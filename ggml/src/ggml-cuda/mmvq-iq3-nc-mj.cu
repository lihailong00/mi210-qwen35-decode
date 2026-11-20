// MMVQ-NC A: N-column IQ3_S / IQ3_XXS multi-job launcher (qkv + z, q/k/v: 2..3 GEMVs on the same activation in one
// launch), iq3gemv-nc.cuh. Units concatenated in job order as in the single-token multi-job launcher.
#include "common.cuh"
#include "mmvq-iq3-nc.cuh"

#if defined(GGML_USE_HIP)
#include "iq3gemv-nc.cuh"
#include <climits>

namespace {
constexpr int MJL = 16, MJNW = 16, MJP = 2;

template <class T0, class T1, int NC, int SM>
void launch_multi_t(const iq3nc_multi_args & a, const iq3g::nc::mjnc_args & ma, cudaStream_t stream) {
    using namespace iq3g;
    const int items = a.K/128;
    const size_t tabs = nc::tab_bytes<T0, T1>();
    const size_t full = tabs + nc::rec_bytes<NC>(items);
    if (full <= 65536) {
        nc::gemv_mj<T0, T1, MJP, MJL, MJNW, NC, SM, false><<<a.ncu, dim3(64, MJNW), full, stream>>>(a.y, a.ycb, a.K, ma, items);
    } else {
        const int hph = (int) ((65536 - tabs)/nc::lay<NC>::CSB);
        const size_t lds = tabs + nc::rec_bytes<NC>(hph);
        const int grid = std::max(a.ncu, (ma.utot + MJNW*nc::PH_MAXU - 1)/(MJNW*nc::PH_MAXU));
        nc::gemv_mj<T0, T1, MJP, MJL, MJNW, NC, SM, true><<<grid, dim3(64, MJNW), lds, stream>>>(a.y, a.ycb, a.K, ma, hph);
    }
}
template <class T0, class T1, int SM>
bool launch_multi_nc(const iq3nc_multi_args & a, const iq3g::nc::mjnc_args & ma, cudaStream_t stream) {
    switch (a.ncols) {
        case 2: launch_multi_t<T0, T1, 2, SM>(a, ma, stream); return true;
        case 3: launch_multi_t<T0, T1, 3, SM>(a, ma, stream); return true;
        case 4: launch_multi_t<T0, T1, 4, SM>(a, ma, stream); return true;
        case 5: launch_multi_t<T0, T1, 5, SM>(a, ma, stream); return true;
        case 6: launch_multi_t<T0, T1, 6, SM>(a, ma, stream); return true;
        default: return false;
    }
}
template <class T0, class T1>
bool launch_multi_sm(const iq3nc_multi_args & a, const iq3g::nc::mjnc_args & ma, cudaStream_t stream) {
    if (a.sm == 8) return launch_multi_nc<T0, T1, 8>(a, ma, stream);
    if (a.sm == 0) return launch_multi_nc<T0, T1, 0>(a, ma, stream);
    return false;
}
} // namespace

bool iq3nc_launch_multi(const iq3nc_multi_args & a, cudaStream_t stream) {
    using namespace iq3g;
    if (a.n < 2 || a.n > MJ_MAX) {
        return false;
    }
    bool hs = false, hx = false;
    for (int i = 0; i < a.n; ++i) {
        hs |= a.t[i] == GGML_TYPE_IQ3_S;
        hx |= a.t[i] == GGML_TYPE_IQ3_XXS;
        if (a.t[i] != GGML_TYPE_IQ3_S && a.t[i] != GGML_TYPE_IQ3_XXS) {
            return false;
        }
    }
    int rows = 0;
    for (int i = 0; i < a.n; ++i) {
        rows += a.N[i];
    }
    if (iq3mf_use_shape(a.ncols, rows, a.K)) {
        // MFMA form: one launch (mixed types: both tables, per-job type) if a full K phase fits next to the tables,
        // else one launch per type (the jobs keep their own rows, outputs and arithmetic)
        // second form, one column group: one launch per type (no K phases next to the two tables; measured faster
        // than one launch with phases)
        const bool split2 = iq3mf2_enabled() && (a.ncols + 3)/4 == 1;
        const bool one = !(hs && hx) || (!split2 && (a.ncols + 3)/4 <= 2 && iq3mf_fit8(GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, a.ncols));
        for (int pass = 0; pass < (one ? 1 : 2); ++pass) {
            iq3mf_desc d{};
            for (int j = 0; j < 3; ++j) {
                d.job[j] = iq3mf_job{};
                d.job[j].ubeg = INT_MAX;
            }
            d.y = a.y; d.ycb = a.ycb; d.nc = a.ncols; d.K = a.K; d.ncu = a.ncu; d.sm = a.sm; d.mode = 0;
            const ggml_type tp = pass == 0 ? (hs ? GGML_TYPE_IQ3_S : GGML_TYPE_IQ3_XXS) : GGML_TYPE_IQ3_XXS;
            d.t0 = one ? (hs ? GGML_TYPE_IQ3_S : GGML_TYPE_IQ3_XXS) : tp;
            d.t1 = one && hs && hx ? GGML_TYPE_IQ3_XXS : GGML_TYPE_COUNT;
            int ub = 0, nj = 0;
            for (int i = 0; i < a.n; ++i) {
                if (!one && a.t[i] != tp) {
                    continue;
                }
                iq3mf_job & J = d.job[nj++];
                J.w    = a.w[i];
                J.dst  = a.dst[i];
                J.dcs  = a.dcs[i];
                J.nh   = (uint32_t) a.N[i]*(uint32_t) (a.K/128);
                J.ubeg = ub;
                J.tsel = d.t1 != GGML_TYPE_COUNT && a.t[i] == GGML_TYPE_IQ3_XXS ? 1 : 0;
                J.U    = a.N[i]/4;
                ub += a.N[i]/4;
            }
            d.utot = ub;
            if (nj > 0 && !iq3mf_launch(d, stream)) {
                return false;
            }
        }
        return true;
    }
    nc::mjnc_args ma{};
    for (int j = 0; j < MJ_MAX; ++j) {
        ma.job[j].ubeg = INT_MAX;
    }
    int ub = 0;
    for (int i = 0; i < a.n; ++i) {
        ma.job[i].w    = a.w[i];
        ma.job[i].dst  = a.dst[i];
        ma.job[i].nh   = (uint32_t) a.N[i]*(uint32_t) (a.K/128);
        ma.job[i].ubeg = ub;
        ma.job[i].tsel = (hs && hx && a.t[i] == GGML_TYPE_IQ3_XXS) ? 1 : 0;
        ma.job[i].U    = a.N[i]/4;
        ma.job[i].dcs  = a.dcs[i];
        ub += a.N[i]/4;
    }
    ma.njob = a.n;
    ma.utot = ub;
    if (hs && hx) return launch_multi_sm<iq3s_t,   iq3xxs_t>(a, ma, stream);
    if (hs)       return launch_multi_sm<iq3s_t,   none_t  >(a, ma, stream);
    return               launch_multi_sm<iq3xxs_t, none_t  >(a, ma, stream);
}
#endif
