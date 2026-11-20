// MMVQ-NC A: N-column IQ3_S / IQ3_XXS GEMV launchers, single / residual / SwiGLU gate+up (iq3gemv-nc.cuh).
// Config as the single-token launcher (mmvq-iq3.cu iq3g_launch): P = 2, L = 16, NW = 16, persistent grid = CUs;
// K phases (up to nc::PH_MAXU units per wave) when the NC activation record sets do not fit next to the tables.
// ncols >= GGML_MMVQ_NC_IQ3_MFMA_MIN (and every ncols above the dp4a range, up to 16): the int8-MFMA form (iq3gemv-mf.cuh).
#include "common.cuh"
#include "mmvq-iq3-nc.cuh"

#if defined(GGML_USE_HIP)
#include "iq3gemv-nc.cuh"
#include <algorithm>
#include <climits>
#include <cstdlib>

namespace {
constexpr int NCL = 16, NCNW = 16, NCP = 2;

template <class TA, class TB, int NC, int SM>
void launch_single_t(const iq3nc_single_args & a, cudaStream_t stream) {
    using namespace iq3g;
    const int items = a.K/128;
    const size_t tabs = nc::tab_bytes<TA, TB>();
    const size_t full = tabs + nc::rec_bytes<NC>(items);
    if (full <= 65536) {
        nc::gemv<TA, TB, NCP, NCL, NCNW, NC, SM, false><<<a.ncu, dim3(64, NCNW), full, stream>>>(
            a.wa, a.wb, a.y, a.ycb, a.dst, a.dcs, a.K, a.N, a.res, a.rcs, items);
    } else {
        const int hph = (int) ((65536 - tabs)/nc::lay<NC>::CSB);
        const size_t lds = tabs + nc::rec_bytes<NC>(hph);
        const int U = a.N/(64/NCL);
        const int grid = std::max(a.ncu, (U + NCNW*nc::PH_MAXU - 1)/(NCNW*nc::PH_MAXU));
        nc::gemv<TA, TB, NCP, NCL, NCNW, NC, SM, true><<<grid, dim3(64, NCNW), lds, stream>>>(
            a.wa, a.wb, a.y, a.ycb, a.dst, a.dcs, a.K, a.N, a.res, a.rcs, hph);
    }
}
template <class TA, class TB, int SM>
bool launch_single_nc(const iq3nc_single_args & a, cudaStream_t stream) {
    switch (a.ncols) {
        case 2: launch_single_t<TA, TB, 2, SM>(a, stream); return true;
        case 3: launch_single_t<TA, TB, 3, SM>(a, stream); return true;
        case 4: launch_single_t<TA, TB, 4, SM>(a, stream); return true;
        case 5: launch_single_t<TA, TB, 5, SM>(a, stream); return true;
        case 6: launch_single_t<TA, TB, 6, SM>(a, stream); return true;
        default: return false;
    }
}
template <class TA, class TB>
bool launch_single_sm(const iq3nc_single_args & a, cudaStream_t stream) {
    if (a.sm == 8) return launch_single_nc<TA, TB, 8>(a, stream);
    if (a.sm == 0) return launch_single_nc<TA, TB, 0>(a, stream);
    return false;
}
template <class TA>
bool launch_single_tb(const iq3nc_single_args & a, cudaStream_t stream) {
    using namespace iq3g;
    switch (a.tb) {
        case GGML_TYPE_COUNT:   return launch_single_sm<TA, none_t>(a, stream);
        case GGML_TYPE_IQ3_S:   return launch_single_sm<TA, iq3s_t>(a, stream);
        case GGML_TYPE_IQ3_XXS: return launch_single_sm<TA, iq3xxs_t>(a, stream);
        default: return false;
    }
}
} // namespace

// ---- MFMA form
bool iq3mf_enabled() {
    static const bool enabled = [] { const char * v = getenv("GGML_MMVQ_NC_IQ3_MFMA"); return v == nullptr || v[0] != '0'; }();
    return enabled;
}
static int iq3mf_min_env() {   // GGML_MMVQ_NC_IQ3_MFMA_MIN, 0 if not set
    static const int m = [] { const char * v = getenv("GGML_MMVQ_NC_IQ3_MFMA_MIN"); return v ? std::max(2, atoi(v)) : 0; }();
    return m;
}
int iq3mf_min() {
    // N = 2..3: the dp4a form is faster than the first MFMA form; with the second form the choice is per shape
    // (iq3mf_use_shape), 2 is the smallest column count that can take it
    return iq3mf_min_env() ? iq3mf_min_env() : (iq3mf2_enabled() ? 2 : 4);
}
bool iq3mf_use(const int ncols) {
    // above the dp4a form's range the MFMA form is the only one (whatever the minimum says)
    return iq3mf_enabled() && ncols >= 2 && ncols <= IQ3MF_MAX && (ncols >= iq3mf_min() || ncols > IQ3NC_MAX);
}
bool iq3mf_use_shape(const int ncols, const int rows, const int K) {
    if (!iq3mf_use(ncols)) {
        return false;
    }
    if (ncols > IQ3NC_MAX || iq3mf_min_env() || !iq3mf2_enabled()) {
        return true;
    }
    // second form, measured in the model (verification step, per node): the dp4a form stays faster for the small
    // matrices (K/V projections, 1024 rows) and at 2 columns except for the K-phased ffn_down (K = 17408)
    if (rows < 4096) {
        return false;
    }
    return ncols >= 3 || K >= 16384;
}
bool iq3mf2_enabled() {
    static const bool enabled = [] { const char * v = getenv("GGML_MMVQ_NC_IQ3_MF2"); return v == nullptr || v[0] != '0'; }();
    return enabled;
}
bool iq3mf_glu_split(const ggml_type tgate, const ggml_type tup, const int ncols) {
    // one launch keeps two accumulator sets: up to 2 column groups; and a full K phase must fit next to both tables
    // (the gate/up launch runs the second form at one column group, and at two for an IQ3_XXS pair: see launch_t)
    const bool mf2 = iq3mf2_enabled() && ((ncols + 3)/4 == 1 || (tgate == GGML_TYPE_IQ3_XXS && tup == GGML_TYPE_IQ3_XXS));
    return (ncols + 3)/4 > 2 || !iq3mf_fit8(tgate, tup, ncols, mf2);
}
static void iq3mf_job_init(iq3mf_desc & d) {
    for (int j = 0; j < 3; ++j) {
        d.job[j] = iq3mf_job{};
        d.job[j].ubeg = INT_MAX;
    }
}
static bool iq3mf_launch_single(const iq3nc_single_args & a, cudaStream_t stream) {
    iq3mf_desc d{};
    iq3mf_job_init(d);
    d.y = a.y; d.ycb = a.ycb; d.nc = a.ncols; d.K = a.K; d.ncu = a.ncu; d.sm = a.sm;
    d.utot = a.N/4;
    iq3mf_job & J = d.job[0];
    J.nh = (uint32_t) a.N*(uint32_t) (a.K/128);
    J.ubeg = 0;
    J.U = a.N/4;
    if (a.tb == GGML_TYPE_COUNT) {
        d.mode = 0; d.t0 = a.ta; d.t1 = GGML_TYPE_COUNT;
        J.w = a.wa; J.dst = a.dst; J.dcs = a.dcs; J.aux = a.res; J.acs = a.rcs;
        return iq3mf_launch(d, stream);
    }
    if (!iq3mf_glu_split(a.ta, a.tb, a.ncols)) {
        d.mode = 1; d.t0 = a.ta; d.t1 = a.tb;
        J.w = a.wa; J.w2 = a.wb; J.dst = a.dst; J.dcs = a.dcs;
        return iq3mf_launch(d, stream);
    }
    if (!a.gsc) {
        return false;
    }
    // split gate / up: gate pass (fma accumulation, raw group sums into the scratch), then the up pass with the epilogue
    d.mode = 2; d.t0 = a.ta; d.t1 = GGML_TYPE_COUNT;
    J.w = a.wa; J.dst = a.gsc; J.dcs = a.N;
    if (!iq3mf_launch(d, stream)) {
        return false;
    }
    d.mode = 3; d.t0 = a.tb;
    J.w = a.wb; J.dst = a.dst; J.dcs = a.dcs; J.aux = a.gsc; J.acs = a.N;
    return iq3mf_launch(d, stream);
}

bool iq3nc_launch_single(const iq3nc_single_args & a, cudaStream_t stream) {
    if (iq3mf_use_shape(a.ncols, a.N, a.K)) {
        return iq3mf_launch_single(a, stream);
    }
    switch (a.ta) {
        case GGML_TYPE_IQ3_S:   return launch_single_tb<iq3g::iq3s_t>(a, stream);
        case GGML_TYPE_IQ3_XXS: return launch_single_tb<iq3g::iq3xxs_t>(a, stream);
        default: return false;
    }
}
#endif
