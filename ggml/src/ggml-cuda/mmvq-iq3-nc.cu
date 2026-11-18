// MMVQ-NC A: N-column IQ3_S / IQ3_XXS GEMV launchers, single / residual / SwiGLU gate+up (iq3gemv-nc.cuh).
// Config as the single-token launcher (mmvq-iq3.cu iq3g_launch): P = 2, L = 16, NW = 16, persistent grid = CUs;
// K phases (up to nc::PH_MAXU units per wave) when the NC activation record sets do not fit next to the tables.
#include "common.cuh"
#include "mmvq-iq3-nc.cuh"

#if defined(GGML_USE_HIP)
#include "iq3gemv-nc.cuh"

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

bool iq3nc_launch_single(const iq3nc_single_args & a, cudaStream_t stream) {
    switch (a.ta) {
        case GGML_TYPE_IQ3_S:   return launch_single_tb<iq3g::iq3s_t>(a, stream);
        case GGML_TYPE_IQ3_XXS: return launch_single_tb<iq3g::iq3xxs_t>(a, stream);
        default: return false;
    }
}
#endif
