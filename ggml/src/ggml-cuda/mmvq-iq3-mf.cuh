// MMVQ-NC A (MFMA form): launcher of the int8-MFMA N-column IQ3 kernels (iq3gemv-mf.cuh), one translation unit per
// column-group count CG (mmvq-iq3-mf<CG>.cu) so the instantiations compile in parallel.
#pragma once
#include "common.cuh"
#include "mmvq-iq3-nc.cuh"

#if defined(GGML_USE_HIP)
#include "iq3gemv-mf.cuh"
#include <algorithm>
#include <cstdlib>

namespace iq3mf_detail {
constexpr int NW = 16;

template <class T0, class T1, int CG, int SM, int MODE, int R>
void launch(const iq3mf_desc & d, const int grid, cudaStream_t stream) {
    using namespace iq3g;
    mf::args a{};
    for (int j = 0; j < mf::MJ; ++j) {
        a.job[j].w    = d.job[j].w;
        a.job[j].w2   = d.job[j].w2;
        a.job[j].dst  = d.job[j].dst;
        a.job[j].aux  = d.job[j].aux;
        a.job[j].nh   = d.job[j].nh;
        a.job[j].ubeg = d.job[j].ubeg;
        a.job[j].tsel = d.job[j].tsel;
        a.job[j].U    = d.job[j].U;
        a.job[j].dcs  = d.job[j].dcs;
        a.job[j].acs  = d.job[j].acs;
    }
    a.y    = d.y;
    a.ycb  = d.ycb;
    a.nc   = d.nc;
    a.K    = d.K;
    a.utot = d.utot;
    const int items = d.K/128;
    const size_t tabs = nc::tab_bytes<T0, T1>();
    const int cs = mf::csb(d.nc);
    const int fit = (int) ((65536 - tabs)/cs);
    int hph = fit >= items ? items : (fit >= 8 ? fit/8*8 : fit);
    hph = std::min(hph, mf::SPT*NW*64/(4*d.nc));
    a.hph = hph;
    const size_t lds = tabs + (size_t) hph*cs;
    mf::gemv<T0, T1, CG, SM, MODE, NW, R><<<grid, dim3(64, NW), lds, stream>>>(a);
}
template <class T0, class T1, int CG, int MODE>
bool launch_sm(const iq3mf_desc & d, cudaStream_t stream) {
    // one block per CU (LDS), every block with work; R = 2 units per wave step (one type, no gate/up) when every wave
    // has at least two units (GGML_MMVQ_NC_IQ3_MFMA_R = 1 or 2 forces it)
    const int grid = std::max(1, std::min(d.ncu, (d.utot + NW - 1)/NW));
    constexpr bool R2 = std::is_same<T1, iq3g::none_t>::value && MODE != 1 && CG <= 2;   // (CG > 2: registers)
    static const int rf = [] { const char * v = getenv("GGML_MMVQ_NC_IQ3_MFMA_R"); return v ? atoi(v) : 0; }();
    const bool r2 = R2 && (rf == 2 || (rf != 1 && d.utot >= 2*grid*NW));
    if (d.sm == 8) { if constexpr (R2) { if (r2) { launch<T0, T1, CG, 8, MODE, 2>(d, grid, stream); return true; } } launch<T0, T1, CG, 8, MODE, 1>(d, grid, stream); return true; }
    if (d.sm == 0) { if constexpr (R2) { if (r2) { launch<T0, T1, CG, 0, MODE, 2>(d, grid, stream); return true; } } launch<T0, T1, CG, 0, MODE, 1>(d, grid, stream); return true; }
    return false;
}
template <int CG>
bool dispatch(const iq3mf_desc & d, cudaStream_t stream) {
    using namespace iq3g;
    const bool s0 = d.t0 == GGML_TYPE_IQ3_S, x0 = d.t0 == GGML_TYPE_IQ3_XXS;
    const bool s1 = d.t1 == GGML_TYPE_IQ3_S, x1 = d.t1 == GGML_TYPE_IQ3_XXS, n1 = d.t1 == GGML_TYPE_COUNT;
    switch (d.mode) {
        case 0:
            if (s0 && n1) return launch_sm<iq3s_t,   none_t,   CG, 0>(d, stream);
            if (x0 && n1) return launch_sm<iq3xxs_t, none_t,   CG, 0>(d, stream);
            if (s0 && x1) return launch_sm<iq3s_t,   iq3xxs_t, CG, 0>(d, stream);
            return false;
        case 1:
            if constexpr (CG <= 2) {
                if (s0 && s1) return launch_sm<iq3s_t,   iq3s_t,   CG, 1>(d, stream);
                if (x0 && x1) return launch_sm<iq3xxs_t, iq3xxs_t, CG, 1>(d, stream);
                if (s0 && x1) return launch_sm<iq3s_t,   iq3xxs_t, CG, 1>(d, stream);
                if (x0 && s1) return launch_sm<iq3xxs_t, iq3s_t,   CG, 1>(d, stream);
            }
            return false;
        case 2:
            if (s0 && n1) return launch_sm<iq3s_t,   none_t, CG, 2>(d, stream);
            if (x0 && n1) return launch_sm<iq3xxs_t, none_t, CG, 2>(d, stream);
            return false;
        case 3:
            if (s0 && n1) return launch_sm<iq3s_t,   none_t, CG, 3>(d, stream);
            if (x0 && n1) return launch_sm<iq3xxs_t, none_t, CG, 3>(d, stream);
            return false;
        default:
            return false;
    }
}
} // namespace iq3mf_detail

#define IQ3MF_INSTANTIATE(CG) \
    bool iq3mf_launch_cg##CG(const iq3mf_desc & d, cudaStream_t stream) { return iq3mf_detail::dispatch<CG>(d, stream); }
#endif
