// MMVQ-NC A (MFMA form): launcher of the int8-MFMA N-column IQ3 kernels (iq3gemv-mf.cuh), one translation unit per
// column-group count CG (mmvq-iq3-mf<CG>.cu) so the instantiations compile in parallel.
#pragma once
#include "common.cuh"
#include "mmvq-iq3-nc.cuh"

#if defined(GGML_USE_HIP)
#include "iq3gemv-mf.cuh"
#include "iq3gemv-mf2.cuh"
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
    // second form (iq3gemv-mf2.cuh, GGML_MMVQ_NC_IQ3_MF2, default on): same results, other LDS record layout, one unit
    // per step (R = 1). From two column groups on, the first form stays for the gate/up pairs with an IQ3_S matrix (the
    // second form spills registers there) and for the mixed-type multi-job launches (measured faster in the model).
    constexpr bool XX  = std::is_same<T0, iq3xxs_t>::value && std::is_same<T1, iq3xxs_t>::value;
    constexpr bool SEL = !std::is_same<T1, none_t>::value && MODE != 1;
    constexpr bool MF2 = R == 1 && (CG == 1 || ((MODE != 1 || XX) && !SEL));
    const bool mf2on = MF2 && iq3mf2_enabled();
    const int cs = mf2on ? mf2::cs2(d.nc) : mf::csb(d.nc);
    const int fit = (int) ((65536 - tabs)/cs);
    int hph = fit >= items ? items : (fit >= 8 ? fit/8*8 : fit);
    hph = std::min(hph, mf::SPT*NW*64/(4*d.nc));
    a.hph = hph;
    const size_t lds = tabs + (size_t) hph*cs;
    if constexpr (MF2) {
        if (mf2on) {
            mf2::gemv<T0, T1, CG, SM, MODE, NW><<<grid, dim3(64, NW), lds, stream>>>(a);
            return;
        }
    }
    mf::gemv<T0, T1, CG, SM, MODE, NW, R><<<grid, dim3(64, NW), lds, stream>>>(a);
}
template <class T0, class T1, int CG, int MODE>
bool launch_sm(const iq3mf_desc & d, cudaStream_t stream) {
    // R = 2 units per wave step (first form; one type, no gate/up) when every wave has at least two units
    // (GGML_MMVQ_NC_IQ3_MFMA_R = 1 or 2 forces it).
    // One block per CU (LDS). All CUs (default; GGML_MMVQ_NC_IQ3_MF_GRID=0: only as many blocks as units/NW, the
    // previous rule): the kernels are bound by per-CU throughput (LDS lookups, VALU), so 1280 units (5120 rows) run
    // faster as 12-13 waves on each of 104 CUs than as 16 waves on 80 CUs. Only the unit -> wave ownership changes.
    static const bool gall = [] { const char * v = getenv("GGML_MMVQ_NC_IQ3_MF_GRID"); return v == nullptr || v[0] != '0'; }();
    const int grid = gall ? std::max(1, std::min(d.ncu, d.utot)) : std::max(1, std::min(d.ncu, (d.utot + NW - 1)/NW));
    constexpr bool R2 = std::is_same<T1, iq3g::none_t>::value && MODE != 1 && CG <= 2;   // (CG > 2: registers)
    static const int rf = [] { const char * v = getenv("GGML_MMVQ_NC_IQ3_MFMA_R"); return v ? atoi(v) : 0; }();
    // (the second form runs R = 1: two units per step need more than 128 VGPRs there)
    const bool r2 = R2 && (rf == 2 || (rf != 1 && !iq3mf2_enabled() && d.utot >= 2*grid*NW));
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
