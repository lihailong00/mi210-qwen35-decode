// MMVQ-NC A: launchers of the N-column IQ3_S / IQ3_XXS kernels (iq3gemv-nc.cuh, instantiated in mmvq-iq3-nc.cu);
// the tensor-level host logic is in mmvq-iq3.cu (ggml_cuda_mmvq_iq3_nc / _multi_nc).
#pragma once
#include "common.cuh"

#if defined(GGML_USE_HIP)
static constexpr int IQ3NC_MIN = 2;
static constexpr int IQ3NC_MAX = 6;   // instantiated column counts 2..6

struct iq3nc_single_args {
    const uint8_t    * wa;      // TA weights (GLU: gate)
    const uint8_t    * wb;      // TB weights (GLU: up) or nullptr
    ggml_type          ta;
    ggml_type          tb;      // GGML_TYPE_COUNT: no second matrix
    const block_q8_1 * y;       // column c at y + c*ycb
    int                ycb;
    float            * dst;     // column c at dst + c*dcs
    int                dcs;
    const float      * res;     // residual column c at res + c*rcs, or nullptr
    int                rcs;
    int                K, N, sm, ncols, ncu;
};
struct iq3nc_multi_args {
    const uint8_t    * w[3];
    ggml_type          t[3];
    float            * dst[3];
    int                dcs[3];
    int                N[3];
    int                n;
    const block_q8_1 * y;
    int                ycb;
    int                K, sm, ncols, ncu;
};
// false: configuration not instantiated (nothing launched)
bool iq3nc_launch_single(const iq3nc_single_args & a, cudaStream_t stream);
bool iq3nc_launch_multi(const iq3nc_multi_args & a, cudaStream_t stream);
#endif
