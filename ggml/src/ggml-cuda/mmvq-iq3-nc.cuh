// MMVQ-NC A: launchers of the N-column IQ3_S / IQ3_XXS kernels (iq3gemv-nc.cuh, instantiated in mmvq-iq3-nc.cu;
// MFMA form iq3gemv-mf.cuh, instantiated in mmvq-iq3-mf<CG>.cu); the tensor-level host logic is in mmvq-iq3.cu
// (ggml_cuda_mmvq_iq3_nc / _multi_nc).
#pragma once
#include "common.cuh"

// largest column count the IQ3 N-column family takes (0: off); honours GGML_MMVQ_NC_MAX, GGML_MMVQ_NC_IQ3 and
// GGML_MMVQ_NC_IQ3_MFMA (also declared in mmvq.cuh by the routing patch; identical declaration)
int ggml_cuda_mmvq_iq3_nc_max();

#if defined(GGML_USE_HIP)
static constexpr int IQ3NC_MIN = 2;
static constexpr int IQ3NC_MAX = 6;    // instantiated column counts 2..6 of the dp4a form
static constexpr int IQ3MF_MAX = 16;   // MFMA form: 2..16

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
    float            * gsc;     // MFMA form, split gate/up: scratch for the gate results (N*ncols floats), or nullptr
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

// ---- MFMA form (GGML_MMVQ_NC_IQ3_MFMA, default on; 0 = the dp4a form only, ncols <= IQ3NC_MAX)
bool iq3mf_enabled();
int  iq3mf_min();            // smallest column count that takes the MFMA form (GGML_MMVQ_NC_IQ3_MFMA_MIN, default 4)
bool iq3mf_use(int ncols);   // MFMA form for this column count
// the GLU launch needs a gate scratch (split gate / up passes) for these shapes
bool iq3mf_glu_split(ggml_type tgate, ggml_type tup, int ncols);

struct iq3mf_job {
    const uint8_t * w;      // weights (MODE 1: gate)
    const uint8_t * w2;     // MODE 1: up weights
    float         * dst;    // column c at dst + c*dcs
    const float   * aux;    // MODE 0: residual or nullptr; MODE 3: gate results; column c at aux + c*acs
    uint32_t        nh;
    int             ubeg;   // unused jobs: INT_MAX
    int             tsel;   // MODE 0 with t1 != COUNT: 0 = t0, 1 = t1
    int             U;
    int             dcs;
    int             acs;
};
struct iq3mf_desc {
    iq3mf_job          job[3];
    const block_q8_1 * y;
    int                ycb, nc, K, utot, ncu, sm;
    ggml_type          t0, t1;   // t1 = GGML_TYPE_COUNT: one type
    int                mode;     // 0 plain/residual, 1 gate/up, 2 gate pass, 3 up pass (see iq3gemv-mf.cuh)
};
bool iq3mf_launch_cg1(const iq3mf_desc & d, cudaStream_t stream);
bool iq3mf_launch_cg2(const iq3mf_desc & d, cudaStream_t stream);
bool iq3mf_launch_cg3(const iq3mf_desc & d, cudaStream_t stream);
bool iq3mf_launch_cg4(const iq3mf_desc & d, cudaStream_t stream);
static inline bool iq3mf_launch(const iq3mf_desc & d, cudaStream_t stream) {
    switch ((d.nc + 3)/4) {
        case 1: return iq3mf_launch_cg1(d, stream);
        case 2: return iq3mf_launch_cg2(d, stream);
        case 3: return iq3mf_launch_cg3(d, stream);
        case 4: return iq3mf_launch_cg4(d, stream);
        default: return false;
    }
}
// LDS bytes of the codebook tables of a launch
static inline size_t iq3mf_tab_bytes(const ggml_type t0, const ggml_type t1) {
    auto tb = [](const ggml_type t) -> size_t { return t == GGML_TYPE_IQ3_S ? 32768 : t == GGML_TYPE_IQ3_XXS ? 16384 : 0; };
    return tb(t0) + (t1 != t0 ? tb(t1) : 0);
}
// a K phase of 8 chunks (16 items: every item lane busy) fits next to the tables
static inline bool iq3mf_fit8(const ggml_type t0, const ggml_type t1, const int nc) {
    const size_t cs = (size_t) nc*144 + (nc % 2 == 0 ? 16 : 0);
    return iq3mf_tab_bytes(t0, t1) + 8*cs <= 65536;
}
#endif
