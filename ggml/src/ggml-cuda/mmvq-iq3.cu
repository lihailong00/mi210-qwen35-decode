// Single-token GEMV for IQ3_S / IQ3_XXS on a lossless SoA re-layout (iq3gemv.cuh, gemv-iq3 workstream).
// The re-laid-out copy (112 B / 98 B per 256 weights, vs 134 B for the retained codex 4-bit packed copy) is created
// once per weight tensor with ggml_cuda_get_repacked (cached with the weight buffer, original kept for MMQ/get_rows).
// Codebooks are decoded from sign-expanded LDS tables; integer per-32 sums are identical to vec_dot_iq3_*_q8_1.
// Epilogues: residual add (fusion->x_bias), fused SwiGLU gate/up (fusion->gate, same or mixed IQ3_S/IQ3_XXS types).
// Multi-job launches (ggml_cuda_mmvq_iq3_multi): 2-3 GEMVs on the same activation (qkv + z, k + v) in one kernel.
// GGML_MMVQ_IQ3=0 disables this path (then the retained codex packed path / stock mmvq is used as before);
// GGML_MMVQ_IQ3_MULTI=0 disables only the multi-job grouping, GGML_MMVQ_IQ3_PF=0 selects the previous prologue,
// GGML_MMVQ_IQ3_SM=0 the previous (unit-major) weight layout.
#include "common.cuh"
#include "iq3-repack.cuh"
#include "mmvq-fused-host.cuh"

#if defined(GGML_USE_HIP)
#include "iq3gemv.cuh"
#include <cstdlib>
#include <climits>

static constexpr int IQ3G_RG = 4;   // row interleave of the repacked copies (= rows per wave of every config below)

static bool iq3g_env_on(const char * name) {
    const char * v = getenv(name);
    return v == nullptr || v[0] != '0';
}
static bool iq3g_enabled() {
    static const bool enabled = iq3g_env_on("GGML_MMVQ_IQ3");
    return enabled;
}
static bool iq3g_multi_enabled() {
    static const bool enabled = iq3g_env_on("GGML_MMVQ_IQ3_MULTI");
    return enabled;
}
// prologue of the single / gate+up kernel (GGML_MMVQ_IQ3_PF, default 5): 0 = previous one (every thread loads an activation
// block, threads past the end all load the last one), 5 = only the owners of the blocks load them, 3 = split prologue
// with the first item prefetched under the staging (single GEMVs only)
static int iq3g_pf_mode() {
    static const int mode = [] {
        const char * v = getenv("GGML_MMVQ_IQ3_PF");
        return v == nullptr ? 5 : atoi(v);
    }();
    return mode;
}
// GGML_MMVQ_IQ3_SMALLN=1: single GEMVs with N <= 1024 left to the codex packed kernel (when CODEX_IQ3_PREPACK /
// _XXS_PREPACK allow it); off by default (faster in the micro-benchmark, neutral end to end)
static bool iq3g_smalln_codex() {
    static const bool enabled = [] { const char * v = getenv("GGML_MMVQ_IQ3_SMALLN"); return v != nullptr && v[0] == '1'; }();
    return enabled;
}

// Layout of the re-laid-out copies: step-major (the chunks one 16-lane wave step covers, CPS = 8, are contiguous for
// all units, so all waves read one moving window of the matrix) when K/128 is a multiple of 8, else unit-major.
// The layout is part of the cache key, so every kernel that reads a tensor uses the same one. GGML_MMVQ_IQ3_SM=0:
// unit-major everywhere (previous layout).
static constexpr int IQ3G_SM = 8;
static int iq3g_sm(const int64_t K) {
    static const bool enabled = iq3g_env_on("GGML_MMVQ_IQ3_SM");
    return enabled && (K/128) % IQ3G_SM == 0 ? IQ3G_SM : 0;
}
static int iq3g_kind(const ggml_tensor * w) {
    return 0x1000 + (int) w->type*16 + IQ3G_RG + (iq3g_sm(w->ne[0]) ? 0x800 : 0);
}

static bool iq3g_weight_ok(const ggml_tensor * w) {
    if (!w || (w->type != GGML_TYPE_IQ3_S && w->type != GGML_TYPE_IQ3_XXS)) {
        return false;
    }
    if (w->view_src || !ggml_is_matrix(w) || !ggml_is_contiguous(w)) {
        return false;
    }
    const int64_t K = w->ne[0], N = w->ne[1];
    if (K % 256 != 0 || N % IQ3G_RG != 0 || K > 28672 || N > (1 << 24)) {
        return false;
    }
    if (!w->buffer || ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return false;
    }
    return GGML_CUDA_CC_IS_CDNA2(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
}

template <int RG>
static void iq3g_repack_fn(const void * src, void * dst, const ggml_tensor * t, hipStream_t stream) {
    iq3g::repack(t->type, src, dst, (int) t->ne[1], (int) t->ne[0], RG, stream, iq3g_sm(t->ne[0]));
}

static const uint8_t * iq3g_get_soa(const ggml_tensor * w, const int device, hipStream_t stream) {
    const size_t nb = (size_t) ggml_nelements(w)/QK_K;
    return (const uint8_t *) ggml_cuda_get_repacked(w, device, stream, iq3g_kind(w), iq3g::soa_bytes(w->type, nb),
            iq3g_repack_fn<IQ3G_RG>);
}

// mixed-type gate/up fusion (accepted by ggml_cuda_should_fuse_mul_mat) only when this path will surely handle it:
// both re-laid-out copies must already exist (they are created by the unfused calls of the first evaluation).
bool ggml_cuda_mmvq_iq3_fuse_mixed_ok(const ggml_tensor * up_w, const ggml_tensor * gate_w) {
    if (!iq3g_enabled() || !iq3g_weight_ok(up_w) || !iq3g_weight_ok(gate_w) || up_w->type == gate_w->type ||
            !ggml_are_same_shape(up_w, gate_w)) {
        return false;
    }
    if (iq3g::lds_bytes<iq3g::iq3s_t, iq3g::iq3xxs_t>((int) up_w->ne[0]) > 65536) {
        return false;
    }
    return ggml_cuda_has_repacked(up_w, iq3g_kind(up_w)) && ggml_cuda_has_repacked(gate_w, iq3g_kind(gate_w));
}

template <class TA, class TB, int P, int PF, int SM>
static bool iq3g_launch(const uint8_t * wa, const uint8_t * wb, const block_q8_1 * y, float * dst, const int K, const int N,
        const float * residual, const int ncu, hipStream_t stream) {
    constexpr int L = 16, R = 1, NW = 16;
    static_assert((64/L)*R == IQ3G_RG, "kernel config must match the repack row interleave");
    static_assert(SM == 0 || SM == L*P/4, "step-major layout: chunks per wave step");
    const size_t smem = iq3g::lds_bytes<TA, TB>(K);
    if (smem > 65536) {
        return false;
    }
    if (PF >= 3 && K/32 > NW*64) {   // the split prologue stages one activation block per thread
        return false;
    }
    iq3g::gemv<TA, TB, P, L, R, NW, PF, 0, 1, SM><<<ncu, dim3(64, NW), smem, stream>>>(wa, wb, y, dst, K, N, residual);
    return true;
}
template <class TA, class TB>
static bool iq3g_launch_pf(const uint8_t * wa, const uint8_t * wb, const block_q8_1 * y, float * dst, const int K, const int N,
        const float * residual, const int ncu, hipStream_t stream, const int sm) {
    const bool pf5 = iq3g_pf_mode() == 5;
    if (sm) {
        return (pf5 && iq3g_launch<TA, TB, 2, 5, IQ3G_SM>(wa, wb, y, dst, K, N, residual, ncu, stream)) ||
               iq3g_launch<TA, TB, 2, 0, IQ3G_SM>(wa, wb, y, dst, K, N, residual, ncu, stream);
    }
    return (pf5 && iq3g_launch<TA, TB, 2, 5, 0>(wa, wb, y, dst, K, N, residual, ncu, stream)) ||
           iq3g_launch<TA, TB, 2, 0, 0>(wa, wb, y, dst, K, N, residual, ncu, stream);
}

bool ggml_cuda_mmvq_iq3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const block_q8_1 * src1_q8_1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    if (!iq3g_enabled() || !iq3g_weight_ok(src0)) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || dst->ne[1] != 1 || dst->ne[0] != src0->ne[1]) {
        return false;
    }
    const ggml_tensor * gate = nullptr;
    const float * residual = nullptr;
    if (fusion) {
        if (fusion->gate_bias || fusion->x_scale || fusion->gate_scale || fusion->shared_up) {
            return false;
        }
        if (fusion->gate) {
            if (fusion->glu_op != GGML_GLU_OP_SWIGLU || fusion->x_bias || !iq3g_weight_ok(fusion->gate) ||
                    !ggml_are_same_shape(fusion->gate, src0)) {
                return false;
            }
            gate = fusion->gate;
        } else if (fusion->x_bias) {
            if (fusion->x_bias->type != GGML_TYPE_F32 || !ggml_are_same_shape(fusion->x_bias, dst) ||
                    !ggml_is_contiguous(fusion->x_bias)) {
                return false;
            }
            residual = (const float *) fusion->x_bias->data;
        }
    }
    const int device = ggml_cuda_get_device();
    hipStream_t stream = ctx.stream();
    // small single GEMVs (attn_k/v, 1024 rows) are latency bound: the table staging costs more than it saves there, the
    // retained codex packed kernel is faster (5.8 vs 8.6 us) - leave them to it when its packed copy exists or can be made
    if (!gate && !residual && src0->ne[1] <= 1024 && iq3g_smalln_codex() && codex_iq3_get_packed(src0, device, stream) != nullptr) {
        return false;
    }
    const uint8_t * wu = iq3g_get_soa(src0, device, stream);
    const uint8_t * wg = gate ? iq3g_get_soa(gate, device, stream) : nullptr;
    if (!wu || (gate && !wg)) {
        return false;
    }
    const int K = (int) src0->ne[0];
    const int N = (int) src0->ne[1];
    const int ncu = ggml_cuda_info().devices[device].nsm;
    float * d = (float *) dst->data;
    const bool us = src0->type == GGML_TYPE_IQ3_S;
    const int  sm = iq3g_sm(K);   // = the layout iq3g_get_soa created the copies with
    bool ok = false;
    using namespace iq3g;
    if (!gate) {
        ok = us ? iq3g_launch_pf<iq3s_t,   none_t>(wu, nullptr, src1_q8_1, d, K, N, residual, ncu, stream, sm)
                : iq3g_launch_pf<iq3xxs_t, none_t>(wu, nullptr, src1_q8_1, d, K, N, residual, ncu, stream, sm);
    } else {
        // TA = gate (weights wg), TB = up (weights wu): dst = silu(gate.y) * (up.y)
        const bool gs = gate->type == GGML_TYPE_IQ3_S;
        if (gs && us)        ok = iq3g_launch_pf<iq3s_t,   iq3s_t  >(wg, wu, src1_q8_1, d, K, N, nullptr, ncu, stream, sm);
        else if (gs && !us)  ok = iq3g_launch_pf<iq3s_t,   iq3xxs_t>(wg, wu, src1_q8_1, d, K, N, nullptr, ncu, stream, sm);
        else if (!gs && us)  ok = iq3g_launch_pf<iq3xxs_t, iq3s_t  >(wg, wu, src1_q8_1, d, K, N, nullptr, ncu, stream, sm);
        else                 ok = iq3g_launch_pf<iq3xxs_t, iq3xxs_t>(wg, wu, src1_q8_1, d, K, N, nullptr, ncu, stream, sm);
    }
    if (ok) {
        CUDA_CHECK(cudaGetLastError());
    }
    return ok;
}

// ------------------------------------------------------------------------------------------------ multi-job
static constexpr int IQ3G_MJ_NW = 16;
static constexpr int IQ3G_MJ_L  = 16;

template <int SM>
static void iq3g_launch_mj(const bool hs, const bool hx, const block_q8_1 * y, const int K, const iq3g::mj_args & a, const int ncu,
        hipStream_t stream) {
    using namespace iq3g;
    constexpr int NW = IQ3G_MJ_NW, L = IQ3G_MJ_L;
    static_assert(64/L == IQ3G_RG, "kernel config must match the repack row interleave");
    if (hs && hx) {
        gemv_mj<iq3s_t, iq3xxs_t, 2, L, NW, 1, 0, SM><<<ncu, dim3(64, NW), lds_bytes_mj<iq3s_t, iq3xxs_t>(K, 1, NW, L), stream>>>(y, K, a);
    } else if (hs) {
        gemv_mj<iq3s_t, none_t, 2, L, NW, 1, 0, SM><<<ncu, dim3(64, NW), lds_bytes_mj<iq3s_t, none_t>(K, 1, NW, L), stream>>>(y, K, a);
    } else {
        gemv_mj<iq3xxs_t, none_t, 2, L, NW, 1, 0, SM><<<ncu, dim3(64, NW), lds_bytes_mj<iq3xxs_t, none_t>(K, 1, NW, L), stream>>>(y, K, a);
    }
}

bool ggml_cuda_mmvq_iq3_multi_ok(const ggml_tensor * const * w, const int n) {
    if (!iq3g_enabled() || !iq3g_multi_enabled() || n < 2 || n > iq3g::MJ_MAX) {
        return false;
    }
    bool hs = false, hx = false;
    int64_t rows = 0;
    for (int i = 0; i < n; ++i) {
        if (!iq3g_weight_ok(w[i]) || w[i]->ne[0] != w[0]->ne[0]) {
            return false;
        }
        hs |= w[i]->type == GGML_TYPE_IQ3_S;
        hx |= w[i]->type == GGML_TYPE_IQ3_XXS;
        rows += w[i]->ne[1];
    }
    const int K = (int) w[0]->ne[0];
    if (rows > (1 << 24) || K/32 > IQ3G_MJ_NW*64) {
        return false;
    }
    using namespace iq3g;
    const size_t lds = (hs && hx) ? lds_bytes_mj<iq3s_t, iq3xxs_t>(K, 1, IQ3G_MJ_NW, IQ3G_MJ_L) :
                       hs         ? lds_bytes_mj<iq3s_t, none_t  >(K, 1, IQ3G_MJ_NW, IQ3G_MJ_L) :
                                    lds_bytes_mj<iq3xxs_t, none_t>(K, 1, IQ3G_MJ_NW, IQ3G_MJ_L);
    return lds <= 65536;
}

bool ggml_cuda_mmvq_iq3_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const * w, ggml_tensor * const * dst, const int n,
        const block_q8_1 * src1_q8_1) {
    if (!ggml_cuda_mmvq_iq3_multi_ok(w, n)) {
        return false;
    }
    const int device = ggml_cuda_get_device();
    hipStream_t stream = ctx.stream();
    const int K = (int) w[0]->ne[0];
    iq3g::mj_args a{};
    for (int j = 0; j < iq3g::MJ_MAX; ++j) {
        a.job[j].ubeg = INT_MAX;
    }
    bool hs = false, hx = false;
    for (int i = 0; i < n; ++i) {
        hs |= w[i]->type == GGML_TYPE_IQ3_S;
        hx |= w[i]->type == GGML_TYPE_IQ3_XXS;
    }
    int ub = 0;
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * d = dst[i];
        if (d->type != GGML_TYPE_F32 || !ggml_is_contiguous(d) || ggml_nelements(d) != w[i]->ne[1]) {
            return false;
        }
        const uint8_t * soa = iq3g_get_soa(w[i], device, stream);   // nullptr during graph capture if not created yet
        if (!soa) {
            return false;
        }
        a.job[i].w    = soa;
        a.job[i].dst  = (float *) d->data;
        a.job[i].res  = nullptr;
        a.job[i].nh   = (uint32_t) (w[i]->ne[1]*(K/128));
        a.job[i].ubeg = ub;
        a.job[i].tsel = (hs && hx && w[i]->type == GGML_TYPE_IQ3_XXS) ? 1 : 0;
        a.job[i].U    = (int) (w[i]->ne[1] / IQ3G_RG);
        ub += (int) (w[i]->ne[1] / IQ3G_RG);
    }
    a.njob = n;
    a.utot = ub;
    const int ncu = ggml_cuda_info().devices[device].nsm;
    // no first-item prefetch under the staging (PFM 0): slower with IQ3_XXS in the bench and for IQ3_S groups in situ
    if (iq3g_sm(K)) {
        iq3g_launch_mj<IQ3G_SM>(hs, hx, src1_q8_1, K, a, ncu, stream);
    } else {
        iq3g_launch_mj<0>(hs, hx, src1_q8_1, K, a, ncu, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
#else
bool ggml_cuda_mmvq_iq3_fuse_mixed_ok(const ggml_tensor *, const ggml_tensor *) {
    return false;
}
bool ggml_cuda_mmvq_iq3(ggml_backend_cuda_context &, const ggml_tensor *, const block_q8_1 *, ggml_tensor *, const ggml_cuda_mm_fusion_args_host *) {
    return false;
}
bool ggml_cuda_mmvq_iq3_multi_ok(const ggml_tensor * const *, int) {
    return false;
}
bool ggml_cuda_mmvq_iq3_multi(ggml_backend_cuda_context &, const ggml_tensor * const *, ggml_tensor * const *, int, const block_q8_1 *) {
    return false;
}
#endif
