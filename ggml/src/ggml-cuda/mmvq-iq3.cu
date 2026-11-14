// Single-token GEMV for IQ3_S / IQ3_XXS on a lossless SoA re-layout (iq3gemv.cuh, gemv-iq3 workstream).
// The re-laid-out copy (112 B / 98 B per 256 weights, vs 134 B for the retained codex 4-bit packed copy) is created
// once per weight tensor with ggml_cuda_get_repacked (cached with the weight buffer, original kept for MMQ/get_rows).
// Codebooks are decoded from sign-expanded LDS tables; integer per-32 sums are identical to vec_dot_iq3_*_q8_1.
// Epilogues: residual add (fusion->x_bias), fused SwiGLU gate/up (fusion->gate, same or mixed IQ3_S/IQ3_XXS types).
// GGML_MMVQ_IQ3=0 disables this path (then the retained codex packed path / stock mmvq is used as before).
#include "common.cuh"
#include "iq3-repack.cuh"
#include "mmvq-fused-host.cuh"

#if defined(GGML_USE_HIP)
#include "iq3gemv.cuh"
#include <cstdlib>

static constexpr int IQ3G_RG = 4;   // row interleave of the repacked copies (= rows per wave of every config below)

static bool iq3g_enabled() {
    static const bool enabled = [] {
        const char * v = getenv("GGML_MMVQ_IQ3");
        return v == nullptr || v[0] != '0';
    }();
    return enabled;
}

static int iq3g_kind(const ggml_type t) {
    return 0x1000 + (int) t*16 + IQ3G_RG;
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
    iq3g::repack(t->type, src, dst, (int) t->ne[1], (int) t->ne[0], RG, stream);
}

static const uint8_t * iq3g_get_soa(const ggml_tensor * w, const int device, hipStream_t stream) {
    const size_t nb = (size_t) ggml_nelements(w)/QK_K;
    return (const uint8_t *) ggml_cuda_get_repacked(w, device, stream, iq3g_kind(w->type), iq3g::soa_bytes(w->type, nb),
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
    return ggml_cuda_has_repacked(up_w, iq3g_kind(up_w->type)) && ggml_cuda_has_repacked(gate_w, iq3g_kind(gate_w->type));
}

template <class TA, class TB, int P, int PF>
static bool iq3g_launch(const uint8_t * wa, const uint8_t * wb, const block_q8_1 * y, float * dst, const int K, const int N,
        const float * residual, const int ncu, hipStream_t stream) {
    constexpr int L = 16, R = 1, NW = 16;
    static_assert((64/L)*R == IQ3G_RG, "kernel config must match the repack row interleave");
    const size_t smem = iq3g::lds_bytes<TA, TB>(K);
    if (smem > 65536) {
        return false;
    }
    iq3g::gemv<TA, TB, P, L, R, NW, PF><<<ncu, dim3(64, NW), smem, stream>>>(wa, wb, y, dst, K, N, residual);
    return true;
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
    bool ok = false;
    using namespace iq3g;
    if (!gate) {
        ok = us ? iq3g_launch<iq3s_t,   none_t, 2, 0>(wu, nullptr, src1_q8_1, d, K, N, residual, ncu, stream)
                : iq3g_launch<iq3xxs_t, none_t, 4, 0>(wu, nullptr, src1_q8_1, d, K, N, residual, ncu, stream);
    } else {
        // TA = gate (weights wg), TB = up (weights wu): dst = silu(gate.y) * (up.y)
        const bool gs = gate->type == GGML_TYPE_IQ3_S;
        if (gs && us)        ok = iq3g_launch<iq3s_t,   iq3s_t,   2, 0>(wg, wu, src1_q8_1, d, K, N, nullptr, ncu, stream);
        else if (gs && !us)  ok = iq3g_launch<iq3s_t,   iq3xxs_t, 2, 0>(wg, wu, src1_q8_1, d, K, N, nullptr, ncu, stream);
        else if (!gs && us)  ok = iq3g_launch<iq3xxs_t, iq3s_t,   2, 1>(wg, wu, src1_q8_1, d, K, N, nullptr, ncu, stream);
        else                 ok = iq3g_launch<iq3xxs_t, iq3xxs_t, 2, 0>(wg, wu, src1_q8_1, d, K, N, nullptr, ncu, stream);
    }
    if (ok) {
        CUDA_CHECK(cudaGetLastError());
    }
    return ok;
}
#else
bool ggml_cuda_mmvq_iq3_fuse_mixed_ok(const ggml_tensor *, const ggml_tensor *) {
    return false;
}
bool ggml_cuda_mmvq_iq3(ggml_backend_cuda_context &, const ggml_tensor *, const block_q8_1 *, ggml_tensor *, const ggml_cuda_mm_fusion_args_host *) {
    return false;
}
#endif
