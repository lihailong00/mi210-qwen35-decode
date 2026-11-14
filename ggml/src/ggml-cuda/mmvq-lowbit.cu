// Single-token GEMV for low-bit types (IQ2_S, IQ2_XS, IQ2_XXS, Q2_K, IQ1_M) using the SoA re-layout kernels of the
// gemv-lowbit workstream (lowbit-gemv.cuh). The re-laid-out copy (same size, lossless) is created once per weight
// tensor and cached with the weight buffer; the activation comes from the regular quantize_q8_1 step.
#include "common.cuh"
#include "vecdotq.cuh"
#include "iq3-repack.cuh"
#include "mmvq-fused-host.cuh"

#if defined(GGML_USE_HIP)
#include "lowbit-gemv.cuh"
#include <cstdlib>

template <typename block_t, void (*K)(const block_t *, uint8_t *, size_t)>
static void lowbit_repack_launch(const void * src, void * dst, const ggml_tensor * t, hipStream_t stream) {
    const size_t nb = ggml_nelements(t)/QK_K;
    K<<<(nb + 255)/256, 256, 0, stream>>>((const block_t *) src, (uint8_t *) dst, nb);
}

static bool lowbit_enabled() {
    static const bool enabled = [] {
        const char * v = getenv("GGML_MMVQ_LOWBIT");
        return v == nullptr || v[0] != '0';
    }();
    return enabled;
}

template <class T>
static void lowbit_launch(const lb::soa & s, const block_q8_1 * y, float * dst, const int K, const int N, const float * residual,
        const int ncu, hipStream_t stream) {
    constexpr int L = 16, R = 2, NW = 16;
    const size_t smem = lb::gemv_lds_bytes<T>(K);
    lb::gemv<T, L, R, NW, false><<<ncu, dim3(64, NW), smem, stream>>>(s, y, dst, K, N, residual);
}

bool ggml_cuda_mmvq_lowbit(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const block_q8_1 * src1_q8_1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    if (!lowbit_enabled()) {
        return false;
    }
    const ggml_type type = src0->type;
    if (type != GGML_TYPE_IQ2_S && type != GGML_TYPE_IQ2_XS && type != GGML_TYPE_IQ2_XXS && type != GGML_TYPE_Q2_K && type != GGML_TYPE_IQ1_M) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    // stock mmvq is faster for small N (latency bound); lb kernels need N % 8 == 0
    if (K % QK_K != 0 || N % 8 != 0 || N < 2048 || K > 32768 || !ggml_is_matrix(src0) || !ggml_is_contiguous(src0) ||
            !ggml_is_contiguous(dst) || dst->ne[1] != 1) {
        return false;
    }
    const float * residual = nullptr;
    if (fusion) {
        if (fusion->gate || fusion->gate_bias || fusion->x_scale || fusion->gate_scale || fusion->shared_up) {
            return false;
        }
        if (fusion->x_bias) {
            if (fusion->x_bias->type != GGML_TYPE_F32 || !ggml_are_same_shape(fusion->x_bias, dst) || !ggml_is_contiguous(fusion->x_bias)) {
                return false;
            }
            residual = (const float *) fusion->x_bias->data;
        }
    }
    const int device = ggml_cuda_get_device();
    if (!GGML_CUDA_CC_IS_CDNA2(ggml_cuda_info().devices[device].cc)) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    ggml_cuda_repack_fn fn = nullptr;
    switch (type) {
        case GGML_TYPE_IQ2_S:   fn = lowbit_repack_launch<block_iq2_s,   lb::repack_iq2_s>;   break;
        case GGML_TYPE_IQ2_XS:  fn = lowbit_repack_launch<block_iq2_xs,  lb::repack_iq2_xs>;  break;
        case GGML_TYPE_IQ2_XXS: fn = lowbit_repack_launch<block_iq2_xxs, lb::repack_iq2_xxs>; break;
        case GGML_TYPE_Q2_K:    fn = lowbit_repack_launch<block_q2_K,    lb::repack_q2_K>;    break;
        case GGML_TYPE_IQ1_M:   fn = lowbit_repack_launch<block_iq1_m,   lb::repack_iq1_m>;   break;
        default: return false;
    }
    const void * soa_data = ggml_cuda_get_repacked(src0, device, stream, 0x100 + (int) type, ggml_nbytes(src0), fn);
    if (!soa_data) {
        return false;
    }
    const lb::soa s = lb::soa_make(type, soa_data, (size_t) ggml_nelements(src0)/QK_K);
    const int ncu = ggml_cuda_info().devices[device].nsm;
    float * dst_d = (float *) dst->data;
    switch (type) {
        case GGML_TYPE_IQ2_S:   lowbit_launch<lb::iq2_s_t<true>>  (s, src1_q8_1, dst_d, (int) K, (int) N, residual, ncu, stream); break;
        case GGML_TYPE_IQ2_XS:  lowbit_launch<lb::iq2_xs_t<true>> (s, src1_q8_1, dst_d, (int) K, (int) N, residual, ncu, stream); break;
        case GGML_TYPE_IQ2_XXS: lowbit_launch<lb::iq2_xxs_t<true>>(s, src1_q8_1, dst_d, (int) K, (int) N, residual, ncu, stream); break;
        case GGML_TYPE_Q2_K:    lowbit_launch<lb::q2_K>           (s, src1_q8_1, dst_d, (int) K, (int) N, residual, ncu, stream); break;
        case GGML_TYPE_IQ1_M:   lowbit_launch<lb::iq1_m>          (s, src1_q8_1, dst_d, (int) K, (int) N, residual, ncu, stream); break;
        default: return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
#else
bool ggml_cuda_mmvq_lowbit(ggml_backend_cuda_context &, const ggml_tensor *, const block_q8_1 *, ggml_tensor *, const ggml_cuda_mm_fusion_args_host *) {
    return false;
}
#endif
