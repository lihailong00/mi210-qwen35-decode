// Single-token GEMV for low-bit types (IQ2_S, IQ2_XS, IQ2_XXS, Q2_K, IQ1_M) using the SoA re-layout kernels of the
// gemv-lowbit workstream (lowbit-gemv.cuh). The re-laid-out copy (same size, lossless) is created once per weight
// tensor and cached with the weight buffer; the activation comes from the regular quantize_q8_1 step.
#include "common.cuh"
#include "vecdotq.cuh"
#include "iq3-repack.cuh"
#include "mmvq-fused-host.cuh"

#if defined(GGML_USE_HIP)
#include <cstdlib>
#include "lowbit-gemv.cuh"
#include "quantize.cuh"   // MMVQ-NC lowbit (verify)
#include <algorithm>
#include <atomic>
#include <vector>

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
    // lb::launch needs N % 64 == 0 (faster than stock mmvq even for 1024 rows)
    if (K % QK_K != 0 || N % 64 != 0 || K > 32768 || !ggml_is_matrix(src0) || !ggml_is_contiguous(src0) ||
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
    float * dst_d = (float *) dst->data;
    if (!lb::launch(type, soa_data, src1_q8_1, dst_d, (int) K, (int) N, stream, residual)) {  // per-shape tuned configs
        return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// ================================================================================================ MMVQ-NC lowbit
// N-column GEMVs (ne11 = 2..nc_max) on the same SoA copy (same repack kind, no extra VRAM): one pass over the weights
// for all columns, per column bit-identical to ggml_cuda_mmvq_lowbit on that column alone (see lb::gemv_nc).
// Switches: GGML_MMVQ_NC=0 (all families) / GGML_MMVQ_NC_LOWBIT=0 / GGML_MMVQ_LOWBIT=0 -> previous behaviour.
// GGML_MMVQ_NC_MAX (default 4, 2..6). Debug: GGML_MMVQ_NC_VERIFY=n checks the first n launches column by column
// against the N=1 launcher on a separately quantized column (bitwise); GGML_MMVQ_NC_BENCH=reps times NC vs N=1.
static int lowbit_env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}
// (master switch GGML_MMVQ_NC and GGML_MMVQ_NC_MAX: the shared parse in mmvq.cu)
static bool lowbit_nc_enabled() {
    static const bool enabled = lowbit_enabled() && ggml_cuda_mmvq_nc_enabled() && lowbit_env_int("GGML_MMVQ_NC_LOWBIT", 1) != 0;
    return enabled;
}
static int lowbit_nc_max() {
    return std::min(ggml_cuda_mmvq_nc_max(), LB_NC_MAX);
}
static bool lowbit_type(ggml_type type) {
    return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_Q2_K || type == GGML_TYPE_IQ1_M;
}
static bool lowbit_nc_shape_ok(const ggml_tensor * w, int ncols) {
    if (!lowbit_nc_enabled() || ncols < 2 || ncols > lowbit_nc_max() || !lowbit_type(w->type)) {
        return false;
    }
    const int64_t K = w->ne[0];
    const int64_t N = w->ne[1];
    // lb::launch_ok: the single-token path runs this kernel for the shape too (it is the per-column reference)
    return K % QK_K == 0 && N % 64 == 0 && K <= 32768 && ggml_is_matrix(w) && ggml_is_contiguous(w) &&
           lb::launch_ok(w->type, (int) K, (int) N) &&
           GGML_CUDA_CC_IS_CDNA2(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
}

bool ggml_cuda_mmvq_lowbit_nc_ok(const ggml_tensor * w, const ggml_tensor * gate_w, bool residual, int ncols) {
    GGML_UNUSED(residual);   // plain and residual epilogues, as at N=1
    return gate_w == nullptr && lowbit_nc_shape_ok(w, ncols) && ggml_cuda_has_repacked(w, 0x100 + (int) w->type);
}

// the SoA copy of w (the one the N=1 path uses; created here unless the stream is capturing)
static const void * lowbit_nc_soa(ggml_backend_cuda_context & ctx, const ggml_tensor * w) {
    ggml_cuda_repack_fn fn = nullptr;
    switch (w->type) {
        case GGML_TYPE_IQ2_S:   fn = lowbit_repack_launch<block_iq2_s,   lb::repack_iq2_s>;   break;
        case GGML_TYPE_IQ2_XS:  fn = lowbit_repack_launch<block_iq2_xs,  lb::repack_iq2_xs>;  break;
        case GGML_TYPE_IQ2_XXS: fn = lowbit_repack_launch<block_iq2_xxs, lb::repack_iq2_xxs>; break;
        case GGML_TYPE_Q2_K:    fn = lowbit_repack_launch<block_q2_K,    lb::repack_q2_K>;    break;
        case GGML_TYPE_IQ1_M:   fn = lowbit_repack_launch<block_iq1_m,   lb::repack_iq1_m>;   break;
        default: return nullptr;
    }
    return ggml_cuda_get_repacked(w, ggml_cuda_get_device(), ctx.stream(), 0x100 + (int) w->type, ggml_nbytes(w), fn);
}

// called by the fusion predicate before _nc_ok: make the SoA copy now if it does not exist yet (not possible while
// the stream is capturing; _nc_ok is then false), so that a fused N-column call works on the first evaluation too
void ggml_cuda_mmvq_lowbit_nc_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * w, int ncols) {
    if (lowbit_nc_shape_ok(w, ncols)) {
        (void) lowbit_nc_soa(ctx, w);
    }
}

// the MUL_MAT node of dst (dst itself or the ADD it was fused into); debug only
static const ggml_tensor * lowbit_nc_find_mm(const ggml_tensor * dst, const ggml_tensor * src0) {
    if (dst->op == GGML_OP_MUL_MAT && dst->src[0] == src0) {
        return dst;
    }
    for (int i = 0; i < 2; ++i) {
        const ggml_tensor * s = dst->src[i];
        if (s && s->op == GGML_OP_MUL_MAT && s->src[0] == src0) {
            return s;
        }
    }
    return nullptr;
}

static void lowbit_nc_check(ggml_backend_cuda_context & ctx, ggml_type type, const void * soa, const block_q8_1 * y, int ys, ggml_tensor * dst,
        int ldd, int K, int N, int nc, const float * residual, int ldr, const ggml_tensor * src0, bool verify, int bench) {
    cudaStream_t stream = ctx.stream();
    float * dst_d = (float *) dst->data;
    // residual snapshot first: the fused ADD may write in place
    ggml_cuda_pool_alloc<float> res_copy(ctx.pool());
    if (residual) {
        res_copy.alloc((size_t) nc*ldr);
        CUDA_CHECK(cudaMemcpyAsync(res_copy.get(), residual, (size_t) nc*ldr*sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
    const float * r0 = residual ? res_copy.get() : nullptr;
    GGML_ASSERT(lb::launch_nc(type, soa, y, ys, dst_d, ldd, K, N, nc, stream, r0, ldr));
    std::vector<float> got((size_t) nc*N), ref((size_t) N);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int c = 0; c < nc; ++c) {
        CUDA_CHECK(cudaMemcpy(got.data() + (size_t) c*N, dst_d + (size_t) c*ldd, N*sizeof(float), cudaMemcpyDeviceToHost));
    }
    const ggml_tensor * mm = lowbit_nc_find_mm(dst, src0);
    const ggml_tensor * src1 = mm ? mm->src[1] : nullptr;
    const int64_t K_pad = GGML_PAD(K, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char>  q1(ctx.pool(), K_pad/QK8_1*sizeof(block_q8_1));
    ggml_cuda_pool_alloc<float> d1(ctx.pool(), N);
    std::vector<char> qa(K/QK8_1*sizeof(block_q8_1)), qb(qa.size());
    if (verify) {
        long long bad = 0, bad_q8 = 0;
        for (int c = 0; c < nc; ++c) {
            const block_q8_1 * yc = y + (size_t) c*ys;
            if (src1) {   // the column quantized on its own, as at N=1
                const int64_t s11 = src1->nb[1]/sizeof(float);
                quantize_row_q8_1_cuda((const float *) src1->data + c*s11, nullptr, q1.get(), src0->type, K, s11, s11, s11, K_pad, 1, 1, 1, stream);
                CUDA_CHECK(cudaMemcpyAsync(qa.data(), q1.get(), qa.size(), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(qb.data(), yc, qb.size(), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                bad_q8 += memcmp(qa.data(), qb.data(), qa.size()) != 0;
                yc = (const block_q8_1 *) q1.get();
            }
            GGML_ASSERT(lb::launch(type, soa, yc, d1.get(), K, N, stream, r0 ? r0 + (size_t) c*ldr : nullptr));
            CUDA_CHECK(cudaMemcpyAsync(ref.data(), d1.get(), N*sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (int i = 0; i < N; ++i) {
                bad += memcmp(&ref[i], &got[(size_t) c*N + i], sizeof(float)) != 0;
            }
        }
        fprintf(stderr, "[mmvq-nc lowbit] verify %s K=%d N=%d nc=%d res=%d: mismatches=%lld/%lld q8_col_mismatch=%lld%s\n", ggml_type_name(type), K, N, nc,
                residual != nullptr, bad, (long long) nc*N, bad_q8, src1 ? "" : " (q8 not rechecked)");
    }
    if (bench > 0) {
        hipEvent_t e0, e1, e2;
        CUDA_CHECK(hipEventCreate(&e0)); CUDA_CHECK(hipEventCreate(&e1)); CUDA_CHECK(hipEventCreate(&e2));
        for (int i = 0; i < 3; ++i) lb::launch_nc(type, soa, y, ys, dst_d, ldd, K, N, nc, stream, r0, ldr);
        CUDA_CHECK(hipEventRecord(e0, stream));
        for (int i = 0; i < bench; ++i) lb::launch_nc(type, soa, y, ys, dst_d, ldd, K, N, nc, stream, r0, ldr);
        CUDA_CHECK(hipEventRecord(e1, stream));
        for (int i = 0; i < bench; ++i) lb::launch(type, soa, y, d1.get(), K, N, stream, r0);
        CUDA_CHECK(hipEventRecord(e2, stream));
        CUDA_CHECK(hipEventSynchronize(e2));
        float t_nc = 0, t_1 = 0;
        CUDA_CHECK(hipEventElapsedTime(&t_nc, e0, e1)); CUDA_CHECK(hipEventElapsedTime(&t_1, e1, e2));
        const double bytes = (double) ggml_nbytes(src0);
        fprintf(stderr, "[mmvq-nc lowbit] bench %s K=%d N=%d nc=%d: nc %.2f us  n1 %.2f us  ratio %.3f  floor(1.45TB/s) %.2f us\n", ggml_type_name(type), K, N, nc,
                1e3*t_nc/bench, 1e3*t_1/bench, t_nc/t_1, bytes/1.45e12*1e6);
        CUDA_CHECK(hipEventDestroy(e0)); CUDA_CHECK(hipEventDestroy(e1)); CUDA_CHECK(hipEventDestroy(e2));
        // the result of the real launch (dst may alias the residual: recompute from the snapshot)
        GGML_ASSERT(lb::launch_nc(type, soa, y, ys, dst_d, ldd, K, N, nc, stream, r0, ldr));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool ggml_cuda_mmvq_lowbit_nc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const block_q8_1 * y, int ncols, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    if (!lowbit_nc_shape_ok(src0, ncols)) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32 || dst->ne[0] != src0->ne[1] || dst->ne[1] != ncols || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            dst->nb[0] != sizeof(float)) {
        return false;
    }
    const float * residual = nullptr;
    int ldr = 0;
    if (fusion) {
        if (fusion->gate || fusion->gate_bias || fusion->x_scale || fusion->gate_scale || fusion->shared_up) {
            return false;
        }
        if (fusion->x_bias) {
            const ggml_tensor * xb = fusion->x_bias;
            if (xb->type != GGML_TYPE_F32 || !ggml_are_same_shape(xb, dst) || xb->nb[0] != sizeof(float)) {
                return false;
            }
            residual = (const float *) xb->data;
            ldr = (int) (xb->nb[1]/sizeof(float));
        }
    }
    const ggml_type type = src0->type;
    cudaStream_t stream = ctx.stream();
    const void * soa_data = lowbit_nc_soa(ctx, src0);
    if (!soa_data) {
        return false;
    }
    const int K = (int) src0->ne[0];
    const int N = (int) src0->ne[1];
    const int ys  = (int) (GGML_PAD(K, MATRIX_ROW_PADDING)/QK8_1);   // column stride of the quantized activation
    const int ldd = (int) (dst->nb[1]/sizeof(float));

    static std::atomic<int> verify_left{lowbit_env_int("GGML_MMVQ_NC_VERIFY", 0)};
    static const int bench = lowbit_env_int("GGML_MMVQ_NC_BENCH", 0);
    if (verify_left.load() > 0 || bench > 0) {
        hipStreamCaptureStatus status;
        CUDA_CHECK(hipStreamIsCapturing(stream, &status));
        if (status == hipStreamCaptureStatusNone) {
            const bool verify = verify_left.fetch_sub(1) > 0;
            lowbit_nc_check(ctx, type, soa_data, y, ys, dst, ldd, K, N, ncols, residual, ldr, src0, verify, bench);
            return true;
        }
    }
    if (!lb::launch_nc(type, soa_data, y, ys, (float *) dst->data, ldd, K, N, ncols, stream, residual, ldr)) {
        return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
// ================================================================================================ MMVQ-NC lowbit end
#else
bool ggml_cuda_mmvq_lowbit(ggml_backend_cuda_context &, const ggml_tensor *, const block_q8_1 *, ggml_tensor *, const ggml_cuda_mm_fusion_args_host *) {
    return false;
}
// MMVQ-NC lowbit
bool ggml_cuda_mmvq_lowbit_nc_ok(const ggml_tensor *, const ggml_tensor *, bool, int) {
    return false;
}
void ggml_cuda_mmvq_lowbit_nc_prepare(ggml_backend_cuda_context &, const ggml_tensor *, int) {
}
bool ggml_cuda_mmvq_lowbit_nc(ggml_backend_cuda_context &, const ggml_tensor *, const block_q8_1 *, int, ggml_tensor *, const ggml_cuda_mm_fusion_args_host *) {
    return false;
}
#endif
