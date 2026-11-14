// Host dispatch for the single-token GEMV kernels with in-kernel q8_1 activation quantization (mmvq-fused.cuh).
#define vec_dot_iq3_s_q8_1 vec_dot_iq3_s_q8_1_original
#define vec_dot_iq3_xxs_q8_1 vec_dot_iq3_xxs_q8_1_original
#include "vecdotq.cuh"
#undef vec_dot_iq3_s_q8_1
#undef vec_dot_iq3_xxs_q8_1
#include "vecdotq-iq3-sign.cuh"
#include "iq3-repack.cuh"
#include "mmvq-fused.cuh"
#include "mmvq-fused-host.cuh"

#include <cstdlib>

using namespace mmvqf;

#include <mutex>
#include <map>

// per (device, stream) cooperative-quantization state + global q8 buffer, allocated once (outside graph capture),
// freed by ggml_cuda_mmvq_fused_release when the stream is destroyed
static std::mutex coop_mtx;
static std::map<std::pair<int, cudaStream_t>, char *> coop_bufs;

static bool get_coop(const int device, cudaStream_t stream, coop_state ** st, char ** q8) {
    std::mutex & mtx = coop_mtx;
    std::map<std::pair<int, cudaStream_t>, char *> & bufs = coop_bufs;
    static const bool disabled = [] {
        const char * v = getenv("GGML_MMVQF_COOP");
        return v != nullptr && v[0] == '0';
    }();
    if (disabled) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mtx);
    auto it = bufs.find({device, stream});
    if (it == bufs.end()) {
        hipStreamCaptureStatus status;
        CUDA_CHECK(hipStreamIsCapturing(stream, &status));
        if (status != hipStreamCaptureStatusNone) {
            return false;
        }
        char * mem = nullptr;
        const size_t bytes = 256 + lds_y_bytes<true>(COOP_KMAX);
        CUDA_CHECK(cudaMalloc(&mem, bytes));
        CUDA_CHECK(cudaMemset(mem, 0, bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
        it = bufs.emplace(std::make_pair(device, stream), mem).first;
    }
    // ticket/ready and done in separate 128 B lines
    *st = (coop_state *) it->second;
    *q8 = it->second + 256;
    return true;
}

void ggml_cuda_mmvq_fused_release(const int device, cudaStream_t stream) {
    char * mem = nullptr;
    {
        std::lock_guard<std::mutex> lock(coop_mtx);
        const auto it = coop_bufs.find({device, stream});
        if (it == coop_bufs.end()) {
            return;
        }
        mem = it->second;
        coop_bufs.erase(it);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(mem));
}

// GGML_MMVQF: 0 = off, 1 (default) = only where measured faster in situ, 2 = every supported GEMV
static int mmvqf_mode() {
    static const int mode = [] {
        const char * v = getenv("GGML_MMVQF");
        return v == nullptr ? 1 : atoi(v);
    }();
    return mode;
}
static bool mmvqf_enabled() {
    return mmvqf_mode() > 0;
}
static bool mmvqf_wanted(const ggml_type type, const int64_t N, const int64_t K) {
    if (mmvqf_mode() >= 2) {
        return true;
    }
    // LM head sized Q4_K GEMV: the persistent pipelined kernel beats mul_mat_vec_q by ~25% (595 vs 775 us)
    return type == GGML_TYPE_Q4_K && N >= 65536 && K % 1024 == 0;
}

template <class D, int NW, int L, int R, bool has_gate>
static void launch_pp(const params & p, const int ncu, const int wg_per_cu, cudaStream_t stream) {
    constexpr int G  = 64/L;
    constexpr int RW = NW*G*R;
    const int need = (p.N + RW - 1) / RW;
    const int smem_base = pp_smem_bytes<D, NW, L, R>(p.K, 0);
    int wpc = wg_per_cu;
    while (wpc > 1 && wpc*(smem_base + 4096) > 65536) {
        --wpc;
    }
    const int maxg  = std::max(1, ncu*wpc);
    const int iters = (need + maxg - 1) / maxg;
    const int grid  = (need + iters - 1) / iters;
    const int ngroups = (p.N + R - 1) / R;
    const int niter = (ngroups + grid*NW*G - 1) / (grid*NW*G);
    const int smem = pp_smem_bytes<D, NW, L, R>(p.K, niter);
    params pp = p;
    pp.coop = nullptr;
    mmvqf_pp_kernel<D, NW, L, R, has_gate><<<grid, dim3(64, NW), smem, stream>>>(pp);
}

template <class D, int NW, int L, int R, bool has_gate>
static void launch_loop(const params & p, const int ncu, const int wg_per_cu, cudaStream_t stream) {
    constexpr int G  = 64/L;
    constexpr int RW = NW*G*R;
    const int need  = (p.N + RW - 1) / RW;
    const int smem  = smem_bytes<D>(p.K);
    int wpc = wg_per_cu;
    while (wpc > 1 && wpc*smem > 65536) {
        --wpc;
    }
    const int maxg  = std::max(1, ncu*wpc);
    const int iters = (need + maxg - 1) / maxg;
    const int grid  = (need + iters - 1) / iters;
    mmvqf_kernel<D, NW, L, R, has_gate><<<grid, dim3(64, NW), smem, stream>>>(p);
}

template <class D, int NW, int L, int R, int U, bool has_gate>
static void launch_batch(const params & p, const int ncu, const int wg_per_cu, cudaStream_t stream) {
    constexpr int G  = 64/L;
    constexpr int RW = NW*G*R;
    const int need = (p.N + RW - 1) / RW;
    const int smem_base = pp_smem_bytes<D, NW, L, R>(p.K, 0);
    int wpc = wg_per_cu;
    while (wpc > 1 && wpc*(smem_base + 2048) > 65536) {
        --wpc;
    }
    const int maxg  = std::max(1, ncu*wpc);
    const int iters = (need + maxg - 1) / maxg;
    const int grid  = (need + iters - 1) / iters;
    const int ngroups = (p.N + R - 1) / R;
    const int niter = (ngroups + grid*NW*G - 1) / (grid*NW*G);
    const int smem = pp_smem_bytes<D, NW, L, R>(p.K, niter);
    params pp = p;
    if (grid < (p.K/8 + NW*64 - 1)/(NW*64)) {
        pp.coop = nullptr;  // not enough workgroups to cover the quantizer slices
    }
    mmvqf_batch_kernel<D, NW, L, R, U, has_gate><<<grid, dim3(64, NW), smem, stream>>>(pp);
}

// shape-dependent configuration (tuned on MI210 with real weights)
template <class D, int L, int WPC_BIGK, bool has_gate>
static void launch_batch_auto(const params & p, const int ncu, cudaStream_t stream) {
    if (p.K >= 16384) {
        launch_batch<D, 16, L, 1, 4, has_gate>(p, ncu, WPC_BIGK, stream);
    } else if (p.N >= 4096) {
        launch_batch<D, 8, L, 1, 2, has_gate>(p, ncu, 4, stream);
    } else {
        launch_batch<D, 4, L, 1, 2, has_gate>(p, ncu, 8, stream);
    }
}

template <ggml_type type, bool has_gate>
static void launch_generic(const params & p, const int ncu, cudaStream_t stream) {
    launch_loop<dot_gen<type>, 4, 32, 2, has_gate>(p, ncu, 4, stream);
}

template <bool has_gate>
static bool dispatch(const ggml_type type, const bool packed, const params & p, const int ncu, cudaStream_t stream) {
    const int K = p.K;
    switch (type) {
        case GGML_TYPE_IQ3_S:
            if (packed) {
                if ((K/32) % 32) return false;
                launch_batch_auto<pipe_pk<false>, 32, 2, has_gate>(p, ncu, stream);
            } else {
                launch_generic<GGML_TYPE_IQ3_S, has_gate>(p, ncu, stream);
            }
            return true;
        case GGML_TYPE_IQ3_XXS:
            if (packed) {
                if ((K/32) % 32) return false;
                launch_batch_auto<pipe_pk<true>, 32, 2, has_gate>(p, ncu, stream);
            } else {
                launch_generic<GGML_TYPE_IQ3_XXS, has_gate>(p, ncu, stream);
            }
            return true;
        case GGML_TYPE_IQ4_XS:
            if ((K/32) % 32) return false;
            launch_batch_auto<pipe_iq4xs, 32, 1, has_gate>(p, ncu, stream);
            return true;
        case GGML_TYPE_Q4_K:
            if ((K/64) % 16) return false;
            if (p.N >= 65536) {
                launch_pp<pipe_q4k, 8, 16, 1, has_gate>(p, ncu, 2, stream);
            } else {
                launch_batch_auto<pipe_q4k, 16, 1, has_gate>(p, ncu, stream);
            }
            return true;
        case GGML_TYPE_Q2_K:    launch_generic<GGML_TYPE_Q2_K,    has_gate>(p, ncu, stream); return true;
        case GGML_TYPE_Q3_K:    launch_generic<GGML_TYPE_Q3_K,    has_gate>(p, ncu, stream); return true;
        case GGML_TYPE_Q5_K:    launch_generic<GGML_TYPE_Q5_K,    has_gate>(p, ncu, stream); return true;
        case GGML_TYPE_Q6_K:    launch_generic<GGML_TYPE_Q6_K,    has_gate>(p, ncu, stream); return true;
        case GGML_TYPE_IQ2_XXS: launch_generic<GGML_TYPE_IQ2_XXS, has_gate>(p, ncu, stream); return true;
        case GGML_TYPE_IQ2_XS:  launch_generic<GGML_TYPE_IQ2_XS,  has_gate>(p, ncu, stream); return true;
        case GGML_TYPE_IQ2_S:   launch_generic<GGML_TYPE_IQ2_S,   has_gate>(p, ncu, stream); return true;
        case GGML_TYPE_IQ1_M:   launch_generic<GGML_TYPE_IQ1_M,   has_gate>(p, ncu, stream); return true;
        case GGML_TYPE_IQ4_NL:  launch_generic<GGML_TYPE_IQ4_NL,  has_gate>(p, ncu, stream); return true;
        case GGML_TYPE_Q4_0:    launch_generic<GGML_TYPE_Q4_0,    has_gate>(p, ncu, stream); return true;
        case GGML_TYPE_Q8_0:    launch_generic<GGML_TYPE_Q8_0,    has_gate>(p, ncu, stream); return true;
        default:
            return false;
    }
}

static bool type_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_mmvq_fused_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (!mmvqf_enabled()) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_AMD(cc) || ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != 64) {
        return false;
    }
    if (!type_supported(src0->type) || !mmvqf_wanted(src0->type, src0->ne[1], src0->ne[0])) {
        return false;
    }
    const int64_t K = src0->ne[0];
    if (K % 256 != 0 || K > 32768 || src0->ne[1] > INT32_MAX/2) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || src1->ne[0] != K) {
        return false;
    }
    if (!ggml_is_matrix(src0) || !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (((uintptr_t) src1->data) % 16 != 0) {
        return false;
    }
    return true;
}

// Q4_K single-token GEMV on the persistent pipelined kernel with the activation already quantized (activation cache):
// the prologue only stages q8_1 into LDS. Used for all Q4_K shapes with enough rows.
bool ggml_cuda_mmvq_fused_preq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const block_q8_1 * q8, const int * sums,
        ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    static const bool enabled = [] {
        const char * v = getenv("GGML_MMVQF_Q4K");
        return v == nullptr || v[0] != '0';
    }();
    static const int iq4xs_cfg = [] {   // experimental: 0 off, 1..3 configs
        const char * v = getenv("GGML_MMVQF_IQ4XS");
        return v ? atoi(v) : 0;
    }();
    const bool is_q4k   = src0->type == GGML_TYPE_Q4_K && enabled;
    const bool is_iq4xs = src0->type == GGML_TYPE_IQ4_XS && iq4xs_cfg > 0;
    if (!mmvqf_enabled() || (!is_q4k && !is_iq4xs) || !q8 || !sums) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    if (K % 1024 != 0 || K > 32768 || N < 4096 || N > INT32_MAX/2 || !ggml_is_matrix(src0) || !ggml_is_contiguous(src0) ||
            !ggml_is_contiguous(dst) || dst->ne[1] != 1 || ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != 64) {
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
    params p{};
    p.q8aos      = q8;
    p.q8sums     = sums;
    p.residual   = residual;
    p.dst        = (float *) dst->data;
    p.K          = (int) K;
    p.N          = (int) N;
    p.vx         = (const char *) src0->data;
    p.stride_row = (int) (src0->nb[1] / ggml_type_size(src0->type));
    const int ncu = ggml_cuda_info().devices[device].nsm;
    if (is_q4k) {
        launch_pp<pipe_q4k, 8, 16, 1, false>(p, ncu, 2, ctx.stream());
    } else if (iq4xs_cfg == 1) {
        launch_pp<pipe_iq4xs, 8, 32, 2, false>(p, ncu, 1, ctx.stream());
    } else if (iq4xs_cfg == 2) {
        launch_pp<pipe_iq4xs, 8, 32, 1, false>(p, ncu, 2, ctx.stream());
    } else {
        launch_pp<pipe_iq4xs, 4, 32, 1, false>(p, ncu, 4, ctx.stream());
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_mmvq_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion, const float * other, const int epi) {
    if (!ggml_cuda_mmvq_fused_supported(src0, src1, dst)) {
        return false;
    }
    const ggml_tensor * gate = nullptr;
    const float * residual = nullptr;
    if (fusion) {
        if (fusion->gate_bias || fusion->x_scale || fusion->gate_scale || fusion->shared_up) {
            return false;
        }
        if (fusion->gate) {
            if (fusion->glu_op != GGML_GLU_OP_SWIGLU || fusion->x_bias || fusion->gate->type != src0->type ||
                    !ggml_are_same_shape(fusion->gate, src0) || !ggml_are_same_stride(fusion->gate, src0) ||
                    !ggml_is_contiguous(fusion->gate)) {
                return false;
            }
            gate = fusion->gate;
        }
        if (fusion->x_bias) {
            if (fusion->x_bias->type != GGML_TYPE_F32 || !ggml_are_same_shape(fusion->x_bias, dst) || !ggml_is_contiguous(fusion->x_bias)) {
                return false;
            }
            residual = (const float *) fusion->x_bias->data;
        }
    }
    if (epi != EPI_NONE && (gate || residual || !other)) {
        return false;
    }

    const int device = ggml_cuda_get_device();
    cudaStream_t stream = ctx.stream();
    const int ncu = ggml_cuda_info().devices[device].nsm;

    params p{};
    p.x        = (const float *) src1->data;
    p.residual = residual;
    p.other    = other;
    p.dst      = (float *) dst->data;
    p.K        = (int) src0->ne[0];
    p.N        = (int) src0->ne[1];
    p.epi      = epi;

    if (!get_coop(device, stream, &p.coop, &p.q8)) {
        p.coop = nullptr;
        p.q8   = nullptr;
    }

    bool packed = false;
#if defined(GGML_USE_HIP)
    if (src0->type == GGML_TYPE_IQ3_S || src0->type == GGML_TYPE_IQ3_XXS) {
        const char * up = (const char *) codex_iq3_get_packed(src0, device, stream);
        const char * gp = gate ? (const char *) codex_iq3_get_packed(gate, device, stream) : nullptr;
        if (up && (!gate || gp)) {
            const size_t count = ggml_nelements(src0)/256;
            p.vx = up;
            p.mx = up + count*128;
            if (gate) {
                p.vg = gp;
                p.mg = gp + count*128;
            }
            p.stride_row = p.K/32;
            packed = true;
        }
    }
#endif
    if (!packed) {
        p.vx = (const char *) src0->data;
        p.vg = gate ? (const char *) gate->data : nullptr;
        p.stride_row = (int) (src0->nb[1] / ggml_type_size(src0->type));
    }

    const bool ok = gate ? dispatch<true>(src0->type, packed, p, ncu, stream) : dispatch<false>(src0->type, packed, p, ncu, stream);
    if (ok) {
        CUDA_CHECK(cudaGetLastError());
    }
    return ok;
}
