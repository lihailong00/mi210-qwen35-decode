// Single-token GEMV for Q6_K (the MTP draft layer of qwen35: eh_proj, attn q/k/v/output, ffn gate/up/down) on the
// q6k SoA re-layout kernels (q6k-gemv.cuh): lossless same-size copy of the weights (created once per tensor outside
// stream capture, cached with the weight buffer), q8_1 activation from the shared per-stream cache, residual epilogue
// (x_bias) and fused same-type gate/up SwiGLU. Not bit-identical to the stock Q6_K kernel (different fp32 accumulation
// order, deterministic). GGML_MMVQ_Q6K=0 disables (falls back to the previous paths exactly).
#include "common.cuh"
#include "iq3-repack.cuh"
#include "mmvq-fused-host.cuh"

#if defined(GGML_USE_HIP)
#include "q6k-gemv.cuh"
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <map>
#include <vector>
#include <cstring>

static bool q6k_enabled() {
    static const bool enabled = [] {
        const char * v = getenv("GGML_MMVQ_Q6K");
        return v == nullptr || v[0] != '0';
    }();
    return enabled;
}

static int q6k_ncu() { return ggml_cuda_info().devices[ggml_cuda_get_device()].nsm; }

// GGML_MMVQ_Q6K_CFG="N,K,L,R,NW,mode;..." (N or K = 0: any; mode 0 row-interleaved, 1 round-tiled, 2 stream-K) overrides
// the tuned config of the matching shapes (first match; the copy layout follows, so set it for the whole process)
struct q6k_ovr { int N, K; q6k::cfg c; };
static const std::vector<q6k_ovr> & q6k_overrides() {
    static const std::vector<q6k_ovr> v = [] {
        std::vector<q6k_ovr> r;
        const char * e = getenv("GGML_MMVQ_Q6K_CFG");
        while (e && *e) {
            int N = 0, K = 0, L = 0, R = 0, NW = 0, m = 0;
            if (sscanf(e, "%d,%d,%d,%d,%d,%d", &N, &K, &L, &R, &NW, &m) == 6 && (L == 16 || L == 32) && (R == 1 || R == 2) &&
                    NW >= 1 && NW <= 16 && m >= 0 && m <= 2) {
                r.push_back({N, K, q6k::cfg{L, R, NW, 0, m == 2, m >= 1}});
            }
            e = strchr(e, ';');
            e = e ? e + 1 : nullptr;
        }
        return r;
    }();
    return v;
}
static q6k::cfg q6k_pick(int N, int K, bool glu) {
    const int ncu = q6k_ncu();
    q6k::cfg c = q6k::pick_cfg(N, K, ncu, glu);
    for (const q6k_ovr & o : q6k_overrides()) {
        if ((o.N == 0 || o.N == N) && (o.K == 0 || o.K == K)) {
            q6k::cfg oc = o.c; oc.grid = ncu;
            if (q6k::shape_ok(oc, K, N, glu)) {
                c = oc;
            }
            break;
        }
    }
    return c;
}

static void q6k_repack_single(const void * src, void * dst, const ggml_tensor * t, hipStream_t stream) {
    q6k::repack_cfg(src, dst, t->ne[1], (int) t->ne[0], q6k_pick((int) t->ne[1], (int) t->ne[0], false), stream);
}
static void q6k_repack_glu(const void * src, void * dst, const ggml_tensor * t, hipStream_t stream) {
    q6k::repack_cfg(src, dst, t->ne[1], (int) t->ne[0], q6k_pick((int) t->ne[1], (int) t->ne[0], true), stream);
}

// per-device scratch + zeroed counters for the stream-K split-unit combine (allocated outside graph capture)
static bool q6k_scratch(const int device, cudaStream_t stream, float ** scratch, unsigned int ** cnt) {
    static std::mutex mtx;
    static std::map<int, char *> bufs;
    std::lock_guard<std::mutex> lock(mtx);
    auto it = bufs.find(device);
    if (it == bufs.end()) {
        hipStreamCaptureStatus status;
        CUDA_CHECK(hipStreamIsCapturing(stream, &status));
        if (status != hipStreamCaptureStatusNone) {
            return false;
        }
        char * mem = nullptr;
        const size_t bytes = q6k::SCRATCH_FLOATS*4 + q6k::CNT_UINTS*4;
        CUDA_CHECK(cudaMalloc(&mem, bytes));
        CUDA_CHECK(cudaMemset(mem, 0, bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
        it = bufs.emplace(device, mem).first;
    }
    *scratch = (float *) it->second;
    *cnt     = (unsigned int *) (it->second + q6k::SCRATCH_FLOATS*4);
    return true;
}

static bool q6k_weight_ok(const ggml_tensor * w) {
    if (!q6k_enabled() || w->type != GGML_TYPE_Q6_K) {
        return false;
    }
    const int64_t K = w->ne[0], N = w->ne[1];
    return K % QK_K == 0 && K <= 32768 && N >= 64 && ggml_is_matrix(w) && ggml_is_contiguous(w) && w->buffer &&
        ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        GGML_CUDA_CC_IS_CDNA2(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
}

// does the q6k path want this single-token GEMV (keeps the in-kernel-quantizing mmvqf path away from it)
bool ggml_cuda_mmvq_q6k_wants(const ggml_tensor * src0) {
    if (!q6k_weight_ok(src0)) {
        return false;
    }
    const int K = (int) src0->ne[0], N = (int) src0->ne[1];
    return q6k::shape_ok(q6k_pick(N, K, false), K, N, false);
}

bool ggml_cuda_mmvq_q6k(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const block_q8_1 * src1_q8_1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    if (!q6k_weight_ok(src0) || !ggml_is_contiguous(dst) || dst->ne[1] != 1) {
        return false;
    }
    const int K = (int) src0->ne[0], N = (int) src0->ne[1];
    const float * residual = nullptr;
    const ggml_tensor * gate = nullptr;
    if (fusion) {
        if (fusion->gate_bias || fusion->x_scale || fusion->gate_scale || fusion->shared_up) {
            return false;
        }
        if (fusion->gate) {
            if (fusion->glu_op != GGML_GLU_OP_SWIGLU || fusion->x_bias || !q6k_weight_ok(fusion->gate) ||
                    !ggml_are_same_shape(fusion->gate, src0)) {
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
    const q6k::cfg c = q6k_pick(N, K, gate != nullptr);
    if (!q6k::shape_ok(c, K, N, gate != nullptr)) {
        return false;
    }
    const int device = ggml_cuda_get_device();
    cudaStream_t stream = ctx.stream();
    float * scratch = nullptr;
    unsigned int * cnt = nullptr;
    if (!q6k_scratch(device, stream, &scratch, &cnt)) {
        return false;
    }
    // cache key: type + layout of the copy (L, R, waves, mode)
    const int mode = c.sk ? 2 : (c.rt ? 1 : 0);
    const int kind = 0x6000000 + (gate ? 0x2000000 : 0) + (c.L/16)*0x100000 + c.R*0x10000 + (c.grid*c.NW)*0x4 + mode;
    const void * up_soa = ggml_cuda_get_repacked(src0, device, stream, kind, ggml_nbytes(src0), gate ? q6k_repack_glu : q6k_repack_single);
    if (!up_soa) {
        return false;
    }
    const size_t nb = (size_t) N*(K/QK_K);
    const q6k::soa s = q6k::soa_make(up_soa, nb);
    bool ok;
    if (gate) {
        const void * gate_soa = ggml_cuda_get_repacked(gate, device, stream, kind, ggml_nbytes(gate), q6k_repack_glu);
        if (!gate_soa) {
            return false;
        }
        ok = q6k::dispatch<true>(c, s, q6k::soa_make(gate_soa, nb), src1_q8_1, (float *) dst->data, K, N, stream, nullptr, scratch, cnt);
    } else {
        ok = q6k::dispatch<false>(c, s, s, src1_q8_1, (float *) dst->data, K, N, stream, residual, scratch, cnt);
    }
    if (!ok) {
        return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// K split of a multi-column launch whose activation records do not fit in LDS (ffn_down, K = 17408, 3-4 columns): two
// launches over K ranges [0, K1) and [K1, K) (K1 = a multiple of 1024 near K/2), the second adds onto the first result.
// The split copy holds both ranges back to back (same total size), a cache entry of its own.
static int q6k_split_k1(int K) { return (K/2/1024)*1024; }
static void q6k_repack_split(const void * src, void * dst, const ggml_tensor * t, hipStream_t stream) {
    const int K = (int) t->ne[0], N = (int) t->ne[1], K1 = q6k_split_k1(K), K2 = K - K1;
    q6k::repack_cfg(src, dst, N, K1, q6k_pick(N, K1, false), stream, K, 0);
    q6k::repack_cfg(src, (char *) dst + (size_t) N*(K1/QK_K)*210, N, K2, q6k_pick(N, K2, false), stream, K, K1);
}

// N-column (2..q6k::NC_MAX) unfused Q6_K GEMVs (the MTP catch-up batch through the draft layer): the weights are read
// once for all columns, on the same copy as the single-token launcher. y column c at y + c*GGML_PAD(K, 512)/32 blocks,
// dst column c at dst->data + c*dst->nb[1]. false: nothing launched (shape, LDS, missing copy during capture).
// Opt-in (GGML_MMVQ_Q6K_NC=1; needs GGML_MMVQ_Q6K on): with the KV-only draft graph the catch-up batch keeps only
// eh_proj / k / v at 2..4 columns (about 40 us per round), no measured end-to-end gain, and the columns are not
// bit-identical to the single-token q6k kernel (K-split launches, other accumulation order): it only moves the draft
// numerics. Draft-only use; no GGML_MMVQ_NC_VERIFY hook. When off, Q6_K is not an N-column type (mmvq.cu
// ggml_cuda_mmvq_nc_type): N-column Q6_K MUL_MATs quantize their activation in the pool and run the stock kernels
// exactly as before.
bool ggml_cuda_mmvq_q6k_nc_enabled() {
    static const bool enabled = [] {
        const char * v = getenv("GGML_MMVQ_Q6K_NC");
        return v != nullptr && v[0] == '1';
    }();
    return enabled && q6k_enabled();
}
bool ggml_cuda_mmvq_q6k_nc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const block_q8_1 * y, int ncols, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    if (!ggml_cuda_mmvq_q6k_nc_enabled() || fusion || ncols < 2 || ncols > q6k::NC_MAX || !q6k_weight_ok(src0) || dst->type != GGML_TYPE_F32 ||
            dst->ne[1] != ncols || dst->ne[2] != 1 || dst->ne[3] != 1 || dst->nb[0] != sizeof(float) || dst->nb[1] % sizeof(float) != 0) {
        return false;
    }
    const int K = (int) src0->ne[0], N = (int) src0->ne[1];
    const q6k::cfg c = q6k_pick(N, K, false);
    const int device = ggml_cuda_get_device();
    cudaStream_t stream = ctx.stream();
    const int ys = (int) (GGML_PAD(K, MATRIX_ROW_PADDING)/QK8_1);
    const int ds = (int) (dst->nb[1]/sizeof(float));
    float * scratch = nullptr;
    unsigned int * cnt = nullptr;
    if (!q6k::shape_ok(c, K, N, false, ncols)) {
        const int K1 = q6k_split_k1(K), K2 = K - K1;
        if (K1 < 1024 || K2 % 1024 != 0) {
            return false;
        }
        const q6k::cfg c1 = q6k_pick(N, K1, false), c2 = q6k_pick(N, K2, false);
        if (!q6k::shape_ok(c1, K1, N, false, ncols) || !q6k::shape_ok(c2, K2, N, false, ncols) || !q6k_scratch(device, stream, &scratch, &cnt)) {
            return false;
        }
        const void * soa = ggml_cuda_get_repacked(src0, device, stream, 0x7000000, ggml_nbytes(src0), q6k_repack_split);
        if (!soa) {
            return false;
        }
        const size_t nb1 = (size_t) N*(K1/QK_K), nb2 = (size_t) N*(K2/QK_K);
        const q6k::soa s1 = q6k::soa_make(soa, nb1), s2 = q6k::soa_make((const char *) soa + nb1*210, nb2);
        float * d = (float *) dst->data;
        if (!q6k::dispatch<false>(c1, s1, s1, y, d, K1, N, stream, nullptr, scratch, cnt, ncols, ys, ds) ||
                !q6k::dispatch<false>(c2, s2, s2, y + K1/QK8_1, d, K2, N, stream, d, scratch, cnt, ncols, ys, ds)) {
            return false;
        }
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    if (!q6k_scratch(device, stream, &scratch, &cnt)) {
        return false;
    }
    const int mode = c.sk ? 2 : (c.rt ? 1 : 0);
    const int kind = 0x6000000 + (c.L/16)*0x100000 + c.R*0x10000 + (c.grid*c.NW)*0x4 + mode;   // = the single launcher's copy
    const void * soa = ggml_cuda_get_repacked(src0, device, stream, kind, ggml_nbytes(src0), q6k_repack_single);
    if (!soa) {
        return false;
    }
    const q6k::soa s = q6k::soa_make(soa, (size_t) N*(K/QK_K));
    if (!q6k::dispatch<false>(c, s, s, y, (float *) dst->data, K, N, stream, nullptr, scratch, cnt, ncols, ys, ds)) {
        return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_mmvq_q6k_nc_enabled() {
    return false;
}
bool ggml_cuda_mmvq_q6k_nc(ggml_backend_cuda_context &, const ggml_tensor *, const block_q8_1 *, int, ggml_tensor *,
        const ggml_cuda_mm_fusion_args_host *) {
    return false;
}
bool ggml_cuda_mmvq_q6k_wants(const ggml_tensor *) {
    return false;
}
bool ggml_cuda_mmvq_q6k(ggml_backend_cuda_context &, const ggml_tensor *, const block_q8_1 *, ggml_tensor *,
        const ggml_cuda_mm_fusion_args_host *) {
    return false;
}

#endif // defined(GGML_USE_HIP)
