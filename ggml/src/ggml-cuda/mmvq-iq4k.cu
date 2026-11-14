// Single-token GEMV for IQ4_XS and Q4_K using the SoA re-layout kernels of the gemv-iq4k workstream (iq4k-gemv.cuh):
// lossless same-size row-interleaved SoA copy of the weights (created once per tensor, cached with the weight buffer),
// q8_1 activation from the shared per-stream cache, residual epilogue (x_bias) and fused same-type gate/up SwiGLU.
// GGML_MMVQ_IQ4K=0 disables (falls back to the stock paths).
// iq4k-r3: grouped launch for GEMVs sharing the activation (ggml_cuda_mmvq_iq4k_group, detected in ggml-cuda.cu);
// GGML_MMVQ_IQ4K_GROUP=0 disables only the grouping.
#include "common.cuh"
#include "vecdotq.cuh"
#include "iq3-repack.cuh"
#include "mmvq-fused-host.cuh"

#if defined(GGML_USE_HIP)
#include "iq4k-gemv.cuh"
#include <cstdlib>
#include <mutex>
#include <map>

static bool iq4k_enabled() {
    static const bool enabled = [] {
        const char * v = getenv("GGML_MMVQ_IQ4K");
        return v == nullptr || v[0] != '0';
    }();
    return enabled;
}

// the layout of the copy (row interleave, round tiling) follows the launch configuration that will consume it
static int iq4k_ncu() { return ggml_cuda_info().devices[ggml_cuda_get_device()].nsm; }
// tuning override for the single-matrix launcher at K <= 8192 (row-interleaved layout, IL = 64/L; the cache key
// includes IL): GGML_MMVQ_IQ4K_X="L:NW" (IQ4_XS), GGML_MMVQ_IQ4K_Q="L:NW" (Q4_K). Default: iq4k::pick_cfg.
static iq4k::cfg iq4k_pick_cfg(ggml_type type, int N, int K, int ncu) {
    iq4k::cfg c = iq4k::pick_cfg(type, N, K, ncu);
    static const char * ex = getenv("GGML_MMVQ_IQ4K_X");
    static const char * eq = getenv("GGML_MMVQ_IQ4K_Q");
    const char * e = type == GGML_TYPE_IQ4_XS ? ex : eq;
    int L = 0, NW = 0;
    if (e && !c.rt && sscanf(e, "%d:%d", &L, &NW) == 2 && (L == 16 || L == 32 || L == 64) && NW >= 1 && NW <= 16) {
        iq4k::cfg o = c; o.L = L; o.NW = NW;
        if (iq4k::shape_ok(o, type, K, N)) {
            c = o;
        }
    }
    return c;
}
static void iq4k_repack_single(const void * src, void * dst, const ggml_tensor * t, hipStream_t stream) {
    const int ncu = iq4k_ncu();
    iq4k::repack_cfg(t->type, src, dst, t->ne[1], (int) t->ne[0], iq4k_pick_cfg(t->type, (int) t->ne[1], (int) t->ne[0], ncu), stream);
}
static void iq4k_repack_glu(const void * src, void * dst, const ggml_tensor * t, hipStream_t stream) {
    iq4k::repack_glu(t->type, src, dst, t->ne[1], (int) t->ne[0], stream, iq4k_ncu());
}

// per-device scratch + zeroed counters for the stream-K split-unit combine (allocated outside graph capture)
static bool iq4k_scratch(const int device, cudaStream_t stream, float ** scratch, unsigned int ** cnt) {
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
        const size_t bytes = iq4k::SCRATCH_FLOATS*4 + iq4k::CNT_UINTS*4;
        CUDA_CHECK(cudaMalloc(&mem, bytes));
        CUDA_CHECK(cudaMemset(mem, 0, bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
        it = bufs.emplace(device, mem).first;
    }
    *scratch = (float *) it->second;
    *cnt     = (unsigned int *) (it->second + iq4k::SCRATCH_FLOATS*4);
    return true;
}

// does the iq4k path want this GEMV (used to keep the in-kernel-quantizing mmvqf path away from it)
bool ggml_cuda_mmvq_iq4k_wants(const ggml_tensor * src0) {
    if (!iq4k_enabled() || (src0->type != GGML_TYPE_IQ4_XS && src0->type != GGML_TYPE_Q4_K)) {
        return false;
    }
    const int64_t K = src0->ne[0], N = src0->ne[1];
    return K % QK_K == 0 && K <= 32768 && N >= 2048 && ggml_is_matrix(src0) && ggml_is_contiguous(src0) &&
        GGML_CUDA_CC_IS_CDNA2(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) &&
        iq4k::shape_ok(iq4k_pick_cfg(src0->type, (int) N, (int) K, ggml_cuda_info().devices[ggml_cuda_get_device()].nsm), src0->type, (int) K, (int) N);
}

// ---- grouped launch (iq4k-r3): GEMVs that share the activation, one launch, round-robin units over the concatenated
// jobs (iq4k::launch_g, RRM). Uses the row-interleaved IL = 4 copy, i.e. the same copy (cache kind) as the single launcher.
static bool iq4k_group_enabled() {
    static const bool enabled = [] {
        const char * v = getenv("GGML_MMVQ_IQ4K_GROUP");
        return iq4k_enabled() && (v == nullptr || v[0] != '0');
    }();
    return enabled;
}
static int iq4k_env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}
// Groups whose members all have N <= 2048 (attn_k + attn_v) run on the small-N multi-job kernel (iq4k::launch_sn, stock
// mmvq structure on the original layout, bit-identical to the unfused path); GGML_MMVQ_IQ4K_GROUP_SMALL=0 disables.
// Groups with a large member (q + k + v, qkv + z, mixed-type gate + up) use the round-robin multi-job kernel
// (iq4k::launch_g): faster in back-to-back launches but slower than the separate tuned launches inside HIP graph
// replay (+110 us/token GPU time), so they are off unless GGML_MMVQ_IQ4K_GROUP_BIG=1 (knobs: _NW waves per workgroup,
// _RR round-robin (1) or balanced split (0)).
static int iq4k_group_small() { static const int v = iq4k_env_int("GGML_MMVQ_IQ4K_GROUP_SMALL", 1); return v; }
static int iq4k_group_big()   { static const int v = iq4k_env_int("GGML_MMVQ_IQ4K_GROUP_BIG", 0); return v; }
static int iq4k_group_nw()    { static const int v = iq4k_env_int("GGML_MMVQ_IQ4K_GROUP_NW", 12); return v; }
static int iq4k_group_rr()    { static const int v = iq4k_env_int("GGML_MMVQ_IQ4K_GROUP_RR", 1); return v; }
static void iq4k_repack_il4(const void * src, void * dst, const ggml_tensor * t, hipStream_t stream) {
    iq4k::repack_il(t->type, src, dst, t->ne[1], (int) t->ne[0], 4, stream);
}

bool ggml_cuda_mmvq_iq4k_group_ok(const ggml_tensor * mm) {
    if (!iq4k_group_enabled() || mm->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * w = mm->src[0];
    const ggml_tensor * x = mm->src[1];
    if ((w->type != GGML_TYPE_IQ4_XS && w->type != GGML_TYPE_Q4_K) || x->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t K = w->ne[0], N = w->ne[1];
    return K % QK_K == 0 && K <= 8192 && N % 64 == 0 && ggml_is_matrix(w) && ggml_is_contiguous(w) &&
        x->ne[0] == K && x->ne[1] == 1 && x->ne[2] == 1 && x->ne[3] == 1 && ggml_is_contiguous(x) &&
        ggml_is_contiguous(mm) && mm->ne[1] == 1 && w->buffer &&
        ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        GGML_CUDA_CC_IS_CDNA2(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
}

bool ggml_cuda_mmvq_iq4k_group(ggml_backend_cuda_context & ctx, ggml_tensor * const * mm, int n) {
    if (n < 2 || n > iq4k::G_MAXJ) {
        return false;
    }
    const ggml_tensor * src1 = mm[0]->src[1];
    const int K = (int) mm[0]->src[0]->ne[0];
    int64_t maxn = 0;
    for (int k = 0; k < n; ++k) {
        if (!ggml_cuda_mmvq_iq4k_group_ok(mm[k]) || mm[k]->src[1] != src1) {
            return false;
        }
        maxn = std::max<int64_t>(maxn, mm[k]->src[0]->ne[1]);
    }
    const int device = ggml_cuda_get_device();
    const int ncu = ggml_cuda_info().devices[device].nsm;
    cudaStream_t stream = ctx.stream();
    iq4k::gjob_h jobs[iq4k::G_MAXJ];
    if (maxn <= 2048) {
        if (!iq4k_group_small()) {
            return false;
        }
        const block_q8_1 * q8 = nullptr;
        if (!ggml_cuda_mmvq_q8_cached(ctx, src1, &q8)) {
            return false;
        }
        for (int k = 0; k < n; ++k) {
            const ggml_tensor * w = mm[k]->src[0];
            jobs[k] = { w->type, w->data, (float *) mm[k]->data, nullptr, (int) w->ne[1] };
        }
        if (!iq4k::launch_sn(jobs, n, q8, K, stream)) {
            return false;
        }
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    if (!iq4k_group_big()) {
        return false;
    }
    for (int k = 0; k < n; ++k) {
        const ggml_tensor * w = mm[k]->src[0];
        const int kind = 0x200 + 16*(int) w->type + 4;   // = the single launcher's copy (L = 16, IL = 4) at K <= 8192
        const void * soa = ggml_cuda_get_repacked(w, device, stream, kind, ggml_nbytes(w), iq4k_repack_il4);
        if (!soa) {
            return false;
        }
        jobs[k] = { w->type, soa, (float *) mm[k]->data, nullptr, (int) w->ne[1] };
    }
    const iq4k::gcfg c = { 16, iq4k_group_nw(), ncu, iq4k_group_rr() != 0 };
    if (!iq4k::g_supported(jobs, n, K, false, c)) {
        return false;
    }
    const block_q8_1 * q8 = nullptr;
    if (!ggml_cuda_mmvq_q8_cached(ctx, src1, &q8)) {
        return false;
    }
    if (!iq4k::launch_g(jobs, n, q8, K, stream, false, c)) {
        return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_mmvq_iq4k(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const block_q8_1 * src1_q8_1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    if (!iq4k_enabled()) {
        return false;
    }
    const ggml_type type = src0->type;
    if (type != GGML_TYPE_IQ4_XS && type != GGML_TYPE_Q4_K) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    // the 1024-row attn_k/attn_v GEMVs are latency bound: stock mmvq is faster there
    if (K % QK_K != 0 || K > 32768 || N < 2048 || !ggml_is_matrix(src0) || !ggml_is_contiguous(src0) ||
            !ggml_is_contiguous(dst) || dst->ne[1] != 1) {
        return false;
    }
    const float * residual = nullptr;
    const ggml_tensor * gate = nullptr;
    if (fusion) {
        if (fusion->gate_bias || fusion->x_scale || fusion->gate_scale || fusion->shared_up) {
            return false;
        }
        if (fusion->gate) {
            if (fusion->glu_op != GGML_GLU_OP_SWIGLU || fusion->x_bias || fusion->gate->type != type ||
                    !ggml_are_same_shape(fusion->gate, src0) || !ggml_is_contiguous(fusion->gate)) {
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
    const int device = ggml_cuda_get_device();
    if (!GGML_CUDA_CC_IS_CDNA2(ggml_cuda_info().devices[device].cc)) {
        return false;
    }
    const int ncu = ggml_cuda_info().devices[device].nsm;
    const iq4k::cfg c = gate ? iq4k::pick_cfg_glu(type, (int) N, (int) K, ncu) : iq4k_pick_cfg(type, (int) N, (int) K, ncu);
    if (!iq4k::shape_ok(c, type, (int) K, (int) N)) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    float * scratch = nullptr;
    unsigned int * cnt = nullptr;
    if (!iq4k_scratch(device, stream, &scratch, &cnt)) {
        return false;
    }
    const int il = (64/c.L)*c.R;
    // cache key: type + layout of the copy (interleave; round tiling depends on L and the number of waves)
    const int kind = c.rt ? 0x10000 + 0x100*(int) type + 0x10*(c.L/16) + (c.grid*c.NW)*0x40000 : 0x200 + 16*(int) type + il;
    ggml_cuda_repack_fn fn = gate ? iq4k_repack_glu : iq4k_repack_single;
    const void * up_soa = ggml_cuda_get_repacked(src0, device, stream, kind, ggml_nbytes(src0), fn);
    if (!up_soa) {
        return false;
    }
    float * dst_d = (float *) dst->data;
    bool ok;
    if (gate) {
        const void * gate_soa = ggml_cuda_get_repacked(gate, device, stream, kind, ggml_nbytes(gate), fn);
        if (!gate_soa) {
            return false;
        }
        ok = iq4k::launch_glu(type, up_soa, gate_soa, src1_q8_1, dst_d, (int) K, (int) N, stream, scratch, cnt, ncu);
    } else {
        const iq4k::soa s = iq4k::soa_make(type, up_soa, (size_t) N*(K/QK_K));
        ok = type == GGML_TYPE_IQ4_XS ?
            iq4k::dispatch<iq4k::iq4xs_t, false>(c, s, s, src1_q8_1, dst_d, (int) K, (int) N, stream, residual, scratch, cnt) :
            iq4k::dispatch<iq4k::q4k_t,   false>(c, s, s, src1_q8_1, dst_d, (int) K, (int) N, stream, residual, scratch, cnt);
    }
    if (!ok) {
        return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
#else
bool ggml_cuda_mmvq_iq4k_group_ok(const ggml_tensor *) {
    return false;
}
bool ggml_cuda_mmvq_iq4k_group(ggml_backend_cuda_context &, ggml_tensor * const *, int) {
    return false;
}
bool ggml_cuda_mmvq_iq4k_wants(const ggml_tensor *) {
    return false;
}
bool ggml_cuda_mmvq_iq4k(ggml_backend_cuda_context &, const ggml_tensor *, const block_q8_1 *, ggml_tensor *, const ggml_cuda_mm_fusion_args_host *) {
    return false;
}
#endif
