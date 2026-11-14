// Single-token GEMV for IQ4_XS and Q4_K using the SoA re-layout kernels of the gemv-iq4k workstream (iq4k-gemv.cuh):
// lossless same-size row-interleaved SoA copy of the weights (created once per tensor, cached with the weight buffer),
// q8_1 activation from the shared per-stream cache, residual epilogue (x_bias) and fused same-type gate/up SwiGLU.
// GGML_MMVQ_IQ4K=0 disables (falls back to the stock paths).
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
static void iq4k_repack_single(const void * src, void * dst, const ggml_tensor * t, hipStream_t stream) {
    iq4k::repack(t->type, src, dst, t->ne[1], (int) t->ne[0], stream, iq4k_ncu());
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
        iq4k::shape_ok(iq4k::pick_cfg(src0->type, (int) N, (int) K, ggml_cuda_info().devices[ggml_cuda_get_device()].nsm), src0->type, (int) K, (int) N);
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
    const iq4k::cfg c = gate ? iq4k::pick_cfg_glu(type, (int) N, (int) K, ncu) : iq4k::pick_cfg(type, (int) N, (int) K, ncu);
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
        ok = iq4k::launch(type, up_soa, src1_q8_1, dst_d, (int) K, (int) N, stream, scratch, cnt, residual, ncu);
    }
    if (!ok) {
        return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
#else
bool ggml_cuda_mmvq_iq4k_wants(const ggml_tensor *) {
    return false;
}
bool ggml_cuda_mmvq_iq4k(ggml_backend_cuda_context &, const ggml_tensor *, const block_q8_1 *, ggml_tensor *, const ggml_cuda_mm_fusion_args_host *) {
    return false;
}
#endif
