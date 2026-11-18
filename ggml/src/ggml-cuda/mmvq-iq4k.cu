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
#include "iq4k-gemv-nc.cuh"   // MMVQ-NC B
#include "mmvq.cuh"            // MMVQ-NC B
#include <string>                 // MMVQ-NC B
#include <vector>                 // MMVQ-NC B
#include <cstdlib>
#include <mutex>
#include <atomic>
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

// r4-mixgroup: the row-interleaved IL = 4 copy (= the single launcher's copy at K <= 8192 with L = 16) for the mixed
// IQ3 + IQ4_XS/Q4_K pair kernel (mixgemv.cuh); nullptr if the path is disabled or the copy cannot be made (capture)
const void * ggml_cuda_mmvq_iq4k_soa_il4(const ggml_tensor * w, int device, cudaStream_t stream) {
    if (!iq4k_enabled() || (w->type != GGML_TYPE_IQ4_XS && w->type != GGML_TYPE_Q4_K)) {
        return nullptr;
    }
    const int kind = 0x200 + 16*(int) w->type + 4;
    return ggml_cuda_get_repacked(w, device, stream, kind, ggml_nbytes(w), iq4k_repack_il4);
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

// MMVQ-NC B begin ------------------------------------------------------------------------------------------------------
// N-column (ne11 = 2..GGML_MMVQ_NC_MAX) IQ4_XS / Q4_K GEMVs (iq4k-gemv-nc.cuh): column j is bit-identical to the
// single-token path on column j alone. Rows >= 2048: the iq4k SoA kernel on the same copy (plain, residual, same-type
// SwiGLU gate/up); rows < 2048: the stock-structure small-N kernel (= gemv_sn / mul_mat_vec_q ncols_dst = 1).
// GGML_MMVQ_NC_IQ4K=0 (or GGML_MMVQ_NC=0 / GGML_MMVQ_IQ4K=0) disables. GGML_MMVQ_NC_VERIFY=n: the first n NC launches are
// recomputed column by column on the single-token path and compared bitwise (eager execution only).
static bool iq4k_nc_enabled() {
    static const bool enabled = [] {
        const char * v = getenv("GGML_MMVQ_NC_IQ4K");
        return iq4k_enabled() && ggml_cuda_mmvq_nc_enabled() && (v == nullptr || v[0] != '0');
    }();
    return enabled;
}
static std::atomic<int> & iq4k_nc_verify_left() {
    static std::atomic<int> left{iq4k_env_int("GGML_MMVQ_NC_VERIFY", 0)};
    return left;
}

// copy cache kind of the single-token launcher for cfg c (see ggml_cuda_mmvq_iq4k)
static int iq4k_nc_kind(ggml_type type, const iq4k::cfg & c) {
    const int il = (64/c.L)*c.R;
    return c.rt ? 0x10000 + 0x100*(int) type + 0x10*(c.L/16) + (c.grid*c.NW)*0x40000 : 0x200 + 16*(int) type + il;
}

// rows < 2048 run on the small-N kernel, which has the structure of stock mul_mat_vec_q<type, 1> WITHOUT its small_k
// variant; stock uses small_k for Q4_K at ncols_dst = 1 on GCN/CDNA when K/QK_K < nwarps*vdr*64/qi = 2*2*64/32 = 8
// (IQ4_XS never: it is in the iq_slow list there), so such Q4_K shapes are left to the old path (MMVQ-NC fix)
static bool iq4k_nc_sn_ok(const ggml_tensor * w) {
    return w->ne[1] >= 2048 || w->type != GGML_TYPE_Q4_K || w->ne[0]/QK_K >= 8;
}
static bool iq4k_nc_weight_ok(const ggml_tensor * w) {
    return (w->type == GGML_TYPE_IQ4_XS || w->type == GGML_TYPE_Q4_K) && w->ne[0] % QK_K == 0 && w->ne[0] <= 32768 &&
        iq4k_nc_sn_ok(w) &&
        ggml_is_matrix(w) && ggml_is_contiguous(w) && w->buffer &&
        GGML_CUDA_CC_IS_CDNA2(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
}

// MTP2-B begin: the int8 matrix-core form (iq4k::gemv_mf) for 3 (GGML_MMVQ_NC_IQ4K_MF_MIN) .. 16 columns and the small-N
// kernel with several rows per workgroup (iq4k::gemv_sn_ncr, rows < 2048) for 2 .. 16 columns.
// GGML_MMVQ_NC_IQ4K_MFMA=0 restores the previous kernels exactly (dp4a gemv_nc / gemv_sn_nc for 2..6 columns, nothing above).
// Every column stays bit-identical to the single-token path (see iq4k-gemv-nc.cuh; GGML_MMVQ_NC_VERIFY checks it).
static bool iq4k_nc_mfma() {
    static const bool on = [] {
        const char * v = getenv("GGML_MMVQ_NC_IQ4K_MFMA");
        return v == nullptr || v[0] != '0';
    }();
    return on;
}
static int iq4k_nc_mf_min() {
    static const int v = std::max(2, iq4k_env_int("GGML_MMVQ_NC_IQ4K_MF_MIN", 3));
    return v;
}
// MTP2-merge: optional per-shape column caps of the N-column path (above a cap the call is declined and the MUL_MAT goes
// to MMQ, which is faster there at 13..16 columns but not bit-identical to the single-token path): the small-N kernel
// (rows < 2048, K/V) up to GGML_MMVQ_NC_IQ4K_SN_MAX columns, the K-phased IQ4_XS ffn_down (K >= 16384) up to
// GGML_MMVQ_NC_IQ4K_DOWN_MAX. Both default to 16 (no cap: every column stays exact). Only with the MFMA form on.
static bool iq4k_nc_cap_ok(const ggml_tensor * w, const int ncols) {
    static const int sn_max   = iq4k_env_int("GGML_MMVQ_NC_IQ4K_SN_MAX", 16);
    static const int down_max = iq4k_env_int("GGML_MMVQ_NC_IQ4K_DOWN_MAX", 16);
    if (!iq4k_nc_mfma()) {
        return true;
    }
    if (w->ne[1] < 2048 && ncols > sn_max) {
        return false;
    }
    if (w->type == GGML_TYPE_IQ4_XS && w->ne[0] >= 16384 && ncols > down_max) {
        return false;
    }
    return true;
}
int ggml_cuda_mmvq_iq4k_nc_max();
// the largest column count family B takes (0: off)
int ggml_cuda_mmvq_iq4k_nc_max() {
    if (!iq4k_nc_enabled()) {
        return 0;
    }
    return std::min(ggml_cuda_mmvq_nc_max(), iq4k_nc_mfma() ? 16 : iq4k::NC_MAX_INST);
}
// kernel choice of an N-column launch: mf = gemv_mf with plan m (column groups / K phases, iq4k::mf_plan), else dp4a
// gemv_nc with plan p
struct iq4k_ncsel { iq4k::cfg c; iq4k::ncplan p; iq4k::mfplan m; bool mf; };
// MTP2-B end

// the single-token configuration of w (gate: the SwiGLU launcher) and the NC plan; false if no NC kernel applies
static bool iq4k_nc_plan(const ggml_tensor * w, bool glu, int ncols, iq4k_ncsel & sl) {   // MTP2-B: + gemv_mf choice
    const ggml_type type = w->type;
    const int K = (int) w->ne[0], N = (int) w->ne[1];
    const int ncu = iq4k_ncu();
    iq4k::cfg & c = sl.c;
    iq4k::ncplan & p = sl.p;
    c = glu ? iq4k::pick_cfg_glu(type, N, K, ncu) : iq4k_pick_cfg(type, N, K, ncu);
    if (!iq4k::nc_cfg_ok(c, type, K, N)) {
        return false;
    }
    // launch bounds of the NC kernel (iq4k::nc_lb): Q4_K SwiGLU with at most 12 waves per workgroup
    if (c.NW*64 > (glu && type == GGML_TYPE_Q4_K ? 768 : 1024)) {
        return false;
    }
    // MTP2-B: gemv_mf from 3 columns (2: the dp4a kernel is faster); the LM head (Q4_K, 248320 rows) keeps the dp4a kernel up to
    // 4 columns (measured faster there: many tiles per wave)
    const int mf_min = N >= 65536 ? std::max(iq4k_nc_mf_min(), 5) : iq4k_nc_mf_min();
    if (iq4k_nc_mfma() && ncols >= mf_min && iq4k::mf_plan(type, c, glu, K, N, ncols, ncu, sl.m)) {
        sl.mf = true;
        return true;
    }
    sl.mf = false;
    if (ncols > iq4k::NC_MAX_INST || !iq4k::nc_plan(type, c, K, N, ncols, p) || !iq4k::nc_mu_ok(p.mu)) {
        return false;
    }
    return !glu || p.mu == 0;
}

bool ggml_cuda_mmvq_iq4k_nc_route(int64_t ne11) {
    return ne11 >= 2 && ne11 <= ggml_cuda_mmvq_iq4k_nc_max();   // MTP2-B: the family maximum
}

bool ggml_cuda_mmvq_iq4k_nc_ok(const ggml_tensor * w, const ggml_tensor * gate_w, bool residual, int ncols, bool capturing) {
    if (!iq4k_nc_enabled() || ncols < 2 || ncols > ggml_cuda_mmvq_iq4k_nc_max() || !iq4k_nc_weight_ok(w) || !iq4k_nc_cap_ok(w, ncols)) {   // MTP2-B
        return false;
    }
    if (w->ne[1] < 2048) {   // small-N kernel: no epilogue (the single-token path of these shapes is stock mmvq)
        return !gate_w && !residual;
    }
    if (residual && gate_w) {
        return false;
    }
    if (gate_w && (gate_w->type != w->type || !ggml_are_same_shape(gate_w, w) || !ggml_is_contiguous(gate_w))) {
        return false;
    }
    iq4k_ncsel sl;
    if (!iq4k_nc_plan(w, gate_w != nullptr, ncols, sl)) {
        return false;
    }
    const iq4k::cfg & c = sl.c;
    // the copies cannot be made during graph capture: a fusion is only promised for tensors whose copy exists (the
    // unfused N-column call creates it in the eager warmup run). Verify mode outside capture: the launcher creates it.
    if (iq4k_nc_verify_left().load() > 0 && !capturing) {
        // (only weights the copy can be made for later: a WEIGHTS buffer, not a view; else the promise could not be kept)
        const auto weights = [](const ggml_tensor * t) {
            return t->buffer && !t->view_src && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        };
        return weights(w) && (!gate_w || weights(gate_w));
    }
    const int kind = iq4k_nc_kind(w->type, c);
    return ggml_cuda_has_repacked(w, kind) && (!gate_w || ggml_cuda_has_repacked(gate_w, kind));
}

bool ggml_cuda_mmvq_iq4k_nc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const block_q8_1 * y, int ncols, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    if (!iq4k_nc_enabled() || ncols < 2 || ncols > ggml_cuda_mmvq_iq4k_nc_max() || !iq4k_nc_weight_ok(src0) || !iq4k_nc_cap_ok(src0, ncols)) {   // MTP2-B
        return false;
    }
    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || dst->ne[0] != src0->ne[1] || dst->ne[1] != ncols ||
            dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }
    const ggml_type type = src0->type;
    const int K = (int) src0->ne[0], N = (int) src0->ne[1];
    const int ys = (int) (GGML_PAD(K, MATRIX_ROW_PADDING)/QK8_1);   // q8_1 blocks per column (cache and pool layouts)
    const ggml_tensor * gate = nullptr;
    const ggml_tensor * res  = nullptr;
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
            res = fusion->x_bias;
        }
    }
    const int device = ggml_cuda_get_device();
    cudaStream_t stream = ctx.stream();
    float * dst_d = (float *) dst->data;
    const int dstr = (int) (dst->nb[1]/sizeof(float));
    if (N < 2048) {
        if (gate || res) {
            return false;
        }
        const bool ok = iq4k_nc_mfma() ? iq4k::launch_sn_ncr(type, ncols, src0->data, y, ys, dst_d, dstr, K, N, stream) :   // MTP2-B: up to 16
                                         iq4k::launch_sn_nc(type, ncols, src0->data, y, ys, dst_d, dstr, K, N, stream);
        if (!ok) {
            return false;
        }
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    iq4k_ncsel sl;
    if (!iq4k_nc_plan(src0, gate != nullptr, ncols, sl)) {
        return false;
    }
    const iq4k::cfg & c = sl.c;
    const iq4k::ncplan & p = sl.p;
    const int kind = iq4k_nc_kind(type, c);
    ggml_cuda_repack_fn fn = gate ? iq4k_repack_glu : iq4k_repack_single;
    const void * up_soa = ggml_cuda_get_repacked(src0, device, stream, kind, ggml_nbytes(src0), fn);
    const void * gate_soa = gate ? ggml_cuda_get_repacked(gate, device, stream, kind, ggml_nbytes(gate), fn) : nullptr;
    if (!up_soa || (gate && !gate_soa)) {
        return false;
    }
    const size_t nb = (size_t) N*(K/QK_K);
    const iq4k::soa s  = iq4k::soa_make(type, up_soa, nb);
    const iq4k::soa sg = gate ? iq4k::soa_make(type, gate_soa, nb) : s;
    const float * res_d = res ? (const float *) res->data : nullptr;
    const int rstr = res ? (int) (res->nb[1]/sizeof(float)) : 0;
    bool ok = true;
    if (sl.mf) {   // MTP2-B: gemv_mf (one launch; column groups are workgroups of the same launch)
        ok = type == GGML_TYPE_IQ4_XS ?
            (gate ? iq4k::launch_mf<iq4k::iq4xs_t, true >(c, sl.m, ncols, s, sg, y, ys, dst_d, dstr, nullptr, 0, K, N, stream) :
                    iq4k::launch_mf<iq4k::iq4xs_t, false>(c, sl.m, ncols, s, sg, y, ys, dst_d, dstr, res_d, rstr, K, N, stream)) :
            (gate ? iq4k::launch_mf<iq4k::q4k_t, true >(c, sl.m, ncols, s, sg, y, ys, dst_d, dstr, nullptr, 0, K, N, stream) :
                    iq4k::launch_mf<iq4k::q4k_t, false>(c, sl.m, ncols, s, sg, y, ys, dst_d, dstr, res_d, rstr, K, N, stream));
    } else if (type == GGML_TYPE_IQ4_XS) {
        ok = gate ? iq4k::launch_nc<iq4k::iq4xs_t, true >(c, p, ncols, s, sg, y, ys, dst_d, dstr, nullptr, 0, K, N, stream) :
                    iq4k::launch_nc<iq4k::iq4xs_t, false>(c, p, ncols, s, sg, y, ys, dst_d, dstr, res_d, rstr, K, N, stream);
    } else {
        ok = gate ? iq4k::launch_nc<iq4k::q4k_t, true >(c, p, ncols, s, sg, y, ys, dst_d, dstr, nullptr, 0, K, N, stream) :
                    iq4k::launch_nc<iq4k::q4k_t, false>(c, p, ncols, s, sg, y, ys, dst_d, dstr, res_d, rstr, K, N, stream);
    }
    if (!ok) {
        return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// ggml_cuda_mmvq_iq4k_nc + GGML_MMVQ_NC_VERIFY: the first n launches are recomputed column by column with the
// single-token path (ggml_cuda_mul_mat_vec_q on one-column views; the residual is snapshotted first since the ADD may be
// in place) and compared bitwise. Eager execution only (no check during graph capture).
bool ggml_cuda_mmvq_iq4k_nc_checked(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const block_q8_1 * y, int ncols, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    std::atomic<int> & left = iq4k_nc_verify_left();
    cudaStream_t stream = ctx.stream();
    bool verify = left.load() > 0;
    if (verify) {
        hipStreamCaptureStatus status;
        CUDA_CHECK(hipStreamIsCapturing(stream, &status));
        verify = status == hipStreamCaptureStatusNone;
    }
    const ggml_tensor * xb = fusion ? fusion->x_bias : nullptr;
    ggml_cuda_pool_alloc<char> snap(ctx.pool());
    if (verify && xb) {
        snap.alloc(ggml_nbytes(xb));
        CUDA_CHECK(cudaMemcpyAsync(snap.get(), xb->data, ggml_nbytes(xb), cudaMemcpyDeviceToDevice, stream));
    }
    if (!ggml_cuda_mmvq_iq4k_nc(ctx, src0, y, ncols, dst, fusion)) {
        return false;
    }
    if (!verify) {
        return true;
    }
    --left;
    const int64_t nrows = dst->ne[0];
    ggml_cuda_pool_alloc<float> ref(ctx.pool(), nrows);
    std::vector<float> a(nrows), b(nrows);
    int64_t bad = 0, first = -1;
    for (int64_t j = 0; j < ncols; ++j) {
        ggml_tensor x1 = *src1;
        x1.ne[1] = 1; x1.nb[2] = x1.nb[1]; x1.nb[3] = x1.nb[1];
        x1.data = (char *) src1->data + j*src1->nb[1];
        x1.view_src = nullptr; x1.view_offs = 0;
        ggml_tensor d1 = *dst;
        d1.ne[1] = 1; d1.nb[2] = d1.nb[1]; d1.nb[3] = d1.nb[1];
        d1.data = ref.get();
        d1.view_src = nullptr; d1.view_offs = 0;
        ggml_tensor r1;
        ggml_cuda_mm_fusion_args_host f1{};
        if (fusion) {
            f1 = *fusion;
            if (xb) {
                r1 = *xb;
                r1.ne[1] = 1; r1.nb[2] = r1.nb[1]; r1.nb[3] = r1.nb[1];
                r1.data = snap.get() + j*xb->nb[1];
                r1.view_src = nullptr; r1.view_offs = 0;
                f1.x_bias = &r1;
            }
        }
        ggml_cuda_mul_mat_vec_q(ctx, src0, &x1, nullptr, &d1, fusion ? &f1 : nullptr);
        CUDA_CHECK(cudaMemcpyAsync(a.data(), ref.get(), nrows*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(b.data(), (const char *) dst->data + j*dst->nb[1], nrows*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int64_t r = 0; r < nrows; ++r) {
            if (memcmp(&a[r], &b[r], sizeof(float)) != 0) {
                if (first < 0) {
                    first = j*nrows + r;
                }
                ++bad;
            }
        }
    }
    const std::string where = bad ? " (first at col " + std::to_string(first/nrows) + " row " + std::to_string(first%nrows) + ")" : "";
    fprintf(stderr, "MMVQ-NC verify iq4k %s %s [%lld x %lld] n=%d %s: %lld mismatches%s\n", ggml_type_name(src0->type), dst->name,
        (long long) src0->ne[0], (long long) src0->ne[1], ncols,
        fusion ? (fusion->gate ? "glu" : xb ? "residual" : "fused") : "plain", (long long) bad, where.c_str());
    return true;
}
// MMVQ-NC B end --------------------------------------------------------------------------------------------------------
#else
// MMVQ-NC B
int ggml_cuda_mmvq_iq4k_nc_max();   // MTP2-B
int ggml_cuda_mmvq_iq4k_nc_max() {
    return 0;
}
bool ggml_cuda_mmvq_iq4k_nc_route(int64_t) {
    return false;
}
bool ggml_cuda_mmvq_iq4k_nc_ok(const ggml_tensor *, const ggml_tensor *, bool, int, bool) {
    return false;
}
bool ggml_cuda_mmvq_iq4k_nc(ggml_backend_cuda_context &, const ggml_tensor *, const block_q8_1 *, int, ggml_tensor *, const ggml_cuda_mm_fusion_args_host *) {
    return false;
}
bool ggml_cuda_mmvq_iq4k_nc_checked(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, const block_q8_1 *, int, ggml_tensor *, const ggml_cuda_mm_fusion_args_host *) {
    return false;
}
const void * ggml_cuda_mmvq_iq4k_soa_il4(const ggml_tensor *, int, cudaStream_t) {
    return nullptr;
}
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
