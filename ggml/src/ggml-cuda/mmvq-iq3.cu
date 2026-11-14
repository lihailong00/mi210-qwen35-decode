// Single-token GEMV for IQ3_S / IQ3_XXS on a lossless SoA re-layout (iq3gemv.cuh, gemv-iq3 workstream).
// The re-laid-out copy (112 B / 98 B per 256 weights, vs 134 B for the retained codex 4-bit packed copy) is created
// once per weight tensor with ggml_cuda_get_repacked (cached with the weight buffer, original kept for MMQ/get_rows).
// Codebooks are decoded from sign-expanded LDS tables; integer per-32 sums are identical to vec_dot_iq3_*_q8_1.
// Epilogues: residual add (fusion->x_bias), fused SwiGLU gate/up (fusion->gate, same or mixed IQ3_S/IQ3_XXS types).
// Multi-job launches (ggml_cuda_mmvq_iq3_multi): 2-3 GEMVs on the same activation (qkv + z, k + v) in one kernel.
// GGML_MMVQ_IQ3=0 disables this path (then the retained codex packed path / stock mmvq is used as before);
// GGML_MMVQ_IQ3_MULTI=0 disables only the multi-job grouping, GGML_MMVQ_IQ3_PF=0 selects the previous prologue,
// GGML_MMVQ_IQ3_SM=0 the previous (unit-major) weight layout.
// Mixed pairs (r4-mixgroup, ggml_cuda_mmvq_mix): an IQ3 GEMV and an IQ4_XS/Q4_K GEMV on the same activation in one
// launch (mixgemv.cuh); GGML_MMVQ_MIX=0 disables.
#include "common.cuh"
#include "iq3-repack.cuh"
#include "mmvq-fused-host.cuh"

#if defined(GGML_USE_HIP)
#include "iq3gemv.cuh"
#include "mixgemv.cuh"
#include <map>
#include <mutex>
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
// prologue of the single / gate+up kernel (GGML_MMVQ_IQ3_PF, default 7): 0 = previous one (every thread loads an activation
// block, threads past the end all load the last one), 5 = only the owners of the blocks load them, 7 = 5 with the grid
// words loaded first and all kernel arguments up front (r4-ffndown; also selects that load order in the multi-job kernel)
static int iq3g_pf_mode() {
    static const int mode = [] {
        const char * v = getenv("GGML_MMVQ_IQ3_PF");
        return v == nullptr ? 7 : atoi(v);
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
    const int pf = iq3g_pf_mode();
    if (sm) {
        return (pf == 7 && iq3g_launch<TA, TB, 2, 7, IQ3G_SM>(wa, wb, y, dst, K, N, residual, ncu, stream)) ||
               (pf == 5 && iq3g_launch<TA, TB, 2, 5, IQ3G_SM>(wa, wb, y, dst, K, N, residual, ncu, stream)) ||
               iq3g_launch<TA, TB, 2, 0, IQ3G_SM>(wa, wb, y, dst, K, N, residual, ncu, stream);
    }
    return (pf == 7 && iq3g_launch<TA, TB, 2, 7, 0>(wa, wb, y, dst, K, N, residual, ncu, stream)) ||
           (pf == 5 && iq3g_launch<TA, TB, 2, 5, 0>(wa, wb, y, dst, K, N, residual, ncu, stream)) ||
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

template <int SM, bool ORD>
static void iq3g_launch_mj(const bool hs, const bool hx, const block_q8_1 * y, const int K, const iq3g::mj_args & a, const int ncu,
        hipStream_t stream) {
    using namespace iq3g;
    constexpr int NW = IQ3G_MJ_NW, L = IQ3G_MJ_L;
    static_assert(64/L == IQ3G_RG, "kernel config must match the repack row interleave");
    if (hs && hx) {
        gemv_mj<iq3s_t, iq3xxs_t, 2, L, NW, 1, 0, SM, ORD><<<ncu, dim3(64, NW), lds_bytes_mj<iq3s_t, iq3xxs_t>(K, 1, NW, L), stream>>>(y, K, a);
    } else if (hs) {
        gemv_mj<iq3s_t, none_t, 2, L, NW, 1, 0, SM, ORD><<<ncu, dim3(64, NW), lds_bytes_mj<iq3s_t, none_t>(K, 1, NW, L), stream>>>(y, K, a);
    } else {
        gemv_mj<iq3xxs_t, none_t, 2, L, NW, 1, 0, SM, ORD><<<ncu, dim3(64, NW), lds_bytes_mj<iq3xxs_t, none_t>(K, 1, NW, L), stream>>>(y, K, a);
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
    const bool ord = iq3g_pf_mode() == 7;
    if (iq3g_sm(K)) {
        if (ord) iq3g_launch_mj<IQ3G_SM, true>(hs, hx, src1_q8_1, K, a, ncu, stream);
        else     iq3g_launch_mj<IQ3G_SM, false>(hs, hx, src1_q8_1, K, a, ncu, stream);
    } else {
        if (ord) iq3g_launch_mj<0, true>(hs, hx, src1_q8_1, K, a, ncu, stream);
        else     iq3g_launch_mj<0, false>(hs, hx, src1_q8_1, K, a, ncu, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// ------------------------------------------------------------------------------------------------ mixed pairs (r4-mixgroup)
// GGML_MMVQ_MIX: 0 = off (the members run one by one as before), 1 (default) = IQ3 job first, 2 = IQ4 job first
static int mixg_mode() {
    static const int mode = [] {
        const char * v = getenv("GGML_MMVQ_MIX");
        return v == nullptr ? 1 : atoi(v);
    }();
    return mode;
}
static bool mixg_is_iq3(const ggml_type t) { return t == GGML_TYPE_IQ3_S  || t == GGML_TYPE_IQ3_XXS; }
static bool mixg_is_iq4(const ggml_type t) { return t == GGML_TYPE_IQ4_XS || t == GGML_TYPE_Q4_K; }

bool ggml_cuda_mmvq_mix_member(const ggml_tensor * mm) {
    if (mixg_mode() == 0 || !iq3g_enabled() || mm->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * w = mm->src[0];
    const ggml_tensor * x = mm->src[1];
    const bool i3 = mixg_is_iq3(w->type), i4 = mixg_is_iq4(w->type);
    if ((!i3 && !i4) || x->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) ||
            x->ne[1] != 1 || x->ne[2] != 1 || x->ne[3] != 1 || x->ne[0] != w->ne[0] || !ggml_is_contiguous(mm) ||
            ggml_nelements(mm) != w->ne[1]) {
        return false;
    }
    const int64_t K = w->ne[0], N = w->ne[1];
    // K: step-major IQ3 layout (K/128 % 8), exact IQ4 steps (K/32 resp. K/64 % 16), one q8_1 block per prologue thread
    if (K % 1024 != 0 || K > 8192 || N % mixg::RG != 0 || N > (1 << 24)) {
        return false;
    }
    if (i3) {
        return iq3g_weight_ok(w);
    }
    // the 1024-row GEMVs stay with their own paths (stock mmvq / iq4k small-N groups)
    return N >= 2048 && !w->view_src && ggml_is_matrix(w) && ggml_is_contiguous(w) && w->buffer &&
        ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        GGML_CUDA_CC_IS_CDNA2(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
}

// split mode (GGML_MMVQ_MIX=3): IQ3 waves per workgroup ~ the IQ3 job's share of the estimated time (bytes of the
// copies / typical efficiency: IQ3 ~0.8, IQ4 ~0.92 of the streaming rate); GGML_MMVQ_MIX_NA=n forces n (1..15)
static int mixg_na(const ggml_tensor * w3, const ggml_tensor * w4) {
    static const int force = [] { const char * v = getenv("GGML_MMVQ_MIX_NA"); return v ? atoi(v) : 0; }();
    if (force > 0 && force < mixg::NW) {
        return force;
    }
    if (mixg_mode() == 4) {
        return -1;   // alternating IQ3 / IQ4 units per wave
    }
    if (mixg_mode() != 3) {
        return 0;
    }
    const double b3 = (double) w3->ne[1]*(w3->type == GGML_TYPE_IQ3_S ? 112.0 : 98.0)/0.80;
    const double b4 = (double) w4->ne[1]*(w4->type == GGML_TYPE_IQ4_XS ? 136.0 : 144.0)/0.92;
    const int na = (int) (mixg::NW*b3/(b3 + b4) + 0.5);
    return std::min(mixg::NW - 1, std::max(1, na));
}
template <class T3, class T4>
static size_t mixg_lds(const int K, const int N4, const int nwv4) {
    return mixg::lds_bytes<T3, T4>(K, N4, nwv4);
}
template <class T3, class T4>
static void mixg_launch(const bool first3, const int na, const uint8_t * w3, float * d3, const int N3, const iq4k::soa & s4, float * d4,
        const int N4, const block_q8_1 * y, const int K, const int ncu, hipStream_t stream, const mixg::glu_args & ga) {
    const int nwv = ncu*mixg::NW;
    const size_t lds = mixg::lds_bytes<T3, T4>(K, N4, na > 0 ? (mixg::NW - na)*ncu : nwv);
    const int y4off = (int) mixg::y4_offset<T3, T4>(K);
    const int rot = (first3 ? N3/mixg::RG : N4/mixg::RG) % nwv;
    // r4-ffndown: the prologue load order of the iq3g kernels (grid words first) also for the pair kernel
    // (old order: GGML_MMVQ_IQ3_PF=5 for all iq3g kernels, GGML_MMVQ_MIX_ORD=0 for the pair kernel only)
    static const bool ord_env = iq3g_env_on("GGML_MMVQ_MIX_ORD");
    const bool ord = ord_env && iq3g_pf_mode() >= 7;
    if (first3) {
        if (ord) mixg::gemv_pair<T3, T4, IQ3G_SM, true,  true ><<<ncu, dim3(64, mixg::NW), lds, stream>>>(w3, d3, N3, s4, d4, N4, y, K, y4off, rot, na, ga);
        else     mixg::gemv_pair<T3, T4, IQ3G_SM, true,  false><<<ncu, dim3(64, mixg::NW), lds, stream>>>(w3, d3, N3, s4, d4, N4, y, K, y4off, rot, na, ga);
    } else {
        if (ord) mixg::gemv_pair<T3, T4, IQ3G_SM, false, true ><<<ncu, dim3(64, mixg::NW), lds, stream>>>(w3, d3, N3, s4, d4, N4, y, K, y4off, rot, na, ga);
        else     mixg::gemv_pair<T3, T4, IQ3G_SM, false, false><<<ncu, dim3(64, mixg::NW), lds, stream>>>(w3, d3, N3, s4, d4, N4, y, K, y4off, rot, na, ga);
    }
}

// zero-initialized counters of the SwiGLU epilogue (per device, allocated outside graph capture; the kernel leaves them zero)
static constexpr int MIXG_CNT = 1 << 14;
static unsigned int * mixg_counters(const int device, hipStream_t stream) {
    static std::mutex mtx;
    static std::map<int, unsigned int *> bufs;
    std::lock_guard<std::mutex> lock(mtx);
    auto it = bufs.find(device);
    if (it == bufs.end()) {
        hipStreamCaptureStatus status;
        CUDA_CHECK(hipStreamIsCapturing(stream, &status));
        if (status != hipStreamCaptureStatusNone) {
            return nullptr;
        }
        unsigned int * p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, MIXG_CNT*sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(p, 0, MIXG_CNT*sizeof(unsigned int)));
        CUDA_CHECK(cudaDeviceSynchronize());
        it = bufs.emplace(device, p).first;
    }
    return it->second;
}
// GGML_MMVQ_MIX_GLU=1: compute the SwiGLU of a gate/up pair (+ its q8_1) in the pair kernel instead of its own node.
// Off by default: the epilogue's serial chain at the end of the kernel (release fence, atomic, acquire, loads, q8 stores,
// ~5 us) costs more than the separate swiglu_q8 node (~3 us) in graph replay.
static bool mixg_glu_enabled() {
    static const bool enabled = [] { const char * v = getenv("GGML_MMVQ_MIX_GLU"); return v != nullptr && v[0] == '1'; }();
    return enabled;
}

bool ggml_cuda_mmvq_mix(ggml_backend_cuda_context & ctx, ggml_tensor * const * mm, const int n, ggml_tensor * glu) {
    if (n != 2 || !ggml_cuda_mmvq_mix_member(mm[0]) || !ggml_cuda_mmvq_mix_member(mm[1]) || mm[0]->src[1] != mm[1]->src[1]) {
        return false;
    }
    const bool a3 = mixg_is_iq3(mm[0]->src[0]->type), b3 = mixg_is_iq3(mm[1]->src[0]->type);
    if (a3 == b3) {
        return false;   // exactly one IQ3 job and one IQ4_XS/Q4_K job
    }
    ggml_tensor * m3 = a3 ? mm[0] : mm[1];
    ggml_tensor * m4 = a3 ? mm[1] : mm[0];
    const ggml_tensor * w3 = m3->src[0];
    const ggml_tensor * w4 = m4->src[0];
    const int K = (int) w3->ne[0];
    if (w4->ne[0] != K || !iq3g_sm(K)) {
        return false;
    }
    const int N3 = (int) w3->ne[1], N4 = (int) w4->ne[1];
    // experiment knob GGML_MMVQ_MIX_CLASS: 1 = only pairs of different N (qkv + z), 2 = only equal N (gate + up), 3 = both
    static const int cls = [] { const char * v = getenv("GGML_MMVQ_MIX_CLASS"); return v ? atoi(v) : 3; }();
    if (!(cls & (N3 == N4 ? 2 : 1))) {
        return false;
    }
    const int device = ggml_cuda_get_device();
    hipStream_t stream = ctx.stream();
    const int ncu = ggml_cuda_info().devices[device].nsm;
    const int nwv = ncu*mixg::NW;
    const bool s3 = w3->type == GGML_TYPE_IQ3_S, x4 = w4->type == GGML_TYPE_IQ4_XS;
    const int na = mixg_na(w3, w4);
    const int nwv4 = na > 0 ? (mixg::NW - na)*ncu : nwv;
    using namespace iq3g;
    const size_t lds = s3 ? (x4 ? mixg_lds<iq3s_t,   iq4k::iq4xs_t>(K, N4, nwv4) : mixg_lds<iq3s_t,   iq4k::q4k_t>(K, N4, nwv4))
                          : (x4 ? mixg_lds<iq3xxs_t, iq4k::iq4xs_t>(K, N4, nwv4) : mixg_lds<iq3xxs_t, iq4k::q4k_t>(K, N4, nwv4));
    if (lds > 65536) {
        return false;
    }
    // SwiGLU epilogue (glu = the GLU node of a gate/up pair): dst = silu(src[0]) * src[1] (not swapped), + its q8_1 blocks
    const ggml_tensor * gate_t = nullptr;
    const ggml_tensor * up_t   = nullptr;
    unsigned int * cnt = nullptr;
    if (glu) {
        if (!mixg_glu_enabled() || na > 0 || N3 != N4 || N3 % 32 != 0 || N3/32 + 2 > MIXG_CNT || glu->op != GGML_OP_GLU ||
                ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ((const int32_t *) glu->op_params)[1] != 0 || !glu->src[1] ||
                glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu) || ggml_nelements(glu) != N3 ||
                !ggml_is_contiguous(glu->src[0]) || !ggml_is_contiguous(glu->src[1])) {
            return false;
        }
        gate_t = glu->src[0];
        up_t   = glu->src[1];
        if (!((gate_t == m3 && up_t == m4) || (gate_t == m4 && up_t == m3))) {
            return false;
        }
        cnt = mixg_counters(device, stream);   // nullptr during graph capture if not allocated yet
        if (!cnt) {
            return false;
        }
    }
    const uint8_t * soa3 = iq3g_get_soa(w3, device, stream);                 // nullptr during graph capture if not created yet
    const void    * soa4 = ggml_cuda_mmvq_iq4k_soa_il4(w4, device, stream);  // = the single launcher's copy at K <= 8192
    if (!soa3 || !soa4) {
        return false;
    }
    const block_q8_1 * q8 = nullptr;
    if (!ggml_cuda_mmvq_q8_cached(ctx, m3->src[1], &q8)) {
        return false;
    }
    mixg::glu_args ga{};
    if (glu) {
        // the GLU output's q8_1 goes to the activation cache (the same buffer as this kernel's input q8: the kernel writes it
        // only after every workgroup has staged the input); after this call the launch must happen
        block_q8_1 * gy = nullptr;
        int * gsum = nullptr;
        if (!ggml_cuda_mmvq_cache_produce(ctx, glu, &gy, &gsum)) {
            return false;
        }
        ga.g    = (const float *) gate_t->data;
        ga.u    = (const float *) up_t->data;
        ga.dst  = (float *) glu->data;
        ga.y    = gy;
        ga.ysum = gsum;
        ga.cnt  = cnt;
        ga.nb   = N3/32;
    }
    const iq4k::soa s4 = iq4k::soa_make(w4->type, soa4, (size_t) N4*(K/QK_K));
    const bool first3 = mixg_mode() != 2;
    float * d3 = (float *) m3->data;
    float * d4 = (float *) m4->data;
    static const int n_empty_pre = [] { const char * v = getenv("GGML_MMVQ_MIX_EMPTY_PRE"); return v ? atoi(v) : 0; }();
    for (int e = 0; e < n_empty_pre; ++e) {
        mixg::empty_kernel<<<1, 64, 0, stream>>>();   // calibration probe: n empty launches before the pair (after a small kernel)
    }
    if (s3 && x4)       mixg_launch<iq3s_t,   iq4k::iq4xs_t>(first3, na, soa3, d3, N3, s4, d4, N4, q8, K, ncu, stream, ga);
    else if (s3)        mixg_launch<iq3s_t,   iq4k::q4k_t  >(first3, na, soa3, d3, N3, s4, d4, N4, q8, K, ncu, stream, ga);
    else if (x4)        mixg_launch<iq3xxs_t, iq4k::iq4xs_t>(first3, na, soa3, d3, N3, s4, d4, N4, q8, K, ncu, stream, ga);
    else                mixg_launch<iq3xxs_t, iq4k::q4k_t  >(first3, na, soa3, d3, N3, s4, d4, N4, q8, K, ncu, stream, ga);
    static const int n_empty = [] { const char * v = getenv("GGML_MMVQ_MIX_EMPTY"); return v ? atoi(v) : 0; }();
    // probe shape (GGML_MMVQ_MIX_EMPTY_CFG): 0 = 1 x 64 threads, no LDS; 1 = 1 x 1024 threads with the pair kernel's LDS;
    // 2 = ncu x 1024 threads with the pair kernel's LDS
    static const int cfg_empty = [] { const char * v = getenv("GGML_MMVQ_MIX_EMPTY_CFG"); return v ? atoi(v) : 0; }();
    for (int e = 0; e < n_empty; ++e) {
        if (cfg_empty == 0) {
            mixg::empty_kernel<<<1, 64, 0, stream>>>();
        } else {
            mixg::empty_kernel<<<cfg_empty == 1 ? 1 : ncu, dim3(64, mixg::NW), lds, stream>>>();
        }
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
bool ggml_cuda_mmvq_mix_member(const ggml_tensor *) {
    return false;
}
bool ggml_cuda_mmvq_mix(ggml_backend_cuda_context &, ggml_tensor * const *, int, ggml_tensor *) {
    return false;
}
#endif
