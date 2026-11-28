#pragma once
#include "common.cuh"

// Single-token GEMV with in-kernel q8_1 activation quantization (AMD wave64 only).
// epi: 0 = none, 1 = dst = silu(other) * (src0 x src1), 2 = dst = silu(src0 x src1) * other (see mmvqf::epi_kind).
bool ggml_cuda_mmvq_fused_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
bool ggml_cuda_mmvq_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion, const float * other = nullptr, int epi = 0);

// Q4_K single-token GEMV with a pre-quantized activation (q8_1 blocks + int sums from the activation cache)
bool ggml_cuda_mmvq_fused_preq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const block_q8_1 * q8, const int * sums,
        ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion);

// invalidate the per-stream cache of the quantized single-token activation (call at the start of every graph evaluation)
void ggml_cuda_mmvq_cache_reset(ggml_backend_cuda_context & ctx);
// drop the cached activation whose f32 source bytes a node writes (the graph loop calls these for every executed node,
// see mmvq.cu): _invalidate with MMVQ_Q8_INV_NODE before a node / the first node of a fused group (starts a new group),
// with MMVQ_Q8_INV_AHEAD before a member that a fusion runs ahead of the rest of its group; _invalidate_group after a
// fused group nodes[0..n) for all its members (keeps what the group produced last for a member)
enum { MMVQ_Q8_INV_NODE = 0, MMVQ_Q8_INV_AHEAD = 1 };
void ggml_cuda_mmvq_cache_invalidate(ggml_backend_cuda_context & ctx, const ggml_tensor * node, int mode);
void ggml_cuda_mmvq_cache_invalidate_group(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes, int n);
// producer API (mmvq.cu): get the cache buffer for the q8_1 blocks + int sums of the single-row activation t
bool ggml_cuda_mmvq_cache_produce(ggml_backend_cuda_context & ctx, const ggml_tensor * t, block_q8_1 ** q8, int ** sums);
// free the per-stream state of the activation cache (mmvq.cu) and of the cooperative quantizer (mmvq-fused.cu) of a
// stream that is about to be destroyed (call from the backend context destructor; handles can be recycled)
void ggml_cuda_mmvq_cache_release(int device, cudaStream_t stream);
void ggml_cuda_mmvq_fused_release(int device, cudaStream_t stream);

// q8-producing variants of activation kernels (q8-producers.cu); return false if not applicable (nothing launched)
bool ggml_cuda_rms_norm_mul_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, ggml_tensor * mul_node);
bool ggml_cuda_swiglu_q8(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_unary_mul_q8(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node);
// CONT(strided view) -> UNARY(SILU/SIGMOID) -> MUL in one kernel: the unary input is read through the view (no copy)
bool ggml_cuda_cont_unary_mul_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * cont_node, ggml_tensor * unary_node, ggml_tensor * mul_node);

// Single-token GEMV for IQ2_S/IQ2_XS/IQ2_XXS/Q2_K/IQ1_M on SoA re-laid-out weights (activation already quantized to q8_1).
// Gated output norm of the linear-attention layers in one kernel (+ q8_1 of the result for the following GEMV):
//   dst = silu(z) * (rms_norm(x) * w), rows of 128 (same arithmetic as rms_norm_f32<32, 1, true, true> + the
//   UNARY(SILU)+MUL op). x, z, dst: [128, nrows] contiguous f32, w: [128].
bool ggml_cuda_rms_norm_gated_q8_supported(const ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * z, const ggml_tensor * dst);
void ggml_cuda_rms_norm_gated_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w, float eps,
        const ggml_tensor * z, ggml_tensor * dst);
// Large-batch variant of the same op for an f16 GEMM consumer (HIP wave64 only): writes only the f16 values of dst
// (bit-identical to rms_norm_f32<32, 1, true, true> + UNARY(SILU)+MUL + the f32 -> f16 conversion) into dst16;
// dst->data is neither read nor written.
bool ggml_cuda_rms_norm_gated_f16_supported(const ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * z, const ggml_tensor * dst);
void ggml_cuda_rms_norm_gated_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w, float eps,
        const ggml_tensor * z, const ggml_tensor * dst, half * dst16);

// Single-token BF16 thin GEMV with an elementwise epilogue (fused graph chains, see ggml-cuda.cu):
//   epi 0: dst = dot (+ bias)
//   epi 1: dst = sigmoid(dot)                        (MUL_MAT -> RESHAPE -> SIGMOID)
//   epi 2: dst = softplus(dot + bias) * mul          (MUL_MAT -> RESHAPE -> ADD -> SOFTPLUS -> MUL)
// Up to two jobs with the same activation run in one launch. Same arithmetic as the unfused ops (bit-identical).
struct ggml_cuda_bf16_epi_job {
    const ggml_tensor * src0;
    ggml_tensor       * dst;
    const float       * bias;
    const float       * mul;
    int                 epi;
};
bool ggml_cuda_mmvf_bf16_thin_epi(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const ggml_cuda_bf16_epi_job * jobs, int njobs);

bool ggml_cuda_mmvq_lowbit(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const block_q8_1 * src1_q8_1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion);

// Single-token GEMV for IQ3_S/IQ3_XXS on the iq3gemv SoA re-layout (mmvq-iq3.cu; replaces the codex packed copies).
// Handles plain, residual (fusion->x_bias) and SwiGLU gate/up (fusion->gate) GEMVs; returns false to fall back.
bool ggml_cuda_mmvq_iq3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const block_q8_1 * src1_q8_1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion);
// true if a gate/up pair of different IQ3 types (IQ3_S + IQ3_XXS) can be fused by ggml_cuda_mmvq_iq3
bool ggml_cuda_mmvq_iq3_fuse_mixed_ok(const ggml_tensor * up_w, const ggml_tensor * gate_w);
// n (2..3) single-token GEMVs with IQ3_S/IQ3_XXS weights w[i] (same K) on the same activation in one launch (iq3gemv
// multi-job kernel), outputs dst[i]; _ok: shape/type check only; the launch returns false to fall back.
bool ggml_cuda_mmvq_iq3_multi_ok(const ggml_tensor * const * w, int n);
bool ggml_cuda_mmvq_iq3_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const * w, ggml_tensor * const * dst, int n,
        const block_q8_1 * src1_q8_1);
// Single-token GEMV for IQ4_XS/Q4_K on row-interleaved SoA re-laid-out weights (mmvq-iq4k.cu; activation already q8_1).
// Supports the residual (x_bias) epilogue and the fused same-type gate/up SwiGLU. GGML_MMVQ_IQ4K=0 disables.
bool ggml_cuda_mmvq_iq4k(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const block_q8_1 * src1_q8_1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion);
bool ggml_cuda_mmvq_iq4k_wants(const ggml_tensor * src0);
// Single-token GEMV for Q6_K on the q6k SoA re-layout (mmvq-q6k.cu, q6k-gemv.cuh; activation already q8_1): plain,
// residual (x_bias) and same-type SwiGLU gate/up. Not bit-identical to the stock Q6_K kernels (only the MTP draft head
// uses Q6_K in the target model). GGML_MMVQ_Q6K=0 disables. _wants: will it take this weight (keeps mmvqf away).
bool ggml_cuda_mmvq_q6k(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const block_q8_1 * src1_q8_1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion);
bool ggml_cuda_mmvq_q6k_wants(const ggml_tensor * src0);
// N-column (2..4) unfused Q6_K GEMVs on the same copy (y column c at y + c*GGML_PAD(K, 512)/32); false: nothing launched.
// Opt-in: GGML_MMVQ_Q6K_NC=1 (_enabled); columns are not bit-identical to the single-token kernel (draft head only).
bool ggml_cuda_mmvq_q6k_nc_enabled();
bool ggml_cuda_mmvq_q6k_nc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const block_q8_1 * y, int ncols, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion);
// iq4k-r3: one launch for 2..4 single-token MUL_MATs that read the same activation (IQ4_XS/Q4_K weights, may be mixed).
// Default: only groups of small GEMVs (attn_k + attn_v, N <= 2048; bit-identical); groups with a large member
// (q/k/v, qkv/z, mixed-type gate/up) only with GGML_MMVQ_IQ4K_GROUP_BIG=1. group_ok: can this node be a member;
// group: run them (false: nothing launched, caller runs the nodes one by one). GGML_MMVQ_IQ4K_GROUP=0 disables.
bool ggml_cuda_mmvq_iq4k_group_ok(const ggml_tensor * mm);
bool ggml_cuda_mmvq_iq4k_group(ggml_backend_cuda_context & ctx, ggml_tensor * const * mm, int n);
// q8_1 blocks of a single-row f32 activation from the per-stream cache (quantized on a miss); false if not usable
bool ggml_cuda_mmvq_q8_cached(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const block_q8_1 ** q8);
// r4-mixgroup: one launch for a pair of single-token GEMVs on the same activation that mixes an IQ3_S/IQ3_XXS job with
// an IQ4_XS/Q4_K job (mixgemv.cuh; linear-attention qkv + z, mixed-type FFN gate + up). member: can this MUL_MAT be
// part of such a pair; mix: run the pair mm[0..1] (false: nothing launched, the caller runs the nodes one by one).
// glu (optional): the SWIGLU node of a gate/up pair, computed in the same launch (+ its q8_1 for ffn_down).
// GGML_MMVQ_MIX=0 disables, =2 runs the IQ4 job first, GGML_MMVQ_MIX_GLU=0 keeps the SwiGLU separate.
// soa_il4: the iq4k IL = 4 copy (mmvq-iq4k.cu).
// MMVQ-NC S: N-column (2 <= ncols <= nc_max) GEMVs for multi-token verification. nc_max: GGML_MMVQ_NC_MAX (default 16,
// 2..16), 0 if GGML_MMVQ_NC=0. q8_cached_nc: q8_1 of a [K, ncols] f32 activation (K % 512 == 0) from the activation cache,
// column c at q8 + c*K/32 (byte-identical to the single-row quantization of every column). fusion_ok: will
// ggml_cuda_mul_mat_vec_q fuse this MUL_MAT (N columns) with a SwiGLU gate (gate_w) or a residual add; it is exact (a
// promised fusion that no NC slot takes aborts) and may create a low-bit SoA copy outside stream capture (ctx).
bool ggml_cuda_mmvq_nc_enabled();
int  ggml_cuda_mmvq_nc_max();
bool ggml_cuda_mmvq_q8_cached_nc(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const block_q8_1 ** q8, const int ** sums);
bool ggml_cuda_mmvq_nc_fusion_ok(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * gate_w, bool residual);
// MMVQ-NC B: IQ4_XS / Q4_K N-column GEMVs (mmvq-iq4k.cu, iq4k-gemv-nc.cuh). _ok: exact predicate of _nc (types, shapes,
// LDS, existing SoA copies); _nc: y = q8_1 columns at y + j*(GGML_PAD(K, 512)/32); false: nothing launched.
// capturing: the stream of the caller is being captured (no SoA copy can be created, also not in verify mode)
bool ggml_cuda_mmvq_iq4k_nc_ok(const ggml_tensor * w, const ggml_tensor * gate_w, bool residual, int ncols, bool capturing);
bool ggml_cuda_mmvq_iq4k_nc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const block_q8_1 * y, int ncols, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion);
// _nc plus the GGML_MMVQ_NC_VERIFY=n debug check (first n launches compared bitwise with the single-token path per column)
bool ggml_cuda_mmvq_iq4k_nc_checked(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const block_q8_1 * y, int ncols, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion);
// Q4_K with ne11 = 4..NC max goes to MMVQ (N-column iq4k kernels) instead of MMQ on CDNA2
bool ggml_cuda_mmvq_iq4k_nc_route(int64_t ne11);
// MMVQ-NC C: low-bit (IQ2_S/IQ2_XS/IQ2_XXS/Q2_K/IQ1_M) N-column GEMVs (mmvq-lowbit.cu, lowbit-gemv.cuh): q8_1 column j at
// y + j*GGML_PAD(K, 512)/32, dst column j at dst->data + j*dst->nb[1]; plain and residual epilogues. _ok is exact (incl.
// the existing SoA copy); _prepare creates the copy outside stream capture. false: nothing launched.
bool ggml_cuda_mmvq_lowbit_nc_ok(const ggml_tensor * w, const ggml_tensor * gate_w, bool residual, int ncols);
void ggml_cuda_mmvq_lowbit_nc_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * w, int ncols);
bool ggml_cuda_mmvq_lowbit_nc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const block_q8_1 * y, int ncols, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion);
// MMVQ-NC A: N-column IQ3_S/IQ3_XXS GEMVs (mmvq-iq3.cu, iq3gemv-nc.cuh), per column bit-identical to the single-token path.
// y: q8_1 activation, column c at y + c*GGML_PAD(K, 512)/32. _ok is exact (true only if the launch will happen).
bool ggml_cuda_mmvq_iq3_nc_ok(const ggml_tensor * w, const ggml_tensor * gate_w, bool residual, int ncols);
bool ggml_cuda_mmvq_iq3_nc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const block_q8_1 * y, int ncols, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion);
bool ggml_cuda_mmvq_iq3_multi_nc_ok(const ggml_tensor * const * w, int n, int ncols);
bool ggml_cuda_mmvq_iq3_multi_nc(ggml_backend_cuda_context & ctx, const ggml_tensor * const * w, ggml_tensor * const * dst, int n,
        const block_q8_1 * y, int ncols);
bool ggml_cuda_mmvq_mix_member(const ggml_tensor * mm);
bool ggml_cuda_mmvq_mix(ggml_backend_cuda_context & ctx, ggml_tensor * const * mm, int n, ggml_tensor * glu = nullptr);
const void * ggml_cuda_mmvq_iq4k_soa_il4(const ggml_tensor * w, int device, cudaStream_t stream);

bool ggml_cuda_mmvq_q2_0(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const block_q8_1 * x,
        ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion);

#if defined(GGML_USE_HIP)
bool ggml_cuda_mmvq_q2_0_fwht_group(ggml_backend_cuda_context & ctx, const ggml_tensor * input,
        const ggml_tensor * signs, ggml_tensor * const * outputs, int count);
bool ggml_cuda_mmvq_q2_0_mfma(ggml_backend_cuda_context & ctx, const ggml_tensor * w,
        const ggml_tensor * src, ggml_tensor * dst);
#endif
