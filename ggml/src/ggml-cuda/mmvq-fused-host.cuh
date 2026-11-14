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
// producer API (mmvq.cu): get the cache buffer for the q8_1 blocks + int sums of the single-row activation t
bool ggml_cuda_mmvq_cache_produce(ggml_backend_cuda_context & ctx, const ggml_tensor * t, block_q8_1 ** q8, int ** sums);

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
// GGML_MMVQ_MIX=0 disables, =2 runs the IQ4 job first. soa_il4: the iq4k IL = 4 copy (mmvq-iq4k.cu).
bool ggml_cuda_mmvq_mix_member(const ggml_tensor * mm);
bool ggml_cuda_mmvq_mix(ggml_backend_cuda_context & ctx, ggml_tensor * const * mm, int n);
const void * ggml_cuda_mmvq_iq4k_soa_il4(const ggml_tensor * w, int device, cudaStream_t stream);
