#pragma once
#include "common.cuh"

// Single-token GEMV with in-kernel q8_1 activation quantization (AMD wave64 only).
// epi: 0 = none, 1 = dst = silu(other) * (src0 x src1), 2 = dst = silu(src0 x src1) * other (see mmvqf::epi_kind).
bool ggml_cuda_mmvq_fused_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
bool ggml_cuda_mmvq_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion, const float * other = nullptr, int epi = 0);

// invalidate the per-stream cache of the quantized single-token activation (call at the start of every graph evaluation)
void ggml_cuda_mmvq_cache_reset(ggml_backend_cuda_context & ctx);
// producer API (mmvq.cu): get the cache buffer for the q8_1 blocks + int sums of the single-row activation t
bool ggml_cuda_mmvq_cache_produce(ggml_backend_cuda_context & ctx, const ggml_tensor * t, block_q8_1 ** q8, int ** sums);

// q8-producing variants of activation kernels (q8-producers.cu); return false if not applicable (nothing launched)
bool ggml_cuda_rms_norm_mul_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, ggml_tensor * mul_node);
bool ggml_cuda_swiglu_q8(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_unary_mul_q8(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node);

// Single-token GEMV for IQ2_S/IQ2_XS/IQ2_XXS/Q2_K/IQ1_M on SoA re-laid-out weights (activation already quantized to q8_1).
bool ggml_cuda_mmvq_lowbit(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const block_q8_1 * src1_q8_1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion);

// Single-token GEMV for IQ3_S/IQ3_XXS on the iq3gemv SoA re-layout (mmvq-iq3.cu; replaces the codex packed copies).
// Handles plain, residual (fusion->x_bias) and SwiGLU gate/up (fusion->gate) GEMVs; returns false to fall back.
bool ggml_cuda_mmvq_iq3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const block_q8_1 * src1_q8_1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion);
// true if a gate/up pair of different IQ3 types (IQ3_S + IQ3_XXS) can be fused by ggml_cuda_mmvq_iq3
bool ggml_cuda_mmvq_iq3_fuse_mixed_ok(const ggml_tensor * up_w, const ggml_tensor * gate_w);
