#pragma once
#include "common.cuh"

// Single-token GEMV with in-kernel q8_1 activation quantization (AMD wave64 only).
// epi: 0 = none, 1 = dst = silu(other) * (src0 x src1), 2 = dst = silu(src0 x src1) * other (see mmvqf::epi_kind).
bool ggml_cuda_mmvq_fused_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
bool ggml_cuda_mmvq_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion, const float * other = nullptr, int epi = 0);
