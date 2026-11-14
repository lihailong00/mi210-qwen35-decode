#pragma once

#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

// Optional epilogue of the decode combine kernel, set by the FLASH_ATTN_EXT + SIGMOID + MUL fusion in ggml-cuda.cu for
// the duration of one flash attention call: gated = sigmoid(gate) * combined, plus the q8_1 of gated into the mmvq
// activation cache for the following single-token GEMV (same arithmetic as gated_q8_kernel<1> / quantize_q8_1).
// If the call has no combine step (single parallel block, stream-k, other kernels) applied stays false.
struct ggml_cuda_fattn_gate_epilogue {
    ggml_backend_cuda_context * ctx      = nullptr;
    const float               * gate     = nullptr;
    ggml_tensor               * mul_node = nullptr;
    bool                        applied  = false;
};
extern thread_local ggml_cuda_fattn_gate_epilogue * ggml_cuda_fattn_epilogue;
