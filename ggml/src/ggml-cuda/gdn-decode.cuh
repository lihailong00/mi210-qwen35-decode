#pragma once

#include "common.cuh"

// fused single-token, single-sequence gated delta net decode (GGML_OP_GDN_DECODE_PREP / GGML_OP_GDN_DECODE_OUT)

bool ggml_cuda_gdn_decode_supported(const ggml_tensor * op);

void ggml_cuda_op_gdn_decode_prep(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gdn_decode_out(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
