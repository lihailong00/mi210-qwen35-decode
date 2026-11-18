#pragma once

#include "common.cuh"

// fused single-token, single-sequence gated delta net decode (GGML_OP_GDN_DECODE_PREP / GGML_OP_GDN_DECODE_OUT)

// device: the device the op would run on (the N-token / rollback-slot variants need a wave64 device)
bool ggml_cuda_gdn_decode_supported(const ggml_tensor * op, int device);

void ggml_cuda_op_gdn_decode_prep(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gdn_decode_out(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same, and also writes the q8_1 quantization of the output (bit-identical to quantize_q8_1) to y and the per-block
// int sums to ysum (H_v*S/QK8_1 blocks), e.g. into the activation cache of the single-token GEMV that consumes it
void ggml_cuda_op_gdn_decode_out_q8(ggml_backend_cuda_context & ctx, ggml_tensor * dst, block_q8_1 * y, int * ysum);
