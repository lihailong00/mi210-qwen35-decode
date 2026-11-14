#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows);

// fused full-attention prep: q/k rms_norm*w + rope, gate copy, k/v cache stores in one kernel
// nodes: rms_q, mul_q, rope_q, cont_gate, rms_k, mul_k, rope_k, set_rows_k, set_rows_v
bool ggml_cuda_attn_prep_supported(const int64_t head_dim, const ggml_type k_type, const ggml_type v_type);

void ggml_cuda_op_attn_prep_fused(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes);
