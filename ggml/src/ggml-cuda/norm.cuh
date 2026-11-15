#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// shared f16 copies of GEMM inputs (ggml-cuda.cu): buffer for the f16 copy of the tensor t that is about to be written
void * ggml_cuda_f16_share_produce(ggml_backend_cuda_context & ctx, const ggml_tensor * t);
void   ggml_cuda_f16_share_cancel(ggml_backend_cuda_context & ctx, const ggml_tensor * t);

// share_f16: the caller executes exactly the two nodes rms_norm (dst) -> mul (mul_tensor) with this call, so the
// kernel may also write the shared f16 copy of the result (ggml_cuda_f16_share_produce)
void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor, bool share_f16 = false);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
