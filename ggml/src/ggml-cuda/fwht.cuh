#include "common.cuh"

bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op);

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);

#if defined(GGML_USE_HIP)
void ggml_cuda_fwht_q2_0_q8(ggml_backend_cuda_context & ctx, const float * src, const float * signs,
        int width, int cols, int8_t * bytes, float * scales);
#endif
