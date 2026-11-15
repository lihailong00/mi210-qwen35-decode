#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

#if defined(GGML_USE_HIP)
bool ggml_cuda_conv_fuse_enabled();
// whether the halo rows of the fused conv fit into the unused part of the concat output
bool ggml_cuda_conv_fuse_geom_ok(int64_t n_t, int64_t C, int64_t col0);
// runs in place of the CONCAT: writes only the parts of its output that are still read later
void ggml_cuda_conv_fuse_prep(ggml_backend_cuda_context & ctx, const ggml_tensor * concat);
#endif // defined(GGML_USE_HIP)
