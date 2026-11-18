#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

// n single-token GEMVs (weights src0[i], outputs dst[i]) that read the same activation src1, in one launch when a
// module supports the group (currently IQ3_S/IQ3_XXS, mmvq-iq3.cu); returns false (nothing launched) otherwise.
bool ggml_cuda_mul_mat_vec_q_multi(ggml_backend_cuda_context & ctx,
    const ggml_tensor * const * src0, const ggml_tensor * src1, ggml_tensor * const * dst, int n);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
// MTP2-P0: the largest column count each N-column family takes (0: off), defined in the family files, and the type-level
// route predicate (mmvq.cu) used by ggml_cuda_should_use_mmvq on CDNA2
int  ggml_cuda_mmvq_iq3_nc_max();      // A: mmvq-iq3.cu
int  ggml_cuda_mmvq_iq4k_nc_max();     // B: mmvq-iq4k.cu
int  ggml_cuda_mmvq_lowbit_nc_max();   // C: mmvq-lowbit.cu
int  ggml_cuda_mmvq_mix_nc_max();      // D: mmvq-iq3.cu
bool ggml_cuda_mmvq_nc_takes(ggml_type type, int64_t ne11);
// unfused MUL_MAT with more than MMVQ_MAX_BATCH_SIZE columns on the MMVQ route: false = declined, nothing launched
bool ggml_cuda_mul_mat_vec_q_nc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
