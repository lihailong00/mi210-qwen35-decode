#define vec_dot_iq3_s_q8_1 vec_dot_iq3_s_q8_1_original
#define vec_dot_iq3_xxs_q8_1 vec_dot_iq3_xxs_q8_1_original
#include "vecdotq.cuh"
#undef vec_dot_iq3_s_q8_1
#undef vec_dot_iq3_xxs_q8_1
#include "vecdotq-iq3-sign.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "iq3-repack.cuh"
#include "mmvq-fused-host.cuh"

#include <cstdint>
#include <type_traits>
#include <map>
#include <mutex>
#include <atomic>

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
// For perf data, see https://github.com/ggml-org/llama.cpp/pull/26705#issuecomment-5569335031
#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
// returns true only for those quants that benefit from prefetch and false otherwise
static constexpr __host__ __device__ bool mmvq_should_prefetch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static __device__ __forceinline__ void mmvq_prefetch_l2(const void * p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
#endif

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5) || defined(RDNA4_M)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: P100, Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_PASCAL || cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ORIN) {
        switch (type) { // tuned for Jetson Orin
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                return ne11 <= 1;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_VOLTA) {
        switch (type) {
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        // MTP2-P0 begin: 2 <= ne11 <= the family maximum of an N-column family (A IQ3, B IQ4_XS/Q4_K, C low-bit) goes to
        // MMVQ; a tensor the family declines at ne11 > MMVQ_MAX_BATCH_SIZE is sent on to MMQ by ggml_cuda_mul_mat_vec_q
        if (GGML_CUDA_CC_IS_CDNA2(cc) && ggml_cuda_mmvq_nc_takes(type, ne11)) {
            return true;
        }
        // MTP2-P0 end
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3) || defined(RDNA4_M)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif !defined(GGML_USE_MUSA) && (__CUDA_ARCH__ == GGML_CUDA_CC_PASCAL || __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
                return 2;
            case 2:
            case 3:
                return type == GGML_TYPE_Q3_K ? 1 : 2;
            case 4:
                return type == GGML_TYPE_Q6_K? 1 : 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

// Low bit types load few weight bytes per thread, a second row per block shares the q8_1 loads and adds memory parallelism.
static constexpr __host__ __device__ bool mmvq_two_rows_ncols1(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
            return true;
        default:
            return false;
    }
}

static constexpr __host__ __device__ int calc_rows_per_block(ggml_type type, int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                if (small_k) {
                    return nwarps;
                }
                if (table_id == MMVQ_PARAMETERS_GB10) {
                    return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S ? 2 : 1;
                }
                return table_id == MMVQ_PARAMETERS_GENERIC && mmvq_two_rows_ncols1(type) ? 2 : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = shared_expert ? 0 : ncols_dst == 1 && ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
        // start the next iterations' weight loads early
        if constexpr (mmvq_should_prefetch(type)) {
            // loop iterations, not blocks; q3_K and q6_K keep the distance in blocks they had with vdr 1
            constexpr int pf_dist = type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q6_K ? 1 : 2;
            const int kbx_pf = kbx + pf_dist*blocks_per_iter;
            if (kbx_pf < blocks_per_row_x) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    const size_t off = (size_t)(kbx_offset + i*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
                    mmvq_prefetch_l2((const char *) vx + off);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            mmvq_prefetch_l2((const char *) vgate + off);
                        }
                    }
                }
            }
        }
#endif

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            case GGML_GLU_OP_SWIGLU_CLAMP:
                                result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block, bool has_fusion = false>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    // fuse gate, bias, scales, and glu_op into the up projection
    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    const float * x_scale    = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            x_scale    = (const float *) fusion.x_scale;
            gate_scale = (const float *) fusion.gate_scale;
        }
    }

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = shared_expert ? 0 : ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            const uint32_t bias_idx = channel_x*stride_channel_dst + row0 + threadIdx.x;

            if constexpr (type == GGML_TYPE_NVFP4) {
                if (x_scale) {
                    result *= x_scale[channel_x];
                }
            }
            if (x_bias) {
                result += x_bias[bias_idx];
            }
            if (use_gate) {
                float gate_value = tmp_gate[threadIdx.x];
                if constexpr (type == GGML_TYPE_NVFP4) {
                    if (gate_scale) {
                        gate_value *= gate_scale[channel_x];
                    }
                }
                if (gate_bias) {
                    gate_value += gate_bias[bias_idx];
                }
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    case GGML_GLU_OP_SWIGLU_CLAMP:
                        result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            }
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, tmp_gate, vgate, x_bias, gate_bias, active_glu, glu_limit, x_scale, gate_scale);
    } else if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(x_scale, gate_scale);
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst + (fusion.shared_up != nullptr));
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;

    if (has_fusion) {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, true>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, false>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    }
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        const int  nwarps = calc_nwarps(type, c_ncols_dst, table_id);
        bool       use    = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    // Whether doubling nwarps pays off on the ncols_dst == 1 path, where K sets the K loop trip count.
    const auto should_halve_iters = [&] {
        if (table_id != MMVQ_PARAMETERS_GB10) {
            return false;
        }

        // Expert rows are gathered per token, so a wider block adds reduction work without reuse.
        if (has_ids) {
            return false;
        }

        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);

        // An odd trip count leaves half the wider block idle for its last iteration, that tail is
        // only affordable once the loop is long enough to dilute it to an eighth of the work (observation).
        const int idle = iters_wide * 2 - iters;

        return idle * 8 <= iters_wide * 2;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;

            // Tag types keep the flags compile-time, so __launch_bounds__ matches what is launched.
            const auto launch = [&](auto small_k_tag, auto halve_iters_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
                // Types the table does not promote would compile a second, identical kernel.
                constexpr bool c_promoted =
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, true) !=
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, false);

                constexpr bool c_halve_iters = decltype(halve_iters_tag)::value && c_promoted;

                const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst + (fusion.shared_up != nullptr),
                                                                              nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            };

            if (should_use_small_k(c_ncols_dst)) {
                launch(std::true_type{},  std::false_type{});
            } else if (should_halve_iters()) {
                launch(std::false_type{}, std::true_type{});
            } else {
                launch(std::false_type{}, std::false_type{});
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// GGML_MMVQ_Q8_CACHE=0 disables the per-stream activation cache (plain host code: also used outside the HIP-only
// section, ggml_cuda_mul_mat_vec_q)
static bool mmvq_q8_cache_disabled() {
    static const bool disabled = [] {
        const char * v = getenv("GGML_MMVQ_Q8_CACHE");
        return v != nullptr && v[0] == '0';
    }();
    return disabled;
}

#if defined(GGML_USE_HIP)
static __global__ void codex_iq3_sum_q8(const block_q8_1 * x, int * sums, int groups) {
    const int g = blockIdx.x*blockDim.x+threadIdx.x;
    if (g >= groups) return;
    int sum = 0;
#pragma unroll
    for (int j = 0; j < 8; j++) sum = ggml_cuda_dp4a(0x01010101,get_int_b4(x[g].qs,j),sum);
    sums[g] = sum;
}

static __device__ __forceinline__ float codex_iq3_block_dot(const uint32_t * codes,
        const codex_iq3_meta & m, const block_q8_1 & x, int sum_x, int group) {
    int value = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const uint32_t pair = codes[j];
        value = ggml_cuda_dp4a(pair & 0x0f0f0f0fu,get_int_b4(x.qs,2*j),value);
        value = ggml_cuda_dp4a((pair>>4) & 0x0f0f0f0fu,get_int_b4(x.qs,2*j+1),value);
    }
    value = 2*value-15*sum_x;
    value *= 1+2*((m.scales[group/2]>>((group&1)*4))&15);
    const float d = __half2float(m.d) * __low2float(x.ds);
    return d*value;
}

alignas(4) static __device__ __constant__ int8_t codex_iq3_xxs_levels[16] = {
     4, 12, 20, 28, 36, 44, 52, 62,
    -4,-12,-20,-28,-36,-44,-52,-62
};

static __device__ __forceinline__ float codex_iq3_xxs_block_dot(const uint32_t * codes,
        const codex_iq3_meta & m, const block_q8_1 & x, int group) {
    int value = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int2 v = get_int_from_table_16(codes[j],codex_iq3_xxs_levels);
        value = ggml_cuda_dp4a(v.x,get_int_b4(x.qs,2*j),value);
        value = ggml_cuda_dp4a(v.y,get_int_b4(x.qs,2*j+1),value);
    }
    const int ls = (m.scales[group/2]>>((group&1)*4))&15;
    value = (ls*value + value/2)/2;
    const float d = __half2float(m.d) * __low2float(x.ds);
    return d*value;
}

// R rows per workgroup of NT threads: every thread has R (2R when fused) independent weight streams in flight and
// the activation / sum loads are shared by the R rows. (R=1, NT=128 is the original retained kernel.)
template<bool fused, bool add_residual, bool xxs, int R, int NT>
__launch_bounds__(NT)
static __global__ void codex_iq3_packed_gemv(const uint32_t * up, const codex_iq3_meta * up_meta,
        const uint32_t * gate, const codex_iq3_meta * gate_meta, const block_q8_1 * x,
        const int * sums, float * out, int groups, const float * residual) {
    constexpr int NWV = NT/64;
    const int row0 = blockIdx.x*R, lane = threadIdx.x;
    float result[R], gate_result[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        result[r] = 0.f;
        gate_result[r] = 0.f;
    }
    for (int g = lane; g < groups; g += NT) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const size_t pos = size_t(row0 + r)*groups+g;
            if constexpr (xxs) {
                result[r] += codex_iq3_xxs_block_dot(up+pos*4,up_meta[pos/8],x[g],g&7);
                if constexpr (fused) gate_result[r] += codex_iq3_xxs_block_dot(gate+pos*4,gate_meta[pos/8],x[g],g&7);
            } else {
                result[r] += codex_iq3_block_dot(up+pos*4,up_meta[pos/8],x[g],sums[g],g&7);
                if constexpr (fused) gate_result[r] += codex_iq3_block_dot(gate+pos*4,gate_meta[pos/8],x[g],sums[g],g&7);
            }
        }
    }
    __shared__ float part[NWV > 1 ? NWV : 1][R][2];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        result[r] = warp_reduce_sum<64>(result[r]);
        if constexpr (fused) gate_result[r] = warp_reduce_sum<64>(gate_result[r]);
        if constexpr (NWV > 1) {
            if (lane % 64 == 0) {
                part[lane/64][r][0] = result[r];
                part[lane/64][r][1] = gate_result[r];
            }
        }
    }
    if constexpr (NWV > 1) {
        __syncthreads();
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float res  = result[r];
            float gres = gate_result[r];
            if constexpr (NWV > 1) {
#pragma unroll
                for (int w = 1; w < NWV; ++w) {
                    res  += part[w][r][0];
                    gres += part[w][r][1];
                }
            }
            if constexpr (fused) res *= ggml_cuda_op_silu_single(gres);
            if constexpr (add_residual) res += residual[row0 + r];
            out[row0 + r] = res;
        }
    }
}

// (rows per workgroup, threads per workgroup) for the packed IQ3 GEMV; tuned on MI210, env-overridable
struct codex_cfg { int R, NT; };
static codex_cfg codex_get_cfg(const bool xxs, const bool fused, const int K) {
    codex_cfg c;
    if (!xxs) {
        c = fused ? codex_cfg{ 1, 128 } : codex_cfg{ 2, 256 };
    } else if (fused) {
        c = codex_cfg{ 1, 64 };
    } else {
        // IQ3_XXS (VALU-heavier decode): thread count by row length (MI210 harness: 17408x5120 46.4 us @64,
        // 5120x6144 18.0 us @192, 5120x17408 46.1 us @256)
        c = K <= 5120 ? codex_cfg{ 2, 64 } : (K <= 8192 ? codex_cfg{ 2, 192 } : codex_cfg{ 2, 256 });
    }
    const char * names[2][2] = { { "GGML_CODEX_S", "GGML_CODEX_S_FUSED" }, { "GGML_CODEX_XXS", "GGML_CODEX_XXS_FUSED" } };
    const char * v = getenv(names[xxs][fused]);   // format "R,NT"
    if (v) {
        int r = 0, nt = 0;
        if (sscanf(v, "%d,%d", &r, &nt) == 2 && (r == 1 || r == 2 || r == 4 || r == 8) && (nt == 64 || nt == 128 || nt == 192 || nt == 256)) {
            c = { r, nt };
        }
    }
    return c;
}
template<bool xxs>
static void codex_iq3_launch_packed(ggml_backend_cuda_context & ctx,
        const uint32_t * up, const uint32_t * gate, const block_q8_1 * q8, float * dst,
        const float * residual, size_t count, int groups, int rows, cudaStream_t stream, const int * sums_pre = nullptr) {
    ggml_cuda_pool_alloc<int> sums_alloc(ctx.pool());
    const int * sums = sums_pre;
    if constexpr (!xxs) {
        if (!sums) {
            sums_alloc.alloc(groups);
            codex_iq3_sum_q8<<<(groups+127)/128,128,0,stream>>>(q8,sums_alloc.get(),groups);
            sums = sums_alloc.get();
        }
    }
    const auto * up_meta   = reinterpret_cast<const codex_iq3_meta *>(up+count*32);
    const auto * gate_meta = gate ? reinterpret_cast<const codex_iq3_meta *>(gate+count*32) : nullptr;
    codex_cfg c = codex_get_cfg(xxs, gate != nullptr, groups*32);
    while (rows % c.R != 0) {
        c.R /= 2;
    }
#define CODEX_LAUNCH(RR, NTT) \
    if (gate) { \
        codex_iq3_packed_gemv<true,false,xxs,RR,NTT><<<rows/RR,NTT,0,stream>>>(up,up_meta,gate,gate_meta,q8,sums,dst,groups,nullptr); \
    } else if (residual) { \
        codex_iq3_packed_gemv<false,true,xxs,RR,NTT><<<rows/RR,NTT,0,stream>>>(up,up_meta,nullptr,nullptr,q8,sums,dst,groups,residual); \
    } else { \
        codex_iq3_packed_gemv<false,false,xxs,RR,NTT><<<rows/RR,NTT,0,stream>>>(up,up_meta,nullptr,nullptr,q8,sums,dst,groups,nullptr); \
    }
    if      (c.R == 1 && c.NT ==  64) { CODEX_LAUNCH(1,  64) }
    else if (c.R == 1 && c.NT == 256) { CODEX_LAUNCH(1, 256) }
    else if (c.R == 2 && c.NT ==  64) { CODEX_LAUNCH(2,  64) }
    else if (c.R == 2 && c.NT == 128) { CODEX_LAUNCH(2, 128) }
    else if (c.R == 2 && c.NT == 192) { CODEX_LAUNCH(2, 192) }
    else if (c.R == 2 && c.NT == 256) { CODEX_LAUNCH(2, 256) }
    else if (c.R == 4 && c.NT ==  64) { CODEX_LAUNCH(4,  64) }
    else if (c.R == 4 && c.NT == 128) { CODEX_LAUNCH(4, 128) }
    else if (c.R == 4 && c.NT == 192) { CODEX_LAUNCH(4, 192) }
    else if (c.R == 4 && c.NT == 256) { CODEX_LAUNCH(4, 256) }
    else if (c.R == 1 && c.NT == 192) { CODEX_LAUNCH(1, 192) }
    else if (c.R == 8 && c.NT ==  64) { CODEX_LAUNCH(8,  64) }
    else                              { CODEX_LAUNCH(1, 128) }
#undef CODEX_LAUNCH
}

// Per-stream cache of the q8_1-quantized activation (+ per-block int sums for the packed IQ3 path): a single-token
// activation, or (MMVQ-NC) the flattened [K, 2..nc_max] activation of a multi-token verification step.
// Consecutive GEMVs of one graph evaluation that consume the same src1 tensor (qkv/z, q/k/v, gate/up) quantize it once.
// The buffer is dedicated (not from the pool), so the cached q8 data itself cannot be overwritten by intermediate ops.
// The entry is dropped
//  - at the start of every graph evaluation (ggml_cuda_mmvq_cache_reset),
//  - whenever a node writes bytes of the cached f32 source (ggml_cuda_mmvq_cache_invalidate, called by the graph loop
//    before every executed node / node group and, for all members of a fused group, after the group): covers in-place
//    ops (res += a@b folded into the GEMV), allocator reuse and writes through views. After a fused group an entry the
//    group itself filled for one of its members (q8 producers: rms_norm*w, swiglu, FA gate, ...) is only checked
//    against the members that follow that member.
// Lookups never happen between a write and its invalidation: a fused group reads cached activations only of tensors
// that no earlier member of the same group writes (the fusions only read their inputs, then write their outputs).
static __global__ void quantize_q8_1_sums(const float * __restrict__ x, block_q8_1 * __restrict__ y, int * __restrict__ sums,
        const int64_t ne10, const int64_t ne10_padded) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= ne10_padded) {
        return;
    }
    const float xi = i < ne10 ? x[i] : 0.0f;
    float amax = fabsf(xi);
    amax = warp_reduce_max<QK8_1>(amax);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    const int sum = warp_reduce_sum<QK8_1>((int) q);
    y[i/QK8_1].qs[i%QK8_1] = q;
    if (i % QK8_1 == 0) {
        y[i/QK8_1].ds = make_half2(d, d * sum);
        sums[i/QK8_1] = sum;
    }
}

// The identity of the cached activation is kept by value: base = root tensor (view_src or the tensor itself) at the
// time of insertion, src = the tensor itself, both only ever compared as pointer values, never dereferenced (the graph
// that owned them may be freed). [lo, hi): the bytes of the f32 source the q8 data was made from (overlap checks).
// epoch: the node group (ggml_cuda_mmvq_cache_invalidate, MMVQ_Q8_INV_NODE) that filled the entry.
struct mmvq_q8_cache_entry {
    char              * buf  = nullptr;
    size_t              size = 0;
    const ggml_tensor * base = nullptr;
    const ggml_tensor * src  = nullptr;
    const void        * data = nullptr;
    const char        * lo   = nullptr;
    const char        * hi   = nullptr;
    int64_t             ne10 = 0;
    uint64_t            epoch = 0;
    bool                valid = false;
};
static std::mutex mmvq_q8_cache_mutex;
static std::map<std::pair<int, cudaStream_t>, mmvq_q8_cache_entry> mmvq_q8_caches;
// number of valid entries (all devices; lets the per-node invalidation skip the lock while nothing is cached) and the
// node-group counter of every device (advanced by the thread that evaluates graphs on that device)
static std::atomic<int> mmvq_q8_n_valid{0};
static std::atomic<uint64_t> mmvq_q8_epoch[GGML_CUDA_MAX_DEVICES] = {};

static void mmvq_q8_entry_drop(mmvq_q8_cache_entry & e) {
    if (e.valid) {
        mmvq_q8_n_valid.fetch_sub(1, std::memory_order_relaxed);
    }
    e.valid = false;
    e.base  = nullptr;
    e.src   = nullptr;
    e.data  = nullptr;
    e.lo    = nullptr;
    e.hi    = nullptr;
}

static void mmvq_q8_entry_fill(mmvq_q8_cache_entry & e, const int device, const ggml_tensor * t, const int64_t ne10) {
    if (!e.valid) {
        mmvq_q8_n_valid.fetch_add(1, std::memory_order_relaxed);
    }
    e.base  = t->view_src ? t->view_src : t;
    e.src   = t;
    e.data  = t->data;
    e.lo    = (const char *) t->data;
    e.hi    = e.lo + ggml_nbytes(t);
    e.ne10  = ne10;
    e.epoch = mmvq_q8_epoch[device].load(std::memory_order_relaxed);
    e.valid = true;
}

void ggml_cuda_mmvq_cache_reset(ggml_backend_cuda_context & ctx) {
    std::lock_guard<std::mutex> lock(mmvq_q8_cache_mutex);
    for (auto & e : mmvq_q8_caches) {
        if (e.first.first == ctx.device) {
            mmvq_q8_entry_drop(e.second);
        }
    }
}

static bool mmvq_q8_overlaps(const mmvq_q8_cache_entry & e, const ggml_tensor * t) {
    const char * lo = (const char *) t->data;
    return lo == nullptr || (lo < e.hi && e.lo < lo + ggml_nbytes(t));
}

// node is about to be written (MMVQ_Q8_INV_NODE: a node or the first node of a fused group, starts a new group;
// MMVQ_Q8_INV_AHEAD: a member that a fusion runs ahead of the rest of its group): drop the entries of this device (any
// stream) whose source bytes it overlaps
void ggml_cuda_mmvq_cache_invalidate(ggml_backend_cuda_context & ctx, const ggml_tensor * node, const int mode) {
    if (mode == MMVQ_Q8_INV_NODE) {
        mmvq_q8_epoch[ctx.device].fetch_add(1, std::memory_order_relaxed);
    }
    if (mmvq_q8_n_valid.load(std::memory_order_relaxed) == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(mmvq_q8_cache_mutex);
    for (auto & it : mmvq_q8_caches) {
        mmvq_q8_cache_entry & e = it.second;
        if (it.first.first == ctx.device && e.valid && mmvq_q8_overlaps(e, node)) {
            mmvq_q8_entry_drop(e);
        }
    }
}

// the fused group nodes[0..n) just ran and may have written every (non-view) member: drop the entries whose source
// bytes a member overlaps, except an entry the group itself filled for its member p (a q8 producer, or a GEMV of the
// group that read p) against the members k <= p (written before p got the contents that were quantized)
void ggml_cuda_mmvq_cache_invalidate_group(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes, const int n) {
    if (mmvq_q8_n_valid.load(std::memory_order_relaxed) == 0) {
        return;
    }
    const uint64_t epoch = mmvq_q8_epoch[ctx.device].load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mmvq_q8_cache_mutex);
    for (auto & it : mmvq_q8_caches) {
        mmvq_q8_cache_entry & e = it.second;
        if (it.first.first != ctx.device || !e.valid) {
            continue;
        }
        int p = -1;
        if (e.epoch == epoch) {
            for (int j = 0; j < n; ++j) {
                if (nodes[j] == e.src) {
                    p = j;
                    break;
                }
            }
        }
        for (int k = p + 1; k < n; ++k) {
            const ggml_tensor * t = nodes[k];
            const bool view_or_noop = ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
                t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;   // (= ggml_cuda_is_view_or_noop)
            if (!view_or_noop && mmvq_q8_overlaps(e, t)) {
                mmvq_q8_entry_drop(e);
                break;
            }
        }
    }
}

// drop the entry of a stream that is about to be destroyed (its handle may be recycled by a new stream)
void ggml_cuda_mmvq_cache_release(const int device, cudaStream_t stream) {
    char * buf = nullptr;
    {
        std::lock_guard<std::mutex> lock(mmvq_q8_cache_mutex);
        const auto it = mmvq_q8_caches.find({device, stream});
        if (it == mmvq_q8_caches.end()) {
            return;
        }
        buf = it->second.buf;
        mmvq_q8_entry_drop(it->second);
        mmvq_q8_caches.erase(it);
    }
    if (buf) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(buf));
    }
}

// Producer side: a kernel that creates the single-row activation t (K % 512 == 0) also writes its q8_1 blocks + sums
// into the cache buffer; returns false if the cache cannot be used (then the producer must not write q8 data).
bool ggml_cuda_mmvq_cache_produce(ggml_backend_cuda_context & ctx, const ggml_tensor * t, block_q8_1 ** q8, int ** sums) {
    const int64_t ne10 = ggml_nelements(t);
    if (mmvq_q8_cache_disabled() || ne10 % MATRIX_ROW_PADDING != 0 || !ggml_is_contiguous(t)) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    const size_t q8_bytes = ne10/QK8_1*sizeof(block_q8_1);
    const size_t need = q8_bytes + ne10/QK8_1*sizeof(int) + 256;
    std::lock_guard<std::mutex> lock(mmvq_q8_cache_mutex);
    mmvq_q8_cache_entry & e = mmvq_q8_caches[{ctx.device, stream}];
    if (e.size < need) {
        hipStreamCaptureStatus status;
        CUDA_CHECK(hipStreamIsCapturing(stream, &status));
        if (status != hipStreamCaptureStatusNone) {
            return false;
        }
        if (e.buf) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaFree(e.buf));
        }
        e.size = std::max(need, (size_t) 1 << 20);
        CUDA_CHECK(cudaMalloc(&e.buf, e.size));
        mmvq_q8_entry_drop(e);
    }
    const size_t sums_off = (q8_bytes + 255) & ~(size_t) 255;
    *q8   = (block_q8_1 *) e.buf;
    *sums = (int *) (e.buf + sums_off);
    mmvq_q8_entry_fill(e, ctx.device, t, ne10);
    return true;
}

// returns false if the cache cannot be used (then the caller quantizes into a pool buffer)
static bool mmvq_q8_cache_get(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const int64_t ne10, const int64_t ne10_padded,
        const block_q8_1 ** q8, const int ** sums) {
    if (mmvq_q8_cache_disabled()) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    const size_t q8_bytes = ne10_padded/QK8_1*sizeof(block_q8_1);
    const size_t need = q8_bytes + ne10_padded/QK8_1*sizeof(int) + 256;
    std::lock_guard<std::mutex> lock(mmvq_q8_cache_mutex);
    mmvq_q8_cache_entry & e = mmvq_q8_caches[{ctx.device, stream}];
    if (e.size < need) {
        hipStreamCaptureStatus status;
        CUDA_CHECK(hipStreamIsCapturing(stream, &status));
        if (status != hipStreamCaptureStatusNone) {
            return false;
        }
        if (e.buf) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaFree(e.buf));
        }
        e.size = std::max(need, (size_t) 1 << 20);
        CUDA_CHECK(cudaMalloc(&e.buf, e.size));
        mmvq_q8_entry_drop(e);
    }
    const size_t sums_off = (q8_bytes + 255) & ~(size_t) 255;
    *q8   = (const block_q8_1 *) e.buf;
    *sums = (const int *) (e.buf + sums_off);
    const ggml_tensor * base_src1 = src1->view_src ? src1->view_src : src1;
    if (e.valid && e.base == base_src1 && e.data == src1->data && e.ne10 == ne10) {
        return true;
    }
    const int64_t nblk = (ne10_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    quantize_q8_1_sums<<<nblk, CUDA_QUANTIZE_BLOCK_SIZE, 0, stream>>>((const float *) src1->data, (block_q8_1 *) e.buf,
            (int *) (e.buf + sums_off), ne10, ne10_padded);
    CUDA_CHECK(cudaGetLastError());
    mmvq_q8_entry_fill(e, ctx.device, src1, ne10);
    return true;
}

bool ggml_cuda_mmvq_q8_cached(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const block_q8_1 ** q8) {
    if (src1->type != GGML_TYPE_F32 || src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || !ggml_is_contiguous(src1)) {
        return false;
    }
    const int64_t ne10 = src1->ne[0];
    const int * sums = nullptr;
    return mmvq_q8_cache_get(ctx, src1, ne10, GGML_PAD(ne10, MATRIX_ROW_PADDING), q8, &sums);
}
// MMVQ-NC S: the [K, ncols] activation quantized as one flat K*ncols row (K % 512 == 0: no padding, no block crosses a
// column, so column c is byte-identical to the single-row quantization of that column); cache key (base, data, K*ncols)
bool ggml_cuda_mmvq_q8_cached_nc(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const block_q8_1 ** q8, const int ** sums) {
    if (src1->type != GGML_TYPE_F32 || !ggml_is_contiguous(src1) || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            src1->ne[0] % MATRIX_ROW_PADDING != 0 || src1->ne[1] < 1 || src1->ne[1] > std::max(1, ggml_cuda_mmvq_nc_max())) {
        return false;
    }
    const int64_t n = src1->ne[0]*src1->ne[1];
    return mmvq_q8_cache_get(ctx, src1, n, n, q8, sums);
}
#else
bool ggml_cuda_mmvq_q8_cached(ggml_backend_cuda_context &, const ggml_tensor *, const block_q8_1 **) {
    return false;
}
bool ggml_cuda_mmvq_q8_cached_nc(ggml_backend_cuda_context &, const ggml_tensor *, const block_q8_1 **, const int **) {
    return false;
}
// the activation cache is HIP-only; callers outside HIP-only code (fattn-common.cuh launch_fattn, fattn-dec.cu,
// q8-producers.cu, ggml-cuda.cu) must link in every build
void ggml_cuda_mmvq_cache_reset(ggml_backend_cuda_context &) {
}
void ggml_cuda_mmvq_cache_invalidate(ggml_backend_cuda_context &, const ggml_tensor *, int) {
}
void ggml_cuda_mmvq_cache_invalidate_group(ggml_backend_cuda_context &, ggml_tensor * const *, int) {
}
bool ggml_cuda_mmvq_cache_produce(ggml_backend_cuda_context &, const ggml_tensor *, block_q8_1 ** q8, int ** sums) {
    *q8   = nullptr;
    *sums = nullptr;
    return false;
}
void ggml_cuda_mmvq_cache_release(int, cudaStream_t) {
}
#endif

// ------------------------------------------------------------------------------------------------ MMVQ-NC S
// N-column GEMVs (2 <= ne11 <= nc_max) for multi-token verification: GGML_MMVQ_NC=0 restores the previous behaviour for
// ne11 > 1 exactly; GGML_MMVQ_NC_MAX (default 16, 2..16) is the largest column count taken. Every family reads the
// switches through these two functions (one parse: any value starting with '0' turns the master switch off).
// Output change (accepted): the kernels are not limited to speculative verification. Every MUL_MAT with 2..nc_max
// columns takes them, also prompt chunks of that size (short prompts, the last chunk of a prompt split into ubatches),
// which ran on the stock ncols > 1 kernels before. The N-column kernels give every column exactly the single-token
// (N=1) result, the stock ones round differently, so greedy output can change; measured on the server without
// speculative decoding (K=0): 2 of 4 prompts differ from the build without MMVQ-NC. This moves the results closer to
// N=1 decode, it is not a regression; GGML_MMVQ_NC=0 restores the previous kernels and outputs exactly.
bool ggml_cuda_mmvq_nc_enabled() {
    static const bool enabled = [] {
        const char * v = getenv("GGML_MMVQ_NC");
        if (v == nullptr) {
            return true;
        }
        while (*v == ' ' || *v == '\t') {
            ++v;
        }
        return !(v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N');
    }();
    return enabled;
}
int ggml_cuda_mmvq_nc_max() {
    static const int n = [] {
        if (!ggml_cuda_mmvq_nc_enabled()) {
            return 0;
        }
        const char * v = getenv("GGML_MMVQ_NC_MAX");
        const int m = v ? atoi(v) : 16;   // MTP2-P0: default 16 (every verification size of MTP K <= 15 / n-gram drafts)
        // MTP2-P0: GGML_MMVQ_NC_MAX 2..16
        return std::min(16, std::max(2, m));
    }();
    return n;
}
// MTP2-P0 begin: does an N-column family take ne11 columns of this weight type (type level; the family launcher still
// checks the tensor, and a declined call with ne11 > MMVQ_MAX_BATCH_SIZE goes to MMQ in ggml_cuda_mul_mat_vec_q)
bool ggml_cuda_mmvq_nc_takes(ggml_type type, int64_t ne11) {
    if (ne11 < 2 || ne11 > 16) {
        return false;
    }
    switch (type) {
        case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ3_XXS:
            return ne11 <= ggml_cuda_mmvq_iq3_nc_max();
        case GGML_TYPE_IQ4_XS: case GGML_TYPE_Q4_K:
            return ne11 <= ggml_cuda_mmvq_iq4k_nc_max();
        case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_Q2_K: case GGML_TYPE_IQ1_M:
            return ne11 <= ggml_cuda_mmvq_lowbit_nc_max();
        default:
            return false;
    }
}
// MTP2-P0 end
// the weight types an N-column family handles: only these read the N-column activation from the q8 cache, so the stock
// ne11 > 1 kernels of every other type keep the pool quantization exactly as before (the cache entry is dropped whenever
// a node writes its source bytes, see above, so a cached N-column activation is always the current one)
static bool ggml_cuda_mmvq_nc_type(const ggml_type t) {
    switch (t) {
        case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ3_XXS:                                         // A
        case GGML_TYPE_IQ4_XS: case GGML_TYPE_Q4_K:                                           // B
        case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_Q2_K: case GGML_TYPE_IQ1_M:   // C
            return true;
        default:
            return false;
    }
}
// will ggml_cuda_mul_mat_vec_q take this N-column MUL_MAT with the fusion (gate_w: SwiGLU gate weights; residual add)?
// Exact: true only if one of the NC slots below will accept the call (a promised fusion that is not taken aborts).
// Invariant shared with the only callers, the SwiGLU and ADD fusion sites of ggml_cuda_try_fuse: besides what is checked
// here they pass nothing but (a) gate + glu_op = SWIGLU (not swapped), dst = the F32 contiguous GLU node of mm's shape,
// gate and up on the same src1, or (b) x_bias = an F32 contiguous residual of mm's shape, dst = the F32 contiguous ADD
// node of mm's shape; no ids, gate_bias, scales or shared experts. Given that, every slot's launcher (_iq3_nc,
// _lowbit_nc, _iq4k_nc) checks nothing beyond its _ok predicate (dst / residual layout, fusion fields), so the slot
// whose _ok is true takes the call; ggml_cuda_mul_mat_vec_q aborts if that is ever violated.
// Side effect (family C): outside stream capture the low-bit SoA copy of mm->src[0] is created here if missing, so the
// fusion can already form on the first evaluation; during capture nothing is allocated and the answer is then false.
bool ggml_cuda_mmvq_nc_fusion_ok(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * gate_w, const bool residual) {
    if (mm->op != GGML_OP_MUL_MAT || mm->src[2] || mm->type != GGML_TYPE_F32 || !ggml_is_contiguous(mm)) {
        return false;
    }
    const ggml_tensor * src1 = mm->src[1];
    if (src1->type != GGML_TYPE_F32 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            mm->ne[1] != src1->ne[1] || src1->ne[1] < 2 || src1->ne[1] > ggml_cuda_mmvq_nc_max()) {
        return false;
    }
    const int ncols = (int) src1->ne[1];
    const ggml_tensor * w = mm->src[0];
    if (ggml_cuda_mmvq_iq3_nc_ok(w, gate_w, residual, ncols)) {   // MMVQ-NC A (slot 1)
        return true;
    }
    ggml_cuda_mmvq_lowbit_nc_prepare(ctx, w, ncols);               // MMVQ-NC C (slot 2)
    if (ggml_cuda_mmvq_lowbit_nc_ok(w, gate_w, residual, ncols)) {
        return true;
    }
    bool capturing = false;   // (the iq4k slot exists in HIP builds only)
#if defined(GGML_USE_HIP)
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    CUDA_CHECK(hipStreamIsCapturing(ctx.stream(), &status));
    capturing = status != hipStreamCaptureStatusNone;
#endif
    return ggml_cuda_mmvq_iq4k_nc_ok(w, gate_w, residual, ncols, capturing);  // MMVQ-NC B (slot 3)
}
// ------------------------------------------------------------------------------------------------ MMVQ-NC S end

// MTP2-P0: decline signal of an unfused N-column call above MMVQ_MAX_BATCH_SIZE (see ggml_cuda_mul_mat_vec_q_nc)
static thread_local bool mmvq_nc_allow_decline = false;
static thread_local bool mmvq_nc_declined      = false;
void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        GGML_ASSERT( !ids || dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc));
        GGML_ASSERT(  ids || dst->ne[1] == 1 || (ne11 >= 2 && ne11 <= ggml_cuda_mmvq_nc_max()));   // MMVQ-NC S: N-column fusion
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->shared_up) {
            GGML_ASSERT(ids && fusion->gate && fusion->shared_gate && fusion->shared_dst);
            GGML_ASSERT(!fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
            GGML_ASSERT(ne11 == 1 && ne03 == 1 && ne13 == 1);
            GGML_ASSERT(fusion->shared_up->type == src0->type && fusion->shared_gate->type == src0->type);
            GGML_ASSERT(ggml_are_same_shape(fusion->shared_up, fusion->shared_gate));
            GGML_ASSERT(ggml_is_contiguous(fusion->shared_up) && ggml_is_contiguous(fusion->shared_gate));
            GGML_ASSERT(fusion->shared_up->ne[0] == ne00 && fusion->shared_up->ne[1] == ne01);
            GGML_ASSERT(fusion->shared_up->nb[1] == nb01 && ggml_is_matrix(fusion->shared_up));
            GGML_ASSERT(fusion->shared_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->shared_dst));
            GGML_ASSERT(fusion->shared_dst->ne[0] == ne0 && fusion->shared_dst->ne[1] == ne2);
            fusion_local.shared_up   = fusion->shared_up->data;
            fusion_local.shared_gate = fusion->shared_gate->data;
            fusion_local.shared_dst  = (float *) fusion->shared_dst->data;
            fusion_local.shared_stride_col_dst = fusion->shared_dst->nb[1] / ts_dst;
        }

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            // mixed IQ3_S/IQ3_XXS gate/up: only the iq3gemv SwiGLU kernel (mmvq-iq3.cu) handles it
            GGML_ASSERT((fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0)) ||
                        (!ids && fusion->glu_op == GGML_GLU_OP_SWIGLU && ggml_cuda_mmvq_iq3_fuse_mixed_ok(src0, fusion->gate)));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
    }

    // single token: GEMV kernels that quantize the activation in-kernel (no quantize_q8_1 / sum launches).
    // Q4_K is handled below on the same kernel with the cached quantized activation when the cache is usable;
    // IQ4_XS/Q4_K GEMVs taken by the iq4k path below skip this (iq4k is faster, incl. the LM head).
    if (!ids && ne11 == 1 && ggml_backend_buffer_get_usage(src0->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
            (src0->type != GGML_TYPE_Q4_K || mmvq_q8_cache_disabled()) &&
            !ggml_cuda_mmvq_iq4k_wants(src0) &&
            ggml_cuda_mmvq_fused(ctx, src0, src1, dst, fusion)) {
        return;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1_alloc(ctx.pool());
    const block_q8_1 * src1_q8_1_ptr = nullptr;
    const int * src1_sums = nullptr;
#if defined(GGML_USE_HIP)
    // MMVQ-NC S: 2 <= ne11 <= nc_max with K % 512 == 0: the flattened activation (same column layout as the pool path)
    const bool use_q8_cache = !ids && ne12 == 1 && ne13 == 1 && ggml_is_contiguous(src1) &&
            (ne11 == 1 ? mmvq_q8_cache_get(ctx, src1, ne10, ne10_padded, &src1_q8_1_ptr, &src1_sums) :
             ne11 <= ggml_cuda_mmvq_nc_max() && ggml_cuda_mmvq_nc_type(src0->type) &&
             ggml_cuda_mmvq_q8_cached_nc(ctx, src1, &src1_q8_1_ptr, &src1_sums));
#else
    const bool use_q8_cache = false;
#endif
    if (!use_q8_cache) {
        src1_q8_1_alloc.alloc(ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1_alloc.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        src1_q8_1_ptr = (const block_q8_1 *) src1_q8_1_alloc.get();
        src1_sums = nullptr;
    }

#if defined(GGML_USE_HIP)
    // IQ3_S / IQ3_XXS on the iq3gemv SoA re-layout (no codex packed copies are created for tensors handled here)
    if (!ids && ne11 == 1 && ne12 == 1 && ne13 == 1 &&
            ggml_cuda_mmvq_iq3(ctx, src0, src1_q8_1_ptr, dst, fusion)) {
        return;
    }
    // MMVQ-NC S: N-column GEMVs; a fused call (accepted by ggml_cuda_mmvq_nc_fusion_ok) must be taken by one of them
    if (!ids && ne11 >= 2 && ne11 <= ggml_cuda_mmvq_nc_max() && ne12 == 1 && ne13 == 1) {
        if (ggml_cuda_mmvq_iq3_nc(ctx, src0, src1_q8_1_ptr, (int) ne11, dst, fusion)) {   // MMVQ-NC A (slot 1)
            return;
        }
        if (ggml_cuda_mmvq_lowbit_nc(ctx, src0, src1_q8_1_ptr, (int) ne11, dst, fusion)) {   // MMVQ-NC C (slot 2)
            return;
        }
        if (ggml_cuda_mmvq_iq4k_nc_checked(ctx, src0, src1, src1_q8_1_ptr, (int) ne11, dst, fusion)) {   // MMVQ-NC B (slot 3)
            return;
        }
        if (fusion) {
            GGML_ABORT("MMVQ-NC: fused N-column GEMV (%s, ncols %d) not taken", ggml_type_name(src0->type), (int) ne11);
        }
    }
    // MTP2-P0 begin: more columns than the stock kernels take (a tensor an N-column family declined: shape, view, missing
    // SoA copy during capture, per-shape cap): nothing is launched, ggml_cuda_mul_mat_vec_q_nc reports the decline and its
    // caller (ggml_cuda_mul_mat) takes the route these batches took before (MMQ, or cuBLAS for types MMQ lacks)
    if (!ids && !fusion && ne11 > MMVQ_MAX_BATCH_SIZE) {
        GGML_ASSERT(mmvq_nc_allow_decline && "N-column MUL_MAT above MMVQ_MAX_BATCH_SIZE outside ggml_cuda_mul_mat_vec_q_nc");
        mmvq_nc_declined = true;
        return;
    }
    // MTP2-P0 end
#endif
    // a mixed-type gate must never reach the kernels below: they decode the gate with src0's type and row stride
    GGML_ASSERT(!fusion || !fusion->gate || fusion->gate->type == src0->type);

#if defined(GGML_USE_HIP)
    if (!ids && ne11 == 1 && ne12 == 1 && ne13 == 1 &&
            ggml_cuda_mmvq_lowbit(ctx, src0, src1_q8_1_ptr, dst, fusion)) {
        return;
    }
    if (!ids && ne11 == 1 && ne12 == 1 && ne13 == 1 &&
            ggml_cuda_mmvq_iq4k(ctx, src0, src1_q8_1_ptr, dst, fusion)) {
        return;
    }
    if (!ids && ne11 == 1 && ne12 == 1 && ne13 == 1 && use_q8_cache &&
            ggml_backend_buffer_get_usage(src0->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
            ggml_cuda_mmvq_fused_preq(ctx, src0, src1_q8_1_ptr, src1_sums, dst, fusion)) {
        return;
    }
#endif

#if defined(GGML_USE_HIP)
    const bool plain_fusion = fusion && !fusion->gate_bias && !fusion->shared_up &&
            !fusion->x_scale && !fusion->gate_scale;
    const bool swiglu = plain_fusion && fusion->gate && !fusion->x_bias &&
            fusion->glu_op == GGML_GLU_OP_SWIGLU;
    const bool add_residual = plain_fusion && !fusion->gate && fusion->x_bias &&
            fusion->x_bias->type == GGML_TYPE_F32 &&
            ggml_are_same_shape(fusion->x_bias, dst) && ggml_is_contiguous(fusion->x_bias);
    if (!ids && (src0->type == GGML_TYPE_IQ3_S || src0->type == GGML_TYPE_IQ3_XXS) &&
            ne11 == 1 && ne12 == 1 && ne13 == 1 &&
            ggml_is_matrix(src0) && ggml_is_contiguous(src0) && ggml_is_contiguous(dst) &&
            (!fusion || swiglu || add_residual)) {
        const int device = ggml_cuda_get_device();
        const auto * up = static_cast<const uint32_t *>(codex_iq3_get_packed(src0,device,stream));
        const auto * gate = swiglu ? static_cast<const uint32_t *>(codex_iq3_get_packed(fusion->gate,device,stream)) : nullptr;
        if (up && (!swiglu || gate)) {
            const size_t count = ggml_nelements(src0)/256;
            const int groups = ne00/32;
            const auto * q8 = src1_q8_1_ptr;
            const auto * residual = add_residual ? static_cast<const float *>(fusion->x_bias->data) : nullptr;
            if (src0->type == GGML_TYPE_IQ3_XXS) {
                codex_iq3_launch_packed<true>(ctx,up,gate,q8,dst_d,residual,count,groups,ne01,stream);
            } else {
                codex_iq3_launch_packed<false>(ctx,up,gate,q8,dst_d,residual,count,groups,ne01,stream,src1_sums);
            }
            CUDA_CHECK(hipGetLastError());
            return;
        }
    }
#endif

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1_ptr, ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream);
}

bool ggml_cuda_mul_mat_vec_q_multi(ggml_backend_cuda_context & ctx,
        const ggml_tensor * const * src0, const ggml_tensor * src1, ggml_tensor * const * dst, const int n) {
#if defined(GGML_USE_HIP)
    if (src1->ne[1] >= 2) {   // MMVQ-NC A: N-column multi-job launch
        const block_q8_1 * q8 = nullptr;
        const int * sums = nullptr;
        return ggml_cuda_mmvq_iq3_multi_nc_ok(src0, n, (int) src1->ne[1]) && ggml_cuda_mmvq_q8_cached_nc(ctx, src1, &q8, &sums) &&
            ggml_cuda_mmvq_iq3_multi_nc(ctx, src0, dst, n, q8, (int) src1->ne[1]);
    }
    if (src1->type != GGML_TYPE_F32 || !ggml_is_contiguous(src1) || src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            !ggml_cuda_mmvq_iq3_multi_ok(src0, n)) {
        return false;
    }
    const int64_t ne10        = src1->ne[0];
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const block_q8_1 * q8 = nullptr;
    const int * sums = nullptr;
    if (!mmvq_q8_cache_get(ctx, src1, ne10, ne10_padded, &q8, &sums)) {
        return false;
    }
    return ggml_cuda_mmvq_iq3_multi(ctx, src0, dst, n, q8);
#else
    GGML_UNUSED_VARS(ctx, src0, src1, dst, n);
    return false;
#endif
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}

// MTP2-P0: an unfused MUL_MAT with MMVQ_MAX_BATCH_SIZE < ne11 <= 16 columns routed here by ggml_cuda_mmvq_nc_takes; false if
// no N-column family took the tensor (nothing was written to dst), then the caller runs the previous route
bool ggml_cuda_mul_mat_vec_q_nc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    mmvq_nc_allow_decline = true;
    mmvq_nc_declined      = false;
    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst, nullptr);
    mmvq_nc_allow_decline = false;
    return !mmvq_nc_declined;
}
