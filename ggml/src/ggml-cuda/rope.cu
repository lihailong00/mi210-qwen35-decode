#include "convert.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "rope.cuh"

#include <cstring>
#include <map>
#include <mutex>
#include <utility>

struct rope_corr_dims {
    float v[2];
};


struct mrope_sections {
    int v[4];
};

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template<bool forward>
static __device__ void rope_yarn(
        const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_norm(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int            n_offs,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride,
                                 const bool           inplace) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 + i1 * s01 + i2 * s02 + i3 * s03;
    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    const auto & store_coaelsced = [&](float x0, float x1) {
        if constexpr (std::is_same_v<float, D>) {
            float2 v = make_float2(x0, x1);
            ggml_cuda_memcpy_1<8>(dst + idst, &v);
        } else if constexpr (std::is_same_v<half, D>) {
            half2 v = make_half2(x0, x1);
            ggml_cuda_memcpy_1<4>(dst + idst, &v);
        }
    };
    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        store_coaelsced(x[ix + 0], x[ix + 1]);
        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const float theta_base = pos[i2]*powf(theta_scale, iw/2.0f);

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta, x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_neox(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int            n_offs,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride,
                                 const bool           inplace) {
    ggml_cuda_pdl_lc();
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;
    ggml_cuda_pdl_sync();

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        dst[idst + i0 / 2 + 0] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 1]);

        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const float theta_base = pos[i2]*powf(theta_scale, iw/2.0f);

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    const float x0 = x[ix + n_offs/2 + 0];
    const float x1 = x[ix + n_offs/2 + n_dims/2];

    dst[idst + n_offs/2 + 0]          = ggml_cuda_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs/2 + n_dims / 2] = ggml_cuda_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_multi(const T *            x,
                                  D *                  dst,
                                  const int            ne00,
                                  const int            ne01,
                                  const int            ne02,
                                  const int            s01,
                                  const int            s02,
                                  const int            s03,
                                  const int            s1,
                                  const int            s2,
                                  const int            s3,
                                  const int            n_dims,
                                  const int            n_offs,
                                  const int32_t *      pos,
                                  const float          freq_scale,
                                  const float          ext_factor,
                                  const float          attn_factor,
                                  const rope_corr_dims corr_dims,
                                  const float          theta_scale,
                                  const float *        freq_factors,
                                  const mrope_sections sections,
                                  const bool           is_imrope,
                                  const int64_t *      row_indices,
                                  const int            set_rows_stride,
                                  const bool           inplace) {
    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ggml_cuda_pdl_sync();

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        dst[idst + i0/2 + 0] = ggml_cuda_cast<D>(x[ix + i0/2 + 0]);
        dst[idst + i0/2 + 1] = ggml_cuda_cast<D>(x[ix + i0/2 + 1]);

        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (iw / 2) % sect_dims;

    float theta_base = 0.0;
    if (is_imrope) {
        if (sector % 3 == 1 && sector < 3 * sections.v[1]) {         // h
            theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, iw / 2.0f);
        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {  // w
            theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, iw / 2.0f);
        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {  // t
            theta_base = pos[i2] * powf(theta_scale, iw / 2.0f);
        } else {
            theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, iw / 2.0f);
        }
    } else {
        if (sector < sections.v[0]) {
            theta_base = pos[i2] * powf(theta_scale, iw / 2.0f);
        } else if (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, iw / 2.0f);
        } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, iw / 2.0f);
        } else if (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, iw / 2.0f);
        }
    }

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    const float x0 = x[ix + n_offs/2 + 0];
    const float x1 = x[ix + n_offs/2 + n_dims/2];

    dst[idst + n_offs/2 + 0]        = ggml_cuda_cast<D>(x0*cos_theta - x1*sin_theta);
    dst[idst + n_offs/2 + n_dims/2] = ggml_cuda_cast<D>(x0*sin_theta + x1*cos_theta);
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_vision(const T *            x,
                                   T *                  dst,
                                   const int            ne00,
                                   const int            ne01,
                                   const int            ne02,
                                   const int            s01,
                                   const int            s02,
                                   const int            s03,
                                   const int            s1,
                                   const int            s2,
                                   const int            s3,
                                   const int            n_dims,
                                   const int32_t *      pos,
                                   const float          freq_scale,
                                   const float          ext_factor,
                                   const float          attn_factor,
                                   const rope_corr_dims corr_dims,
                                   const float          theta_scale,
                                   const float *        freq_factors,
                                   const mrope_sections sections) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ggml_cuda_pdl_sync();
    const int sect_dims = sections.v[0] + sections.v[1];
    const int sec_w     = sections.v[1] + sections.v[0];
    const int sector    = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        const int p = sector;
        theta_base  = pos[i2] * powf(theta_scale, p);
    } else if (sector >= sections.v[0] && sector < sec_w) {
        const int p = sector - sections.v[0];
        theta_base  = pos[i2 + ne02] * powf(theta_scale, p);
    }

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims];

    dst[idst + 0]      = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims] = x0*sin_theta + x1*cos_theta;
}

template <bool forward, typename T, typename D>
static void rope_norm_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            n_offs,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           const bool           inplace,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_norm<forward, false><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    } else {
        rope_norm<forward, true><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    }
}

template <bool forward, typename T, typename D>
static void rope_neox_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            n_offs,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           const bool           inplace,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};

    if (freq_factors == nullptr) {
        ggml_cuda_kernel_launch(rope_neox<forward, false, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    } else {
        ggml_cuda_kernel_launch(rope_neox<forward, true, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    }
}

template <bool forward, typename T, typename D>
static void rope_multi_cuda(const T *            x,
                            D *                  dst,
                            const int            ne00,
                            const int            ne01,
                            const int            ne02,
                            const int            s01,
                            const int            s02,
                            const int            s03,
                            const int            s1,
                            const int            s2,
                            const int            s3,
                            const int            n_dims,
                            const int            n_offs,
                            const int            nr,
                            const int32_t *      pos,
                            const float          freq_scale,
                            const float          freq_base,
                            const float          ext_factor,
                            const float          attn_factor,
                            const rope_corr_dims corr_dims,
                            const float *        freq_factors,
                            const mrope_sections sections,
                            const bool           is_imrope,
                            const int64_t *      row_indices,
                            const int            set_rows_stride,
                            const bool           inplace,
                            cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(rope_multi<forward, false, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope, row_indices, set_rows_stride, inplace);
    } else {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(rope_multi<forward, true, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope, row_indices, set_rows_stride, inplace);
    }
}

template <bool forward, typename T>
static void rope_vision_cuda(const T *            x,
                             T *                  dst,
                             const int            ne00,
                             const int            ne01,
                             const int            ne02,
                             const int            s01,
                             const int            s02,
                             const int            s03,
                             const int            s1,
                             const int            s2,
                             const int            s3,
                             const int            n_dims,
                             const int            nr,
                             const int32_t *      pos,
                             const float          freq_scale,
                             const float          freq_base,
                             const float          ext_factor,
                             const float          attn_factor,
                             const rope_corr_dims corr_dims,
                             const float *        freq_factors,
                             const mrope_sections sections,
                             cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);
    // break down (head_dim, heads, seq) into (CUDA_ROPE_BLOCK_SIZE, x, heads * seq)
    // where x ~= ceil(head_dim / CUDA_ROPE_BLOCK_SIZE);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_vision<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    } else {
        rope_vision<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    }
}

template <bool forward>
void ggml_cuda_op_rope_impl(ggml_backend_cuda_context & ctx,
                            ggml_tensor *               dst,
                            const ggml_tensor *         set_rows = nullptr) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;

    void *          dst_d           = dst->data;
    const int64_t * row_indices     = nullptr;
    ggml_type       dst_type        = dst->type;
    int             set_rows_stride = 0;

    if (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d           = set_rows->data;
        row_indices     = (const int64_t *) set_rows->src[1]->data;
        dst_type        = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    // When not fused, src0 and dst types must match
    // When fused (ROPE+VIEW+SET_ROWS), src0 may be F32 and dst may be F16
    GGML_ASSERT(src0->type == dst->type || (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16));

    const int64_t ne00 = src0->ne[0]; // head dims
    const int64_t ne01 = src0->ne[1]; // num heads
    const int64_t ne02 = src0->ne[2]; // num heads
    const int64_t nr = ggml_nrows(src0);

    const size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    const size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    const size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    const size_t s1 = dst->nb[1] / ggml_type_size(dst->type);
    const size_t s2 = dst->nb[2] / ggml_type_size(dst->type);
    const size_t s3 = dst->nb[3] / ggml_type_size(dst->type);

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    const int n_offs     = ((int32_t *) dst->op_params)[15];
    mrope_sections sections;

    // when dst aliases src0, the channels outside the rotated window already hold the correct data
    const bool inplace = dst_d == src0->data;

    // RoPE alteration for extended context
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections.v,  (int32_t *) dst->op_params + 11, sizeof(int)*4);

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 || sections.v[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne00/2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    const int32_t * pos = (const int32_t *) src1_d;

    const float * freq_factors = nullptr;
    if (src2 != nullptr) {
        freq_factors = (const float *) src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    // compute
    if (is_neox) {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_mrope && !is_vision) {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_multi_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                   s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                   ext_factor, attn_factor, corr_dims, freq_factors, sections, is_imrope,
                                                   row_indices, set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_multi_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, sections, is_imrope,
                                                  row_indices, set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_multi_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, sections, is_imrope,
                                                 row_indices, set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_vision) {
        if (src0->type == GGML_TYPE_F32) {
            rope_vision_cuda<forward>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_vision_cuda<forward>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    }
}

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<true>(ctx, dst);
}

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<false>(ctx, dst);
}

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rope, ggml_tensor * set_rows) {
    ggml_cuda_op_rope_impl<true>(ctx, rope, set_rows);
}

// position index (0..3) used by multi-section rope (mrope / imrope) for rotary pair i (= i0/2),
// identical to the section selection in rope_multi
static __device__ __forceinline__ int rope_multi_pos_idx(const int i, const mrope_sections sections, const bool is_imrope) {
    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w     = sections.v[1] + sections.v[0];
    const int sector    = i % sect_dims;
    if (is_imrope) {
        if (sector % 3 == 1 && sector < 3 * sections.v[1]) {         // h
            return 1;
        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {  // w
            return 2;
        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {  // t
            return 0;
        }
        return 3;
    }
    if (sector < sections.v[0]) {
        return 0;
    } else if (sector < sec_w) {
        return 1;
    } else if (sector < sec_w + sections.v[2]) {
        return 2;
    }
    return 3;
}

// fused RMS_NORM + MUL + ROPE (+ VIEW + SET_ROWS)
// one block per row: block_reduce gives the norm scale, then each thread applies mul and rope to the elements it owns
// is_multi: mrope/imrope (neox-style pairs, per-section positions pos[channel + nchannels*k])
template <int block_size, bool has_ff, typename D>
static __global__ void rms_norm_mul_rope_f32(
        const float * x, D * dst, const int ncols, const int nchannels, const int nsamples,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const float eps,
        const float * mul,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed,
        const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const int n_dims, const int32_t * pos,
        const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale,
        const float * freq_factors,
        const int64_t * row_indices, const int set_rows_stride,
        const bool is_neox, const bool is_multi, const bool is_imrope, const mrope_sections sections) {
    ggml_cuda_pdl_lc();
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const float * xc = x + sample*s03 + channel*s02 + row*s01;

            const uint32_t mul_row     = fastmodulo(row,     mul_nrows_packed);
            const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
            const uint32_t mul_sample  = fastmodulo(sample,  mul_nsamples_packed);
            const float * mulc = mul + mul_sample*mul_s03 + mul_channel*mul_s02 + mul_row*mul_s01;

            float tmp = 0.0f;

            for (int col = tid; col < ncols; col += block_size) {
                const float xi = xc[col];
                tmp += xi * xi;
            }

            tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

            const float scale = rsqrtf(tmp/ncols + eps);

            int64_t idst = sample*s3 + channel*s2 + row*s1;
            if (set_rows_stride != 0) {
                idst = row*s1 + row_indices[channel]*set_rows_stride;
            }
            D * dstc = dst + idst;

            for (int i0 = 2*tid; i0 < ncols; i0 += 2*block_size) {
                int ix0;
                int ix1;
                if ((is_neox || is_multi) && i0 < n_dims) {
                    ix0 = i0/2;
                    ix1 = i0/2 + n_dims/2;
                } else {
                    ix0 = i0 + 0;
                    ix1 = i0 + 1;
                }

                const float x0 = scale * xc[ix0] * mulc[fastmodulo(ix0, mul_ncols_packed)];
                const float x1 = scale * xc[ix1] * mulc[fastmodulo(ix1, mul_ncols_packed)];

                if (i0 >= n_dims) {
                    dstc[ix0] = ggml_cuda_cast<D>(x0);
                    dstc[ix1] = ggml_cuda_cast<D>(x1);
                    continue;
                }

                const int   p_idx       = is_multi ? rope_multi_pos_idx(i0/2, sections, is_imrope) : 0;
                const float theta_base  = pos[channel + nchannels*p_idx]*powf(theta_scale, i0/2.0f);
                const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

                float cos_theta;
                float sin_theta;
                rope_yarn<true>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

                dstc[ix0] = ggml_cuda_cast<D>(x0*cos_theta - x1*sin_theta);
                dstc[ix1] = ggml_cuda_cast<D>(x0*sin_theta + x1*cos_theta);
            }

            if constexpr (block_size > WARP_SIZE) {
                // sync is needed as we reuse s_sum across block_reduce invocations, see #26385
                __syncthreads();
            }
        }
    }
}

template <typename D>
static void rms_norm_mul_rope_cuda(
        const float * x, D * dst,
        const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const float eps,
        const float * mul,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint32_t mul_ncols, const uint32_t mul_nrows,
        const uint32_t mul_nchannels, const uint32_t mul_nsamples,
        const int n_dims, const int32_t * pos,
        const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims,
        const float * freq_factors,
        const int64_t * row_indices, const int set_rows_stride,
        const bool is_neox, const bool is_multi, const bool is_imrope, const mrope_sections sections, cudaStream_t stream) {
    GGML_ASSERT(ncols % 2 == 0);

    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32*sizeof(float), stream};
        if (freq_factors == nullptr) {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<256, false, D>, launch_params,
                x, dst, ncols, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox, is_multi, is_imrope, sections);
        } else {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<256, true, D>, launch_params,
                x, dst, ncols, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox, is_multi, is_imrope, sections);
        }
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32*sizeof(float), stream};
        if (freq_factors == nullptr) {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<1024, false, D>, launch_params,
                x, dst, ncols, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox, is_multi, is_imrope, sections);
        } else {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<1024, true, D>, launch_params,
                x, dst, ncols, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox, is_multi, is_imrope, sections);
        }
    }
}

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx,
        ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows) {
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * mul_src = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_src->type == GGML_TYPE_F32);
    GGML_ASSERT(rope->type == GGML_TYPE_F32);

    void *          dst_d           = rope->data;
    ggml_type       dst_type        = rope->type;
    const int64_t * row_indices     = nullptr;
    int             set_rows_stride = 0;

    if (set_rows != nullptr) {
        dst_d           = set_rows->data;
        dst_type        = set_rows->type;
        row_indices     = (const int64_t *) set_rows->src[1]->data;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }

    const int n_dims     = ((const int32_t *) rope->op_params)[1];
    const int mode       = ((const int32_t *) rope->op_params)[2];
    const int n_ctx_orig = ((const int32_t *) rope->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (const int32_t *) rope->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) rope->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) rope->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) rope->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) rope->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) rope->op_params + 10, sizeof(float));

    const bool is_neox   = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_multi  = (mode & GGML_ROPE_TYPE_MROPE) && mode != GGML_ROPE_TYPE_VISION;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    GGML_ASSERT(mode != GGML_ROPE_TYPE_VISION);

    mrope_sections sections;
    memcpy(&sections.v, (const int32_t *) rope->op_params + 11, sizeof(int)*4);

    const int32_t * pos = (const int32_t *) rope->src[1]->data;

    const float * freq_factors = rope->src[2] != nullptr ? (const float *) rope->src[2]->data : nullptr;

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    const size_t ts0 = ggml_type_size(x->type);
    GGML_ASSERT(x->nb[0] == ts0);
    const int64_t s01 = x->nb[1] / ts0;
    const int64_t s02 = x->nb[2] / ts0;
    const int64_t s03 = x->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const size_t ts_dst = ggml_type_size(rope->type);
    const int64_t s1 = rope->nb[1] / ts_dst;
    const int64_t s2 = rope->nb[2] / ts_dst;
    const int64_t s3 = rope->nb[3] / ts_dst;

    cudaStream_t stream = ctx.stream();

    if (dst_type == GGML_TYPE_F32) {
        rms_norm_mul_rope_cuda((const float *) x->data, (float *) dst_d,
            x->ne[0], x->ne[1], x->ne[2], x->ne[3], s01, s02, s03, s1, s2, s3, eps,
            (const float *) mul_src->data, mul_s01, mul_s02, mul_s03,
            mul_src->ne[0], mul_src->ne[1], mul_src->ne[2], mul_src->ne[3],
            n_dims, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
            freq_factors, row_indices, set_rows_stride, is_neox, is_multi, is_imrope, sections, stream);
    } else if (dst_type == GGML_TYPE_F16) {
        rms_norm_mul_rope_cuda((const float *) x->data, (half *) dst_d,
            x->ne[0], x->ne[1], x->ne[2], x->ne[3], s01, s02, s03, s1, s2, s3, eps,
            (const float *) mul_src->data, mul_s01, mul_s02, mul_s03,
            mul_src->ne[0], mul_src->ne[1], mul_src->ne[2], mul_src->ne[3],
            n_dims, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
            freq_factors, row_indices, set_rows_stride, is_neox, is_multi, is_imrope, sections, stream);
    } else {
        GGML_ABORT("fatal error");
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// fused full-attention prep (qwen3.5 / qwen3-next style), one launch for the whole chain after the q/k/v projections:
//   q_out           = rope(rms_norm(q_view) * wq)          (q_view: q half of the joint q+gate projection, strided)
//   gate_out        = cont(gate_view)                      (gate half of the joint projection)
//   k_cache[k_idx]  = rope(rms_norm(k) * wk)               (f16 or f32 cache rows, all kv heads of a token per row)
//   v_cache[v_idx]  = v
// grid = (2*n_head + 2*n_head_kv, n_tokens), one wave (64 threads) per head row: blocks [0, nh) q, [nh, nh+nkv) k,
// [nh+nkv, nh+2nkv) v, [nh+2nkv, 2nh+2nkv) gate. The rope arithmetic (theta, yarn, pair order) is identical to
// rope_norm / rope_neox / rope_multi, so results only differ from the unfused path by the norm reduction order.
struct ggml_cuda_attn_prep {
    const float * q_src; int64_t q_s1, q_s2;
    const float * g_src; int64_t g_s1, g_s2;
    const float * k_src; int64_t k_s1, k_s2;
    const float * v_src; int64_t v_s1;
    const float * wq;
    const float * wk;
    float * q_dst; int64_t qd_s1, qd_s2;
    float * g_dst; int64_t gd_s1;
    ggml_type k_type; void * k_dst; int64_t kd_s1; const int64_t * k_idx;
    ggml_type v_type; void * v_dst; int64_t vd_s1; const int64_t * v_idx;
    int n_head, n_head_kv, n_tokens;
    float eps_q, eps_k;
    int n_dims;
    const int32_t * pos;
    int rope_kind; // 0 norm, 1 neox, 2 mrope/imrope
    bool is_imrope;
    mrope_sections sections;
    float freq_scale, ext_factor, attn_factor;
    rope_corr_dims corr_dims;
    float theta_scale;
    const float * freq_factors;
    const float * pow_table; // theta_scale^i, i < n_dims/2 (nullptr: the kernel calls powf)
    int2 * cs_table;         // N1: tagged cos/sin of the last position per rotary pair (nullptr: not used)
};

// Table of theta_scale^i for attn_prep_f32, one immutable device buffer per (device, theta_scale, n_pairs). It is
// filled by attn_prep_pow_init (the same device powf expression as the in-kernel fallback, so the values are
// bit-identical) outside graph capture, and the host waits for the init kernel before publishing the buffer, so every
// attn_prep launch that receives it (any stream, captured or not) only reads finished values. A launch that would
// need a table which does not exist yet while its stream is capturing gets none and calls powf itself (as the first
// launch of the old in-kernel cache did): no per-token launch, no writer racing with readers.
static __global__ void attn_prep_pow_init(float * __restrict__ table, const float theta_scale, const int n_pairs) {
    const int i = threadIdx.x;
    if (i < n_pairs) {
        const int   iw = 2*i;
        const float pw = powf(theta_scale, iw/2.0f);
        table[i] = pw;
    } else {
        table[i] = 0.0f;
    }
}

// half(fma(a, b, c)) rounded once, like the v_fma_mixlo_f16 the compiler forms for ggml_cuda_cast<half>(x0*c - x1*s)
// in rope_multi<float, half> on gfx9 (a plain fmaf + conversion rounds twice and differs by 1 ulp now and then)
static __device__ __forceinline__ half attn_prep_fma_f16(const float a, const float b, const float c) {
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__)
    uint32_t r;
    asm("v_fma_mixlo_f16 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
    const uint16_t r16 = (uint16_t) r;
    half h;
    memcpy(&h, &r16, sizeof(h));
    return h;
#else
    return __float2half(fmaf(a, b, c));
#endif
}

template <int EPT>
static __device__ __forceinline__ void attn_prep_load(const float * src, float * x) {
    if constexpr (EPT == 4) {
        const float4 t = *(const float4 *) src;
        x[0] = t.x; x[1] = t.y; x[2] = t.z; x[3] = t.w;
    } else if constexpr (EPT == 2) {
        const float2 t = *(const float2 *) src;
        x[0] = t.x; x[1] = t.y;
    } else {
#pragma unroll
        for (int j = 0; j < EPT; ++j) {
            x[j] = src[j];
        }
    }
}

template <int EPT, typename D>
static __device__ __forceinline__ void attn_prep_store(D * dst, const float * x) {
    if constexpr (std::is_same_v<D, float> && EPT == 4) {
        *(float4 *) dst = make_float4(x[0], x[1], x[2], x[3]);
    } else if constexpr (std::is_same_v<D, float> && EPT == 2) {
        *(float2 *) dst = make_float2(x[0], x[1]);
    } else if constexpr (std::is_same_v<D, half> && EPT == 4) {
        half2 h[2] = { make_half2(x[0], x[1]), make_half2(x[2], x[3]) };
        ggml_cuda_memcpy_1<8>(dst, h);
    } else if constexpr (std::is_same_v<D, half> && EPT == 2) {
        half2 h = make_half2(x[0], x[1]);
        ggml_cuda_memcpy_1<4>(dst, &h);
    } else {
#pragma unroll
        for (int j = 0; j < EPT; ++j) {
            dst[j] = ggml_cuda_cast<D>(x[j]);
        }
    }
}

// N1: cs_table (GGML_CUDA_ATTN_PREP_CS=0 disables) holds, per rotary pair i < 64, the cos and sin of the latest position
// as two 64-bit words {p, bits(cos)}, {p, bits(sin)} (single-copy atomic stores). Every launch loads the words of its
// pairs together with its other inputs; a word whose tag equals the position the pair needs is the value rope_yarn
// returns for it (the table belongs to one set of rope parameters, and the computation is deterministic), so it is
// used directly; otherwise the lane computes cos/sin as before, and block (0, 0) stores the new words. In the decode
// graph the first full-attention layer of a token computes and the other layers reuse: bit-identical results.
// A word is one aligned 64-bit access (tag in the low half, value bits in the high half), so a reader never pairs a new
// tag with an old value. The table key (attn_prep_cs_table) covers the rope parameters, the head size and the cache
// types (kernel instantiation); launches with freq_factors do not use a table (the key cannot cover their contents).
static __device__ __forceinline__ int2 attn_prep_cs_load(const int2 * w) {
#if defined(GGML_USE_HIP)
    const unsigned long long u = __hip_atomic_load((const unsigned long long *) w, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    const unsigned long long u = *(const volatile unsigned long long *) w;
#endif // defined(GGML_USE_HIP)
    return make_int2((int) (uint32_t) u, (int) (uint32_t) (u >> 32));
}
static __device__ __forceinline__ void attn_prep_cs_store(int2 * w, const int tag, const int bits) {
    const unsigned long long u = ((unsigned long long) (uint32_t) bits << 32) | (unsigned long long) (uint32_t) tag;
#if defined(GGML_USE_HIP)
    __hip_atomic_store((unsigned long long *) w, u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    *(volatile unsigned long long *) w = u;
#endif // defined(GGML_USE_HIP)
}
template <int HD, bool has_ff, typename DK, typename DV>
__launch_bounds__(64, 1)
static __global__ void attn_prep_f32(
        const float * __restrict__ q_src, const int64_t q_s1, const int64_t q_s2,
        const float * __restrict__ g_src, const int64_t g_s1, const int64_t g_s2,
        const float * __restrict__ k_src, const int64_t k_s1, const int64_t k_s2,
        const float * __restrict__ v_src, const int64_t v_s1,
        const float * __restrict__ wq, const float * __restrict__ wk,
        float * __restrict__ q_dst, const int64_t qd_s1, const int64_t qd_s2,
        float * __restrict__ g_dst, const int64_t gd_s1,
        DK * __restrict__ k_dst, const int64_t kd_s1, const int64_t * __restrict__ k_idx,
        DV * __restrict__ v_dst, const int64_t vd_s1, const int64_t * __restrict__ v_idx,
        const int n_head, const int n_head_kv, const int n_tokens,
        const float eps_q, const float eps_k,
        const int n_dims, const int32_t * __restrict__ pos, const int rope_kind, const bool is_imrope,
        const mrope_sections sections,
        const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale,
        const float * __restrict__ freq_factors, const float * __restrict__ pow_table, int2 * cs_table) {
    constexpr int EPT = HD / 64;
    static_assert(HD % 64 == 0, "bad head size");

    const int tid = threadIdx.x;
    const int t   = blockIdx.y;
    const int b   = blockIdx.x;
    const int e0  = tid * EPT;

    float x[EPT];

    if (b < n_head + n_head_kv) {
        // q or k head: rms_norm * w, then rope
        const bool    is_q = b < n_head;
        const int     h    = is_q ? b : b - n_head;
        const float * src  = is_q ? q_src + t*q_s2 + h*q_s1 : k_src + t*k_s2 + h*k_s1;
        const float * w    = is_q ? wq : wk;
        const int64_t krow = is_q ? 0 : k_idx[t]; // uniform, issued early
        // early load of the tabulated theta_scale^i of this lane (first pass); the table is complete before launch
        const float pw_pref = pow_table ? pow_table[tid] : 0.0f;
        int2 cs_c = make_int2(INT_MIN, 0);
        int2 cs_s = make_int2(INT_MIN, 0);
        if (cs_table != nullptr && tid < n_dims / 2) {
            cs_c = attn_prep_cs_load(cs_table + 2*tid);
            cs_s = attn_prep_cs_load(cs_table + 2*tid + 1);
        }

        float wv[EPT];
        attn_prep_load<EPT>(src + e0, x);
        attn_prep_load<EPT>(w + e0, wv);

        // cos/sin of every rotary pair, one lane per pair (instead of one per element), computed while the row loads
        // are in flight: this kernel is latency bound, and the transcendental chain dominated it. The arithmetic
        // (powf/cosf/sinf, rope_yarn, pair order) is the one of rope_multi/rope_neox/rope_norm, bit for bit.
        const int n_pairs = n_dims / 2;
        int p4[4];
        p4[0] = pos[t];
        if (rope_kind == 2) {
            p4[1] = pos[t + n_tokens];
            p4[2] = pos[t + 2*n_tokens];
            p4[3] = pos[t + 3*n_tokens];
        }
        // theta_scale^i depends only on the model: it comes from a device-side table computed with the same powf
        // (bit-identical, see attn_prep_pow_init); powf was ~1/3 of this latency-bound kernel
        const bool pow_cached = pow_table != nullptr;
        __shared__ float s_cos[HD/2];
        __shared__ float s_sin[HD/2];
        for (int i = tid; i < n_pairs; i += 64) {
            int p;
            if (rope_kind == 2) {
                const int p_idx = rope_multi_pos_idx(i, sections, is_imrope);
                p = p_idx == 0 ? p4[0] : (p_idx == 1 ? p4[1] : (p_idx == 2 ? p4[2] : p4[3]));
            } else {
                p = p4[0];
            }
            if (i == tid && cs_c.x == p && cs_s.x == p) {
                s_cos[i] = __int_as_float(cs_c.y);
                s_sin[i] = __int_as_float(cs_s.y);
                continue;
            }
            const int   iw          = 2*i;
            const float pw          = pow_cached ? (i == tid ? pw_pref : pow_table[i]) : powf(theta_scale, iw/2.0f);
            const float theta_base  = p*pw;
            const float freq_factor = has_ff ? freq_factors[i] : 1.0f;
            float cos_theta;
            float sin_theta;
            rope_yarn<true>(theta_base/freq_factor, freq_scale, corr_dims, 2*i, ext_factor, attn_factor, cos_theta, sin_theta);
            s_cos[i] = cos_theta;
            s_sin[i] = sin_theta;
            if (cs_table != nullptr && i == tid && b == 0 && t == 0) {
                attn_prep_cs_store(cs_table + 2*i,     p, __float_as_int(cos_theta));
                attn_prep_cs_store(cs_table + 2*i + 1, p, __float_as_int(sin_theta));
            }
        }

        // sum of squares in the order of rms_norm_f32<128, 1, true> for 256 columns (float4 per thread, 32-lane
        // trees, then the two halves), so that the fused result matches the unfused rms_norm + mul bit for bit
        float sum = 0.0f;
        if constexpr (EPT == 4) {
            sum += x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3];
        } else {
#pragma unroll
            for (int j = 0; j < EPT; ++j) {
                sum += x[j] * x[j];
            }
        }
        sum = warp_reduce_sum<32>(sum);
        if constexpr (ggml_cuda_get_physical_warp_size() == 64) {
            sum += __shfl_xor_sync(0xffffffff, sum, 32, 64);
        } else {
            __shared__ float s_red[2];
            if (tid % 32 == 0) {
                s_red[tid / 32] = sum;
            }
            __syncthreads();
            sum = s_red[0] + s_red[1];
        }
        const float scale = rsqrtf(sum / HD + (is_q ? eps_q : eps_k));

        __shared__ float ys[HD];
#pragma unroll
        for (int j = 0; j < EPT; ++j) {
            x[j] = scale * x[j] * wv[j];
            ys[e0 + j] = x[j];
        }
        __syncthreads();

        // rotated element e of pair (ia, ib): fma(P, cos, M*sin) with (P, M) = (x0, -x1) for ia and (x1, x0) for ib.
        // This is the exact contraction the compiler emits for rope_multi/rope_neox (x0*cos - x1*sin ->
        // fma(x0, cos, -(x1*sin)), x0*sin + x1*cos -> fma(x1, cos, x0*sin)); for an f16 destination the fma is
        // converted directly so that it becomes the same single-rounding v_fma_mixlo_f16 as in rope_multi<float, half>.
        float P[EPT];
        float M[EPT];
        float C[EPT];
        float S[EPT];
#pragma unroll
        for (int j = 0; j < EPT; ++j) {
            const int e = e0 + j;
            int i, ia, ib; // rotary pair index, element indices of the pair (first, second)
            if (rope_kind == 0) {
                i  = e / 2;
                ia = 2*i;
                ib = ia + 1;
            } else {
                const int half_dims = n_dims / 2;
                i  = e < half_dims ? e : e - half_dims;
                ia = i;
                ib = i + half_dims;
            }
            const bool rot = e < n_dims;
            i  = rot ? i  : 0;
            ia = rot ? ia : 0;
            ib = rot ? ib : 0;
            const float x0 = ys[ia];
            const float x1 = ys[ib];
            P[j] = e == ia ? x0 : x1;
            M[j] = e == ia ? -x1 : x0;
            C[j] = s_cos[i];
            S[j] = s_sin[i];
        }

        if (is_q) {
            float out[EPT];
#pragma unroll
            for (int j = 0; j < EPT; ++j) {
                out[j] = e0 + j < n_dims ? fmaf(P[j], C[j], M[j]*S[j]) : x[j];
            }
            attn_prep_store<EPT>(q_dst + t*qd_s2 + h*qd_s1 + e0, out);
        } else if constexpr (std::is_same_v<DK, half>) {
            half out[EPT];
#pragma unroll
            for (int j = 0; j < EPT; ++j) {
                out[j] = e0 + j < n_dims ? attn_prep_fma_f16(P[j], C[j], M[j]*S[j]) : __float2half(x[j]);
            }
            ggml_cuda_memcpy_1<EPT*sizeof(half)>(k_dst + krow*kd_s1 + h*HD + e0, out);
        } else {
            float out[EPT];
#pragma unroll
            for (int j = 0; j < EPT; ++j) {
                out[j] = e0 + j < n_dims ? fmaf(P[j], C[j], M[j]*S[j]) : x[j];
            }
            attn_prep_store<EPT>(k_dst + krow*kd_s1 + h*HD + e0, out);
        }
    } else if (b < n_head + 2*n_head_kv) {
        // v head: copy (convert) into the cache row
        const int h = b - n_head - n_head_kv;
        const int64_t vrow = v_idx[t];
        attn_prep_load<EPT>(v_src + t*v_s1 + h*HD + e0, x);
        attn_prep_store<EPT>(v_dst + vrow*vd_s1 + h*HD + e0, x);
    } else {
        // gate head: make contiguous
        const int h = b - n_head - 2*n_head_kv;
        attn_prep_load<EPT>(g_src + t*g_s2 + h*g_s1 + e0, x);
        attn_prep_store<EPT>(g_dst + t*gd_s1 + h*HD + e0, x);
    }
}

template <int HD, typename DK, typename DV>
static void attn_prep_launch(const ggml_cuda_attn_prep & p, cudaStream_t stream) {
    const dim3 block_nums(2*p.n_head + 2*p.n_head_kv, p.n_tokens, 1);
    const dim3 block_dims(64, 1, 1);
    if (p.freq_factors == nullptr) {
        attn_prep_f32<HD, false, DK, DV><<<block_nums, block_dims, 0, stream>>>(
            p.q_src, p.q_s1, p.q_s2, p.g_src, p.g_s1, p.g_s2, p.k_src, p.k_s1, p.k_s2, p.v_src, p.v_s1,
            p.wq, p.wk, p.q_dst, p.qd_s1, p.qd_s2, p.g_dst, p.gd_s1,
            (DK *) p.k_dst, p.kd_s1, p.k_idx, (DV *) p.v_dst, p.vd_s1, p.v_idx,
            p.n_head, p.n_head_kv, p.n_tokens, p.eps_q, p.eps_k,
            p.n_dims, p.pos, p.rope_kind, p.is_imrope, p.sections,
            p.freq_scale, p.ext_factor, p.attn_factor, p.corr_dims, p.theta_scale, p.freq_factors, p.pow_table, p.cs_table);
    } else {
        attn_prep_f32<HD, true, DK, DV><<<block_nums, block_dims, 0, stream>>>(
            p.q_src, p.q_s1, p.q_s2, p.g_src, p.g_s1, p.g_s2, p.k_src, p.k_s1, p.k_s2, p.v_src, p.v_s1,
            p.wq, p.wk, p.q_dst, p.qd_s1, p.qd_s2, p.g_dst, p.gd_s1,
            (DK *) p.k_dst, p.kd_s1, p.k_idx, (DV *) p.v_dst, p.vd_s1, p.v_idx,
            p.n_head, p.n_head_kv, p.n_tokens, p.eps_q, p.eps_k,
            p.n_dims, p.pos, p.rope_kind, p.is_imrope, p.sections,
            p.freq_scale, p.ext_factor, p.attn_factor, p.corr_dims, p.theta_scale, p.freq_factors, p.pow_table, p.cs_table);
    }
}

template <int HD>
static void attn_prep_launch_types(const ggml_cuda_attn_prep & p, cudaStream_t stream) {
    if (p.k_type == GGML_TYPE_F16 && p.v_type == GGML_TYPE_F16) {
        attn_prep_launch<HD, half, half>(p, stream);
    } else if (p.k_type == GGML_TYPE_F32 && p.v_type == GGML_TYPE_F32) {
        attn_prep_launch<HD, float, float>(p, stream);
    } else if (p.k_type == GGML_TYPE_F16 && p.v_type == GGML_TYPE_F32) {
        attn_prep_launch<HD, half, float>(p, stream);
    } else {
        attn_prep_launch<HD, float, half>(p, stream);
    }
}

static std::mutex attn_prep_pow_mutex;
static std::map<std::pair<int, uint64_t>, float *> attn_prep_pow_tables; // (device, key) -> 256 floats, never rewritten

// the immutable table for (device, theta_scale, n_pairs), created on first use outside graph capture; nullptr if it
// does not exist yet and the stream is capturing (or n_pairs is out of range)
static const float * attn_prep_pow_table(const int device, cudaStream_t stream, const float theta_scale, const int n_pairs) {
    if (n_pairs <= 0 || n_pairs > 256) {
        return nullptr;
    }
    uint32_t ts_bits;
    memcpy(&ts_bits, &theta_scale, sizeof(ts_bits));
    const uint64_t key = ((uint64_t) n_pairs << 32) | ts_bits;
    std::lock_guard<std::mutex> lock(attn_prep_pow_mutex);
    const auto it = attn_prep_pow_tables.find({device, key});
    if (it != attn_prep_pow_tables.end()) {
        return it->second;
    }
#if defined(GGML_USE_HIP)
    hipStreamCaptureStatus status;
    CUDA_CHECK(hipStreamIsCapturing(stream, &status));
    if (status != hipStreamCaptureStatusNone) {
        return nullptr;
    }
#else
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    if (status != cudaStreamCaptureStatusNone) {
        return nullptr;
    }
#endif // defined(GGML_USE_HIP)
    float * table = nullptr;
    CUDA_CHECK(cudaMalloc(&table, 256*sizeof(float)));
    attn_prep_pow_init<<<1, 256, 0, stream>>>(table, theta_scale, n_pairs);
    CUDA_CHECK(cudaGetLastError());
    // once per key and device: the table is complete before any launch (on any stream) can be handed the pointer
    CUDA_CHECK(cudaStreamSynchronize(stream));
    attn_prep_pow_tables.emplace(std::make_pair(device, key), table);
    return table;
}

static __global__ void attn_prep_cs_init(int2 * table) {
    table[threadIdx.x] = make_int2(INT_MIN, 0); // no position matches
}

static std::mutex attn_prep_cs_mutex;
static std::map<std::pair<int, std::vector<uint32_t>>, int2 *> attn_prep_cs_tables;

// the cos/sin table of one set of rope parameters on one device, created on first use outside graph capture (nullptr
// while capturing before it exists, or when disabled)
static int2 * attn_prep_cs_table(const int device, cudaStream_t stream, const ggml_cuda_attn_prep & p, const int head_dim,
        const ggml_type k_type, const ggml_type v_type) {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_ATTN_PREP_CS");
        return e == nullptr || e[0] != '0';
    }();
    if (!enabled || p.n_dims / 2 > 64 || p.n_dims <= 0 || p.freq_factors != nullptr) {
        return nullptr;
    }
    std::vector<uint32_t> key;
    auto put = [&](const void * v, size_t n) {
        for (size_t i = 0; i < n; i += 4) {
            uint32_t u = 0;
            memcpy(&u, (const char *) v + i, 4);
            key.push_back(u);
        }
    };
    put(&p.n_dims, 4); put(&p.rope_kind, 4); put(&p.freq_scale, 4); put(&p.ext_factor, 4); put(&p.attn_factor, 4);
    put(&p.theta_scale, 4); put(p.corr_dims.v, 8); put(p.sections.v, 16);
    const int imr = p.is_imrope ? 1 : 0;
    put(&imr, 4);
    const int inst[3] = { head_dim, (int) k_type, (int) v_type };   // the attn_prep_f32 instantiation
    put(inst, 12);
    std::lock_guard<std::mutex> lock(attn_prep_cs_mutex);
    const auto it = attn_prep_cs_tables.find({device, key});
    if (it != attn_prep_cs_tables.end()) {
        return it->second;
    }
#if defined(GGML_USE_HIP)
    hipStreamCaptureStatus status;
    CUDA_CHECK(hipStreamIsCapturing(stream, &status));
    if (status != hipStreamCaptureStatusNone) {
        return nullptr;
    }
#else
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    if (status != cudaStreamCaptureStatusNone) {
        return nullptr;
    }
#endif // defined(GGML_USE_HIP)
    int2 * table = nullptr;
    CUDA_CHECK(cudaMalloc(&table, 128*sizeof(int2)));
    attn_prep_cs_init<<<1, 128, 0, stream>>>(table);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    attn_prep_cs_tables.emplace(std::make_pair(device, key), table);
    return table;
}

bool ggml_cuda_attn_prep_supported(const int64_t head_dim, const ggml_type k_type, const ggml_type v_type) {
    return (head_dim == 128 || head_dim == 256) &&
        (k_type == GGML_TYPE_F16 || k_type == GGML_TYPE_F32) && (v_type == GGML_TYPE_F16 || v_type == GGML_TYPE_F32);
}

// nodes: rms_q, mul_q, rope_q, cont_gate, rms_k, mul_k, rope_k, set_rows_k, set_rows_v (validated by the caller)
void ggml_cuda_op_attn_prep_fused(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes) {
    const ggml_tensor * rms_q  = nodes[0];
    const ggml_tensor * mul_q  = nodes[1];
    const ggml_tensor * rope_q = nodes[2];
    const ggml_tensor * cont   = nodes[3];
    const ggml_tensor * rms_k  = nodes[4];
    const ggml_tensor * mul_k  = nodes[5];
    const ggml_tensor * sr_k   = nodes[7];
    const ggml_tensor * sr_v   = nodes[8];

    const ggml_tensor * q  = rms_q->src[0];
    const ggml_tensor * k  = rms_k->src[0];
    const ggml_tensor * g  = cont->src[0];
    const ggml_tensor * v  = sr_v->src[0];
    const ggml_tensor * wq = mul_q->src[0] == rms_q ? mul_q->src[1] : mul_q->src[0];
    const ggml_tensor * wk = mul_k->src[0] == rms_k ? mul_k->src[1] : mul_k->src[0];

    ggml_cuda_attn_prep p;
    p.q_src = (const float *) q->data; p.q_s1 = q->nb[1]/sizeof(float); p.q_s2 = q->nb[2]/sizeof(float);
    p.g_src = (const float *) g->data; p.g_s1 = g->nb[1]/sizeof(float); p.g_s2 = g->nb[2]/sizeof(float);
    p.k_src = (const float *) k->data; p.k_s1 = k->nb[1]/sizeof(float); p.k_s2 = k->nb[2]/sizeof(float);
    p.v_src = (const float *) v->data; p.v_s1 = v->nb[1]/sizeof(float);
    p.wq = (const float *) wq->data;
    p.wk = (const float *) wk->data;
    p.q_dst = (float *) rope_q->data; p.qd_s1 = rope_q->nb[1]/sizeof(float); p.qd_s2 = rope_q->nb[2]/sizeof(float);
    p.g_dst = (float *) cont->data;   p.gd_s1 = cont->nb[1]/sizeof(float);
    p.k_type = sr_k->type; p.k_dst = sr_k->data; p.kd_s1 = sr_k->nb[1]/ggml_type_size(sr_k->type);
    p.k_idx  = (const int64_t *) sr_k->src[1]->data;
    p.v_type = sr_v->type; p.v_dst = sr_v->data; p.vd_s1 = sr_v->nb[1]/ggml_type_size(sr_v->type);
    p.v_idx  = (const int64_t *) sr_v->src[1]->data;
    p.n_head    = q->ne[1];
    p.n_head_kv = k->ne[1];
    p.n_tokens  = q->ne[2];
    memcpy(&p.eps_q, rms_q->op_params, sizeof(float));
    memcpy(&p.eps_k, rms_k->op_params, sizeof(float));

    const int32_t * op = (const int32_t *) rope_q->op_params;
    p.n_dims = op[1];
    const int mode       = op[2];
    const int n_ctx_orig = op[4];
    float freq_base, beta_fast, beta_slow;
    memcpy(&freq_base,     op +  5, sizeof(float));
    memcpy(&p.freq_scale,  op +  6, sizeof(float));
    memcpy(&p.ext_factor,  op +  7, sizeof(float));
    memcpy(&p.attn_factor, op +  8, sizeof(float));
    memcpy(&beta_fast,     op +  9, sizeof(float));
    memcpy(&beta_slow,     op + 10, sizeof(float));
    memcpy(&p.sections.v,  op + 11, sizeof(int)*4);
    p.rope_kind    = (mode & GGML_ROPE_TYPE_MROPE) ? 2 : ((mode & GGML_ROPE_TYPE_NEOX) ? 1 : 0);
    p.is_imrope    = mode == GGML_ROPE_TYPE_IMROPE;
    p.pos          = (const int32_t *) rope_q->src[1]->data;
    p.freq_factors = rope_q->src[2] ? (const float *) rope_q->src[2]->data : nullptr;
    p.theta_scale  = powf(freq_base, -2.0f/p.n_dims); // same as ggml_cuda_op_rope_impl
    ggml_rope_yarn_corr_dims(p.n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, p.corr_dims.v);

    cudaStream_t stream = ctx.stream();
    p.pow_table = attn_prep_pow_table(ctx.device, stream, p.theta_scale, p.n_dims / 2);
    p.cs_table  = attn_prep_cs_table(ctx.device, stream, p, (int) q->ne[0], p.k_type, p.v_type);
    switch (q->ne[0]) {
        case 128: attn_prep_launch_types<128>(p, stream); break;
        case 256: attn_prep_launch_types<256>(p, stream); break;
        default: GGML_ABORT("unsupported head size");
    }
}
