#include "gdn-decode.cuh"
#include "convert.cuh"
#include "quantize.cuh"
#include "unary.cuh"

#include <vector>

// Fused single-token gated delta net decode for one sequence (qwen3.5 / qwen3-next linear attention).
//
// GGML_OP_GDN_DECODE_PREP (one launch): conv1d + silu over [conv state | qkv] for the q/k channels with their conv
//   state shifted in place, L2 norm of the q/k heads, the alpha/beta projections and the gates
//   exp(softplus(alpha + dt)*a), sigmoid(beta).
// GGML_OP_GDN_DECODE_OUT (one launch, one block per v head): conv1d + silu of the head's v channels (conv state
//   shifted in place), the delta rule on the recurrent state (read and written in place in the cache) and the gated
//   output norm rms_norm(o)*w*silu(z).
//
// The arithmetic (operation order, reduction trees, lane mapping) deliberately mirrors the unfused CUDA chain
//   ssm_conv_f32<silu>, rms_norm_f32<32,1,vec>(+scale), mul_mat_vec_f<bf16,128>, add, softplus*a, sigmoid,
//   gated_delta_net_cuda<128, 4 columns per warp>, rms_norm_f32<32,1,vec>(*w), silu(z)*x
// so that the fused path is bit-identical with it on the same device (verified on gfx90a, tests/test-gdn-decode).
//
// Both kernels first issue all their global loads (the state rows are read speculatively from the row they write,
// which is the row s_copy[0] points to in the steady state) so that the dependent load chains collapse to about one
// memory round trip.

#define GDN_DEC_S         128 // head size (S_k == S_v)
#define GDN_DEC_DCONV     4   // conv kernel size
#define GDN_DEC_MMV_BLOCK 128 // threads per alpha/beta row (mul_mat_vec_f uses 128 threads for these rows)
#define GDN_DEC_OUT_BLOCK 1024 // gdn_decode_out_f32: one block per v head
#define GDN_DEC_SPLIT_BLOCK 256 // gdn_decode_out_split_f32: S/16 blocks per v head (wave64)

// Floating point contraction: the reference kernels end up with a specific mix of fused (fma) and unfused
// (mul + add) operations, chosen by the compiler (e.g. vectorized v_pk_mul_f32 products are never fused).
// To reproduce their results bit for bit independently of how this file gets compiled, every product that
// feeds an addition is made opaque (rounded on its own, never contracted) and fusions are explicit (fmaf).
static __device__ __forceinline__ float gdn_dec_mul(float a, float b) {
    float p = a * b;
#if defined(GGML_USE_HIP)
    asm volatile("" : "+v"(p));
#else
    asm volatile("" : "+f"(p));
#endif
    return p;
}

// sum of squares of a float4: ((x*x + y*y) + z*z) + w*w with rounded products (== rms_norm_f32 vec path)
static __device__ __forceinline__ float gdn_dec_sumsq4(const float4 v) {
    return ((gdn_dec_mul(v.x, v.x) + gdn_dec_mul(v.y, v.y)) + gdn_dec_mul(v.z, v.z)) + gdn_dec_mul(v.w, v.w);
}

// conv1d over [x0, x1, x2, x_new] (== ssm_conv_f32 for one token: products rounded, summed in order, + bias 0)
static __device__ __forceinline__ float gdn_dec_conv_silu(const float * xc, const float * wc) {
    float sumf = 0.0f;
#pragma unroll
    for (int j = 0; j < GDN_DEC_DCONV; j++) {
        sumf = sumf + gdn_dec_mul(xc[j], wc[j]);
    }
    sumf = 0.0f + sumf; // the reference kernel adds the (absent) conv bias
    return ggml_cuda_op_silu_single(sumf);
}

static __device__ __forceinline__ float gdn_dec_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x)); // == op_sigmoid (unary.cu)
}

static __device__ __forceinline__ float gdn_dec_softplus(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x)); // == op_softplus (unary.cu)
}

// output of head h, elements 4*tid .. 4*tid + 3 (tid < 32); with q8: also the q8_1 quantization for the consuming
// single-token GEMV, bit-identical to quantize_q8_1 (8 threads hold one block of 32 values) + the per-block int sums
template <bool q8>
static __device__ __forceinline__ void gdn_dec_store_out(
        float * out, const float4 o, const int h, const int tid, block_q8_1 * y, int * ysum) {
    ((float4 *) (out + h*GDN_DEC_S))[tid] = o;
    if constexpr (q8) {
        float amax = fmaxf(fmaxf(fabsf(o.x), fabsf(o.y)), fmaxf(fabsf(o.z), fabsf(o.w)));
        amax = warp_reduce_max<QK8_1/4>(amax);
        const float  d  = amax / 127.0f;
        const int8_t q0 = amax == 0.0f ? 0 : roundf(o.x / d);
        const int8_t q1 = amax == 0.0f ? 0 : roundf(o.y / d);
        const int8_t q2 = amax == 0.0f ? 0 : roundf(o.z / d);
        const int8_t q3 = amax == 0.0f ? 0 : roundf(o.w / d);
        int sum = (int) q0 + (int) q1 + (int) q2 + (int) q3;
        sum = warp_reduce_sum<QK8_1/4>(sum);
        const int ib = (h*GDN_DEC_S)/QK8_1 + tid/(QK8_1/4);
        const uint32_t packed = (uint32_t) (uint8_t) q0 | ((uint32_t) (uint8_t) q1 << 8) |
                                ((uint32_t) (uint8_t) q2 << 16) | ((uint32_t) (uint8_t) q3 << 24);
        ((uint32_t *) y[ib].qs)[tid % (QK8_1/4)] = packed;
        if (tid % (QK8_1/4) == 0) {
            y[ib].ds  = make_half2(d, d * sum);
            ysum[ib] = sum;
        }
    }
}

template <typename T>
static __device__ __forceinline__ float2 gdn_dec_load_w2(const T * w, int col2);

template <>
__device__ __forceinline__ float2 gdn_dec_load_w2<float>(const float * w, int col2) {
    return ((const float2 *) w)[col2];
}

template <>
__device__ __forceinline__ float2 gdn_dec_load_w2<nv_bfloat16>(const nv_bfloat16 * w, int col2) {
    // same conversion as mul_mat_vec_f (bf16 pair loaded as one int)
    const int tmpx = ((const int *) w)[col2];
    return make_float2(ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]),
                       ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]));
}

// grid: 2*H_k blocks for the q/k heads (conv + L2 norm), H_v blocks for the alpha/beta projections + gates
// out: [2*H_k*S + 2*H_v] = l2norm(q) | l2norm(k) | exp(gate) | sigmoid(beta)
template <typename Tw>
static __global__ void __launch_bounds__(2*GDN_DEC_MMV_BLOCK)
gdn_decode_prep_f32(
        const float   * __restrict__ qkv,
        const float   * __restrict__ x,
        const Tw      * __restrict__ w_alpha,
        const Tw      * __restrict__ w_beta,
        const float   * __restrict__ dt_bias,
        const float   * __restrict__ a,
        const float   * __restrict__ conv_w,
        float         *              conv_state,
        const int32_t * __restrict__ s_copy,
        float         * __restrict__ out,
        const int                    H_k,
        const int                    H_v,
        const int                    ncols2,
        const int64_t                w_row_stride,
        const int64_t                cs_row_stride,
        const int                    dst_row,
        const float                  eps_l2,
        const float                  scale_l2) {
    constexpr int S         = GDN_DEC_S;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_red     = 2*GDN_DEC_MMV_BLOCK/warp_size;

    const int b   = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float s_y[S];
    __shared__ float s_red[n_red];

    if (b < 2*H_k) {
        // q/k head b (channels b*S ... b*S + S - 1): conv1d + silu, conv state shifted in place, then L2 norm
        if (tid < S) {
            const int ch = b*S + tid;
            float * cs_dst = conv_state + (int64_t) dst_row*cs_row_stride + (int64_t) ch*(GDN_DEC_DCONV - 1);

            float xc[GDN_DEC_DCONV];
            float wc[GDN_DEC_DCONV];
#pragma unroll
            for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
                xc[j] = cs_dst[j]; // speculative: read row == written row
            }
            xc[GDN_DEC_DCONV - 1] = qkv[ch];
#pragma unroll
            for (int j = 0; j < GDN_DEC_DCONV; j++) {
                wc[j] = conv_w[(int64_t) ch*GDN_DEC_DCONV + j];
            }
            const int32_t src_row = s_copy[0];
            if (src_row != dst_row) {
                const float * cs_src = conv_state + (int64_t) src_row*cs_row_stride + (int64_t) ch*(GDN_DEC_DCONV - 1);
#pragma unroll
                for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
                    xc[j] = cs_src[j];
                }
            }

            s_y[tid] = gdn_dec_conv_silu(xc, wc);

#pragma unroll
            for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
                cs_dst[j] = xc[j + 1];
            }
        }

        // L2 norm == rms_norm_f32<32, 1, true, do_scale>(eps_l2) followed by the 1/sqrt(S) scale
        __syncthreads();
        if (tid < 32) {
            const float4 xs = ((const float4 *) s_y)[tid];
            float tmp = gdn_dec_sumsq4(xs);
            tmp = warp_reduce_sum<32>(tmp);
            const float scale = rsqrtf(tmp / S + eps_l2);
            float4 r = make_float4(scale*xs.x, scale*xs.y, scale*xs.z, scale*xs.w);
            r.x *= scale_l2;
            r.y *= scale_l2;
            r.z *= scale_l2;
            r.w *= scale_l2;
            ((float4 *) (out + b*S))[tid] = r;
        }
        return;
    }

    // alpha/beta projections of v head h: threads [0, 128) -> alpha row h, [128, 256) -> beta row h.
    // == mul_mat_vec_f<T, float, 1, 128>: same per-thread accumulation order (col2 = t, t + 128, ...) and the
    // same reduction tree; the loads of a batch are all issued before the first product.
    const int    h    = b - 2*H_k;
    const int    half = tid / GDN_DEC_MMV_BLOCK;
    const int    t    = tid % GDN_DEC_MMV_BLOCK;
    const Tw   * w    = (half == 0 ? w_alpha : w_beta) + h*w_row_stride;
    const float2 * x2 = (const float2 *) x;

    // full batches without bounds checks (so that all loads of a batch are in flight together), then the tail
    constexpr int nb = 20;
    float sumf = 0.0f;
    int c0 = t;
    for (; c0 + (nb - 1)*GDN_DEC_MMV_BLOCK < ncols2; c0 += nb*GDN_DEC_MMV_BLOCK) {
        float2 wv[nb];
        float2 xv[nb];
#pragma unroll
        for (int u = 0; u < nb; u++) {
            wv[u] = gdn_dec_load_w2<Tw>(w, c0 + u*GDN_DEC_MMV_BLOCK);
            xv[u] = x2[c0 + u*GDN_DEC_MMV_BLOCK];
        }
#pragma unroll
        for (int u = 0; u < nb; u++) {
            sumf = sumf + gdn_dec_mul(wv[u].x, xv[u].x);
            sumf = sumf + gdn_dec_mul(wv[u].y, xv[u].y);
        }
    }
    for (; c0 < ncols2; c0 += GDN_DEC_MMV_BLOCK) {
        const float2 wv = gdn_dec_load_w2<Tw>(w, c0);
        const float2 xv = x2[c0];
        sumf = sumf + gdn_dec_mul(wv.x, xv.x);
        sumf = sumf + gdn_dec_mul(wv.y, xv.y);
    }

    sumf = warp_reduce_sum<warp_size>(sumf);
    if (tid % warp_size == 0) {
        s_red[tid / warp_size] = sumf;
    }
    __syncthreads();

    if (tid == 0) {
        // second stage of the block reduction: butterfly over the per-warp sums of each row
        constexpr int nw = GDN_DEC_MMV_BLOCK/warp_size;
        float va[nw];
        float vb[nw];
#pragma unroll
        for (int i = 0; i < nw; i++) {
            va[i] = s_red[i];
            vb[i] = s_red[nw + i];
        }
#pragma unroll
        for (int off = nw/2; off > 0; off >>= 1) {
#pragma unroll
            for (int i = 0; i < off; i++) {
                va[i] = va[i] + va[i + off];
                vb[i] = vb[i] + vb[i + off];
            }
        }
        const float alpha = va[0] + dt_bias[h];
        const float gate  = gdn_dec_softplus(alpha) * a[h];
        out[2*H_k*S + h]       = expf(gate);
        out[2*H_k*S + H_v + h] = gdn_dec_sigmoid(vb[0]);
        // arrival counter of head h for gdn_decode_out_split_f32
        ((unsigned int *) (out + 2*H_k*S + 2*H_v))[h] = 0u;
    }
}

// one block per v head: conv of the head's v channels, delta rule (== gated_delta_net_cuda<128, false, false, 4>),
// gated rms norm
template <bool q8>
static __global__ void __launch_bounds__(GDN_DEC_OUT_BLOCK)
gdn_decode_out_f32(
        const float   * __restrict__ prep,
        const float   * __restrict__ qkv,
        const float   * __restrict__ conv_w,
        float         *              conv_state,
        const float   * __restrict__ z,
        const float   * __restrict__ norm_w,
        float         *              ssm_state,
        const int32_t * __restrict__ s_copy,
        float         * __restrict__ out,
        const int                    H_k,
        const int                    H_v,
        const int64_t                cs_row_stride,
        const int64_t                ss_row_stride,
        const int                    dst_row,
        const float                  scale,
        const float                  eps,
        block_q8_1    * __restrict__ y,
        int           * __restrict__ ysum) {
    constexpr int S             = GDN_DEC_S;
    constexpr int warp_size     = ggml_cuda_get_physical_warp_size() < S ? ggml_cuda_get_physical_warp_size() : S;
    constexpr int cols_per_warp = 4;
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S / lanes_per_col;
    constexpr int n_warps       = GDN_DEC_OUT_BLOCK / warp_size;
    constexpr int cols_per_pass = n_warps * cols_per_warp;
    constexpr int n_pass        = S / cols_per_pass;
    static_assert(S % cols_per_pass == 0, "unsupported block size");

    const int h           = blockIdx.x;
    const int tid         = threadIdx.x;
    const int lane        = tid % warp_size;
    const int warp        = tid / warp_size;
    const int col_in_warp = lane / lanes_per_col;
    const int lane_in_col = lane - col_in_warp*lanes_per_col;
    const int hk          = h % H_k;

    __shared__ float s_v[S];
    __shared__ float s_o[S];

    // ---- issue all loads first
    // the state is stored transposed: M[col][i] = S[i][col], each column is contiguous
    float * st_dst = ssm_state + (int64_t) dst_row*ss_row_stride + (int64_t) h*S*S;

    float s_shard[n_pass][rows_per_lane];
#pragma unroll
    for (int p = 0; p < n_pass; p++) {
        const int col = p*cols_per_pass + warp*cols_per_warp + col_in_warp;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[p][r] = st_dst[col*S + r*lanes_per_col + lane_in_col]; // speculative: read row == written row
        }
    }

    const float * q = prep + hk*S;
    const float * k = prep + H_k*S + hk*S;
    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r*lanes_per_col + lane_in_col;
        k_reg[r] = k[i];
        q_reg[r] = q[i];
    }
    const float g_val    = prep[2*H_k*S + h];
    const float beta_val = prep[2*H_k*S + H_v + h];

    // v channels of this head
    const int ch = 2*H_k*S + h*S + tid;
    float * cs_dst = conv_state + (int64_t) dst_row*cs_row_stride + (int64_t) ch*(GDN_DEC_DCONV - 1);
    float xc[GDN_DEC_DCONV];
    float wc[GDN_DEC_DCONV];
    if (tid < S) {
#pragma unroll
        for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
            xc[j] = cs_dst[j]; // speculative
        }
        xc[GDN_DEC_DCONV - 1] = qkv[ch];
#pragma unroll
        for (int j = 0; j < GDN_DEC_DCONV; j++) {
            wc[j] = conv_w[(int64_t) ch*GDN_DEC_DCONV + j];
        }
    }

    // inputs of the output norm
    float4 zz = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 nw = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (tid < 32) {
        zz = ((const float4 *) (z + h*S))[tid];
        nw = ((const float4 *) norm_w)[tid];
    }

    const int32_t src_row = s_copy[0];
    if (src_row != dst_row) {
        const float * st_src = ssm_state + (int64_t) src_row*ss_row_stride + (int64_t) h*S*S;
#pragma unroll
        for (int p = 0; p < n_pass; p++) {
            const int col = p*cols_per_pass + warp*cols_per_warp + col_in_warp;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[p][r] = st_src[col*S + r*lanes_per_col + lane_in_col];
            }
        }
        if (tid < S) {
            const float * cs_src = conv_state + (int64_t) src_row*cs_row_stride + (int64_t) ch*(GDN_DEC_DCONV - 1);
#pragma unroll
            for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
                xc[j] = cs_src[j];
            }
        }
    }

    // ---- conv1d + silu of the v channels (conv state shifted in place)
    if (tid < S) {
        s_v[tid] = gdn_dec_conv_silu(xc, wc);
#pragma unroll
        for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
            cs_dst[j] = xc[j + 1];
        }
    }
    __syncthreads();

    // ---- delta rule
    // the fused/unfused pattern below is the one of gated_delta_net_cuda<128, false, false, 4> as compiled
    // for gfx90a (rows 0-3 of kv fused, 4-7 unfused; state rows 0-5 fused, 6-7 unfused; attn fused)
#pragma unroll
    for (int p = 0; p < n_pass; p++) {
        const int col = p*cols_per_pass + warp*cols_per_warp + col_in_warp;

        // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            if (r < rows_per_lane/2) {
                kv_shard = fmaf(s_shard[p][r], k_reg[r], kv_shard);
            } else {
                kv_shard = kv_shard + gdn_dec_mul(s_shard[p][r], k_reg[r]);
            }
        }
        float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

        // delta[col] = (v[col] - g * kv[col]) * beta
        float delta_col = beta_val * fmaf(-g_val, kv_col, s_v[col]);

        // S[i][col] = g * S[i][col] + k[i] * delta[col];  attn[col] = (S^T @ q)[col]
        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            if (r < rows_per_lane - 2) {
                s_shard[p][r] = fmaf(g_val, s_shard[p][r], gdn_dec_mul(k_reg[r], delta_col));
            } else {
                s_shard[p][r] = gdn_dec_mul(g_val, s_shard[p][r]) + gdn_dec_mul(k_reg[r], delta_col);
            }
            attn_partial = fmaf(s_shard[p][r], q_reg[r], attn_partial);
        }

        float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

        if (lane_in_col == 0) {
            s_o[col] = attn_col * scale;
        }

#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            st_dst[col*S + r*lanes_per_col + lane_in_col] = s_shard[p][r];
        }
    }

    __syncthreads();

    // ---- gated norm == rms_norm_f32<32, 1, true, do_multiply>(eps) * w, then silu(z) * x
    if (tid < 32) {
        const float4 xs = ((const float4 *) s_o)[tid];
        float tmp = gdn_dec_sumsq4(xs);
        tmp = warp_reduce_sum<32>(tmp);
        const float sc = rsqrtf(tmp / S + eps);
        float4 r = make_float4(sc*xs.x, sc*xs.y, sc*xs.z, sc*xs.w);
        r.x *= nw.x;
        r.y *= nw.y;
        r.z *= nw.z;
        r.w *= nw.w;
        float4 o;
        o.x = ggml_cuda_op_silu_single(zz.x) * r.x;
        o.y = ggml_cuda_op_silu_single(zz.y) * r.y;
        o.z = ggml_cuda_op_silu_single(zz.z) * r.z;
        o.w = ggml_cuda_op_silu_single(zz.w) * r.w;
        gdn_dec_store_out<q8>(out, o, h, tid, y, ysum);
    }
}

// several blocks per v head (each owns cols_per_block state columns, like gated_delta_net_cuda): conv of the
// block's v channels, delta rule; the last block of a head to finish applies the gated rms norm to the head's
// output (per-head arrival counters: tail of the prep result, zeroed by the prep kernel)
template <bool q8>
static __global__ void __launch_bounds__(GDN_DEC_SPLIT_BLOCK)
gdn_decode_out_split_f32(
        const float   * __restrict__ prep,
        const float   * __restrict__ qkv,
        const float   * __restrict__ conv_w,
        float         *              conv_state,
        const float   * __restrict__ z,
        const float   * __restrict__ norm_w,
        float         *              ssm_state,
        const int32_t * __restrict__ s_copy,
        float         *              out,
        const int                    H_k,
        const int                    H_v,
        const int64_t                cs_row_stride,
        const int64_t                ss_row_stride,
        const int                    dst_row,
        const float                  scale,
        const float                  eps,
        block_q8_1    * __restrict__ y,
        int           * __restrict__ ysum) {
    constexpr int S              = GDN_DEC_S;
    constexpr int warp_size      = ggml_cuda_get_physical_warp_size() < S ? ggml_cuda_get_physical_warp_size() : S;
    constexpr int cols_per_warp  = 4;
    constexpr int lanes_per_col  = warp_size / cols_per_warp;
    constexpr int rows_per_lane  = S / lanes_per_col;
    constexpr int n_warps        = GDN_DEC_SPLIT_BLOCK / warp_size;
    constexpr int cols_per_block = n_warps * cols_per_warp;
    constexpr int split          = S / cols_per_block;
    static_assert(S % cols_per_block == 0, "unsupported block size");

    const int h           = blockIdx.x / split;
    const int part        = blockIdx.x % split;
    const int tid         = threadIdx.x;
    const int lane        = tid % warp_size;
    const int warp        = tid / warp_size;
    const int col_in_warp = lane / lanes_per_col;
    const int lane_in_col = lane - col_in_warp*lanes_per_col;
    const int hk          = h % H_k;
    const int col         = part*cols_per_block + warp*cols_per_warp + col_in_warp;

    __shared__ float s_v[cols_per_block];
    __shared__ int   s_last;

    unsigned int * counters = (unsigned int *) (prep + 2*H_k*S + 2*H_v);

    // ---- issue all loads first
    // the state is stored transposed: M[col][i] = S[i][col], each column is contiguous
    float * st_dst = ssm_state + (int64_t) dst_row*ss_row_stride + (int64_t) h*S*S + (int64_t) col*S;

    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = st_dst[r*lanes_per_col + lane_in_col]; // speculative: read row == written row
    }

    const float * q = prep + hk*S;
    const float * k = prep + H_k*S + hk*S;
    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r*lanes_per_col + lane_in_col;
        k_reg[r] = k[i];
        q_reg[r] = q[i];
    }
    const float g_val    = prep[2*H_k*S + h];
    const float beta_val = prep[2*H_k*S + H_v + h];

    // v channels of this block's columns
    const int ch = 2*H_k*S + h*S + part*cols_per_block + tid;
    float * cs_dst = conv_state + (int64_t) dst_row*cs_row_stride + (int64_t) ch*(GDN_DEC_DCONV - 1);
    float xc[GDN_DEC_DCONV];
    float wc[GDN_DEC_DCONV];
    if (tid < cols_per_block) {
#pragma unroll
        for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
            xc[j] = cs_dst[j]; // speculative
        }
        xc[GDN_DEC_DCONV - 1] = qkv[ch];
#pragma unroll
        for (int j = 0; j < GDN_DEC_DCONV; j++) {
            wc[j] = conv_w[(int64_t) ch*GDN_DEC_DCONV + j];
        }
    }

    const int32_t src_row = s_copy[0];
    if (src_row != dst_row) {
        const float * st_src = ssm_state + (int64_t) src_row*ss_row_stride + (int64_t) h*S*S + (int64_t) col*S;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] = st_src[r*lanes_per_col + lane_in_col];
        }
        if (tid < cols_per_block) {
            const float * cs_src = conv_state + (int64_t) src_row*cs_row_stride + (int64_t) ch*(GDN_DEC_DCONV - 1);
#pragma unroll
            for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
                xc[j] = cs_src[j];
            }
        }
    }

    // ---- conv1d + silu of the v channels (conv state shifted in place)
    if (tid < cols_per_block) {
        s_v[tid] = gdn_dec_conv_silu(xc, wc);
#pragma unroll
        for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
            cs_dst[j] = xc[j + 1];
        }
    }
    __syncthreads();

    // ---- delta rule (fused/unfused pattern of gated_delta_net_cuda<128, false, false, 4>, see gdn_decode_out_f32)
    {
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            if (r < rows_per_lane/2) {
                kv_shard = fmaf(s_shard[r], k_reg[r], kv_shard);
            } else {
                kv_shard = kv_shard + gdn_dec_mul(s_shard[r], k_reg[r]);
            }
        }
        const float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

        const float delta_col = beta_val * fmaf(-g_val, kv_col, s_v[col - part*cols_per_block]);

        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            if (r < rows_per_lane - 2) {
                s_shard[r] = fmaf(g_val, s_shard[r], gdn_dec_mul(k_reg[r], delta_col));
            } else {
                s_shard[r] = gdn_dec_mul(g_val, s_shard[r]) + gdn_dec_mul(k_reg[r], delta_col);
            }
            attn_partial = fmaf(s_shard[r], q_reg[r], attn_partial);
        }

        const float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

        if (lane_in_col == 0) {
            out[h*S + col] = attn_col * scale;
        }

#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            st_dst[r*lanes_per_col + lane_in_col] = s_shard[r];
        }
    }

    // ---- the last block of the head applies the gated norm
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        s_last = atomicAdd(&counters[h], 1u) == (unsigned int) (split - 1);
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    __threadfence();

    // == rms_norm_f32<32, 1, true, do_multiply>(eps) * w, then silu(z) * x
    if (tid < 32) {
        const float4 xs = ((const float4 *) (out + h*S))[tid]; // L1 was invalidated by the fence
        const float4 zz = ((const float4 *) (z + h*S))[tid];
        const float4 nw = ((const float4 *) norm_w)[tid];
        float tmp = gdn_dec_sumsq4(xs);
        tmp = warp_reduce_sum<32>(tmp);
        const float sc = rsqrtf(tmp / S + eps);
        float4 r = make_float4(sc*xs.x, sc*xs.y, sc*xs.z, sc*xs.w);
        r.x *= nw.x;
        r.y *= nw.y;
        r.z *= nw.z;
        r.w *= nw.w;
        float4 o;
        o.x = ggml_cuda_op_silu_single(zz.x) * r.x;
        o.y = ggml_cuda_op_silu_single(zz.y) * r.y;
        o.z = ggml_cuda_op_silu_single(zz.z) * r.z;
        o.w = ggml_cuda_op_silu_single(zz.w) * r.w;
        gdn_dec_store_out<q8>(out, o, h, tid, y, ysum);
    }
}

#if defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define GDN_DEC_DIET 1
#else
#define GDN_DEC_DIET 0
#endif

#if GDN_DEC_DIET
// "Diet" variants of gdn_decode_prep_f32<bf16> and gdn_decode_out_split_f32 for the qwen3.5 shapes on wave64 devices
// (H_k = 16, H_v = 48, S = 128, conv 4, bf16 alpha/beta rows of 5120 values). In the model every dispatch starts with a
// cold instruction cache, so the code size on the critical path is a large part of these short kernels' time.
// Same arithmetic as the generic kernels (bit-identical results), less code:
// - constant shapes: no integer division (h % H_k), 32-bit offsets, a single load batch for the alpha/beta rows;
// - the dst-row base pointers come from the host (the src row is still read from s_copy on the device);
// - DPP / ds_swizzle reductions with the same butterflies instead of ds_bpermute (float addition is commutative, so
//   every lane still gets the bit-identical value), exact max / int sums in any order;
// - the gated-norm + q8_1 epilogue on one full wave: element e = lane + 64*pass, two rolled passes (one silu and one
//   division of code instead of four of each).
#define GDN_DIET_HK    16
#define GDN_DIET_HV    48
#define GDN_DIET_NCOL2 2560

template <int ctrl>
static __device__ __forceinline__ float gdn_diet_dpp(const float v) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), ctrl, 0xF, 0xF, false));
}

template <int ctrl>
static __device__ __forceinline__ int gdn_diet_dpp_i(const int v) {
    return __builtin_amdgcn_update_dpp(0, v, ctrl, 0xF, 0xF, false);
}

// lane ^ 16 within aligned groups of 32 lanes
static __device__ __forceinline__ float gdn_diet_swz16(const float v) {
    return __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F));
}

// == warp_reduce_sum<16> (butterfly xor 8, 4, 2, 1). row_ror:8 is xor 8 within a DPP row of 16 lanes; after that step
// v[j] == v[j ^ 8], so row_ror:4 delivers the value of lane j ^ 4; quad_perm [2,3,0,1] / [1,0,3,2] are xor 2 / xor 1
static __device__ __forceinline__ float gdn_diet_sum16(float v) {
    v += gdn_diet_dpp<0x128>(v);
    v += gdn_diet_dpp<0x124>(v);
    v += gdn_diet_dpp<0x4E>(v);
    v += gdn_diet_dpp<0xB1>(v);
    return v;
}

// == warp_reduce_sum<32>: xor 16 with ds_swizzle, then as above
static __device__ __forceinline__ float gdn_diet_sum32(float v) {
    v += gdn_diet_swz16(v);
    return gdn_diet_sum16(v);
}

// == warp_reduce_sum<64>: xor 32 with ds_bpermute, then as above
static __device__ __forceinline__ float gdn_diet_sum64(float v, const int tid) {
    v += __int_as_float(__builtin_amdgcn_ds_bpermute(((tid ^ 32) & 63) << 2, __float_as_int(v)));
    return gdn_diet_sum32(v);
}

// max / int sum over aligned groups of 32 lanes (exact operations, the order does not matter)
static __device__ __forceinline__ float gdn_diet_max32(float v) {
    v = fmaxf(v, gdn_diet_dpp<0xB1>(v));  // xor 1
    v = fmaxf(v, gdn_diet_dpp<0x4E>(v));  // xor 2
    v = fmaxf(v, gdn_diet_dpp<0x141>(v)); // row_half_mirror: the other quad of the 8
    v = fmaxf(v, gdn_diet_dpp<0x140>(v)); // row_mirror: the other 8 of the row
    v = fmaxf(v, gdn_diet_swz16(v));
    return v;
}

static __device__ __forceinline__ int gdn_diet_isum32(int v) {
    v += gdn_diet_dpp_i<0xB1>(v);
    v += gdn_diet_dpp_i<0x4E>(v);
    v += gdn_diet_dpp_i<0x141>(v);
    v += gdn_diet_dpp_i<0x140>(v);
    v += __builtin_amdgcn_ds_swizzle(v, 0x401F);
    return v;
}

// == gdn_decode_prep_f32<nv_bfloat16> for the shapes above; cs_dst = conv state row dst_row
// early (N1, GGML_GDN_PREP_EARLY=0 disables): dt_bias[h] and a[h] are loaded together with the projection rows instead
// of after the reduction barrier, and the beta gate (sigmoid) runs in wave 2 while wave 0 runs the alpha gate chain,
// instead of both in thread 0 one after the other (same operations on the same values: bit-identical)
template <bool warm, bool early = false>
static __global__ void __launch_bounds__(2*GDN_DEC_MMV_BLOCK)
gdn_decode_prep_diet(
        const float       * __restrict__ qkv,
        const float       * __restrict__ x,
        const nv_bfloat16 * __restrict__ w_alpha,
        const nv_bfloat16 * __restrict__ w_beta,
        const float       * __restrict__ dt_bias,
        const float       * __restrict__ a,
        const float       * __restrict__ conv_w,
        float             *              cs_dst,
        const int32_t     * __restrict__ s_copy,
        float             * __restrict__ out,
        const int64_t                    cs_row_stride,
        const int                        dst_row,
        const float                      eps_l2,
        const float                      scale_l2) {
    constexpr int S  = GDN_DEC_S;
    constexpr int HK = GDN_DIET_HK;
    constexpr int HV = GDN_DIET_HV;
    constexpr int nb = GDN_DIET_NCOL2/GDN_DEC_MMV_BLOCK; // one batch of loads per thread

    const int b   = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float s_y[S];
    __shared__ float s_red[2*GDN_DEC_MMV_BLOCK/64];

    if (b < 2*HK) {
        if (tid < S) {
            const int ch = b*S + tid;
            float * cs = cs_dst + ch*(GDN_DEC_DCONV - 1);

            float xc[GDN_DEC_DCONV];
            float wc[GDN_DEC_DCONV];
#pragma unroll
            for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
                xc[j] = cs[j]; // speculative: read row == written row
            }
            xc[GDN_DEC_DCONV - 1] = qkv[ch];
#pragma unroll
            for (int j = 0; j < GDN_DEC_DCONV; j++) {
                wc[j] = conv_w[ch*GDN_DEC_DCONV + j];
            }
            const int32_t src_row = s_copy[0];
            if (src_row != dst_row) {
                const float * cs_src = cs + (int64_t) (src_row - dst_row)*cs_row_stride;
#pragma unroll
                for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
                    xc[j] = cs_src[j];
                }
            }

            s_y[tid] = gdn_dec_conv_silu(xc, wc);

#pragma unroll
            for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
                cs[j] = xc[j + 1];
            }
        }

        __syncthreads();
        if (tid < 32) {
            const float4 xs = ((const float4 *) s_y)[tid];
            float tmp = gdn_dec_sumsq4(xs);
            tmp = gdn_diet_sum32(tmp);
            const float scale = rsqrtf(tmp / S + eps_l2);
            float4 r = make_float4(scale*xs.x, scale*xs.y, scale*xs.z, scale*xs.w);
            r.x *= scale_l2;
            r.y *= scale_l2;
            r.z *= scale_l2;
            r.w *= scale_l2;
            ((float4 *) (out + b*S))[tid] = r;
        }
        return;
    }

    // alpha/beta projections of v head h (threads [0, 128) -> alpha row h, [128, 256) -> beta row h); the row of a wave
    // is wave-uniform
    const int h    = b - 2*HK;
    const int half = __builtin_amdgcn_readfirstlane(tid) / GDN_DEC_MMV_BLOCK;
    const int t    = tid % GDN_DEC_MMV_BLOCK;
    const nv_bfloat16 * w  = (half == 0 ? w_alpha : w_beta) + h*(2*GDN_DIET_NCOL2);
    const float2      * x2 = (const float2 *) x;

    if constexpr (!warm) {
        float dt_e = 0.0f;
        float a_e  = 0.0f;
        if constexpr (early) {
            if (half == 0) {
                dt_e = dt_bias[h];
                a_e  = a[h];
            }
        }
        float2 wv[nb];
        float2 xv[nb];
#pragma unroll
        for (int u = 0; u < nb; u++) {
            wv[u] = gdn_dec_load_w2<nv_bfloat16>(w, t + u*GDN_DEC_MMV_BLOCK);
            xv[u] = x2[t + u*GDN_DEC_MMV_BLOCK];
        }
        float sumf = 0.0f;
#pragma unroll
        for (int u = 0; u < nb; u++) {
            sumf = sumf + gdn_dec_mul(wv[u].x, xv[u].x);
            sumf = sumf + gdn_dec_mul(wv[u].y, xv[u].y);
        }

        sumf = gdn_diet_sum64(sumf, tid);
        if (tid % 64 == 0) {
            s_red[tid / 64] = sumf;
        }
        __syncthreads();

        if constexpr (early) {
            if (tid == 0) {
                const float alpha = (s_red[0] + s_red[1]) + dt_e;
                const float gate  = gdn_dec_softplus(alpha) * a_e;
                out[2*HK*S + h] = expf(gate);
                ((unsigned int *) (out + 2*HK*S + 2*HV))[h] = 0u;
            } else if (tid == 2*64) {
                out[2*HK*S + HV + h] = gdn_dec_sigmoid(s_red[2] + s_red[3]);
            }
            return;
        }
        if (tid == 0) {
            const float alpha = (s_red[0] + s_red[1]) + dt_bias[h];
            const float gate  = gdn_dec_softplus(alpha) * a[h];
            out[2*HK*S + h]      = expf(gate);
            out[2*HK*S + HV + h] = gdn_dec_sigmoid(s_red[2] + s_red[3]);
            ((unsigned int *) (out + 2*HK*S + 2*HV))[h] = 0u;
        }
        return;
    }

    const float dt_h = dt_bias[h];
    const float a_h  = a[h];

    // warm: the gate math (softplus = log1pf(expf(x)) alone is ~130 instructions) runs twice through the same code: in
    // the first pass on dummy inputs while the loads are in flight, which pulls its code into the cold instruction
    // cache, then on the projections (the result of the first pass is discarded).
    int    wr[nb]; // bf16 pairs, converted after the first pass
    float2 xv[nb];
    float va  = 0.0f;
    float vb  = 0.0f;
    float dtv = 0.0f;
    float av  = 0.0f;
    float gate_exp;
    float beta_sig;
#pragma unroll 1
    for (int pass = 0; ; pass++) {
        if (pass == 0) {
            int zoff = 0; // opaque 0: keeps the address arithmetic next to the loads (uniform base + 32-bit offsets)
            asm volatile("" : "+s"(zoff));
            const int    * wp = (const int *) w + zoff;
            const float2 * xp = x2 + zoff;
#pragma unroll
            for (int u = 0; u < nb; u++) {
                wr[u] = wp[t + u*GDN_DEC_MMV_BLOCK];
                xv[u] = xp[t + u*GDN_DEC_MMV_BLOCK];
            }
        }
        const float alpha = va + dtv;
        const float gate  = gdn_dec_softplus(alpha) * av;
        gate_exp = expf(gate);
        beta_sig = gdn_dec_sigmoid(vb);
        asm volatile("" :: "v"(gate_exp), "v"(beta_sig)); // keep the first pass (not sunk behind the loop)
        if (pass != 0) {
            break;
        }

        float sumf = 0.0f;
#pragma unroll
        for (int u = 0; u < nb; u++) {
            const float2 wv = gdn_dec_load_w2<nv_bfloat16>((const nv_bfloat16 *) &wr[u], 0); // same conversion
            sumf = sumf + gdn_dec_mul(wv.x, xv[u].x);
            sumf = sumf + gdn_dec_mul(wv.y, xv[u].y);
        }

        sumf = gdn_diet_sum64(sumf, tid);
        if (tid % 64 == 0) {
            s_red[tid / 64] = sumf;
        }
        __syncthreads();

        // second stage of the block reduction (== the nw = 2 butterfly of gdn_decode_prep_f32), in every thread
        va  = s_red[0] + s_red[1];
        vb  = s_red[2] + s_red[3];
        dtv = dt_h;
        av  = a_h;
    }

    if (tid == 0) {
        out[2*HK*S + h]      = gate_exp;
        out[2*HK*S + HV + h] = beta_sig;
        ((unsigned int *) (out + 2*HK*S + 2*HV))[h] = 0u;
    }
}

// == gdn_decode_out_split_f32<q8> for the shapes above (wave64: 16 lanes per state column, 16 columns per block);
// cs_dst / ss_dst = conv / recurrent state row dst_row
template <bool q8>
static __global__ void __launch_bounds__(GDN_DEC_SPLIT_BLOCK)
gdn_decode_out_split_diet(
        const float   * __restrict__ prep,
        const float   * __restrict__ qkv,
        const float   * __restrict__ conv_w,
        float         *              cs_dst,
        const float   * __restrict__ z,
        const float   * __restrict__ norm_w,
        float         *              ss_dst,
        const int32_t * __restrict__ s_copy,
        float         *              out,
        const int64_t                cs_row_stride,
        const int64_t                ss_row_stride,
        const int                    dst_row,
        const float                  scale,
        const float                  eps,
        block_q8_1    * __restrict__ y,
        int           * __restrict__ ysum) {
    constexpr int S              = GDN_DEC_S;
    constexpr int HK             = GDN_DIET_HK;
    constexpr int HV             = GDN_DIET_HV;
    constexpr int lanes_per_col  = 16;
    constexpr int rows_per_lane  = S / lanes_per_col;
    constexpr int cols_per_block = GDN_DEC_SPLIT_BLOCK / lanes_per_col;
    constexpr int split          = S / cols_per_block;

    const int h           = blockIdx.x / split;
    const int part        = blockIdx.x % split;
    const int tid         = threadIdx.x;
    const int lane_in_col = tid % lanes_per_col;
    const int col         = part*cols_per_block + tid / lanes_per_col;
    const int hk          = h % HK;

    __shared__ float s_v[cols_per_block];
    __shared__ int   s_last;

    unsigned int * counters = (unsigned int *) (prep + 2*HK*S + 2*HV);

    // ---- issue all loads first (state stored transposed: column col of head h is contiguous)
    float * st = ss_dst + (unsigned) (h*S*S + col*S + lane_in_col);

    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = st[r*lanes_per_col]; // speculative: read row == written row
    }

    const float * q = prep + (unsigned) (hk*S + lane_in_col);
    const float * k = q + HK*S;
    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        k_reg[r] = k[r*lanes_per_col];
        q_reg[r] = q[r*lanes_per_col];
    }
    const float g_val    = prep[2*HK*S + h];
    const float beta_val = prep[2*HK*S + HV + h];

    const int32_t src_row = s_copy[0];

    // ---- conv1d + silu of this block's v channels (conv state shifted in place)
    if (tid < cols_per_block) {
        const unsigned ch = 2*HK*S + h*S + part*cols_per_block + tid;
        float * cs = cs_dst + ch*(GDN_DEC_DCONV - 1);
        float xc[GDN_DEC_DCONV];
        float wc[GDN_DEC_DCONV];
#pragma unroll
        for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
            xc[j] = cs[j]; // speculative
        }
        xc[GDN_DEC_DCONV - 1] = qkv[ch];
#pragma unroll
        for (int j = 0; j < GDN_DEC_DCONV; j++) {
            wc[j] = conv_w[ch*GDN_DEC_DCONV + j];
        }
        if (src_row != dst_row) {
            const float * cs_src = cs + (int64_t) (src_row - dst_row)*cs_row_stride;
#pragma unroll
            for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
                xc[j] = cs_src[j];
            }
        }
        s_v[tid] = gdn_dec_conv_silu(xc, wc);
#pragma unroll
        for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
            cs[j] = xc[j + 1];
        }
    }

    if (src_row != dst_row) {
        const float * st_src = st + (int64_t) (src_row - dst_row)*ss_row_stride;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] = st_src[r*lanes_per_col];
        }
    }
    __syncthreads();

    // ---- delta rule (fused/unfused pattern of gated_delta_net_cuda<128, false, false, 4>, see gdn_decode_out_f32)
    {
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            if (r < rows_per_lane/2) {
                kv_shard = fmaf(s_shard[r], k_reg[r], kv_shard);
            } else {
                kv_shard = kv_shard + gdn_dec_mul(s_shard[r], k_reg[r]);
            }
        }
        const float kv_col = gdn_diet_sum16(kv_shard);

        const float delta_col = beta_val * fmaf(-g_val, kv_col, s_v[tid / lanes_per_col]);

        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            if (r < rows_per_lane - 2) {
                s_shard[r] = fmaf(g_val, s_shard[r], gdn_dec_mul(k_reg[r], delta_col));
            } else {
                s_shard[r] = gdn_dec_mul(g_val, s_shard[r]) + gdn_dec_mul(k_reg[r], delta_col);
            }
            attn_partial = fmaf(s_shard[r], q_reg[r], attn_partial);
        }

        const float attn_col = gdn_diet_sum16(attn_partial);

        if (lane_in_col == 0) {
            out[h*S + col] = attn_col * scale;
        }

#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            st[r*lanes_per_col] = s_shard[r];
        }
    }

    // ---- the last block of the head applies the gated norm
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        s_last = atomicAdd(&counters[h], 1u) == (unsigned int) (split - 1);
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    __threadfence();

    // == rms_norm_f32<32, 1, true, do_multiply>(eps) * w, then silu(z) * x (+ q8_1): both halves of wave 0 redo the
    // 32-lane reduction of the float4 sums of squares (identical totals), then lane l handles elements l and l + 64
    if (tid < 64) {
        const float * oh = out + h*S;
        const float4 xs = ((const float4 *) oh)[tid % 32]; // L1 was invalidated by the fence
        float x0 = oh[tid];
        float x1 = oh[tid + 64];
        float z0 = z[h*S + tid];
        float z1 = z[h*S + tid + 64];
        float w0 = norm_w[tid];
        float w1 = norm_w[tid + 64];
        float tmp = gdn_dec_sumsq4(xs);
        tmp = gdn_diet_sum32(tmp);
        const float sc = rsqrtf(tmp / S + eps);
        int e = h*S + tid;
#pragma unroll 1
        for (int pass = 0; pass < 2; pass++) {
            float r = sc*x0;
            r *= w0;
            const float o = ggml_cuda_op_silu_single(z0) * r;
            out[e] = o;
            if constexpr (q8) {
                const float  amax = gdn_diet_max32(fabsf(o));
                const float  d    = amax / 127.0f;
                const int8_t qv   = amax == 0.0f ? 0 : roundf(o / d);
                const int    sum  = gdn_diet_isum32(qv);
                const int    ib   = e / QK8_1;
                y[ib].qs[e % QK8_1] = qv;
                if (e % QK8_1 == 0) {
                    y[ib].ds  = make_half2(d, d * sum);
                    ysum[ib] = sum;
                }
            }
            x0 = x1;
            z0 = z1;
            w0 = w1;
            e += 64;
        }
    }
}

// == gdn_decode_out_split_diet<q8>, but the epilogue code (silu, the q8_1 divisions and reductions) first runs once on
// dummy inputs (no stores) in every block while the state loads are in flight, which pulls it into the cold
// instruction cache before the last block of the head needs it: one rolled loop, pass -1 = dummy pass + main work
template <bool q8>
static __global__ void __launch_bounds__(GDN_DEC_SPLIT_BLOCK)
gdn_decode_out_split_diet_warm(
        const float   * __restrict__ prep,
        const float   * __restrict__ qkv,
        const float   * __restrict__ conv_w,
        float         *              cs_dst,
        const float   * __restrict__ z,
        const float   * __restrict__ norm_w,
        float         *              ss_dst,
        const int32_t * __restrict__ s_copy,
        float         *              out,
        const int64_t                cs_row_stride,
        const int64_t                ss_row_stride,
        const int                    dst_row,
        const float                  scale,
        const float                  eps,
        block_q8_1    * __restrict__ y,
        int           * __restrict__ ysum) {
    constexpr int S              = GDN_DEC_S;
    constexpr int HK             = GDN_DIET_HK;
    constexpr int HV             = GDN_DIET_HV;
    constexpr int lanes_per_col  = 16;
    constexpr int rows_per_lane  = S / lanes_per_col;
    constexpr int cols_per_block = GDN_DEC_SPLIT_BLOCK / lanes_per_col;
    constexpr int split          = S / cols_per_block;

    const int h           = blockIdx.x / split;
    const int part        = blockIdx.x % split;
    const int tid         = threadIdx.x;
    const int lane_in_col = tid % lanes_per_col;
    const int col         = part*cols_per_block + tid / lanes_per_col;
    const int hk          = h % HK;

    __shared__ float s_v[cols_per_block];
    __shared__ int   s_last;

    unsigned int * counters = (unsigned int *) (prep + 2*HK*S + 2*HV);

    // ---- issue all loads first (state stored transposed: column col of head h is contiguous)
    float * st = ss_dst + (unsigned) (h*S*S + col*S + lane_in_col);

    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = st[r*lanes_per_col]; // speculative: read row == written row
    }

    const float * q = prep + (unsigned) (hk*S + lane_in_col);
    const float * k = q + HK*S;
    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        k_reg[r] = k[r*lanes_per_col];
        q_reg[r] = q[r*lanes_per_col];
    }
    const float g_val    = prep[2*HK*S + h];
    const float beta_val = prep[2*HK*S + HV + h];

    const unsigned ch = 2*HK*S + h*S + part*cols_per_block + tid;
    float * cs = cs_dst + ch*(GDN_DEC_DCONV - 1);
    float xc[GDN_DEC_DCONV] = {0.0f, 0.0f, 0.0f, 0.0f};
    float wc[GDN_DEC_DCONV] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (tid < cols_per_block) {
#pragma unroll
        for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
            xc[j] = cs[j]; // speculative
        }
        xc[GDN_DEC_DCONV - 1] = qkv[ch];
#pragma unroll
        for (int j = 0; j < GDN_DEC_DCONV; j++) {
            wc[j] = conv_w[ch*GDN_DEC_DCONV + j];
        }
    }

    const int32_t src_row = s_copy[0];

    // epilogue inputs; dummies (non-zero, so that the division path runs) in pass -1
    float sc = 1.0f;
    float x0 = 1.0f;
    float x1 = 1.0f;
    float z0 = 1.0f;
    float z1 = 1.0f;
    float w0 = 1.0f;
    float w1 = 1.0f;
    int   e  = 0;
#pragma unroll 1
    for (int pass = -1; pass < 2; pass++) {
        // == rms_norm_f32<32, 1, true, do_multiply>(eps) * w, then silu(z) * x (+ q8_1), element e (see the diet kernel)
        {
            float r = sc*x0;
            r *= w0;
            const float o = ggml_cuda_op_silu_single(z0) * r;
            asm volatile("" :: "v"(o));
            if (pass >= 0) {
                out[e] = o;
            }
            if constexpr (q8) {
                const float  amax = gdn_diet_max32(fabsf(o));
                const float  d    = amax / 127.0f;
                const int8_t qv   = amax == 0.0f ? 0 : roundf(o / d);
                const int    sum  = gdn_diet_isum32(qv);
                asm volatile("" :: "v"(d), "v"((int) qv), "v"(sum));
                if (pass >= 0) {
                    const int ib = e / QK8_1;
                    y[ib].qs[e % QK8_1] = qv;
                    if (e % QK8_1 == 0) {
                        y[ib].ds  = make_half2(d, d * sum);
                        ysum[ib] = sum;
                    }
                }
            }
            x0 = x1;
            z0 = z1;
            w0 = w1;
            e += 64;
        }
        if (pass >= 0) {
            continue;
        }

        // ---- pass -1: conv1d + silu of this block's v channels (conv state shifted in place)
        if (tid < cols_per_block) {
            if (src_row != dst_row) {
                const float * cs_src = cs + (int64_t) (src_row - dst_row)*cs_row_stride;
#pragma unroll
                for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
                    xc[j] = cs_src[j];
                }
            }
            s_v[tid] = gdn_dec_conv_silu(xc, wc);
#pragma unroll
            for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
                cs[j] = xc[j + 1];
            }
        }

        if (src_row != dst_row) {
            const float * st_src = st + (int64_t) (src_row - dst_row)*ss_row_stride;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r] = st_src[r*lanes_per_col];
            }
        }
        __syncthreads();

        // ---- delta rule (fused/unfused pattern of gated_delta_net_cuda<128, false, false, 4>)
        {
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                if (r < rows_per_lane/2) {
                    kv_shard = fmaf(s_shard[r], k_reg[r], kv_shard);
                } else {
                    kv_shard = kv_shard + gdn_dec_mul(s_shard[r], k_reg[r]);
                }
            }
            const float kv_col = gdn_diet_sum16(kv_shard);

            const float delta_col = beta_val * fmaf(-g_val, kv_col, s_v[tid / lanes_per_col]);

            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                if (r < rows_per_lane - 2) {
                    s_shard[r] = fmaf(g_val, s_shard[r], gdn_dec_mul(k_reg[r], delta_col));
                } else {
                    s_shard[r] = gdn_dec_mul(g_val, s_shard[r]) + gdn_dec_mul(k_reg[r], delta_col);
                }
                attn_partial = fmaf(s_shard[r], q_reg[r], attn_partial);
            }

            const float attn_col = gdn_diet_sum16(attn_partial);

            if (lane_in_col == 0) {
                out[h*S + col] = attn_col * scale;
            }

#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                st[r*lanes_per_col] = s_shard[r];
            }
        }

        // ---- the last block of the head applies the gated norm (passes 0 and 1, wave 0)
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            s_last = atomicAdd(&counters[h], 1u) == (unsigned int) (split - 1);
        }
        __syncthreads();
        if (!s_last || tid >= 64) {
            return;
        }
        __threadfence();

        const float * oh = out + h*S;
        const float4 xs = ((const float4 *) oh)[tid % 32]; // L1 was invalidated by the fence
        x0 = oh[tid];
        x1 = oh[tid + 64];
        z0 = z[h*S + tid];
        z1 = z[h*S + tid + 64];
        w0 = norm_w[tid];
        w1 = norm_w[tid + 64];
        float tmp = gdn_dec_sumsq4(xs);
        tmp = gdn_diet_sum32(tmp);
        sc = rsqrtf(tmp / S + eps);
        e  = h*S + tid;
        // wait for these loads here, not at the loop header (where pass -1 would wait for the state loads)
        asm volatile("" : "+v"(x0), "+v"(x1), "+v"(z0), "+v"(z1), "+v"(w0), "+v"(w1), "+v"(sc));
    }
}


// ---- N-token variants (MTP verification: n_tok = 1..GDN_NT_MAX consecutive tokens of one sequence, K rollback slots)
// Every token runs exactly the arithmetic of the single-token kernels above (same operations, same operand rounding,
// same reduction trees and lane mapping), with the conv window and the recurrent state carried from token to token:
// - conv: every (token, channel) pair runs gdn_dec_conv_silu over its window of [state | x_0 .. x_{n-1}] (staged in LDS);
// - alpha/beta: per token the mul_mat_vec_f<bf16, 128> ncols = 1 accumulation and tree of gdn_decode_prep_diet;
// - delta rule: the state shard stays in registers across the tokens, per token the step of gdn_decode_out_split_diet;
// - gated norm (+ q8_1 of every token into the flattened [H_v*S, n_tok] activation) as gdn_decode_out_split_diet.
// Snapshots: slot s (row dst_row + s*slot_stride) gets the recurrent state after token n-1-s for s < min(n, K) and the
// conv window ending at token n-1-s for s < min(n+1, K) (slot n = the conv state before the first token), as
// build_recurrent_attn / build_conv_state write them. In every block all state reads (row s_copy[0]) come before a block
// barrier that precedes its state writes, and a block reads and writes only its own channels / state columns (a thread
// only its own state shard), so a slot row may be the source row.
#define GDN_NT_MAX 16 // tokens per launch
#define GDN_NT_NCB 1  // alpha/beta: tokens per block (1: the code of a block is the size of the single-token one)

// Conv of NC channels for the n_tok tokens, all (token, channel) pairs in parallel: the window of token t is rows
// t .. t+3 of win[3 + n_tok][NC] (rows 0..2: the conv state read from the source row, row 3 + t: qkv of token t), so every
// token runs exactly gdn_dec_conv_silu over its window. Slot s gets rows n_tok-s .. n_tok-s+2 (slot n_tok: the old
// state); all reads of the state rows happen in the staging, before the block barrier that precedes these writes.
template <int NC>
static __device__ __forceinline__ void gdn_nt_conv_pairs(const float * win, const float * cw, float * y,
        float * cs_ch0 /* conv state row dst_row, channel 0 of the block */, const int64_t cs_slot, const int K,
        const int n_tok, const int tid, const int nthreads) {
#pragma unroll 1
    for (int p = tid; p < n_tok*NC; p += nthreads) {
        const int t = p / NC;
        const int c = p % NC;
        float xc[GDN_DEC_DCONV];
        float wc[GDN_DEC_DCONV];
#pragma unroll
        for (int j = 0; j < GDN_DEC_DCONV; j++) {
            xc[j] = win[(t + j)*NC + c];
            wc[j] = cw[c*GDN_DEC_DCONV + j];
        }
        y[t*NC + c] = gdn_dec_conv_silu(xc, wc);
    }
    const int Kw = min(n_tok + 1, K);
#pragma unroll 1
    for (int p = tid; p < Kw*NC; p += nthreads) {
        const int s = p / NC;
        const int c = p % NC;
        float * d = cs_ch0 + (int64_t) s*cs_slot + c*(GDN_DEC_DCONV - 1);
#pragma unroll
        for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
            d[j] = win[(n_tok - s + j)*NC + c];
        }
    }
}

// stage the conv inputs of channel ch0 + c (thread c < NC: state, thread NC + c < 2*NC: weights)
template <int NC>
static __device__ __forceinline__ void gdn_nt_conv_stage(float * win, float * cw, const float * __restrict__ conv_w,
        const float * cs_ch0, const int64_t src_off, const int ch0, const int tid) {
    if (tid < NC) {
        const float * cs = cs_ch0 + tid*(GDN_DEC_DCONV - 1);
        float xc[GDN_DEC_DCONV - 1];
#pragma unroll
        for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
            xc[j] = cs[j]; // speculative: read row == written row
        }
        if (src_off != 0) {
#pragma unroll
            for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
                xc[j] = cs[src_off + j];
            }
        }
#pragma unroll
        for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
            win[j*NC + tid] = xc[j];
        }
    } else if (tid < 2*NC) {
        const int c = tid - NC;
        const float4 w4 = *(const float4 *) (conv_w + (int64_t) (ch0 + c)*GDN_DEC_DCONV);
        cw[c*GDN_DEC_DCONV + 0] = w4.x;
        cw[c*GDN_DEC_DCONV + 1] = w4.y;
        cw[c*GDN_DEC_DCONV + 2] = w4.z;
        cw[c*GDN_DEC_DCONV + 3] = w4.w;
    }
}

// grid: 2*H_k blocks (q/k head: conv of the n_tok tokens, L2 norms) + H_v*ceil(n_tok/GDN_NT_NCB) blocks (alpha/beta of
// head h for up to GDN_NT_NCB tokens, gates); out column t = [l2norm(q) | l2norm(k) | exp(gate) | sigmoid(beta) | scratch]
template <typename Tw>
static __global__ void __launch_bounds__(2*GDN_DEC_MMV_BLOCK)
gdn_decode_prep_nt(
        const float   * __restrict__ qkv,
        const float   * __restrict__ x,
        const Tw      * __restrict__ w_alpha,
        const Tw      * __restrict__ w_beta,
        const float   * __restrict__ dt_bias,
        const float   * __restrict__ a,
        const float   * __restrict__ conv_w,
        float         *              cs_dst,   // conv state row dst_row
        const int32_t * __restrict__ s_copy,
        float         * __restrict__ out,
        const int                    H_k,
        const int                    H_v,
        const int                    n_tok,
        const int                    K,
        const int                    ncols2,
        const int64_t                w_row_stride,
        const int64_t                cs_row_stride,
        const int64_t                cs_slot,  // floats between conv state slots
        const int                    dst_row,
        const float                  eps_l2,
        const float                  scale_l2) {
    constexpr int S = GDN_DEC_S;
    const int P        = 2*H_k*S + 3*H_v;
    const int conv_dim = (2*H_k + H_v)*S;

    const int b   = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float s_y[GDN_NT_MAX][S];
    __shared__ float s_win[GDN_NT_MAX + GDN_DEC_DCONV - 1][S];
    __shared__ float s_cw[S*GDN_DEC_DCONV];
    __shared__ float s_red[GDN_NT_NCB][2*GDN_DEC_MMV_BLOCK/64];

    if (b < 2*H_k) {
        const int32_t src_row = s_copy[0];
        float * cs_ch0 = cs_dst + (int64_t) b*S*(GDN_DEC_DCONV - 1);
        gdn_nt_conv_stage<S>(&s_win[0][0], s_cw, conv_w, cs_ch0, (int64_t) (src_row - dst_row)*cs_row_stride, b*S, tid);
        // qkv of the head for every token: element i = t*S + channel, 4 per thread and iteration (loads in flight
        // together, clamped index: unconditional)
#pragma unroll 1
        for (int i0 = 0; i0 < n_tok*S; i0 += 4*2*GDN_DEC_MMV_BLOCK) {
            float qv[4];
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int i = min(i0 + tid + u*2*GDN_DEC_MMV_BLOCK, n_tok*S - 1);
                qv[u] = qkv[(int64_t) (i / S)*conv_dim + b*S + i % S];
            }
#pragma unroll
            for (int u = 0; u < 4; u++) {
                const int i = i0 + tid + u*2*GDN_DEC_MMV_BLOCK;
                if (i < n_tok*S) {
                    s_win[GDN_DEC_DCONV - 1 + i / S][i % S] = qv[u];
                }
            }
        }
        __syncthreads();
        gdn_nt_conv_pairs<S>(&s_win[0][0], s_cw, &s_y[0][0], cs_ch0, cs_slot, K, n_tok, tid, 2*GDN_DEC_MMV_BLOCK);
        __syncthreads();
        // L2 norm of token t on the half wave t % 8 (== the tid < 32 code of gdn_decode_prep_diet)
        const int l = tid % 32;
        for (int t = tid / 32; t < n_tok; t += 2*GDN_DEC_MMV_BLOCK/32) {
            const float4 xs = ((const float4 *) s_y[t])[l];
            float tmp = gdn_dec_sumsq4(xs);
            tmp = gdn_diet_sum32(tmp);
            const float scale = rsqrtf(tmp / S + eps_l2);
            float4 r = make_float4(scale*xs.x, scale*xs.y, scale*xs.z, scale*xs.w);
            r.x *= scale_l2;
            r.y *= scale_l2;
            r.z *= scale_l2;
            r.w *= scale_l2;
            ((float4 *) (out + (int64_t) t*P + b*S))[l] = r;
        }
        return;
    }

    // alpha/beta projections of v head h for the tokens c0 .. c0 + nc - 1 (threads [0, 128) -> alpha row h,
    // [128, 256) -> beta row h); per token the accumulation order and the reduction tree of gdn_decode_prep_f32/_diet
    const int bb   = b - 2*H_k;
    const int h    = bb % H_v;
    const int c0   = (bb / H_v)*GDN_NT_NCB;
    const int nc   = min(GDN_NT_NCB, n_tok - c0);
    const int half = __builtin_amdgcn_readfirstlane(tid) / GDN_DEC_MMV_BLOCK;
    const int t    = tid % GDN_DEC_MMV_BLOCK;
    const Tw     * w  = (half == 0 ? w_alpha : w_beta) + h*w_row_stride;
    const float2 * x2 = (const float2 *) x;

    constexpr int nb = 20;
    float sumf[GDN_NT_NCB];
#pragma unroll
    for (int c = 0; c < GDN_NT_NCB; c++) {
        sumf[c] = 0.0f;
    }
    int col2 = t;
    for (; col2 + (nb - 1)*GDN_DEC_MMV_BLOCK < ncols2; col2 += nb*GDN_DEC_MMV_BLOCK) {
        float2 wv[nb];
        float2 xv[GDN_NT_NCB][nb];
#pragma unroll
        for (int u = 0; u < nb; u++) {
            wv[u] = gdn_dec_load_w2<Tw>(w, col2 + u*GDN_DEC_MMV_BLOCK);
        }
#pragma unroll
        for (int c = 0; c < GDN_NT_NCB; c++) {
            const float2 * xc = x2 + (int64_t) (c0 + min(c, nc - 1))*ncols2; // clamped: unconditional loads
#pragma unroll
            for (int u = 0; u < nb; u++) {
                xv[c][u] = xc[col2 + u*GDN_DEC_MMV_BLOCK];
            }
        }
#pragma unroll
        for (int c = 0; c < GDN_NT_NCB; c++) {
#pragma unroll
            for (int u = 0; u < nb; u++) {
                sumf[c] = sumf[c] + gdn_dec_mul(wv[u].x, xv[c][u].x);
                sumf[c] = sumf[c] + gdn_dec_mul(wv[u].y, xv[c][u].y);
            }
        }
    }
    for (; col2 < ncols2; col2 += GDN_DEC_MMV_BLOCK) {
        const float2 wv = gdn_dec_load_w2<Tw>(w, col2);
#pragma unroll
        for (int c = 0; c < GDN_NT_NCB; c++) {
            const float2 xv = x2[(int64_t) (c0 + min(c, nc - 1))*ncols2 + col2];
            sumf[c] = sumf[c] + gdn_dec_mul(wv.x, xv.x);
            sumf[c] = sumf[c] + gdn_dec_mul(wv.y, xv.y);
        }
    }

#pragma unroll
    for (int c = 0; c < GDN_NT_NCB; c++) {
        const float v = gdn_diet_sum64(sumf[c], tid);
        if (tid % 64 == 0) {
            s_red[c][tid / 64] = v;
        }
    }
    __syncthreads();

    if (tid < nc) {
        const int c = tid;
        const float alpha = (s_red[c][0] + s_red[c][1]) + dt_bias[h];
        const float gate  = gdn_dec_softplus(alpha) * a[h];
        float * o = out + (int64_t) (c0 + c)*P;
        o[2*H_k*S + h]       = expf(gate);
        o[2*H_k*S + H_v + h] = gdn_dec_sigmoid(s_red[c][2] + s_red[c][3]);
    }
    if (c0 == 0 && tid == 0) {
        // arrival counter of head h for gdn_decode_out_nt (scratch tail of column 0)
        ((unsigned int *) (out + 2*H_k*S + 2*H_v))[h] = 0u;
    }
}

// dynamic LDS of gdn_decode_out_nt (floats): q|k [n][2*S], g/beta [n][2], v [n][16], conv window [n + 3][16], weights
static size_t gdn_nt_out_lds_bytes(const int n_tok) {
    return sizeof(float)*((size_t) n_tok*(2*GDN_DEC_S + 2 + 16) + (size_t) (n_tok + GDN_DEC_DCONV - 1)*16 + 16*GDN_DEC_DCONV);
}

// H_v*8 blocks of 256 threads (16 lanes per state column, 16 columns per block, like gdn_decode_out_split_diet)
template <bool q8, int HK, int HV> // HK, HV > 0: compile-time head counts (smaller code), 0: the runtime values
static __global__ void __launch_bounds__(GDN_DEC_SPLIT_BLOCK)
gdn_decode_out_nt(
        const float   * __restrict__ prep,
        const float   * __restrict__ qkv,
        const float   * __restrict__ conv_w,
        float         *              cs_dst,  // conv state row dst_row
        const float   * __restrict__ z,
        const float   * __restrict__ norm_w,
        float         *              ss_dst,  // recurrent state row dst_row
        const int32_t * __restrict__ s_copy,
        float         *              out,
        const int                    H_k_rt,
        const int                    H_v_rt,
        const int                    n_tok,
        const int                    K,
        const int64_t                cs_row_stride,
        const int64_t                ss_row_stride,
        const int64_t                cs_slot,
        const int64_t                ss_slot,
        const int                    dst_row,
        const float                  scale,
        const float                  eps,
        block_q8_1    * __restrict__ y,
        int           * __restrict__ ysum) {
    constexpr int S              = GDN_DEC_S;
    constexpr int lanes_per_col  = 16;
    constexpr int rows_per_lane  = S / lanes_per_col;
    constexpr int cols_per_block = GDN_DEC_SPLIT_BLOCK / lanes_per_col;
    constexpr int split          = S / cols_per_block;

    const int H_k      = HK > 0 ? HK : H_k_rt;
    const int H_v      = HV > 0 ? HV : H_v_rt;
    const int P        = 2*H_k*S + 3*H_v;
    const int HVS      = H_v*S;
    const int conv_dim = (2*H_k + H_v)*S;

    const int h           = blockIdx.x / split;
    const int part        = blockIdx.x % split;
    const int tid         = threadIdx.x;
    const int lane_in_col = tid % lanes_per_col;
    const int colb        = tid / lanes_per_col;
    const int col         = part*cols_per_block + colb;
    const int hk          = h % H_k;

    extern __shared__ float gdn_nt_lds[];
    float * s_qk  = gdn_nt_lds;                          // [n][2*S]: q | k of head hk
    float * s_gb  = s_qk + n_tok*2*S;                    // [n][2]: exp(gate), sigmoid(beta) of head h
    float * s_v   = s_gb + n_tok*2;                      // [n][16]: conv output of the block's v channels
    float * s_win = s_v + n_tok*cols_per_block;          // [n + 3][16]: conv windows
    float * s_cw  = s_win + (n_tok + GDN_DEC_DCONV - 1)*cols_per_block; // [16][4]
    __shared__ int s_last;

    unsigned int * counters = (unsigned int *) (prep + 2*H_k*S + 2*H_v);

    // ---- issue all loads first (registers), then the LDS stores: the state shard (stored transposed: column col is
    // contiguous), q/k of the first 4 tokens (element tid of the 2*S q/k values per token, clamped token index:
    // unconditional), g/beta, the conv inputs of the block's v channels (state: thread c < 16, weights: thread 16 + c,
    // qkv of token colb: thread (colb, lane))
    float * st = ss_dst + (unsigned) h*S*S + col*S + lane_in_col;

    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = st[r*lanes_per_col]; // speculative: read row == written row
    }

    static_assert(2*S == GDN_DEC_SPLIT_BLOCK, "one q/k element per thread and token");
    const float * pq = prep + (tid < S ? hk*S + tid : H_k*S + hk*S + tid - S);
    float qk[4];
#pragma unroll
    for (int u = 0; u < 4; u++) {
        qk[u] = pq[(unsigned) min(u, n_tok - 1)*P];
    }
    const float gb = prep[(unsigned) min(tid / 2, n_tok - 1)*P + 2*H_k*S + (tid % 2)*H_v + h];

    const int32_t src_row = s_copy[0];
    const int ch0 = 2*H_k*S + h*S + part*cols_per_block;
    float * cs_ch0 = cs_dst + (unsigned) ch0*(GDN_DEC_DCONV - 1);
    const float * cs_t = cs_ch0 + min(tid, cols_per_block - 1)*(GDN_DEC_DCONV - 1);
    float xc[GDN_DEC_DCONV - 1];
#pragma unroll
    for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
        xc[j] = cs_t[j]; // speculative: read row == written row
    }
    const float4 w4 = *(const float4 *) (conv_w + (unsigned) (ch0 + tid % cols_per_block)*GDN_DEC_DCONV);
    const float  xq = qkv[(unsigned) min(colb, n_tok - 1)*conv_dim + ch0 + lane_in_col];

    if (src_row != dst_row) {
        const float * st_src = st + (int64_t) (src_row - dst_row)*ss_row_stride;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] = st_src[r*lanes_per_col];
        }
        const float * cs_src = cs_t + (int64_t) (src_row - dst_row)*cs_row_stride;
#pragma unroll
        for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
            xc[j] = cs_src[j];
        }
    }

#pragma unroll
    for (int u = 0; u < 4; u++) {
        if (u < n_tok) {
            s_qk[u*2*S + tid] = qk[u];
        }
    }
#pragma unroll 1
    for (int t0 = 4; t0 < n_tok; t0 += 4) { // tokens 4 .. n_tok-1: 4 per iteration, loads in flight together
        float qk4[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
            qk4[u] = pq[(unsigned) min(t0 + u, n_tok - 1)*P];
        }
#pragma unroll
        for (int u = 0; u < 4; u++) {
            if (t0 + u < n_tok) {
                s_qk[(t0 + u)*2*S + tid] = qk4[u];
            }
        }
    }
    if (tid < 2*n_tok) {
        s_gb[tid] = gb;
    }
    if (tid < cols_per_block) {
#pragma unroll
        for (int j = 0; j < GDN_DEC_DCONV - 1; j++) {
            s_win[j*cols_per_block + tid] = xc[j];
        }
    } else if (tid < 2*cols_per_block) {
        const int c = tid - cols_per_block;
        s_cw[c*GDN_DEC_DCONV + 0] = w4.x;
        s_cw[c*GDN_DEC_DCONV + 1] = w4.y;
        s_cw[c*GDN_DEC_DCONV + 2] = w4.z;
        s_cw[c*GDN_DEC_DCONV + 3] = w4.w;
    }
    if (colb < n_tok) {
        s_win[(GDN_DEC_DCONV - 1 + colb)*cols_per_block + lane_in_col] = xq;
    }
    __syncthreads();
    gdn_nt_conv_pairs<cols_per_block>(s_win, s_cw, s_v, cs_ch0, cs_slot, K, n_tok, tid, GDN_DEC_SPLIT_BLOCK);
    __syncthreads();

    // ---- delta rule, token by token (step of gdn_decode_out_split_diet)
#pragma unroll 1
    for (int t = 0; t < n_tok; t++) {
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            q_reg[r] = s_qk[t*2*S + r*lanes_per_col + lane_in_col];
            k_reg[r] = s_qk[t*2*S + S + r*lanes_per_col + lane_in_col];
        }
        const float g_val    = s_gb[2*t + 0];
        const float beta_val = s_gb[2*t + 1];

        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            if (r < rows_per_lane/2) {
                kv_shard = fmaf(s_shard[r], k_reg[r], kv_shard);
            } else {
                kv_shard = kv_shard + gdn_dec_mul(s_shard[r], k_reg[r]);
            }
        }
        const float kv_col = gdn_diet_sum16(kv_shard);

        const float delta_col = beta_val * fmaf(-g_val, kv_col, s_v[t*cols_per_block + colb]);

        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            if (r < rows_per_lane - 2) {
                s_shard[r] = fmaf(g_val, s_shard[r], gdn_dec_mul(k_reg[r], delta_col));
            } else {
                s_shard[r] = gdn_dec_mul(g_val, s_shard[r]) + gdn_dec_mul(k_reg[r], delta_col);
            }
            attn_partial = fmaf(s_shard[r], q_reg[r], attn_partial);
        }

        const float attn_col = gdn_diet_sum16(attn_partial);

        if (lane_in_col == 0) {
            out[(unsigned) t*HVS + h*S + col] = attn_col * scale;
        }

        const int s = n_tok - 1 - t; // snapshot slot of the state after token t (slot 0 = the final state)
        if (s < K) {
            float * d = st + (int64_t) s*ss_slot;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                d[r*lanes_per_col] = s_shard[r];
            }
        }
    }

    // ---- the last block of the head applies the gated norm to every token (token t on wave t % 4)
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        s_last = atomicAdd(&counters[h], 1u) == (unsigned int) (split - 1);
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    __threadfence();

    // == the epilogue of gdn_decode_out_split_diet per token: both halves of the wave redo the 32-lane reduction of the
    // float4 sums of squares, lane l handles elements l and l + 64
    const int lane = tid % 64;
    for (int t = tid / 64; t < n_tok; t += GDN_DEC_SPLIT_BLOCK/64) {
        const float * oh = out + (unsigned) t*HVS + h*S;
        const float * zh = z   + (unsigned) t*HVS + h*S;
        const float4 xs = ((const float4 *) oh)[lane % 32]; // L1 was invalidated by the fence
        float x0 = oh[lane];
        float x1 = oh[lane + 64];
        float z0 = zh[lane];
        float z1 = zh[lane + 64];
        float w0 = norm_w[lane];
        float w1 = norm_w[lane + 64];
        float tmp = gdn_dec_sumsq4(xs);
        tmp = gdn_diet_sum32(tmp);
        const float sc = rsqrtf(tmp / S + eps);
        int e = t*HVS + h*S + lane;
#pragma unroll 1
        for (int pass = 0; pass < 2; pass++) {
            float r = sc*x0;
            r *= w0;
            const float o = ggml_cuda_op_silu_single(z0) * r;
            out[e] = o;
            if constexpr (q8) {
                const float  amax = gdn_diet_max32(fabsf(o));
                const float  d    = amax / 127.0f;
                const int8_t qv   = amax == 0.0f ? 0 : roundf(o / d);
                const int    sum  = gdn_diet_isum32(qv);
                const int    ib   = e / QK8_1;
                y[ib].qs[e % QK8_1] = qv;
                if (e % QK8_1 == 0) {
                    y[ib].ds  = make_half2(d, d * sum);
                    ysum[ib] = sum;
                }
            }
            x0 = x1;
            z0 = z1;
            w0 = w1;
            e += 64;
        }
    }
}

// GGML_GDN_DIET: 0 = the generic kernels, 1 (default) = diet kernels, 2 = + the warm-up pass of the prep gates,
// 3 = + the warm-up pass of the out epilogue (all bit-identical; 2 and 3 measured within noise of 1, experimental)
static int gdn_dec_diet_level() {
    static const int level = [] {
        const char * e = getenv("GGML_GDN_DIET");
        return e == nullptr ? 1 : atoi(e);
    }();
    return level;
}

static bool gdn_dec_diet_enabled() {
    return gdn_dec_diet_level() != 0;
}
#endif // GDN_DEC_DIET

#if GDN_DEC_DIET
// ---- N-token launchers + GGML_GDN_DECODE_N_VERIFY

static bool gdn_nt_is_multi(const ggml_tensor * dst) {
    const ggml_tensor * qkv = dst->src[dst->op == GGML_OP_GDN_DECODE_PREP ? 0 : 1];
    return qkv->ne[1] > 1 || ggml_get_op_params_i32(dst, 5) > 1;
}

// GGML_GDN_DECODE_N_VERIFY=n (debug): the first n N-token PREP/OUT pairs that run outside graph capture are recomputed
// token by token with the single-token ops (on copies of the source state rows) and compared bit for bit: prep columns,
// outputs, q8_1 blocks + sums, and every conv / recurrent state slot row the N-token ops wrote
struct gdn_nt_verify_state {
    int                 left    = -1;
    const ggml_tensor * prep    = nullptr; // PREP node whose source rows were stashed
    float             * cs_pre  = nullptr;
    float             * ss_pre  = nullptr;
    size_t              cs_size = 0;
    size_t              ss_size = 0;
    int64_t             n_calls = 0;
    int64_t             n_bad   = 0;
};
static gdn_nt_verify_state gdn_nt_verify;

static bool gdn_nt_verify_active(cudaStream_t stream) {
    if (gdn_nt_verify.left < 0) {
        const char * e = getenv("GGML_GDN_DECODE_N_VERIFY");
        gdn_nt_verify.left = e ? std::max(0, atoi(e)) : 0;
    }
    if (gdn_nt_verify.left == 0) {
        return false;
    }
    hipStreamCaptureStatus status; // (HIP-only code: GDN_DEC_DIET)
    CUDA_CHECK(hipStreamIsCapturing(stream, &status));
    return status == hipStreamCaptureStatusNone;
}

static void gdn_nt_prep_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * qkv     = dst->src[0];
    const ggml_tensor * x       = dst->src[1];
    const ggml_tensor * w_alpha = dst->src[2];
    const ggml_tensor * w_beta  = dst->src[3];
    const ggml_tensor * dt_bias = dst->src[4];
    const ggml_tensor * a       = dst->src[5];
    const ggml_tensor * conv_w  = dst->src[6];
    ggml_tensor       * cs      = dst->src[7];
    const ggml_tensor * s_copy  = dst->src[8];

    const int   dst_row     = ggml_get_op_params_i32(dst, 0);
    const int   H_k         = ggml_get_op_params_i32(dst, 1);
    const int   H_v         = ggml_get_op_params_i32(dst, 2);
    const float eps_l2      = ggml_get_op_params_f32(dst, 4);
    const int   K           = std::max(1, ggml_get_op_params_i32(dst, 5));
    const int   slot_stride = ggml_get_op_params_i32(dst, 6);
    const int   n_tok       = (int) qkv->ne[1];

    GGML_ASSERT(n_tok >= 1 && n_tok <= GDN_NT_MAX);

    const float   scale_l2      = 1.0f/sqrtf((float) GDN_DEC_S);
    const int     ncols2        = (int) (x->ne[0] / 2);
    const int64_t w_row_stride  = (int64_t) (w_alpha->nb[1] / ggml_element_size(w_alpha));
    const int64_t cs_row_stride = (int64_t) (cs->nb[1] / sizeof(float));
    cudaStream_t stream = ctx.stream();

    if (gdn_nt_verify_active(stream)) {
        // stash the source rows before anything is written
        int32_t src_row = 0;
        CUDA_CHECK(cudaMemcpyAsync(&src_row, s_copy->data, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const size_t cs_bytes = cs->nb[1];
        if (gdn_nt_verify.cs_size < cs_bytes) {
            if (gdn_nt_verify.cs_pre) {
                CUDA_CHECK(cudaFree(gdn_nt_verify.cs_pre));
            }
            CUDA_CHECK(cudaMalloc(&gdn_nt_verify.cs_pre, cs_bytes));
            gdn_nt_verify.cs_size = cs_bytes;
        }
        CUDA_CHECK(cudaMemcpyAsync(gdn_nt_verify.cs_pre, (const char *) cs->data + src_row*cs->nb[1], cs_bytes,
            cudaMemcpyDeviceToDevice, stream));
        gdn_nt_verify.prep = dst;
        // the recurrent state row is stashed when the matching OUT op arrives (PREP does not touch it)
    }

    const int ncg = (n_tok + GDN_NT_NCB - 1) / GDN_NT_NCB;
    const dim3 grid(2*H_k + H_v*ncg, 1, 1);
    const dim3 block(2*GDN_DEC_MMV_BLOCK, 1, 1);
    float * cs_dst = (float *) cs->data + (int64_t) dst_row*cs_row_stride;
    const int64_t cs_slot = (int64_t) slot_stride*cs_row_stride;
    if (w_alpha->type == GGML_TYPE_BF16) {
        gdn_decode_prep_nt<nv_bfloat16><<<grid, block, 0, stream>>>(
            (const float *) qkv->data, (const float *) x->data,
            (const nv_bfloat16 *) w_alpha->data, (const nv_bfloat16 *) w_beta->data,
            (const float *) dt_bias->data, (const float *) a->data, (const float *) conv_w->data,
            cs_dst, (const int32_t *) s_copy->data, (float *) dst->data,
            H_k, H_v, n_tok, K, ncols2, w_row_stride, cs_row_stride, cs_slot, dst_row, eps_l2, scale_l2);
    } else {
        GGML_ASSERT(w_alpha->type == GGML_TYPE_F32);
        gdn_decode_prep_nt<float><<<grid, block, 0, stream>>>(
            (const float *) qkv->data, (const float *) x->data,
            (const float *) w_alpha->data, (const float *) w_beta->data,
            (const float *) dt_bias->data, (const float *) a->data, (const float *) conv_w->data,
            cs_dst, (const int32_t *) s_copy->data, (float *) dst->data,
            H_k, H_v, n_tok, K, ncols2, w_row_stride, cs_row_stride, cs_slot, dst_row, eps_l2, scale_l2);
    }
    CUDA_CHECK(cudaGetLastError());
}

static void gdn_nt_out_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, block_q8_1 * y, int * ysum) {
    const ggml_tensor * prep   = dst->src[0];
    const ggml_tensor * qkv    = dst->src[1];
    const ggml_tensor * conv_w = dst->src[2];
    ggml_tensor       * cs     = dst->src[3];
    const ggml_tensor * z      = dst->src[4];
    const ggml_tensor * norm_w = dst->src[5];
    ggml_tensor       * ss     = dst->src[6];
    const ggml_tensor * s_copy = dst->src[7];

    const int   dst_row     = ggml_get_op_params_i32(dst, 0);
    const int   H_k         = ggml_get_op_params_i32(dst, 1);
    const int   H_v         = ggml_get_op_params_i32(dst, 2);
    const float eps         = ggml_get_op_params_f32(dst, 4);
    const int   K           = std::max(1, ggml_get_op_params_i32(dst, 5));
    const int   slot_stride = ggml_get_op_params_i32(dst, 6);
    const int   n_tok       = (int) qkv->ne[1];

    GGML_ASSERT(n_tok >= 1 && n_tok <= GDN_NT_MAX);

    const float   scale         = 1.0f / sqrtf((float) GDN_DEC_S);
    const int64_t cs_row_stride = (int64_t) (cs->nb[1] / sizeof(float));
    const int64_t ss_row_stride = (int64_t) (ss->nb[1] / sizeof(float));
    constexpr int split = GDN_DEC_S / (GDN_DEC_SPLIT_BLOCK / 16);
    cudaStream_t stream = ctx.stream();

    const bool verify = gdn_nt_verify_active(stream) && gdn_nt_verify.prep == prep;
    if (verify) {
        int32_t src_row = 0;
        CUDA_CHECK(cudaMemcpyAsync(&src_row, s_copy->data, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const size_t ss_bytes = ss->nb[1];
        if (gdn_nt_verify.ss_size < ss_bytes) {
            if (gdn_nt_verify.ss_pre) {
                CUDA_CHECK(cudaFree(gdn_nt_verify.ss_pre));
            }
            CUDA_CHECK(cudaMalloc(&gdn_nt_verify.ss_pre, ss_bytes));
            gdn_nt_verify.ss_size = ss_bytes;
        }
        CUDA_CHECK(cudaMemcpyAsync(gdn_nt_verify.ss_pre, (const char *) ss->data + src_row*ss->nb[1], ss_bytes,
            cudaMemcpyDeviceToDevice, stream));
    }

    float * cs_dst = (float *) cs->data + (int64_t) dst_row*cs_row_stride;
    float * ss_dst = (float *) ss->data + (int64_t) dst_row*ss_row_stride;
    const int64_t cs_slot = (int64_t) slot_stride*cs_row_stride;
    const int64_t ss_slot = (int64_t) slot_stride*ss_row_stride;
    auto launch = [&](auto q8_c) {
        constexpr bool Q8 = decltype(q8_c)::value;
        // the model shape of the diet kernels with compile-time head counts, any other shape with the runtime ones
        auto * kernel = H_k == GDN_DIET_HK && H_v == GDN_DIET_HV ? gdn_decode_out_nt<Q8, GDN_DIET_HK, GDN_DIET_HV> :
                                                                   gdn_decode_out_nt<Q8, 0, 0>;
        kernel<<<H_v*split, GDN_DEC_SPLIT_BLOCK, gdn_nt_out_lds_bytes(n_tok), stream>>>(
            (const float *) prep->data, (const float *) qkv->data, (const float *) conv_w->data, cs_dst,
            (const float *) z->data, (const float *) norm_w->data, ss_dst, (const int32_t *) s_copy->data,
            (float *) dst->data, H_k, H_v, n_tok, K, cs_row_stride, ss_row_stride, cs_slot, ss_slot, dst_row,
            scale, eps, y, ysum);
    };
    if (y != nullptr) {
        launch(std::true_type{});
    } else {
        launch(std::false_type{});
    }
    CUDA_CHECK(cudaGetLastError());

    if (!verify) {
        return;
    }
    gdn_nt_verify.prep = nullptr;
    gdn_nt_verify.left--;

    // ---- reference: the single-token ops token by token on scratch copies of the source rows (row 0 of 1-row states)
    const ggml_tensor * pdst = prep;
    const int64_t P        = prep->ne[0];
    const int64_t HVS      = dst->ne[0];
    const int64_t conv_dim = qkv->ne[0];
    const int     Kw       = std::min(n_tok + 1, K);
    const int     Ks       = std::min(n_tok, K);
    const int64_t nb8      = HVS*n_tok/QK8_1;

    ggml_cuda_pool_alloc<float>      cs_ref(ctx.pool(), cs_row_stride);
    ggml_cuda_pool_alloc<float>      ss_ref(ctx.pool(), ss_row_stride);
    ggml_cuda_pool_alloc<float>      cs_snap(ctx.pool(), cs_row_stride*Kw);
    ggml_cuda_pool_alloc<float>      ss_snap(ctx.pool(), ss_row_stride*Ks);
    ggml_cuda_pool_alloc<float>      prep_ref(ctx.pool(), P*n_tok);
    ggml_cuda_pool_alloc<float>      out_ref(ctx.pool(), HVS*n_tok);
    ggml_cuda_pool_alloc<block_q8_1> y_ref(ctx.pool(), y ? nb8 : 1);
    ggml_cuda_pool_alloc<int>        s_ref(ctx.pool(), y ? nb8 : 1);
    ggml_cuda_pool_alloc<int32_t>    zero_row(ctx.pool(), 1);
    CUDA_CHECK(cudaMemsetAsync(zero_row.get(), 0, sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemcpyAsync(cs_ref.get(), gdn_nt_verify.cs_pre, cs_row_stride*sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ss_ref.get(), gdn_nt_verify.ss_pre, ss_row_stride*sizeof(float), cudaMemcpyDeviceToDevice, stream));
    if (Kw > n_tok) {
        CUDA_CHECK(cudaMemcpyAsync(cs_snap.get() + (int64_t) n_tok*cs_row_stride, gdn_nt_verify.cs_pre,
            cs_row_stride*sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }

    // fake single-token tensors (copies of the real nodes with the data pointers / shapes of token t)
    ggml_tensor t_qkv = *qkv;
    ggml_tensor t_x   = *pdst->src[1];
    ggml_tensor t_z   = *z;
    ggml_tensor t_cs  = *cs;
    ggml_tensor t_ss  = *ss;
    ggml_tensor t_sc  = *s_copy;
    ggml_tensor t_prep = *pdst;
    ggml_tensor t_out  = *dst;
    for (ggml_tensor * t : { &t_qkv, &t_x, &t_z, &t_prep, &t_out }) {
        t->ne[1] = 1;
        t->nb[2] = t->nb[3] = t->nb[1];
    }
    t_cs.ne[1] = 1;
    t_ss.ne[1] = 1;
    t_cs.data  = cs_ref.get();
    t_ss.data  = ss_ref.get();
    t_sc.data  = zero_row.get();
    t_prep.src[0] = &t_qkv;
    t_prep.src[1] = &t_x;
    t_prep.src[7] = &t_cs;
    t_prep.src[8] = &t_sc;
    t_out.src[0]  = &t_prep;
    t_out.src[1]  = &t_qkv;
    t_out.src[3]  = &t_cs;
    t_out.src[4]  = &t_z;
    t_out.src[6]  = &t_ss;
    t_out.src[7]  = &t_sc;
    for (ggml_tensor * t : { &t_prep, &t_out }) {
        ggml_set_op_params_i32(t, 0, 0);
        ggml_set_op_params_i32(t, 5, 1);
        ggml_set_op_params_i32(t, 6, 0);
    }
    for (int t = 0; t < n_tok; ++t) {
        t_qkv.data  = (char *) qkv->data + (size_t) t*conv_dim*sizeof(float);
        t_x.data    = (char *) pdst->src[1]->data + (size_t) t*pdst->src[1]->nb[1];
        t_z.data    = (char *) z->data + (size_t) t*HVS*sizeof(float);
        t_prep.data = prep_ref.get() + t*P;
        t_out.data  = out_ref.get() + t*HVS;
        ggml_cuda_op_gdn_decode_prep(ctx, &t_prep);
        if (y) {
            ggml_cuda_op_gdn_decode_out_q8(ctx, &t_out, y_ref.get() + t*HVS/QK8_1, s_ref.get() + t*HVS/QK8_1);
        } else {
            ggml_cuda_op_gdn_decode_out(ctx, &t_out);
        }
        const int s = n_tok - 1 - t;
        if (s < Kw) {
            CUDA_CHECK(cudaMemcpyAsync(cs_snap.get() + (int64_t) s*cs_row_stride, cs_ref.get(),
                cs_row_stride*sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }
        if (s < Ks) {
            CUDA_CHECK(cudaMemcpyAsync(ss_snap.get() + (int64_t) s*ss_row_stride, ss_ref.get(),
                ss_row_stride*sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }
    }

    auto fetch = [&](const void * p, size_t bytes) {
        std::vector<uint8_t> v(bytes);
        CUDA_CHECK(cudaMemcpyAsync(v.data(), p, bytes, cudaMemcpyDeviceToHost, stream));
        return v;
    };
    auto count_f = [](const std::vector<uint8_t> & a, const std::vector<uint8_t> & b, int64_t off, int64_t n) {
        int64_t bad = 0;
        const uint32_t * pa = (const uint32_t *) a.data() + off;
        const uint32_t * pb = (const uint32_t *) b.data() + off;
        for (int64_t i = 0; i < n; ++i) {
            bad += pa[i] != pb[i];
        }
        return bad;
    };
    const auto h_prep  = fetch(prep->data, P*n_tok*sizeof(float));
    const auto h_prepr = fetch(prep_ref.get(), P*n_tok*sizeof(float));
    const auto h_out   = fetch(dst->data, HVS*n_tok*sizeof(float));
    const auto h_outr  = fetch(out_ref.get(), HVS*n_tok*sizeof(float));
    std::vector<std::vector<uint8_t>> h_cs, h_csr, h_ss, h_ssr;
    for (int s = 0; s < Kw; ++s) {
        h_cs.push_back(fetch(cs_dst + s*cs_slot, cs_row_stride*sizeof(float)));
        h_csr.push_back(fetch(cs_snap.get() + s*cs_row_stride, cs_row_stride*sizeof(float)));
    }
    for (int s = 0; s < Ks; ++s) {
        h_ss.push_back(fetch(ss_dst + s*ss_slot, ss_row_stride*sizeof(float)));
        h_ssr.push_back(fetch(ss_snap.get() + s*ss_row_stride, ss_row_stride*sizeof(float)));
    }
    std::vector<uint8_t> h_y, h_yr, h_s, h_sr;
    if (y) {
        h_y  = fetch(y, nb8*sizeof(block_q8_1));
        h_yr = fetch(y_ref.get(), nb8*sizeof(block_q8_1));
        h_s  = fetch(ysum, nb8*sizeof(int));
        h_sr = fetch(s_ref.get(), nb8*sizeof(int));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int64_t pv = 2*(int64_t) H_k*GDN_DEC_S + 2*H_v; // prep values without the scratch tail
    int64_t bad_prep = 0;
    for (int t = 0; t < n_tok; ++t) {
        bad_prep += count_f(h_prep, h_prepr, t*P, pv);
    }
    const int64_t bad_out = count_f(h_out, h_outr, 0, HVS*n_tok);
    int64_t bad_cs = 0;
    for (int s = 0; s < Kw; ++s) {
        bad_cs += count_f(h_cs[s], h_csr[s], 0, (int64_t) (cs->ne[0]));
    }
    int64_t bad_ss = 0;
    for (int s = 0; s < Ks; ++s) {
        bad_ss += count_f(h_ss[s], h_ssr[s], 0, (int64_t) (ss->ne[0]));
    }
    int64_t bad_q8 = 0;
    if (y) {
        // mismatching q8_1 blocks (quants + scales) or block sums
        for (int64_t i = 0; i < nb8; ++i) {
            bad_q8 += memcmp(h_y.data() + i*sizeof(block_q8_1), h_yr.data() + i*sizeof(block_q8_1), sizeof(block_q8_1)) != 0 ||
                      memcmp(h_s.data() + i*sizeof(int), h_sr.data() + i*sizeof(int), sizeof(int)) != 0;
        }
    }
    const bool bad = bad_prep || bad_out || bad_cs || bad_ss || bad_q8;
    gdn_nt_verify.n_calls++;
    gdn_nt_verify.n_bad += bad;
    GGML_LOG_WARN("[GDN_DECODE_N_VERIFY] call %lld n_tok=%d K=%d dst_row=%d q8=%d: mismatches prep=%lld out=%lld cs=%lld (%d slots) "
        "ss=%lld (%d slots) q8=%lld | total %lld calls, %lld bad\n",
        (long long) gdn_nt_verify.n_calls, n_tok, K, dst_row, y != nullptr, (long long) bad_prep, (long long) bad_out,
        (long long) bad_cs, Kw, (long long) bad_ss, Ks, (long long) bad_q8,
        (long long) gdn_nt_verify.n_calls, (long long) gdn_nt_verify.n_bad);
}
#endif // GDN_DEC_DIET


bool ggml_cuda_gdn_decode_supported(const ggml_tensor * op, int device) {
    const int S = ggml_get_op_params_i32(op, 3);
    if (S != GDN_DEC_S) {
        return false;
    }
    // N tokens or rollback slots: wave64 HIP devices only (the kernels use the DPP reductions of the diet kernels)
    {
        const ggml_tensor * qkv = op->src[op->op == GGML_OP_GDN_DECODE_PREP ? 0 : 1];
        if (qkv->ne[1] > 1 || ggml_get_op_params_i32(op, 5) > 1) {
#if GDN_DEC_DIET
            if (qkv->ne[1] > GDN_NT_MAX || ggml_cuda_info().devices[device].warp_size != 64) {
                return false;
            }
#else
            GGML_UNUSED(device);
            return false;
#endif // GDN_DEC_DIET
        }
    }
    if (op->op == GGML_OP_GDN_DECODE_PREP) {
        const ggml_tensor * x       = op->src[1];
        const ggml_tensor * w_alpha = op->src[2];
        const ggml_tensor * w_beta  = op->src[3];
        const ggml_tensor * conv_w  = op->src[6];
        if (conv_w->ne[0] != GDN_DEC_DCONV) {
            return false;
        }
        if (w_alpha->type != w_beta->type || (w_alpha->type != GGML_TYPE_BF16 && w_alpha->type != GGML_TYPE_F32)) {
            return false;
        }
        if (x->ne[0] % 2 != 0 || w_alpha->nb[1] % 4 != 0 || w_beta->nb[1] != w_alpha->nb[1]) {
            return false;
        }
        return true;
    }
    if (op->op == GGML_OP_GDN_DECODE_OUT) {
        const ggml_tensor * conv_w = op->src[2];
        return conv_w->ne[0] == GDN_DEC_DCONV;
    }
    return false;
}

void ggml_cuda_op_gdn_decode_prep(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * qkv     = dst->src[0];
    const ggml_tensor * x       = dst->src[1];
    const ggml_tensor * w_alpha = dst->src[2];
    const ggml_tensor * w_beta  = dst->src[3];
    const ggml_tensor * dt_bias = dst->src[4];
    const ggml_tensor * a       = dst->src[5];
    const ggml_tensor * conv_w  = dst->src[6];
    ggml_tensor       * cs      = dst->src[7];
    const ggml_tensor * s_copy  = dst->src[8];

    const int   dst_row = ggml_get_op_params_i32(dst, 0);
    const int   H_k     = ggml_get_op_params_i32(dst, 1);
    const int   H_v     = ggml_get_op_params_i32(dst, 2);
    const int   S       = ggml_get_op_params_i32(dst, 3);
    const float eps_l2  = ggml_get_op_params_f32(dst, 4);

    GGML_ASSERT(S == GDN_DEC_S && conv_w->ne[0] == GDN_DEC_DCONV);
    GGML_ASSERT(cs->nb[1] % sizeof(float) == 0);

#if GDN_DEC_DIET
    if (gdn_nt_is_multi(dst)) {
        gdn_nt_prep_launch(ctx, dst);
        return;
    }
#endif // GDN_DEC_DIET

    const float scale_l2 = 1.0f/sqrtf((float) S); // same value as build_gdn_l2_norm uses

    const int     ncols2        = (int) (x->ne[0] / 2);
    const int64_t w_row_stride  = (int64_t) (w_alpha->nb[1] / ggml_element_size(w_alpha));
    const int64_t cs_row_stride = (int64_t) (cs->nb[1] / sizeof(float));

    cudaStream_t stream = ctx.stream();

    const dim3 grid(2*H_k + H_v, 1, 1);
    const dim3 block(2*GDN_DEC_MMV_BLOCK, 1, 1);

#if GDN_DEC_DIET
    if (gdn_dec_diet_enabled() && H_k == GDN_DIET_HK && H_v == GDN_DIET_HV && w_alpha->type == GGML_TYPE_BF16 &&
            ncols2 == GDN_DIET_NCOL2 && w_row_stride == 2*GDN_DIET_NCOL2 &&
            ggml_cuda_info().devices[ctx.device].warp_size == 64) {
        static const bool early = [] {
            const char * e = getenv("GGML_GDN_PREP_EARLY");
            return e == nullptr || e[0] != '0';
        }();
        auto * kernel = gdn_dec_diet_level() == 1 ? (early ? gdn_decode_prep_diet<false, true> : gdn_decode_prep_diet<false>) :
                                                    gdn_decode_prep_diet<true>;
        kernel<<<grid, block, 0, stream>>>(
            (const float *) qkv->data, (const float *) x->data,
            (const nv_bfloat16 *) w_alpha->data, (const nv_bfloat16 *) w_beta->data,
            (const float *) dt_bias->data, (const float *) a->data, (const float *) conv_w->data,
            (float *) cs->data + (int64_t) dst_row*cs_row_stride, (const int32_t *) s_copy->data, (float *) dst->data,
            cs_row_stride, dst_row, eps_l2, scale_l2);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#endif // GDN_DEC_DIET

    if (w_alpha->type == GGML_TYPE_BF16) {
        gdn_decode_prep_f32<nv_bfloat16><<<grid, block, 0, stream>>>(
            (const float *) qkv->data, (const float *) x->data,
            (const nv_bfloat16 *) w_alpha->data, (const nv_bfloat16 *) w_beta->data,
            (const float *) dt_bias->data, (const float *) a->data, (const float *) conv_w->data,
            (float *) cs->data, (const int32_t *) s_copy->data, (float *) dst->data,
            H_k, H_v, ncols2, w_row_stride, cs_row_stride, dst_row, eps_l2, scale_l2);
    } else {
        GGML_ASSERT(w_alpha->type == GGML_TYPE_F32);
        gdn_decode_prep_f32<float><<<grid, block, 0, stream>>>(
            (const float *) qkv->data, (const float *) x->data,
            (const float *) w_alpha->data, (const float *) w_beta->data,
            (const float *) dt_bias->data, (const float *) a->data, (const float *) conv_w->data,
            (float *) cs->data, (const int32_t *) s_copy->data, (float *) dst->data,
            H_k, H_v, ncols2, w_row_stride, cs_row_stride, dst_row, eps_l2, scale_l2);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_gdn_decode_out(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gdn_decode_out_q8(ctx, dst, nullptr, nullptr);
}

void ggml_cuda_op_gdn_decode_out_q8(ggml_backend_cuda_context & ctx, ggml_tensor * dst, block_q8_1 * y, int * ysum) {
    const ggml_tensor * prep   = dst->src[0];
    const ggml_tensor * qkv    = dst->src[1];
    const ggml_tensor * conv_w = dst->src[2];
    ggml_tensor       * cs     = dst->src[3];
    const ggml_tensor * z      = dst->src[4];
    const ggml_tensor * norm_w = dst->src[5];
    ggml_tensor       * ss     = dst->src[6];
    const ggml_tensor * s_copy = dst->src[7];

    const int   dst_row = ggml_get_op_params_i32(dst, 0);
    const int   H_k     = ggml_get_op_params_i32(dst, 1);
    const int   H_v     = ggml_get_op_params_i32(dst, 2);
    const int   S       = ggml_get_op_params_i32(dst, 3);
    const float eps     = ggml_get_op_params_f32(dst, 4);

    GGML_ASSERT(S == GDN_DEC_S && conv_w->ne[0] == GDN_DEC_DCONV);
    GGML_ASSERT(cs->nb[1] % sizeof(float) == 0 && ss->nb[1] % sizeof(float) == 0);

#if GDN_DEC_DIET
    if (gdn_nt_is_multi(dst)) {
        GGML_ASSERT((y == nullptr) == (ysum == nullptr));
        gdn_nt_out_launch(ctx, dst, y, ysum);
        return;
    }
#endif // GDN_DEC_DIET

    const float scale = 1.0f / sqrtf((float) S); // same value as the gated_delta_net kernel uses

    cudaStream_t stream = ctx.stream();

    // GGML_GDN_DECODE_SPLIT=0: one block per v head instead of S/16 blocks per head (same results)
    static const bool use_split = [] {
        const char * e = getenv("GGML_GDN_DECODE_SPLIT");
        return e == nullptr || atoi(e) != 0;
    }();

    const int warp_size = std::min(ggml_cuda_info().devices[ctx.device].warp_size, S);
    const int split     = S / ((GDN_DEC_SPLIT_BLOCK / warp_size) * 4);

    GGML_ASSERT((y == nullptr) == (ysum == nullptr));

    // GGML_GDN_DECODE_Q8_SELFTEST=1 (debug, needs GGML_CUDA_DISABLE_GRAPHS=1): run the q8 variant into scratch buffers
    // and compare its q8_1 output with quantize_q8_1 of the fp32 result
    static const bool q8_selftest = [] {
        const char * e = getenv("GGML_GDN_DECODE_Q8_SELFTEST");
        return e != nullptr && atoi(e) != 0;
    }();
    const int64_t n_out = (int64_t) H_v*S;
    const int64_t nb8   = n_out/QK8_1;
    ggml_cuda_pool_alloc<block_q8_1> y_test(ctx.pool());
    ggml_cuda_pool_alloc<int>        s_test(ctx.pool());
    const bool selftest = q8_selftest && y == nullptr;
    if (selftest) {
        y    = y_test.alloc(nb8);
        ysum = s_test.alloc(nb8);
    }

    const bool q8 = y != nullptr;

    auto launch = [&](auto q8_c) {
        constexpr bool Q8 = decltype(q8_c)::value;
#if GDN_DEC_DIET
        if (use_split && gdn_dec_diet_enabled() && H_k == GDN_DIET_HK && H_v == GDN_DIET_HV && warp_size == 64) {
            const int64_t cs_row_stride = (int64_t) (cs->nb[1] / sizeof(float));
            const int64_t ss_row_stride = (int64_t) (ss->nb[1] / sizeof(float));
            auto * kernel = gdn_dec_diet_level() >= 3 ? gdn_decode_out_split_diet_warm<Q8> : gdn_decode_out_split_diet<Q8>;
            kernel<<<H_v*split, GDN_DEC_SPLIT_BLOCK, 0, stream>>>(
                (const float *) prep->data, (const float *) qkv->data, (const float *) conv_w->data,
                (float *) cs->data + (int64_t) dst_row*cs_row_stride, (const float *) z->data,
                (const float *) norm_w->data, (float *) ss->data + (int64_t) dst_row*ss_row_stride,
                (const int32_t *) s_copy->data, (float *) dst->data, cs_row_stride, ss_row_stride,
                dst_row, scale, eps, y, ysum);
            return;
        }
#endif // GDN_DEC_DIET
        if (use_split && split >= 1) {
            gdn_decode_out_split_f32<Q8><<<H_v*split, GDN_DEC_SPLIT_BLOCK, 0, stream>>>(
                (const float *) prep->data, (const float *) qkv->data, (const float *) conv_w->data, (float *) cs->data,
                (const float *) z->data, (const float *) norm_w->data, (float *) ss->data, (const int32_t *) s_copy->data,
                (float *) dst->data, H_k, H_v, (int64_t) (cs->nb[1] / sizeof(float)), (int64_t) (ss->nb[1] / sizeof(float)),
                dst_row, scale, eps, y, ysum);
        } else {
            gdn_decode_out_f32<Q8><<<H_v, GDN_DEC_OUT_BLOCK, 0, stream>>>(
                (const float *) prep->data, (const float *) qkv->data, (const float *) conv_w->data, (float *) cs->data,
                (const float *) z->data, (const float *) norm_w->data, (float *) ss->data, (const int32_t *) s_copy->data,
                (float *) dst->data, H_k, H_v, (int64_t) (cs->nb[1] / sizeof(float)), (int64_t) (ss->nb[1] / sizeof(float)),
                dst_row, scale, eps, y, ysum);
        }
    };
    if (q8) {
        launch(std::true_type{});
    } else {
        launch(std::false_type{});
    }
    CUDA_CHECK(cudaGetLastError());

    if (selftest) {
        ggml_cuda_pool_alloc<block_q8_1> y_ref(ctx.pool(), nb8);
        quantize_row_q8_1_cuda((const float *) dst->data, nullptr, y_ref.get(), GGML_TYPE_Q8_0,
            n_out, n_out, n_out, n_out, n_out, 1, 1, 1, stream);
        std::vector<block_q8_1> hf(nb8), hr(nb8);
        std::vector<int> hs(nb8);
        CUDA_CHECK(cudaMemcpyAsync(hf.data(), y, nb8*sizeof(block_q8_1), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(hr.data(), y_ref.get(), nb8*sizeof(block_q8_1), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(hs.data(), ysum, nb8*sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int64_t bad = 0;
        for (int64_t i = 0; i < nb8; ++i) {
            int sum = 0;
            for (int j = 0; j < QK8_1; ++j) {
                sum += hr[i].qs[j];
            }
            if (memcmp(&hf[i], &hr[i], sizeof(block_q8_1)) != 0 || hs[i] != sum) {
                bad++;
            }
        }
        static int64_t n_calls = 0;
        static int64_t n_bad   = 0;
        n_calls++;
        n_bad += bad;
        if (n_calls == 1 || n_calls % 480 == 0 || bad) {
            GGML_LOG_WARN("%s: q8 selftest: %lld calls, %lld mismatching blocks (last call: %lld of %lld)\n", __func__,
                (long long) n_calls, (long long) n_bad, (long long) bad, (long long) nb8);
        }
    }
}
