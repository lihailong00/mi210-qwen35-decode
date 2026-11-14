#include "gdn-decode.cuh"
#include "convert.cuh"
#include "unary.cuh"

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
        const float                  eps) {
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
        ((float4 *) (out + h*S))[tid] = o;
    }
}

// several blocks per v head (each owns cols_per_block state columns, like gated_delta_net_cuda): conv of the
// block's v channels, delta rule; the last block of a head to finish applies the gated rms norm to the head's
// output (per-head arrival counters: tail of the prep result, zeroed by the prep kernel)
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
        const float                  eps) {
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
        ((float4 *) (out + h*S))[tid] = o;
    }
}

bool ggml_cuda_gdn_decode_supported(const ggml_tensor * op) {
    const int S = ggml_get_op_params_i32(op, 3);
    if (S != GDN_DEC_S) {
        return false;
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

    const float scale_l2 = 1.0f/sqrtf((float) S); // same value as build_gdn_l2_norm uses

    const int     ncols2        = (int) (x->ne[0] / 2);
    const int64_t w_row_stride  = (int64_t) (w_alpha->nb[1] / ggml_element_size(w_alpha));
    const int64_t cs_row_stride = (int64_t) (cs->nb[1] / sizeof(float));

    cudaStream_t stream = ctx.stream();

    const dim3 grid(2*H_k + H_v, 1, 1);
    const dim3 block(2*GDN_DEC_MMV_BLOCK, 1, 1);

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

    const float scale = 1.0f / sqrtf((float) S); // same value as the gated_delta_net kernel uses

    cudaStream_t stream = ctx.stream();

    // GGML_GDN_DECODE_SPLIT=0: one block per v head instead of S/16 blocks per head (same results)
    static const bool use_split = [] {
        const char * e = getenv("GGML_GDN_DECODE_SPLIT");
        return e == nullptr || atoi(e) != 0;
    }();

    const int warp_size = std::min(ggml_cuda_info().devices[ctx.device].warp_size, S);
    const int split     = S / ((GDN_DEC_SPLIT_BLOCK / warp_size) * 4);

    if (use_split && split >= 1) {
        gdn_decode_out_split_f32<<<H_v*split, GDN_DEC_SPLIT_BLOCK, 0, stream>>>(
            (const float *) prep->data, (const float *) qkv->data, (const float *) conv_w->data, (float *) cs->data,
            (const float *) z->data, (const float *) norm_w->data, (float *) ss->data, (const int32_t *) s_copy->data,
            (float *) dst->data, H_k, H_v, (int64_t) (cs->nb[1] / sizeof(float)), (int64_t) (ss->nb[1] / sizeof(float)),
            dst_row, scale, eps);
    } else {
        gdn_decode_out_f32<<<H_v, GDN_DEC_OUT_BLOCK, 0, stream>>>(
            (const float *) prep->data, (const float *) qkv->data, (const float *) conv_w->data, (float *) cs->data,
            (const float *) z->data, (const float *) norm_w->data, (float *) ss->data, (const int32_t *) s_copy->data,
            (float *) dst->data, H_k, H_v, (int64_t) (cs->nb[1] / sizeof(float)), (int64_t) (ss->nb[1] / sizeof(float)),
            dst_row, scale, eps);
    }
    CUDA_CHECK(cudaGetLastError());
}
