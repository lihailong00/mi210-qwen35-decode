#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

#if defined(GGML_USE_HIP)
// Long-sequence conv through 64-channel x 64-token LDS tiles: rows of the channel-major input are read
// as contiguous runs and the token-major output is written 64 channels at a time. The per-element
// arithmetic and its order are the same as in ssm_conv_long_token_f32, so the output is identical; that
// kernel's 128x35 tiles read one scattered float per thread and reach ~0.4 TB/s on MI210.
// GGML_CUDA_SSM_CONV_TILE=0 restores it.
template <bool apply_silu, int d_conv>
static __global__ void __launch_bounds__(256) ssm_conv_tile_f32(
        const float * __restrict__ src0, const float * __restrict__ src1, const float * __restrict__ bias,
        const int64_t x_s1, const int64_t x_s2, const int64_t w_s1,
        float * __restrict__ dst, const int64_t y_s1, const int64_t y_s2,
        const int64_t n_t, const int64_t nr) {
    constexpr int TS   = 64;
    constexpr int NCOL = TS + d_conv - 1;
    __shared__ float smem[TS*NCOL];

    const int64_t c0 = (int64_t) blockIdx.y*TS;
    const int64_t t0 = (int64_t) blockIdx.z*TS;
    const float * x  = src0 + blockIdx.x*x_s2 + c0*x_s1 + t0;
    float *       y  = dst  + blockIdx.x*y_s2 + t0*y_s1 + c0;

    const int tid = threadIdx.x;
    const int lo  = tid % TS;
    const int hi  = tid / TS;

    const int64_t cols_left = n_t + d_conv - 1 - t0;
    const int     n_cols    = cols_left < NCOL ? (int) cols_left : NCOL;

    for (int idx = tid; idx < TS*NCOL; idx += 256) {
        const int c   = idx / NCOL;
        const int col = idx - c*NCOL;
        if (col < n_cols && c0 + c < nr) {
            smem[idx] = x[c*x_s1 + col];
        }
    }
    __syncthreads();

    if (c0 + lo >= nr) {
        return;
    }

    float w[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; j++) {
        w[j] = src1[(c0 + lo)*w_s1 + j];
    }
    const float b = bias != nullptr ? bias[c0 + lo] : 0.0f;

    for (int p = 0; p < TS/4; ++p) {
        const int t = p*4 + hi;
        if (t0 + t < n_t) {
            float sumf = 0.0f;
#pragma unroll
            for (int j = 0; j < d_conv; j++) {
                sumf += smem[lo*NCOL + t + j] * w[j];
            }
            sumf += b;
            y[t*y_s1 + lo] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
        }
    }
}

static bool ggml_cuda_ssm_conv_tile_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_SSM_CONV_TILE");
        return e == nullptr || atoi(e) != 0;
    }();
    return enabled;
}

// Conv-input fusion. The delta-net graphs build the conv input as concat(conv_state [d_conv-1, C], transposed
// token-major projection x [C, n_t]) -> channel-major [d_conv-1 + n_t, C], and SSM_CONV + SILU turns it back into a
// token-major [C, n_t] tensor. Element (c, t) of the result only needs x[t-3..t][c] (the state for t < 3), so the
// whole thing is a 4-row stencil over the token-major data of each sequence: one 16-byte load and one 16-byte
// store per 4 channels and token, no transposes and no LDS. The products are rounded and summed in the same order
// as in ssm_conv_tile_f32 (as compiled for gfx90a: separate multiplies and adds), so the output is identical.
// Measured on MI210 (C 10240, n_t 4096): 0.74 ms concat + 0.67 ms conv -> 0.02 ms + 0.37 ms.
//
// The concat output itself is not materialized. Only what is still read is written into its buffer (by
// conv_fuse_prep_f32, which runs in place of the CONCAT):
//   - columns [col0, ne0) of every row: the views that feed the conv state update,
//   - columns [0, 3): the old conv state (its own buffer may be reused before the SSM_CONV node runs),
//   - the 3 token rows in front of every chunk of L tokens ("halo"), packed into the otherwise unused columns
//     [4, col0): the stencil output may be allocated on top of x, and then the block of the previous chunk
//     overwrites these rows concurrently.
// The chunk length L trades latency against throughput: a thread walks its L tokens serially (about 0.56 us per
// token on MI210, so a 256-token chunk cannot finish in less than 145 us however small the batch is), while short
// chunks cost more halo rows and are slower once the GPU is saturated. About 16 chunks per sequence is the measured
// optimum from 256 to 4096 tokens (ggml_cuda_conv_fuse_chunk), e.g. conv kernels per layer, C 10240, in us:
//   n_t    separate   L=256   L=128   L=64   L=32
//   256      95        154      89     58     47
//   512     187        155      95     71     68
//   1024    367        169     119    113    133
//   2048    722        217     204    234    233
//   4096   1405        383     437    423    446
// GGML_CUDA_CONV_FUSE=0 restores the separate concat and conv kernels; GGML_CUDA_CONV_FUSE_CHUNK fixes L.
static __host__ __device__ __forceinline__ int conv_fuse_cap(const int64_t col0) {
    return (int) ((col0 - 8) & ~(int64_t) 3);
}

// position of element c (c % 4 == 0) of halo row r inside the concat buffer; 4 consecutive channels stay in
// one buffer row and start 16-byte aligned
static __device__ __forceinline__ int64_t conv_fuse_halo_idx(const int r, const int c, const int C, const int cap, const int64_t ne0) {
    const int     m   = r*C + c;
    const int     row = m / cap;
    const int     col = m - row*cap;
    const int64_t o   = (int64_t) row*ne0;
    return o + 4 + ((4 - (int) (o & 3)) & 3) + col;
}

static __global__ void __launch_bounds__(256) conv_fuse_prep_f32(
        const float * __restrict__ st, const float * __restrict__ x, float * __restrict__ cb,
        const int C, const int64_t n_t, const int64_t ne0, const int64_t col0, const int L) {
    const int c = (blockIdx.x*256 + threadIdx.x)*4;
    if (c >= C) {
        return;
    }
    st += (int64_t) blockIdx.z*3*C;
    x  += (int64_t) blockIdx.z*n_t*C;
    cb += (int64_t) blockIdx.z*ne0*C;
    if (blockIdx.y == 0) {
        const float4 s0 = *(const float4 *) (st + (int64_t) c*3);
        const float4 s1 = *(const float4 *) (st + (int64_t) c*3 + 4);
        const float4 s2 = *(const float4 *) (st + (int64_t) c*3 + 8);
        float * d = cb + (int64_t) c*ne0;
        d[0]       = s0.x; d[1]         = s0.y; d[2]         = s0.z;
        d[ne0]     = s0.w; d[ne0 + 1]   = s1.x; d[ne0 + 2]   = s1.y;
        d[2*ne0]   = s1.z; d[2*ne0 + 1] = s1.w; d[2*ne0 + 2] = s2.x;
        d[3*ne0]   = s2.y; d[3*ne0 + 1] = s2.z; d[3*ne0 + 2] = s2.w;
        for (int64_t col = col0; col < ne0; ++col) {
            const float4 v = *(const float4 *) (x + (col - 3)*C + c);
            d[col]         = v.x;
            d[ne0 + col]   = v.y;
            d[2*ne0 + col] = v.z;
            d[3*ne0 + col] = v.w;
        }
        return;
    }
    const int     r = blockIdx.y - 1;
    const int     q = 1 + r/3;
    const int64_t t = (int64_t) q*L - 3 + (r - (q - 1)*3);
    *(float4 *) (cb + conv_fuse_halo_idx(r, c, C, conv_fuse_cap(col0), ne0)) = *(const float4 *) (x + t*C + c);
}

static __device__ __forceinline__ float conv_fuse_tap4_silu(const float x0, const float x1, const float x2, const float x3, const float4 w) {
    // the compiler does not contract the products of the kernels above into fma; never do it here either
#pragma clang fp contract(off)
    float sumf = 0.0f;
    sumf += x0 * w.x;
    sumf += x1 * w.y;
    sumf += x2 * w.z;
    sumf += x3 * w.w;
    return ggml_cuda_op_silu_single(sumf);
}

// tokens t-3 .. t of 4 channels -> token t of the 4 channels; w<i> holds the 4 taps of channel i
static __device__ __forceinline__ float4 conv_fuse_tap4x4_silu(
        const float4 x0, const float4 x1, const float4 x2, const float4 x3,
        const float4 w0, const float4 w1, const float4 w2, const float4 w3) {
    return make_float4(
        conv_fuse_tap4_silu(x0.x, x1.x, x2.x, x3.x, w0),
        conv_fuse_tap4_silu(x0.y, x1.y, x2.y, x3.y, w1),
        conv_fuse_tap4_silu(x0.z, x1.z, x2.z, x3.z, w2),
        conv_fuse_tap4_silu(x0.w, x1.w, x2.w, x3.w, w3));
}

// x and y are either the same buffer or do not overlap
static __global__ void __launch_bounds__(256) ssm_conv4_silu_stream_f32(
        const float * x, const float * cb, const float * w, float * y,
        const int C, const int64_t n_t, const int64_t ne0, const int64_t col0, const int64_t w_s1, const int L) {
    const int c = (blockIdx.x*256 + threadIdx.x)*4;
    if (c >= C) {
        return;
    }
    const float4 w0 = *(const float4 *) (w + (int64_t) (c + 0)*w_s1);
    const float4 w1 = *(const float4 *) (w + (int64_t) (c + 1)*w_s1);
    const float4 w2 = *(const float4 *) (w + (int64_t) (c + 2)*w_s1);
    const float4 w3 = *(const float4 *) (w + (int64_t) (c + 3)*w_s1);
    x  += (int64_t) blockIdx.z*n_t*C;
    y  += (int64_t) blockIdx.z*n_t*C;
    cb += (int64_t) blockIdx.z*ne0*C;

    float4 a, b, d; // tokens t-3, t-2, t-1 of the 4 channels
    if (blockIdx.y == 0) {
        const float * s = cb + (int64_t) c*ne0;
        a = make_float4(s[0], s[ne0],     s[2*ne0],     s[3*ne0]);
        b = make_float4(s[1], s[ne0 + 1], s[2*ne0 + 1], s[3*ne0 + 1]);
        d = make_float4(s[2], s[ne0 + 2], s[2*ne0 + 2], s[3*ne0 + 2]);
    } else {
        const int r   = (blockIdx.y - 1)*3;
        const int cap = conv_fuse_cap(col0);
        a = *(const float4 *) (cb + conv_fuse_halo_idx(r,     c, C, cap, ne0));
        b = *(const float4 *) (cb + conv_fuse_halo_idx(r + 1, c, C, cap, ne0));
        d = *(const float4 *) (cb + conv_fuse_halo_idx(r + 2, c, C, cap, ne0));
    }

    const int64_t t0 = (int64_t) blockIdx.y*L;
    const int64_t t1 = t0 + L < n_t ? t0 + L : n_t;
    x += t0*C + c;
    y += t0*C + c;
    int64_t t = t0;
    // 4 tokens per step: the loads are issued together (x and y may be the same buffer, so they have to
    // come before the stores explicitly)
    for (; t + 4 <= t1; t += 4) {
        const float4 e0 = *(const float4 *) x;
        const float4 e1 = *(const float4 *) (x + C);
        const float4 e2 = *(const float4 *) (x + 2*(int64_t) C);
        const float4 e3 = *(const float4 *) (x + 3*(int64_t) C);
        const float4 o0 = conv_fuse_tap4x4_silu(a,  b,  d,  e0, w0, w1, w2, w3);
        const float4 o1 = conv_fuse_tap4x4_silu(b,  d,  e0, e1, w0, w1, w2, w3);
        const float4 o2 = conv_fuse_tap4x4_silu(d,  e0, e1, e2, w0, w1, w2, w3);
        const float4 o3 = conv_fuse_tap4x4_silu(e0, e1, e2, e3, w0, w1, w2, w3);
        *(float4 *) y                   = o0;
        *(float4 *) (y + C)             = o1;
        *(float4 *) (y + 2*(int64_t) C) = o2;
        *(float4 *) (y + 3*(int64_t) C) = o3;
        a = e1;
        b = e2;
        d = e3;
        x += 4*(int64_t) C;
        y += 4*(int64_t) C;
    }
    for (; t < t1; ++t) {
        const float4 e = *(const float4 *) x;
        *(float4 *) y = conv_fuse_tap4x4_silu(a, b, d, e, w0, w1, w2, w3);
        a = b;
        b = d;
        d = e;
        x += C;
        y += C;
    }
}

bool ggml_cuda_conv_fuse_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_CONV_FUSE");
        return e == nullptr || atoi(e) != 0;
    }();
    return enabled;
}

// tokens per chunk for sequences of n_t tokens: n_t/16 rounded up to a multiple of 32, at most 256 (see above)
static int ggml_cuda_conv_fuse_chunk(const int64_t n_t) {
    static const int chunk_env = [] {
        const char * e = getenv("GGML_CUDA_CONV_FUSE_CHUNK");
        const int    v = e == nullptr ? 0 : atoi(e);
        return v >= 16 ? v : 0;
    }();
    if (chunk_env != 0) {
        return chunk_env;
    }
    return (int) std::min<int64_t>(256, std::max<int64_t>(32, (n_t + 511)/512*32));
}

bool ggml_cuda_conv_fuse_geom_ok(const int64_t n_t, const int64_t C, const int64_t col0) {
    const int64_t L   = ggml_cuda_conv_fuse_chunk(n_t);
    const int64_t nq  = (n_t + L - 1)/L;
    const int64_t cap = conv_fuse_cap(col0);
    // the prep kernel uses one grid row for the state / trailing columns and one per halo row
    if (cap < 4 || C % 4 != 0 || nq*3*C >= INT32_MAX || 1 + (nq - 1)*3 > 65535) {
        return false;
    }
    return ((nq - 1)*3*C + cap - 1)/cap <= C;
}

void ggml_cuda_conv_fuse_prep(ggml_backend_cuda_context & ctx, const ggml_tensor * concat) {
    const ggml_tensor * st = concat->src[0];
    const ggml_tensor * x  = concat->src[1];
    const int64_t n_t = x->ne[0];
    const int64_t C   = concat->ne[1];
    const int     L   = ggml_cuda_conv_fuse_chunk(n_t);
    const int64_t nq  = (n_t + L - 1)/L;
    const dim3 grid((C/4 + 255)/256, 1 + (nq - 1)*3, concat->ne[2]);
    conv_fuse_prep_f32<<<grid, 256, 0, ctx.stream()>>>(
        (const float *) st->data, (const float *) x->data, (float *) concat->data,
        (int) C, n_t, concat->ne[0], ctx.conv_fuse_col0, L);
}
#endif // defined(GGML_USE_HIP)

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
#if defined(GGML_USE_HIP)
            if (n_t >= 64 && nr % 64 == 0 && dst_nb0 == (int) sizeof(float) && ggml_cuda_ssm_conv_tile_enabled()) {
                const dim3 blocks(n_s, nr/64, (n_t + 63)/64);
                ssm_conv_tile_f32<apply_silu, kNC><<<blocks, 256, 0, stream>>>(
                    src0, src1, bias, src0_nb1/(int) sizeof(float), src0_nb2/(int) sizeof(float), src1_nb1/(int) sizeof(float),
                    dst, dst_nb1/(int) sizeof(float), dst_nb2/(int) sizeof(float), n_t, nr);
                return;
            }
#endif // defined(GGML_USE_HIP)
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 7:  launch_kernel(std::integral_constant<int, 7 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 7, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

#if defined(GGML_USE_HIP)
    if (ctx.conv_fuse_concat == src0) {
        // planned in ggml_cuda_plan_conv_fuse(): src0 was not materialized, read its sources instead. The plan
        // guarantees that this node is reached through the SSM_CONV + SILU fusion.
        GGML_ASSERT(fuse_silu && !fuse_bias && nc == 4);
        const int L = ggml_cuda_conv_fuse_chunk(n_t);
        const dim3 grid((nr/4 + 255)/256, (n_t + L - 1)/L, n_s);
        ssm_conv4_silu_stream_f32<<<grid, 256, 0, ctx.stream()>>>(
            (const float *) src0->src[1]->data, (const float *) src0->data, (const float *) src1->data, (float *) out->data,
            (int) nr, n_t, src0->ne[0], ctx.conv_fuse_col0, src1->nb[1]/sizeof(float), L);
        ctx.conv_fuse_concat = nullptr;
        return;
    }
#endif // defined(GGML_USE_HIP)

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}
