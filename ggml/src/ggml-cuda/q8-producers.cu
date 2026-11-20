// Single-row (decode) variants of the kernels that produce GEMV activations, which additionally write the q8_1
// quantization of their output (bit-identical to quantize_q8_1, + per-block int sums) into the per-stream activation
// cache of mmvq.cu. The following single-token GEMVs that consume the tensor then skip their quantize launch.
// The fp32 output is computed with exactly the same arithmetic as the regular kernels.
#include "common.cuh"
#include "unary.cuh"
#include "convert.cuh"
#include "mmvq-fused-host.cuh"

#include <vector>

#if defined(GGML_USE_HIP)

// (int8_t) roundf(v / d) without the IEEE division in the common case (bit-exact, see mmvq-fused.cuh q8_round_div)
static __device__ __forceinline__ int q8p_round_div(const float v, const float d, const float inv_d) {
    const float t  = v * inv_d;
    const float fr = __builtin_amdgcn_fractf(fabsf(t));
    if (fabsf(fr - 0.5f) > 3.0e-5f) {
        return (int) rintf(t);
    }
    return (int) roundf(v / d);
}

// reductions over aligned groups of 8 lanes with DPP (no LDS round trip): quad_perm xor1, xor2, then row_half_mirror
static __device__ __forceinline__ float q8p_max8(float v) {
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0xB1, 0xF, 0xF, false)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x4E, 0xF, 0xF, false)));
    v = fmaxf(v, __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x141, 0xF, 0xF, false)));
    return v;
}
static __device__ __forceinline__ int q8p_sum8(int v) {
    v += __builtin_amdgcn_update_dpp(0, v, 0xB1, 0xF, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x4E, 0xF, 0xF, false);
    v += __builtin_amdgcn_update_dpp(0, v, 0x141, 0xF, 0xF, false);
    return v;
}

// warp_reduce_sum<32> with the same butterfly (xor 16, 8, 4, 2, 1; float addition is commutative, so every lane gets
// the bit-identical total) but with ds_swizzle / DPP instead of ds_bpermute: no lane-address arithmetic, less code
static __device__ __forceinline__ float q8p_sum32(float v) {
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F)); // xor 16
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x201F)); // xor 8
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x101F)); // xor 4
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x4E, 0xF, 0xF, false)); // xor 2
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0xB1, 0xF, 0xF, false)); // xor 1
    return v;
}

// block_reduce<SUM, 1024> (32-lane groups, partial sums of the 32 groups in LDS) with q8p_sum32
static __device__ __forceinline__ float q8p_block_sum1024(float v, float * s_sum) {
    v = q8p_sum32(v);
    if (threadIdx.x % 32 == 0) {
        s_sum[threadIdx.x / 32] = v;
    }
    __syncthreads();
    return q8p_sum32(s_sum[threadIdx.x % 32]);
}

static __device__ __forceinline__ float q8p_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// one value per thread, 32 consecutive lanes = one q8_1 block (same reduction as quantize_q8_1)
static __device__ __forceinline__ void q8p_quantize_1(const float v, const int64_t i, block_q8_1 * __restrict__ y, int * __restrict__ ysum) {
    float amax = fabsf(v);
    amax = warp_reduce_max<QK8_1>(amax);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : q8p_round_div(v, d, 1.0f / d);
    const int sum = warp_reduce_sum<QK8_1>((int) q);
    y[i/QK8_1].qs[i%QK8_1] = q;
    if (i % QK8_1 == 0) {
        y[i/QK8_1].ds = make_half2(d, d * sum);
        ysum[i/QK8_1] = sum;
    }
}

// dst = op(x) * g, op = silu (OP 0) or sigmoid (OP 1), plus q8_1 of dst
template <int OP>
static __global__ void __launch_bounds__(256) gated_q8_kernel(const float * __restrict__ x, const float * __restrict__ g, float * __restrict__ dst,
        block_q8_1 * __restrict__ y, int * __restrict__ ysum, const int64_t k) {
    const int64_t i = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    const float xv = x[i];
    const float a  = OP == 0 ? ggml_cuda_op_silu_single(xv) : q8p_sigmoid(xv);
    const float r  = a * g[i];
    dst[i] = r;
    q8p_quantize_1(r, i, y, ysum);
}

// rms_norm(x) * w for one row (+ q8_1), same arithmetic as rms_norm_f32<1024, 4, true, true>. The q8_1 epilogue is
// VALU-heavy, so the row is split over gridDim.x workgroups of 1024 threads: every workgroup computes the identical
// full-row reduction (bit-identical scale), then outputs its own slice of 64 float4 (8 q8_1 blocks) with threads
// 0..63. (One workgroup doing all 1280 float4 of a 5120 row spent ~4 us in situ on the q8 math of 16 waves on one CU.)
// SMALL: q = roundf(r / d) directly (the formula of quantize_q8_1, bit-identical results) instead of the
// round-div fast path with its out-of-line division fallbacks, and the block reduction with swizzle/DPP (same
// butterfly): ~85 instructions less code (398 -> ~315). In the model each dispatch
// starts with a cold instruction cache, so code size on the critical path is a large part of this kernel's cost.
template <int VPT, bool SMALL, bool MR = false>
static __global__ void __launch_bounds__(1024) rms_norm_mul_q8_kernel(const float * __restrict__ x, const float * __restrict__ w,
        float * __restrict__ dst, block_q8_1 * __restrict__ y, int * __restrict__ ysum, const int ncols, const float eps) {
    constexpr int block_size = 1024;
    constexpr int per_wg     = 64;    // float4 per workgroup slice
    __shared__ float s_sum[32];
    const int tid = threadIdx.x;
    if constexpr (MR) {   // MTP2-O2: row blockIdx.y of a [ncols, nrows] tensor; the per-row arithmetic is unchanged
        x    += (int64_t) blockIdx.y*ncols;
        dst  += (int64_t) blockIdx.y*ncols;
        y    += blockIdx.y*(ncols/QK8_1);
        ysum += blockIdx.y*(ncols/QK8_1);
    }
    const int ncols4 = ncols / 4;
    const float4 * x4   = (const float4 *) x;
    const float4 * mul4 = (const float4 *) w;
    float4       * dst4 = (float4 *) dst;

    // slice operands are loaded together with the reduction inputs (shorter dependent latency chain)
    const int  io  = blockIdx.x*per_wg + tid;
    const bool out = tid < per_wg && io < ncols4;
    float4 xo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 mo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (out) {
        xo = x4[io];
        mo = mul4[io];
    }
    float4 xs[VPT];
    float  tmp = 0.0f;
#pragma unroll
    for (int k = 0; k < VPT; k++) {
        const int i = tid + k*block_size;
        if (i < ncols4) {
            xs[k] = x4[i];
        }
    }
#pragma unroll
    for (int k = 0; k < VPT; k++) {
        const int i = tid + k*block_size;
        if (i < ncols4) {
            tmp += xs[k].x*xs[k].x + xs[k].y*xs[k].y + xs[k].z*xs[k].z + xs[k].w*xs[k].w;
        }
    }
    if constexpr (SMALL) {
        tmp = q8p_block_sum1024(tmp, s_sum);
    } else {
        tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    }
    const float scale = rsqrtf(tmp / ncols + eps);

    if (out) {
        const int i = io;
        float4 r = make_float4(scale*xo.x, scale*xo.y, scale*xo.z, scale*xo.w);
        r.x *= mo.x;
        r.y *= mo.y;
        r.z *= mo.z;
        r.w *= mo.w;
        dst4[i] = r;

        // q8_1: 8 consecutive threads hold one block of 32 values (per_wg % 8 == 0, so whole groups are active)
        float amax = fmaxf(fmaxf(fabsf(r.x), fabsf(r.y)), fmaxf(fabsf(r.z), fabsf(r.w)));
        amax = q8p_max8(amax);
        const float d = amax / 127.0f;
        int8_t q0, q1, q2, q3;
        if constexpr (SMALL) {
            q0 = amax == 0.0f ? 0 : (int8_t) roundf(r.x / d);
            q1 = amax == 0.0f ? 0 : (int8_t) roundf(r.y / d);
            q2 = amax == 0.0f ? 0 : (int8_t) roundf(r.z / d);
            q3 = amax == 0.0f ? 0 : (int8_t) roundf(r.w / d);
        } else {
            const float inv_d = 1.0f / d;
            q0 = amax == 0.0f ? 0 : q8p_round_div(r.x, d, inv_d);
            q1 = amax == 0.0f ? 0 : q8p_round_div(r.y, d, inv_d);
            q2 = amax == 0.0f ? 0 : q8p_round_div(r.z, d, inv_d);
            q3 = amax == 0.0f ? 0 : q8p_round_div(r.w, d, inv_d);
        }
        int sum = (int) q0 + (int) q1 + (int) q2 + (int) q3;
        sum = q8p_sum8(sum);
        const uint32_t packed = (uint32_t) (uint8_t) q0 | ((uint32_t) (uint8_t) q1 << 8) |
                                ((uint32_t) (uint8_t) q2 << 16) | ((uint32_t) (uint8_t) q3 << 24);
        ((uint32_t *) y[i/8].qs)[i%8] = packed;
        if (i % 8 == 0) {
            y[i/8].ds = make_half2(d, d * sum);
            ysum[i/8] = sum;
        }
    }
}

// ------------------------------------------------------------------------------------- N1: shaped 5120-column norm
// rms_norm_mul_q8_kernel<2, true> for exactly 5120 columns with 256 threads (4 waves) instead of 1024, and less code
// on the critical path. Bit-identical outputs (fp32 row and q8_1 blocks + sums) by construction:
//  - reduction: thread t holds the partial sums of the virtual lanes v = t + 256*k (k < 4) of the 1024-lane kernel
//    (v < 256 also adds element v + 1024, as there). The 32-lane butterflies of the four values are the same as in
//    q8p_sum32 (lanes t and v have the same index within their 32-lane group, so the partner values are the same);
//    xor 8 and xor 4 use DPP row rotations: after the xor 16 step lanes congruent mod 16 hold equal values, after xor 8
//    those congruent mod 8, and row_ror:8 / row_ror:4 read a lane of the same class as lane^8 / lane^4. Float
//    addition is commutative, so every add sees bitwise the same operands. The 32 group sums go through LDS in the
//    same order as before (group t/32 + 8k), and the second butterfly is unchanged.
//  - the squares are rounded on their own and summed ((xx + yy) + zz) + ww, as the compiled 1024-thread kernel does
//    (v_pk_mul_f32 products, no contraction).
//  - x / 5120 and amax / 127 are correctly rounded divisions computed as q0 = x*r, q = fma(fma(-q0, b, x), r, q0)
//    with r = 1/b: checked equal to the IEEE quotient for all 2^32 inputs on gfx90a (infinities keep q0), for 5120
//    only for |x| >= 1e-30 (tiny sums take the IEEE division; amax / 127 differs only for -0, and amax >= +0).
//  - roundf(r / d) is taken as rintf(r * rcp(d)) unless the product is within 4e-5 of a half-integer (the error of
//    the product is below 2.8e-5 for |r / d| <= 127.0001), in which case the whole group of 4 recomputes roundf(r / d).
// GGML_Q8_NORM_SHAPED=0 disables (previous kernel). For 5120 columns the shaped kernel replaces both previous variants,
// so GGML_Q8_NORM_SMALL=0 has no effect there unless GGML_Q8_NORM_SHAPED=0 (same bits either way). The multi-row
// shaped kernel is compared with the single-row path by GGML_Q8P_NC_VERIFY, the single-row one by
// GGML_Q8_NORM_SHAPED_VERIFY.
#define Q8P_SHAPED_NCOLS 5120
#define Q8P_SHAPED_NT    256

static __device__ __forceinline__ float q8p_div_const(const float x, const float b, const float r) {
    const float q0 = x * r;
    const float e  = fmaf(-q0, b, x);
    const float q  = fmaf(e, r, q0);
    return e == e ? q : q0;
}

static __device__ __forceinline__ float q8p_rmul(const float a, const float b) {
    float p = a * b;
    asm volatile("" : "+v"(p));
    return p;
}

static __device__ __forceinline__ float q8p_sumsq4(const float4 v) {
    return ((q8p_rmul(v.x, v.x) + q8p_rmul(v.y, v.y)) + q8p_rmul(v.z, v.z)) + q8p_rmul(v.w, v.w);
}

template <int CTRL>
static __device__ __forceinline__ float q8p_dpp(const float v) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xF, 0xF, false));
}

// == q8p_sum32 (xor 16 swizzle; xor 8 = row_ror:8, xor 4 = row_ror:4 on the equal-value classes; xor 2, xor 1)
static __device__ __forceinline__ float q8p_sum32_dpp(float v) {
    v += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F)); // xor 16
    v += q8p_dpp<0x128>(v); // row_ror:8
    v += q8p_dpp<0x124>(v); // row_ror:4
    v += q8p_dpp<0x4E>(v);  // xor 2
    v += q8p_dpp<0xB1>(v);  // xor 1
    return v;
}

template <bool MR>
static __global__ void __launch_bounds__(Q8P_SHAPED_NT) rms_norm_mul_q8_shaped_kernel(const float * __restrict__ x,
        const float * __restrict__ w, float * __restrict__ dst, block_q8_1 * __restrict__ y, int * __restrict__ ysum,
        const float eps) {
    constexpr int ncols4 = Q8P_SHAPED_NCOLS / 4;      // 1280
    constexpr int per_wg = 64;                        // float4 of the output slice of a workgroup
    constexpr int NV     = 1024 / Q8P_SHAPED_NT;      // virtual lanes per thread
    constexpr int NL     = ncols4 / Q8P_SHAPED_NT;    // float4 loads per thread (5)
    static_assert(NL == NV + 1 && ncols4 - 1024 == Q8P_SHAPED_NT, "thread mapping");
    __shared__ float s_sum[32];
    const int tid = threadIdx.x;
    if constexpr (MR) {
        x    += (int64_t) blockIdx.y*Q8P_SHAPED_NCOLS;
        dst  += (int64_t) blockIdx.y*Q8P_SHAPED_NCOLS;
        y    += blockIdx.y*(Q8P_SHAPED_NCOLS/QK8_1);
        ysum += blockIdx.y*(Q8P_SHAPED_NCOLS/QK8_1);
    }
    const float4 * x4 = (const float4 *) x;
    const int  io  = blockIdx.x*per_wg + tid;
    const bool out = tid < per_wg;
    float4 xo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 mo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (out) {
        xo = x4[io];
        mo = ((const float4 *) w)[io];
    }
    float4 xs[NL];
#pragma unroll
    for (int k = 0; k < NL; ++k) {
        xs[k] = x4[tid + k*Q8P_SHAPED_NT];
    }
    float v[NV];
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        v[k] = q8p_sumsq4(xs[k]);
    }
    v[0] = v[0] + q8p_sumsq4(xs[NV]); // virtual lane tid also holds element tid + 1024
#pragma unroll
    for (int k = 0; k < NV; ++k) {
        v[k] = q8p_sum32_dpp(v[k]);
    }
    if (tid % 32 == 0) {
#pragma unroll
        for (int k = 0; k < NV; ++k) {
            s_sum[tid/32 + k*(Q8P_SHAPED_NT/32)] = v[k];
        }
    }
    __syncthreads();
    if (!out) {
        return;
    }
    const float tmp   = q8p_sum32_dpp(s_sum[tid % 32]);
    // (the fma-corrected quotient by 5120 is exact for |tmp| >= 1e-30; below that the IEEE division, uniform branch)
    const float mean  = tmp >= 1.0e-30f ? q8p_div_const(tmp, (float) Q8P_SHAPED_NCOLS, 1.0f/Q8P_SHAPED_NCOLS) : tmp / Q8P_SHAPED_NCOLS;
    const float scale = rsqrtf(mean + eps);

    float4 r = make_float4(scale*xo.x, scale*xo.y, scale*xo.z, scale*xo.w);
    r.x *= mo.x;
    r.y *= mo.y;
    r.z *= mo.z;
    r.w *= mo.w;
    ((float4 *) dst)[io] = r;

    float amax = fmaxf(fmaxf(fabsf(r.x), fabsf(r.y)), fmaxf(fabsf(r.z), fabsf(r.w)));
    amax = q8p_max8(amax);
    const float d     = q8p_div_const(amax, 127.0f, 1.0f/127.0f);
    const float inv_d = __builtin_amdgcn_rcpf(d);
    float t0 = r.x * inv_d;
    float t1 = r.y * inv_d;
    float t2 = r.z * inv_d;
    float t3 = r.w * inv_d;
    const float m0 = fabsf(__builtin_amdgcn_fractf(fabsf(t0)) - 0.5f);
    const float m1 = fabsf(__builtin_amdgcn_fractf(fabsf(t1)) - 0.5f);
    const float m2 = fabsf(__builtin_amdgcn_fractf(fabsf(t2)) - 0.5f);
    const float m3 = fabsf(__builtin_amdgcn_fractf(fabsf(t3)) - 0.5f);
    // NaN products (d = 0 or NaN) and denormal d (v_rcp_f32 range) take the IEEE formula as well
    if (!(fminf(fminf(m0, m1), fminf(m2, m3)) > 4.0e-5f) || !(m0 == m0 && m1 == m1 && m2 == m2 && m3 == m3) ||
            !(d >= 1.17549435e-38f)) {
        t0 = roundf(r.x / d);
        t1 = roundf(r.y / d);
        t2 = roundf(r.z / d);
        t3 = roundf(r.w / d);
    } else {
        t0 = rintf(t0);
        t1 = rintf(t1);
        t2 = rintf(t2);
        t3 = rintf(t3);
    }
    const int8_t q0 = amax == 0.0f ? 0 : (int8_t) t0;
    const int8_t q1 = amax == 0.0f ? 0 : (int8_t) t1;
    const int8_t q2 = amax == 0.0f ? 0 : (int8_t) t2;
    const int8_t q3 = amax == 0.0f ? 0 : (int8_t) t3;
    int sum = (int) q0 + (int) q1 + (int) q2 + (int) q3;
    sum = q8p_sum8(sum);
    const uint32_t packed = (uint32_t) (uint8_t) q0 | ((uint32_t) (uint8_t) q1 << 8) |
                            ((uint32_t) (uint8_t) q2 << 16) | ((uint32_t) (uint8_t) q3 << 24);
    ((uint32_t *) y[io/8].qs)[io%8] = packed;
    if (io % 8 == 0) {
        y[io/8].ds = make_half2(d, d * sum);
        ysum[io/8] = sum;
    }
}

static bool q8p_norm_shaped() {
    static const bool shaped = [] {
        const char * v = getenv("GGML_Q8_NORM_SHAPED");
        return v == nullptr || v[0] != '0';
    }();
    return shaped;
}

// In-place variant of rms_norm_mul_q8_kernel (dst overlaps x, e.g. the output norm that ggml-alloc places over l_out):
// a single workgroup loads the whole row into registers before the reduction barrier and only then stores, so no load
// can observe an already normalized element (the multi-workgroup kernel re-reads x in every workgroup while others
// store their slices). Same reduction and per-element arithmetic as rms_norm_mul_q8_kernel, so dst and the q8_1 blocks
// are bit-identical to it. No __restrict__ on x/w/dst: they may alias.
template <int VPT, bool MR = false>
static __global__ void __launch_bounds__(1024) rms_norm_mul_q8_inplace_kernel(const float * x, const float * w,
        float * dst, block_q8_1 * __restrict__ y, int * __restrict__ ysum, const int ncols, const float eps) {
    constexpr int block_size = 1024;
    __shared__ float s_sum[32];
    const int tid = threadIdx.x;
    if constexpr (MR) {   // MTP2-O2: row blockIdx.x of a [ncols, nrows] tensor; the per-row arithmetic is unchanged
        x    += (int64_t) blockIdx.x*ncols;
        dst  += (int64_t) blockIdx.x*ncols;
        y    += blockIdx.x*(ncols/QK8_1);
        ysum += blockIdx.x*(ncols/QK8_1);
    }
    const int ncols4 = ncols / 4;
    const float4 * x4   = (const float4 *) x;
    const float4 * mul4 = (const float4 *) w;
    float4       * dst4 = (float4 *) dst;

    float4 xs[VPT];
    float  tmp = 0.0f;
#pragma unroll
    for (int k = 0; k < VPT; k++) {
        const int i = tid + k*block_size;
        if (i < ncols4) {
            xs[k] = x4[i];
        }
    }
#pragma unroll
    for (int k = 0; k < VPT; k++) {
        const int i = tid + k*block_size;
        if (i < ncols4) {
            tmp += xs[k].x*xs[k].x + xs[k].y*xs[k].y + xs[k].z*xs[k].z + xs[k].w*xs[k].w;
        }
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float scale = rsqrtf(tmp / ncols + eps);

#pragma unroll
    for (int k = 0; k < VPT; k++) {
        const int i = tid + k*block_size;
        // ncols4 % 128 == 0: aligned groups of 8 lanes (one q8_1 block) are either fully active or fully inactive
        if (i < ncols4) {
            const float4 xo = xs[k];   // register copy loaded before the barrier (x may alias dst)
            const float4 mo = mul4[i]; // w may alias dst only exactly: this thread reads element i before storing it
            float4 r = make_float4(scale*xo.x, scale*xo.y, scale*xo.z, scale*xo.w);
            r.x *= mo.x;
            r.y *= mo.y;
            r.z *= mo.z;
            r.w *= mo.w;
            dst4[i] = r;

            float amax = fmaxf(fmaxf(fabsf(r.x), fabsf(r.y)), fmaxf(fabsf(r.z), fabsf(r.w)));
            amax = q8p_max8(amax);
            const float d = amax / 127.0f;
            const float inv_d = 1.0f / d;
            const int8_t q0 = amax == 0.0f ? 0 : q8p_round_div(r.x, d, inv_d);
            const int8_t q1 = amax == 0.0f ? 0 : q8p_round_div(r.y, d, inv_d);
            const int8_t q2 = amax == 0.0f ? 0 : q8p_round_div(r.z, d, inv_d);
            const int8_t q3 = amax == 0.0f ? 0 : q8p_round_div(r.w, d, inv_d);
            int sum = (int) q0 + (int) q1 + (int) q2 + (int) q3;
            sum = q8p_sum8(sum);
            const uint32_t packed = (uint32_t) (uint8_t) q0 | ((uint32_t) (uint8_t) q1 << 8) |
                                    ((uint32_t) (uint8_t) q2 << 16) | ((uint32_t) (uint8_t) q3 << 24);
            ((uint32_t *) y[i/8].qs)[i%8] = packed;
            if (i % 8 == 0) {
                y[i/8].ds = make_half2(d, d * sum);
                ysum[i/8] = sum;
            }
        }
    }
}

// one row of 128 per 32-thread block (as rms_norm_f32<32, 1, true, true>): rms_norm(x) * w, then silu(z) * that (as the
// UNARY(SILU)+MUL op), + q8_1 of the result (8 consecutive threads hold one block of 32 values)
static __global__ void __launch_bounds__(32) rms_norm_gated_q8_kernel(const float * __restrict__ x, const float * __restrict__ w,
        const float * __restrict__ z, float * __restrict__ dst, block_q8_1 * __restrict__ y, int * __restrict__ ysum,
        const int ncols, const float eps) {
    __shared__ float s_sum[32];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float4 * x4 = (const float4 *) (x + (int64_t) row*ncols);
    const float4 * z4 = (const float4 *) (z + (int64_t) row*ncols);
    const float4 * m4 = (const float4 *) w;
    float4       * d4 = (float4 *) (dst + (int64_t) row*ncols);

    const float4 xs = x4[tid];
    const float4 m  = m4[tid];
    const float4 zv = z4[tid];
    float tmp = 0.0f;
    tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
    tmp = block_reduce<block_reduce_method::SUM, 32>(tmp, s_sum);
    const float scale = rsqrtf(tmp / ncols + eps);

    float4 r = make_float4(scale*xs.x, scale*xs.y, scale*xs.z, scale*xs.w);
    r.x *= m.x;
    r.y *= m.y;
    r.z *= m.z;
    r.w *= m.w;
    float4 o;
    o.x = ggml_cuda_op_silu_single(zv.x) * r.x;
    o.y = ggml_cuda_op_silu_single(zv.y) * r.y;
    o.z = ggml_cuda_op_silu_single(zv.z) * r.z;
    o.w = ggml_cuda_op_silu_single(zv.w) * r.w;
    d4[tid] = o;

    if (y) {
        const int i = row*(ncols/4) + tid;   // float4 index in the flattened tensor
        float amax = fmaxf(fmaxf(fabsf(o.x), fabsf(o.y)), fmaxf(fabsf(o.z), fabsf(o.w)));
        amax = q8p_max8(amax);
        const float d = amax / 127.0f;
        const float inv_d = 1.0f / d;
        const int8_t q0 = amax == 0.0f ? 0 : q8p_round_div(o.x, d, inv_d);
        const int8_t q1 = amax == 0.0f ? 0 : q8p_round_div(o.y, d, inv_d);
        const int8_t q2 = amax == 0.0f ? 0 : q8p_round_div(o.z, d, inv_d);
        const int8_t q3 = amax == 0.0f ? 0 : q8p_round_div(o.w, d, inv_d);
        int sum = (int) q0 + (int) q1 + (int) q2 + (int) q3;
        sum = q8p_sum8(sum);
        const uint32_t packed = (uint32_t) (uint8_t) q0 | ((uint32_t) (uint8_t) q1 << 8) |
                                ((uint32_t) (uint8_t) q2 << 16) | ((uint32_t) (uint8_t) q3 << 24);
        ((uint32_t *) y[i/8].qs)[i%8] = packed;
        if (i % 8 == 0) {
            y[i/8].ds = make_half2(d, d * sum);
            ysum[i/8] = sum;
        }
    }
}

// Large-batch (prefill) variant: rows of 128, the f16 copy of silu(z) * (rms_norm(x) * w) only (the input of the
// following f16 GEMM; the f32 result has no other consumer and is not written). One 32-lane group per row, 8 rows per
// 256-thread block. Per row exactly the arithmetic of rms_norm_f32<32, 1, true, true> (same per-lane partial sums and
// the same width-32 xor reduction), the UNARY(SILU)+MUL op and the f32 -> f16 conversion of the GEMM input, so the
// f16 values are bit-identical to that sequence of three kernels. (Two or four rows per group are slower on gfx90a
// and, once unrolled, let the compiler contract the sums of squares differently: not bit-identical.)
struct alignas(8) gnorm_half4 { half v[4]; };

static __global__ void __launch_bounds__(256) rms_norm_gated_f16_kernel(const float * __restrict__ x, const float * __restrict__ w,
        const float * __restrict__ z, half * __restrict__ dst, const int64_t nrows, const int ncols, const float eps) {
    const int     tid = threadIdx.x & 31;
    const int64_t row = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= nrows) {
        return;   // the whole 32-lane group (the reduction never leaves it)
    }
    const float4 xs = ((const float4 *) x)[row*32 + tid];
    const float4 zv = ((const float4 *) z)[row*32 + tid];
    const float4 m  = ((const float4 *) w)[tid];

    float tmp = 0.0f;
    tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
    tmp = warp_reduce_sum(tmp);   // width 32 = block_reduce<SUM, 32>
    const float scale = rsqrtf(tmp / ncols + eps);

    float4 r = make_float4(scale*xs.x, scale*xs.y, scale*xs.z, scale*xs.w);
    r.x *= m.x;
    r.y *= m.y;
    r.z *= m.z;
    r.w *= m.w;
    gnorm_half4 h;
    h.v[0] = ggml_cuda_cast<half>(ggml_cuda_op_silu_single(zv.x) * r.x);
    h.v[1] = ggml_cuda_cast<half>(ggml_cuda_op_silu_single(zv.y) * r.y);
    h.v[2] = ggml_cuda_cast<half>(ggml_cuda_op_silu_single(zv.z) * r.z);
    h.v[3] = ggml_cuda_cast<half>(ggml_cuda_op_silu_single(zv.w) * r.w);
    ((gnorm_half4 *) dst)[row*32 + tid] = h;
}

// dst = op(x) * g with x read through a strided view (row length ne0, row strides s1/s2 in floats), + q8_1 of dst;
// same arithmetic as CONT + gated_q8_kernel
template <int OP>
static __global__ void __launch_bounds__(256) gated_q8_strided_kernel(const float * __restrict__ x, const uint32_t ne0, const uint32_t ne1,
        const int64_t s1, const int64_t s2, const float * __restrict__ g, float * __restrict__ dst,
        block_q8_1 * __restrict__ y, int * __restrict__ ysum, const uint32_t k) {
    const uint32_t i = blockDim.x*blockIdx.x + threadIdx.x;   // k <= 65536
    if (i >= k) {
        return;
    }
    const uint32_t i0 = i % ne0;
    const uint32_t ir = i / ne0;
    const uint32_t i1 = ir % ne1;
    const uint32_t i2 = ir / ne1;
    const float xv = x[i0 + i1*s1 + i2*s2];
    const float a  = OP == 0 ? ggml_cuda_op_silu_single(xv) : q8p_sigmoid(xv);
    const float r  = a * g[i];
    dst[i] = r;
    if (y) {
        q8p_quantize_1(r, i, y, ysum);
    }
}

// byte ranges of a and b intersect (ggml_nbytes covers the full extent of strided views)
static bool q8p_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

// an elementwise producer (thread i reads element i of src, then writes element i of dst) is safe in place only if src
// and dst coincide exactly; any other overlap would let one thread read an element another thread already wrote
static bool q8p_partial_overlap(const ggml_tensor * src, const ggml_tensor * dst) {
    return q8p_overlap(src, dst) && (src->data != dst->data || ggml_nbytes(src) != ggml_nbytes(dst));
}

static bool q8p_single_row_f32(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[1] == 1 && t->ne[2] == 1 && t->ne[3] == 1 &&
           t->ne[0] % 512 == 0 && ((uintptr_t) t->data) % 16 == 0;
}

// any contiguous f32 tensor with a single "token" worth of elements, treated as a flat vector
static bool q8p_flat_f32(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nelements(t) % 512 == 0 && ggml_nelements(t) <= 65536 &&
           ((uintptr_t) t->data) % 16 == 0;
}

static bool q8p_enabled() {
    static const bool enabled = [] {
        const char * v = getenv("GGML_Q8_PRODUCERS");
        return v == nullptr || v[0] != '0';
    }();
    return enabled;
}

static bool q8p_norm_enabled() {
    static const bool enabled = [] {
        const char * v = getenv("GGML_Q8_PRODUCERS_NORM");
        return v == nullptr || v[0] != '0';
    }();
    return enabled;
}

// ------------------------------------------------------------------------------------------------- MTP2-O2: N rows
// Multi-row (2..nc_max tokens, multi-token verification) variants of the single-row producers: every row runs exactly
// the single-row kernel (rms_norm*w: the same template with a row offset, one grid row per token; the elementwise
// gated kernels: the same kernel over the flat tensor). The q8_1 blocks of the flat [K, N] tensor are the per-column
// blocks (K % 512 == 0), i.e. the layout the N-column GEMVs read from the activation cache (ggml_cuda_mmvq_q8_cached_nc),
// so they skip their own quantize launch. Only for N <= ggml_cuda_mmvq_nc_max() (where the consumer is an N-column
// GEMV). GGML_Q8P_NC=0 disables (previous behaviour exactly); GGML_Q8P_NC_VERIFY=<n> compares the first n multi-row
// norm calls row by row with the single-row kernel (debug, sync).
int ggml_cuda_mmvq_nc_max();

static bool q8p_nc_enabled() {
    static const bool enabled = [] {
        const char * v = getenv("GGML_Q8P_NC");
        return v == nullptr || v[0] != '0';
    }();
    return enabled;
}

// contiguous f32 [ne0, n] with 2 <= n <= nc_max rows, ne0 % 512 == 0
static bool q8p_nc_rows_f32(const ggml_tensor * t) {
    return q8p_nc_enabled() && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[1] >= 2 &&
           t->ne[1] <= ggml_cuda_mmvq_nc_max() && t->ne[2] == 1 && t->ne[3] == 1 && t->ne[0] % 512 == 0 &&
           t->ne[0] <= 65536 && ((uintptr_t) t->data) % 16 == 0;
}

static void q8p_norm_launch(ggml_backend_cuda_context & ctx, const bool inplace, const bool small, const float * x, const float * w,
        float * dst, block_q8_1 * y, int * ysum, const int ncols, const int nrows, const float eps) {
    if (inplace) {
        if (ncols <= 8192) {
            rms_norm_mul_q8_inplace_kernel<2, true><<<nrows, 1024, 0, ctx.stream()>>>(x, w, dst, y, ysum, ncols, eps);
        } else {
            rms_norm_mul_q8_inplace_kernel<4, true><<<nrows, 1024, 0, ctx.stream()>>>(x, w, dst, y, ysum, ncols, eps);
        }
    } else {
        const dim3 grid(ncols / 256, nrows, 1);
        if (ncols == Q8P_SHAPED_NCOLS && q8p_norm_shaped()) {
            rms_norm_mul_q8_shaped_kernel<true><<<grid, Q8P_SHAPED_NT, 0, ctx.stream()>>>(x, w, dst, y, ysum, eps);
        } else if (ncols <= 8192) {
            if (small) {
                rms_norm_mul_q8_kernel<2, true, true><<<grid, 1024, 0, ctx.stream()>>>(x, w, dst, y, ysum, ncols, eps);
            } else {
                rms_norm_mul_q8_kernel<2, false, true><<<grid, 1024, 0, ctx.stream()>>>(x, w, dst, y, ysum, ncols, eps);
            }
        } else {
            rms_norm_mul_q8_kernel<4, false, true><<<grid, 1024, 0, ctx.stream()>>>(x, w, dst, y, ysum, ncols, eps);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

static bool q8p_norm_small() {
    static const bool small = [] {
        const char * v = getenv("GGML_Q8_NORM_SMALL");
        return v == nullptr || v[0] != '0';
    }();
    return small;
}

static bool ggml_cuda_rms_norm_mul_q8_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, ggml_tensor * mul_node) {
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * w = mul_node->src[0] == rms_norm ? mul_node->src[1] : mul_node->src[0];
    if (!q8p_nc_rows_f32(x) || !q8p_nc_rows_f32(mul_node) || !q8p_nc_rows_f32(rms_norm) ||
            !ggml_are_same_shape(x, mul_node) || !ggml_are_same_shape(x, rms_norm) ||
            w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || w->ne[0] != x->ne[0] || ggml_nelements(w) != x->ne[0] ||
            ((uintptr_t) w->data) % 16 != 0 || x->ne[0] > 16384) {
        return false;
    }
    // in place only on exactly coinciding x/dst (each row's workgroup loads its row before storing it); w never overlaps
    if (q8p_overlap(w, mul_node) || (q8p_overlap(x, mul_node) && x->data != mul_node->data)) {
        return false;
    }
    const bool inplace = q8p_overlap(x, mul_node);
    block_q8_1 * y = nullptr;
    int * ysum = nullptr;
    if (!ggml_cuda_mmvq_cache_produce(ctx, mul_node, &y, &ysum)) {
        return false;
    }
    float eps;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    const int ncols = (int) x->ne[0];
    const int nrows = (int) x->ne[1];
    const bool small = q8p_norm_small();
    static int verify_left = getenv("GGML_Q8P_NC_VERIFY") ? atoi(getenv("GGML_Q8P_NC_VERIFY")) : 0;
    if (verify_left > 0 && !inplace) {
        // debug: each row through the single-row launch of ggml_cuda_rms_norm_mul_q8 into scratch, then the multi-row
        // kernel, compare dst and q8 bytes per row
        verify_left--;
        const int nblk = ncols / QK8_1;
        std::vector<float> ref((size_t) ncols*nrows), got((size_t) ncols*nrows);
        std::vector<block_q8_1> rq((size_t) nblk*nrows), gq((size_t) nblk*nrows);
        std::vector<int> rs((size_t) nblk*nrows), gs((size_t) nblk*nrows);
        ggml_cuda_pool_alloc<float> scratch(ctx.pool(), (size_t) ncols);
        ggml_cuda_pool_alloc<block_q8_1> sq(ctx.pool(), (size_t) nblk);
        ggml_cuda_pool_alloc<int> ss(ctx.pool(), (size_t) nblk);
        for (int r = 0; r < nrows; ++r) {
            const float * xr = (const float *) x->data + (size_t) r*ncols;
            if (ncols <= 8192) {
                if (small) {
                    rms_norm_mul_q8_kernel<2, true><<<ncols/256, 1024, 0, ctx.stream()>>>(xr, (const float *) w->data, scratch.ptr, sq.ptr, ss.ptr, ncols, eps);
                } else {
                    rms_norm_mul_q8_kernel<2, false><<<ncols/256, 1024, 0, ctx.stream()>>>(xr, (const float *) w->data, scratch.ptr, sq.ptr, ss.ptr, ncols, eps);
                }
            } else {
                rms_norm_mul_q8_kernel<4, false><<<ncols/256, 1024, 0, ctx.stream()>>>(xr, (const float *) w->data, scratch.ptr, sq.ptr, ss.ptr, ncols, eps);
            }
            CUDA_CHECK(cudaMemcpyAsync(ref.data() + (size_t) r*ncols, scratch.ptr, ncols*sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
            CUDA_CHECK(cudaMemcpyAsync(rq.data() + (size_t) r*nblk, sq.ptr, nblk*sizeof(block_q8_1), cudaMemcpyDeviceToHost, ctx.stream()));
            CUDA_CHECK(cudaMemcpyAsync(rs.data() + (size_t) r*nblk, ss.ptr, nblk*sizeof(int), cudaMemcpyDeviceToHost, ctx.stream()));
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        }
        q8p_norm_launch(ctx, false, small, (const float *) x->data, (const float *) w->data, (float *) mul_node->data, y, ysum, ncols, nrows, eps);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        CUDA_CHECK(cudaMemcpy(got.data(), mul_node->data, got.size()*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gq.data(), y, gq.size()*sizeof(block_q8_1), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gs.data(), ysum, gs.size()*sizeof(int), cudaMemcpyDeviceToHost));
        size_t d_out = 0, d_q8 = 0;
        for (size_t e = 0; e < got.size(); ++e) {
            d_out += memcmp(&got[e], &ref[e], sizeof(float)) != 0;
        }
        for (size_t b = 0; b < gq.size(); ++b) {
            d_q8 += memcmp(&gq[b], &rq[b], sizeof(block_q8_1)) != 0 || gs[b] != rs[b];
        }
        fprintf(stderr, "q8p_nc verify %s: %d rows x %d: out %zu/%zu differ, q8 %zu/%zu differ%s\n", mul_node->name, nrows, ncols,
            d_out, got.size(), d_q8, gq.size(), d_out || d_q8 ? "  <-- MISMATCH" : "");
        return true;
    }
    q8p_norm_launch(ctx, inplace, small, (const float *) x->data, (const float *) w->data, (float *) mul_node->data, y, ysum, ncols, nrows, eps);
    return true;
}
// ------------------------------------------------------------------------------------------------- MTP2-O2 end

// GGML_Q8_NORM_SHAPED_VERIFY=<n> (debug): the first n single-row shaped launches outside graph capture are compared
// bit for bit (fp32 row, q8_1 blocks, sums) with rms_norm_mul_q8_kernel<2, true> run into scratch buffers
static void q8p_shaped_verify(ggml_backend_cuda_context & ctx, const float * x, const float * w, const float * dst,
        const block_q8_1 * y, const int * ysum, const float eps) {
    static int left = getenv("GGML_Q8_NORM_SHAPED_VERIFY") ? atoi(getenv("GGML_Q8_NORM_SHAPED_VERIFY")) : 0;
    if (left <= 0) {
        return;
    }
    hipStreamCaptureStatus status;
    CUDA_CHECK(hipStreamIsCapturing(ctx.stream(), &status));
    if (status != hipStreamCaptureStatusNone) {
        return;
    }
    left--;
    constexpr int n = Q8P_SHAPED_NCOLS, nb = n/QK8_1;
    ggml_cuda_pool_alloc<float>      rd(ctx.pool(), n);
    ggml_cuda_pool_alloc<block_q8_1> rq(ctx.pool(), nb);
    ggml_cuda_pool_alloc<int>        rs(ctx.pool(), nb);
    rms_norm_mul_q8_kernel<2, true><<<n/256, 1024, 0, ctx.stream()>>>(x, w, rd.ptr, rq.ptr, rs.ptr, n, eps);
    std::vector<float> a(n), b(n);
    std::vector<block_q8_1> qa(nb), qb(nb);
    std::vector<int> sa(nb), sb(nb);
    CUDA_CHECK(cudaMemcpyAsync(a.data(), dst, n*sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(b.data(), rd.ptr, n*sizeof(float), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(qa.data(), y, nb*sizeof(block_q8_1), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(qb.data(), rq.ptr, nb*sizeof(block_q8_1), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(sa.data(), ysum, nb*sizeof(int), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(sb.data(), rs.ptr, nb*sizeof(int), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    const int md = memcmp(a.data(), b.data(), n*sizeof(float)) != 0;
    const int mq = memcmp(qa.data(), qb.data(), nb*sizeof(block_q8_1)) != 0;
    const int ms = memcmp(sa.data(), sb.data(), nb*sizeof(int)) != 0;
    fprintf(stderr, "q8p_shaped verify: dst %s q8 %s sums %s%s\n", md ? "DIFF" : "ok", mq ? "DIFF" : "ok", ms ? "DIFF" : "ok",
            md || mq || ms ? " MISMATCH" : "");
}

bool ggml_cuda_rms_norm_mul_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, ggml_tensor * mul_node) {
    if (!q8p_enabled() || !q8p_norm_enabled() || ggml_cuda_info().devices[ctx.device].warp_size != 64) {
        return false;
    }
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * w = mul_node->src[0] == rms_norm ? mul_node->src[1] : mul_node->src[0];
    // MTP2-O2 begin
    if (x->ne[1] > 1) {
        return ggml_cuda_rms_norm_mul_q8_rows(ctx, rms_norm, mul_node);
    }
    // MTP2-O2 end
    if (!q8p_single_row_f32(x) || !q8p_single_row_f32(mul_node) || !q8p_single_row_f32(rms_norm) ||
            w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || w->ne[0] != x->ne[0] || ggml_nelements(w) != x->ne[0] ||
            ((uintptr_t) w->data) % 16 != 0 || x->ne[0] > 16384) {
        return false;
    }
    // x is read in full by every workgroup of the multi-workgroup kernel, so an output that overlaps x (in place, as
    // ggml-alloc does for the output norm) needs the single-workgroup kernel. w is read element by element: it may
    // coincide with dst exactly (also single-workgroup, so the __restrict__ of the multi-workgroup kernel always holds)
    if (q8p_partial_overlap(w, mul_node)) {
        return false;
    }
    const bool inplace = q8p_overlap(x, mul_node) || q8p_overlap(w, mul_node);
    block_q8_1 * y = nullptr;
    int * ysum = nullptr;
    if (!ggml_cuda_mmvq_cache_produce(ctx, mul_node, &y, &ysum)) {
        return false;
    }
    float eps;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    if (inplace) {
        if (x->ne[0] <= 8192) {
            rms_norm_mul_q8_inplace_kernel<2><<<1, 1024, 0, ctx.stream()>>>((const float *) x->data, (const float *) w->data,
                    (float *) mul_node->data, y, ysum, (int) x->ne[0], eps);
        } else {
            rms_norm_mul_q8_inplace_kernel<4><<<1, 1024, 0, ctx.stream()>>>((const float *) x->data, (const float *) w->data,
                    (float *) mul_node->data, y, ysum, (int) x->ne[0], eps);
        }
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    // VPT only bounds the per-thread loop (elements beyond ncols are skipped), so 2 and 4 give identical results;
    // one workgroup per 64 float4 (ne0 % 512 == 0)
    const int nwg = (int) (x->ne[0] / 256);
    if (x->ne[0] == Q8P_SHAPED_NCOLS && q8p_norm_shaped()) {
        rms_norm_mul_q8_shaped_kernel<false><<<nwg, Q8P_SHAPED_NT, 0, ctx.stream()>>>((const float *) x->data,
                (const float *) w->data, (float *) mul_node->data, y, ysum, eps);
        CUDA_CHECK(cudaGetLastError());
        q8p_shaped_verify(ctx, (const float *) x->data, (const float *) w->data, (const float *) mul_node->data, y, ysum, eps);
        return true;
    }
    static const bool small = [] {
        const char * v = getenv("GGML_Q8_NORM_SMALL");
        return v == nullptr || v[0] != '0';
    }();
    if (x->ne[0] <= 8192) {
        if (small) {
            rms_norm_mul_q8_kernel<2, true><<<nwg, 1024, 0, ctx.stream()>>>((const float *) x->data, (const float *) w->data, (float *) mul_node->data,
                    y, ysum, (int) x->ne[0], eps);
        } else {
            rms_norm_mul_q8_kernel<2, false><<<nwg, 1024, 0, ctx.stream()>>>((const float *) x->data, (const float *) w->data, (float *) mul_node->data,
                    y, ysum, (int) x->ne[0], eps);
        }
    } else {
        rms_norm_mul_q8_kernel<4, false><<<nwg, 1024, 0, ctx.stream()>>>((const float *) x->data, (const float *) w->data, (float *) mul_node->data,
                y, ysum, (int) x->ne[0], eps);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

static bool q8p_gnorm_enabled() {
    static const bool enabled = [] {
        const char * v = getenv("GGML_Q8_PRODUCERS_GNORM");
        return v == nullptr || v[0] != '0';
    }();
    return enabled;
}

bool ggml_cuda_rms_norm_gated_q8_supported(const ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * z, const ggml_tensor * dst) {
    if (!q8p_gnorm_enabled() || ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != 64) {
        return false;
    }
    auto ok = [](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[0] == 128 && ((uintptr_t) t->data) % 16 == 0;
    };
    return ok(x) && ok(w) && ok(z) && ok(dst) && ggml_nelements(w) == 128 && ggml_are_same_shape(x, dst) &&
           ggml_nelements(z) == ggml_nelements(dst) && ggml_nrows(dst) <= 65535;
}

void ggml_cuda_rms_norm_gated_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w, float eps,
        const ggml_tensor * z, ggml_tensor * dst) {
    block_q8_1 * y = nullptr;
    int * ysum = nullptr;
    if (!q8p_enabled() || !q8p_flat_f32(dst) || !ggml_cuda_mmvq_cache_produce(ctx, dst, &y, &ysum)) {
        y = nullptr;
        ysum = nullptr;
    }
    rms_norm_gated_q8_kernel<<<ggml_nrows(dst), 32, 0, ctx.stream()>>>((const float *) x->data, (const float *) w->data,
            (const float *) z->data, (float *) dst->data, y, ysum, 128, eps);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_rms_norm_gated_f16_supported(const ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * z, const ggml_tensor * dst) {
    if (ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != 64) {
        return false;
    }
    auto ok = [](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[0] == 128 && ((uintptr_t) t->data) % 16 == 0;
    };
    return ok(x) && ok(w) && ok(z) && ok(dst) && ggml_nelements(w) == 128 && ggml_are_same_shape(x, dst) &&
           ggml_nelements(z) == ggml_nelements(dst) && ggml_nrows(dst) <= (int64_t) (UINT32_MAX / 256) * 8;   // grid below 2^32 threads
}

void ggml_cuda_rms_norm_gated_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w, float eps,
        const ggml_tensor * z, const ggml_tensor * dst, half * dst16) {
    const int64_t nrows = ggml_nrows(dst);
    rms_norm_gated_f16_kernel<<<(nrows + 7) / 8, 256, 0, ctx.stream()>>>((const float *) x->data, (const float *) w->data,
            (const float *) z->data, dst16, nrows, 128, eps);
    CUDA_CHECK(cudaGetLastError());
}

static bool gated_q8_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * g, ggml_tensor * dst, const int op) {
    if (!q8p_enabled() || ggml_cuda_info().devices[ctx.device].warp_size != 64) {
        return false;
    }
    // MTP2-O2: or N rows (2..nc_max) of a [K % 512 == 0, N] tensor: the same elementwise kernel, per-column q8_1 blocks
    const bool flat = q8p_flat_f32(x) && q8p_flat_f32(g) && q8p_flat_f32(dst);
    if (!(flat || (q8p_nc_rows_f32(x) && q8p_nc_rows_f32(g) && q8p_nc_rows_f32(dst))) ||
            !ggml_are_same_shape(x, dst) || !ggml_are_same_shape(g, dst)) {
        return false;
    }
    // elementwise kernel: in place only on exactly coinciding tensors
    if (q8p_partial_overlap(x, dst) || q8p_partial_overlap(g, dst)) {
        return false;
    }
    block_q8_1 * y = nullptr;
    int * ysum = nullptr;
    if (!ggml_cuda_mmvq_cache_produce(ctx, dst, &y, &ysum)) {
        return false;
    }
    const int64_t k = ggml_nelements(dst);
    const int64_t nblk = (k + 255) / 256;
    if (op == 0) {
        gated_q8_kernel<0><<<nblk, 256, 0, ctx.stream()>>>((const float *) x->data, (const float *) g->data, (float *) dst->data, y, ysum, k);
    } else {
        gated_q8_kernel<1><<<nblk, 256, 0, ctx.stream()>>>((const float *) x->data, (const float *) g->data, (float *) dst->data, y, ysum, k);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_swiglu_q8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (!src1 || ((const int32_t *) dst->op_params)[1] != 0) {  // separate gate/up tensors, not swapped
        return false;
    }
    return gated_q8_launch(ctx, src0, src1, dst, 0);
}

bool ggml_cuda_cont_unary_mul_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * cont_node, ggml_tensor * unary_node, ggml_tensor * mul_node) {
    if (!q8p_enabled() || ggml_cuda_info().devices[ctx.device].warp_size != 64) {
        return false;
    }
    static const bool enabled = [] {
        const char * v = getenv("GGML_Q8_PRODUCERS_CONT");
        return v == nullptr || v[0] != '0';
    }();
    const ggml_unary_op uop = ggml_get_unary_op(unary_node);
    if (!enabled || (uop != GGML_UNARY_OP_SILU && uop != GGML_UNARY_OP_SIGMOID) || unary_node->src[0] != cont_node) {
        return false;
    }
    const ggml_tensor * v = cont_node->src[0];   // strided source of the copy
    const ggml_tensor * g = mul_node->src[0] == unary_node ? mul_node->src[1] : mul_node->src[0];
    if (v->type != GGML_TYPE_F32 || cont_node->type != GGML_TYPE_F32 || v->nb[0] != sizeof(float) || v->ne[3] != 1 ||
            v->nb[1] % sizeof(float) != 0 || v->nb[2] % sizeof(float) != 0 || ggml_nelements(v) != ggml_nelements(cont_node) ||
            !ggml_is_contiguous(cont_node) || !q8p_flat_f32(g) || !q8p_flat_f32(mul_node) ||
            ggml_nelements(g) != ggml_nelements(mul_node) || ggml_nelements(cont_node) != ggml_nelements(mul_node) ||
            !ggml_are_same_shape(g, mul_node) || !ggml_are_same_shape(unary_node, cont_node)) {
        return false;
    }
    // the elementwise index of the copy output must match the flattened mul index: same element count, and the unary
    // output has the copy's shape; the mul operands are contiguous with the mul's shape (no broadcasting)
    if (ggml_nelements(unary_node) != ggml_nelements(mul_node)) {
        return false;
    }
    // the strided source is read in a different order than dst is written: no overlap at all; g elementwise: exact only
    if (q8p_overlap(v, mul_node) || q8p_partial_overlap(g, mul_node)) {
        return false;
    }
    block_q8_1 * y = nullptr;
    int * ysum = nullptr;
    if (!ggml_cuda_mmvq_cache_produce(ctx, mul_node, &y, &ysum)) {
        y = nullptr;
        ysum = nullptr;
    }
    const int64_t k = ggml_nelements(mul_node);
    const int64_t nblk = (k + 255) / 256;
    const int64_t s1 = v->nb[1] / sizeof(float);
    const int64_t s2 = v->nb[2] / sizeof(float);
    if (uop == GGML_UNARY_OP_SILU) {
        gated_q8_strided_kernel<0><<<nblk, 256, 0, ctx.stream()>>>((const float *) v->data, (uint32_t) v->ne[0], (uint32_t) v->ne[1], s1, s2,
                (const float *) g->data, (float *) mul_node->data, y, ysum, (uint32_t) k);
    } else {
        gated_q8_strided_kernel<1><<<nblk, 256, 0, ctx.stream()>>>((const float *) v->data, (uint32_t) v->ne[0], (uint32_t) v->ne[1], s1, s2,
                (const float *) g->data, (float *) mul_node->data, y, ysum, (uint32_t) k);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_unary_mul_q8(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node) {
    const ggml_unary_op uop = ggml_get_unary_op(unary_node);
    if (uop != GGML_UNARY_OP_SILU && uop != GGML_UNARY_OP_SIGMOID) {
        return false;
    }
    const ggml_tensor * x = unary_node->src[0];
    const ggml_tensor * g = mul_node->src[0] == unary_node ? mul_node->src[1] : mul_node->src[0];
    return gated_q8_launch(ctx, x, g, mul_node, uop == GGML_UNARY_OP_SILU ? 0 : 1);
}

#else

bool ggml_cuda_rms_norm_mul_q8(ggml_backend_cuda_context &, const ggml_tensor *, ggml_tensor *) { return false; }
bool ggml_cuda_rms_norm_gated_q8_supported(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *) { return false; }
void ggml_cuda_rms_norm_gated_q8(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, float, const ggml_tensor *, ggml_tensor *) {}
bool ggml_cuda_rms_norm_gated_f16_supported(const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *) { return false; }
void ggml_cuda_rms_norm_gated_f16(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, float, const ggml_tensor *,
        const ggml_tensor *, half *) {}
bool ggml_cuda_swiglu_q8(ggml_backend_cuda_context &, ggml_tensor *) { return false; }
bool ggml_cuda_unary_mul_q8(ggml_backend_cuda_context &, ggml_tensor *, ggml_tensor *) { return false; }
bool ggml_cuda_cont_unary_mul_q8(ggml_backend_cuda_context &, const ggml_tensor *, ggml_tensor *, ggml_tensor *) { return false; }

#endif
