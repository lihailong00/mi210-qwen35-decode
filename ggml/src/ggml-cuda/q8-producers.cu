// Single-row (decode) variants of the kernels that produce GEMV activations, which additionally write the q8_1
// quantization of their output (bit-identical to quantize_q8_1, + per-block int sums) into the per-stream activation
// cache of mmvq.cu. The following single-token GEMVs that consume the tensor then skip their quantize launch.
// The fp32 output is computed with exactly the same arithmetic as the regular kernels.
#include "common.cuh"
#include "unary.cuh"
#include "mmvq-fused-host.cuh"

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

// rms_norm(x) * w for one row, block 1024 / 4 float4 per thread: same arithmetic as rms_norm_f32<1024, 4, true, true>
template <int VPT>
static __global__ void __launch_bounds__(1024) rms_norm_mul_q8_kernel(const float * __restrict__ x, const float * __restrict__ w,
        float * __restrict__ dst, block_q8_1 * __restrict__ y, int * __restrict__ ysum, const int ncols, const float eps) {
    constexpr int block_size = 1024;
    __shared__ float s_sum[32];
    const int tid = threadIdx.x;
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
            tmp += xs[k].x*xs[k].x + xs[k].y*xs[k].y + xs[k].z*xs[k].z + xs[k].w*xs[k].w;
        }
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float scale = rsqrtf(tmp / ncols + eps);

#pragma unroll
    for (int k = 0; k < VPT; k++) {
        const int i = tid + k*block_size;
        if (i < ncols4) {
            float4 r = make_float4(scale*xs[k].x, scale*xs[k].y, scale*xs[k].z, scale*xs[k].w);
            const float4 m = mul4[i];
            r.x *= m.x;
            r.y *= m.y;
            r.z *= m.z;
            r.w *= m.w;
            dst4[i] = r;

            // q8_1: 8 consecutive threads hold one block of 32 values (ncols4 % 8 == 0, so whole groups are active)
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

bool ggml_cuda_rms_norm_mul_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, ggml_tensor * mul_node) {
    if (!q8p_enabled() || ggml_cuda_info().devices[ctx.device].warp_size != 64) {
        return false;
    }
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * w = mul_node->src[0] == rms_norm ? mul_node->src[1] : mul_node->src[0];
    if (!q8p_single_row_f32(x) || !q8p_single_row_f32(mul_node) || !q8p_single_row_f32(rms_norm) ||
            w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || w->ne[0] != x->ne[0] || ggml_nelements(w) != x->ne[0] ||
            ((uintptr_t) w->data) % 16 != 0 || x->ne[0] > 16384) {
        return false;
    }
    block_q8_1 * y = nullptr;
    int * ysum = nullptr;
    if (!ggml_cuda_mmvq_cache_produce(ctx, mul_node, &y, &ysum)) {
        return false;
    }
    float eps;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    rms_norm_mul_q8_kernel<4><<<1, 1024, 0, ctx.stream()>>>((const float *) x->data, (const float *) w->data, (float *) mul_node->data,
            y, ysum, (int) x->ne[0], eps);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

static bool gated_q8_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * g, ggml_tensor * dst, const int op) {
    if (!q8p_enabled() || ggml_cuda_info().devices[ctx.device].warp_size != 64) {
        return false;
    }
    if (!q8p_flat_f32(x) || !q8p_flat_f32(g) || !q8p_flat_f32(dst) ||
            !ggml_are_same_shape(x, dst) || !ggml_are_same_shape(g, dst)) {
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
bool ggml_cuda_swiglu_q8(ggml_backend_cuda_context &, ggml_tensor *) { return false; }
bool ggml_cuda_unary_mul_q8(ggml_backend_cuda_context &, ggml_tensor *, ggml_tensor *) { return false; }

#endif
