#pragma once
// Single-token (ne11 == 1) GEMV kernels with in-kernel q8_1 quantization of the fp32 activation.
// Every workgroup quantizes the whole activation vector into LDS. The quantization is bit-identical to
// quantize_q8_1 (d = amax/127, q = roundf(x/d), ds = (d, d*sum)), so the integer dot products are
// unchanged; only the fp32 accumulation order differs from the stock mul_mat_vec_q.
//
// The includer must include vecdotq.cuh (+ vecdotq-iq3-sign.cuh) and iq3-repack.cuh before this header.

#include "common.cuh"
#include "unary.cuh"

namespace mmvqf {

enum epi_kind : int {
    EPI_NONE       = 0,
    EPI_GLU_GATE   = 1, // dst = silu(other[row]) * acc   (other = output of the gate GEMV, acc = up)
    EPI_GLU_UP     = 2, // dst = silu(acc) * other[row]   (acc = gate, other = output of the up GEMV)
};

// per-stream state for the cooperative activation quantization (zero-initialized, reset by the last workgroup)
struct coop_state {
    unsigned int ready;      // number of quantizer workgroups that finished
    unsigned int pad0[31];
    unsigned int done;       // number of workgroups that copied the q8 vector
    unsigned int pad1[31];
};
static_assert(sizeof(coop_state) == 256, "coop_state layout");
static constexpr int COOP_KMAX = 32768;
// global q8 buffer layout == SoA LDS layout: qs[K] | ds[K/32] (half2) | sum[K/32] (int)

struct params {
    coop_state  * coop;      // nullptr: every workgroup quantizes x itself
    char        * q8;        // global q8 buffer for the cooperative / pre-quantized modes
    int           q8_pre;    // 1: q8 already holds the quantized activation (SoA): just copy it to LDS
    const char  * vx;        // weights
    const char  * vg;        // gate weights (same type/layout as vx) for the fused swiglu, or nullptr
    const char  * mx;        // side data of vx (packed IQ3 meta)
    const char  * mg;        // side data of vg
    const float * x;         // fp32 activation [K]
    const float * residual;  // optional [N], added to the result
    const float * other;     // optional [N], GLU partner (see epi_kind)
    float       * dst;       // [N]
    int K;
    int N;
    int stride_row;          // row stride in units of the weight blocks used by the dot policy
    int epi;
};

// ------------------------------------------------------------------------------------------------
// activation quantization into LDS
// two layouts: AoS block_q8_1 (for the stock vec_dot functions) or SoA (int8 qs[K] 16B aligned, d/ds per block)

// (int8_t) roundf(v / d), bit-exact, mostly without the IEEE division:
// t = v * (1/d) differs from fl(v/d) by < 2.5 ulp (< 1.9e-5 for |t| <= 127.x); rounding to nearest can only differ
// when t is that close to a half-integer, in which case the exact division is used.
static __device__ __forceinline__ int q8_round_div(const float v, const float d, const float inv_d) {
    const float t  = v * inv_d;
    const float fr = __builtin_amdgcn_fractf(fabsf(t));
    if (fabsf(fr - 0.5f) > 3.0e-5f) {
        return (int) rintf(t);
    }
    return (int) roundf(v / d);
}

template <int NT, bool SOA>
static __device__ __forceinline__ void quantize_x_lds(const float * __restrict__ x, char * __restrict__ smem, const int K) {
    const int tid = threadIdx.y*64 + threadIdx.x;
    for (int c = tid; c < K/8; c += NT) {
        const float4 a = ((const float4 *) x)[2*c + 0];
        const float4 b = ((const float4 *) x)[2*c + 1];
        const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        float amax = fabsf(v[0]);
#pragma unroll
        for (int k = 1; k < 8; ++k) {
            amax = fmaxf(amax, fabsf(v[k]));
        }
        amax = fmaxf(amax, __shfl_xor(amax, 1, 64));
        amax = fmaxf(amax, __shfl_xor(amax, 2, 64));
        const float d = amax / 127.0f;
        const float inv_d = 1.0f / d;
        int sum = 0;
        uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int8_t q = amax == 0.0f ? 0 : q8_round_div(v[k], d, inv_d);
            sum += q;
            if (k < 4) {
                w0 |= uint32_t(uint8_t(q)) << (8*k);
            } else {
                w1 |= uint32_t(uint8_t(q)) << (8*(k-4));
            }
        }
        sum += __shfl_xor(sum, 1, 64);
        sum += __shfl_xor(sum, 2, 64);
        const int ib = c / 4;
        const int is = c % 4;
        if constexpr (SOA) {
            int2 * qs = (int2 *) smem;
            qs[c] = make_int2(w0, w1);
            if (is == 0) {
                half2 * ds = (half2 *)(smem + K);
                int   * sm = (int   *)(smem + K + 4*(K/32));
                ds[ib] = make_half2(d, d*sum);
                sm[ib] = sum;
            }
        } else {
            block_q8_1 * ys = (block_q8_1 *) smem;
            int * q32 = (int *) ys[ib].qs;
            q32[2*is + 0] = w0;
            q32[2*is + 1] = w1;
            if (is == 0) {
                ys[ib].ds = make_half2(d, d*sum);
            }
        }
    }
}

template <bool SOA>
static constexpr __host__ __device__ int lds_y_bytes(int K) {
    return SOA ? K + 8*(K/32) : (K/32)*int(sizeof(block_q8_1));
}

// Quantize chunks [c0, c1) (8 floats each) of x into the SoA layout at dst (global memory).
static __device__ __forceinline__ void quantize_x_soa_range(const float * __restrict__ x, char * __restrict__ dst, const int K,
        const int c0, const int c1, const int tid, const int nt) {
    for (int c = c0 + tid; c < c1; c += nt) {
        const float4 a = ((const float4 *) x)[2*c + 0];
        const float4 b = ((const float4 *) x)[2*c + 1];
        const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        float amax = fabsf(v[0]);
#pragma unroll
        for (int k = 1; k < 8; ++k) {
            amax = fmaxf(amax, fabsf(v[k]));
        }
        amax = fmaxf(amax, __shfl_xor(amax, 1, 64));
        amax = fmaxf(amax, __shfl_xor(amax, 2, 64));
        const float d = amax / 127.0f;
        const float inv_d = 1.0f / d;
        int sum = 0;
        uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int8_t q = amax == 0.0f ? 0 : q8_round_div(v[k], d, inv_d);
            sum += q;
            if (k < 4) {
                w0 |= uint32_t(uint8_t(q)) << (8*k);
            } else {
                w1 |= uint32_t(uint8_t(q)) << (8*(k-4));
            }
        }
        sum += __shfl_xor(sum, 1, 64);
        sum += __shfl_xor(sum, 2, 64);
        ((int2 *) dst)[c] = make_int2(w0, w1);
        if (c % 4 == 0) {
            ((half2 *)(dst + K))[c/4]          = make_half2(d, d*sum);
            ((int   *)(dst + K + 4*(K/32)))[c/4] = sum;
        }
    }
}

// standalone quantization of x (K floats) into the global SoA q8 buffer (bit-exact with quantize_q8_1)
static __global__ void __launch_bounds__(256) mmvqf_quantize_soa(const float * __restrict__ x, char * __restrict__ q8, const int K) {
    quantize_x_soa_range(x, q8, K, blockIdx.x*256, min(K/8, (int)(blockIdx.x + 1)*256), threadIdx.x, 256);
}

// Cooperative activation quantization: workgroups 0..nq-1 quantize a slice each into the global buffer and signal;
// every workgroup waits for all slices and copies the SoA q8 vector into LDS. Workgroups are dispatched in order,
// so the quantizer workgroups are always resident and never wait on anything. The last workgroup to pass resets the state.
template <int NT>
static __device__ __forceinline__ void coop_quantize_to_lds(const params & p, char * __restrict__ smem, const int K, const int tid) {
    const int nchunk = K/8;
    const int nq = (nchunk + NT - 1) / NT;
    if ((int) blockIdx.x < nq) {
        quantize_x_soa_range(p.x, p.q8, K, blockIdx.x*NT, min(nchunk, (int)(blockIdx.x + 1)*NT), tid, NT);
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            atomicAdd(&p.coop->ready, 1u);
        }
    }
    if (tid == 0) {
        while (__hip_atomic_load(&p.coop->ready, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < (unsigned int) nq) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
    const int n16 = lds_y_bytes<true>(K) / 16;
    const int4 * src = (const int4 *) p.q8;
    for (int i = tid; i < n16; i += NT) {
        ((int4 *) smem)[i] = src[i];
    }
    if (tid == 0) {
        const unsigned int d = atomicAdd(&p.coop->done, 1u);
        if (d == gridDim.x - 1) {
            atomicExch(&p.coop->ready, 0u);
            atomicExch(&p.coop->done,  0u);
        }
    }
}

// SoA accessors
static __device__ __forceinline__ const int * soa_qs(const char * smem, int ib) {
    return (const int *)(smem + 32*ib);
}
static __device__ __forceinline__ float soa_d(const char * smem, int K, int ib) {
    return __low2float(((const half2 *)(smem + K))[ib]);
}
static __device__ __forceinline__ float2 soa_ds(const char * smem, int K, int ib) {
    return __half22float2(((const half2 *)(smem + K))[ib]);
}
static __device__ __forceinline__ int soa_sum(const char * smem, int K, int ib) {
    return ((const int *)(smem + K + 4*(K/32)))[ib];
}

// ------------------------------------------------------------------------------------------------
// dot policies. Each processes one "chunk" of one row; chunks of a row are t = 0 .. nchunks-1.

static constexpr __host__ __device__ int vdr_of(ggml_type type) {
    switch (type) {
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
        case GGML_TYPE_IQ1_S:   return VDR_IQ1_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ1_M:   return VDR_IQ1_M_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        default:                return 1;
    }
}

template <ggml_type type>
static __device__ __forceinline__ float vec_dot_stock(const void * vx, const block_q8_1 * y, const int kbx, const int iqs) {
    if constexpr (type == GGML_TYPE_Q2_K)    { return vec_dot_q2_K_q8_1(vx, y, kbx, iqs); }
    if constexpr (type == GGML_TYPE_Q3_K)    { return vec_dot_q3_K_q8_1(vx, y, kbx, iqs); }
    if constexpr (type == GGML_TYPE_Q4_K)    { return vec_dot_q4_K_q8_1(vx, y, kbx, iqs); }
    if constexpr (type == GGML_TYPE_Q5_K)    { return vec_dot_q5_K_q8_1(vx, y, kbx, iqs); }
    if constexpr (type == GGML_TYPE_Q6_K)    { return vec_dot_q6_K_q8_1(vx, y, kbx, iqs); }
    if constexpr (type == GGML_TYPE_IQ2_XXS) { return vec_dot_iq2_xxs_q8_1(vx, y, kbx, iqs); }
    if constexpr (type == GGML_TYPE_IQ2_XS)  { return vec_dot_iq2_xs_q8_1(vx, y, kbx, iqs); }
    if constexpr (type == GGML_TYPE_IQ2_S)   { return vec_dot_iq2_s_q8_1(vx, y, kbx, iqs); }
    if constexpr (type == GGML_TYPE_IQ3_XXS) { return vec_dot_iq3_xxs_q8_1(vx, y, kbx, iqs); }
    if constexpr (type == GGML_TYPE_IQ3_S)   { return vec_dot_iq3_s_q8_1(vx, y, kbx, iqs); }
    if constexpr (type == GGML_TYPE_IQ1_S)   { return vec_dot_iq1_s_q8_1(vx, y, kbx, iqs); }
    if constexpr (type == GGML_TYPE_IQ1_M)   { return vec_dot_iq1_m_q8_1(vx, y, kbx, iqs); }
    if constexpr (type == GGML_TYPE_IQ4_NL)  { return vec_dot_iq4_nl_q8_1(vx, y, kbx, iqs); }
    if constexpr (type == GGML_TYPE_IQ4_XS)  { return vec_dot_iq4_xs_q8_1(vx, y, kbx, iqs); }
    if constexpr (type == GGML_TYPE_Q4_0)    { return vec_dot_q4_0_q8_1(vx, y, kbx, iqs); }
    if constexpr (type == GGML_TYPE_Q4_1)    { return vec_dot_q4_1_q8_1(vx, y, kbx, iqs); }
    if constexpr (type == GGML_TYPE_Q5_0)    { return vec_dot_q5_0_q8_1(vx, y, kbx, iqs); }
    if constexpr (type == GGML_TYPE_Q5_1)    { return vec_dot_q5_1_q8_1(vx, y, kbx, iqs); }
    if constexpr (type == GGML_TYPE_Q8_0)    { return vec_dot_q8_0_q8_1(vx, y, kbx, iqs); }
    return 0.0f;
}

// ---- LDS-codebook versions of the codebook vec_dots (same integer math as the stock functions) ----

static __device__ __forceinline__ float dot_iq3_s_tab(const uint32_t * __restrict__ grid, const void * __restrict__ vbq,
        const block_q8_1 * __restrict__ bq8_1, const int kbx, const int iqs) {
    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs = (const uint8_t *) &qs_packed;
    const int qh = bq3->qh[iqs/2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8 = (const uint8_t *) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t g0 = grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
        const uint32_t g1 = grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];
        const uint32_t signs = signs_packed_8[l0/2];
        const int grid_l = iq3_apply_signs(g0, signs & 15u);
        const int grid_h = iq3_apply_signs(g1, signs >> 4);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float dot_iq3_xxs_tab(const uint32_t * __restrict__ grid, const void * __restrict__ vbq,
        const block_q8_1 * __restrict__ bq8_1, const int kbx, const int iqs) {
    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t g0 = grid[q3[l0 + 0]];
        const uint32_t g1 = grid[q3[l0 + 1]];
        const uint8_t sign_code = aux32 >> (7*l0/2);
        const uint32_t signs = sign_code ^ ((__popc(sign_code) & 1u) << 7);
        const int grid_l = iq3_apply_signs(g0, signs & 15u);
        const int grid_h = iq3_apply_signs(g1, signs >> 4);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float dot_iq2_s_tab(const uint32_t * __restrict__ grid, const void * __restrict__ vbq,
        const block_q8_1 * __restrict__ bq8_1, const int kbx, const int iqs) {
    const block_iq2_s * bq2 = (const block_iq2_s *) vbq + kbx;
    const int       qs_packed = get_int_b2(bq2->qs, iqs/2);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;
    const int qh = bq2->qh[iqs/2];
    const int       signs_packed_32 = get_int_b2(bq2->qs, QK_K/32 + iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;
    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int gi = (qs[l0/2] | ((qh << (8-l0)) & 0x300));
        const int g0 = grid[2*gi+0], g1 = grid[2*gi+1];
        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(g0 ^ signs0, signs0);
        const int grid_h = __vsub4(g1 ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        if (l0 < 4) { sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0); sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0); }
        else        { sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1); sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1); }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float dot_iq2_xs_tab(const uint32_t * __restrict__ grid, const void * __restrict__ vbq,
        const block_q8_1 * __restrict__ bq8_1, const int kbx, const int iqs) {
    const block_iq2_xs * bq2 = (const block_iq2_xs *) vbq + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t * q2 = (const uint16_t *) &q2_packed;
    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int gi = q2[l0/2] & 0x1FF;
        const uint32_t gx = grid[2*gi+0], gy = grid[2*gi+1];
        const uint32_t signs = unpack_ksigns(q2[l0/2] >> 9);
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(gx ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(gy ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        if (l0 < 4) { sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0); sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0); }
        else        { sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1); sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1); }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float dot_iq2_xxs_tab(const uint32_t * __restrict__ grid, const void * __restrict__ vbq,
        const block_q8_1 * __restrict__ bq8_1, const int kbx, const int iqs) {
    const block_iq2_xxs * bq2 = (const block_iq2_xxs *) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t * aux8 = (const uint8_t *) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const int gi = aux8[k0/2];
        const uint32_t gx = grid[2*gi+0], gy = grid[2*gi+1];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(gx ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(gy ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

static __device__ __forceinline__ float dot_iq1_m_tab(const uint32_t * __restrict__ grid, const void * __restrict__ vbq,
        const block_q8_1 * __restrict__ bq8_1, const int kbx, const int iqs) {
    const block_iq1_m * bq1 = (const block_iq1_m *) vbq + kbx;
    const int       qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;
    int   sumi[2] = {0};
    float sumf[2] = {0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2*iqs + l0/4] >> (4 * ((l0/2) % 2));
        const int grid_v = grid[qs[l0/2] | ((qhl & 0x07) << 8)];
        const int grid0 = (grid_v >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid_v >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi[l0/4] = ggml_cuda_dp4a(grid0, u0, sumi[l0/4]);
        sumi[l0/4] = ggml_cuda_dp4a(grid1, u1, sumi[l0/4]);
        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f*IQ1M_DELTA/0x08);
        int sumy = 0;
        sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
        sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
        sumf[l0/4] += delta*sumy;
    }
    const uint16_t * sc = (const uint16_t *) bq1->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = __half2float(scale.f16) * __low2float(bq8_1[iqs].ds);
    const int tmp = sc[iqs/2] >> (6*(iqs%2));
    const int sc0 = 2*((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2*((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}

template <ggml_type type> struct tab_info { static constexpr int words = 0; };
template <> struct tab_info<GGML_TYPE_IQ3_S>   { static constexpr int words = 512;  static __device__ __forceinline__ const uint32_t * src() { return (const uint32_t *) iq3s_grid; } };
template <> struct tab_info<GGML_TYPE_IQ3_XXS> { static constexpr int words = 256;  static __device__ __forceinline__ const uint32_t * src() { return (const uint32_t *) iq3xxs_grid; } };
template <> struct tab_info<GGML_TYPE_IQ2_S>   { static constexpr int words = 2048; static __device__ __forceinline__ const uint32_t * src() { return (const uint32_t *) iq2s_grid; } };
template <> struct tab_info<GGML_TYPE_IQ2_XS>  { static constexpr int words = 1024; static __device__ __forceinline__ const uint32_t * src() { return (const uint32_t *) iq2xs_grid; } };
template <> struct tab_info<GGML_TYPE_IQ2_XXS> { static constexpr int words = 512;  static __device__ __forceinline__ const uint32_t * src() { return (const uint32_t *) iq2xxs_grid; } };
template <> struct tab_info<GGML_TYPE_IQ1_M>   { static constexpr int words = 2048; static __device__ __forceinline__ const uint32_t * src() { return (const uint32_t *) iq1s_grid_gpu; } };

// generic policy: stock vec_dot (or its LDS-codebook twin), AoS block_q8_1 activations in LDS
template <ggml_type type>
struct dot_gen {
    static constexpr bool soa_y      = false;
    static constexpr int  qk         = ggml_cuda_type_traits<type>::qk;
    static constexpr int  qi         = ggml_cuda_type_traits<type>::qi;
    static constexpr int  vdr        = vdr_of(type);
    static constexpr int  cpb        = qi/vdr;          // chunks per block
    static constexpr int  tab_words  = tab_info<type>::words;

    static __device__ __forceinline__ void load_table(uint32_t * tab, const int tid, const int nt) {
        if constexpr (tab_words > 0) {
            const uint32_t * g = tab_info<type>::src();
            for (int i = tid; i < tab_words; i += nt) {
                tab[i] = g[i];
            }
        }
    }

    static __device__ __forceinline__ float dot(const char * vx, const char * mx, const int row, const int t,
            const char * smem, const uint32_t * tab, const int stride_row, const int K) {
        const int kbx = t / cpb;
        const int kqs = vdr*(t % cpb);
        const block_q8_1 * y = (const block_q8_1 *) smem + kbx*(qk/QK8_1);
        const int ib = row*stride_row + kbx;
        if constexpr (type == GGML_TYPE_IQ3_S)   { return dot_iq3_s_tab  (tab, vx, y, ib, kqs); }
        if constexpr (type == GGML_TYPE_IQ3_XXS) { return dot_iq3_xxs_tab(tab, vx, y, ib, kqs); }
        if constexpr (type == GGML_TYPE_IQ2_S)   { return dot_iq2_s_tab  (tab, vx, y, ib, kqs); }
        if constexpr (type == GGML_TYPE_IQ2_XS)  { return dot_iq2_xs_tab (tab, vx, y, ib, kqs); }
        if constexpr (type == GGML_TYPE_IQ2_XXS) { return dot_iq2_xxs_tab(tab, vx, y, ib, kqs); }
        if constexpr (type == GGML_TYPE_IQ1_M)   { return dot_iq1_m_tab  (tab, vx, y, ib, kqs); }
        if constexpr (tab_words == 0)            { return vec_dot_stock<type>(vx, y, ib, kqs); }
        GGML_UNUSED_VARS(mx, K);
        return 0.0f;
    }
};

// retained packed IQ3 (4-bit expanded codes, 16 B per 32 weights + 6 B meta per 256), SoA activations
template <bool xxs>
struct dot_pk {
    static constexpr bool soa_y     = true;
    static constexpr int  cpb       = 8;   // 8 groups of 32 per 256 block
    static constexpr int  qk        = 256;
    static constexpr int  tab_words = 0;

    static __device__ __forceinline__ void load_table(uint32_t *, const int, const int) {}

    static __device__ __forceinline__ float dot(const char * vx, const char * mx, const int row, const int t,
            const char * smem, const uint32_t *, const int stride_row, const int K) {
        // stride_row = groups per row
        const size_t pos = (size_t) row*stride_row + t;
        const uint4 c = ((const uint4 *) vx)[pos];
        const codex_iq3_meta * m = (const codex_iq3_meta *) mx + pos/8;
        const int * y = soa_qs(smem, t);
        const int4 y0 = ((const int4 *) y)[0];
        const int4 y1 = ((const int4 *) y)[1];
        const int g = t & 7;
        const int sc = (m->scales[g/2] >> ((g & 1)*4)) & 15;
        int value = 0;
        if constexpr (xxs) {
            alignas(4) static constexpr int8_t lv[16] = { 4, 12, 20, 28, 36, 44, 52, 62, -4,-12,-20,-28,-36,-44,-52,-62 };
            const int2 v0 = get_int_from_table_16(c.x, lv);
            const int2 v1 = get_int_from_table_16(c.y, lv);
            const int2 v2 = get_int_from_table_16(c.z, lv);
            const int2 v3 = get_int_from_table_16(c.w, lv);
            value = ggml_cuda_dp4a(v0.x, y0.x, value);
            value = ggml_cuda_dp4a(v0.y, y0.y, value);
            value = ggml_cuda_dp4a(v1.x, y0.z, value);
            value = ggml_cuda_dp4a(v1.y, y0.w, value);
            value = ggml_cuda_dp4a(v2.x, y1.x, value);
            value = ggml_cuda_dp4a(v2.y, y1.y, value);
            value = ggml_cuda_dp4a(v3.x, y1.z, value);
            value = ggml_cuda_dp4a(v3.y, y1.w, value);
            value = (sc*value + value/2)/2;
        } else {
            value = ggml_cuda_dp4a( c.x       & 0x0f0f0f0fu, y0.x, value);
            value = ggml_cuda_dp4a((c.x >> 4) & 0x0f0f0f0fu, y0.y, value);
            value = ggml_cuda_dp4a( c.y       & 0x0f0f0f0fu, y0.z, value);
            value = ggml_cuda_dp4a((c.y >> 4) & 0x0f0f0f0fu, y0.w, value);
            value = ggml_cuda_dp4a( c.z       & 0x0f0f0f0fu, y1.x, value);
            value = ggml_cuda_dp4a((c.z >> 4) & 0x0f0f0f0fu, y1.y, value);
            value = ggml_cuda_dp4a( c.w       & 0x0f0f0f0fu, y1.z, value);
            value = ggml_cuda_dp4a((c.w >> 4) & 0x0f0f0f0fu, y1.w, value);
            value = 2*value - 15*soa_sum(smem, K, t);
            value *= 1 + 2*sc;
        }
        const float d = __half2float(m->d) * soa_d(smem, K, t);
        return d*value;
    }
};

// IQ4_XS with SoA activations: chunk = one 32-weight sub-block
struct dot_iq4xs {
    static constexpr bool soa_y     = true;
    static constexpr int  cpb       = 8;
    static constexpr int  qk        = 256;
    static constexpr int  tab_words = 0;

    static __device__ __forceinline__ void load_table(uint32_t *, const int, const int) {}

    static __device__ __forceinline__ float dot(const char * vx, const char *, const int row, const int t,
            const char * smem, const uint32_t *, const int stride_row, const int K) {
        const int kbx = t / 8;
        const int ib  = t % 8;
        const block_iq4_xs * b = (const block_iq4_xs *) vx + (row*stride_row + kbx);
        const uint2 q01 = ((const uint2 *)(b->qs + 16*ib))[0];
        const uint2 q23 = ((const uint2 *)(b->qs + 16*ib))[1];
        const int * y = soa_qs(smem, t);
        const int4 y0 = ((const int4 *) y)[0];
        const int4 y1 = ((const int4 *) y)[1];
        int sumi = 0;
        const int2 v0 = get_int_from_table_16(q01.x, kvalues_iq4nl);
        const int2 v1 = get_int_from_table_16(q01.y, kvalues_iq4nl);
        const int2 v2 = get_int_from_table_16(q23.x, kvalues_iq4nl);
        const int2 v3 = get_int_from_table_16(q23.y, kvalues_iq4nl);
        sumi = ggml_cuda_dp4a(v0.x, y0.x, sumi);
        sumi = ggml_cuda_dp4a(v0.y, y1.x, sumi);
        sumi = ggml_cuda_dp4a(v1.x, y0.y, sumi);
        sumi = ggml_cuda_dp4a(v1.y, y1.y, sumi);
        sumi = ggml_cuda_dp4a(v2.x, y0.z, sumi);
        sumi = ggml_cuda_dp4a(v2.y, y1.z, sumi);
        sumi = ggml_cuda_dp4a(v3.x, y0.w, sumi);
        sumi = ggml_cuda_dp4a(v3.y, y1.w, sumi);
        const int ls = ((b->scales_l[ib/2] >> (4*(ib & 1))) & 0x0F) | (((b->scales_h >> (2*ib)) & 0x03) << 4);
        sumi *= ls - 32;
        const float d = __half2float(b->d) * soa_d(smem, K, t);
        return d * sumi;
    }
};

// Q4_K with SoA activations: chunk = pair of 32-weight sub-blocks (32 bytes of qs, 16B aligned)
struct dot_q4k {
    static constexpr bool soa_y     = true;
    static constexpr int  cpb       = 4;
    static constexpr int  qk        = 256;
    static constexpr int  tab_words = 0;

    static __device__ __forceinline__ void load_table(uint32_t *, const int, const int) {}

    static __device__ __forceinline__ float dot(const char * vx, const char *, const int row, const int t,
            const char * smem, const uint32_t *, const int stride_row, const int K) {
        const int kbx = t / 4;
        const int j   = t % 4;      // sub-block pair: sub-blocks 2j (low nibbles), 2j+1 (high nibbles)
        const block_q4_K * b = (const block_q4_K *) vx + (row*stride_row + kbx);
        const int4 qa = ((const int4 *)(b->qs + 32*j))[0];
        const int4 qb = ((const int4 *)(b->qs + 32*j))[1];
        const int ib0 = 2*(t);       // q8 block index of sub-block 2j in the row: kbx*8 + 2j
        const int * ya = soa_qs(smem, ib0 + 0);
        const int * yb = soa_qs(smem, ib0 + 1);
        const int4 ya0 = ((const int4 *) ya)[0];
        const int4 ya1 = ((const int4 *) ya)[1];
        const int4 yb0 = ((const int4 *) yb)[0];
        const int4 yb1 = ((const int4 *) yb)[1];
        const int q[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
        const int ua[8] = {ya0.x, ya0.y, ya0.z, ya0.w, ya1.x, ya1.y, ya1.z, ya1.w};
        const int ub[8] = {yb0.x, yb0.y, yb0.z, yb0.w, yb1.x, yb1.y, yb1.z, yb1.w};
        int dot_a = 0, dot_b = 0, sum_a = 0, sum_b = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            dot_a = ggml_cuda_dp4a( q[k]       & 0x0F0F0F0F, ua[k], dot_a);
            dot_b = ggml_cuda_dp4a((q[k] >> 4) & 0x0F0F0F0F, ub[k], dot_b);
        }
        sum_a = soa_sum(smem, K, ib0 + 0);
        sum_b = soa_sum(smem, K, ib0 + 1);
        // scales/mins of sub-blocks 2j and 2j+1 (get_scale_min_k4)
        const uint8_t * s = b->scales;
        int sc_a, m_a, sc_b, m_b;
        const int ja = 2*j, jb = 2*j + 1;
        if (j < 2) {
            sc_a = s[ja] & 63; m_a = s[ja + 4] & 63;
            sc_b = s[jb] & 63; m_b = s[jb + 4] & 63;
        } else {
            sc_a = (s[ja + 4] & 0xF) | ((s[ja - 4] >> 6) << 4); m_a = (s[ja + 4] >> 4) | ((s[ja] >> 6) << 4);
            sc_b = (s[jb + 4] & 0xF) | ((s[jb - 4] >> 6) << 4); m_b = (s[jb + 4] >> 4) | ((s[jb] >> 6) << 4);
        }
        const float2 dm = __half22float2(b->dm);
        const float da = soa_d(smem, K, ib0 + 0);
        const float db = soa_d(smem, K, ib0 + 1);
        const float sumf_d = da*(dot_a*sc_a) + db*(dot_b*sc_b);
        const float sumf_m = da*(sum_a*m_a)  + db*(sum_b*m_b);
        return dm.x*sumf_d - dm.y*sumf_m;
    }
};

// ------------------------------------------------------------------------------------------------
// the kernel: persistent workgroups, L lanes per row, R rows per lane group, NW waves per workgroup

template <class D, int NW, int L, int R, bool has_gate, int NCH = 0>
__launch_bounds__(NW*64, 1)
static __global__ void mmvqf_kernel(const params p) {
    extern __shared__ __align__(16) char mmvqf_smem[];
    constexpr int NT = NW*64;
    constexpr int G  = 64/L;

    const int K = p.K;
    char     * smem = mmvqf_smem;
    uint32_t * tab  = (uint32_t *)(smem + ((lds_y_bytes<D::soa_y>(K) + 15) & ~15));

    const int tid = threadIdx.y*64 + threadIdx.x;
    D::load_table(tab, tid, NT);
    quantize_x_lds<NT, D::soa_y>(p.x, smem, K);
    __syncthreads();

    const int lane    = threadIdx.x % L;
    const int grp     = threadIdx.x / L;
    const int nch     = (K / D::qk) * D::cpb;
    const int ngroups = (p.N + R - 1) / R;

    for (int g = (blockIdx.x*NW + threadIdx.y)*G + grp; g < ngroups; g += gridDim.x*NW*G) {
        const int row0 = g*R;
        int rows[R];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            rows[i] = min(row0 + i, p.N - 1);
        }
        float acc[R]  = {0.0f};
        float accg[R] = {0.0f};
        if constexpr (NCH > 0) {
            // compile-time trip count: fully unrolled so that all weight loads of the row group can be issued early
#pragma unroll
            for (int k = 0; k < NCH; ++k) {
                const int t = lane + k*L;
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    acc[i] += D::dot(p.vx, p.mx, rows[i], t, smem, tab, p.stride_row, K);
                    if constexpr (has_gate) {
                        accg[i] += D::dot(p.vg, p.mg, rows[i], t, smem, tab, p.stride_row, K);
                    }
                }
            }
        } else {
            for (int t = lane; t < nch; t += L) {
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    acc[i] += D::dot(p.vx, p.mx, rows[i], t, smem, tab, p.stride_row, K);
                    if constexpr (has_gate) {
                        accg[i] += D::dot(p.vg, p.mg, rows[i], t, smem, tab, p.stride_row, K);
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < R; ++i) {
            acc[i] = warp_reduce_sum<L>(acc[i]);
            if constexpr (has_gate) {
                accg[i] = warp_reduce_sum<L>(accg[i]);
            }
        }
        if (lane < R) {
            float r = acc[0];
            float rg = accg[0];
#pragma unroll
            for (int i = 1; i < R; ++i) {
                if (lane == i) {
                    r  = acc[i];
                    rg = accg[i];
                }
            }
            const int row = row0 + lane;
            if (row < p.N) {
                if constexpr (has_gate) {
                    r *= ggml_cuda_op_silu_single(rg);
                }
                if (p.epi == EPI_GLU_GATE) {
                    r = ggml_cuda_op_silu_single(p.other[row]) * r;
                } else if (p.epi == EPI_GLU_UP) {
                    r = ggml_cuda_op_silu_single(r) * p.other[row];
                }
                if (p.residual) {
                    r += p.residual[row];
                }
                p.dst[row] = r;
            }
        }
    }
}

template <class D>
static constexpr __host__ __device__ int smem_bytes(int K) {
    return ((lds_y_bytes<D::soa_y>(K) + 15) & ~15) + 4*D::tab_words;
}

// ------------------------------------------------------------------------------------------------
// software-pipelined kernel: policies split into load (raw registers only, no ALU on loaded data) and compute.
// The next chunk (or the first chunk of the next row group) is loaded before the current one is consumed,
// and the very first loads are issued before the activation prologue.

// packed IQ3 (codex layout: codes[count*32 u32] then meta[count] 6 B)
// MM: meta layout. 0 = codex 6-byte meta (2 small loads), 1 = 8-byte meta {d, pad, scales[4]} (one 8-byte load),
// 2 = no meta loads (timing experiments only, wrong results)
template <bool xxs, int MM = 0>
struct pipe_pk {
    static constexpr bool soa_y     = true;
    static constexpr int  cpb       = 8;
    static constexpr int  qk        = 256;
    static constexpr int  tab_words = 0;
    struct raw_t { uint4 c; uint32_t d; uint32_t s; };

    static __device__ __forceinline__ void load_table(uint32_t *, const int, const int) {}

    static __device__ __forceinline__ raw_t load(const char * vx, const char * mx, const int row, const int t, const int stride_row) {
        const size_t pos = (size_t) row*stride_row + t;
        raw_t r;
        r.c = ((const uint4 *) vx)[pos];
        if constexpr (MM == 0) {
            const codex_iq3_meta * m = (const codex_iq3_meta *) mx + pos/8;
            r.d = *(const uint16_t *) &m->d;
            r.s = m->scales[(t & 7)/2];
        } else if constexpr (MM == 1) {
            const uint2 m = ((const uint2 *) mx)[pos/8];
            r.d = m.x;
            r.s = m.y;
        } else {
            r.d = 0x3c00;
            r.s = 0x55555555;
        }
        return r;
    }

    static __device__ __forceinline__ float compute(const raw_t & r, const char * smem, const int t, const int K, const uint32_t *) {
        const int * y = soa_qs(smem, t);
        const int4 y0 = ((const int4 *) y)[0];
        const int4 y1 = ((const int4 *) y)[1];
        const uint32_t sbyte = MM == 0 ? r.s : (r.s >> (8*((t & 7)/2)));
        const int sc = (sbyte >> ((t & 1)*4)) & 15;
        int value = 0;
        if constexpr (xxs) {
            alignas(4) static constexpr int8_t lv[16] = { 4, 12, 20, 28, 36, 44, 52, 62, -4,-12,-20,-28,-36,-44,-52,-62 };
            const int2 v0 = get_int_from_table_16(r.c.x, lv);
            const int2 v1 = get_int_from_table_16(r.c.y, lv);
            const int2 v2 = get_int_from_table_16(r.c.z, lv);
            const int2 v3 = get_int_from_table_16(r.c.w, lv);
            value = ggml_cuda_dp4a(v0.x, y0.x, value);
            value = ggml_cuda_dp4a(v0.y, y0.y, value);
            value = ggml_cuda_dp4a(v1.x, y0.z, value);
            value = ggml_cuda_dp4a(v1.y, y0.w, value);
            value = ggml_cuda_dp4a(v2.x, y1.x, value);
            value = ggml_cuda_dp4a(v2.y, y1.y, value);
            value = ggml_cuda_dp4a(v3.x, y1.z, value);
            value = ggml_cuda_dp4a(v3.y, y1.w, value);
            value = (sc*value + value/2)/2;
        } else {
            value = ggml_cuda_dp4a( r.c.x       & 0x0f0f0f0fu, y0.x, value);
            value = ggml_cuda_dp4a((r.c.x >> 4) & 0x0f0f0f0fu, y0.y, value);
            value = ggml_cuda_dp4a( r.c.y       & 0x0f0f0f0fu, y0.z, value);
            value = ggml_cuda_dp4a((r.c.y >> 4) & 0x0f0f0f0fu, y0.w, value);
            value = ggml_cuda_dp4a( r.c.z       & 0x0f0f0f0fu, y1.x, value);
            value = ggml_cuda_dp4a((r.c.z >> 4) & 0x0f0f0f0fu, y1.y, value);
            value = ggml_cuda_dp4a( r.c.w       & 0x0f0f0f0fu, y1.z, value);
            value = ggml_cuda_dp4a((r.c.w >> 4) & 0x0f0f0f0fu, y1.w, value);
            value = 2*value - 15*soa_sum(smem, K, t);
            value *= 1 + 2*sc;
        }
        const float d = __half2float(__ushort_as_half((uint16_t) (r.d & 0xFFFF))) * soa_d(smem, K, t);
        return d*value;
    }
};

// IQ4_XS, original layout: header (d, scales_h, scales_l) 8 B + 16 B of qs per 32-weight sub-block
struct pipe_iq4xs {
    static constexpr bool soa_y     = true;
    static constexpr int  cpb       = 8;
    static constexpr int  qk        = 256;
    static constexpr int  tab_words = 0;
    struct raw_t { uint2 h; uint2 a; uint2 b; };

    static __device__ __forceinline__ void load_table(uint32_t *, const int, const int) {}

    static __device__ __forceinline__ raw_t load(const char * vx, const char *, const int row, const int t, const int stride_row) {
        const block_iq4_xs * blk = (const block_iq4_xs *) vx + (row*stride_row + t/8);
        raw_t r;
        r.h = *(const uint2 *) blk;
        const uint2 * q = (const uint2 *)(blk->qs + 16*(t % 8));
        r.a = q[0];
        r.b = q[1];
        return r;
    }

    static __device__ __forceinline__ float compute(const raw_t & r, const char * smem, const int t, const int K, const uint32_t *) {
        const int ib = t % 8;
        const int * y = soa_qs(smem, t);
        const int4 y0 = ((const int4 *) y)[0];
        const int4 y1 = ((const int4 *) y)[1];
        const int2 v0 = get_int_from_table_16(r.a.x, kvalues_iq4nl);
        const int2 v1 = get_int_from_table_16(r.a.y, kvalues_iq4nl);
        const int2 v2 = get_int_from_table_16(r.b.x, kvalues_iq4nl);
        const int2 v3 = get_int_from_table_16(r.b.y, kvalues_iq4nl);
        int sumi = 0;
        sumi = ggml_cuda_dp4a(v0.x, y0.x, sumi);
        sumi = ggml_cuda_dp4a(v0.y, y1.x, sumi);
        sumi = ggml_cuda_dp4a(v1.x, y0.y, sumi);
        sumi = ggml_cuda_dp4a(v1.y, y1.y, sumi);
        sumi = ggml_cuda_dp4a(v2.x, y0.z, sumi);
        sumi = ggml_cuda_dp4a(v2.y, y1.z, sumi);
        sumi = ggml_cuda_dp4a(v3.x, y0.w, sumi);
        sumi = ggml_cuda_dp4a(v3.y, y1.w, sumi);
        // h.x = d | scales_h << 16, h.y = scales_l[4]
        const int sl = (r.h.y >> (4*ib)) & 0x0F;                 // scales_l[ib/2] >> 4*(ib&1)
        const int sh = (r.h.x >> (16 + 2*ib)) & 0x03;
        const int ls = sl | (sh << 4);
        sumi *= ls - 32;
        const float d = __half2float(__ushort_as_half((uint16_t)(r.h.x & 0xFFFF))) * soa_d(smem, K, t);
        return d * sumi;
    }
};

// Q4_K, original layout: chunk = 32 bytes of qs (sub-blocks 2j: low nibbles, 2j+1: high nibbles) + 16 B header
struct pipe_q4k {
    static constexpr bool soa_y     = true;
    static constexpr int  cpb       = 4;
    static constexpr int  qk        = 256;
    static constexpr int  tab_words = 0;
    struct raw_t { uint4 h; int4 qa; int4 qb; };

    static __device__ __forceinline__ void load_table(uint32_t *, const int, const int) {}

    static __device__ __forceinline__ raw_t load(const char * vx, const char *, const int row, const int t, const int stride_row) {
        const block_q4_K * blk = (const block_q4_K *) vx + (row*stride_row + t/4);
        raw_t r;
        r.h  = *(const uint4 *) blk;
        r.qa = ((const int4 *)(blk->qs + 32*(t % 4)))[0];
        r.qb = ((const int4 *)(blk->qs + 32*(t % 4)))[1];
        return r;
    }

    static __device__ __forceinline__ float compute(const raw_t & r, const char * smem, const int t, const int K, const uint32_t *) {
        const int j   = t % 4;
        const int ib0 = 2*t;
        const int * ya = soa_qs(smem, ib0 + 0);
        const int * yb = soa_qs(smem, ib0 + 1);
        const int4 ya0 = ((const int4 *) ya)[0];
        const int4 ya1 = ((const int4 *) ya)[1];
        const int4 yb0 = ((const int4 *) yb)[0];
        const int4 yb1 = ((const int4 *) yb)[1];
        const int q[8]  = {r.qa.x, r.qa.y, r.qa.z, r.qa.w, r.qb.x, r.qb.y, r.qb.z, r.qb.w};
        const int ua[8] = {ya0.x, ya0.y, ya0.z, ya0.w, ya1.x, ya1.y, ya1.z, ya1.w};
        const int ub[8] = {yb0.x, yb0.y, yb0.z, yb0.w, yb1.x, yb1.y, yb1.z, yb1.w};
        int dot_a = 0, dot_b = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            dot_a = ggml_cuda_dp4a( q[k]       & 0x0F0F0F0F, ua[k], dot_a);
            dot_b = ggml_cuda_dp4a((q[k] >> 4) & 0x0F0F0F0F, ub[k], dot_b);
        }
        const int sum_a = soa_sum(smem, K, ib0 + 0);
        const int sum_b = soa_sum(smem, K, ib0 + 1);
        // header bytes: 0..3 dm, 4..15 scales[12]
        const uint32_t hw[4] = {r.h.x, r.h.y, r.h.z, r.h.w};
        auto sbyte = [&](int i) -> int { return (hw[1 + i/4] >> (8*(i % 4))) & 0xFF; };
        const int ja = 2*j, jb = 2*j + 1;
        int sc_a, m_a, sc_b, m_b;
        if (j < 2) {
            sc_a = sbyte(ja) & 63; m_a = sbyte(ja + 4) & 63;
            sc_b = sbyte(jb) & 63; m_b = sbyte(jb + 4) & 63;
        } else {
            sc_a = (sbyte(ja + 4) & 0xF) | ((sbyte(ja - 4) >> 6) << 4); m_a = (sbyte(ja + 4) >> 4) | ((sbyte(ja) >> 6) << 4);
            sc_b = (sbyte(jb + 4) & 0xF) | ((sbyte(jb - 4) >> 6) << 4); m_b = (sbyte(jb + 4) >> 4) | ((sbyte(jb) >> 6) << 4);
        }
        const float2 dm = __half22float2(*(const half2 *) &hw[0]);
        const float da = soa_d(smem, K, ib0 + 0);
        const float db = soa_d(smem, K, ib0 + 1);
        const float sumf_d = da*(dot_a*sc_a) + db*(dot_b*sc_b);
        const float sumf_m = da*(sum_a*m_a)  + db*(sum_b*m_b);
        return dm.x*sumf_d - dm.y*sumf_m;
    }
};

// epilogue helper shared by the kernels: lane i < R of the L-group writes row row0+i
template <int L, int R, bool has_gate>
static __device__ __forceinline__ void epilogue(const params & p, float (&acc)[R], float (&accg)[R], const int lane, const int row0) {
#pragma unroll
    for (int i = 0; i < R; ++i) {
        acc[i] = warp_reduce_sum<L>(acc[i]);
        if constexpr (has_gate) {
            accg[i] = warp_reduce_sum<L>(accg[i]);
        }
    }
    if (lane < R) {
        float r  = acc[0];
        float rg = accg[0];
#pragma unroll
        for (int i = 1; i < R; ++i) {
            if (lane == i) {
                r  = acc[i];
                rg = accg[i];
            }
        }
        const int row = row0 + lane;
        if (row < p.N) {
            if constexpr (has_gate) {
                r *= ggml_cuda_op_silu_single(rg);
            }
            if (p.epi == EPI_GLU_GATE) {
                r = ggml_cuda_op_silu_single(p.other[row]) * r;
            } else if (p.epi == EPI_GLU_UP) {
                r = ggml_cuda_op_silu_single(r) * p.other[row];
            }
            if (p.residual) {
                r += p.residual[row];
            }
            p.dst[row] = r;
        }
    }
}

// Ping-pong pipelined kernel. Every L-lane group walks a flat sequence of steps (row group j, chunk k);
// the loads of step s+1 are issued (unconditionally, clamped) before step s is computed, alternating between
// two register buffers (manual 2x unroll) so that the waitcnt pass can wait precisely for the older buffer.
// Residual / GLU-partner values of all rows of the workgroup are staged in LDS during the prologue so that the
// epilogue issues no global loads. Requires nch % L == 0.
template <class D, int NW, int L, int R, bool has_gate>
__launch_bounds__(NW*64, 1)
static __global__ void mmvqf_pp_kernel(const params p) {
    extern __shared__ __align__(16) char mmvqf_smem[];
    constexpr int NT = NW*64;
    constexpr int G  = 64/L;
    constexpr int RW = NW*G*R;    // rows per workgroup iteration
    using raw_t = typename D::raw_t;

    const int K = p.K;
    char     * smem = mmvqf_smem;
    const int ybytes = (lds_y_bytes<D::soa_y>(K) + 15) & ~15;
    uint32_t * tab  = (uint32_t *)(smem + ybytes);

    const int lane    = threadIdx.x % L;
    const int grp     = threadIdx.x / L;
    const int nch     = (K / D::qk) * D::cpb;
    const int spg     = nch / L;                       // steps per row group
    const int ngroups = (p.N + R - 1) / R;
    const int gstride = gridDim.x*NW*G;
    const int niter   = (ngroups + gstride - 1) / gstride;
    const int nsteps  = niter*spg;
    const int wg_g0   = blockIdx.x*NW*G;
    const int g0      = wg_g0 + threadIdx.y*G + grp;
    float * side = (float *)(smem + ybytes + 4*D::tab_words);   // [niter][RW] residual/other values

    raw_t bufA[R], bufB[R];
    raw_t bufAg[has_gate ? R : 1], bufBg[has_gate ? R : 1];

    int jn = 0, kn = 0;
    auto load_step = [&](raw_t (&b)[R], raw_t (&bg)[has_gate ? R : 1]) {
        const int row0 = min(g0 + min(jn, niter - 1)*gstride, ngroups - 1)*R;
        const int t = lane + kn*L;
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const int row = min(row0 + i, p.N - 1);
            b[i] = D::load(p.vx, p.mx, row, t, p.stride_row);
            if constexpr (has_gate) {
                bg[i] = D::load(p.vg, p.mg, row, t, p.stride_row);
            }
        }
        if (++kn == spg) { kn = 0; ++jn; }
    };

    // first loads before the prologue
    load_step(bufA, bufAg);

    const int tid = threadIdx.y*64 + threadIdx.x;
    D::load_table(tab, tid, NT);
    quantize_x_lds<NT, D::soa_y>(p.x, smem, K);
    if (p.residual || p.other) {
        const float * src = p.residual ? p.residual : p.other;
        for (int i = tid; i < niter*RW; i += NT) {
            const int it = i / RW, r = i % RW;
            const int row = min((wg_g0 + it*gstride)*R + r, p.N - 1);
            side[i] = src[row];
        }
    }
    __syncthreads();

    float acc[R]  = {0.0f};
    float accg[R] = {0.0f};
    int j = 0, k = 0;

    auto compute_step = [&](raw_t (&b)[R], raw_t (&bg)[has_gate ? R : 1]) {
        const int t = lane + k*L;
#pragma unroll
        for (int i = 0; i < R; ++i) {
            acc[i] += D::compute(b[i], smem, t, K, tab);
            if constexpr (has_gate) {
                accg[i] += D::compute(bg[i], smem, t, K, tab);
            }
        }
        if (k == spg - 1) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
                acc[i] = warp_reduce_sum<L>(acc[i]);
                if constexpr (has_gate) {
                    accg[i] = warp_reduce_sum<L>(accg[i]);
                }
            }
            const int g = g0 + j*gstride;
            if (lane < R && g < ngroups) {
                float r  = acc[0];
                float rg = accg[0];
#pragma unroll
                for (int i = 1; i < R; ++i) {
                    if (lane == i) {
                        r  = acc[i];
                        rg = accg[i];
                    }
                }
                const int row = g*R + lane;
                if (row < p.N) {
                    if constexpr (has_gate) {
                        r *= ggml_cuda_op_silu_single(rg);
                    }
                    const float sv = side[j*RW + (g - wg_g0 - j*gstride)*R + lane];
                    if (p.epi == EPI_GLU_GATE) {
                        r = ggml_cuda_op_silu_single(sv) * r;
                    } else if (p.epi == EPI_GLU_UP) {
                        r = ggml_cuda_op_silu_single(r) * sv;
                    } else if (p.residual) {
                        r += sv;
                    }
                    p.dst[row] = r;
                }
            }
#pragma unroll
            for (int i = 0; i < R; ++i) {
                acc[i]  = 0.0f;
                accg[i] = 0.0f;
            }
        }
        if (++k == spg) { k = 0; ++j; }
    };

    for (int s = 0; s < nsteps; s += 2) {
        load_step(bufB, bufBg);          // step s+1 (clamped when past the end)
        compute_step(bufA, bufAg);       // step s
        if (s + 1 >= nsteps) {
            break;
        }
        load_step(bufA, bufAg);          // step s+2
        compute_step(bufB, bufBg);       // step s+1
    }
}

// Batch kernel: each L-lane group processes its rows in batches of U chunks per row; all loads of a batch are issued
// together (the first batch before the activation prologue), then computed. Latency is hidden by the other waves.
// Requires nch % L == 0. Residual / GLU-partner values are staged in LDS during the prologue.
template <class D, int NW, int L, int R, int U, bool has_gate, bool YG = false>
__launch_bounds__(NW*64, 1)
static __global__ void mmvqf_batch_kernel(const params p) {
    extern __shared__ __align__(16) char mmvqf_smem[];
    constexpr int NT = NW*64;
    constexpr int G  = 64/L;
    constexpr int RW = NW*G*R;
    using raw_t = typename D::raw_t;

    const int K = p.K;
    char     * smem   = mmvqf_smem;
    const int  ybytes = (lds_y_bytes<D::soa_y>(K) + 15) & ~15;
    uint32_t * tab    = (uint32_t *)(smem + ybytes);
    float    * side   = (float *)(smem + ybytes + 4*D::tab_words);

    const int lane    = threadIdx.x % L;
    const int grp     = threadIdx.x / L;
    const int nch     = (K / D::qk) * D::cpb;
    const int spr     = nch / L;
    const int nb      = (spr + U - 1) / U;
    const int ngroups = (p.N + R - 1) / R;
    const int gstride = gridDim.x*NW*G;
    const int niter   = (ngroups + gstride - 1) / gstride;
    const int wg_g0   = blockIdx.x*NW*G;
    const int g0      = wg_g0 + threadIdx.y*G + grp;

    raw_t raw[U][R];
    raw_t rawg[U][has_gate ? R : 1];

    auto load_batch = [&](const int j, const int kb) {
        const int row0 = min(g0 + min(j, niter - 1)*gstride, ngroups - 1)*R;
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int t = lane + min(kb + u, spr - 1)*L;
#pragma unroll
            for (int i = 0; i < R; ++i) {
                const int row = min(row0 + i, p.N - 1);
                raw[u][i] = D::load(p.vx, p.mx, row, t, p.stride_row);
                if constexpr (has_gate) {
                    rawg[u][i] = D::load(p.vg, p.mg, row, t, p.stride_row);
                }
            }
        }
    };

    load_batch(0, 0);

    const int tid = threadIdx.y*64 + threadIdx.x;
    D::load_table(tab, tid, NT);
    const char * ybase = YG ? p.q8 : smem;
    if constexpr (YG) {
        // activation already quantized (SoA) in global memory: no prologue
    } else if constexpr (D::soa_y) {
        if (p.q8_pre) {
            const int n16 = lds_y_bytes<true>(K) / 16;
            for (int i = tid; i < n16; i += NT) {
                ((int4 *) smem)[i] = ((const int4 *) p.q8)[i];
            }
        } else if (p.coop) {
            coop_quantize_to_lds<NT>(p, smem, K, tid);
        } else {
            quantize_x_lds<NT, true>(p.x, smem, K);
        }
    } else {
        quantize_x_lds<NT, false>(p.x, smem, K);
    }
    if constexpr (!YG) {
        if (p.residual || p.other) {
            const float * src = p.residual ? p.residual : p.other;
            for (int i = tid; i < niter*RW; i += NT) {
                const int it = i / RW, r = i % RW;
                const int row = min((wg_g0 + it*gstride)*R + r, p.N - 1);
                side[i] = src[row];
            }
        }
        __syncthreads();
    } else if constexpr (D::tab_words > 0) {
        __syncthreads();
    }

    float acc[R]  = {0.0f};
    float accg[R] = {0.0f};
    int j = 0, kb = 0;
    const int nsteps = niter*nb;
    for (int s = 0; ; ) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (U == 1 || kb + u < spr) {
                const int t = lane + (kb + u)*L;
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    acc[i] += D::compute(raw[u][i], ybase, t, K, tab);
                    if constexpr (has_gate) {
                        accg[i] += D::compute(rawg[u][i], ybase, t, K, tab);
                    }
                }
            }
        }
        if (kb + U >= spr) {
#pragma unroll
            for (int i = 0; i < R; ++i) {
                acc[i] = warp_reduce_sum<L>(acc[i]);
                if constexpr (has_gate) {
                    accg[i] = warp_reduce_sum<L>(accg[i]);
                }
            }
            const int g = g0 + j*gstride;
            if (lane < R && g < ngroups) {
                float r  = acc[0];
                float rg = accg[0];
#pragma unroll
                for (int i = 1; i < R; ++i) {
                    if (lane == i) {
                        r  = acc[i];
                        rg = accg[i];
                    }
                }
                const int row = g*R + lane;
                if (row < p.N) {
                    if constexpr (has_gate) {
                        r *= ggml_cuda_op_silu_single(rg);
                    }
                    float sv = 0.0f;
                    if constexpr (YG) {
                        if (p.epi != EPI_NONE) {
                            sv = p.other[row];
                        } else if (p.residual) {
                            sv = p.residual[row];
                        }
                    } else {
                        sv = side[j*RW + (g - wg_g0 - j*gstride)*R + lane];
                    }
                    if (p.epi == EPI_GLU_GATE) {
                        r = ggml_cuda_op_silu_single(sv) * r;
                    } else if (p.epi == EPI_GLU_UP) {
                        r = ggml_cuda_op_silu_single(r) * sv;
                    } else if (p.residual) {
                        r += sv;
                    }
                    p.dst[row] = r;
                }
            }
#pragma unroll
            for (int i = 0; i < R; ++i) {
                acc[i]  = 0.0f;
                accg[i] = 0.0f;
            }
        }
        if (++s >= nsteps) {
            break;
        }
        kb += U;
        if (kb >= spr) {
            kb = 0;
            ++j;
        }
        load_batch(j, kb);
    }
}

template <class D, int NW, int L, int R>
static constexpr __host__ __device__ int pp_smem_bytes(int K, int niter) {
    return ((lds_y_bytes<D::soa_y>(K) + 15) & ~15) + 4*D::tab_words + 4*niter*NW*(64/L)*R;
}

template <class D, int NW, int L, int R, bool has_gate>
__launch_bounds__(NW*64, 1)
static __global__ void mmvqf_pipe_kernel(const params p) {
    extern __shared__ __align__(16) char mmvqf_smem[];
    constexpr int NT = NW*64;
    constexpr int G  = 64/L;
    using raw_t = typename D::raw_t;

    const int K = p.K;
    char     * smem = mmvqf_smem;
    uint32_t * tab  = (uint32_t *)(smem + ((lds_y_bytes<D::soa_y>(K) + 15) & ~15));

    const int lane    = threadIdx.x % L;
    const int grp     = threadIdx.x / L;
    const int nch     = (K / D::qk) * D::cpb;
    const int ngroups = (p.N + R - 1) / R;
    const int gstride = gridDim.x*NW*G;

    int g = (blockIdx.x*NW + threadIdx.y)*G + grp;

    // first loads, before the prologue
    raw_t cur[R];
    raw_t curg[has_gate ? R : 1];
    {
        const int row0 = min(g, ngroups - 1)*R;
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const int row = min(row0 + i, p.N - 1);
            cur[i] = D::load(p.vx, p.mx, row, lane, p.stride_row);
            if constexpr (has_gate) {
                curg[i] = D::load(p.vg, p.mg, row, lane, p.stride_row);
            }
        }
    }

    const int tid = threadIdx.y*64 + threadIdx.x;
    D::load_table(tab, tid, NT);
    quantize_x_lds<NT, D::soa_y>(p.x, smem, K);
    __syncthreads();

    for (; g < ngroups; g += gstride) {
        const int row0 = g*R;
        float acc[R]  = {0.0f};
        float accg[R] = {0.0f};
        for (int t = lane; t < nch; t += L) {
            // prefetch: next chunk of this row group, or the first chunk of the next row group
            int tn = t + L;
            int gn = g;
            if (tn >= nch) {
                tn = lane;
                gn = min(g + gstride, ngroups - 1);
            }
            raw_t nxt[R];
            raw_t nxtg[has_gate ? R : 1];
#pragma unroll
            for (int i = 0; i < R; ++i) {
                const int row = min(gn*R + i, p.N - 1);
                nxt[i] = D::load(p.vx, p.mx, row, tn, p.stride_row);
                if constexpr (has_gate) {
                    nxtg[i] = D::load(p.vg, p.mg, row, tn, p.stride_row);
                }
            }
#pragma unroll
            for (int i = 0; i < R; ++i) {
                acc[i] += D::compute(cur[i], smem, t, K, tab);
                if constexpr (has_gate) {
                    accg[i] += D::compute(curg[i], smem, t, K, tab);
                }
            }
#pragma unroll
            for (int i = 0; i < R; ++i) {
                cur[i] = nxt[i];
                if constexpr (has_gate) {
                    curg[i] = nxtg[i];
                }
            }
        }
#pragma unroll
        for (int i = 0; i < R; ++i) {
            acc[i] = warp_reduce_sum<L>(acc[i]);
            if constexpr (has_gate) {
                accg[i] = warp_reduce_sum<L>(accg[i]);
            }
        }
        if (lane < R) {
            float r  = acc[0];
            float rg = accg[0];
#pragma unroll
            for (int i = 1; i < R; ++i) {
                if (lane == i) {
                    r  = acc[i];
                    rg = accg[i];
                }
            }
            const int row = row0 + lane;
            if (row < p.N) {
                if constexpr (has_gate) {
                    r *= ggml_cuda_op_silu_single(rg);
                }
                if (p.epi == EPI_GLU_GATE) {
                    r = ggml_cuda_op_silu_single(p.other[row]) * r;
                } else if (p.epi == EPI_GLU_UP) {
                    r = ggml_cuda_op_silu_single(r) * p.other[row];
                }
                if (p.residual) {
                    r += p.residual[row];
                }
                p.dst[row] = r;
            }
        }
    }
}

} // namespace mmvqf
