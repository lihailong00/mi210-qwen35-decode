// lowbit-gemv.cuh - low-bit GEMV building blocks for gfx90a (MI210): IQ2_S, IQ2_XS, IQ2_XXS, Q2_K, IQ1_M
// (from the gemv-lowbit workstream, lb4.cuh; integration adds an optional residual epilogue)
//
// Design (see READY.md):
//  * weights are re-laid out once (lossless, same size) into an SoA format ("lb SoA"): per-sub-block 8-byte stream A,
//    small side stream B, per-block scale stream C -> one wide fully coalesced load per item instead of 4-5 small ones.
//  * sign bits are stored transposed so the per-dword sign selector is 1-2 ALU ops (no sign table lookups);
//    IQ2_XXS/XS ksigns parity bits are reconstructed in-register (XOR fold), signs applied to y via v_perm(y, -y).
//  * codebooks in LDS; activation y staged in LDS as per-32-value records (odd dword stride: conflict free).
// Numerics: per-32-value integer sums and their integer scale math are identical to llama.cpp vec_dot_*_q8_1;
// only the fp32 accumulation order differs.
#pragma once
#include "common.cuh"
#include "vecdotq.cuh"

namespace lb {

// ---------------------------------------------------------------- helpers
static __device__ __forceinline__ uint32_t perm(uint32_t hi, uint32_t lo, uint32_t sel) { return __builtin_amdgcn_perm(hi, lo, sel); }
static __device__ __forceinline__ int dp4a(int a, int b, int c) { return __builtin_amdgcn_sdot4(a, b, c, false); }
static __device__ __forceinline__ int mul24(int a, int b) { return __mul24(a, b); }
static __device__ __forceinline__ int mad24(int a, int b, int c) { return __mul24(a, b) + c; }
// truncating signed division by 2^s (== C '/')
template <int s> static __device__ __forceinline__ int sdiv(int x) { return (x + (int) ((uint32_t) (x >> 31) >> (32 - s))) >> s; }
static __host__ __device__ __forceinline__ uint32_t spread4(uint32_t s4) { return (s4 * 0x00204081u) & 0x01010101u; } // 4 bits -> 0x01/byte
// bytewise negation (no cross-byte carry), valid for all int8 values
static __device__ __forceinline__ int neg8(int v) {
    const uint32_t nx = ~(uint32_t) v;
    return (int) (((nx & 0x7F7F7F7Fu) + 0x01010101u) ^ (nx & 0x80808080u));
}
// v_perm selector picking byte b of y (lo) or of -y (hi) for the 4 sign bits at S bits {pos, pos+8, pos+16, pos+24}
template <int pos> static __device__ __forceinline__ uint32_t ysel(const uint32_t S) {
    const uint32_t t = pos >= 2 ? (S >> (pos - 2)) : (S << (2 - pos));
    return (t & 0x04040404u) | 0x03020100u;
}
// ksigns: replace bits 28..31 of a transposed sign word by the parity bits of the 4 groups (8th sign = parity of 7)
static __device__ __forceinline__ uint32_t ksign_fix(const uint32_t S) {
    const uint32_t x = S & 0x0FFFFFFFu;
    uint32_t y = x ^ (x >> 4);
    y ^= y >> 16;
    y ^= y >> 8;
    return x | (y << 28);
}
// signed 4-weight dot: YS (y-side): dp4a(g, sel ? -y : y) with -y from the y record; else (g-side): dp4a(sel ? -g : g, y)
template <bool YS> static __device__ __forceinline__ int sdot(const uint32_t g, const int * __restrict__ yr, const int j, const uint32_t sel, const int acc) {
    if constexpr (YS) return dp4a((int) g, (int) perm((uint32_t) yr[8 + j], (uint32_t) yr[j], sel), acc);
    else { const uint32_t ng = (~g) + 0x01010101u; return dp4a((int) perm(ng, g, sel), yr[j], acc); }  // IQ2 grid bytes in [8,43]
}
static __device__ __forceinline__ float h2f(uint32_t h) { return __half2float(__ushort_as_half((unsigned short) h)); }

// transposed sign layout (host+device, used by repack): weight w = 8k+i of a 32-weight sub-block
//   dword j = 2k + (i>>2), byte b = i&3, bit position in S = 8*b + pos(j), pos(2k) = k, pos(2k+1) = 4+k
static __host__ __device__ __forceinline__ int tbit(int w) { const int k = w >> 3, i = w & 7; return 8*(i & 3) + (i < 4 ? k : 4 + k); }

// ---------------------------------------------------------------- SoA stream geometry
// nb = number of 256-weight blocks in the tensor (nrows * K/256), streams back to back: A | B | C
struct soa { const uint8_t * A; const uint8_t * B; const uint8_t * C; };
static __host__ __device__ inline size_t soa_bytes_A(ggml_type t) { return t == GGML_TYPE_IQ1_M ? 32 : 64; }
static __host__ __device__ inline size_t soa_bytes_B(ggml_type t) {
    switch (t) { case GGML_TYPE_IQ2_S: return 16; case GGML_TYPE_IQ2_XS: return 8; case GGML_TYPE_Q2_K: return 16; case GGML_TYPE_IQ1_M: return 16; default: return 0; }
}
static __host__ __device__ inline soa soa_make(ggml_type t, const void * base, size_t nb) {
    soa s; s.A = (const uint8_t *) base; s.B = s.A + nb*soa_bytes_A(t); s.C = s.B + nb*soa_bytes_B(t); return s;
}

// ---------------------------------------------------------------- repack kernels (orig -> SoA), one thread per block
// dst must not alias src. Total size equals ggml_nbytes of the tensor.
static __global__ void repack_iq2_s(const block_iq2_s * __restrict__ src, uint8_t * __restrict__ dst, const size_t nb) {
    const size_t g = blockIdx.x*(size_t) blockDim.x + threadIdx.x;
    if (g >= nb) return;
    const block_iq2_s & b = src[g];
    soa s = soa_make(GGML_TYPE_IQ2_S, dst, nb);
    uint8_t * A = (uint8_t *) s.A + g*64; uint8_t * B = (uint8_t *) s.B + g*16; uint8_t * C = (uint8_t *) s.C + g*2;
    for (int i = 0; i < 8; ++i) {
        uint32_t S = 0;
        for (int w = 0; w < 32; ++w) S |= (uint32_t) ((b.qs[32 + 4*i + (w >> 3)] >> (w & 7)) & 1) << tbit(w);
        for (int j = 0; j < 4; ++j) A[8*i + j] = b.qs[4*i + j];
        memcpy(A + 8*i + 4, &S, 4);
        B[2*i] = b.qh[i]; B[2*i + 1] = b.scales[i];
    }
    memcpy(C, &b.d, 2);
}
static __global__ void repack_iq2_xxs(const block_iq2_xxs * __restrict__ src, uint8_t * __restrict__ dst, const size_t nb) {
    const size_t g = blockIdx.x*(size_t) blockDim.x + threadIdx.x;
    if (g >= nb) return;
    const block_iq2_xxs & b = src[g];
    soa s = soa_make(GGML_TYPE_IQ2_XXS, dst, nb);
    uint8_t * A = (uint8_t *) s.A + g*64; uint8_t * C = (uint8_t *) s.C + g*2;
    for (int i = 0; i < 8; ++i) {
        const uint32_t q2  = (uint32_t) b.qs[4*i + 0] | (uint32_t) b.qs[4*i + 1] << 16;
        const uint32_t aux = (uint32_t) b.qs[4*i + 2] | (uint32_t) b.qs[4*i + 3] << 16;
        uint32_t S = (aux >> 28) << 28;                       // scale nibble at 28..31 (parity slots)
        for (int k = 0; k < 4; ++k) for (int ii = 0; ii < 7; ++ii) S |= ((aux >> (7*k + ii)) & 1u) << tbit(8*k + ii);
        memcpy(A + 8*i, &q2, 4); memcpy(A + 8*i + 4, &S, 4);
    }
    memcpy(C, &b.d, 2);
}
static __global__ void repack_iq2_xs(const block_iq2_xs * __restrict__ src, uint8_t * __restrict__ dst, const size_t nb) {
    const size_t g = blockIdx.x*(size_t) blockDim.x + threadIdx.x;
    if (g >= nb) return;
    const block_iq2_xs & b = src[g];
    soa s = soa_make(GGML_TYPE_IQ2_XS, dst, nb);
    uint8_t * A = (uint8_t *) s.A + g*64; uint8_t * B = (uint8_t *) s.B + g*8; uint8_t * C = (uint8_t *) s.C + g*2;
    for (int i = 0; i < 8; ++i) {
        uint32_t Q = 0, S = 0;
        for (int k = 0; k < 4; ++k) {
            const uint32_t w = b.qs[4*i + k];
            Q |= (w & 0xFF) << (8*k);
            S |= ((w >> 8) & 1u) << (28 + k);                 // idx bit 8 at 28..31 (parity slots)
            for (int ii = 0; ii < 7; ++ii) S |= ((w >> (9 + ii)) & 1u) << tbit(8*k + ii);
        }
        memcpy(A + 8*i, &Q, 4); memcpy(A + 8*i + 4, &S, 4);
        B[i] = b.scales[i];
    }
    memcpy(C, &b.d, 2);
}
static __global__ void repack_q2_K(const block_q2_K * __restrict__ src, uint8_t * __restrict__ dst, const size_t nb) {
    const size_t g = blockIdx.x*(size_t) blockDim.x + threadIdx.x;
    if (g >= nb) return;
    const block_q2_K & b = src[g];
    soa s = soa_make(GGML_TYPE_Q2_K, dst, nb);
    memcpy((uint8_t *) s.A + g*64, b.qs, 64);
    memcpy((uint8_t *) s.B + g*16, b.scales, 16);
    memcpy((uint8_t *) s.C + g*4, &b.dm, 4);
}
static __global__ void repack_iq1_m(const block_iq1_m * __restrict__ src, uint8_t * __restrict__ dst, const size_t nb) {
    const size_t g = blockIdx.x*(size_t) blockDim.x + threadIdx.x;
    if (g >= nb) return;
    const block_iq1_m & b = src[g];
    soa s = soa_make(GGML_TYPE_IQ1_M, dst, nb);
    memcpy((uint8_t *) s.A + g*32, b.qs, 32);
    memcpy((uint8_t *) s.B + g*16, b.qh, 16);
    memcpy((uint8_t *) s.C + g*8, b.scales, 8);
}

// ---------------------------------------------------------------- y records in LDS (one per 32 values)
//  IQ2_*: 17 dwords {qs[8], -qs[8], d}       Q2_K: 11 dwords {qs[8], d, sum(qs[0..15]), sum(qs[16..31])}
//  IQ1_M: 13 dwords {qs[8], d, sum of each 8 values [4]}
template <int MODE> struct yrec;  // MODE 0: IQ2, 1: Q2_K, 2: IQ1_M
template <> struct yrec<0> { static constexpr int W = 17, D = 16; };
template <> struct yrec<1> { static constexpr int W = 11, D = 8; };
template <> struct yrec<2> { static constexpr int W = 13, D = 8; };
template <> struct yrec<3> { static constexpr int W = 9,  D = 8; };   // IQ2 g-side signs: {qs[8], d}

// stage y records from global block_q8_1 (K % 256 == 0); one thread per y dword
template <int MODE>
static __device__ __forceinline__ void stage_y(int * __restrict__ rec, const block_q8_1 * __restrict__ yg, const int K, const int tid, const int nt) {
    constexpr int W = yrec<MODE>::W;
    for (int i = tid; i < K/4; i += nt) {
        const int b = i >> 3, w = i & 7;
        const int v = ((const int *) yg[b].qs)[w];
        rec[b*W + w] = v;
        if (MODE == 0) rec[b*W + 8 + w] = neg8(v);
    }
    for (int b = tid; b < K/32; b += nt) {
        ((float *) rec)[b*W + yrec<MODE>::D] = __low2float(yg[b].ds);
        if (MODE == 1 || MODE == 2) {
            const int * q = (const int *) yg[b].qs;
            int s[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) s[j] = dp4a(q[2*j+1], 0x01010101, dp4a(q[2*j], 0x01010101, 0));
            if (MODE == 1) { rec[b*W + 9] = s[0] + s[1]; rec[b*W + 10] = s[2] + s[3]; }
            else { rec[b*W + 9] = s[0]; rec[b*W + 10] = s[1]; rec[b*W + 11] = s[2]; rec[b*W + 12] = s[3]; }
        }
    }
}

// ---------------------------------------------------------------- IQ2_S
// LDS tables: grid [0,2048) words | qhx [2048,2304): byte k = 2-bit field k of qh
template <bool YS = true> struct iq2_s_t {
    static constexpr ggml_type type = GGML_TYPE_IQ2_S;
    static constexpr int TAB_WORDS = 2304, YMODE = YS ? 0 : 3, IPB = 4 /* items per block */, SPI = 2 /* q8 blocks per item */;
    struct raw { uint4 a; uint32_t b; uint32_t d; };
    static __device__ __forceinline__ void stage_tables(uint32_t * t, int tid, int nt) {
        const uint4 * g = (const uint4 *) iq2s_grid;
        for (int i = tid; i < 512; i += nt) ((uint4 *) t)[i] = g[i];
        for (int i = tid; i < 256; i += nt) t[2048 + i] = (i & 3) | ((i >> 2) & 3) << 8 | ((i >> 4) & 3) << 16 | ((i >> 6) & 3) << 24;
    }
    static __device__ __forceinline__ raw load(const soa & s, const uint32_t kbx, const uint32_t it) {
        raw r;
        const uint32_t sb = kbx*8 + 2*it;
        r.a = *(const uint4 *) (s.A + sb*8);
        r.b = *(const uint32_t *) (s.B + sb*2);
        r.d = *(const uint16_t *) (s.C + kbx*2);
        return r;
    }
    static __device__ __forceinline__ float compute(const uint32_t * __restrict__ tab, const raw & r, const int * __restrict__ yr0) {
        const float dw = h2f(r.d);
        float sum = 0.0f;
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const uint32_t qs = p ? r.a.z : r.a.x;
            const uint32_t S  = p ? r.a.w : r.a.y;
            const uint32_t qhsc = r.b >> (16*p);
            const uint32_t qhx = tab[2048 + (qhsc & 0xFF)];
            const uint32_t X = perm(qhx, qs, 0x05010400u), Z = perm(qhx, qs, 0x07030602u);
            const uint2 g0 = ((const uint2 *) tab)[X & 0xFFFF], g1 = ((const uint2 *) tab)[X >> 16];
            const uint2 g2 = ((const uint2 *) tab)[Z & 0xFFFF], g3 = ((const uint2 *) tab)[Z >> 16];
            const int * yr = yr0 + p*yrec<YMODE>::W;
            int s0 = 0, s1 = 0;
            s0 = sdot<YS>(g0.x, yr, 0, ysel<0>(S), s0);
            s0 = sdot<YS>(g0.y, yr, 1, ysel<4>(S), s0);
            s0 = sdot<YS>(g1.x, yr, 2, ysel<1>(S), s0);
            s0 = sdot<YS>(g1.y, yr, 3, ysel<5>(S), s0);
            s1 = sdot<YS>(g2.x, yr, 4, ysel<2>(S), s1);
            s1 = sdot<YS>(g2.y, yr, 5, ysel<6>(S), s1);
            s1 = sdot<YS>(g3.x, yr, 6, ysel<3>(S), s1);
            s1 = sdot<YS>(g3.y, yr, 7, ysel<7>(S), s1);
            const int sc = (qhsc >> 8) & 0xFF;
            const int sumi = sdiv<2>(mad24(s0, sc & 0xF, mad24(s1, sc >> 4, sdiv<1>(s0 + s1))));
            sum += (dw * __int_as_float(yr[yrec<YMODE>::D])) * (float) sumi;
        }
        return sum;
    }
};

// ---------------------------------------------------------------- IQ2_XXS
// LDS tables: grid [0,512) words
template <bool YS = true> struct iq2_xxs_t {
    static constexpr ggml_type type = GGML_TYPE_IQ2_XXS;
    static constexpr int TAB_WORDS = 512, YMODE = YS ? 0 : 3, IPB = 4, SPI = 2;
    struct raw { uint4 a; uint32_t d; };
    static __device__ __forceinline__ void stage_tables(uint32_t * t, int tid, int nt) {
        const uint4 * g = (const uint4 *) iq2xxs_grid;
        for (int i = tid; i < 128; i += nt) ((uint4 *) t)[i] = g[i];
    }
    static __device__ __forceinline__ raw load(const soa & s, const uint32_t kbx, const uint32_t it) {
        raw r;
        r.a = *(const uint4 *) (s.A + (kbx*8 + 2*it)*8);
        r.d = *(const uint16_t *) (s.C + kbx*2);
        return r;
    }
    static __device__ __forceinline__ float compute(const uint32_t * __restrict__ tab, const raw & r, const int * __restrict__ yr0) {
        const float dw = h2f(r.d);
        float sum = 0.0f;
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const uint32_t q2 = p ? r.a.z : r.a.x;
            const uint32_t S0 = p ? r.a.w : r.a.y;
            const uint32_t S = ksign_fix(S0);
            const uint2 g0 = ((const uint2 *) tab)[q2 & 0xFF],         g1 = ((const uint2 *) tab)[(q2 >> 8) & 0xFF];
            const uint2 g2 = ((const uint2 *) tab)[(q2 >> 16) & 0xFF], g3 = ((const uint2 *) tab)[q2 >> 24];
            const int * yr = yr0 + p*yrec<YMODE>::W;
            int s = 0;
            s = sdot<YS>(g0.x, yr, 0, ysel<0>(S), s);
            s = sdot<YS>(g0.y, yr, 1, ysel<4>(S), s);
            s = sdot<YS>(g1.x, yr, 2, ysel<1>(S), s);
            s = sdot<YS>(g1.y, yr, 3, ysel<5>(S), s);
            s = sdot<YS>(g2.x, yr, 4, ysel<2>(S), s);
            s = sdot<YS>(g2.y, yr, 5, ysel<6>(S), s);
            s = sdot<YS>(g3.x, yr, 6, ysel<3>(S), s);
            s = sdot<YS>(g3.y, yr, 7, ysel<7>(S), s);
            const int ls = (int) ((S0 >> 27) | 1u);
            const int sumi = sdiv<3>(mul24(s, ls));
            sum += (dw * __int_as_float(yr[yrec<YMODE>::D])) * (float) sumi;
        }
        return sum;
    }
};

// ---------------------------------------------------------------- IQ2_XS
// LDS tables: grid [0,1024) words
template <bool YS = true> struct iq2_xs_t {
    static constexpr ggml_type type = GGML_TYPE_IQ2_XS;
    static constexpr int TAB_WORDS = 1024, YMODE = YS ? 0 : 3, IPB = 4, SPI = 2;
    struct raw { uint4 a; uint32_t b; uint32_t d; };
    static __device__ __forceinline__ void stage_tables(uint32_t * t, int tid, int nt) {
        const uint4 * g = (const uint4 *) iq2xs_grid;
        for (int i = tid; i < 256; i += nt) ((uint4 *) t)[i] = g[i];
    }
    static __device__ __forceinline__ raw load(const soa & s, const uint32_t kbx, const uint32_t it) {
        raw r;
        const uint32_t sb = kbx*8 + 2*it;
        r.a = *(const uint4 *) (s.A + sb*8);
        r.b = *(const uint16_t *) (s.B + sb);
        r.d = *(const uint16_t *) (s.C + kbx*2);
        return r;
    }
    static __device__ __forceinline__ float compute(const uint32_t * __restrict__ tab, const raw & r, const int * __restrict__ yr0) {
        const float dw = h2f(r.d);
        float sum = 0.0f;
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const uint32_t Q  = p ? r.a.z : r.a.x;
            const uint32_t S0 = p ? r.a.w : r.a.y;
            const uint32_t qh4 = spread4(S0 >> 28);            // byte k = bit 8 of idx k
            const uint32_t X = perm(qh4, Q, 0x05010400u), Z = perm(qh4, Q, 0x07030602u);
            const uint32_t S = ksign_fix(S0);
            const uint2 g0 = ((const uint2 *) tab)[X & 0xFFFF], g1 = ((const uint2 *) tab)[X >> 16];
            const uint2 g2 = ((const uint2 *) tab)[Z & 0xFFFF], g3 = ((const uint2 *) tab)[Z >> 16];
            const int * yr = yr0 + p*yrec<YMODE>::W;
            int s0 = 0, s1 = 0;
            s0 = sdot<YS>(g0.x, yr, 0, ysel<0>(S), s0);
            s0 = sdot<YS>(g0.y, yr, 1, ysel<4>(S), s0);
            s0 = sdot<YS>(g1.x, yr, 2, ysel<1>(S), s0);
            s0 = sdot<YS>(g1.y, yr, 3, ysel<5>(S), s0);
            s1 = sdot<YS>(g2.x, yr, 4, ysel<2>(S), s1);
            s1 = sdot<YS>(g2.y, yr, 5, ysel<6>(S), s1);
            s1 = sdot<YS>(g3.x, yr, 6, ysel<3>(S), s1);
            s1 = sdot<YS>(g3.y, yr, 7, ysel<7>(S), s1);
            const int sc = (r.b >> (8*p)) & 0xFF;
            const int sumi = sdiv<2>(mad24(s0, sc & 0xF, mad24(s1, sc >> 4, sdiv<1>(s0 + s1))));
            sum += (dw * __int_as_float(yr[yrec<YMODE>::D])) * (float) sumi;
        }
        return sum;
    }
};

typedef iq2_s_t<true>   iq2_s;
typedef iq2_xs_t<true>  iq2_xs;
typedef iq2_xxs_t<true> iq2_xxs;

// ---------------------------------------------------------------- Q2_K (item = half block = 4 q8 blocks)
struct q2_K {
    static constexpr ggml_type type = GGML_TYPE_Q2_K;
    static constexpr int TAB_WORDS = 0, YMODE = 1, IPB = 2, SPI = 4;
    struct raw { uint4 q0, q1; uint2 sc; uint32_t dm; };
    static __device__ __forceinline__ void stage_tables(uint32_t *, int, int) { }
    static __device__ __forceinline__ raw load(const soa & s, const uint32_t kbx, const uint32_t ih) {
        raw r;
        const uint32_t hb = kbx*2 + ih;
        r.q0 = *(const uint4 *) (s.A + hb*32);
        r.q1 = *(const uint4 *) (s.A + hb*32 + 16);
        r.sc = *(const uint2 *) (s.B + hb*8);
        r.dm = *(const uint32_t *) (s.C + kbx*4);
        return r;
    }
    static __device__ __forceinline__ float compute(const uint32_t * __restrict__, const raw & r, const int * __restrict__ yr0) {
        const uint32_t v[8] = {r.q0.x, r.q0.y, r.q0.z, r.q0.w, r.q1.x, r.q1.y, r.q1.z, r.q1.w};
        float sd = 0.0f, sm = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int * yr = yr0 + i*11;
            int s0 = 0, s1 = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                // shift i: bits 2i..2i+1 of every byte; i<3 masked in place (value scaled by 4^i), i==3 via >>4 (scaled by 4)
                const int x = i < 3 ? (int) (v[j] & (0x03030303u << (2*i))) : (int) ((v[j] >> 4) & 0x0C0C0C0Cu);
                if (j < 4) s0 = dp4a(x, yr[j], s0); else s1 = dp4a(x, yr[j], s1);
            }
            const uint32_t scb = (i < 2 ? r.sc.x : r.sc.y) >> (16*(i & 1));
            const int SD = mad24(s0, scb & 0xF, mul24(s1, (scb >> 8) & 0xF));
            const int SM = mad24(yr[9], (scb >> 4) & 0xF, mul24(yr[10], (scb >> 12) & 0xF));
            const float d8 = __int_as_float(yr[8]);
            const float inv = i == 0 ? 1.0f : i == 1 ? 0.25f : i == 2 ? 0.0625f : 0.25f;
            sd += (d8*inv) * (float) SD;
            sm += d8 * (float) SM;
        }
        const float2 dm = __half22float2(*(const half2 *) &r.dm);
        return dm.x*sd - dm.y*sm;
    }
};

// ---------------------------------------------------------------- IQ1_M
// LDS tables: pre-split grid [0,4096) words: entry i = {g & 0x0F0F0F0F, (g >> 4) & 0x0F0F0F0F}, g = iq1s_grid_gpu[i]
struct iq1_m {
    static constexpr ggml_type type = GGML_TYPE_IQ1_M;
    static constexpr int TAB_WORDS = 4096, YMODE = 2, IPB = 4, SPI = 2;
    struct raw { uint2 a; uint32_t b; uint2 sc; };
    static __device__ __forceinline__ void stage_tables(uint32_t * t, int tid, int nt) {
        for (int i = tid; i < 2048; i += nt) { const uint32_t g = iq1s_grid_gpu[i]; ((uint2 *) t)[i] = make_uint2(g & 0x0F0F0F0Fu, (g >> 4) & 0x0F0F0F0Fu); }
    }
    static __device__ __forceinline__ raw load(const soa & s, const uint32_t kbx, const uint32_t it) {
        raw r;
        const uint32_t sb = kbx*8 + 2*it;
        r.a  = *(const uint2 *) (s.A + sb*4);
        r.b  = *(const uint32_t *) (s.B + sb*2);
        r.sc = *(const uint2 *) (s.C + kbx*8);
        return r;
    }
    static __device__ __forceinline__ float compute(const uint32_t * __restrict__ tab, const raw & r, const int * __restrict__ yr0, const int ib0) {
        // block scale (fp16 assembled from the top nibbles of the 4 scale words)
        const uint32_t u16 = ((r.sc.x & 0xFFFF) >> 12) | ((r.sc.x >> 24) & 0x00F0) | (((r.sc.y & 0xFFFF) >> 4) & 0x0F00) | ((r.sc.y >> 16) & 0xF000);
        const float dblk = h2f(u16);
        float sum = 0.0f;
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int ib = ib0 + p;
            const uint32_t Q = p ? r.a.y : r.a.x;
            const uint32_t qh = (r.b >> (16*p)) & 0xFFFF;
            const uint32_t qhb = (qh & 0x0707u) | ((qh >> 4) & 0x0707u) << 16;   // bytes: n0&7, n2&7, n1&7, n3&7
            const uint32_t X = perm(qhb, Q, 0x06010400u), Z = perm(qhb, Q, 0x07030502u);
            const uint2 g0 = ((const uint2 *) tab)[X & 0xFFFF], g1 = ((const uint2 *) tab)[X >> 16];
            const uint2 g2 = ((const uint2 *) tab)[Z & 0xFFFF], g3 = ((const uint2 *) tab)[Z >> 16];
            const int * yr = yr0 + p*13;
            const int i0 = dp4a((int) g0.y, yr[1], dp4a((int) g0.x, yr[0], 0));
            const int i1 = dp4a((int) g1.y, yr[3], dp4a((int) g1.x, yr[2], 0));
            const int i2 = dp4a((int) g2.y, yr[5], dp4a((int) g2.x, yr[4], 0));
            const int i3 = dp4a((int) g3.y, yr[7], dp4a((int) g3.x, yr[6], 0));
            // delta multiplier (x8): bit 3 of nibble k -> -9, else -7
            const int m0 = -7 - 2*(int) ((qh >>  3) & 1), m1 = -7 - 2*(int) ((qh >>  7) & 1);
            const int m2 = -7 - 2*(int) ((qh >> 11) & 1), m3 = -7 - 2*(int) ((qh >> 15) & 1);
            const int S0 = mad24(i0 + i1, 8, mad24(yr[9], m0, mul24(yr[10], m1)));
            const int S1 = mad24(i2 + i3, 8, mad24(yr[11], m2, mul24(yr[12], m3)));
            const uint32_t scw = (ib < 4 ? r.sc.x : r.sc.y) >> (16*((ib >> 1) & 1));
            const uint32_t tmp = (scw & 0xFFFF) >> (6*(ib & 1));
            const int sc0 = 2*(int) (tmp & 7) + 1, sc1 = 2*(int) ((tmp >> 3) & 7) + 1;
            sum += (dblk * __int_as_float(yr[8])) * ((float) mad24(S0, sc0, mul24(S1, sc1)) * 0.125f);
        }
        return sum;
    }
};

// IQ1_M needs the sub-block index within the block for its scales; uniform compute wrapper:
template <class T> static __device__ __forceinline__ float compute_item(const uint32_t * tab, const typename T::raw & r, const int * yr0, int it) {
    if constexpr (T::type == GGML_TYPE_IQ1_M) return T::compute(tab, r, yr0, 2*it); else { (void) it; return T::compute(tab, r, yr0); }
}

// ---------------------------------------------------------------- reference kernel
// Persistent workgroups (grid ~ 2 x #CU); y staged once per workgroup; L lanes per row, R rows per lane group,
// NW waves per workgroup, work units of (64/L)*R rows distributed round-robin over all waves.
// K % 256 == 0, N % ((64/L)*R) == 0.
template <class T, int L, int R, int NW, bool PF = false>
__launch_bounds__(NW*64, 1)
__global__ void gemv(const soa s, const block_q8_1 * __restrict__ yg, float * __restrict__ dst, const int K, const int N,
        const float * __restrict__ residual = nullptr) {
    extern __shared__ uint32_t smem[];
    uint32_t * tab = smem;
    int * yrec_base = (int *) (smem + T::TAB_WORDS);
    constexpr int G = 64/L;
    constexpr int YW = yrec<T::YMODE>::W;
    const int tid = threadIdx.y*64 + threadIdx.x;
    const int lane = threadIdx.x % L, grp = threadIdx.x / L;
    const int bpr = K / QK_K;
    const int items = bpr*T::IPB;
    const int U = N / (G*R);
    const int ustep = gridDim.x*NW;
    // CU-balanced round robin: consecutive units go to different workgroups first
    int u = threadIdx.y*gridDim.x + blockIdx.x;
    typename T::raw cur[R];
    if (PF && u < U) {   // first loads in flight while tables / y are staged
        const int row0 = (u*G + grp)*R;
#pragma unroll
        for (int r = 0; r < R; ++r) cur[r] = T::load(s, (uint32_t) (row0 + r)*bpr + lane / T::IPB, lane % T::IPB);
    }
    T::stage_tables(tab, tid, NW*64);
    stage_y<T::YMODE>(yrec_base, yg, K, tid, NW*64);
    __syncthreads();
    for (; u < U; u += ustep) {
        const int row0 = (u*G + grp)*R;
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] = 0.0f;
        for (int it = lane; it < items; it += L) {
            const uint32_t kb = it / T::IPB, ii = it % T::IPB;
            const int * yr0 = yrec_base + (kb*8 + ii*T::SPI)*YW;
            if constexpr (PF) {
                typename T::raw nxt[R];
                int itn = it + L, rown = row0;
                if (itn >= items) { itn = lane; rown = min(row0 + ustep*G*R, N - R); }
#pragma unroll
                for (int r = 0; r < R; ++r) nxt[r] = T::load(s, (uint32_t) (rown + r)*bpr + itn / T::IPB, itn % T::IPB);
#pragma unroll
                for (int r = 0; r < R; ++r) acc[r] += compute_item<T>(tab, cur[r], yr0, ii);
#pragma unroll
                for (int r = 0; r < R; ++r) cur[r] = nxt[r];
            } else {
#pragma unroll
                for (int r = 0; r < R; ++r) cur[r] = T::load(s, (uint32_t) (row0 + r)*bpr + kb, ii);
#pragma unroll
                for (int r = 0; r < R; ++r) acc[r] += compute_item<T>(tab, cur[r], yr0, ii);
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float v = warp_reduce_sum<L>(acc[r]);
            if (lane == 0) dst[row0 + r] = residual ? v + residual[row0 + r] : v;
        }
    }
}

template <class T> static inline size_t gemv_lds_bytes(int K) { return (size_t) T::TAB_WORDS*4 + (size_t) (K/32)*yrec<T::YMODE>::W*4; }

} // namespace lb
