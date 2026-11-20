// lowbit-gemv.cuh: lowbit.cuh v3 from the gemv-lowbit workstream (/lab/gemv/lowbit, branch gemv-lowbit), unmodified.
// lowbit.cuh - low-bit GEMV building blocks for gfx90a (MI210): IQ2_S, IQ2_XS, IQ2_XXS, Q2_K, IQ1_M
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

#if defined(__HIP_DEVICE_COMPILE__)
typedef __attribute__((address_space(3))) const int yint;   // y record words in LDS
#else
typedef const int yint;
#endif
typedef yint * yptr;

// ---------------------------------------------------------------- helpers
static __device__ __forceinline__ uint32_t perm(uint32_t hi, uint32_t lo, uint32_t sel) { return __builtin_amdgcn_perm(hi, lo, sel); }
static __device__ __forceinline__ int dp4a(int a, int b, int c) { return __builtin_amdgcn_sdot4(a, b, c, false); }
#ifndef LB_DOT0_ASM
#define LB_DOT0_ASM 0
#endif
// first dot of a chain: VOP3P form with inline-constant 0 accumulator (avoids a v_mov 0 + v_dot4c)
static __device__ __forceinline__ int dp4a0(int a, int b) {
#if LB_DOT0_ASM && defined(__HIP_DEVICE_COMPILE__)
    int r; asm("v_dot4_i32_i8 %0, %1, %2, 0" : "=v"(r) : "v"(a), "v"(b)); return r;
#else
    return __builtin_amdgcn_sdot4(a, b, 0, false);
#endif
}
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
#ifndef LB_SEL_ANDOR
#define LB_SEL_ANDOR 1
#endif
static __device__ __forceinline__ uint32_t sconst(uint32_t v) {   // opaque uniform constant, lets v_and_or_b32 form
#if LB_SEL_ANDOR
    uint32_t r; asm("s_mov_b32 %0, %1" : "=s"(r) : "i"(v)); return r;
#else
    return v;
#endif
}
static __device__ __forceinline__ uint32_t vconst(uint32_t v) {   // constant in a VGPR (gfx9 VOP3 may read only one SGPR)
#if LB_SEL_ANDOR
    uint32_t r; asm("v_mov_b32 %0, %1" : "=v"(r) : "i"(v)); return r;
#else
    return v;
#endif
}
template <int pos> static __device__ __forceinline__ uint32_t ysel(const uint32_t S, const uint32_t m4, const uint32_t base) {
    uint32_t t;
    if constexpr (pos >= 2) t = S >> (pos - 2); else t = S << (2 - pos);
    return (t & m4) | base;
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
template <bool YS> static __device__ __forceinline__ int sdot(const uint32_t g, yptr yr, const int j, const uint32_t sel, const int acc) {
    if constexpr (YS) return dp4a((int) g, (int) perm((uint32_t) yr[8 + j], (uint32_t) yr[j], sel), acc);
    else { const uint32_t ng = (~g) + 0x01010101u; return dp4a((int) perm(ng, g, sel), yr[j], acc); }  // IQ2 grid bytes in [8,43]
}
template <bool YS> static __device__ __forceinline__ int sdot0(const uint32_t g, yptr yr, const int j, const uint32_t sel) {
    if constexpr (YS) return dp4a0((int) g, (int) perm((uint32_t) yr[8 + j], (uint32_t) yr[j], sel));
    else { const uint32_t ng = (~g) + 0x01010101u; return dp4a0((int) perm(ng, g, sel), yr[j]); }
}
// LDS table access. LB_TAB0: tables live at LDS address 0 (first in the dynamic segment, no static __shared__ in the
// kernel) -> addresses are plain idx*size, foldable into SDWA shifts. Otherwise relative to the passed table pointer.
#ifndef LB_TAB0
#define LB_TAB0 1
#endif
typedef __attribute__((address_space(3))) const uint2    lds_u2;
typedef __attribute__((address_space(3))) const uint32_t lds_u32;
static __device__ __forceinline__ uint2 tab_u2(const uint32_t * tab, uint32_t word_off, uint32_t idx) {
#if LB_TAB0 && defined(__HIP_DEVICE_COMPILE__)
    (void) tab;
    typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));
    const u32x2 v = *(__attribute__((address_space(3))) const u32x2 *) (size_t) (word_off*4 + idx*8);
    return make_uint2(v.x, v.y);
#else
    return ((const uint2 *) (tab + word_off))[idx];
#endif
}
static __device__ __forceinline__ uint32_t tab_u32(const uint32_t * tab, uint32_t word_off, uint32_t idx) {
#if LB_TAB0 && defined(__HIP_DEVICE_COMPILE__)
    (void) tab; return *(lds_u32 *) (size_t) (word_off*4 + idx*4);
#else
    return tab[word_off + idx];
#endif
}
// grid lookups from a packed pair of 16-bit indices X = idx0 | idx1 << 16 (byte address idx*8 via SDWA)
static __device__ __forceinline__ void tab_pair(const uint32_t * tab, const uint32_t X, uint2 & g0, uint2 & g1) {
#if LB_TAB0 && defined(__HIP_DEVICE_COMPILE__)
    (void) tab;
    uint32_t a0, a1;
    const uint32_t three = 3;
    asm("v_lshlrev_b32_sdwa %0, %1, %2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0" : "=v"(a0) : "v"(three), "v"(X));
    asm("v_lshlrev_b32_sdwa %0, %1, %2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1" : "=v"(a1) : "v"(three), "v"(X));
    typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));
    const u32x2 v0 = *(__attribute__((address_space(3))) const u32x2 *) (size_t) a0;
    const u32x2 v1 = *(__attribute__((address_space(3))) const u32x2 *) (size_t) a1;
    g0 = make_uint2(v0.x, v0.y); g1 = make_uint2(v1.x, v1.y);
#else
    g0 = tab_u2(tab, 0, X & 0xFFFF); g1 = tab_u2(tab, 0, X >> 16);
#endif
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
template <> struct yrec<2> { static constexpr int W = 15, D = 8; };   // IQ1_M: {qs[8], d, A01, A23, B0..B3}
template <> struct yrec<3> { static constexpr int W = 9,  D = 8; };   // IQ2 g-side signs: {qs[8], d}

// stage y records from global block_q8_1 (K % 256 == 0); one thread per y dword
#ifndef LB_STAGE_REC
#define LB_STAGE_REC 1
#endif
template <int MODE>
static __device__ __forceinline__ void stage_y(int * __restrict__ rec, const block_q8_1 * __restrict__ yg, const int K, const int tid, const int nt) {
    constexpr int W = yrec<MODE>::W;
    if (LB_STAGE_REC && K/32 <= nt) {   // one thread per 32-value record: all global loads in one round trip
        if (tid < K/32) {
            const uint8_t * b = (const uint8_t *) (yg + tid);
            const uint4 q0 = *(const uint4 *) (b + 4), q1 = *(const uint4 *) (b + 20);   // qs (4-byte aligned)
            const uint32_t ds = *(const uint32_t *) b;
            const int q[8] = {(int) q0.x, (int) q0.y, (int) q0.z, (int) q0.w, (int) q1.x, (int) q1.y, (int) q1.z, (int) q1.w};
            int * r = rec + tid*W;
#pragma unroll
            for (int w = 0; w < 8; ++w) r[w] = q[w];
            if constexpr (MODE == 0) {
#pragma unroll
                for (int w = 0; w < 8; ++w) r[8 + w] = neg8(q[w]);
            }
            ((float *) r)[yrec<MODE>::D] = __half2float(__ushort_as_half((unsigned short) (ds & 0xFFFF)));
            if constexpr (MODE == 1 || MODE == 2) {
                int s4[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) s4[j] = dp4a(q[2*j+1], 0x01010101, dp4a(q[2*j], 0x01010101, 0));
                if constexpr (MODE == 1) { r[9] = s4[0] + s4[1]; r[10] = s4[2] + s4[3]; }
                else { r[9] = -7*(s4[0] + s4[1]); r[10] = -7*(s4[2] + s4[3]); r[11] = -2*s4[0]; r[12] = -2*s4[1]; r[13] = -2*s4[2]; r[14] = -2*s4[3]; }
            }
        }
        return;
    }
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
            else {   // IQ1_M delta terms (x8): -7*sum (both delta signs) and the extra -2*sum when the delta bit is set
                rec[b*W + 9] = -7*(s[0] + s[1]); rec[b*W + 10] = -7*(s[2] + s[3]);
                rec[b*W + 11] = -2*s[0]; rec[b*W + 12] = -2*s[1]; rec[b*W + 13] = -2*s[2]; rec[b*W + 14] = -2*s[3];
            }
        }
    }
}

// ---------------------------------------------------------------- IQ2 common item load (P = 1 or 2 sub-blocks per item)
template <int P> struct a_words { uint32_t w[2*P]; };
template <int P> static __device__ __forceinline__ a_words<P> load_a8(const uint8_t * p) {
    a_words<P> r;
    if constexpr (P == 2) { const uint4 v = *(const uint4 *) p; r.w[0] = v.x; r.w[1] = v.y; r.w[2] = v.z; r.w[3] = v.w; }
    else { const uint2 v = *(const uint2 *) p; r.w[0] = v.x; r.w[1] = v.y; }
    return r;
}

// ---------------------------------------------------------------- IQ2 grid entries (GN: table also holds -g)
// Sign modes: YS = y-side (signs applied to y via the -y copy in the y record); !YS = g-side (signs applied to the
// grid bytes); GN (only with !YS) = the LDS grid table stores {g, -g} (16 B entries), saving the -g computation.
struct gent { uint32_t x, y, nx, ny; };
static __device__ __forceinline__ gent mk_gent(const uint2 g) { return gent{g.x, g.y, 0x01010100u - g.x, 0x01010100u - g.y}; }
static __device__ __forceinline__ uint4 tab_u4(const uint32_t * tab, uint32_t idx) {
#if LB_TAB0 && defined(__HIP_DEVICE_COMPILE__)
    (void) tab;
    typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
    const u32x4 v = *(__attribute__((address_space(3))) const u32x4 *) (size_t) (idx*16);
    return make_uint4(v.x, v.y, v.z, v.w);
#else
    return ((const uint4 *) tab)[idx];
#endif
}
static __device__ __forceinline__ void tab_pair4(const uint32_t * tab, const uint32_t X, uint4 & g0, uint4 & g1) {
#if LB_TAB0 && defined(__HIP_DEVICE_COMPILE__)
    (void) tab;
    uint32_t a0, a1;
    const uint32_t four = 4;
    asm("v_lshlrev_b32_sdwa %0, %1, %2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0" : "=v"(a0) : "v"(four), "v"(X));
    asm("v_lshlrev_b32_sdwa %0, %1, %2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1" : "=v"(a1) : "v"(four), "v"(X));
    typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
    const u32x4 v0 = *(__attribute__((address_space(3))) const u32x4 *) (size_t) a0;
    const u32x4 v1 = *(__attribute__((address_space(3))) const u32x4 *) (size_t) a1;
    g0 = make_uint4(v0.x, v0.y, v0.z, v0.w); g1 = make_uint4(v1.x, v1.y, v1.z, v1.w);
#else
    g0 = tab_u4(tab, X & 0xFFFF); g1 = tab_u4(tab, X >> 16);
#endif
}
template <bool GN> static __device__ __forceinline__ void tab_pairG(const uint32_t * tab, const uint32_t X, gent & a, gent & b) {
    if constexpr (GN) { uint4 u0, u1; tab_pair4(tab, X, u0, u1); a = gent{u0.x, u0.y, u0.z, u0.w}; b = gent{u1.x, u1.y, u1.z, u1.w}; }
    else { uint2 g0, g1; tab_pair(tab, X, g0, g1); a = mk_gent(g0); b = mk_gent(g1); }
}
template <bool GN> static __device__ __forceinline__ gent tab_g1(const uint32_t * tab, const uint32_t idx) {
    if constexpr (GN) { const uint4 u = tab_u4(tab, idx); return gent{u.x, u.y, u.z, u.w}; }
    else return mk_gent(tab_u2(tab, 0, idx));
}
template <bool GN> static __device__ __forceinline__ void stage_grid(uint32_t * t, const uint4 * g, const int n16, const int tid, const int nt) {
    for (int i = tid; i < n16; i += nt) {
        const uint4 v = g[i];   // two 8-byte grid entries
        if constexpr (GN) {
            ((uint4 *) t)[2*i]     = make_uint4(v.x, v.y, 0x01010100u - v.x, 0x01010100u - v.y);
            ((uint4 *) t)[2*i + 1] = make_uint4(v.z, v.w, 0x01010100u - v.z, 0x01010100u - v.w);
        } else {
            ((uint4 *) t)[i] = v;
        }
    }
}
// signed dot of one dword: YS: dp4a(g, sel ? -y : y); else dp4a(sel ? -g : g, y)
template <bool YS> static __device__ __forceinline__ int sdg(const uint32_t g, const uint32_t ng, yptr yr, const int j, const uint32_t sel, const int acc) {
    if constexpr (YS) return dp4a((int) g, (int) perm((uint32_t) yr[8 + j], (uint32_t) yr[j], sel), acc);
    else return dp4a((int) perm(ng, g, sel), yr[j], acc);
}
template <bool YS> static __device__ __forceinline__ int sdg0(const uint32_t g, const uint32_t ng, yptr yr, const int j, const uint32_t sel) {
    if constexpr (YS) return dp4a0((int) g, (int) perm((uint32_t) yr[8 + j], (uint32_t) yr[j], sel));
    else return dp4a0((int) perm(ng, g, sel), yr[j]);
}
// 32-weight signed dot: groups a..d (8 weights each), sign word S (parity already fixed for ksigns)
template <bool YS> static __device__ __forceinline__ void dot32(const gent & a, const gent & b, const gent & c, const gent & d,
        yptr yr, const uint32_t S, const uint32_t M4, const uint32_t SB, int & s0, int & s1) {
    s0 = sdg0<YS>(a.x, a.nx, yr, 0, ysel<0>(S, M4, SB));
    s0 = sdg<YS>(a.y, a.ny, yr, 1, ysel<4>(S, M4, SB), s0);
    s0 = sdg<YS>(b.x, b.nx, yr, 2, ysel<1>(S, M4, SB), s0);
    s0 = sdg<YS>(b.y, b.ny, yr, 3, ysel<5>(S, M4, SB), s0);
    s1 = sdg0<YS>(c.x, c.nx, yr, 4, ysel<2>(S, M4, SB));
    s1 = sdg<YS>(c.y, c.ny, yr, 5, ysel<6>(S, M4, SB), s1);
    s1 = sdg<YS>(d.x, d.nx, yr, 6, ysel<3>(S, M4, SB), s1);
    s1 = sdg<YS>(d.y, d.ny, yr, 7, ysel<7>(S, M4, SB), s1);
}

// ---------------------------------------------------------------- IQ2_S
// LDS tables: grid [0,G) words (G = 2048, or 4096 with GN) | qhx [G,G+256): byte k = 2-bit field k of qh
template <bool YS = true, int P = 2, bool GN = false> struct iq2_s_t {
    static_assert(!(YS && GN), "GN is a g-side mode");
    static constexpr ggml_type type = GGML_TYPE_IQ2_S;
    static constexpr int QHX = GN ? 4096 : 2048;
    static constexpr int TAB_WORDS = QHX + 256, YMODE = YS ? 0 : 3, IPB = 8/P /* items per block */, SPI = P /* q8 blocks per item */;
    struct raw { a_words<P> a; uint32_t b; uint32_t d; };
    static __device__ __forceinline__ void stage_tables(uint32_t * t, int tid, int nt) {
        stage_grid<GN>(t, (const uint4 *) iq2s_grid, 512, tid, nt);
        for (int i = tid; i < 256; i += nt) t[QHX + i] = (i & 3) | ((i >> 2) & 3) << 8 | ((i >> 4) & 3) << 16 | ((i >> 6) & 3) << 24;
    }
    static __device__ __forceinline__ raw load(const soa & s, const uint32_t kbx, const uint32_t it) {
        raw r;
        const uint32_t sb = kbx*8 + P*it;
        r.a = load_a8<P>(s.A + sb*8);
        r.b = P == 2 ? *(const uint32_t *) (s.B + sb*2) : (uint32_t) *(const uint16_t *) (s.B + sb*2);
        r.d = *(const uint16_t *) (s.C + kbx*2);
        return r;
    }
    static __device__ __forceinline__ float compute(const uint32_t * __restrict__ tab, const raw & r, yptr yr0) {
        const float dw = h2f(r.d);
        const uint32_t M4 = vconst(0x04040404u), SB = sconst(0x03020100u);
        float sum = 0.0f;
#pragma unroll
        for (int p = 0; p < P; ++p) {
            const uint32_t qs = r.a.w[2*p], S = r.a.w[2*p + 1];
            const uint32_t qhsc = r.b >> (16*p);
            const uint32_t qhx = tab_u32(tab, QHX, qhsc & 0xFF);
            const uint32_t X = perm(qhx, qs, 0x05010400u), Z = perm(qhx, qs, 0x07030602u);
            gent g0, g1, g2, g3;
            tab_pairG<GN>(tab, X, g0, g1); tab_pairG<GN>(tab, Z, g2, g3);
            yptr yr = yr0 + p*yrec<YMODE>::W;
            int s0, s1;
            dot32<YS>(g0, g1, g2, g3, yr, S, M4, SB, s0, s1);
            const int sc = (qhsc >> 8) & 0xFF;
            const int sumi = sdiv<2>(mad24(s0, sc & 0xF, mad24(s1, sc >> 4, sdiv<1>(s0 + s1))));
            sum += (dw * __int_as_float(yr[yrec<YMODE>::D])) * (float) sumi;
        }
        return sum;
    }
};

// ---------------------------------------------------------------- IQ2_XXS
// LDS tables: grid [0,512) words (1024 with GN)
template <bool YS = true, int P = 2, bool GN = false> struct iq2_xxs_t {
    static_assert(!(YS && GN), "GN is a g-side mode");
    static constexpr ggml_type type = GGML_TYPE_IQ2_XXS;
    static constexpr int TAB_WORDS = GN ? 1024 : 512, YMODE = YS ? 0 : 3, IPB = 8/P, SPI = P;
    struct raw { a_words<P> a; uint32_t d; };
    static __device__ __forceinline__ void stage_tables(uint32_t * t, int tid, int nt) {
        stage_grid<GN>(t, (const uint4 *) iq2xxs_grid, 128, tid, nt);
    }
    static __device__ __forceinline__ raw load(const soa & s, const uint32_t kbx, const uint32_t it) {
        raw r;
        r.a = load_a8<P>(s.A + (kbx*8 + P*it)*8);
        r.d = *(const uint16_t *) (s.C + kbx*2);
        return r;
    }
    static __device__ __forceinline__ float compute(const uint32_t * __restrict__ tab, const raw & r, yptr yr0) {
        const float dw = h2f(r.d);
        const uint32_t M4 = vconst(0x04040404u), SB = sconst(0x03020100u);
        float sum = 0.0f;
#pragma unroll
        for (int p = 0; p < P; ++p) {
            const uint32_t q2 = r.a.w[2*p], S0 = r.a.w[2*p + 1];
            const uint32_t S = ksign_fix(S0);
            const gent g0 = tab_g1<GN>(tab, q2 & 0xFF),         g1 = tab_g1<GN>(tab, (q2 >> 8) & 0xFF);
            const gent g2 = tab_g1<GN>(tab, (q2 >> 16) & 0xFF), g3 = tab_g1<GN>(tab, q2 >> 24);
            yptr yr = yr0 + p*yrec<YMODE>::W;
            int s0, s1;
            dot32<YS>(g0, g1, g2, g3, yr, S, M4, SB, s0, s1);
            const int ls = (int) ((S0 >> 27) | 1u);
            const int sumi = sdiv<3>(mul24(s0 + s1, ls));
            sum += (dw * __int_as_float(yr[yrec<YMODE>::D])) * (float) sumi;
        }
        return sum;
    }
};

// ---------------------------------------------------------------- IQ2_XS
// LDS tables: grid [0,1024) words (2048 with GN)
template <bool YS = true, int P = 2, bool GN = false> struct iq2_xs_t {
    static_assert(!(YS && GN), "GN is a g-side mode");
    static constexpr ggml_type type = GGML_TYPE_IQ2_XS;
    static constexpr int TAB_WORDS = GN ? 2048 : 1024, YMODE = YS ? 0 : 3, IPB = 8/P, SPI = P;
    struct raw { a_words<P> a; uint32_t b; uint32_t d; };
    static __device__ __forceinline__ void stage_tables(uint32_t * t, int tid, int nt) {
        stage_grid<GN>(t, (const uint4 *) iq2xs_grid, 256, tid, nt);
    }
    static __device__ __forceinline__ raw load(const soa & s, const uint32_t kbx, const uint32_t it) {
        raw r;
        const uint32_t sb = kbx*8 + P*it;
        r.a = load_a8<P>(s.A + sb*8);
        r.b = P == 2 ? (uint32_t) *(const uint16_t *) (s.B + sb) : (uint32_t) s.B[sb];
        r.d = *(const uint16_t *) (s.C + kbx*2);
        return r;
    }
    static __device__ __forceinline__ float compute(const uint32_t * __restrict__ tab, const raw & r, yptr yr0) {
        const float dw = h2f(r.d);
        const uint32_t M4 = vconst(0x04040404u), SB = sconst(0x03020100u);
        float sum = 0.0f;
#pragma unroll
        for (int p = 0; p < P; ++p) {
            const uint32_t Q = r.a.w[2*p], S0 = r.a.w[2*p + 1];
            const uint32_t qh4 = spread4(S0 >> 28);            // byte k = bit 8 of idx k
            const uint32_t X = perm(qh4, Q, 0x05010400u), Z = perm(qh4, Q, 0x07030602u);
            const uint32_t S = ksign_fix(S0);
            gent g0, g1, g2, g3;
            tab_pairG<GN>(tab, X, g0, g1); tab_pairG<GN>(tab, Z, g2, g3);
            yptr yr = yr0 + p*yrec<YMODE>::W;
            int s0, s1;
            dot32<YS>(g0, g1, g2, g3, yr, S, M4, SB, s0, s1);
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
#ifndef LB_Q2K_SPLIT
#define LB_Q2K_SPLIT 0
#endif
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
    static __device__ __forceinline__ float compute(const uint32_t * __restrict__, const raw & r, yptr yr0) {
        const uint32_t v[8] = {r.q0.x, r.q0.y, r.q0.z, r.q0.w, r.q1.x, r.q1.y, r.q1.z, r.q1.w};
        float sd = 0.0f, sm = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            yptr yr = yr0 + i*11;
            int s0 = 0, s1 = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                // shift i: bits 2i..2i+1 of every byte; i<3 masked in place (value scaled by 4^i), i==3 via >>4 (scaled by 4)
                const int x = i < 3 ? (int) (v[j] & (0x03030303u << (2*i))) : (int) ((v[j] >> 4) & 0x0C0C0C0Cu);
                if (j == 0) s0 = dp4a0(x, yr[j]); else if (j < 4) s0 = dp4a(x, yr[j], s0);
                else if (j == 4) s1 = dp4a0(x, yr[j]); else s1 = dp4a(x, yr[j], s1);
            }
            const uint32_t scb = (i < 2 ? r.sc.x : r.sc.y) >> (16*(i & 1));
            const int SD = mad24(s0, scb & 0xF, mul24(s1, (scb >> 8) & 0xF));
            const int SM = mad24(yr[9], (scb >> 4) & 0xF, mul24(yr[10], (scb >> 12) & 0xF));
            const float d8 = __int_as_float(yr[8]);
            const float inv = i == 0 ? 1.0f : i == 1 ? 0.25f : i == 2 ? 0.0625f : 0.25f;
            sd += (d8*inv) * (float) SD;
            sm += d8 * (float) SM;
#if LB_Q2K_SPLIT
            if (i == 1) asm volatile("" ::: "memory");   // keep the 2nd half's LDS loads below (caps live VGPRs)
#endif
        }
        const float2 dm = __half22float2(*(const half2 *) &r.dm);
        return dm.x*sd - dm.y*sm;
    }
};

// ---------------------------------------------------------------- IQ1_M
// LDS tables: pre-split grid [0,4096) words: entry i = {g & 0x0F0F0F0F, (g >> 4) & 0x0F0F0F0F}, g = iq1s_grid_gpu[i]
template <int P = 2, bool PK = false> struct iq1_m_t {
    static constexpr ggml_type type = GGML_TYPE_IQ1_M;
    static constexpr int TAB_WORDS = PK ? 2048 : 4096, YMODE = 2, IPB = 8/P, SPI = P;
    struct raw { uint32_t a[P]; uint32_t b; uint2 sc; };
    static __device__ __forceinline__ void stage_tables(uint32_t * t, int tid, int nt) {
        if constexpr (PK) { for (int i = tid; i < 512; i += nt) ((uint4 *) t)[i] = ((const uint4 *) iq1s_grid_gpu)[i]; }
        else for (int i = tid; i < 2048; i += nt) { const uint32_t g = iq1s_grid_gpu[i]; ((uint2 *) t)[i] = make_uint2(g & 0x0F0F0F0Fu, (g >> 4) & 0x0F0F0F0Fu); }
    }
    static __device__ __forceinline__ raw load(const soa & s, const uint32_t kbx, const uint32_t it) {
        raw r;
        const uint32_t sb = kbx*8 + P*it;
        if constexpr (P == 2) { const uint2 v = *(const uint2 *) (s.A + sb*4); r.a[0] = v.x; r.a[1] = v.y; r.b = *(const uint32_t *) (s.B + sb*2); }
        else { r.a[0] = *(const uint32_t *) (s.A + sb*4); r.b = *(const uint16_t *) (s.B + sb*2); }
        r.sc = *(const uint2 *) (s.C + kbx*8);
        return r;
    }
    static __device__ __forceinline__ float compute(const uint32_t * __restrict__ tab, const raw & r, yptr yr0, const int ib0) {
        // block scale (fp16 assembled from the top nibbles of the 4 scale words)
        const uint32_t u16 = ((r.sc.x & 0xFFFF) >> 12) | ((r.sc.x >> 24) & 0x00F0) | (((r.sc.y & 0xFFFF) >> 4) & 0x0F00) | ((r.sc.y >> 16) & 0xF000);
        const float dblk = h2f(u16);
        const uint32_t scw16 = ((ib0 < 4 ? r.sc.x : r.sc.y) >> (16*((ib0 >> 1) & 1))) & 0xFFFF;   // sc[ib0/2] (P <= 2)
        float sum = 0.0f;
#pragma unroll
        for (int p = 0; p < P; ++p) {
            const uint32_t Q = r.a[p];
            const uint32_t qh = (r.b >> (16*p)) & 0xFFFF;
            const uint32_t qhb = (qh & 0x0707u) | ((qh >> 4) & 0x0707u) << 16;   // bytes: n0&7, n2&7, n1&7, n3&7
            const uint32_t X = perm(qhb, Q, 0x06010400u), Z = perm(qhb, Q, 0x07030502u);
            uint2 g0, g1, g2, g3;
            if constexpr (PK) {
                const uint32_t w0 = tab_u32(tab, 0, X & 0xFFFF), w1 = tab_u32(tab, 0, X >> 16);
                const uint32_t w2 = tab_u32(tab, 0, Z & 0xFFFF), w3 = tab_u32(tab, 0, Z >> 16);
                g0 = make_uint2(w0 & 0x0F0F0F0Fu, (w0 >> 4) & 0x0F0F0F0Fu); g1 = make_uint2(w1 & 0x0F0F0F0Fu, (w1 >> 4) & 0x0F0F0F0Fu);
                g2 = make_uint2(w2 & 0x0F0F0F0Fu, (w2 >> 4) & 0x0F0F0F0Fu); g3 = make_uint2(w3 & 0x0F0F0F0Fu, (w3 >> 4) & 0x0F0F0F0Fu);
            } else {
                tab_pair(tab, X, g0, g1); tab_pair(tab, Z, g2, g3);
            }
            yptr yr = yr0 + p*yrec<2>::W;
            // two 16-weight dot chains (values 0..2 per weight; delta handled via the y sums below)
            const int i01 = dp4a((int) g1.y, yr[3], dp4a((int) g1.x, yr[2], dp4a((int) g0.y, yr[1], dp4a0((int) g0.x, yr[0]))));
            const int i23 = dp4a((int) g3.y, yr[7], dp4a((int) g3.x, yr[6], dp4a((int) g2.y, yr[5], dp4a0((int) g2.x, yr[4]))));
            // delta (x8): -7*sum8 (yr[9], yr[10]) plus -2*sum8 (yr[11..14]) when bit 3 of nibble k is set
            const int b0 = (int) ((qh >> 3) & 1), b1 = (int) ((qh >> 7) & 1), b2 = (int) ((qh >> 11) & 1), b3 = (int) ((qh >> 15) & 1);
            const int S0 = mad24(b1, yr[12], mad24(b0, yr[11], 8*i01 + yr[9]));
            const int S1 = mad24(b3, yr[14], mad24(b2, yr[13], 8*i23 + yr[10]));
            const uint32_t tmp = scw16 >> (6*((ib0 + p) & 1));
            const int sc0 = 2*(int) (tmp & 7) + 1, sc1 = 2*(int) ((tmp >> 3) & 7) + 1;
            sum += (dblk * __int_as_float(yr[8])) * ((float) mad24(S0, sc0, mul24(S1, sc1)) * 0.125f);
        }
        return sum;
    }
};

typedef iq1_m_t<2> iq1_m;
typedef iq1_m_t<2, true> iq1_m_pk;   // packed 8 KB grid table (measured faster)

// IQ1_M needs the sub-block index within the block for its scales; uniform compute wrapper:
template <class T> static __device__ __forceinline__ float compute_item(const uint32_t * tab, const typename T::raw & r, yptr yr0, int it) {
    if constexpr (T::type == GGML_TYPE_IQ1_M) return T::compute(tab, r, yr0, T::SPI*it); else { (void) it; return T::compute(tab, r, yr0); }
}

// ---------------------------------------------------------------- reference kernel
// Persistent workgroups (grid ~ 2 x #CU); y staged once per workgroup; L lanes per row, R rows per lane group,
// NW waves per workgroup, work units of (64/L)*R rows distributed round-robin over all waves.
// K % 256 == 0, N % ((64/L)*R) == 0.
// sum over groups of L lanes (result valid in every lane of the group). L=16/32/64: DPP row_ror butterfly within
// 16-lane rows (no LDS) + shuffles across rows;
// deterministic order. Other L: generic warp_reduce_sum.
#ifndef LB_DPP_REDUCE
#define LB_DPP_REDUCE 1
#endif
template <int L> static __device__ __forceinline__ float group_sum(float v) {
#if LB_DPP_REDUCE && defined(__HIP_DEVICE_COMPILE__)
    if constexpr (L == 16 || L == 32 || L == 64) {
        v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x128, 0xF, 0xF, false));   // row_ror:8
        v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x124, 0xF, 0xF, false));   // row_ror:4
        v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x122, 0xF, 0xF, false));   // row_ror:2
        v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x121, 0xF, 0xF, false));   // row_ror:1
        if constexpr (L >= 32) v += __shfl_xor(v, 16, L);
        if constexpr (L == 64) v += __shfl_xor(v, 32, 64);
        return v;
    }
#endif
    return warp_reduce_sum<L>(v);
}

template <class T>
static __device__ __forceinline__ yptr yrec_ptr(int * yrec_base, const uint32_t rec) {
#if LB_TAB0 && defined(__HIP_DEVICE_COMPILE__)
    constexpr int YW = yrec<T::YMODE>::W;
    if constexpr (T::type != GGML_TYPE_Q2_K) {
        uint32_t ya = T::TAB_WORDS*4 + __umul24(rec, YW*4);
        asm("" : "+v"(ya));   // opaque base: lets the backend fold record word offsets into ds_read2 offsets
        return (yptr) (size_t) ya;
    } else {
        return (yptr) (yrec_base + rec*YW);   // (measured faster for Q2_K)
    }
#else
    return yrec_base + rec*yrec<T::YMODE>::W;
#endif
}

// Options (all measured slower on MI210, default off): PF = first-item load issued before staging,
// SS = steady-state register prefetch of the next item. GN (trait option) = {g,-g} grid table.
#ifndef LB_MINWG
#define LB_MINWG 1    // 2: cap VGPRs so two NW=16 workgroups fit per CU
#endif
template <class T, int L, int R, int NW, bool PF = false, bool SS = false>
__launch_bounds__(NW*64, (NW == 16 && LB_MINWG == 2) ? 8 : 1)   /* 2nd arg = min waves per SIMD */
__global__ void gemv(const soa s, const block_q8_1 * __restrict__ yg, float * __restrict__ dst, const int K, const int N,
        const float * __restrict__ residual = nullptr) {
    extern __shared__ uint32_t smem[];   // [tables (at LDS address 0, see LB_TAB0) | y records]
    uint32_t * tab = smem;
    int * yrec_base = (int *) (smem + T::TAB_WORDS);
#if LB_TAB0 && defined(__HIP_DEVICE_COMPILE__)
    // contract of LB_TAB0: no static __shared__ in this kernel, so the dynamic segment starts at LDS address 0
    if ((uint32_t) (size_t) (__attribute__((address_space(3))) uint32_t *) smem != 0) __builtin_trap();
#endif
    constexpr int G = 64/L;
    const int tid = threadIdx.y*64 + threadIdx.x;
    const int lane = threadIdx.x % L, grp = threadIdx.x / L;
    const int bpr = K / QK_K;
    const int items = bpr*T::IPB;
    const int U = N / (G*R);
    const int ustep = gridDim.x*NW;
    // CU-balanced round robin: consecutive units go to different workgroups first
    int u = threadIdx.y*gridDim.x + blockIdx.x;
    typename T::raw pre[R];
    if (PF && u < U) {
        const int row0 = (u*G + grp)*R;
#pragma unroll
        for (int r = 0; r < R; ++r) pre[r] = T::load(s, (uint32_t) (row0 + r)*bpr + lane / T::IPB, lane % T::IPB);
    }
    T::stage_tables(tab, tid, NW*64);
    stage_y<T::YMODE>(yrec_base, yg, K, tid, NW*64);
    __syncthreads();
    bool first = PF;
    for (; u < U; u += ustep) {
        const int row0 = (u*G + grp)*R;
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] = 0.0f;
        int it = lane;
        if (PF && first) {   // peeled first item (lane < items always holds: items >= L)
            first = false;
            const yptr yr0 = yrec_ptr<T>(yrec_base, (it / T::IPB)*8 + (it % T::IPB)*T::SPI);
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r] += compute_item<T>(tab, pre[r], yr0, it % T::IPB);
            it += L;
        }
        if constexpr (SS) {
            // steady-state register prefetch: loads of item it+L are in flight while item it is computed
            typename T::raw cur[R];
#pragma unroll
            for (int r = 0; r < R; ++r) cur[r] = T::load(s, (uint32_t) (row0 + r)*bpr + it / T::IPB, it % T::IPB);
            for (; it < items; it += L) {
                const uint32_t kb = it / T::IPB, ii = it % T::IPB;
                const int itn = min(it + L, items - 1);
                typename T::raw nxt[R];
#pragma unroll
                for (int r = 0; r < R; ++r) nxt[r] = T::load(s, (uint32_t) (row0 + r)*bpr + itn / T::IPB, itn % T::IPB);
                const yptr yr0 = yrec_ptr<T>(yrec_base, kb*8 + ii*T::SPI);
#pragma unroll
                for (int r = 0; r < R; ++r) acc[r] += compute_item<T>(tab, cur[r], yr0, ii);
#pragma unroll
                for (int r = 0; r < R; ++r) cur[r] = nxt[r];
            }
        } else {
        for (; it < items; it += L) {
            const uint32_t kb = it / T::IPB, ii = it % T::IPB;
            const yptr yr0 = yrec_ptr<T>(yrec_base, kb*8 + ii*T::SPI);
            typename T::raw cur[R];
#pragma unroll
            for (int r = 0; r < R; ++r) cur[r] = T::load(s, (uint32_t) (row0 + r)*bpr + kb, ii);
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r] += compute_item<T>(tab, cur[r], yr0, ii);
        }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float v = group_sum<L>(acc[r]);
            if (lane == 0) dst[row0 + r] = residual ? v + residual[row0 + r] : v;
        }
    }
}

// dynamic LDS bytes to pass at launch
template <class T> static inline size_t gemv_lds_bytes(int K) { return (size_t) T::TAB_WORDS*4 + (size_t) (K/32)*yrec<T::YMODE>::W*4; }
template <class T> static inline size_t gemv_lds_total(int K) { return (size_t) T::TAB_WORDS*4 + (size_t) (K/32)*yrec<T::YMODE>::W*4; }


// ================================================================================================
// Host-side launcher with tuned configurations (MI210, measured with lowbit_bench).
//   soa_base : repacked weights (lb::repack_<type>, same byte size as the ggml tensor), nb = N*K/256
//   y        : q8_1 activation (K/32 block_q8_1) in global memory; staged into LDS by the kernel
//   dst      : float[N]
// Requirements: K % 256 == 0, N % 64 == 0 (all model shapes satisfy this).
struct launch_cfg { int L, R, NW, grid; bool ys; int P; bool gn; };

#ifndef LB_GN_S
#define LB_GN_S 0     // IQ2_S large-N: g-side with {g,-g} grid table
#endif
#ifndef LB_GN_XXS
#define LB_GN_XXS 0   // IQ2_XXS large-N: g-side with {g,-g} grid table
#endif
static inline launch_cfg pick_cfg(ggml_type t, int N, int K) {
    const bool bigK = K > 8192;
    switch (t) {
        case GGML_TYPE_IQ2_S:
            if (bigK)       return launch_cfg{64, 1, 16, 208, false, 1, false};
            if (N <= 2048)  return launch_cfg{64, 1, 16, 104, false, 2, false};
            if (N <= 10240) return launch_cfg{32, 1, 16, 208, false, 1, false};
            return launch_cfg{16, 1, 16, 208, false, 2, LB_GN_S != 0};
        case GGML_TYPE_IQ2_XS:
            if (bigK)       return launch_cfg{64, 1, 16, 208, false, 1, false};
            if (N <= 10240) return launch_cfg{32, 1, 16, 208, true, 1, false};
            return launch_cfg{16, 1, 16, 208, true, 2, false};
        case GGML_TYPE_IQ2_XXS:
            if (bigK)       return launch_cfg{64, 1, 16, 208, false, 1, false};
            if (N >= 16384) return launch_cfg{16, 1, 16, 208, false, 2, LB_GN_XXS != 0};
            if (N >= 12288) return launch_cfg{16, 2, 16, 104, true, 2, false};
            return launch_cfg{32, 1, 16, 208, true, 1, false};
        case GGML_TYPE_Q2_K:
            if (bigK)       return launch_cfg{16, 1, 16, 208, true, 2, false};
            if (N >= 16384) return launch_cfg{16, 1, 16, 104, true, 2, false};
            if (N >= 10240) return launch_cfg{16, 1, 8, 208, true, 2, false};
            return launch_cfg{16, 1, 16, 208, true, 2, false};
        case GGML_TYPE_IQ1_M:
            return launch_cfg{16, 1, 16, 208, true, 2, false};
        default:
            return launch_cfg{0, 0, 0, 0, false, 0, false};
    }
}

template <class T, int L, int R, int NW>
static inline bool try_launch(const launch_cfg & c, const soa & s, const block_q8_1 * y, float * dst, int K, int N, hipStream_t st,
        const float * residual) {
    if (c.L != L || c.R != R || c.NW != NW) return false;
    if (gemv_lds_total<T>(K) > 65536) return false;
    hipLaunchKernelGGL((gemv<T, L, R, NW>), dim3(c.grid), dim3(64, NW), gemv_lds_bytes<T>(K), st, s, y, dst, K, N, residual);
    return true;
}

// returns false if the type / shape is not supported (caller falls back to the stock path)
// residual (optional): dst[row] = dot(row) + residual[row]
static inline bool launch(ggml_type t, const void * soa_base, const block_q8_1 * y, float * dst, int K, int N, hipStream_t st,
        const float * residual = nullptr) {
    if (K % 256 != 0 || N % 64 != 0) return false;
    const size_t nb = (size_t) N*(K/QK_K);
    const soa s = soa_make(t, soa_base, nb);
    const launch_cfg c = pick_cfg(t, N, K);
#define LB_TRY(...) if (try_launch<__VA_ARGS__>(c, s, y, dst, K, N, st, residual)) return true;
    switch (t) {
        case GGML_TYPE_IQ2_S:
            if (c.P == 1)    { LB_TRY(iq2_s_t<false, 1>, 64, 1, 16) LB_TRY(iq2_s_t<false, 1>, 32, 1, 16) }
            else if (c.gn)   { LB_TRY(iq2_s_t<false, 2, true>, 16, 1, 16) }
            else             { LB_TRY(iq2_s_t<false, 2>, 16, 1, 16) LB_TRY(iq2_s_t<false, 2>, 64, 1, 16) }
            break;
        case GGML_TYPE_IQ2_XS:
            if (c.P == 1)    { if (c.ys) { LB_TRY(iq2_xs_t<true, 1>, 32, 1, 16) } else { LB_TRY(iq2_xs_t<false, 1>, 64, 1, 16) } }
            else             { LB_TRY(iq2_xs_t<true, 2>, 16, 1, 16) }
            break;
        case GGML_TYPE_IQ2_XXS:
            if (c.P == 1)    { if (c.ys) { LB_TRY(iq2_xxs_t<true, 1>, 32, 1, 16) } else { LB_TRY(iq2_xxs_t<false, 1>, 64, 1, 16) } }
            else if (c.gn)   { LB_TRY(iq2_xxs_t<false, 2, true>, 16, 1, 16) }
            else if (c.ys)   { LB_TRY(iq2_xxs_t<true, 2>, 16, 2, 16) }
            else             { LB_TRY(iq2_xxs_t<false, 2>, 16, 1, 16) }
            break;
        case GGML_TYPE_Q2_K:
            LB_TRY(q2_K, 16, 1, 16) LB_TRY(q2_K, 16, 1, 8)
            break;
        case GGML_TYPE_IQ1_M:
            LB_TRY(iq1_m_pk, 16, 1, 16)
            break;
        default: break;
    }
#undef LB_TRY
    return false;
}

// repack launcher: src = original ggml blocks (device), dst = separate buffer of the same size (device)
static inline bool repack(ggml_type t, const void * src, void * dst, size_t nb, hipStream_t st) {
    const int nthr = 256; const int nblk = (int) ((nb + nthr - 1)/nthr);
    switch (t) {
        case GGML_TYPE_IQ2_S:   hipLaunchKernelGGL(repack_iq2_s,   dim3(nblk), dim3(nthr), 0, st, (const block_iq2_s *)   src, (uint8_t *) dst, nb); return true;
        case GGML_TYPE_IQ2_XS:  hipLaunchKernelGGL(repack_iq2_xs,  dim3(nblk), dim3(nthr), 0, st, (const block_iq2_xs *)  src, (uint8_t *) dst, nb); return true;
        case GGML_TYPE_IQ2_XXS: hipLaunchKernelGGL(repack_iq2_xxs, dim3(nblk), dim3(nthr), 0, st, (const block_iq2_xxs *) src, (uint8_t *) dst, nb); return true;
        case GGML_TYPE_Q2_K:    hipLaunchKernelGGL(repack_q2_K,    dim3(nblk), dim3(nthr), 0, st, (const block_q2_K *)    src, (uint8_t *) dst, nb); return true;
        case GGML_TYPE_IQ1_M:   hipLaunchKernelGGL(repack_iq1_m,   dim3(nblk), dim3(nthr), 0, st, (const block_iq1_m *)   src, (uint8_t *) dst, nb); return true;
        default: return false;
    }
}

static inline bool supported(ggml_type t) {
    return t == GGML_TYPE_IQ2_S || t == GGML_TYPE_IQ2_XS || t == GGML_TYPE_IQ2_XXS || t == GGML_TYPE_Q2_K || t == GGML_TYPE_IQ1_M;
}

// MMVQ-NC lowbit: would launch() (the single-token launcher) take this shape? Same configuration choice and the same LDS limit
// as try_launch; the N-column path must only run where the single-token path is this kernel too (else its reference is
// stock mmvq, e.g. IQ1_M at K >= 30720, and the columns would not be bit-identical).
template <class T, int L, int R, int NW>
static inline bool try_launch_ok(const launch_cfg & c, int K) {
    return c.L == L && c.R == R && c.NW == NW && gemv_lds_total<T>(K) <= 65536;
}
static inline bool launch_ok(ggml_type t, int K, int N) {
    if (K % 256 != 0 || N % 64 != 0) return false;
    const launch_cfg c = pick_cfg(t, N, K);
#define LB_OK(...) if (try_launch_ok<__VA_ARGS__>(c, K)) return true;
    switch (t) {
        case GGML_TYPE_IQ2_S:
            if (c.P == 1)    { LB_OK(iq2_s_t<false, 1>, 64, 1, 16) LB_OK(iq2_s_t<false, 1>, 32, 1, 16) }
            else if (c.gn)   { LB_OK(iq2_s_t<false, 2, true>, 16, 1, 16) }
            else             { LB_OK(iq2_s_t<false, 2>, 16, 1, 16) LB_OK(iq2_s_t<false, 2>, 64, 1, 16) }
            break;
        case GGML_TYPE_IQ2_XS:
            if (c.P == 1)    { if (c.ys) { LB_OK(iq2_xs_t<true, 1>, 32, 1, 16) } else { LB_OK(iq2_xs_t<false, 1>, 64, 1, 16) } }
            else             { LB_OK(iq2_xs_t<true, 2>, 16, 1, 16) }
            break;
        case GGML_TYPE_IQ2_XXS:
            if (c.P == 1)    { if (c.ys) { LB_OK(iq2_xxs_t<true, 1>, 32, 1, 16) } else { LB_OK(iq2_xxs_t<false, 1>, 64, 1, 16) } }
            else if (c.gn)   { LB_OK(iq2_xxs_t<false, 2, true>, 16, 1, 16) }
            else if (c.ys)   { LB_OK(iq2_xxs_t<true, 2>, 16, 2, 16) }
            else             { LB_OK(iq2_xxs_t<false, 2>, 16, 1, 16) }
            break;
        case GGML_TYPE_Q2_K:
            LB_OK(q2_K, 16, 1, 16) LB_OK(q2_K, 16, 1, 8)
            break;
        case GGML_TYPE_IQ1_M:
            LB_OK(iq1_m_pk, 16, 1, 16)
            break;
        default: break;
    }
#undef LB_OK
    return false;
}

// ================================================================================================ MMVQ-NC lowbit
// N-column GEMV (NC = 2..6 activation columns, one pass over the weights). Column c: q8_1 blocks at yg + c*ys,
// output at dst + c*ldd, optional residual at residual + c*ldr.
// Per column bit-identical to gemv<> on that column alone when L and the item size P equal those of the N=1 config
// (pick_cfg): lane l then accumulates the same items l, l+L, ... in the same order, followed by the same
// group_sum<L> and epilogue, and each item's fp32 value is the same expression of the same exact integer dot products.
// The fp32 expression is spelled out with explicit roundings in compute_nc (contraction off, fma only where the N=1
// kernels have one), because the compiler's contraction of compute() depends on how it vectorizes the surrounding
// code (measured: compute() of the N=1 kernels is unfused for the IQ2/IQ1_M types and an fma chain for Q2_K; inlined
// NC times into one loop it fuses differently). N=1 code as compiled (gfx90a, ROCm 7.2):
//   IQ2_*  : sum = 0; for p < P: sum += (dw*d8_p) * (float) sumi_p          (no fma; P = 1: fma(dw*d8, sumi, 0))
//   IQ1_M  : sum = 0; for p < P: sum += (dblk*d8_p) * ((float) isum_p * 0.125f)   (no fma)
//   Q2_K   : sd = fma(d8_i*inv_i, SD_i, sd), sm = fma(d8_i, SM_i, sm) for i = 0..3 from 0; dm.x*sd - dm.y*sm
//   then acc += item (one add per item, item order per lane as above).
// Free to differ (no value changes): the sign side (YS or g-side: the integer dot is exact either way), the grid
// table form, rows per lane group R, NW and the grid (they only change which wave owns a row).
// The g-side form (YMODE 3: 9-word y records) is used for the IQ2 types: the grid lookups and signed weights are
// shared by all columns, each column adds one LDS read + one dot per dword.
#define LB_NC_MAX 6

template <class T> struct nc_traits;
template <int P, bool GN> struct nc_traits<iq2_s_t<false, P, GN>>   { static constexpr int kind = 0; };
template <int P, bool GN> struct nc_traits<iq2_xs_t<false, P, GN>>  { static constexpr int kind = 1; };
template <int P, bool GN> struct nc_traits<iq2_xxs_t<false, P, GN>> { static constexpr int kind = 2; };
template <> struct nc_traits<q2_K>     { static constexpr int kind = 3; };
template <> struct nc_traits<iq1_m_pk> { static constexpr int kind = 4; };

// item value for NC columns (column c: y records at yr0 + c*colw); out[c] is what compute_item<T> gives on column c
template <class T, int NC>
static __device__ __forceinline__ void compute_nc(const uint32_t * __restrict__ tab, const typename T::raw & r, yptr yr0, const int colw,
        const int it, float (&out)[NC]) {
#pragma clang fp contract(off)
    constexpr int kind = nc_traits<T>::kind;
    if constexpr (kind <= 2) {   // IQ2_S / IQ2_XS / IQ2_XXS, g-side signs (YMODE 3), P = T::SPI sub-blocks per item
        constexpr int P = T::SPI;
        constexpr int W = yrec<3>::W;
        static_assert(T::YMODE == 3, "NC uses g-side IQ2 traits");
        const float dw = h2f(r.d);
        const uint32_t M4 = vconst(0x04040404u), SB = sconst(0x03020100u);
        float sum[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) sum[c] = 0.0f;
#pragma unroll
        for (int p = 0; p < P; ++p) {
            const uint32_t q = r.a.w[2*p], S0 = r.a.w[2*p + 1];
            gent g0, g1, g2, g3;
            uint32_t S;
            int sc = 0;
            if constexpr (kind == 0) {
                const uint32_t qhsc = r.b >> (16*p);
                const uint32_t qhx = tab_u32(tab, T::QHX, qhsc & 0xFF);
                const uint32_t X = perm(qhx, q, 0x05010400u), Z = perm(qhx, q, 0x07030602u);
                tab_pairG<false>(tab, X, g0, g1); tab_pairG<false>(tab, Z, g2, g3);
                S = S0;
                sc = (qhsc >> 8) & 0xFF;
            } else if constexpr (kind == 1) {
                const uint32_t qh4 = spread4(S0 >> 28);
                const uint32_t X = perm(qh4, q, 0x05010400u), Z = perm(qh4, q, 0x07030602u);
                S = ksign_fix(S0);
                tab_pairG<false>(tab, X, g0, g1); tab_pairG<false>(tab, Z, g2, g3);
                sc = (r.b >> (8*p)) & 0xFF;
            } else {
                S = ksign_fix(S0);
                g0 = tab_g1<false>(tab, q & 0xFF);         g1 = tab_g1<false>(tab, (q >> 8) & 0xFF);
                g2 = tab_g1<false>(tab, (q >> 16) & 0xFF); g3 = tab_g1<false>(tab, q >> 24);
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                yptr yr = yr0 + c*colw + p*W;
                int s0, s1;
                dot32<false>(g0, g1, g2, g3, yr, S, M4, SB, s0, s1);
                int sumi;
                if constexpr (kind == 2) sumi = sdiv<3>(mul24(s0 + s1, (int) ((S0 >> 27) | 1u)));
                else                     sumi = sdiv<2>(mad24(s0, sc & 0xF, mad24(s1, sc >> 4, sdiv<1>(s0 + s1))));
                const float m = dw * __int_as_float(yr[yrec<3>::D]);
                sum[c] = sum[c] + m * (float) sumi;
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) out[c] = sum[c];
    } else if constexpr (kind == 3) {   // Q2_K
        const uint32_t v[8] = {r.q0.x, r.q0.y, r.q0.z, r.q0.w, r.q1.x, r.q1.y, r.q1.z, r.q1.w};
        float sd[NC], sm[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) { sd[c] = 0.0f; sm[c] = 0.0f; }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int x[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) x[j] = i < 3 ? (int) (v[j] & (0x03030303u << (2*i))) : (int) ((v[j] >> 4) & 0x0C0C0C0Cu);
            const uint32_t scb = (i < 2 ? r.sc.x : r.sc.y) >> (16*(i & 1));
            const float inv = i == 0 ? 1.0f : i == 1 ? 0.25f : i == 2 ? 0.0625f : 0.25f;
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                yptr yr = yr0 + c*colw + i*11;
                int s0 = dp4a0(x[0], yr[0]);
                s0 = dp4a(x[1], yr[1], s0); s0 = dp4a(x[2], yr[2], s0); s0 = dp4a(x[3], yr[3], s0);
                int s1 = dp4a0(x[4], yr[4]);
                s1 = dp4a(x[5], yr[5], s1); s1 = dp4a(x[6], yr[6], s1); s1 = dp4a(x[7], yr[7], s1);
                const int SD = mad24(s0, scb & 0xF, mul24(s1, (scb >> 8) & 0xF));
                const int SM = mad24(yr[9], (scb >> 4) & 0xF, mul24(yr[10], (scb >> 12) & 0xF));
                const float d8 = __int_as_float(yr[8]);
                sd[c] = __builtin_fmaf(d8*inv, (float) SD, sd[c]);
                sm[c] = __builtin_fmaf(d8, (float) SM, sm[c]);
            }
        }
        const float2 dm = __half22float2(*(const half2 *) &r.dm);
#pragma unroll
        for (int c = 0; c < NC; ++c) out[c] = dm.x*sd[c] - dm.y*sm[c];
    } else {   // IQ1_M (packed grid table), P = 2
        constexpr int P = T::SPI;
        const int ib0 = T::SPI*it;
        const uint32_t u16 = ((r.sc.x & 0xFFFF) >> 12) | ((r.sc.x >> 24) & 0x00F0) | (((r.sc.y & 0xFFFF) >> 4) & 0x0F00) | ((r.sc.y >> 16) & 0xF000);
        const float dblk = h2f(u16);
        const uint32_t scw16 = ((ib0 < 4 ? r.sc.x : r.sc.y) >> (16*((ib0 >> 1) & 1))) & 0xFFFF;
        float sum[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) sum[c] = 0.0f;
#pragma unroll
        for (int p = 0; p < P; ++p) {
            const uint32_t Q = r.a[p];
            const uint32_t qh = (r.b >> (16*p)) & 0xFFFF;
            const uint32_t qhb = (qh & 0x0707u) | ((qh >> 4) & 0x0707u) << 16;
            const uint32_t X = perm(qhb, Q, 0x06010400u), Z = perm(qhb, Q, 0x07030502u);
            const uint32_t w0 = tab_u32(tab, 0, X & 0xFFFF), w1 = tab_u32(tab, 0, X >> 16);
            const uint32_t w2 = tab_u32(tab, 0, Z & 0xFFFF), w3 = tab_u32(tab, 0, Z >> 16);
            const uint2 g0 = make_uint2(w0 & 0x0F0F0F0Fu, (w0 >> 4) & 0x0F0F0F0Fu), g1 = make_uint2(w1 & 0x0F0F0F0Fu, (w1 >> 4) & 0x0F0F0F0Fu);
            const uint2 g2 = make_uint2(w2 & 0x0F0F0F0Fu, (w2 >> 4) & 0x0F0F0F0Fu), g3 = make_uint2(w3 & 0x0F0F0F0Fu, (w3 >> 4) & 0x0F0F0F0Fu);
            const int b0 = (int) ((qh >> 3) & 1), b1 = (int) ((qh >> 7) & 1), b2 = (int) ((qh >> 11) & 1), b3 = (int) ((qh >> 15) & 1);
            const uint32_t tmp = scw16 >> (6*((ib0 + p) & 1));
            const int sc0 = 2*(int) (tmp & 7) + 1, sc1 = 2*(int) ((tmp >> 3) & 7) + 1;
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                yptr yr = yr0 + c*colw + p*yrec<2>::W;
                const int i01 = dp4a((int) g1.y, yr[3], dp4a((int) g1.x, yr[2], dp4a((int) g0.y, yr[1], dp4a0((int) g0.x, yr[0]))));
                const int i23 = dp4a((int) g3.y, yr[7], dp4a((int) g3.x, yr[6], dp4a((int) g2.y, yr[5], dp4a0((int) g2.x, yr[4]))));
                const int S0 = mad24(b1, yr[12], mad24(b0, yr[11], 8*i01 + yr[9]));
                const int S1 = mad24(b3, yr[14], mad24(b2, yr[13], 8*i23 + yr[10]));
                const float m = dblk * __int_as_float(yr[8]);
                const float f = (float) mad24(S0, sc0, mul24(S1, sc1)) * 0.125f;
                sum[c] = sum[c] + m * f;
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) out[c] = sum[c];
    }
}
template <class T, int L, int R, int NW, int NC>
__launch_bounds__(NW*64, 1)
__global__ void gemv_nc(const soa s, const block_q8_1 * __restrict__ yg, const int ys, float * __restrict__ dst, const int ldd,
        const int K, const int N, const float * __restrict__ residual, const int ldr) {
#pragma clang fp contract(off)
    extern __shared__ uint32_t smem[];   // [tables (at LDS address 0) | y records of column 0 | column 1 | ...]
    uint32_t * tab = smem;
    int * yrec_base = (int *) (smem + T::TAB_WORDS);
#if LB_TAB0 && defined(__HIP_DEVICE_COMPILE__)
    if ((uint32_t) (size_t) (__attribute__((address_space(3))) uint32_t *) smem != 0) __builtin_trap();
#endif
    constexpr int G = 64/L;
    const int tid = threadIdx.y*64 + threadIdx.x;
    const int lane = threadIdx.x % L, grp = threadIdx.x / L;
    const int bpr = K / QK_K;
    const int items = bpr*T::IPB;
    const int U = N / (G*R);
    const int ustep = gridDim.x*NW;
    const int colw = (K/32)*yrec<T::YMODE>::W;   // y record words per column
    int u = threadIdx.y*gridDim.x + blockIdx.x;
    T::stage_tables(tab, tid, NW*64);
#pragma unroll
    for (int c = 0; c < NC; ++c) stage_y<T::YMODE>(yrec_base + c*colw, yg + (size_t) c*ys, K, tid, NW*64);
    __syncthreads();
    for (; u < U; u += ustep) {
        const int row0 = (u*G + grp)*R;
        float acc[NC][R];
#pragma unroll
        for (int c = 0; c < NC; ++c)
#pragma unroll
            for (int r = 0; r < R; ++r) acc[c][r] = 0.0f;
        for (int it = lane; it < items; it += L) {
            const uint32_t kb = it / T::IPB, ii = it % T::IPB;
            const yptr yr0 = yrec_ptr<T>(yrec_base, kb*8 + ii*T::SPI);
            typename T::raw cur[R];
#pragma unroll
            for (int r = 0; r < R; ++r) cur[r] = T::load(s, (uint32_t) (row0 + r)*bpr + kb, ii);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float v[NC];
                compute_nc<T, NC>(tab, cur[r], yr0, colw, ii, v);
#pragma unroll
                for (int c = 0; c < NC; ++c) acc[c][r] = acc[c][r] + v[c];
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c)
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float v = group_sum<L>(acc[c][r]);
                if (lane == 0) {
                    const int row = row0 + r;
                    dst[(size_t) c*ldd + row] = residual ? v + residual[(size_t) c*ldr + row] : v;
                }
            }
    }
}

// K-chunked variant for shapes where the y records of all NC columns do not fit next to the tables (ffn_down,
// K = 17408): y is staged KCB 256-blocks at a time and each wave keeps the accumulators of its UM units (unit k:
// u0 + k*ustep) across the chunks. Lane l still takes items l, l+L, ... in increasing order (a chunk boundary only
// pauses it), so the result per column is the same as gemv<> / gemv_nc<>.
template <class T, int L, int R, int NW, int NC, int UM>
__launch_bounds__(NW*64, 1)
__global__ void gemv_nck(const soa s, const block_q8_1 * __restrict__ yg, const int ys, float * __restrict__ dst, const int ldd,
        const int K, const int N, const float * __restrict__ residual, const int ldr, const int KCB) {
#pragma clang fp contract(off)
    extern __shared__ uint32_t smem[];   // [tables (at LDS address 0) | chunk y records of column 0 | column 1 | ...]
    uint32_t * tab = smem;
    int * yrec_base = (int *) (smem + T::TAB_WORDS);
#if LB_TAB0 && defined(__HIP_DEVICE_COMPILE__)
    if ((uint32_t) (size_t) (__attribute__((address_space(3))) uint32_t *) smem != 0) __builtin_trap();
#endif
    constexpr int G = 64/L;
    const int tid = threadIdx.y*64 + threadIdx.x;
    const int lane = threadIdx.x % L, grp = threadIdx.x / L;
    const int bpr = K / QK_K;
    const int U = N / (G*R);
    const int ustep = gridDim.x*NW;
    const int colw = KCB*8*yrec<T::YMODE>::W;   // y record words per column per chunk
    const int u0 = threadIdx.y*gridDim.x + blockIdx.x;
    T::stage_tables(tab, tid, NW*64);
    float acc[UM][NC][R];
#pragma unroll
    for (int k = 0; k < UM; ++k)
#pragma unroll
        for (int c = 0; c < NC; ++c)
#pragma unroll
            for (int r = 0; r < R; ++r) acc[k][c][r] = 0.0f;
    int it0 = lane;   // next item of this lane
    for (int kb0 = 0; kb0 < bpr; kb0 += KCB) {
        const int nkb = min(KCB, bpr - kb0);
        __syncthreads();   // the previous chunk (and the tables, first time) are done with
#pragma unroll
        for (int c = 0; c < NC; ++c) stage_y<T::YMODE>(yrec_base + c*colw, yg + (size_t) c*ys + (size_t) kb0*(QK_K/QK8_1), nkb*QK_K, tid, NW*64);
        __syncthreads();
        const int it_end = (kb0 + nkb)*T::IPB;
#pragma unroll
        for (int k = 0; k < UM; ++k) {
            const int u = u0 + k*ustep;
            if (u >= U) break;
            const int row0 = (u*G + grp)*R;
            for (int it = it0; it < it_end; it += L) {
                const uint32_t kb = it / T::IPB, ii = it % T::IPB;
                const yptr yr0 = yrec_ptr<T>(yrec_base, (kb - kb0)*8 + ii*T::SPI);
                typename T::raw cur[R];
#pragma unroll
                for (int r = 0; r < R; ++r) cur[r] = T::load(s, (uint32_t) (row0 + r)*bpr + kb, ii);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float v[NC];
                    compute_nc<T, NC>(tab, cur[r], yr0, colw, ii, v);
#pragma unroll
                    for (int c = 0; c < NC; ++c) acc[k][c][r] = acc[k][c][r] + v[c];
                }
            }
        }
        if (it0 < it_end) it0 += ((it_end - it0 + L - 1)/L)*L;
    }
#pragma unroll
    for (int k = 0; k < UM; ++k) {
        const int u = u0 + k*ustep;
        if (u >= U) break;
        const int row0 = (u*G + grp)*R;
#pragma unroll
        for (int c = 0; c < NC; ++c)
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float v = group_sum<L>(acc[k][c][r]);
                if (lane == 0) {
                    const int row = row0 + r;
                    dst[(size_t) c*ldd + row] = residual ? v + residual[(size_t) c*ldr + row] : v;
                }
            }
    }
}

template <class T> static inline size_t gemv_nc_lds(int K, int nc) {
    return (size_t) T::TAB_WORDS*4 + (size_t) nc*(K/32)*yrec<T::YMODE>::W*4;
}

static inline int nc_env(const char * name, int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

static inline int nc_num_cu() {
    static const int n = [] {
        int dev = 0, v = 0;
        if (hipGetDevice(&dev) != hipSuccess || hipDeviceGetAttribute(&v, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess || v <= 0) {
            (void) hipGetLastError();
            v = 104;
        }
        return v;
    }();
    return n;
}

// K-chunked launch of all nc columns; 0 if not applicable (then column passes)
template <class T, int L, int R, int NW>
static inline int launch_nck_t(const soa & s, const block_q8_1 * y, int ys, float * dst, int ldd, int K, int N, int nc, int grid,
        hipStream_t st, const float * residual, int ldr) {
    // only the configs whose y records can overflow LDS at NC <= 6 and where one K-chunked pass beats column passes
    // (measured, ffn_down 5120x17408 at NC = 4: IQ2_S 114 -> 86 us; Q2_K: column passes of 2 are faster, 2x45 us
    // vs 97 us): the K > 8192 IQ2 ones (L = 64) and IQ1_M (K = 5120 at NC = 6)
    if constexpr (!(L == 64 || T::type == GGML_TYPE_IQ1_M)) {
        GGML_UNUSED_VARS(s, y, ys, dst, ldd, K, N, nc, grid, st, residual, ldr);
        return 0;
    } else {
    static const int enabled = nc_env("GGML_MMVQ_NC_LB_KCHUNK", 1);
    static const int grid_k  = nc_env("GGML_MMVQ_NC_LB_KGRID", 0);
    if (!enabled) return 0;
    const int bpr = K/QK_K;
    const size_t per_blk = (size_t) nc*8*yrec<T::YMODE>::W*4;   // LDS bytes per 256-block of all columns
    static const int lds_cap = std::min(nc_env("GGML_MMVQ_NC_LB_KLDS", 65536), 65536);   // LDS per workgroup (32768: 2 per CU; at most 64 KB)
    const int kcb_max = (int) (((size_t) lds_cap - (size_t) T::TAB_WORDS*4)/per_blk);
    if (kcb_max < 1) return 0;
    const int nchunks = (bpr + kcb_max - 1)/kcb_max;
    const int KCB = (bpr + nchunks - 1)/nchunks;
    grid = grid_k > 0 ? grid_k : nc_num_cu();   // one workgroup per CU (the chunk fills the LDS)
    const int U = N/((64/L)*R);
    const int um = (U + grid*NW - 1)/(grid*NW);
    const size_t lds = (size_t) T::TAB_WORDS*4 + (size_t) KCB*per_blk;
    const dim3 b(64, NW);
#define LB_NCK(NC_, UM_) if (nc == NC_ && um == UM_) { hipLaunchKernelGGL((gemv_nck<T, L, R, NW, NC_, UM_>), dim3(grid), b, lds, st, s, y, ys, dst, ldd, K, N, residual, ldr, KCB); return NC_; }
#define LB_NCK_U(NC_) LB_NCK(NC_, 1) LB_NCK(NC_, 2) LB_NCK(NC_, 3) LB_NCK(NC_, 4)
    LB_NCK_U(2) LB_NCK_U(3) LB_NCK_U(4) LB_NCK_U(5) LB_NCK_U(6)
#undef LB_NCK_U
#undef LB_NCK
    return 0;
    }
}

// tuning knobs for the NC launches (debug: GGML_MMVQ_NC_LB_GRID overrides the grid of every NC launch)
static inline int nc_grid_override() {
    static const int g = [] { const char * v = getenv("GGML_MMVQ_NC_LB_GRID"); return v ? atoi(v) : 0; }();
    return g;
}

// largest column count <= nc whose y records fit next to the tables (1 = only the N=1 kernel fits)
template <class T> static inline int nc_fit(int K, int nc) {
    while (nc > 1 && gemv_nc_lds<T>(K, nc) > 65536) --nc;
    return nc;
}

template <class T, int L, int R, int NW>
static inline int launch_nc_t(const soa & s, const block_q8_1 * y, int ys, float * dst, int ldd, int K, int N, int nc, int grid,
        hipStream_t st, const float * residual, int ldr) {
    const int fit = nc_fit<T>(K, nc);
    if (fit < nc) {   // all columns in one pass over the weights with K-chunked y staging
        const int done = launch_nck_t<T, L, R, NW>(s, y, ys, dst, ldd, K, N, nc, grid, st, residual, ldr);
        if (done) return done;
    }
    if (fit < nc && fit >= 2) {   // column passes of balanced sizes (4 = 2+2, not 3+1)
        const int passes = (nc + fit - 1)/fit;
        nc = (nc + passes - 1)/passes;
    } else {
        nc = fit;
    }
    if (N % ((64/L)*R) != 0) return 0;
    const size_t lds = gemv_nc_lds<T>(K, nc);
    const dim3 b(64, NW);
    switch (nc) {
        case 2: hipLaunchKernelGGL((gemv_nc<T, L, R, NW, 2>), dim3(grid), b, lds, st, s, y, ys, dst, ldd, K, N, residual, ldr); return 2;
        case 3: hipLaunchKernelGGL((gemv_nc<T, L, R, NW, 3>), dim3(grid), b, lds, st, s, y, ys, dst, ldd, K, N, residual, ldr); return 3;
        case 4: hipLaunchKernelGGL((gemv_nc<T, L, R, NW, 4>), dim3(grid), b, lds, st, s, y, ys, dst, ldd, K, N, residual, ldr); return 4;
        case 5: hipLaunchKernelGGL((gemv_nc<T, L, R, NW, 5>), dim3(grid), b, lds, st, s, y, ys, dst, ldd, K, N, residual, ldr); return 5;
        case 6: hipLaunchKernelGGL((gemv_nc<T, L, R, NW, 6>), dim3(grid), b, lds, st, s, y, ys, dst, ldd, K, N, residual, ldr); return 6;
        default: return 0;   // 1 column (or none) fits: the caller runs the N=1 kernel
    }
}

typedef iq2_s_t<false, 1> nc_s1; typedef iq2_s_t<false, 2> nc_s2; typedef iq2_xs_t<false, 1> nc_xs1; typedef iq2_xs_t<false, 2> nc_xs2;
typedef iq2_xxs_t<false, 1> nc_xxs1; typedef iq2_xxs_t<false, 2> nc_xxs2;
// one NC launch for up to nc columns; returns the number of columns done (0: run column 0 with launch())
static inline int launch_nc_chunk(ggml_type t, const soa & s, const block_q8_1 * y, int ys, float * dst, int ldd, int K, int N, int nc,
        hipStream_t st, const float * residual, int ldr) {
    const launch_cfg c = pick_cfg(t, N, K);   // L and P fix the fp32 accumulation order: keep them
    const int grid = nc_grid_override() > 0 ? nc_grid_override() : c.grid;
    static const int rr = nc_env("GGML_MMVQ_NC_LB_R", 1);   // tuning: rows per lane group
#define LB_NC(T, L_) if (c.L == L_) return rr == 2 ? launch_nc_t<T, L_, 2, 16>(s, y, ys, dst, ldd, K, N, nc, grid, st, residual, ldr) : launch_nc_t<T, L_, 1, 16>(s, y, ys, dst, ldd, K, N, nc, grid, st, residual, ldr);
    switch (t) {
        case GGML_TYPE_IQ2_S:
            if (c.P == 1) { LB_NC(nc_s1, 64) LB_NC(nc_s1, 32) }
            else          { LB_NC(nc_s2, 64) LB_NC(nc_s2, 16) }
            break;
        case GGML_TYPE_IQ2_XS:
            if (c.P == 1) { LB_NC(nc_xs1, 64) LB_NC(nc_xs1, 32) }
            else          { LB_NC(nc_xs2, 16) }
            break;
        case GGML_TYPE_IQ2_XXS:
            if (c.P == 1) { LB_NC(nc_xxs1, 64) LB_NC(nc_xxs1, 32) }
            else          { LB_NC(nc_xxs2, 16) }
            break;
        case GGML_TYPE_Q2_K:
            LB_NC(q2_K, 16)
            break;
        case GGML_TYPE_IQ1_M:
            LB_NC(iq1_m_pk, 16)
            break;
        default: break;
    }
#undef LB_NC
    return 0;
}

// ------------------------------------------------------------------------------------------------ MTP2-C: MFMA N-column
// N-column GEMV (NC = 2..16) with the integer dot products on the matrix cores (v_mfma_i32_4x4x4i8: 16 blocks of 4 lanes).
// Lane = 4*b + q. Block b is an item lane slot: item lane l = b + 16*h (h = 0..L/16-1, L = the N=1 config's lanes per row).
//   B operand (lane 4*b + q): row q of the 4-row unit, decoded at item lane l exactly as the N=1 lane l of that row does it
//     (the sign-applied int8 weights, 4 per dword);
//   A operand (lane 4*b + i): column 4*cg + i of the y records (q8 dwords).
//   D[i] of lane 4*b + q = the exact int32 sum of row q, column 4*cg + i over the chain.
// So every per-row value (dw, sub-block scales) stays in its own lane, and the 4 D values are 4 columns of that row.
// Integer stage: the chains cover the N=1 dp4a chain halves (dwords 0..3 / 4..7 of a 32-weight group), int32 sums are
// exact in any order. Exact integer identities (all integers involved are < 2^24 in magnitude, so every fp32 step on
// them below is exact and gives the N=1 integers bit for bit):
//   IQ2_S / IQ2_XS: chain s0 (dwords 0..3) continued to s01 (0..7); sdiv<2>(s0*sa + s1*sb + sdiv<1>(s0 + s1)) =
//     trunc(0.25*fma(s0, sa - sb, fma(s01, sb, trunc(0.5*s01))))
//   IQ2_XXS: sdiv<3>(s01*ls) = trunc(s01*(ls/8)) (s01*ls < 2^24 and ls/8 exact: the product is exact)
//   Q2_K: weights prescaled by the 4-bit scale (bytes sc*((v >> 2g) & 3) <= 45, one chain of 8) give SDn = SD/4^g',
//     fma(d8*inv, SD, sd) = fma(d8, SDn, sd) (d8*inv = d8/4^g' exactly, so the same exact product);
//     SM = fma(ma, Y01, mb*Y23) with the float y half sums (exact)
//   IQ1_M: bytes 8*g + 9 - 2*delta (<= 25), chains started at -16*sum(y) give S0 / S1 directly
//     (= 8*i01 + y9 + b0*y11 + b1*y12 of the N=1 code); mad24(S0, sc0, mul24(S1, sc1)) = fma(S0, sc0, S1*sc1)
// Float stage per (row, column, item lane l): the N=1 fp32 sequence (contraction off, fma only where N=1 has one; packed
// fp32 ops round per element like the scalar ones). Lane l accumulates items l, l+L, ... in increasing order (K chunks
// only pause it; steps past the row end are computed on a clamped item and dropped by a select) and group_sum<L> is the
// same butterfly (item lane l ^ 8, ^ 4, ^ 2, ^ 1 = lane ^ 32, ^ 16, ^ 8, ^ 4, then the cross-wave ^ 16, ^ 32 of L > 16;
// identical to the row_ror form at lane 0 because fp add is commutative). So every output is bit-identical to the N=1
// kernel of that column. The first item term t = x0 instead of 0 + x0 differs only in the sign of a zero, which the
// accumulator (starting at +0, never -0) absorbs.
// y records of all columns are staged per K chunk (KCB 256-blocks, aligned to L items): q8 dwords per column (record
// stride RS words) + d8 (and the Q2_K / IQ1_M y sums) as [record][column] arrays read 4 columns at a time.
// Switch: GGML_MMVQ_NC_LOWBIT_MFMA=0 -> the dp4a NC kernels above exactly (and N <= LB_NC_MAX). Tuning / debug:
// GGML_MMVQ_NC_LB_MF_MIN (smallest NC taken), GGML_MMVQ_NC_LB_MF_LDS (LDS cap per workgroup), GGML_MMVQ_NC_LB_MF_GRID,
// GGML_MMVQ_NC_LB_MF_NW (waves per workgroup).
#define LB_NC_MAX_MF 16
#ifndef LB_MF_SCHED_STEP
#define LB_MF_SCHED_STEP 1
#endif

typedef int lb_i32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int lb_u32x4 __attribute__((ext_vector_type(4)));
static __device__ __forceinline__ lb_i32x4 mf4(const uint32_t a, const uint32_t b, const lb_i32x4 c) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_amdgcn_mfma_i32_4x4x4i8((int) a, (int) b, c, 0, 0, 0);
#else
    GGML_UNUSED_VARS(a, b);
    return c;
#endif
}
static __device__ __forceinline__ float mf_add(const float a, const float b) {
#pragma clang fp contract(off)
    return a + b;
}
// packed fp32 (v_pk_mul / v_pk_add / v_pk_fma round per element exactly like the scalar ops; never contracted)
typedef float lb_f2 __attribute__((ext_vector_type(2)));
static __device__ __forceinline__ lb_f2 f2_mul(const lb_f2 a, const lb_f2 b) {
#pragma clang fp contract(off)
    return a*b;
}
static __device__ __forceinline__ lb_f2 f2_add(const lb_f2 a, const lb_f2 b) {
#pragma clang fp contract(off)
    return a + b;
}
static __device__ __forceinline__ lb_f2 f2_sub(const lb_f2 a, const lb_f2 b) {
#pragma clang fp contract(off)
    return a - b;
}
static __device__ __forceinline__ lb_f2 f2_fma(const lb_f2 a, const lb_f2 b, const lb_f2 c) { return __builtin_elementwise_fma(a, b, c); }
static __device__ __forceinline__ lb_f2 f2_trunc(const lb_f2 a) { return __builtin_elementwise_trunc(a); }
static __device__ __forceinline__ lb_f2 f2_s(const float a) { lb_f2 r; r.x = a; r.y = a; return r; }
static __device__ __forceinline__ lb_f2 f2_cvt(const int a, const int b) { lb_f2 r; r.x = (float) a; r.y = (float) b; return r; }
static __device__ __forceinline__ lb_f2 f2_u(const uint32_t a, const uint32_t b) { lb_f2 r; r.x = __uint_as_float(a); r.y = __uint_as_float(b); return r; }
// value of lane (lane ^ M) within 32 lanes (ds_swizzle bit mode)
template <int M> static __device__ __forceinline__ float mf_swz(const float v) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (M << 10) | 0x1F));
#else
    return v;
#endif
}
// value of lane (lane + R) mod 16 within its row of 16 (DPP row_ror)
template <int R> static __device__ __forceinline__ float mf_ror(const float v) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x120 | R, 0xF, 0xF, false));
#else
    return v;
#endif
}
// 16-byte LDS read at a byte address (tables at LDS address 0, see LB_TAB0)
static __device__ __forceinline__ lb_u32x4 mf_lds4(const uint32_t a) {
#if defined(__HIP_DEVICE_COMPILE__)
    return *(__attribute__((address_space(3))) const lb_u32x4 *) (size_t) a;
#else
    GGML_UNUSED(a);
    return lb_u32x4{0, 0, 0, 0};
#endif
}

// per type: y records per item (NG), q8 record stride RS (words, multiple of 4) in the column arrays, NX extra
// [record][column] arrays (Q2_K: float y sums of qs[0..15], qs[16..31]; IQ1_M: int -16*the same sums)
template <class T> struct mf_k {
    static constexpr int kind = nc_traits<T>::kind;
    static constexpr int NG = T::SPI;
    static constexpr int RS = kind == 3 ? 12 : 8;
    static constexpr int NX = kind >= 3 ? 2 : 0;
};

// sign-applied grid words of one 32-weight IQ2 group (the dp4a operands of dot32<false>, same order)
static __device__ __forceinline__ void mf_iq2_words(const gent & a, const gent & b, const gent & c, const gent & d, const uint32_t S,
        uint32_t (&w)[8]) {
    const uint32_t M4 = vconst(0x04040404u), SB = sconst(0x03020100u);
    w[0] = perm(a.nx, a.x, ysel<0>(S, M4, SB)); w[1] = perm(a.ny, a.y, ysel<4>(S, M4, SB));
    w[2] = perm(b.nx, b.x, ysel<1>(S, M4, SB)); w[3] = perm(b.ny, b.y, ysel<5>(S, M4, SB));
    w[4] = perm(c.nx, c.x, ysel<2>(S, M4, SB)); w[5] = perm(c.ny, c.y, ysel<6>(S, M4, SB));
    w[6] = perm(d.nx, d.x, ysel<3>(S, M4, SB)); w[7] = perm(d.ny, d.y, ysel<7>(S, M4, SB));
}

// B words of group g of the item of this lane's row
template <class T>
static __device__ __forceinline__ void mf_group(const uint32_t * __restrict__ tab, const typename T::raw & r, const int g, uint32_t (&w)[8]) {
    constexpr int kind = mf_k<T>::kind;
    if constexpr (kind == 0) {
        const uint32_t q = r.a.w[2*g], S = r.a.w[2*g + 1];
        const uint32_t qhsc = r.b >> (16*g);
        const uint32_t qhx = tab_u32(tab, T::QHX, qhsc & 0xFF);
        const uint32_t X = perm(qhx, q, 0x05010400u), Z = perm(qhx, q, 0x07030602u);
        gent g0, g1, g2, g3;
        tab_pairG<false>(tab, X, g0, g1); tab_pairG<false>(tab, Z, g2, g3);
        mf_iq2_words(g0, g1, g2, g3, S, w);
    } else if constexpr (kind == 1) {
        const uint32_t q = r.a.w[2*g], S0 = r.a.w[2*g + 1];
        const uint32_t qh4 = spread4(S0 >> 28);
        const uint32_t X = perm(qh4, q, 0x05010400u), Z = perm(qh4, q, 0x07030602u);
        gent g0, g1, g2, g3;
        tab_pairG<false>(tab, X, g0, g1); tab_pairG<false>(tab, Z, g2, g3);
        mf_iq2_words(g0, g1, g2, g3, ksign_fix(S0), w);
    } else if constexpr (kind == 2) {
        const uint32_t q = r.a.w[2*g], S0 = r.a.w[2*g + 1];
        const gent g0 = tab_g1<false>(tab, q & 0xFF),         g1 = tab_g1<false>(tab, (q >> 8) & 0xFF);
        const gent g2 = tab_g1<false>(tab, (q >> 16) & 0xFF), g3 = tab_g1<false>(tab, q >> 24);
        mf_iq2_words(g0, g1, g2, g3, ksign_fix(S0), w);
    } else if constexpr (kind == 3) {   // prescaled: sc*((v >> 2g) & 3) via a v_perm table {0, sc, 2sc, 3sc}
        const uint32_t v[8] = {r.q0.x, r.q0.y, r.q0.z, r.q0.w, r.q1.x, r.q1.y, r.q1.z, r.q1.w};
        const uint32_t scb = (g < 2 ? r.sc.x : r.sc.y) >> (16*(g & 1));
        const uint32_t Ta = (uint32_t) __mul24((int) (scb & 0xF), 0x030201) << 8, Tb = (uint32_t) __mul24((int) ((scb >> 8) & 0xF), 0x030201) << 8;
        const int sh = g < 3 ? 2*g : 6;
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = perm(0u, j < 4 ? Ta : Tb, (v[j] >> sh) & 0x03030303u);
    } else {
        const uint32_t Q = r.a[g];
        const uint32_t qh = (r.b >> (16*g)) & 0xFFFF;
        const uint32_t qhb = (qh & 0x0707u) | ((qh >> 4) & 0x0707u) << 16;
        const uint32_t X = perm(qhb, Q, 0x06010400u), Z = perm(qhb, Q, 0x07030502u);
        const uint32_t w0 = tab_u32(tab, 0, X & 0xFFFF), w1 = tab_u32(tab, 0, X >> 16);
        const uint32_t w2 = tab_u32(tab, 0, Z & 0xFFFF), w3 = tab_u32(tab, 0, Z >> 16);
        // per 8 weights: + 9 - 2*delta, i.e. 0x07 | (delta ? 0 : 2) per byte
        const uint32_t k0 = (qh & 0x0008u) ? 0u : 0x02020202u, k1 = (qh & 0x0080u) ? 0u : 0x02020202u;
        const uint32_t k2 = (qh & 0x0800u) ? 0u : 0x02020202u, k3 = (qh & 0x8000u) ? 0u : 0x02020202u;
        w[0] = (((w0 << 3) & 0x78787878u) | 0x07070707u) + k0; w[1] = (((w0 >> 1) & 0x78787878u) | 0x07070707u) + k0;
        w[2] = (((w1 << 3) & 0x78787878u) | 0x07070707u) + k1; w[3] = (((w1 >> 1) & 0x78787878u) | 0x07070707u) + k1;
        w[4] = (((w2 << 3) & 0x78787878u) | 0x07070707u) + k2; w[5] = (((w2 >> 1) & 0x78787878u) | 0x07070707u) + k2;
        w[6] = (((w3 << 3) & 0x78787878u) | 0x07070707u) + k3; w[7] = (((w3 >> 1) & 0x78787878u) | 0x07070707u) + k3;
    }
}

// word offset of column c in the q8 column arrays (sw: bank swizzle of column pairs)
static __host__ __device__ __forceinline__ int mf_cb(const int c, const int colw, const int sw) { return c*colw + ((c >> 1) & 1)*sw; }

// stage nrec records of ncp columns (columns >= nc: copies of column nc-1)
template <class T>
static __device__ __forceinline__ void mf_stage(uint32_t * __restrict__ yA, float * __restrict__ d8s, float * __restrict__ x0s, float * __restrict__ x1s,
        const block_q8_1 * __restrict__ yg, const int ys, const int nc, const int ncp, const int colw, const int sw, const int nrec, const int tid, const int nt) {
    typedef mf_k<T> MK;
    for (int t = tid; t < ncp*nrec; t += nt) {
        const int c = t / nrec, k = t - c*nrec;
        const int cs = c < nc ? c : nc - 1;
        const uint8_t * bp = (const uint8_t *) (yg + (size_t) cs*ys + k);
        const uint4 q0 = *(const uint4 *) (bp + 4), q1 = *(const uint4 *) (bp + 20);
        const uint32_t ds = *(const uint32_t *) bp;
        {
            uint4 * r = (uint4 *) (yA + mf_cb(c, colw, sw) + k*MK::RS);
            r[0] = q0; r[1] = q1;
        }
        d8s[k*ncp + c] = __half2float(__ushort_as_half((unsigned short) (ds & 0xFFFF)));
        if constexpr (MK::NX) {
            const int q[8] = {(int) q0.x, (int) q0.y, (int) q0.z, (int) q0.w, (int) q1.x, (int) q1.y, (int) q1.z, (int) q1.w};
            int s4[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) s4[j] = dp4a(q[2*j+1], 0x01010101, dp4a(q[2*j], 0x01010101, 0));
            if constexpr (MK::kind == 3) {
                x0s[k*ncp + c] = (float) (s4[0] + s4[1]);
                x1s[k*ncp + c] = (float) (s4[2] + s4[3]);
            } else {
                x0s[k*ncp + c] = __int_as_float(-16*(s4[0] + s4[1]));
                x1s[k*ncp + c] = __int_as_float(-16*(s4[2] + s4[3]));
            }
        }
    }
}

// one item lane slot of one step: MFMA integer sums + the N=1 fp32 item value of this lane's row for its CG*4 columns,
// added to acc (only when valid). rec = chunk record index of the item's first group; byte addresses: colb (column
// arrays), d8b / x0b / x1b ([record][column] arrays)
template <class T, int CG>
static __device__ __forceinline__ void mf_item(const uint32_t * __restrict__ tab, const typename T::raw & r, const uint32_t ii,
        const uint32_t rec, const uint32_t colb, const uint32_t cgs, const uint32_t d8b, const uint32_t x0b, const uint32_t x1b,
        const bool valid, float (&acc)[CG][4]) {
#pragma clang fp contract(off)
    typedef mf_k<T> MK;
    constexpr int kind = MK::kind, NG = MK::NG, RS = MK::RS, NCP = 4*CG;
    float dw = 0.0f, dmy = 0.0f;
    if constexpr (kind <= 2) dw = h2f(r.d);
    if constexpr (kind == 3) { const float2 dm = __half22float2(*(const half2 *) &r.dm); dw = dm.x; dmy = dm.y; }
    if constexpr (kind == 4) {
        const uint32_t u16 = ((r.sc.x & 0xFFFF) >> 12) | ((r.sc.x >> 24) & 0x00F0) | (((r.sc.y & 0xFFFF) >> 4) & 0x0F00) | ((r.sc.y >> 16) & 0xF000);
        dw = h2f(u16);
    }
    uint32_t scw16 = 0;
    int ib0 = 0;
    if constexpr (kind == 4) {
        ib0 = T::SPI*(int) ii;
        scw16 = ((ib0 < 4 ? r.sc.x : r.sc.y) >> (16*((ib0 >> 1) & 1))) & 0xFFFF;
    }
    GGML_UNUSED(ii);
    // per-row B words and constants of every group first (shared by all columns)
    uint32_t w[NG][8];
    float c0[NG], c1[NG];
#pragma unroll
    for (int g = 0; g < NG; ++g) {
        mf_group<T>(tab, r, g, w[g]);
        c0[g] = 0.0f; c1[g] = 0.0f;
        if constexpr (kind == 0 || kind == 1) {
            const uint32_t sc = kind == 0 ? (r.b >> (16*g + 8)) & 0xFF : (r.b >> (8*g)) & 0xFF;
            c0[g] = (float) ((int) (sc & 0xF) - (int) (sc >> 4)); c1[g] = (float) (int) (sc >> 4);
        } else if constexpr (kind == 2) {
            c0[g] = (float) (int) ((r.a.w[2*g + 1] >> 27) | 1u) * 0.125f;
        } else if constexpr (kind == 3) {
            const uint32_t scb = (g < 2 ? r.sc.x : r.sc.y) >> (16*(g & 1));
            c0[g] = (float) (int) ((scb >> 4) & 0xF); c1[g] = (float) (int) ((scb >> 12) & 0xF);
        } else {
            const uint32_t tmp = scw16 >> (6*((ib0 + g) & 1));
            c0[g] = (float) (2*(int) (tmp & 7) + 1); c1[g] = (float) (2*(int) ((tmp >> 3) & 7) + 1);
        }
    }
    const lb_f2 dw2 = f2_s(dw), dmi = f2_s(dmy);
    // (column group cg, group g) steps n = cg*NG + g, software pipelined: LDS operands of step n+2 and the MFMA chains
    // of step n+1 are issued before the fp32 stage of step n
    constexpr int NI = CG*NG;
    struct ldo { lb_u32x4 A0, A1, D8, X0, X1; };
    struct cro { lb_i32x4 s0, s1; lb_u32x4 D8, X0; lb_u32x4 X1; };
    auto ld = [&](const int n) {
        const int cg = n / NG, g = n % NG;
        const uint32_t rg = rec + (uint32_t) g;
        const uint32_t ya = colb + (uint32_t) cg*cgs + rg*(RS*4);
        ldo o;
        o.A0 = mf_lds4(ya); o.A1 = mf_lds4(ya + 16);
        o.D8 = mf_lds4(d8b + (rg*NCP + 4*cg)*4);
        if constexpr (MK::NX) { o.X0 = mf_lds4(x0b + (rg*NCP + 4*cg)*4); o.X1 = mf_lds4(x1b + (rg*NCP + 4*cg)*4); }
        else { o.X0 = lb_u32x4{0, 0, 0, 0}; o.X1 = o.X0; }
        return o;
    };
    auto chain = [&](const ldo & o, const int n) {
        const int g = n % NG;
        cro c;
        c.D8 = o.D8; c.X0 = o.X0; c.X1 = o.X1;
        lb_i32x4 s0 = {0, 0, 0, 0}, s1 = {0, 0, 0, 0};
        if constexpr (kind == 0 || kind == 1) {   // s0 = dwords 0..3, s1 = s01 = dwords 0..7
            s0 = mf4(o.A0.x, w[g][0], s0); s0 = mf4(o.A0.y, w[g][1], s0); s0 = mf4(o.A0.z, w[g][2], s0); s0 = mf4(o.A0.w, w[g][3], s0);
            s1 = mf4(o.A1.x, w[g][4], s0); s1 = mf4(o.A1.y, w[g][5], s1); s1 = mf4(o.A1.z, w[g][6], s1); s1 = mf4(o.A1.w, w[g][7], s1);
        } else if constexpr (kind == 2 || kind == 3) {   // one chain of 8
            s1 = mf4(o.A0.x, w[g][0], s1); s1 = mf4(o.A0.y, w[g][1], s1); s1 = mf4(o.A0.z, w[g][2], s1); s1 = mf4(o.A0.w, w[g][3], s1);
            s1 = mf4(o.A1.x, w[g][4], s1); s1 = mf4(o.A1.y, w[g][5], s1); s1 = mf4(o.A1.z, w[g][6], s1); s1 = mf4(o.A1.w, w[g][7], s1);
        } else {   // two chains of 4 started at -16*sum(y)
            s0 = lb_i32x4{(int) o.X0.x, (int) o.X0.y, (int) o.X0.z, (int) o.X0.w};
            s1 = lb_i32x4{(int) o.X1.x, (int) o.X1.y, (int) o.X1.z, (int) o.X1.w};
            s0 = mf4(o.A0.x, w[g][0], s0); s1 = mf4(o.A1.x, w[g][4], s1);
            s0 = mf4(o.A0.y, w[g][1], s0); s1 = mf4(o.A1.y, w[g][5], s1);
            s0 = mf4(o.A0.z, w[g][2], s0); s1 = mf4(o.A1.z, w[g][6], s1);
            s0 = mf4(o.A0.w, w[g][3], s0); s1 = mf4(o.A1.w, w[g][7], s1);
        }
        c.s0 = s0; c.s1 = s1;
        return c;
    };
    lb_f2 t0[2], t1[2];   // IQ2 / IQ1_M: item sum; Q2_K: sd, sm (columns 2j, 2j+1) of the current column group
    auto epi = [&](const cro & c, const int n) {
        const int cg = n / NG, g = n % NG;
        const lb_f2 c02 = f2_s(c0[g]), c12 = f2_s(c1[g]);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int i0 = 2*j, i1 = 2*j + 1;
            const lb_f2 d82 = f2_u(c.D8[i0], c.D8[i1]);
            if constexpr (kind == 3) {
                const lb_f2 sd = f2_cvt(c.s1[i0], c.s1[i1]);
                const lb_f2 sm = f2_fma(c02, f2_u(c.X0[i0], c.X0[i1]), f2_mul(c12, f2_u(c.X1[i0], c.X1[i1])));
                t0[j] = f2_fma(d82, sd, g == 0 ? f2_s(0.0f) : t0[j]);
                t1[j] = f2_fma(d82, sm, g == 0 ? f2_s(0.0f) : t1[j]);
            } else {
                lb_f2 si;
                if constexpr (kind == 0 || kind == 1) {
                    const lb_f2 a0 = f2_cvt(c.s0[i0], c.s0[i1]), a01 = f2_cvt(c.s1[i0], c.s1[i1]);
                    const lb_f2 hh = f2_trunc(f2_mul(a01, f2_s(0.5f)));
                    si = f2_trunc(f2_mul(f2_fma(a0, c02, f2_fma(a01, c12, hh)), f2_s(0.25f)));
                } else if constexpr (kind == 2) {
                    si = f2_trunc(f2_mul(f2_cvt(c.s1[i0], c.s1[i1]), c02));
                } else {
                    si = f2_mul(f2_fma(f2_cvt(c.s0[i0], c.s0[i1]), c02, f2_mul(f2_cvt(c.s1[i0], c.s1[i1]), c12)), f2_s(0.125f));
                }
                const lb_f2 x = f2_mul(f2_mul(dw2, d82), si);
                t0[j] = g == 0 ? x : f2_add(t0[j], x);
            }
        }
        if (g == NG - 1) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const lb_f2 item = kind == 3 ? f2_sub(f2_mul(dw2, t0[j]), f2_mul(dmi, t1[j])) : t0[j];
                const float n0 = mf_add(acc[cg][2*j], item.x), n1 = mf_add(acc[cg][2*j + 1], item.y);
                acc[cg][2*j]     = valid ? n0 : acc[cg][2*j];
                acc[cg][2*j + 1] = valid ? n1 : acc[cg][2*j + 1];
            }
        }
    };
    if constexpr (kind == 3) {   // Q2_K (4 groups of B words live): no look-ahead, it would exceed the register budget
#pragma unroll
        for (int n = 0; n < NI; ++n) {
            epi(chain(ld(n), n), n);
#if defined(__HIP_DEVICE_COMPILE__)
            if constexpr (LB_MF_SCHED_STEP) __builtin_amdgcn_sched_barrier(0);
#endif
        }
    } else {
        ldo l1 = ld(0), l2;
        cro c1c = chain(l1, 0), c2c;
        if constexpr (NI > 1) l1 = ld(1);
#pragma unroll
        for (int n = 0; n < NI; ++n) {
            if (n + 2 < NI) l2 = ld(n + 2);
            if (n + 1 < NI) c2c = chain(l1, n + 1);
            epi(c1c, n);
#if defined(__HIP_DEVICE_COMPILE__)
            if constexpr (LB_MF_SCHED_STEP) __builtin_amdgcn_sched_barrier(0);   // keeps the pipeline order (bounds live ranges)
#endif
            c1c = c2c; l1 = l2;
        }
    }
}

// L = 16*H lanes per row in the N=1 config: H waves form a wave group that shares a unit; wave h of the group takes
// the item lanes l = b + 16*h (items l, l+L, ...). The group sum: each wave runs the l ^ 8, ^ 4, ^ 2, ^ 1 butterfly on
// its 16 item lanes, then wave 0 adds the H partial sums as the N=1 shuffles do (^16: v0 + v1; ^32: (v0 + v1) + (v2 + v3)),
// exchanged through LDS.
template <class T, int H, int CG, int UM, int NW>
__launch_bounds__(NW*64, 1)
__global__ void gemv_mf(const soa s, const block_q8_1 * __restrict__ yg, const int ys, const int nc, float * __restrict__ dst, const int ldd,
        const int K, const int N, const float * __restrict__ residual, const int ldr, const int KCB, const int colw, const int sw) {
#pragma clang fp contract(off)
    extern __shared__ uint32_t smem[];   // [tables (at LDS address 0) | q8 column arrays | d8 [rec][col] | y sums | group sums]
    typedef mf_k<T> MK;
    constexpr int L = 16*H, NWG = NW/H, NCP = 4*CG;
    static_assert(NW % H == 0, "wave groups");
    const int nrec = KCB*8;
    uint32_t * tab = smem;
    uint32_t * yA  = smem + T::TAB_WORDS;
    float * d8s = (float *) (yA + NCP*colw + sw);
    float * x0s = d8s + nrec*NCP;
    float * x1s = x0s + (MK::NX ? nrec*NCP : 0);
    float * red = x1s + (MK::NX ? nrec*NCP : 0);
#if LB_TAB0 && defined(__HIP_DEVICE_COMPILE__)
    if ((uint32_t) (size_t) (__attribute__((address_space(3))) uint32_t *) smem != 0) __builtin_trap();
#endif
    const int tid = threadIdx.y*64 + threadIdx.x;
    const int b = threadIdx.x >> 2, q = threadIdx.x & 3;
    const int h = threadIdx.y % H, grp = threadIdx.y / H;
    const int bpr = K / QK_K;
    const int U = N/4;                        // units of 4 rows
    const int ustep = gridDim.x*NWG;
    const int nchunks = (bpr + KCB - 1)/KCB;
    const int u0 = grp*gridDim.x + blockIdx.x;
    // LDS byte addresses
    const uint32_t yAb = (uint32_t) T::TAB_WORDS*4;
    const uint32_t d8b = yAb + (uint32_t) (NCP*colw + sw)*4;
    const uint32_t x0b = d8b + (uint32_t) (nrec*NCP)*4;
    const uint32_t x1b = x0b + (uint32_t) (MK::NX ? nrec*NCP : 0)*4;
    const uint32_t colb = yAb + (uint32_t) mf_cb(q, colw, sw)*4;   // column 4*cg + q of the A operand: colb + cg*cgs
    const uint32_t cgs  = (uint32_t) (4*colw)*4;
    T::stage_tables(tab, tid, NW*64);
    bool staged = false;
    // batches of UM units per wave group (one batch unless U > UM*ustep); the batch and chunk loops are uniform over the
    // workgroup (barriers inside), a wave without a unit in the batch only stages
    for (int bt = 0; (int) blockIdx.x + bt*UM*ustep < U; ++bt) {
        const int ub = u0 + bt*UM*ustep;
        float acc[UM][CG][4];
#pragma unroll
        for (int k = 0; k < UM; ++k)
#pragma unroll
            for (int cg = 0; cg < CG; ++cg)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[k][cg][i] = 0.0f;
        for (int kb0 = 0; kb0 < bpr; kb0 += KCB) {
            const int nkb = min(KCB, bpr - kb0);
            if (nchunks > 1 || !staged) {
                __syncthreads();
                mf_stage<T>(yA, d8s, x0s, x1s, yg + (size_t) kb0*(QK_K/QK8_1), ys, nc, NCP, colw, sw, nkb*8, tid, NW*64);
                __syncthreads();
                staged = true;
            }
            const uint32_t it_beg = (uint32_t) kb0*T::IPB, it_end = (uint32_t) (kb0 + nkb)*T::IPB;
            const uint32_t st0 = it_beg - it_beg % L;   // steps cover [st0, it_end); items outside [it_beg, it_end) are dropped
            const uint32_t l0 = (uint32_t) (b + 16*h);
#pragma unroll
            for (int k = 0; k < UM; ++k) {
                const int u = ub + k*ustep;
                if (u >= U) break;   // uniform per wave
                const uint32_t rowb = (uint32_t) (4*u + q)*(uint32_t) bpr;
                uint32_t it = st0 + l0, itc = it < it_beg ? it_beg : it < it_end ? it : it_end - 1;
                typename T::raw cur = T::load(s, rowb + itc/T::IPB, itc % T::IPB);
                for (uint32_t st = st0; st < it_end; st += L) {
                    const uint32_t itn = st + L + l0 < it_end ? st + L + l0 : it_end - 1;   // next step (clamped, unconditional)
                    const typename T::raw nxt = T::load(s, rowb + itn/T::IPB, itn % T::IPB);
                    it = st + l0;
                    const bool valid = it >= it_beg && it < it_end;
                    itc = it < it_beg ? it_beg : valid ? it : it_end - 1;
                    mf_item<T, CG>(tab, cur, itc % T::IPB, (itc - it_beg)*T::SPI, colb, cgs, d8b, x0b, x1b, valid, acc[k]);
                    cur = nxt;
                }
            }
        }
#pragma unroll
        for (int k = 0; k < UM; ++k) {
            const int u = ub + k*ustep;
            float v[CG][4];
#pragma unroll
            for (int cg = 0; cg < CG; ++cg)
#pragma unroll
                for (int i = 0; i < 4; ++i) {   // group_sum<16> over the item lanes b (lane bits 2..5)
                    float x = acc[k][cg][i];
                    x += __shfl_xor(x, 32, 64);
                    x += mf_swz<16>(x);
                    // ^8, ^4 as row rotations: equal to the xor partners at every lane the stored lanes 0..3 depend on
                    x += mf_ror<8>(x);
                    x += mf_ror<4>(x);
                    v[cg][i] = x;
                }
            if constexpr (H > 1) {
                __syncthreads();   // (uniform: every wave runs all UM slots) previous slot's sums are read
                if (b == 0) {
#pragma unroll
                    for (int cg = 0; cg < CG; ++cg)
#pragma unroll
                        for (int i = 0; i < 4; ++i) red[((grp*H + h)*NCP + cg*4 + i)*4 + q] = v[cg][i];
                }
                __syncthreads();
                if (h == 0 && b == 0) {
#pragma unroll
                    for (int cg = 0; cg < CG; ++cg)
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const float * rp = red + ((grp*H)*NCP + cg*4 + i)*4 + q;
                            if constexpr (H == 2) v[cg][i] = v[cg][i] + rp[NCP*4];                                   // ^16
                            else v[cg][i] = (v[cg][i] + rp[NCP*4]) + (rp[2*NCP*4] + rp[3*NCP*4]);                    // ^16, ^32
                        }
                }
            }
            if (u >= U || h != 0) continue;
#pragma unroll
            for (int cg = 0; cg < CG; ++cg)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int c = 4*cg + i;
                    if (b == 0 && c < nc) {
                        const int row = 4*u + q;
                        dst[(uint32_t) (c*ldd + row)] = residual ? v[cg][i] + residual[(uint32_t) (c*ldr + row)] : v[cg][i];
                    }
                }
        }
    }
}

static inline int mf_env_mfma() {
    static const int v = nc_env("GGML_MMVQ_NC_LOWBIT_MFMA", 1);
    return v;
}
static inline int mf_min_nc() {
    static const int v = std::max(2, nc_env("GGML_MMVQ_NC_LB_MF_MIN", 3));
    return v;
}
static inline int mf_gcd(int a, int b) { while (b) { const int t = a % b; a = b; b = t; } return a; }

// column layout of the q8 column arrays: stride colw >= need (multiple of 4 words) and column-pair swizzle sw chosen so
// that the 16-byte A reads (8 lanes per LDS cycle: blocks 2k, 2k+1 x columns 0..3) spread over the 32 banks
static inline void mf_layout(int need, int spi_rs, int CG, int & colw, int & sw) {
    int best = 1 << 30; colw = need; sw = 0;
    for (int pad = 0; pad < 32; pad += 4) {
        for (int s = 0; s <= pad; s += 4) {   // s <= pad: shifted columns never overlap the next one
            const int cw = need + pad;
            int cost = 0;
            for (int cg = 0; cg < CG; ++cg) {
                for (int g8 = 0; g8 < 8; ++g8) {   // 8 groups of 8 lanes
                    int cnt[32] = {0}, m = 0;
                    for (int ln = 8*g8; ln < 8*g8 + 8; ++ln) {
                        const int bb = ln >> 2, i = ln & 3;
                        const int c = 4*cg + i;
                        const int a = mf_cb(c, cw, s) + bb*spi_rs;
                        for (int e = 0; e < 4; ++e) m = std::max(m, ++cnt[(a + e) & 31]);
                    }
                    cost += m;
                }
            }
            if (cost < best) { best = cost; colw = cw; sw = s; }
        }
    }
}

template <class T, int H, int CG, int NW>
static inline bool launch_mf_t(const soa & s, const block_q8_1 * y, int ys, float * dst, int ldd, int K, int N, int nc,
        hipStream_t st, const float * residual, int ldr) {
    typedef mf_k<T> MK;
    constexpr int L = 16*H, NCP = 4*CG;
    const int bpr = K/QK_K;
    if (N % 4 != 0) return false;
    static const int lds_env = nc_env("GGML_MMVQ_NC_LB_MF_LDS", 0);
    static const int grid_o  = nc_env("GGML_MMVQ_NC_LB_MF_GRID", 0);
    const int tabb = T::TAB_WORDS*4;
    int align = L/mf_gcd(L, T::IPB);                      // KCB*IPB % L == 0 preferred (no partial steps at chunk ends)
    const size_t redb = H > 1 ? (size_t) NW*NCP*4*4 : 0;  // wave group sums
    // per 256-block of K: q8 dwords of nc columns + d8 (+ sums) of NCP columns (+ column padding, at most 60 words)
    const size_t per_blk = (size_t) 8*NCP*(MK::RS + 1 + MK::NX)*4;
    // default LDS cap: two workgroups per CU when the chunks stay large, else one
    const int lds_cap = lds_env > 0 ? std::min(lds_env, 65536) : 65536;
    auto lds_of = [&](int kcb) { return (size_t) tabb + (size_t) kcb*per_blk + (size_t) NCP*128 + 128 + redb; };
    int KCB = bpr;
    if (lds_of(KCB) > (size_t) lds_cap) {
        int kmax = (bpr/align)*align;
        while (kmax > 0 && lds_of(kmax) > (size_t) lds_cap) kmax -= align;
        if (kmax <= 0) {   // unaligned chunks
            align = 1; kmax = bpr;
            while (kmax > 0 && lds_of(kmax) > (size_t) lds_cap) --kmax;
            if (kmax <= 0) return false;
        }
        const int nch = (bpr + kmax - 1)/kmax;
        KCB = (((bpr + nch - 1)/nch + align - 1)/align)*align;
        if (KCB > kmax) KCB = kmax;
    }
    int colw, sw;
    mf_layout(KCB*8*MK::RS, MK::NG*MK::RS, CG, colw, sw);
    const size_t lds = (size_t) tabb + (size_t) (NCP*colw + sw)*4 + (size_t) KCB*8*NCP*(1 + MK::NX)*4 + redb;
    if (lds > 65536) return false;
    const int ncu = nc_num_cu();
    static int occ_lds = -1, occ_n = 1;   // resident workgroups per CU (registers, LDS), per instantiation
    if (occ_lds != (int) lds) {
        int n = 0;
        if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, gemv_mf<T, H, CG, 1, NW>, NW*64, lds) != hipSuccess || n < 1) { (void) hipGetLastError(); n = 1; }
        occ_lds = (int) lds; occ_n = n;
    }
    const int wpc = occ_n;
    int grid = grid_o > 0 ? grid_o : wpc*ncu;
    const int U = N/4;
    const int nch = (bpr + KCB - 1)/KCB;
    const int upw = (U + grid*(NW/H) - 1)/(grid*(NW/H));  // units per wave group
    // units kept across the K chunks per wave group: bounded by the register budget (spills beyond); more units per wave
    // than that run as batches (each restages the chunks)
    static const int um_env = nc_env("GGML_MMVQ_NC_LB_MF_UM", 0);
    constexpr int um_cap = (CG == 1 && MK::kind <= 2) ? 4 : 3;   // UM = 4 is instantiated only there (LB_MF(4) below)
    const int umax = um_env > 0 ? std::min(um_env, um_cap) : MK::kind == 4 ? (CG >= 3 ? 1 : CG == 2 ? 2 : 3) : MK::kind == 3 ? (CG == 1 ? 2 : 3) : CG >= 3 ? 2 : 3;
    // MTP3-BCD: one more unit per wave group when that saves a whole batch (the IQ2 ffn_down at 3..4 columns: 1280 units on
    // 416 wave groups; um = 3 left a second batch of 32 units that restaged every K chunk). GGML_MMVQ_NC_LB_MF_UM4=0: off
    static const int um4_env = nc_env("GGML_MMVQ_NC_LB_MF_UM4", 1);
    int um = nch > 1 ? std::min(upw, umax) : 1;
    if (nch > 1 && um_env == 0 && um4_env && um == 3 && upw == 4 && CG == 1 && MK::kind <= 2) um = 4;
    static const int dbg = nc_env("GGML_MMVQ_NC_LB_MF_DEBUG", 0);
    if (dbg) {
        fprintf(stderr, "[lb mf] type=%d L=%d CG=%d NW=%d K=%d N=%d nc=%d KCB=%d colw=%d sw=%d lds=%zu wpc=%d grid=%d um=%d nch=%d\n",
                (int) T::type, L, CG, NW, K, N, nc, KCB, colw, sw, lds, wpc, grid, um, nch);
    }
    const dim3 blk(64, NW);
#define LB_MF(UM_) if (um == UM_) { hipLaunchKernelGGL((gemv_mf<T, H, CG, UM_, NW>), dim3(grid), blk, lds, st, s, y, ys, nc, dst, ldd, K, N, residual, ldr, KCB, colw, sw); return true; }
    LB_MF(1) LB_MF(2) LB_MF(3)
    if constexpr (CG == 1 && MK::kind <= 2) { LB_MF(4) }
#undef LB_MF
    return false;
}

template <class T, int H>
static inline bool launch_mf_h(const soa & s, const block_q8_1 * y, int ys, float * dst, int ldd, int K, int N, int nc,
        hipStream_t st, const float * residual, int ldr) {
    static const int nwo = nc_env("GGML_MMVQ_NC_LB_MF_NW", 0);   // tuning: 8 or 16 waves per workgroup (0: by config)
    // 16 waves cap the kernel at 128 VGPRs: enough for every type but Q2_K beyond 4 columns
    const bool nw8 = nwo == 8 || (nwo != 16 && mf_k<T>::kind == 3 && nc > 4);
    switch ((nc + 3)/4) {
        case 1: return nw8 ? launch_mf_t<T, H, 1, 8>(s, y, ys, dst, ldd, K, N, nc, st, residual, ldr) : launch_mf_t<T, H, 1, 16>(s, y, ys, dst, ldd, K, N, nc, st, residual, ldr);
        case 2: return nw8 ? launch_mf_t<T, H, 2, 8>(s, y, ys, dst, ldd, K, N, nc, st, residual, ldr) : launch_mf_t<T, H, 2, 16>(s, y, ys, dst, ldd, K, N, nc, st, residual, ldr);
        case 3: return nw8 ? launch_mf_t<T, H, 3, 8>(s, y, ys, dst, ldd, K, N, nc, st, residual, ldr) : launch_mf_t<T, H, 3, 16>(s, y, ys, dst, ldd, K, N, nc, st, residual, ldr);
        case 4: return nw8 ? launch_mf_t<T, H, 4, 8>(s, y, ys, dst, ldd, K, N, nc, st, residual, ldr) : launch_mf_t<T, H, 4, 16>(s, y, ys, dst, ldd, K, N, nc, st, residual, ldr);
        default: return false;
    }
}

// all nc columns in one MFMA launch; false: not taken (nothing launched)
static inline bool launch_mf(ggml_type t, const soa & s, const block_q8_1 * y, int ys, float * dst, int ldd, int K, int N, int nc,
        hipStream_t st, const float * residual, int ldr) {
    if (nc < 2 || nc > LB_NC_MAX_MF) return false;
    const launch_cfg c = pick_cfg(t, N, K);   // L and P fix the fp32 accumulation order: keep them
#define LB_MFT(T, L_) if (c.L == L_) return launch_mf_h<T, L_/16>(s, y, ys, dst, ldd, K, N, nc, st, residual, ldr);
    switch (t) {
        case GGML_TYPE_IQ2_S:
            if (c.P == 1) { LB_MFT(nc_s1, 64) LB_MFT(nc_s1, 32) }
            else          { LB_MFT(nc_s2, 64) LB_MFT(nc_s2, 16) }
            break;
        case GGML_TYPE_IQ2_XS:
            if (c.P == 1) { LB_MFT(nc_xs1, 64) LB_MFT(nc_xs1, 32) }
            else          { LB_MFT(nc_xs2, 16) }
            break;
        case GGML_TYPE_IQ2_XXS:
            if (c.P == 1) { LB_MFT(nc_xxs1, 64) LB_MFT(nc_xxs1, 32) }
            else          { LB_MFT(nc_xxs2, 16) }
            break;
        case GGML_TYPE_Q2_K:
            LB_MFT(q2_K, 16)
            break;
        case GGML_TYPE_IQ1_M:
            LB_MFT(iq1_m_pk, 16)
            break;
        default: break;
    }
#undef LB_MFT
    return false;
}
// largest column count launch_nc takes (MFMA on: 16, else the dp4a kernels' 6)
static inline int nc_max_cols() {
    return mf_env_mfma() ? LB_NC_MAX_MF : LB_NC_MAX;
}
// ------------------------------------------------------------------------------------------------ MTP2-C end

// all nc columns (column passes when the y records of all columns do not fit in LDS; a single remaining column runs
// the N=1 launcher itself). Returns false only for unsupported types / shapes, all checked before anything is launched:
// launch_ok is exactly the condition under which launch() (the single-column fallback) succeeds, and a pass of 2+
// columns either launches or reports 0 columns done (then launch() runs that column), so nothing fails mid-way.
static inline bool launch_nc(ggml_type t, const void * soa_base, const block_q8_1 * y, int ys, float * dst, int ldd, int K, int N, int nc,
        hipStream_t st, const float * residual, int ldr) {
    if (K % 256 != 0 || N % 64 != 0 || nc < 2 || nc > nc_max_cols() || !supported(t) || !launch_ok(t, K, N)) return false;
    const launch_cfg c = pick_cfg(t, N, K);
    if (c.L == 0) return false;
    const soa s = soa_make(t, soa_base, (size_t) N*(K/QK_K));
    if (mf_env_mfma() && nc >= mf_min_nc() && launch_mf(t, s, y, ys, dst, ldd, K, N, nc, st, residual, ldr)) return true;   // MTP2-C
    for (int c0 = 0; c0 < nc; ) {
        const float * res = residual ? residual + (size_t) c0*ldr : nullptr;
        int done = nc - c0 >= 2 ? launch_nc_chunk(t, s, y + (size_t) c0*ys, ys, dst + (size_t) c0*ldd, ldd, K, N, std::min(nc - c0, LB_NC_MAX), st, res, ldr) : 0;
        if (done == 0) {
            if (!launch(t, soa_base, y + (size_t) c0*ys, dst + (size_t) c0*ldd, K, N, st, res)) return false;
            done = 1;
        }
        c0 += done;
    }
    return true;
}
// ================================================================================================ MMVQ-NC lowbit end

} // namespace lb
