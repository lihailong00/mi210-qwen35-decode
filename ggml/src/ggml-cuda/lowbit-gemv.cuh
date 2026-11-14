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

} // namespace lb
