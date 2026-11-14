// iq4k.cuh - single-token (ne11 == 1) GEMV for IQ4_XS and Q4_K on gfx90a (MI210).
//
// Design (see READY.md):
//  * weights are re-laid out once (lossless, same byte size) into an SoA format: Q stream (the 128 B of 4-bit codes per
//    256-weight block, 16-byte aligned) | S stream (the per-block header: IQ4_XS 8 B {d, scales_h, scales_l},
//    Q4_K 16 B {dm, scales[12]}). Q4_K codes are reordered per row so that both 16-byte halves of a 64-weight pair are
//    loaded fully coalesced. One wide load per lane item, several items in flight per lane (ping-pong pipelining across
//    rows/units), multiple rows per wave (L lanes per row), persistent grid.
//  * activation y = q8_1 (block_q8_1) staged once per workgroup in LDS as 16-byte aligned records
//    (IQ4_XS: {qs[8], d, pad[3]} per 32 values; Q4_K: {qs_a[8], qs_b[8], d_a, d_b, sum_a, sum_b} per 64 values).
//  * IQ4_NL code -> value lookup via v_perm (same as get_int_from_table_16).
// Numerics: the integer sums per 32-weight sub-block and their integer scale math are exactly those of llama.cpp
// vec_dot_iq4_xs_q8_1 / vec_dot_q4_K_q8_1; only the fp32 accumulation order differs.
#pragma once
#include "common.cuh"
#include "vecdotq.cuh"
#include <type_traits>

namespace iq4k {

static __device__ __forceinline__ uint32_t perm(uint32_t hi, uint32_t lo, uint32_t sel) { return __builtin_amdgcn_perm(hi, lo, sel); }
static __device__ __forceinline__ int dp4a(int a, int b, int c) { return __builtin_amdgcn_sdot4(a, b, c, false); }
static __device__ __forceinline__ float h2f(uint32_t h) { return __half2float(__ushort_as_half((unsigned short) (h & 0xFFFF))); }
static __device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }


// ---------------------------------------------------------------- SoA stream geometry
// nb = number of 256-weight blocks (nrows * K/256). Streams back to back: Q (nb*128 B) | S (nb*SB B). Same total size.
struct soa { const uint8_t * Q; const uint8_t * S; };
static __host__ __device__ inline int s_bytes(ggml_type t) { return t == GGML_TYPE_Q4_K ? 16 : 8; }
static __host__ __device__ inline soa soa_make(ggml_type t, const void * base, size_t nb) {
    soa s; s.Q = (const uint8_t *) base; s.S = s.Q + nb*128; (void) t; return s;
}

// ---------------------------------------------------------------- repack kernels (orig -> SoA), one thread per block
// Row interleave IL: groups of IL consecutive rows ("units") are interleaved at 16-byte granularity so that the IL*L
// lanes of a wave read one contiguous 1 KB piece per load instruction:
//   Q chunk (row, c) at (row/IL)*cpr*IL + c*IL + row%IL   (cpr = 16-byte chunks per row = 8*bpr)
//   S entry (row, b) at (row/IL)*bpr*IL + b*IL + row%IL
// IQ4_XS: chunk c = sub-block it (natural order). Q4_K: pair p = 4*blk + j: low 16 B half -> chunk p,
// high half -> chunk 4*bpr + p. IL = 1 is the plain row-major SoA layout. nrows % IL == 0. dst must not alias src.
static __host__ __device__ __forceinline__ size_t il_chunk(size_t row, size_t c, size_t cpr, int IL) { return (row/IL)*cpr*IL + c*IL + row%IL; }
static __global__ void repack_iq4_xs(const block_iq4_xs * __restrict__ src, uint8_t * __restrict__ dst, const int64_t nb, const int bpr, const int IL) {
    const int64_t g = blockIdx.x*(int64_t) blockDim.x + threadIdx.x;
    if (g >= nb) return;
    const uint32_t * b = (const uint32_t *) (src + g);    // 136 B, 8-byte aligned: {d, scales_h} {scales_l} qs[128]
    const int64_t row = g / bpr; const int blk = (int) (g % bpr);
    uint32_t * S = (uint32_t *) (dst + nb*128 + il_chunk(row, blk, bpr, IL)*8);
    S[0] = b[0]; S[1] = b[1];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        uint32_t * Q = (uint32_t *) (dst + il_chunk(row, 8*blk + j, 8*bpr, IL)*16);
        Q[0] = b[2 + 4*j]; Q[1] = b[3 + 4*j]; Q[2] = b[4 + 4*j]; Q[3] = b[5 + 4*j];
    }
}
static __global__ void repack_q4_K(const block_q4_K * __restrict__ src, uint8_t * __restrict__ dst, const int64_t nb, const int bpr, const int IL) {
    const int64_t g = blockIdx.x*(int64_t) blockDim.x + threadIdx.x;
    if (g >= nb) return;
    const uint4 * b = (const uint4 *) (src + g);          // 144 B, 16-byte aligned: {dm, scales[12]} qs[128]
    const int64_t row = g / bpr; const int blk = (int) (g % bpr);
    *(uint4 *) (dst + nb*128 + il_chunk(row, blk, bpr, IL)*16) = b[0];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        *(uint4 *) (dst + il_chunk(row, 4*blk + j,         8*bpr, IL)*16) = b[1 + 2*j];
        *(uint4 *) (dst + il_chunk(row, 4*bpr + 4*blk + j, 8*bpr, IL)*16) = b[2 + 2*j];
    }
}

// Round-tiled layout (RT, used with round-robin units and exact ipr % L == 0): unit u = rows [u*IL, (u+1)*IL),
// round k = units [k*NWV, (k+1)*NWV) (NWV = number of waves of the launch), nur = units in round k, pos = u - k*NWV.
// A unit takes q = ipr/L steps; the data of one wave step (L items x IL rows) is one contiguous chunk, and within a
// round the step chunks are stored step-major: chunk (k, kk, pos) at k*NWV*q + kk*nur + pos. So at every step the
// whole grid reads one contiguous slab (like a grid-stride stream). Q chunk = L*IL 16-byte pieces (l*IL + ri),
// S chunk = (L/IPB)*IL header entries ((l/IPB)*IL + ri). Q4_K: plane A (low halves) | plane B (high halves) | S.
static __host__ __device__ __forceinline__ size_t rt_chunk(size_t u, int kk, int q, int NWV, int nunits) {
    const size_t k = u / NWV, pos = u % NWV;
    const size_t nur = (size_t) min(NWV, nunits - (int) (k*NWV));
    return k*NWV*q + kk*nur + pos;
}
static __global__ void repack_rt_iq4_xs(const block_iq4_xs * __restrict__ src, uint8_t * __restrict__ dst, const int64_t nb, const int bpr,
        const int IL, const int L, const int NWV) {
    const int64_t g = blockIdx.x*(int64_t) blockDim.x + threadIdx.x;
    if (g >= nb) return;
    const uint32_t * b = (const uint32_t *) (src + g);
    const int64_t row = g / bpr; const int blk = (int) (g % bpr);
    const int ipr = bpr*8, q = ipr/L, nunits = (int) (nb/bpr)/IL;
    const size_t u = row / IL; const int ri = (int) (row % IL);
    {   // header of block blk: item 8*blk
        const int it = 8*blk, kk = it / L, l = it % L;
        const size_t e = rt_chunk(u, kk, q, NWV, nunits)*(L/8)*IL + (l/8)*IL + ri;
        uint32_t * S = (uint32_t *) (dst + nb*128 + e*8);
        S[0] = b[0]; S[1] = b[1];
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int it = 8*blk + j, kk = it / L, l = it % L;
        const size_t c = rt_chunk(u, kk, q, NWV, nunits)*L*IL + l*IL + ri;
        uint32_t * Q = (uint32_t *) (dst + c*16);
        Q[0] = b[2 + 4*j]; Q[1] = b[3 + 4*j]; Q[2] = b[4 + 4*j]; Q[3] = b[5 + 4*j];
    }
}
static __global__ void repack_rt_q4_K(const block_q4_K * __restrict__ src, uint8_t * __restrict__ dst, const int64_t nb, const int bpr,
        const int IL, const int L, const int NWV) {
    const int64_t g = blockIdx.x*(int64_t) blockDim.x + threadIdx.x;
    if (g >= nb) return;
    const uint4 * b = (const uint4 *) (src + g);
    const int64_t row = g / bpr; const int blk = (int) (g % bpr);
    const int ipr = bpr*4, q = ipr/L, nunits = (int) (nb/bpr)/IL;
    const size_t u = row / IL; const int ri = (int) (row % IL);
    {
        const int it = 4*blk, kk = it / L, l = it % L;
        const size_t e = rt_chunk(u, kk, q, NWV, nunits)*(L/4)*IL + (l/4)*IL + ri;
        *(uint4 *) (dst + nb*128 + e*16) = b[0];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int it = 4*blk + j, kk = it / L, l = it % L;
        const size_t c = rt_chunk(u, kk, q, NWV, nunits)*L*IL + l*IL + ri;
        *(uint4 *) (dst + c*16)         = b[1 + 2*j];
        *(uint4 *) (dst + nb*64 + c*16) = b[2 + 2*j];
    }
}

// Stream-K round-tiled layout (SK+RT): W = nunits*q steps split into ranges of T = ceil(W/NWV) steps (nact active
// waves, the last one gets `last` steps). Step s belongs to wave w = s/T at local step j = s%T; its chunk is stored
// step-major over the waves: j*nact + w (j < last), last*nact + (j-last)*(nact-1) + w (j >= last). Every grid step
// reads one contiguous slab and every wave does the same number of steps.
struct skgeo { int T, nact, last; };
static __host__ __device__ __forceinline__ skgeo sk_geo(const int64_t W, const int NWV) {
    skgeo g; g.T = (int) ((W + NWV - 1)/NWV); g.nact = (int) ((W + g.T - 1)/g.T); g.last = (int) (W - (int64_t) (g.nact - 1)*g.T); return g;
}
static __host__ __device__ __forceinline__ size_t sk_chunk(const int w, const int j, const skgeo & g) {
    return j < g.last ? (size_t) j*g.nact + w : (size_t) g.last*g.nact + (size_t) (j - g.last)*(g.nact - 1) + w;
}
static __host__ __device__ __forceinline__ size_t sk_chunk_of_step(const int64_t st, const skgeo & g) {
    return sk_chunk((int) (st / g.T), (int) (st % g.T), g);
}
template <int IPB, int SB>
static __global__ void repack_skrt(const uint8_t * __restrict__ src, uint8_t * __restrict__ dst, const int64_t nb, const int bpr,
        const int IL, const int L, const int NWV) {
    const int64_t g = blockIdx.x*(int64_t) blockDim.x + threadIdx.x;
    if (g >= nb) return;
    constexpr int BB = SB + 128;   // block bytes: header + codes
    const uint8_t * b = src + g*BB;
    const int64_t row = g / bpr; const int blk = (int) (g % bpr);
    const int ipr = bpr*IPB, q = ipr/L, nunits = (int) (nb/bpr)/IL;
    const skgeo geo = sk_geo((int64_t) nunits*q, NWV);
    const int64_t u = row / IL; const int ri = (int) (row % IL);
    {
        const int it = IPB*blk, kk = it / L, l = it % L;
        const size_t e = sk_chunk_of_step(u*q + kk, geo)*(L/IPB)*IL + (l/IPB)*IL + ri;
        if (SB == 8) { ((uint32_t *) (dst + nb*128 + e*8))[0] = ((const uint32_t *) b)[0]; ((uint32_t *) (dst + nb*128 + e*8))[1] = ((const uint32_t *) b)[1]; }
        else *(uint4 *) (dst + nb*128 + e*16) = *(const uint4 *) b;
    }
#pragma unroll
    for (int jj = 0; jj < IPB; ++jj) {
        const int it = IPB*blk + jj, kk = it / L, l = it % L;
        const size_t c = sk_chunk_of_step(u*q + kk, geo)*L*IL + l*IL + ri;
        if (IPB == 8) {   // IQ4_XS: 16 B of codes per sub-block (8-byte aligned source)
            const uint32_t * sq = (const uint32_t *) (b + SB + 16*jj);
            uint32_t * Q = (uint32_t *) (dst + c*16);
            Q[0] = sq[0]; Q[1] = sq[1]; Q[2] = sq[2]; Q[3] = sq[3];
        } else {          // Q4_K: pair jj = 32 B -> plane A (low half) / plane B (high half)
            *(uint4 *) (dst + c*16)         = *(const uint4 *) (b + SB + 32*jj);
            *(uint4 *) (dst + nb*64 + c*16) = *(const uint4 *) (b + SB + 32*jj + 16);
        }
    }
}

// ---------------------------------------------------------------- IQ4_NL lookup (v_perm, as get_int_from_table_16)
// kvalues_iq4nl = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}
struct tab4 { uint32_t t0, t1, t2, t3; };
static __device__ __forceinline__ uint32_t vconst(uint32_t v) {   // keep the constant in a VGPR (gfx9 VOP3: 1 SGPR/literal)
#if defined(__HIP_DEVICE_COMPILE__)
    uint32_t r; asm("v_mov_b32 %0, %1" : "=v"(r) : "i"(v)); return r;
#else
    return v;
#endif
}
static __device__ __forceinline__ uint32_t sconst(uint32_t v) {   // opaque uniform constant (SGPR), lets v_and_or_b32 form
#if defined(__HIP_DEVICE_COMPILE__)
    uint32_t r; asm("s_mov_b32 %0, %1" : "=s"(r) : "i"(v)); return r;
#else
    return v;
#endif
}
static __device__ __forceinline__ void lookup8(const uint32_t q, const tab4 & T, const uint32_t m4, const uint32_t sb,
        uint32_t & lo, uint32_t & hi) {
    const uint32_t qe = q & 0x07070707u;
    const uint32_t u  = q >> 4;
    const uint32_t qo = u & 0x07070707u;
    const uint32_t me = ((q >> 1) & m4) | sb;
    const uint32_t mo = ((u >> 1) & m4) | sb;
    const uint32_t ae = perm(T.t1, T.t0, qe), be = perm(T.t3, T.t2, qe);
    const uint32_t ao = perm(T.t1, T.t0, qo), bo = perm(T.t3, T.t2, qo);
    lo = perm(be, ae, me);
    hi = perm(bo, ao, mo);
}

// ---------------------------------------------------------------- type traits
// IQ4_XS: item = one 32-weight sub-block (16 B of codes); y record = 12 words {qs[8], d, 0, 0, 0}
struct iq4xs_t {
    static constexpr ggml_type type = GGML_TYPE_IQ4_XS;
    static constexpr int IPB = 8, YW = 12, YPI = 1 /* q8 blocks per item */;
    struct raw { uint4 q; uint2 h; };
    struct lc { uint32_t shl, shh; tab4 T; uint32_t m4, sb; };
    static __device__ __forceinline__ lc lane_consts(const int lane) {
        lc c; const int ib = lane & 7;
        c.shl = 4*ib; c.shh = 16 + 2*ib;
        c.T.t0 = vconst(0xBFAD9881u); c.T.t1 = vconst(0xF6EADDCFu); c.T.t2 = vconst(0x26190D01u); c.T.t3 = vconst(0x71594535u);
        c.m4 = sconst(0x04040404u); c.sb = vconst(0x03020100u);
        return c;
    }
    template <int IL>
    static __device__ __forceinline__ raw load(const soa & s, const uint32_t u, const uint32_t ri, const uint32_t it, const uint32_t bpr) {
        raw r;
        r.q = *(const uint4 *) (s.Q + (size_t) (((u*bpr*8 + it)*IL + ri)*16));
        r.h = *(const uint2 *) (s.S + (size_t) (((u*bpr + it/8)*IL + ri)*8));
        return r;
    }
    // RT layout: qc = 16-byte chunk index, se = header entry index
    static __device__ __forceinline__ raw load_rt(const soa & s, const uint32_t qc, const uint32_t se, const size_t nb) {
        raw r; (void) nb;
        r.q = *(const uint4 *) (s.Q + (size_t) qc*16);
        r.h = *(const uint2 *) (s.S + (size_t) se*8);
        return r;
    }
    // generic: 16-byte chunk index qc, header entry index se (32-bit byte offsets -> saddr + voffset loads)
    static __device__ __forceinline__ raw load_c(const soa & s, const uint32_t qc, const uint32_t se, const uint32_t boff) {
        raw r; (void) boff;
        r.q = *(const uint4 *) (s.Q + qc*16u);
#ifdef IQ4K_PROBE_NOS
        r.h = *(const uint2 *) (s.S + (se & 7u)*8u);   // timing probe only: header loads hit one cached line
#else
        r.h = *(const uint2 *) (s.S + se*8u);
#endif
        return r;
    }
    static __device__ __forceinline__ void put_y(uint32_t * ys, const int b, const uint4 q0, const uint4 q1, const float d) {
        uint32_t * r = ys + 12*b;
        *(uint4 *) r = q0; *(uint4 *) (r + 4) = q1; *(uint4 *) (r + 8) = make_uint4(__float_as_uint(d), 0u, 0u, 0u);
    }
    static __device__ __forceinline__ float compute(const raw & r, const uint32_t * yr, const lc & c) {
        const uint4 y0 = *(const uint4 *) yr;
        const uint4 y1 = *(const uint4 *) (yr + 4);
        const float d8 = __uint_as_float(yr[8]);
        uint32_t lo, hi;
        int s;
        lookup8(r.q.x, c.T, c.m4, c.sb, lo, hi); s = dp4a((int) lo, (int) y0.x, 0); s = dp4a((int) hi, (int) y1.x, s);
        lookup8(r.q.y, c.T, c.m4, c.sb, lo, hi); s = dp4a((int) lo, (int) y0.y, s); s = dp4a((int) hi, (int) y1.y, s);
        lookup8(r.q.z, c.T, c.m4, c.sb, lo, hi); s = dp4a((int) lo, (int) y0.z, s); s = dp4a((int) hi, (int) y1.z, s);
        lookup8(r.q.w, c.T, c.m4, c.sb, lo, hi); s = dp4a((int) lo, (int) y0.w, s); s = dp4a((int) hi, (int) y1.w, s);
        const int ls = (int) (((r.h.y >> c.shl) & 0xFu) | (((r.h.x >> c.shh) & 3u) << 4));
        const float d = h2f(r.h.x) * d8;
        return d * (float) __mul24(s, ls - 32);
    }
};

// Q4_K: item = one 64-weight pair (sub-blocks 2j, 2j+1 = 32 B of codes, two coalesced 16 B loads) + 16 B header;
// y record = 20 words {qs_a[8], qs_b[8], d_a, d_b, sum_a, sum_b}
struct q4k_t {
    static constexpr ggml_type type = GGML_TYPE_Q4_K;
    static constexpr int IPB = 4, YW = 20, YPI = 2;
    struct raw { uint4 a, b, h; };
    struct lc { uint32_t sh; bool hi; };
    static __device__ __forceinline__ lc lane_consts(const int lane) {
        lc c; const int j = lane & 3; c.sh = 16*(j & 1); c.hi = j >= 2; return c;
    }
    template <int IL>
    static __device__ __forceinline__ raw load(const soa & s, const uint32_t u, const uint32_t ri, const uint32_t it, const uint32_t bpr) {
        raw r;
        const uint32_t base = u*bpr*8*IL + ri;
        r.a = *(const uint4 *) (s.Q + (size_t) ((base + it*IL)*16));
        r.b = *(const uint4 *) (s.Q + (size_t) ((base + (4*bpr + it)*IL)*16));
        r.h = *(const uint4 *) (s.S + (size_t) (((u*bpr + it/4)*IL + ri)*16));
        return r;
    }
    static __device__ __forceinline__ raw load_rt(const soa & s, const uint32_t qc, const uint32_t se, const size_t nb) {
        raw r;
        r.a = *(const uint4 *) (s.Q + (size_t) qc*16);
        r.b = *(const uint4 *) (s.Q + nb*64 + (size_t) qc*16);
        r.h = *(const uint4 *) (s.S + (size_t) se*16);
        return r;
    }
    // generic: boff = chunk distance of the high half (row-interleaved: 4*bpr*IL, round-tiled: nb*4)
    static __device__ __forceinline__ raw load_c(const soa & s, const uint32_t qc, const uint32_t se, const uint32_t boff) {
        raw r;
        r.a = *(const uint4 *) (s.Q + qc*16u);
        r.b = *(const uint4 *) (s.Q + (qc + boff)*16u);
#ifdef IQ4K_PROBE_NOS
        r.h = *(const uint4 *) (s.S + (se & 3u)*16u);   // timing probe only
#else
        r.h = *(const uint4 *) (s.S + se*16u);
#endif
        return r;
    }
    static __device__ __forceinline__ void put_y(uint32_t * ys, const int b, const uint4 q0, const uint4 q1, const float d) {
        uint32_t * r = ys + 20*(b >> 1);
        const int h = b & 1;
        *(uint4 *) (r + 8*h) = q0; *(uint4 *) (r + 8*h + 4) = q1;
        int s = dp4a((int) q0.x, 0x01010101, 0); s = dp4a((int) q0.y, 0x01010101, s); s = dp4a((int) q0.z, 0x01010101, s); s = dp4a((int) q0.w, 0x01010101, s);
        s = dp4a((int) q1.x, 0x01010101, s); s = dp4a((int) q1.y, 0x01010101, s); s = dp4a((int) q1.z, 0x01010101, s); s = dp4a((int) q1.w, 0x01010101, s);
        r[16 + h] = __float_as_uint(d);
        r[18 + h] = (uint32_t) s;
    }
    static __device__ __forceinline__ float compute(const raw & r, const uint32_t * yr, const lc & c) {
        const uint4 ya0 = *(const uint4 *) yr,       ya1 = *(const uint4 *) (yr + 4);
        const uint4 yb0 = *(const uint4 *) (yr + 8), yb1 = *(const uint4 *) (yr + 12);
        const uint4 dd  = *(const uint4 *) (yr + 16);
        const uint32_t q[8]  = {r.a.x, r.a.y, r.a.z, r.a.w, r.b.x, r.b.y, r.b.z, r.b.w};
        const uint32_t ua[8] = {ya0.x, ya0.y, ya0.z, ya0.w, ya1.x, ya1.y, ya1.z, ya1.w};
        const uint32_t ub[8] = {yb0.x, yb0.y, yb0.z, yb0.w, yb1.x, yb1.y, yb1.z, yb1.w};
        int dota = 0, dotb = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            dota = dp4a((int) (q[w] & 0x0F0F0F0Fu), (int) ua[w], dota);
            dotb = dp4a((int) ((q[w] >> 4) & 0x0F0F0F0Fu), (int) ub[w], dotb);
        }
        // scales/mins of sub-blocks 2j (a) and 2j+1 (b) (get_scale_min_k4), j = lane & 3 (constant per lane)
        const uint32_t t1 = r.h.y >> c.sh, t2 = r.h.z >> c.sh, t3 = r.h.w >> c.sh;
        const uint32_t SC = c.hi ? ((t3 & 0x0F0Fu) | ((t1 >> 2) & 0x3030u)) : (t1 & 0x3F3Fu);
        const uint32_t MM = c.hi ? (((t3 >> 4) & 0x0F0Fu) | ((t2 >> 2) & 0x3030u)) : (t2 & 0x3F3Fu);
        const int sca = (int) (SC & 0xFFu), scb = (int) (SC >> 8);
        const int ma  = (int) (MM & 0xFFu), mb  = (int) (MM >> 8);
        const float da = __uint_as_float(dd.x), db = __uint_as_float(dd.y);
        const float sumf_d = da*(float) __mul24(dota, sca) + db*(float) __mul24(dotb, scb);
        const float sumf_m = da*(float) __mul24((int) dd.z, ma) + db*(float) __mul24((int) dd.w, mb);
        const float2 dm = __half22float2(*(const half2 *) &r.h.x);
        return dm.x*sumf_d - dm.y*sumf_m;
    }
};

// ---------------------------------------------------------------- y staging (global block_q8_1 -> LDS records)
// diet-gemv: the same pointer, opaque to the load/store vectorizer - the ds load is not merged with the two qs loads, so
// the qs words arrive in the register order of the record writes (no v_mov shuffle, see iq3g::ld_u32_opaque)
static __device__ __forceinline__ uint32_t ld_u32_opaque(const uint8_t * p) {
#if defined(__HIP_DEVICE_COMPILE__)
    asm("" : "+v"(p));
    return *(const __attribute__((address_space(1))) uint32_t *) (size_t) p;
#else
    return *(const uint32_t *) p;
#endif
}
template <class T>
static __device__ __forceinline__ void stage_y(uint32_t * __restrict__ ys, const block_q8_1 * __restrict__ yg, const int K,
        const int tid, const int nt) {
    for (int b = tid; b < K/32; b += nt) {
        const uint8_t * p = (const uint8_t *) (yg + b);
        const uint32_t ds = ld_u32_opaque(p);
        const uint4 q0 = *(const uint4 *) (p + 4);
        const uint4 q1 = *(const uint4 *) (p + 20);
        T::put_y(ys, b, q0, q1, h2f(ds));
    }
}

// ---------------------------------------------------------------- group reduction (sum over L lanes, deterministic)
template <int L> static __device__ __forceinline__ float group_sum(float v) {
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr (L == 8) {
        v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x141, 0xF, 0xF, false));   // row_half_mirror
        v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xF, 0xF, false));    // quad_perm 1,0,3,2
        v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xF, 0xF, false));    // quad_perm 2,3,0,1
        return v;
    } else {
        v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x128, 0xF, 0xF, false));   // row_ror:8
        v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x124, 0xF, 0xF, false));   // row_ror:4
        v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x122, 0xF, 0xF, false));   // row_ror:2
        v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x121, 0xF, 0xF, false));   // row_ror:1
        if constexpr (L >= 32) v += __shfl_xor(v, 16, 64);
        if constexpr (L == 64) v += __shfl_xor(v, 32, 64);
        return v;
    }
#else
    return v;
#endif
}

// ---------------------------------------------------------------- the GEMV kernel (stream-K balanced)
// Persistent workgroups of blockDim.y waves (y staged once per workgroup in LDS). A wave holds G = 64/L lane groups; a
// lane group computes R rows; a "unit" = G*R consecutive rows, stored row-interleaved (IL = G*R, see repack).
// Each unit takes q = ceil(items_per_row/L) steps (one item per lane per step). The flat step stream of all units
// (W = nunits*q steps) is split into equal contiguous ranges, one per wave (wave w = wy*gridDim.x + blockIdx.x, so
// consecutive ranges run on different CUs): every wave does the same amount of work (no tail imbalance).
// A unit cut by a range boundary is finished by two waves: both store their partial row sums to `scratch` and bump
// cnt[boundary]; the second arriver adds first + second part (fixed order: deterministic), applies the epilogue and
// resets the counter (graph replay safe; cnt must be zero-initialized once). Needs nwaves <= nunits.
// Pipelining: the loads of step s+1 are issued before step s is computed (ping-pong buffers, manual 2x unroll, an even
// number of steps, no conditional memory operations in the loop); the first step is loaded before the y staging.
// Results go to LDS during the loop and are written after it.
// GLU: dst = (W_up . y) * silu(W_gate . y) (same type/shape); residual (optional): dst = W . y + residual.
// Requirements: K % 256 == 0, N % (G*R) == 0.
#ifndef IQ4K_PF2
#define IQ4K_PF2 0   // 1: load both ping-pong buffers before the y staging (measured 0.1-2.5 us slower)
#endif
template <class T, int L, int R, bool GLU, bool SK, bool RT = false, bool EX = false, int U = 1, bool PF2 = IQ4K_PF2 != 0>
__launch_bounds__(1024, 1)
__global__ void gemv(const soa s, const soa sg, const block_q8_1 * __restrict__ yg, float * __restrict__ dst,
        const int K, const int N, const float * __restrict__ residual, float * __restrict__ scratch, unsigned int * __restrict__ cnt) {
    extern __shared__ __align__(16) uint32_t ysm[];
    constexpr int G = 64/L, GR = G*R, NV = GLU ? 2 : 1;
    const int tid  = threadIdx.y*64 + threadIdx.x;
    const int lane = threadIdx.x % L;
    const int grp  = threadIdx.x / L;
    const uint32_t bpr = K / QK_K;
    const int ipr    = bpr*T::IPB;
    const int q      = (ipr + L*U - 1) / (L*U);   // steps per unit (U items per lane per step)
    const bool exact = ipr % L == 0;
    const int nunits = N / GR;
    const int wy     = __builtin_amdgcn_readfirstlane(threadIdx.y);   // wave index in the workgroup (uniform)
    const int nwaves = gridDim.x*blockDim.y;
    const int w      = wy*gridDim.x + blockIdx.x;
    const int64_t W  = (int64_t) nunits*q;
    // SK: contiguous range [b, e) of the global step stream (units u = step/q).
    // !SK: round robin units w, w + nwaves, ... (all complete); local unit counter cu -> global unit w + cu*nwaves.
    // diet-gemv: !SK without signed divisions - myu by one unsigned division (operands >= 0), maxu = ceil(nunits/nwaves)
    // = units of wave 0 = myu + (wave 0 has a unit at myu*nwaves), the last step (ue, ke) = (myu - 1, q - 1) below
    const int myu    = SK ? 0 : (w < nunits ? (int) ((uint32_t) (nunits - 1 - w)/(uint32_t) nwaves) + 1 : 0);
    const skgeo geo  = sk_geo(W, nwaves);
    // SK+RT: equal ranges of geo.T steps (waves >= nact idle); plain SK: proportional ranges
    const int b      = SK ? (RT ? (w < geo.nact ? w*geo.T : (int) W) : (int) ((int64_t) w*W/nwaves)) : 0;
    const int e      = SK ? (RT ? (w < geo.nact - 1 ? b + geo.T : (int) W) : (int) ((int64_t) (w + 1)*W/nwaves)) : myu*q;
    const int maxu   = SK ? (int) (((W + nwaves - 1)/nwaves + q - 1)/q) + 1 : myu + (myu*nwaves < nunits ? 1 : 0);
    const int u0     = b / q;
    auto gunit = [&](const int cu) -> int { return SK ? cu : w + cu*nwaves; };
    float * ow = (float *) (ysm + (K/32/T::YPI)*T::YW) + wy*maxu*NV*GR;   // this wave's unit results [maxu][NV][GR]
    const typename T::lc c = T::lane_consts(lane);

    typedef typename T::raw raw;
    raw A[U*R], B[U*R];
    raw Ag[GLU ? U*R : 1], Bg[GLU ? U*R : 1];

    // last real step; !SK: e = myu*q, so (e - 1)/q = myu - 1 and (e - 1) % q = q - 1 (only used when e > b = 0, i.e. myu >= 1)
    const int ue = SK ? (e - 1)/q : myu - 1, ke = SK ? (e - 1) % q : q - 1;
    int ls = b, lu = u0, lk = SK ? b % q : 0;     // load cursor
    const size_t nbk = (size_t) N*bpr;
    // EX (exact: ipr % L == 0, round robin or round-tiled): chunk indices = uniform base + lane constant
    uint32_t qlane[R], slane[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { qlane[r] = lane*GR + grp*R + r; slane[r] = (lane/T::IPB)*GR + grp*R + r; }
    const uint32_t boff = RT ? (uint32_t) (nbk*4) : 4*bpr*GR;
    auto load_step = [&](raw (&bb)[U*R], raw (&bg)[GLU ? U*R : 1]) {
        const bool past = ls >= e;
        if constexpr (EX) {
            const int kr = past ? ue : lu, kk = past ? ke : lk;
            uint32_t qs_, ss_;
            if constexpr (SK && RT) {
                const int j = (past ? e - 1 : ls) - b;
                const uint32_t ch = (uint32_t) sk_chunk(w, j, geo);
                qs_ = ch*(U*L*GR); ss_ = ch*((U*L/T::IPB)*GR);
            } else if constexpr (RT) {
                const int nur = min(nwaves, nunits - kr*nwaves);
                const uint32_t ch = (uint32_t) (kr*nwaves*q + kk*nur + w);
                qs_ = ch*(U*L*GR); ss_ = ch*((U*L/T::IPB)*GR);
            } else {
                const uint32_t un = (uint32_t) gunit(kr);
                qs_ = (un*bpr*8 + kk*U*L)*GR; ss_ = (un*bpr + kk*(U*L/T::IPB))*GR;
            }
#pragma unroll
            for (int uu = 0; uu < U; ++uu)
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t qo = qs_ + uu*(L*GR) + qlane[r], so = ss_ + uu*((L/T::IPB)*GR) + slane[r];
                bb[uu*R + r] = T::load_c(s, qo, so, boff);
                if constexpr (GLU) bg[uu*R + r] = T::load_c(sg, qo, so, boff);
            }
        } else if constexpr (RT) {   // round-robin only: round = local unit index, position = w (requires ipr % L == 0)
            const int kr = past ? ue : lu, kk = past ? ke : lk;
            const int nur = min(nwaves, nunits - kr*nwaves);
            const uint32_t ch = (uint32_t) (kr*nwaves*q + kk*nur + w);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t ri = grp*R + r;
                const uint32_t qc = ch*(L*GR) + lane*GR + ri;
                const uint32_t se = ch*((L/T::IPB)*GR) + (lane/T::IPB)*GR + ri;
                bb[r] = T::load_rt(s, qc, se, nbk);
                if constexpr (GLU) bg[r] = T::load_rt(sg, qc, se, nbk);
            }
        } else {
            const uint32_t un = (uint32_t) gunit(past ? ue : lu);
            const uint32_t it = (uint32_t) min(lane + L*(past ? ke : lk), ipr - 1);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                bb[r] = T::template load<GR>(s, un, grp*R + r, it, bpr);
                if constexpr (GLU) bg[r] = T::template load<GR>(sg, un, grp*R + r, it, bpr);
            }
        }
        ++ls; if (++lk == q) { lk = 0; ++lu; }
    };

    // PF2: both ping-pong buffers (steps 0 and 1) are loaded before the y staging, so the pipeline starts without a bubble
    if (e > b) { load_step(A, Ag); if constexpr (PF2) load_step(B, Bg); }
    stage_y<T>(ysm, yg, K, tid, blockDim.y*64);
    __syncthreads();
    if (e <= b) return;

    float acc[R], accg[GLU ? R : 1];
#pragma unroll
    for (int r = 0; r < R; ++r) { acc[r] = 0.0f; if constexpr (GLU) accg[r] = 0.0f; }
    int cs = b, cu = u0, ck = b % q;              // compute cursor
    auto compute_step = [&](raw (&bb)[U*R], raw (&bg)[GLU ? U*R : 1]) {
#pragma unroll
        for (int uu = 0; uu < U; ++uu) {
            const int itl = lane + L*(ck*U + uu);
            const uint32_t * yr = EX ? ysm + lane*T::YW + (ck*U + uu)*(L*T::YW) : ysm + min(itl, ipr - 1)*T::YW;
            const bool valid = EX || exact || itl < ipr;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float v = T::compute(bb[uu*R + r], yr, c);
                acc[r] += valid ? v : 0.0f;
                if constexpr (GLU) { const float vg = T::compute(bg[uu*R + r], yr, c); accg[r] += valid ? vg : 0.0f; }
            }
        }
        if (ck == q - 1 || cs == e - 1) {         // end of a unit or of the range (uniform)
            if (cs < e) {
                float * o = ow + (cu - u0)*NV*GR + grp*R;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float v = group_sum<L>(acc[r]);
                    if (lane == 0) o[r] = v;
                    if constexpr (GLU) { const float g = group_sum<L>(accg[r]); if (lane == 0) o[GR + r] = g; }
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) { acc[r] = 0.0f; if constexpr (GLU) accg[r] = 0.0f; }
        }
        ++cs; if (++ck == q) { ck = 0; ++cu; }
    };

    const int nst = e - b;
    if constexpr (PF2) {
        for (int st = 0; st < nst; st += 2) {
            compute_step(A, Ag);    // step st
            load_step(A, Ag);       // step st+2 (clamped past the end)
            compute_step(B, Bg);    // step st+1
            load_step(B, Bg);       // step st+3
        }
    } else {
        for (int st = 0; st < nst; st += 2) {
            load_step(B, Bg);       // step st+1 (clamped past the end)
            compute_step(A, Ag);    // step st
            load_step(A, Ag);       // step st+2
            compute_step(B, Bg);    // step st+1
        }
    }

    // ---- epilogue: complete units straight to dst, split units through the two-wave combine
    const bool hs = SK && (b % q) != 0, ts = SK && (e % q) != 0;
    const int c0 = u0 + (hs ? 1 : 0), c1 = ue - (ts ? 1 : 0);
    for (int i = threadIdx.x; i < (c1 - c0 + 1)*GR; i += 64) {
        const int j = i / GR, rr = i % GR, cu = c0 + j, u = gunit(cu);
        const float * o = ow + (cu - u0)*NV*GR;
        float v = o[rr];
        if constexpr (GLU) v *= silu_f(o[GR + rr]);
        if (residual) v += residual[u*GR + rr];
        dst[u*GR + rr] = v;
    }
    auto combine = [&](const int p, const int role, const int u) {
        float * sc = scratch + (size_t) p*2*NV*GR;          // [role][NV][GR]
        const float * o = ow + (u - u0)*NV*GR;
        if (threadIdx.x < GR) {
#pragma unroll
            for (int nv = 0; nv < NV; ++nv) __hip_atomic_store(sc + (role*NV + nv)*GR + threadIdx.x, o[nv*GR + threadIdx.x], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __threadfence();
        unsigned int old = 0;
        if (threadIdx.x == 0) old = __hip_atomic_fetch_add(cnt + p, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        old = __shfl(old, 0, 64);
        if (old == 1) {
            __threadfence();
            if (threadIdx.x < GR) {
                const int rr = threadIdx.x;
                float t[NV];
#pragma unroll
                for (int nv = 0; nv < NV; ++nv) {
                    const float mine  = o[nv*GR + rr];
                    const float other = __hip_atomic_load(sc + ((1 - role)*NV + nv)*GR + rr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                    t[nv] = role == 0 ? mine + other : other + mine;   // first part + second part
                }
                float v = t[0];
                if constexpr (GLU) v *= silu_f(t[1]);
                if (residual) v += residual[u*GR + rr];
                dst[u*GR + rr] = v;
            }
            if (threadIdx.x == 0) __hip_atomic_store(cnt + p, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    };
    if constexpr (SK) {
        if (hs) combine(w, 1, u0);
        if (ts) combine(w + 1, 0, ue);
    }
}

template <class T> static inline size_t lds_bytes(int K) { return (size_t) (K/32/T::YPI)*T::YW*4; }
// + per-wave unit results: blockDim.y * maxu * NV * G*R floats
template <class T> static inline size_t lds_bytes_total(int K, int N, int L, int R, int NW, int grid, bool glu = false, bool sk = false, int U = 1) {
    const int G = 64/L, ipb = T::IPB, ipr = (K/256)*ipb, q = (ipr + L*U - 1)/(L*U), nunits = N/(G*R), nwaves = grid*NW;
    const int64_t W = (int64_t) nunits*q;
    const int maxu = sk ? (int) (((W + nwaves - 1)/nwaves + q - 1)/q) + 1 : (nunits + nwaves - 1)/nwaves;
    return lds_bytes<T>(K) + (size_t) NW*maxu*(glu ? 2 : 1)*G*R*4;
}

// ================================================================================================
// Host side
struct cfg { int L, R, NW, U, grid; bool sk; bool rt = false; };   // sk: stream-K ranges (else round-robin units); rt: round-tiled layout

// tuned on MI210 (iq4k_bench, real weights, rotating copies > L2)
static inline cfg pick_cfg(ggml_type t, int N, int K, int ncu) {
    (void) N;
    if (t == GGML_TYPE_IQ4_XS) {
        if (K > 8192) return cfg{32, 1, 13, 1, ncu, false, true};   // round-tiled layout (ffn_down)
        return cfg{16, 1, 15, 1, ncu, false};
    }
    if (K > 8192) return cfg{16, 1, 10, 1, ncu, false, true};
    return cfg{16, 1, 12, 1, ncu, false};
}

// scratch for the split-unit combine: >= nwaves*2*2*8 floats; cnt: >= nwaves+1 zero-initialized uints
static constexpr size_t SCRATCH_FLOATS = 1 << 16;
static constexpr size_t CNT_UINTS      = 1 << 13;

template <class T, int L, int R, bool GLU>
static inline void launch_k(const cfg & c, const soa & s, const soa & sg, const block_q8_1 * y, float * dst, int K, int N,
        hipStream_t st, const float * residual, float * scratch, unsigned int * cnt) {
    const size_t lds = lds_bytes_total<T>(K, N, L, R, c.NW, c.grid, GLU, c.sk, c.U);
    const bool ex = ((K/QK_K)*T::IPB) % L == 0;
    if (c.U == 2 && c.rt && !c.sk) { hipLaunchKernelGGL((gemv<T, L, R, GLU, false, true, true, 2>),  dim3(c.grid), dim3(64, c.NW), lds, st, s, sg, y, dst, K, N, residual, scratch, cnt); return; }
    if (c.U == 2 && !c.rt && !c.sk) { hipLaunchKernelGGL((gemv<T, L, R, GLU, false, false, true, 2>), dim3(c.grid), dim3(64, c.NW), lds, st, s, sg, y, dst, K, N, residual, scratch, cnt); return; }
    if (c.sk && c.rt) hipLaunchKernelGGL((gemv<T, L, R, GLU, true, true, true>), dim3(c.grid), dim3(64, c.NW), lds, st, s, sg, y, dst, K, N, residual, scratch, cnt);
    else if (c.sk) hipLaunchKernelGGL((gemv<T, L, R, GLU, true>),               dim3(c.grid), dim3(64, c.NW), lds, st, s, sg, y, dst, K, N, residual, scratch, cnt);
    else if (c.rt) hipLaunchKernelGGL((gemv<T, L, R, GLU, false, true, true>),  dim3(c.grid), dim3(64, c.NW), lds, st, s, sg, y, dst, K, N, residual, scratch, cnt);
    else if (ex)   hipLaunchKernelGGL((gemv<T, L, R, GLU, false, false, true>), dim3(c.grid), dim3(64, c.NW), lds, st, s, sg, y, dst, K, N, residual, scratch, cnt);
    else           hipLaunchKernelGGL((gemv<T, L, R, GLU, false>),              dim3(c.grid), dim3(64, c.NW), lds, st, s, sg, y, dst, K, N, residual, scratch, cnt);
}

static inline bool supported(ggml_type t) { return t == GGML_TYPE_IQ4_XS || t == GGML_TYPE_Q4_K; }

static inline bool shape_ok(const cfg & c, ggml_type t, int K, int N) {
    const int ipb = t == GGML_TYPE_Q4_K ? 4 : 8;
    const int GR = (64/c.L)*c.R;
    return K % 256 == 0 && (K/256)*ipb >= c.L && N % GR == 0 && (!c.sk || N/GR >= c.grid*c.NW) && c.grid*c.NW + 1 <= (int) CNT_UINTS &&
        (!c.rt || (((K/256)*ipb) % c.L == 0 && c.L % ipb == 0)) &&
        (c.U == 1 || (c.U == 2 && !c.sk && ((K/256)*ipb) % (2*c.L) == 0)) &&
        (size_t) c.grid*c.NW*2*2*GR <= SCRATCH_FLOATS && (size_t) N*(K/256)*144 < (1ull << 32);
}

static inline cfg pick_cfg_glu(ggml_type t, int N, int K, int ncu) {
    (void) N; (void) K;
    if (t == GGML_TYPE_IQ4_XS) return cfg{16, 1, 15, 1, ncu, false};
    return cfg{16, 1, 12, 1, ncu, false};
}

template <class T, bool GLU>
static inline bool dispatch(const cfg & c, const soa & s, const soa & sg, const block_q8_1 * y, float * dst, int K, int N,
        hipStream_t st, const float * residual, float * scratch, unsigned int * cnt) {
#define IQ4K_TRY(L_, R_) if (c.L == L_ && c.R == R_) { launch_k<T, L_, R_, GLU>(c, s, sg, y, dst, K, N, st, residual, scratch, cnt); return true; }
    IQ4K_TRY(16, 1) IQ4K_TRY(32, 1) IQ4K_TRY(64, 1)
#undef IQ4K_TRY
    return false;
}

// returns false if the type / shape is not supported (caller falls back to the stock path)
// soa_base: repacked weights (iq4k::repack, same byte size as the ggml tensor); y: K/32 block_q8_1; dst: float[N]
// residual (optional): dst[row] = dot(row) + residual[row]. scratch/cnt: see SCRATCH_FLOATS/CNT_UINTS (cnt zeroed once).
static inline bool launch(ggml_type t, const void * soa_base, const block_q8_1 * y, float * dst, int K, int N, hipStream_t st,
        float * scratch, unsigned int * cnt, const float * residual = nullptr, int ncu = 104) {
    if (!supported(t)) return false;
    const cfg c = pick_cfg(t, N, K, ncu);
    if (!shape_ok(c, t, K, N)) return false;
    const soa s = soa_make(t, soa_base, (size_t) N*(K/QK_K));
    if (t == GGML_TYPE_IQ4_XS) return dispatch<iq4xs_t, false>(c, s, s, y, dst, K, N, st, residual, scratch, cnt);
    return dispatch<q4k_t, false>(c, s, s, y, dst, K, N, st, residual, scratch, cnt);
}

// fused same-type gate/up: dst[row] = (W_up . y)[row] * silu((W_gate . y)[row]); both repacked with layout_il_glu
static inline bool launch_glu(ggml_type t, const void * soa_up, const void * soa_gate, const block_q8_1 * y, float * dst, int K, int N,
        hipStream_t st, float * scratch, unsigned int * cnt, int ncu = 104) {
    if (!supported(t)) return false;
    const cfg c = pick_cfg_glu(t, N, K, ncu);
    if (!shape_ok(c, t, K, N)) return false;
    const size_t nb = (size_t) N*(K/QK_K);
    const soa s = soa_make(t, soa_up, nb), sg = soa_make(t, soa_gate, nb);
    if (t == GGML_TYPE_IQ4_XS) return dispatch<iq4xs_t, true>(c, s, sg, y, dst, K, N, st, nullptr, scratch, cnt);
    return dispatch<q4k_t, true>(c, s, sg, y, dst, K, N, st, nullptr, scratch, cnt);
}

// IL = row interleave; must equal (64/L)*R of the configuration used to launch (iq4k::layout_il(type, N, K))
static inline bool repack_il(ggml_type t, const void * src, void * dst, int64_t nrows, int K, int IL, hipStream_t st) {
    const int64_t nb = nrows*(K/QK_K);
    const int nthr = 256; const int nblk = (int) ((nb + nthr - 1)/nthr);
    switch (t) {
        case GGML_TYPE_IQ4_XS: hipLaunchKernelGGL(repack_iq4_xs, dim3(nblk), dim3(nthr), 0, st, (const block_iq4_xs *) src, (uint8_t *) dst, nb, K/QK_K, IL); return true;
        case GGML_TYPE_Q4_K:   hipLaunchKernelGGL(repack_q4_K,   dim3(nblk), dim3(nthr), 0, st, (const block_q4_K *)   src, (uint8_t *) dst, nb, K/QK_K, IL); return true;
        default: return false;
    }
}
// repack for a given launch configuration (row-interleaved or round-tiled layout)
static inline bool repack_cfg(ggml_type t, const void * src, void * dst, int64_t nrows, int K, const cfg & c, hipStream_t st) {
    const int IL = (64/c.L)*c.R;
    if (!c.rt) return repack_il(t, src, dst, nrows, K, IL, st);
    const int64_t nb = nrows*(K/QK_K);
    const int nthr = 256; const int nblk = (int) ((nb + nthr - 1)/nthr);
    const int nwv = c.grid*c.NW;
    const int Ll = c.L*c.U;   // items per wave step of the layout
    if (c.sk) {
        switch (t) {
            case GGML_TYPE_IQ4_XS: hipLaunchKernelGGL((repack_skrt<8, 8>),  dim3(nblk), dim3(nthr), 0, st, (const uint8_t *) src, (uint8_t *) dst, nb, K/QK_K, IL, c.L, nwv); return true;
            case GGML_TYPE_Q4_K:   hipLaunchKernelGGL((repack_skrt<4, 16>), dim3(nblk), dim3(nthr), 0, st, (const uint8_t *) src, (uint8_t *) dst, nb, K/QK_K, IL, c.L, nwv); return true;
            default: return false;
        }
    }
    switch (t) {
        case GGML_TYPE_IQ4_XS: hipLaunchKernelGGL(repack_rt_iq4_xs, dim3(nblk), dim3(nthr), 0, st, (const block_iq4_xs *) src, (uint8_t *) dst, nb, K/QK_K, IL, Ll, nwv); return true;
        case GGML_TYPE_Q4_K:   hipLaunchKernelGGL(repack_rt_q4_K,   dim3(nblk), dim3(nthr), 0, st, (const block_q4_K *)   src, (uint8_t *) dst, nb, K/QK_K, IL, Ll, nwv); return true;
        default: return false;
    }
}
static inline int layout_il(ggml_type t, int N, int K, int ncu = 104) { const cfg c = pick_cfg(t, N, K, ncu); return (64/c.L)*c.R; }
static inline int layout_il_glu(ggml_type t, int N, int K, int ncu = 104) { const cfg c = pick_cfg_glu(t, N, K, ncu); return (64/c.L)*c.R; }
// repack for the tuned single-matrix launcher (launch); the GLU launcher may need a different interleave (layout_il_glu)
static inline bool repack(ggml_type t, const void * src, void * dst, int64_t nrows, int K, hipStream_t st, int ncu = 104) {
    return repack_cfg(t, src, dst, nrows, K, pick_cfg(t, (int) nrows, K, ncu), st);
}
static inline bool repack_glu(ggml_type t, const void * src, void * dst, int64_t nrows, int K, hipStream_t st, int ncu = 104) {
    return repack_cfg(t, src, dst, nrows, K, pick_cfg_glu(t, (int) nrows, K, ncu), st);
}


// ================================================================================================
// Grouped, workgroup-balanced GEMV ("g" kernel, iq4k-r3)
// One launch computes up to G_MAXJ GEMVs that read the same activation y (same K): phase A = jobs of type TA,
// phase B = jobs of type TB (none_t: no phase B). Every job uses the row-interleaved copy with IL = 64/L (L = 16: the
// same copy as the single-matrix launcher at K <= 8192). The units (GR = 64/L rows) of a phase are concatenated over
// its jobs; workgroup b takes the balanced contiguous unit range [b*U/grid, (b+1)*U/grid) of every phase, and its
// step stream (units x q steps, one 16-byte item per lane per step) is cut into blockDim.y contiguous ranges of
// Tw = ceil(steps/NW) steps, one per wave, so every wave does (almost) the same number of steps whatever N is: no
// round-robin tail, and small N becomes a split-K inside the workgroup. Loads are ping-pong pipelined as in gemv; the
// first step of both phases is loaded before the y staging. A wave stores the group sum of every unit piece it
// computed in LDS slot [unit][part] (part = wave index - first wave of the unit); after a barrier the workgroup adds
// the parts in fixed order (deterministic) and applies the epilogue: plain (+ optional residual per job) or GLU
// (dst = up * silu(gate), gate = phase A job 0, up = phase B job 0, same N).
struct none_t { static constexpr int IPB = 8, YW = 0, YPI = 1; };

static constexpr int G_MAXJ = 4;   // jobs per launch (at most 3 per phase)
struct gjob { const uint8_t * Q; const uint8_t * S; float * dst; const float * res; int u0; int nu; uint32_t boff; int pad_; };
struct gargs {
    gjob j[G_MAXJ];
    const block_q8_1 * y;
    int K, na, nb;        // phase A jobs j[0, na), phase B jobs j[na, na + nb)
    int UA, UB;           // units per phase
    int MA, PA, MB, PB;   // LDS partial slots per phase: max units per workgroup x max parts per unit
    int glu;
};

#define GSEL(i, f) ((i) == 0 ? P.j[0].f : (i) == 1 ? P.j[1].f : (i) == 2 ? P.j[2].f : P.j[3].f)
#ifndef IQ4K_G_STRIDED
#define IQ4K_G_STRIDED 1   // 1: workgroup b owns units b, b + grid, ... (consecutive units on different CUs, like the
                           // round-robin kernel); 0: a contiguous unit range per workgroup
#endif
struct gpg { int ub0, nu, q, Tw, b, e, luf, pf, us; };   // global unit of local unit lu = ub0 + lu*us
static __device__ __forceinline__ gpg g_geo(const int U, const int q, const int wy, const int NW) {
    gpg g;
#if IQ4K_G_STRIDED
    g.ub0 = blockIdx.x; g.us = gridDim.x;
    g.nu  = U > (int) blockIdx.x ? (U - 1 - (int) blockIdx.x)/(int) gridDim.x + 1 : 0;
#else
    g.ub0 = (int) (((int64_t) blockIdx.x*U)/gridDim.x); g.us = 1;
    g.nu  = (int) (((int64_t) (blockIdx.x + 1)*U)/gridDim.x) - g.ub0;
#endif
    g.q   = q;
    const int S = g.nu*q;
    g.Tw  = (S + NW - 1)/NW;
    g.b   = min(wy*g.Tw, S);
    g.e   = min(g.b + g.Tw, S);
    g.luf = g.Tw > 0 ? g.b/q : 0;
    g.pf  = g.Tw > 0 ? wy - (g.luf*q)/g.Tw : 0;
    return g;
}
// round-robin mode (RRM): wave gw (= wy*grid + blockIdx.x, NWV waves) owns complete units i0, i0 + NWV, ... of the phase,
// where unit i is dealt to wave (i + rot) % NWV (phase B is rotated by UA % NWV so that it fills the waves phase A
// leaves idle; GLU: rot = 0, gate and up unit on the same wave).
static __device__ __forceinline__ gpg g_geo_rr(const int U, const int q, const int gw, const int NWV, const int rot) {
    gpg g;
    const int i0 = (gw - rot % NWV + NWV) % NWV;
    g.ub0 = i0; g.us = NWV;
    g.nu  = U > i0 ? (U - 1 - i0)/NWV + 1 : 0;
    g.q   = q;
    g.b   = 0; g.e = g.nu*q; g.Tw = g.e > 0 ? g.e : 1;
    g.luf = 0; g.pf = 0;
    return g;
}

template <class T, int L>
struct gph {
    static constexpr int GR = 64/L;
    typedef typename T::raw raw;
    gpg g;
    uint32_t cpr, bpr, qlane, slane;
    int j0, nj;
    const gargs * Pp;   // job table: read from the kernarg segment (scalar loads); keeping copies of several jobs in
                        // this struct and selecting between them made the compiler index the struct in scratch memory
    // load cursor (clamped at the last real step) and current job
    int ls, lk, lun, lj, jnu, lgu, jnext;
    const uint8_t * Q, * S;
    uint32_t boff;
    // compute cursor
    int cs, ck, cl;
    float acc;

    __device__ __forceinline__ void set_job() {
        const gjob & J = Pp->j[j0 + lj];
        Q = J.Q; S = J.S; boff = J.boff; jnu = J.nu;
        jnext = lj + 1 < nj ? Pp->j[j0 + lj + 1].u0 : 0x7fffffff;
    }
    static __device__ __forceinline__ int steps_per_unit(const int K) { return (int) ((K/QK_K)*T::IPB)/L; }
    __device__ __forceinline__ void init(const gargs & P, const int j0_, const int nj_, const gpg & g_, const int K,
            const int lane, const int grp) {
        Pp = &P; j0 = j0_; nj = nj_;
        bpr = K/QK_K; cpr = 8*bpr;
        g = g_;
        const int q = g.q;
        qlane = lane*GR + grp; slane = (lane/T::IPB)*GR + grp;
        // load cursor at the first step of the range (no steps: some valid step, never computed)
        const int st = g.e > g.b ? g.b : 0;
        const int gu = g.nu > 0 ? g.ub0 + (st/q)*g.us : 0;
        lk = g.nu > 0 ? st - (st/q)*q : 0;
        lj = 0;
        for (int k = 1; k < nj; ++k) lj += gu >= Pp->j[j0 + k].u0;
        lun = gu - Pp->j[j0 + lj].u0;
        lgu = gu;
        ls = st;
        set_job();
        cs = g.b; ck = g.nu > 0 ? g.b - (g.b/q)*q : 0; cl = g.luf;
        acc = 0.0f;
    }
    __device__ __forceinline__ void load(raw & r) {
        const soa s = {Q, S};
        r = T::load_c(s, (lun*cpr + lk*L)*GR + qlane, (lun*bpr + lk*(L/T::IPB))*GR + slane, boff);
        // branch-free cursor advance (selects, like the round-robin kernel); only a job switch branches (rare, uniform)
        const int adv  = ls < g.e - 1 ? 1 : 0;
        const bool wrap = adv && lk + 1 == g.q;
        ls  += adv;
        lk   = wrap ? 0 : lk + adv;
        lgu += wrap ? g.us : 0;
        lun += wrap ? g.us : 0;
        if (lgu >= jnext) {
            do { lun = lgu - jnext; ++lj; set_job(); } while (lgu >= jnext);
        }
    }
    __device__ __forceinline__ void compute(const raw & r, const uint32_t * ys, const typename T::lc & c, float * part, const int PP,
            const int lane, const int grp) {
        acc += T::compute(r, ys + lane*T::YW + ck*(L*T::YW), c);
        if (ck == g.q - 1 || cs == g.e - 1) {
            if (cs < g.e) {
                const float v = group_sum<L>(acc);
                const int p = cl == g.luf ? g.pf : 0;
                if (lane == 0) part[(cl*PP + p)*GR + grp] = v;
            }
            acc = 0.0f;
        }
        ++cs; if (++ck == g.q) { ck = 0; ++cl; }
    }
    __device__ __forceinline__ void run(raw & X0, raw & X1, const uint32_t * ys, const typename T::lc & c, float * part, const int PP,
            const int lane, const int grp) {
        const int nst = g.e - g.b;
        for (int st = 0; st < nst; st += 2) {
            load(X1);                                   // step st+1 (clamped)
            compute(X0, ys, c, part, PP, lane, grp);    // step st
            load(X0);                                   // step st+2 (clamped)
            compute(X1, ys, c, part, PP, lane, grp);    // step st+1 (past the end: not stored)
        }
    }
    // sum of the parts of local unit lu, row r, in wave order (deterministic)
    __device__ __forceinline__ float part_sum(const float * part, const int PP, const int lu, const int r) const {
        const int w0 = (lu*g.q)/g.Tw, w1 = (lu*g.q + g.q - 1)/g.Tw;
        float v = part[(lu*PP)*GR + r];
        for (int p = 1; p <= w1 - w0; ++p) v += part[(lu*PP + p)*GR + r];
        return v;
    }
    // epilogue: per job (uniform loop), the units of this workgroup that belong to it
    __device__ __forceinline__ void store(const float * part, const int PP, const int tid, const int nt) const {
        for (int k = 0; k < nj; ++k) {
            const gjob & J = Pp->j[j0 + k];
            // local units lu with ub0 + lu*us in [J.u0, J.u0 + J.nu)
            const int lo = J.u0 > g.ub0 ? (J.u0 - g.ub0 + g.us - 1)/g.us : 0;
            const int hi = min(g.nu, J.u0 + J.nu > g.ub0 ? (J.u0 + J.nu - g.ub0 + g.us - 1)/g.us : 0);
            float * dst = J.dst;
            const float * res = J.res;
            for (int i = tid; i < (hi - lo)*GR; i += nt) {
                const int lu = lo + i / GR, r = i % GR;
                const int row = (g.ub0 + lu*g.us - J.u0)*GR + r;
                float v = part_sum(part, PP, lu, r);
                if (res) v += res[row];
                dst[row] = v;
            }
        }
    }
};

template <class TA, class TB, int L, bool RRM>
__launch_bounds__(1024, 1)
__global__ void gemv_g(const gargs P) {
    extern __shared__ __align__(16) uint32_t gsm[];
    constexpr bool HB = !std::is_same<TB, none_t>::value;
    constexpr bool SY = std::is_same<TA, TB>::value;
    typedef typename std::conditional<HB, TB, TA>::type TBX;
    constexpr int GR = 64/L;
    const int K = P.K;
    const int NW = blockDim.y, nt = NW*64;
    const int tid = threadIdx.y*64 + threadIdx.x;
    const int lane = threadIdx.x % L, grp = threadIdx.x / L;
    const int wy = __builtin_amdgcn_readfirstlane(threadIdx.y);
    const int ywA = (K/32/TA::YPI)*TA::YW;
    const int ywB = (HB && !SY) ? (K/32/TBX::YPI)*TBX::YW : 0;
    uint32_t * yA = gsm;
    uint32_t * yB = (HB && !SY) ? gsm + ywA : gsm;
    float * partA = (float *) (gsm + ywA + ywB);
    float * partB = partA + (RRM ? NW : 1)*P.MA*P.PA*GR;
    if constexpr (RRM) { partA += wy*P.MA*GR; partB += wy*P.MB*GR; }   // per-wave result slots
    const int gw = wy*(int) gridDim.x + (int) blockIdx.x, NWV = (int) gridDim.x*NW;

    gph<TA, L> pa;
    {
        const int q = gph<TA, L>::steps_per_unit(K);
        pa.init(P, 0, P.na, RRM ? g_geo_rr(P.UA, q, gw, NWV, 0) : g_geo(P.UA, q, wy, NW), K, lane, grp);
    }
    typename TA::raw A0, A1;
    pa.load(A0);
    gph<TBX, L> pb;
    typename TBX::raw B0, B1;
    if constexpr (HB) {
        const int q = gph<TBX, L>::steps_per_unit(K);
        pb.init(P, P.na, P.nb, RRM ? g_geo_rr(P.UB, q, gw, NWV, P.glu ? 0 : P.UA) : g_geo(P.UB, q, wy, NW), K, lane, grp);
        pb.load(B0);
    }
    const typename TA::lc ca = TA::lane_consts(lane);

    for (int b = tid; b < K/32; b += nt) {
        const uint8_t * p = (const uint8_t *) (P.y + b);
        const uint32_t ds = *(const uint32_t *) p;
        const uint4 q0 = *(const uint4 *) (p + 4);
        const uint4 q1 = *(const uint4 *) (p + 20);
        const float d = h2f(ds);
        TA::put_y(yA, b, q0, q1, d);
        if constexpr (HB && !SY) TBX::put_y(yB, b, q0, q1, d);
    }
    __syncthreads();

    pa.run(A0, A1, yA, ca, partA, P.PA, lane, grp);
    if constexpr (HB) {
        const typename TBX::lc cb = TBX::lane_consts(lane);
        pb.run(B0, B1, yB, cb, partB, P.PB, lane, grp);
    }
    // RRM: every wave stores its own complete units (no barrier); else the workgroup combines the parts
    const int et = RRM ? (int) threadIdx.x : tid, ent = RRM ? 64 : nt;
    if constexpr (!RRM) __syncthreads();

    if constexpr (HB) {
        if (P.glu) {   // gate = phase A job 0, up = phase B job 0 (same units)
            float * dst = GSEL(P.na, dst);
            for (int i = et; i < pa.g.nu*GR; i += ent) {
                const int lu = i / GR, r = i % GR;
                const float gv = pa.part_sum(partA, P.PA, lu, r);
                const float uv = pb.part_sum(partB, P.PB, lu, r);
                dst[(pa.g.ub0 + lu*pa.g.us)*GR + r] = uv*silu_f(gv);
            }
            return;
        }
    }
    pa.store(partA, P.PA, et, ent);
    if constexpr (HB) pb.store(partB, P.PB, et, ent);
}

// ---------------------------------------------------------------- host side of the g kernel
struct gcfg { int L, NW, grid; bool rr = true; };   // rr: round-robin complete units per wave (RRM), else balanced split
struct gjob_h { ggml_type type; const void * soa; float * dst; const float * res; int N; };

static inline int g_ipb(ggml_type t) { return t == GGML_TYPE_Q4_K ? 4 : 8; }
// LDS partial slots of a phase: max units per workgroup, max parts per unit
static inline void g_slots(const int U, const int q, const int grid, const int NW, int & M, int & PP) {
    M = 1; PP = 1;
    for (int k = 0; k < 2; ++k) {
        const int nu = k == 0 ? U/grid : (U + grid - 1)/grid;
        if (nu == 0) continue;
        const int S = nu*q, Tw = (S + NW - 1)/NW;
        M = nu > M ? nu : M;
        for (int lu = 0; lu < nu; ++lu) { const int np = (lu*q + q - 1)/Tw - (lu*q)/Tw + 1; PP = np > PP ? np : PP; }
    }
}
static inline size_t g_ywords(ggml_type t, int K) { return t == GGML_TYPE_Q4_K ? (size_t) (K/64)*20 : (size_t) (K/32)*12; }

// jobs: plain mode - any order, IQ4_XS and Q4_K may be mixed (grouped by type into the two phases);
// glu mode - jobs[0] = gate, jobs[1] = up (same N, any of the two types), dst = jobs[1].dst.
// soa = row-interleaved copy with IL = 64/c.L (iq4k::repack_il). Returns false if unsupported (caller falls back).
static inline bool g_supported(const gjob_h * jobs, int nj, int K, bool glu, const gcfg & c) {
    if (nj < 1 || nj > G_MAXJ || K % QK_K != 0 || (glu && nj != 2) || (c.L != 16 && c.L != 32) || c.NW < 1 || c.NW > 16 || c.grid < 1) return false;
    const int GR = 64/c.L;
    int n4 = 0, nq = 0;
    for (int i = 0; i < nj; ++i) {
        const ggml_type t = jobs[i].type;
        if (t != GGML_TYPE_IQ4_XS && t != GGML_TYPE_Q4_K) return false;
        if (((K/QK_K)*g_ipb(t)) % c.L != 0 || jobs[i].N % GR != 0 || jobs[i].N <= 0) return false;
        if ((size_t) jobs[i].N*(K/QK_K)*144 >= (1ull << 32)) return false;
        if (t == GGML_TYPE_IQ4_XS) n4++; else nq++;
    }
    if (glu) return jobs[0].N == jobs[1].N && !jobs[0].res && !jobs[1].res && c.L == 16;
    return n4 <= 3 && nq <= 3 && (c.L == 16 || n4 == 0 || nq == 0);
}

template <class TA, class TB, int L>
static inline void g_launch_k(const gargs & a, const gcfg & c, size_t lds, hipStream_t st) {
    if (c.rr) hipLaunchKernelGGL((gemv_g<TA, TB, L, true>),  dim3(c.grid), dim3(64, c.NW), lds, st, a);
    else      hipLaunchKernelGGL((gemv_g<TA, TB, L, false>), dim3(c.grid), dim3(64, c.NW), lds, st, a);
}

static inline bool launch_g(const gjob_h * jobs, int nj, const block_q8_1 * y, int K, hipStream_t st, bool glu, const gcfg & c) {
    if (!g_supported(jobs, nj, K, glu, c)) return false;
    const int GR = 64/c.L, bpr = K/QK_K;
    // phase order: glu -> (gate, up); plain -> IQ4_XS jobs first, then Q4_K jobs
    int ord[G_MAXJ]; int na = 0, nb = 0;
    ggml_type tA, tB = GGML_TYPE_COUNT;
    if (glu) { ord[0] = 0; ord[1] = 1; na = 1; nb = 1; tA = jobs[0].type; tB = jobs[1].type; }
    else {
        int k = 0;
        for (int i = 0; i < nj; ++i) if (jobs[i].type == GGML_TYPE_IQ4_XS) ord[k++] = i;
        na = k;
        for (int i = 0; i < nj; ++i) if (jobs[i].type == GGML_TYPE_Q4_K) ord[k++] = i;
        nb = k - na;
        if (na == 0) { na = nb; nb = 0; }
        tA = jobs[ord[0]].type; tB = nb > 0 ? jobs[ord[na]].type : GGML_TYPE_COUNT;
    }
    gargs a{};
    a.y = y; a.K = K; a.na = na; a.nb = nb; a.glu = glu ? 1 : 0;
    int U[2] = {0, 0};
    for (int k = 0; k < na + nb; ++k) {
        const gjob_h & J = jobs[ord[k]];
        const int ph = k < na ? 0 : 1;
        const size_t nbk = (size_t) J.N*bpr;
        a.j[k].Q = (const uint8_t *) J.soa; a.j[k].S = (const uint8_t *) J.soa + nbk*128;
        a.j[k].dst = J.dst; a.j[k].res = J.res;
        a.j[k].u0 = U[ph]; a.j[k].nu = J.N/GR; U[ph] += J.N/GR;
        a.j[k].boff = (uint32_t) (4*bpr*GR);
    }
    a.UA = U[0]; a.UB = U[1];
    const int qA = (bpr*g_ipb(tA))/c.L;
    const int NWV = c.grid*c.NW;
    if (c.rr) { a.MA = (a.UA + NWV - 1)/NWV; a.PA = 1; } else g_slots(a.UA, qA, c.grid, c.NW, a.MA, a.PA);
    size_t yw = g_ywords(tA, K);
    if (nb > 0) {
        const int qB = (bpr*g_ipb(tB))/c.L;
        if (c.rr) { a.MB = (a.UB + NWV - 1)/NWV; a.PB = 1; } else g_slots(a.UB, qB, c.grid, c.NW, a.MB, a.PB);
        if (tB != tA) yw += g_ywords(tB, K);
    } else { a.MB = 0; a.PB = 0; }
    const size_t lds = yw*4 + (size_t) (a.MA*a.PA + a.MB*a.PB)*GR*4*(c.rr ? c.NW : 1);
    if (lds > 65536) return false;
    const bool A4 = tA == GGML_TYPE_IQ4_XS;
    if (c.L == 16) {
        if (nb == 0) { if (A4) g_launch_k<iq4xs_t, none_t, 16>(a, c, lds, st); else g_launch_k<q4k_t, none_t, 16>(a, c, lds, st); }
        else if (A4 && tB == GGML_TYPE_Q4_K) g_launch_k<iq4xs_t, q4k_t, 16>(a, c, lds, st);
        else if (!A4 && tB == GGML_TYPE_IQ4_XS) g_launch_k<q4k_t, iq4xs_t, 16>(a, c, lds, st);
        else if (A4) g_launch_k<iq4xs_t, iq4xs_t, 16>(a, c, lds, st);
        else g_launch_k<q4k_t, q4k_t, 16>(a, c, lds, st);
        return true;
    }
    // L = 32: single-type plain only
    if (nb != 0) return false;
    if (A4) g_launch_k<iq4xs_t, none_t, 32>(a, c, lds, st); else g_launch_k<q4k_t, none_t, 32>(a, c, lds, st);
    return true;
}


// ================================================================================================
// Small-N multi-job GEMV ("sn" kernel, iq4k-r3), for the 1024-row attn_k / attn_v GEMVs. These are latency bound, and
// the LDS-staged kernels above lose ~2 us to the y staging there, so this is exactly the mul_mat_vec_q structure of
// llama.cpp for ncols = 1 on GCN/CDNA (2 waves per row, y read from global memory, original GGUF block layout,
// vec_dot_*_q8_1, the same shared-memory + warp reduction -> bit-identical to the stock path), but the rows of up to
// G_MAXJ jobs (IQ4_XS and/or Q4_K) that share the activation are concatenated over blockIdx.x: k and v take one
// launch instead of two. The job (and its type) is uniform per workgroup.
struct snjob { const void * w; float * dst; int r0; int q4k; };
struct snargs { snjob j[G_MAXJ]; const block_q8_1 * y; int bpr, nj; };

template <ggml_type type>
static __device__ __forceinline__ float sn_rowdot(const void * __restrict__ vx, const block_q8_1 * __restrict__ y, const int bpr,
        const int row, const int tid) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk, qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = type == GGML_TYPE_Q4_K ? VDR_Q4_K_Q8_1_MMVQ : VDR_IQ4_XS_Q8_1_MMVQ;
    constexpr int bpi = vdr*2*64/qi;   // blocks per iteration of the 128 threads
    float tmp = 0.0f;
    for (int kbx = tid/(qi/vdr); kbx < bpr; kbx += bpi) {
        const int kby = kbx*(qk/QK8_1);
        const int kqs = vdr*(tid % (qi/vdr));
        if constexpr (type == GGML_TYPE_Q4_K) {
            tmp += vec_dot_q4_K_q8_1(vx, &y[kby], row*bpr + kbx, kqs);
        } else {
            tmp += vec_dot_iq4_xs_q8_1(vx, &y[kby], row*bpr + kbx, kqs);
        }
    }
    return tmp;
}

// keeps a kernarg value in SGPRs: stops the compiler from turning a select between kernarg values into a load from a
// selected kernarg address (a second, dependent round of scalar loads before the first weight load)
template <typename V> static __device__ __forceinline__ V sreg(V v) {
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("" : "+s"(v));
#endif
    return v;
}

__launch_bounds__(128, 1)
static __global__ void gemv_sn(const snargs a) {
    const int b = blockIdx.x;
    const int j = (a.nj > 1 && b >= a.j[1].r0 ? 1 : 0) + (a.nj > 2 && b >= a.j[2].r0 ? 1 : 0) + (a.nj > 3 && b >= a.j[3].r0 ? 1 : 0);
    const void * w0 = sreg(a.j[0].w), * w1 = sreg(a.j[1].w), * w2 = sreg(a.j[2].w), * w3 = sreg(a.j[3].w);
    float * d0 = sreg(a.j[0].dst), * d1 = sreg(a.j[1].dst), * d2 = sreg(a.j[2].dst), * d3 = sreg(a.j[3].dst);
    const int s0 = sreg(a.j[0].r0), s1 = sreg(a.j[1].r0), s2 = sreg(a.j[2].r0), s3 = sreg(a.j[3].r0);
    const int k0 = sreg(a.j[0].q4k), k1 = sreg(a.j[1].q4k), k2 = sreg(a.j[2].q4k), k3 = sreg(a.j[3].q4k);
    const void * w   = j == 0 ? w0 : j == 1 ? w1 : j == 2 ? w2 : w3;
    float      * dst = j == 0 ? d0 : j == 1 ? d1 : j == 2 ? d2 : d3;
    const int    r0  = j == 0 ? s0 : j == 1 ? s1 : j == 2 ? s2 : s3;
    const int    q4k = j == 0 ? k0 : j == 1 ? k1 : j == 2 ? k2 : k3;
    const int row = b - r0;
    const int tid = threadIdx.y*64 + threadIdx.x;
    const float t = q4k ? sn_rowdot<GGML_TYPE_Q4_K>(w, a.y, a.bpr, row, tid) : sn_rowdot<GGML_TYPE_IQ4_XS>(w, a.y, a.bpr, row, tid);
    __shared__ float sh[64];
    if (threadIdx.y == 1) {
        sh[threadIdx.x] = t;
    }
    __syncthreads();
    if (threadIdx.y == 0) {
        float v = t + sh[threadIdx.x];
        v = warp_reduce_sum<64>(v);
        if (threadIdx.x == 0) {
            dst[row] = v;
        }
    }
}

// jobs[k].soa = the ORIGINAL ggml tensor data (no re-layout); all jobs share y (K/32 block_q8_1). false: unsupported.
static inline bool launch_sn(const gjob_h * jobs, int nj, const block_q8_1 * y, int K, hipStream_t st) {
    if (nj < 1 || nj > G_MAXJ || K % QK_K != 0) return false;
    snargs a{};
    int r = 0;
    for (int k = 0; k < nj; ++k) {
        if (jobs[k].type != GGML_TYPE_IQ4_XS && jobs[k].type != GGML_TYPE_Q4_K) return false;
        if ((int64_t) r + jobs[k].N > (1 << 30) || (int64_t) jobs[k].N*(K/QK_K) >= (1ll << 31)) return false;
        a.j[k] = { jobs[k].soa, jobs[k].dst, r, jobs[k].type == GGML_TYPE_Q4_K ? 1 : 0 };
        r += jobs[k].N;
    }
    a.y = y; a.bpr = K/QK_K; a.nj = nj;
    hipLaunchKernelGGL(gemv_sn, dim3(r), dim3(64, 2), 0, st, a);
    return true;
}

} // namespace iq4k
