// iq3gemv.cuh - single-token GEMV for IQ3_S / IQ3_XXS on a lossless SoA re-layout (gfx90a / MI210)
//
// Layout v2 (done once per weight tensor into a separate buffer; the original stays for MMQ / get_rows):
//   chunk = half-block (128 weights = 4 sub-blocks of 32) of one row; nh = 2*nb chunks; items = K/128 chunks per row.
//   Rows are interleaved in groups of RG rows (RG = rows per wave of the kernel config): chunk (row, h) is stored at
//     c = ((row/RG)*items + h)*RG + row%RG       -> every load instruction of a wave reads one contiguous range.
//   IQ3_S  (112 B per 256 weights, +1.8% vs GGUF 110):   Q[nh][32] | F[nh][24] (as FA[nh][16] | FB[nh][8])
//     Q = the 32 qs bytes of the half-block; F = 6 words D0..D5:
//       D0, D1, D2[15:0] = 16 5-bit fields of sub-blocks 0,1 (field i = 8*j + k at bit 5*i), D2[31:16] = d (fp16)
//       D3, D4, D5[15:0] = 16 5-bit fields of sub-blocks 2,3,                                D5[31:16] = 4 scale nibbles
//       field f = hbit | signs4 << 1 (signs4 bit b = sign of weight 4k+b, hbit = bit 8 of grid index k)
//   IQ3_XXS (98 B per 256 weights, same as GGUF):         Q[nh][32] | A[nh][16] | D[nb][2]
//     A = one word per sub-block: byte m = the 7 sign bits of 8-weight group m (bits 0..6), bit 7 = bit m of the scale;
//     D = d (fp16) per block, block chunk index cb = ((row/RG)*(items/2) + b)*RG + row%RG
// Decode: sign-expanded codebooks in LDS (entry = grid word with 4 signs applied), indexed by
//   IQ3_S  : qs | hbit << 8 | signs4 << 9   (8192 words, 32 KB)      address = qs*4 | f << 10
//   IQ3_XXS: qs | signs4 << 8               (4096 words, 16 KB)      address = qs*4 | signs4 << 10
//   -> per 4 weights: 1 SDWA (qs*4), field shift + AND_OR, 1 ds_read_b32, 1 dp4a; no sign arithmetic.
//   IQ3_XXS 8th sign (parity of 7) rebuilt for 4 groups at once with a bytewise XOR fold.
// Numerics: per-32-weight integer sums and the integer scale math are identical to vec_dot_iq3_s_q8_1 /
//   vec_dot_iq3_xxs_q8_1 (vecdotq-iq3-sign.cuh); per sub-block float term (d_w*d_y)*sumi as in the reference,
//   only the fp32 accumulation order differs.
// Activation: q8_1 (block_q8_1) staged by every workgroup into LDS as per-half-block records {qs[32 words], d[4]}.
// LDS contract: tables at LDS address 0 -> kernels must not declare static __shared__ (traps otherwise).
#pragma once
#include "common.cuh"
#include <type_traits>

#ifndef IQ3G_SDWA
#define IQ3G_SDWA 1
#endif

namespace iq3g {

typedef __attribute__((address_space(3))) const int      lds_i32;

static __device__ __forceinline__ int dp4a(int a, int b, int c) { return __builtin_amdgcn_sdot4(a, b, c, false); }
static __device__ __forceinline__ float h2f(uint32_t h) { return __half2float(__ushort_as_half((unsigned short) h)); }
// truncating signed division by 2^s (== C '/')
template <int s> static __device__ __forceinline__ int sdiv(int x) { return (x + (int) ((uint32_t) (x >> 31) >> (32 - s))) >> s; }
static __device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

// ((x >> 8*b) & 0xFF) << 2 in one SDWA instruction (b is a compile-time constant after unrolling)
static __device__ __forceinline__ uint32_t bx4(const uint32_t x, const int b) {
#if IQ3G_SDWA && defined(__HIP_DEVICE_COMPILE__)
    uint32_t r;
    const uint32_t two = 2;
    switch (b) {
        case 0:  asm("v_lshlrev_b32_sdwa %0, %1, %2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0" : "=v"(r) : "v"(two), "v"(x)); break;
        case 1:  asm("v_lshlrev_b32_sdwa %0, %1, %2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1" : "=v"(r) : "v"(two), "v"(x)); break;
        case 2:  asm("v_lshlrev_b32_sdwa %0, %1, %2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2" : "=v"(r) : "v"(two), "v"(x)); break;
        default: asm("v_lshlrev_b32_sdwa %0, %1, %2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3" : "=v"(r) : "v"(two), "v"(x)); break;
    }
    return r;
#else
    return ((x >> (8*b)) & 0xFFu) << 2;
#endif
}
static __device__ __forceinline__ int tab_ld(const uint32_t addr) {
#if defined(__HIP_DEVICE_COMPILE__)
    return *(lds_i32 *) (size_t) addr;
#else
    return (int) addr;
#endif
}
// apply 4 sign bits (bit b -> byte b negated) to a grid word with positive bytes
static __host__ __device__ __forceinline__ uint32_t apply_signs(const uint32_t g, const uint32_t s4) {
    const uint32_t ones = (s4 * 0x00204081u) & 0x01010101u;
    return (g - ones) ^ (ones * 255u);
}

// ------------------------------------------------------------------------------------------------ layouts
static __host__ __device__ inline size_t soa_bytes(ggml_type t, size_t nb) {
    return t == GGML_TYPE_IQ3_S ? nb*112 : t == GGML_TYPE_IQ3_XXS ? nb*98 : 0;
}
static __host__ __device__ __forceinline__ uint32_t chunk_index(uint32_t row, uint32_t h, uint32_t items, uint32_t RG) {
    return ((row / RG)*items + h)*RG + row % RG;
}
// CPS > 0: step-major layout - the CPS chunks that one wave step of a unit (RG rows) covers are contiguous, followed
// by the same step of the next unit (U = N/RG units); all waves then read one moving window of the matrix instead of
// U separate streams. CPS = 0: unit-major (chunk_index). Requires items % CPS == 0.
static __host__ __device__ __forceinline__ uint32_t chunk_index_sm(uint32_t row, uint32_t h, uint32_t items, uint32_t RG, uint32_t U, uint32_t CPS) {
    if (CPS == 0) return chunk_index(row, h, items, RG);
    return (((h / CPS)*U + row / RG)*CPS + h % CPS)*RG + row % RG;
}

// repack: one thread per (row, half-block). dst must not alias src. N % RG == 0, K % 256 == 0, (K/128) % CPS == 0.
static __global__ void repack_iq3_s(const block_iq3_s * __restrict__ src, uint8_t * __restrict__ dst, const int N, const int K, const int RG,
        const int CPS = 0) {
    const size_t t = blockIdx.x*(size_t) blockDim.x + threadIdx.x;
    const uint32_t items = K/128;
    const size_t nh = (size_t) N*items;
    if (t >= nh) return;
    const uint32_t row = (uint32_t) (t / items), h = (uint32_t) (t % items);
    const block_iq3_s & b = src[(size_t) row*(K/256) + h/2];
    const int half = (int) (h & 1);
    uint32_t q[8];
    memcpy(q, b.qs + 32*half, 32);
    uint32_t F[6] = {0, 0, 0, 0, 0, 0};
    uint32_t scales = 0;
    for (int j = 0; j < 4; ++j) {
        const int jg = 4*half + j;
        const uint32_t qh = b.qh[jg];
        for (int k = 0; k < 8; ++k) {
            const uint32_t hb = (qh >> k) & 1u;
            const uint32_t s4 = (b.signs[4*jg + k/2] >> (4*(k & 1))) & 0xFu;
            const uint32_t f = hb | (s4 << 1);
            const int bit = 5*(8*(j & 1) + k);            // position in the part window
            const int w0 = 3*(j >> 1);                     // part window base word
            const uint64_t v = (uint64_t) f << (bit & 31);
            F[w0 + (bit >> 5)] |= (uint32_t) v;
            if ((bit & 31) > 27) F[w0 + (bit >> 5) + 1] |= (uint32_t) (v >> 32);
        }
        scales |= (uint32_t) ((b.scales[jg/2] >> (4*(jg & 1))) & 0xF) << (4*j);
    }
    uint16_t d; memcpy(&d, &b.d, 2);
    F[2] |= (uint32_t) d << 16;
    F[5] |= scales << 16;
    const uint32_t c = chunk_index_sm(row, h, items, (uint32_t) RG, (uint32_t) (N/RG), (uint32_t) CPS);
    uint4 * Q = (uint4 *) dst; uint4 * FA = (uint4 *) (dst + nh*32); uint2 * FB = (uint2 *) (dst + nh*48);
    Q[2*(size_t) c]     = make_uint4(q[0], q[1], q[2], q[3]);
    Q[2*(size_t) c + 1] = make_uint4(q[4], q[5], q[6], q[7]);
    FA[c] = make_uint4(F[0], F[1], F[2], F[3]);
    FB[c] = make_uint2(F[4], F[5]);
}

static __global__ void repack_iq3_xxs(const block_iq3_xxs * __restrict__ src, uint8_t * __restrict__ dst, const int N, const int K, const int RG,
        const int CPS = 0) {
    const size_t t = blockIdx.x*(size_t) blockDim.x + threadIdx.x;
    const uint32_t items = K/128;
    const size_t nh = (size_t) N*items;
    if (t >= nh) return;
    const uint32_t row = (uint32_t) (t / items), h = (uint32_t) (t % items);
    const block_iq3_xxs & b = src[(size_t) row*(K/256) + h/2];
    const int half = (int) (h & 1);
    uint32_t q[8];
    memcpy(q, b.qs + 32*half, 32);
    uint32_t A[4];
    for (int j = 0; j < 4; ++j) {
        const int jg = 4*half + j;
        uint32_t aux; memcpy(&aux, b.qs + 64 + 4*jg, 4);
        const uint32_t sc = aux >> 28;
        uint32_t w = 0;
        for (int m = 0; m < 4; ++m) w |= (((aux >> (7*m)) & 0x7Fu) | (((sc >> m) & 1u) << 7)) << (8*m);
        A[j] = w;
    }
    const uint32_t U = (uint32_t) (N/RG);
    const uint32_t c = chunk_index_sm(row, h, items, (uint32_t) RG, U, (uint32_t) CPS);
    uint4 * Q = (uint4 *) dst; uint4 * AA = (uint4 *) (dst + nh*32); uint16_t * D = (uint16_t *) (dst + nh*48);
    Q[2*(size_t) c]     = make_uint4(q[0], q[1], q[2], q[3]);
    Q[2*(size_t) c + 1] = make_uint4(q[4], q[5], q[6], q[7]);
    AA[c] = make_uint4(A[0], A[1], A[2], A[3]);
    if (half == 0) { uint16_t d; memcpy(&d, &b.d, 2); D[chunk_index_sm(row, h/2, items/2, (uint32_t) RG, U, (uint32_t) (CPS/2))] = d; }
}

// ------------------------------------------------------------------------------------------------ type traits
// item = P sub-blocks (P = 4: whole chunk, P = 2: part p of the chunk) of one row.
// TOFF = byte offset of the type's table in LDS (absolute address).
struct iq3s_t {
    static constexpr ggml_type type = GGML_TYPE_IQ3_S;
    static constexpr int TAB_WORDS = 8192;
    // ld: the loaded registers only (no ALU on loaded data before the prefetch barrier); dec: decoded item
    template <int P> struct ld { uint4 q0, q1, fa; uint2 fb; int p; };
    template <int P> struct dec { uint32_t q[2*P]; uint32_t W[P == 4 ? 6 : 3]; uint32_t d, sc; };
    static __device__ __forceinline__ void stage_table(uint32_t * tab, const int tid, const int nt) {
        for (int e = tid; e < 8192; e += nt) tab[e] = apply_signs(iq3s_grid[e & 511], (uint32_t) e >> 9);
    }
    template <int P>
    static __device__ __forceinline__ ld<P> load(const uint8_t * __restrict__ base, const size_t nh, const uint32_t c, const uint32_t cb, const int p) {
        (void) cb;
        ld<P> r;
        r.fa = *(const uint4 *) (base + nh*32 + c*16u);
        r.fb = *(const uint2 *) (base + nh*48 + c*8u);
        if constexpr (P == 4) {
            r.q0 = *(const uint4 *) (base + c*32u);
            r.q1 = *(const uint4 *) (base + c*32u + 16u);
        } else {
            r.q0 = *(const uint4 *) (base + c*32u + 16u*p);
            r.q1 = r.q0;
        }
        r.p = p;
        return r;
    }
    template <int P>
    static __device__ __forceinline__ dec<P> prep(const ld<P> & l) {
        dec<P> r;
        r.q[0] = l.q0.x; r.q[1] = l.q0.y; r.q[2] = l.q0.z; r.q[3] = l.q0.w;
        if constexpr (P == 4) {
            r.q[4] = l.q1.x; r.q[5] = l.q1.y; r.q[6] = l.q1.z; r.q[7] = l.q1.w;
            r.W[0] = l.fa.x; r.W[1] = l.fa.y; r.W[2] = l.fa.z; r.W[3] = l.fa.w; r.W[4] = l.fb.x; r.W[5] = l.fb.y;
            r.sc = l.fb.y >> 16;
        } else {
            r.W[0] = l.p ? l.fa.w : l.fa.x; r.W[1] = l.p ? l.fb.x : l.fa.y; r.W[2] = l.p ? l.fb.y : l.fa.z;
            r.sc = l.fb.y >> (16 + 8*l.p);
        }
        r.d = l.fa.z >> 16;
        return r;
    }
    template <int P> static __device__ __forceinline__ float dw(const dec<P> & r) { return h2f(r.d); }
    template <int P> static __device__ __forceinline__ uint32_t fold(const ld<P> & r) {
        return r.q0.x ^ r.q0.y ^ r.q0.z ^ r.q0.w ^ r.q1.x ^ r.q1.y ^ r.q1.z ^ r.q1.w ^ r.fa.x ^ r.fa.y ^ r.fa.z ^ r.fa.w ^ r.fb.x ^ r.fb.y;
    }
    // scaled integer sum of sub-block j (0..P-1 of the item) against the 8 y words
    template <int P, int TOFF>
    static __device__ __forceinline__ int sub(const dec<P> & r, const int j, const int (&y)[8]) {
        const int wb = P == 4 ? 3*(j >> 1) : 0;    // part window
        int s = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int bit = 5*(8*(j & 1) + k), w = wb + (bit >> 5), o = bit & 31;
            const uint32_t f = o <= 27 ? (r.W[w] >> o) : __builtin_amdgcn_alignbit(r.W[w + 1], r.W[w], o);
            const uint32_t addr = ((f & 31u) << 10) | bx4(r.q[2*j + (k >> 2)], k & 3);
            s = dp4a(tab_ld(addr + TOFF), y[k], s);
        }
        const int sc = (int) ((r.sc >> (4*j)) & 0xFu);
        return __mul24(s, 1 + 2*sc);
    }
};

struct iq3xxs_t {
    static constexpr ggml_type type = GGML_TYPE_IQ3_XXS;
    static constexpr int TAB_WORDS = 4096;
    template <int P> struct ld { uint4 q0, q1, a; uint32_t d; int p; };
    template <int P> struct dec { uint32_t q[2*P]; uint32_t a[P]; uint32_t d; };
    static __device__ __forceinline__ void stage_table(uint32_t * tab, const int tid, const int nt) {
        for (int e = tid; e < 4096; e += nt) tab[e] = apply_signs(iq3xxs_grid[e & 255], (uint32_t) e >> 8);
    }
    template <int P>
    static __device__ __forceinline__ ld<P> load(const uint8_t * __restrict__ base, const size_t nh, const uint32_t c, const uint32_t cb, const int p) {
        ld<P> r;
        r.a = *(const uint4 *) (base + nh*32 + c*16u);
        if constexpr (P == 4) {
            r.q0 = *(const uint4 *) (base + c*32u);
            r.q1 = *(const uint4 *) (base + c*32u + 16u);
        } else {
            r.q0 = *(const uint4 *) (base + c*32u + 16u*p);
            r.q1 = r.q0;
        }
        r.d = *(const uint16_t *) (base + nh*48 + cb*2u);
        r.p = p;
        return r;
    }
    template <int P>
    static __device__ __forceinline__ dec<P> prep(const ld<P> & l) {
        dec<P> r;
        r.q[0] = l.q0.x; r.q[1] = l.q0.y; r.q[2] = l.q0.z; r.q[3] = l.q0.w;
        if constexpr (P == 4) {
            r.q[4] = l.q1.x; r.q[5] = l.q1.y; r.q[6] = l.q1.z; r.q[7] = l.q1.w;
            r.a[0] = l.a.x; r.a[1] = l.a.y; r.a[2] = l.a.z; r.a[3] = l.a.w;
        } else {
            r.a[0] = l.p ? l.a.z : l.a.x; r.a[1] = l.p ? l.a.w : l.a.y;
        }
        r.d = l.d;
        return r;
    }
    template <int P> static __device__ __forceinline__ float dw(const dec<P> & r) { return h2f(r.d); }
    template <int P> static __device__ __forceinline__ uint32_t fold(const ld<P> & r) {
        return r.q0.x ^ r.q0.y ^ r.q0.z ^ r.q0.w ^ r.q1.x ^ r.q1.y ^ r.q1.z ^ r.q1.w ^ r.a.x ^ r.a.y ^ r.a.z ^ r.a.w ^ r.d;
    }
    template <int P, int TOFF>
    static __device__ __forceinline__ int sub(const dec<P> & r, const int j, const int (&y)[8]) {
        const uint32_t W = r.a[j];
        // full sign bytes: bit 7 of group m = parity of its 7 stored sign bits
        const uint32_t x = W & 0x7F7F7F7Fu;
        uint32_t p = x ^ (x >> 4);
        p ^= p >> 2;
        p ^= p >> 1;
        const uint32_t Wf = x | ((p & 0x01010101u) << 7);
        int s = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint32_t addr = (((Wf >> (4*k)) & 15u) << 10) | bx4(r.q[2*j + (k >> 2)], k & 3);
            s = dp4a(tab_ld(addr + TOFF), y[k], s);
        }
        const int ls = dp4a((int) ((W >> 7) & 0x01010101u), 0x08040201, 0);
        return sdiv<1>(__mul24(ls, s) + sdiv<1>(s));
    }
};

// ------------------------------------------------------------------------------------------------ activation records
// record per half-block: {qs[32 words], d[4] (float)} = 36 words; one thread per 32 activations.
static constexpr int YREC_W = 36;
static __device__ __forceinline__ void stage_y(int * __restrict__ rec, const block_q8_1 * __restrict__ yg, const int K, const int tid, const int nt) {
    for (int s = tid; s < K/32; s += nt) {
        const uint8_t * b = (const uint8_t *) (yg + s);
        const uint4 q0 = *(const uint4 *) (b + 4), q1 = *(const uint4 *) (b + 20);
        const uint32_t ds = *(const uint32_t *) b;
        int * r = rec + (s >> 2)*YREC_W + 8*(s & 3);
        *(int4 *) r       = make_int4((int) q0.x, (int) q0.y, (int) q0.z, (int) q0.w);
        *(int4 *) (r + 4) = make_int4((int) q1.x, (int) q1.y, (int) q1.z, (int) q1.w);
        ((float *) rec)[(s >> 2)*YREC_W + 32 + (s & 3)] = h2f(ds & 0xFFFFu);
    }
}

// Prologue with all global loads issued up front (activation blocks + grid words), then all LDS writes:
// one L2 round trip instead of three serialized ones. TB is staged only if TWO (gate/up of different types).
template <class T> static __device__ __forceinline__ uint32_t grid_word(const int i) {
    if constexpr (T::TAB_WORDS == 8192) return iq3s_grid[i]; else return iq3xxs_grid[i];
}
template <class TA, class TB, bool TWO, int NT>
static __device__ __forceinline__ void prologue(uint32_t * smem, const int toff_b, const int yoff, const block_q8_1 * __restrict__ yg,
        const int K, const int tid) {
    constexpr int GA = TA::TAB_WORDS/16, GB = TWO ? TB::TAB_WORDS/16 : 1;
    if (NT % 512 == 0 && TA::TAB_WORDS % NT == 0 && (!TWO || TB::TAB_WORDS % NT == 0) && K/32 <= NT) {
        const int nrec = K/32;
        const int sy = min(tid, nrec - 1);   // unconditional (clamped) loads and writes: no branches, nothing sinks
        const uint8_t * b = (const uint8_t *) (yg + sy);
        const uint4 q0 = *(const uint4 *) (b + 4), q1 = *(const uint4 *) (b + 20);
        const uint32_t ds = *(const uint32_t *) b;
        const uint32_t ga = grid_word<TA>(tid % GA);
        uint32_t gb = 0;
        if constexpr (TWO) gb = grid_word<TB>(tid % GB);
#pragma unroll
        for (int i = 0; i < TA::TAB_WORDS/NT; ++i) { const int e = tid + NT*i; smem[e] = apply_signs(ga, (uint32_t) (e / GA)); }
        if constexpr (TWO) {
#pragma unroll
            for (int i = 0; i < TB::TAB_WORDS/NT; ++i) { const int e = tid + NT*i; smem[toff_b/4 + e] = apply_signs(gb, (uint32_t) (e / GB)); }
        }
        int * rec = (int *) (smem + yoff/4);
        {   // threads >= nrec rewrite record nrec-1 with identical data (benign)
            int * r = rec + (sy >> 2)*YREC_W + 8*(sy & 3);
            *(int4 *) r       = make_int4((int) q0.x, (int) q0.y, (int) q0.z, (int) q0.w);
            *(int4 *) (r + 4) = make_int4((int) q1.x, (int) q1.y, (int) q1.z, (int) q1.w);
            ((float *) rec)[(sy >> 2)*YREC_W + 32 + (sy & 3)] = h2f(ds & 0xFFFFu);
        }
    }
    if (!(NT % 512 == 0 && TA::TAB_WORDS % NT == 0 && (!TWO || TB::TAB_WORDS % NT == 0)) || K/32 > NT) {
        TA::stage_table(smem, tid, NT);
        if constexpr (TWO) TB::stage_table(smem + toff_b/4, tid, NT);
        stage_y((int *) (smem + yoff/4), yg, K, tid, NT);
    }
}

// Split prologue: all global loads of the staging (activation block + grid word per thread) first (pro_load), the LDS
// writes later (pro_store). The kernels issue the weight loads of their first item in between, so the staging overlaps
// the first HBM round trip (loads return in order: the staging loads, issued first, are consumed while the weight
// prefetch is still in flight; prefetching before the staging loads would serialize them behind the weights).
// Requires NT % 512 == 0, TAB_WORDS % NT == 0 and K/32 <= NT (host checks: pro_split_ok).
template <class TA, class TB, bool TWO> struct pro_regs { uint4 q0, q1; uint32_t ds, ga, gb; };
static __device__ __forceinline__ void pro_fence() {
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_amdgcn_sched_barrier(0);
#endif
}
template <class TA, class TB, bool TWO, int NT>
static __host__ __device__ constexpr bool pro_split_cfg_ok() {
    return NT >= TA::TAB_WORDS/16 && (!TWO || NT >= TB::TAB_WORDS/16);
}
// sign-expanded table from one grid word per thread (thread t: grid word t % GA). Any NT >= GA: with NT % GA != 0
// the first (NT/GA)*GA threads write the 16 sign variants of their grid word in NT/GA parallel groups.
template <class T, int NT>
static __device__ __forceinline__ void tab_store(uint32_t * tab, const uint32_t gw, const int tid) {
    constexpr int GA = T::TAB_WORDS/16;
    constexpr int SL = NT/GA > 0 ? NT/GA : 1;
    if constexpr (T::TAB_WORDS % NT == 0 && NT % GA == 0 && (SL & (SL - 1)) == 0) {
        // diet-gemv: entry e = tid + NT*i has sign index e/GA = s0 + SL*i with s0 = tid/GA < SL (SL a power of 2): s0 and
        // SL*i set disjoint bits, i.e. negate disjoint bytes. All grid bytes are >= 1 (IQ3_S 1..15, IQ3_XXS 4..62), so
        // apply_signs has no borrow between bytes and the bytes of SL*i can be negated on top of apply_signs(gw, s0) with
        // compile-time constants: 2 instructions per entry instead of 7, the same table words.
        const uint32_t base = apply_signs(gw, (uint32_t) (tid / GA));
        tab[tid] = base;
#pragma unroll
        for (int i = 1; i < T::TAB_WORDS/NT; ++i) {
            const uint32_t ones = ((uint32_t) (SL*i) * 0x00204081u) & 0x01010101u;   // constant after unrolling
            tab[tid + NT*i] = (base - ones) ^ (ones * 255u);
        }
    } else if constexpr (T::TAB_WORDS % NT == 0 && NT % GA == 0) {
#pragma unroll
        for (int i = 0; i < T::TAB_WORDS/NT; ++i) { const int e = tid + NT*i; tab[e] = apply_signs(gw, (uint32_t) (e / GA)); }
    } else {
        static_assert(NT >= GA, "table staging needs one thread per grid word");
        constexpr int TPG = NT/GA, SPT = (16 + TPG - 1)/TPG;
        if (tid < TPG*GA) {
            const int g = tid % GA, s0 = tid / GA;
#pragma unroll
            for (int k = 0; k < SPT; ++k) {
                const int s = s0 + TPG*k;
                if (s < 16) tab[s*GA + g] = apply_signs(gw, (uint32_t) s);
            }
        }
    }
}
// CY (conditional activation loads): only the K/32 threads that own a q8_1 block load it (the other waves issue no
// activation loads at all: no hot spot on the last block, no extra address-unit work). Without CY (needed when a
// prefetch follows: the load count must not depend on the path) the other threads load another valid block.
// ORD (r4-ffndown, default via PF 7 / multi-job ORD): the grid words are loaded first - their address needs no kernel
// argument, so they are in flight while the kernel arguments arrive - and the activation address is computed
// unconditionally (all kernel-argument loads are issued together at the kernel start). Without ORD the compiler issues
// the grid-word load only after the activation loads have returned (it moves pro_store's fp16 -> fp32 conversion of the
// owner's d into the conditional load block, with a wait there) and loads the activation pointer only inside that block:
// kernel args -> activation pointer -> activation -> grid word = four serialized round trips at every kernel start.
// diet-gemv: the same pointer, opaque to the load/store vectorizer. Merged with the two 16-byte qs loads of a q8_1 block,
// the 4-byte ds load made the backend split the 36 bytes at 16-byte boundaries of the block ({ds, qs0..2}, {qs3..6},
// {qs7}), so the qs words had to be shuffled into place (8 v_mov) for the ds_write_b128 of the activation record.
// The load goes through an explicit global (addrspace 1) pointer: the laundered generic pointer would become a flat load.
static __device__ __forceinline__ uint32_t ld_u32_opaque(const uint8_t * p) {
#if defined(__HIP_DEVICE_COMPILE__)
    asm("" : "+v"(p));
    return *(const __attribute__((address_space(1))) uint32_t *) (size_t) p;
#else
    return *(const uint32_t *) p;
#endif
}
template <class TA, class TB, bool TWO, int NT, bool CY = false, bool ORD = false>
static __device__ __forceinline__ void pro_load(pro_regs<TA, TB, TWO> & r, const block_q8_1 * __restrict__ yg, const int K, const int tid) {
    constexpr int GA = TA::TAB_WORDS/16;
    if constexpr (ORD) {
        r.ga = grid_word<TA>(tid % GA);
        r.gb = 0;
        if constexpr (TWO) { constexpr int GB = TB::TAB_WORDS/16; r.gb = grid_word<TB>(tid % GB); }
        pro_fence();
#if defined(__HIP_DEVICE_COMPILE__)
        asm volatile("" :: "s"(yg), "s"(K));   // request both kernel arguments together (one scalar round trip)
#endif
        const int nrec = K/32;
        const int sy = CY ? min(tid, nrec - 1) : tid % nrec;
        const uint8_t * b = (const uint8_t *) (yg + sy);
        // diet-gemv: no zero fill for the non-owners (9 v_mov in the prologue) - every consumer of q0/q1/ds (pro_store, the
        // pair kernel's put_y) is guarded by the same tid < K/32
        if (!CY || tid < nrec) {
            r.q0 = *(const uint4 *) (b + 4);
            r.q1 = *(const uint4 *) (b + 20);
            r.ds = ld_u32_opaque(b);   // not merged with the qs loads: qs stays in ds_write_b128 order
        }
        return;
    }
    if (!CY || tid < K/32) {
        const int sy = CY ? tid : tid % (K/32);
        const uint8_t * b = (const uint8_t *) (yg + sy);
        r.q0 = *(const uint4 *) (b + 4);
        r.q1 = *(const uint4 *) (b + 20);
        r.ds = *(const uint32_t *) b;
    } else {
        r.q0 = make_uint4(0, 0, 0, 0); r.q1 = r.q0; r.ds = 0;
    }
    r.ga = grid_word<TA>(tid % GA);
    r.gb = 0;
    if constexpr (TWO) { constexpr int GB = TB::TAB_WORDS/16; r.gb = grid_word<TB>(tid % GB); }
}
template <class TA, class TB, bool TWO, int NT>
static __device__ __forceinline__ void pro_store(const pro_regs<TA, TB, TWO> & r, uint32_t * smem, const int toff_b, const int yoff,
        const int K, const int tid) {
    tab_store<TA, NT>(smem, r.ga, tid);
    if constexpr (TWO) tab_store<TB, NT>(smem + toff_b/4, r.gb, tid);
    if (tid < K/32) {
        const int sy = tid;
        int * rec = (int *) (smem + yoff/4);
        int * rr = rec + (sy >> 2)*YREC_W + 8*(sy & 3);
        *(int4 *) rr       = make_int4((int) r.q0.x, (int) r.q0.y, (int) r.q0.z, (int) r.q0.w);
        *(int4 *) (rr + 4) = make_int4((int) r.q1.x, (int) r.q1.y, (int) r.q1.z, (int) r.q1.w);
        ((float *) rec)[(sy >> 2)*YREC_W + 32 + (sy & 3)] = h2f(r.ds & 0xFFFFu);
    }
}

static __device__ __forceinline__ void sched_fence() {
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_amdgcn_sched_barrier(0);
#endif
}

// ------------------------------------------------------------------------------------------------ reduction
template <int L> static __device__ __forceinline__ float group_sum(float v) {
#if defined(__HIP_DEVICE_COMPILE__)
    // butterflies that stay inside aligned groups of L lanes: quad_perm xor1, xor2, row_half_mirror (8), row_mirror (16)
    if constexpr (L >= 2)  v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xF, 0xF, false));   // quad_perm [1,0,3,2]
    if constexpr (L >= 4)  v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xF, 0xF, false));   // quad_perm [2,3,0,1]
    if constexpr (L >= 8)  v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x141, 0xF, 0xF, false));  // row_half_mirror
    if constexpr (L >= 16) v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x140, 0xF, 0xF, false));  // row_mirror
    if constexpr (L >= 32) v += __shfl_xor(v, 16, 64);
    if constexpr (L >= 64) v += __shfl_xor(v, 32, 64);
    return v;
#else
    return v;
#endif
}

// ------------------------------------------------------------------------------------------------ kernel
// Persistent workgroups; y staged in LDS once per workgroup. A wave processes units of RG = G*R rows (G = 64/L lane
// groups of L lanes, R rows per group; in-unit row index rin = r*G + grp, matching the repack interleave RG).
// Lane l of a group handles items it = l, l+L, ... (item = P sub-blocks). Units are distributed round-robin over waves.
// TB != none_t: fused gate/up SwiGLU (TA = gate type / weights wa, TB = up type / weights wb): dst = silu(gate.y)*(up.y).
// PF: software pipelining of the weight loads (next item loaded before the current one is computed).
struct none_t {
    static constexpr int TAB_WORDS = 0;
    template <int P> struct ld {};
    template <int P> struct dec {};
    template <int P> static __device__ dec<P> prep(const ld<P> &) { return dec<P>{}; }
    template <int P> static __device__ uint32_t fold(const ld<P> &) { return 0; }
};

// SM > 0: weights in the step-major layout with CPS = SM chunks per step (SM must equal L/(4/P)*R... i.e. the chunks
// one wave step of a unit covers: SM = L*P/4 for R = 1)
template <class TA, class TB, int P, int L, int R, int NW, int PF, int DBG = 0, int MINW = 1, int SM = 0>   // DBG: 1 = loads only, 2 = compute only
__launch_bounds__(NW*64, MINW)   /* MINW = min waves per SIMD (8 with NW=16: two workgroups per CU) */
__global__ void gemv(const uint8_t * __restrict__ wa, const uint8_t * __restrict__ wb, const block_q8_1 * __restrict__ yg,
        float * __restrict__ dst, const int K, const int N, const float * __restrict__ residual) {
    static_assert(SM == 0 || (R == 1 && SM == L*P/4), "step-major layout: SM = chunks per wave step");
    constexpr bool GLU  = !std::is_same<TB, none_t>::value;
    constexpr bool SAME = std::is_same<TA, TB>::value;
    constexpr int TOFF_A = 0;
    constexpr int TOFF_B = (GLU && !SAME) ? TA::TAB_WORDS*4 : 0;
    constexpr int YOFF   = TA::TAB_WORDS*4 + ((GLU && !SAME) ? TB::TAB_WORDS*4 : 0);
    constexpr int G  = 64/L;
    constexpr int RG = G*R;
    constexpr int IPC = 4/P;                     // items per chunk
    extern __shared__ uint32_t smem[];
#if defined(__HIP_DEVICE_COMPILE__)
    if ((uint32_t) (size_t) (__attribute__((address_space(3))) uint32_t *) smem != 0) __builtin_trap();
#endif
    const int tid  = threadIdx.y*64 + threadIdx.x;
    const int lane = threadIdx.x % L, grp = threadIdx.x / L;
    const int items = K / 128;                   // chunks per row
    const int ipr   = items*IPC;                 // items per row
    const size_t nh = (size_t) N*items;
    const int U = N / RG;
    const int ustep = gridDim.x*NW;
    int u = threadIdx.y*gridDim.x + blockIdx.x;

    typename TA::template ld<P> ra[R];
    typename TB::template ld<P> rb[GLU ? R : 1];
    auto load_item = [&](const int uu, const int it) {
        const int h = it / IPC, p = it % IPC;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint32_t rin = (uint32_t) (r*G + grp);
            uint32_t c, cb;
            if constexpr (SM == 0) {
                c  = ((uint32_t) uu*(uint32_t) items + (uint32_t) h)*RG + rin;
                cb = ((uint32_t) uu*(uint32_t) (items/2) + (uint32_t) (h >> 1))*RG + rin;
            } else {
                const uint32_t hb = (uint32_t) h >> 1;
                c  = ((((uint32_t) h / SM)*(uint32_t) U + (uint32_t) uu)*SM + (uint32_t) h % SM)*RG + rin;
                cb = (((hb / (SM/2))*(uint32_t) U + (uint32_t) uu)*(SM/2) + hb % (SM/2))*RG + rin;
            }
            ra[r] = TA::template load<P>(wa, nh, c, cb, p);
            if constexpr (GLU) rb[r] = TB::template load<P>(wb, nh, c, cb, p);
        }
    };
    bool first = true;
    if constexpr (PF == 3) {
        // PF 3: staging loads, then the first item's weight loads, then the LDS writes (split prologue)
        static_assert(pro_split_cfg_ok<TA, TB, GLU && !SAME, NW*64>(), "split prologue config");
        pro_regs<TA, TB, GLU && !SAME> pr;
        pro_load<TA, TB, GLU && !SAME, NW*64>(pr, yg, K, tid);
        sched_fence();
        load_item(min(u, U - 1), lane);   // unconditional (clamped): a conditional load makes the staging wait for it (vmcnt(0))
        sched_fence();
        pro_store<TA, TB, GLU && !SAME, NW*64>(pr, smem, TOFF_B, YOFF, K, tid);
    } else if constexpr (PF == 4 || PF == 5 || PF == 7) {
        // PF 4: split prologue without prefetch, activation loads spread over the blocks (no hot spot);
        // PF 5: same, but only the owners of the activation blocks load them (conditional loads);
        // PF 7: PF 5 with the grid words loaded first and all kernel-argument loads up front (pro_load ORD)
        static_assert(pro_split_cfg_ok<TA, TB, GLU && !SAME, NW*64>(), "split prologue config");
        pro_regs<TA, TB, GLU && !SAME> pr;
        pro_load<TA, TB, GLU && !SAME, NW*64, PF == 5 || PF == 7, PF == 7>(pr, yg, K, tid);
        pro_store<TA, TB, GLU && !SAME, NW*64>(pr, smem, TOFF_B, YOFF, K, tid);
    } else {
        if ((PF || DBG == 2) && u < U) load_item(u, lane);   // PF 1: pipelined; PF 2: only the first item is prefetched (before the prologue)
        prologue<TA, TB, GLU && !SAME, NW*64>(smem, TOFF_B, YOFF, yg, K, tid);
    }
    __syncthreads();
    if constexpr (DBG == 4) {   // prologue only (timing)
        if (tid == 0 && smem[YOFF/4 + 5] == 0x12345678u) dst[0] = 1.0f;
        return;
    }

    for (; u < U; u += ustep) {
        float acc[R], accb[GLU ? R : 1];
#pragma unroll
        for (int r = 0; r < R; ++r) { acc[r] = 0.0f; if constexpr (GLU) accb[r] = 0.0f; }
        for (int it = lane; it < ipr; it += L) {
            if ((PF == 0 || PF == 4 || PF == 5 || PF == 7 || ((PF == 2 || PF == 3) && !first)) && DBG != 2) load_item(u, it);
            first = false;
            typename TA::template ld<P> ca[R];
            typename TB::template ld<P> cb[GLU ? R : 1];
#pragma unroll
            for (int r = 0; r < R; ++r) { ca[r] = ra[r]; if constexpr (GLU) cb[r] = rb[r]; }
            if (PF == 1 && DBG != 2) {
                // next item of this unit, or the first item of the next unit (clamped, always valid)
                int nit = it + L, nu = u;
                if (nit >= ipr) { nit = lane; nu = min(u + ustep, U - 1); }
                load_item(nu, nit);
#if defined(__HIP_DEVICE_COMPILE__)
                __builtin_amdgcn_sched_barrier(0);   // keep the prefetch loads ahead of the compute
#endif
            }
            if constexpr (DBG == 1) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    acc[r] += __uint_as_float(TA::template fold<P>(ca[r]) & 0x007FFFFFu);
                    if constexpr (GLU) accb[r] += __uint_as_float(TB::template fold<P>(cb[r]) & 0x007FFFFFu);
                }
                continue;
            }
            const int h = it / IPC, p = it % IPC;
            const uint32_t ya = YOFF + (uint32_t) h*(YREC_W*4) + (uint32_t) p*(32*P);   // qs words of the item
            const uint32_t yd_addr = YOFF + (uint32_t) h*(YREC_W*4) + 128 + (uint32_t) p*(4*P);
            float dy4[P];
#if defined(__HIP_DEVICE_COMPILE__)
            if constexpr (P == 4) {
                const float4 v = *(__attribute__((address_space(3))) const float4 *) (size_t) yd_addr;
                dy4[0] = v.x; dy4[1] = v.y; dy4[2] = v.z; dy4[3] = v.w;
            } else {
                const float2 v = *(__attribute__((address_space(3))) const float2 *) (size_t) yd_addr;
                dy4[0] = v.x; dy4[1] = v.y;
            }
#else
            for (int j = 0; j < P; ++j) dy4[j] = 0.0f;
#endif
            typename TA::template dec<P> da[R];
            typename TB::template dec<P> db[GLU ? R : 1];
            float dwa[R], dwb[GLU ? R : 1];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                da[r] = TA::template prep<P>(ca[r]); dwa[r] = TA::template dw<P>(da[r]);
                if constexpr (GLU) { db[r] = TB::template prep<P>(cb[r]); dwb[r] = TB::template dw<P>(db[r]); }
            }
#pragma unroll
            for (int j = 0; j < P; ++j) {
                int y[8];
#if defined(__HIP_DEVICE_COMPILE__)
                const int4 y0 = *(__attribute__((address_space(3))) const int4 *) (size_t) (ya + 32*j);
                const int4 y1 = *(__attribute__((address_space(3))) const int4 *) (size_t) (ya + 32*j + 16);
                y[0] = y0.x; y[1] = y0.y; y[2] = y0.z; y[3] = y0.w; y[4] = y1.x; y[5] = y1.y; y[6] = y1.z; y[7] = y1.w;
#else
                for (int k = 0; k < 8; ++k) y[k] = 0;
#endif
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    acc[r] += (dwa[r]*dy4[j]) * (float) TA::template sub<P, TOFF_A>(da[r], j, y);
                    if constexpr (GLU) accb[r] += (dwb[r]*dy4[j]) * (float) TB::template sub<P, TOFF_B>(db[r], j, y);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = u*RG + r*G + grp;
            const float v = group_sum<L>(acc[r]);
            if constexpr (GLU) {
                const float vb = group_sum<L>(accb[r]);
                if (lane == 0) dst[row] = vb * silu(v);
            } else {
                if (lane == 0) dst[row] = residual ? v + residual[row] : v;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ multi-job kernel
// One launch for up to MJ_MAX GEMVs that read the same activation (qkv + z, k + v, ...): the jobs' units (RG = 64/L rows
// each, R = 1) are concatenated, every job has its own weights (type T0 or T1, tsel), N, output and optional residual.
// One prologue (codebook tables + activation) for all of them, the staging overlapped with the first weight loads.
// KS > 1: K-split for small N - every unit is processed by KS waves of one workgroup (piece q = items
// [q*ipr/KS, (q+1)*ipr/KS)), partial row sums combined through LDS in piece order (deterministic).
static constexpr int MJ_MAX = 3;
struct mj_job {
    const uint8_t * w;      // SoA copy (chunk interleave RG)
    float         * dst;
    const float   * res;    // residual (dst = dot + res) or nullptr
    uint32_t        nh;     // chunks of the matrix = N*K/128
    int             ubeg;   // first unit of the job in the concatenated unit space (unused jobs: INT_MAX)
    int             tsel;   // 0: T0, 1: T1
    int             U;      // units of the job (N/RG), for the step-major layout
};
struct mj_args {
    mj_job job[MJ_MAX];
    int    njob;
    int    utot;
};
static __device__ __forceinline__ int mj_find(const mj_args & a, const int u) {
    return (u >= a.job[1].ubeg) + (u >= a.job[2].ubeg);
}
template <class T, int P, int RG, int SM = 0>
static __device__ __forceinline__ typename T::template ld<P> mj_load(const uint8_t * __restrict__ w, const uint32_t nh, const int lu,
        const int items, const int grp, const int it, const int U = 0) {
    constexpr int IPC = 4/P;
    const int h = it / IPC, p = it % IPC;
    uint32_t c, cb;
    if constexpr (SM == 0) {
        c  = ((uint32_t) lu*(uint32_t) items + (uint32_t) h)*RG + (uint32_t) grp;
        cb = ((uint32_t) lu*(uint32_t) (items/2) + (uint32_t) (h >> 1))*RG + (uint32_t) grp;
    } else {
        const uint32_t hb = (uint32_t) h >> 1;
        c  = ((((uint32_t) h / SM)*(uint32_t) U + (uint32_t) lu)*SM + (uint32_t) h % SM)*RG + (uint32_t) grp;
        cb = (((hb / (SM/2))*(uint32_t) U + (uint32_t) lu)*(SM/2) + hb % (SM/2))*RG + (uint32_t) grp;
    }
    return T::template load<P>(w, nh, c, cb, p);
}
template <class T, int TOFF, int P>
static __device__ __forceinline__ void mj_item(float & acc, const typename T::template ld<P> & c, const int it, const uint32_t yoff) {
    constexpr int IPC = 4/P;
    const int h = it / IPC, p = it % IPC;
    const uint32_t ya      = yoff + (uint32_t) h*(YREC_W*4) + (uint32_t) p*(32*P);
    const uint32_t yd_addr = yoff + (uint32_t) h*(YREC_W*4) + 128 + (uint32_t) p*(4*P);
    float dy4[P];
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr (P == 4) {
        const float4 v = *(__attribute__((address_space(3))) const float4 *) (size_t) yd_addr;
        dy4[0] = v.x; dy4[1] = v.y; dy4[2] = v.z; dy4[3] = v.w;
    } else {
        const float2 v = *(__attribute__((address_space(3))) const float2 *) (size_t) yd_addr;
        dy4[0] = v.x; dy4[1] = v.y;
    }
#else
    for (int j = 0; j < P; ++j) dy4[j] = 0.0f;
#endif
    const typename T::template dec<P> d = T::template prep<P>(c);
    const float dw = T::template dw<P>(d);
#pragma unroll
    for (int j = 0; j < P; ++j) {
        int y[8];
#if defined(__HIP_DEVICE_COMPILE__)
        const int4 y0 = *(__attribute__((address_space(3))) const int4 *) (size_t) (ya + 32*j);
        const int4 y1 = *(__attribute__((address_space(3))) const int4 *) (size_t) (ya + 32*j + 16);
        y[0] = y0.x; y[1] = y0.y; y[2] = y0.z; y[3] = y0.w; y[4] = y1.x; y[5] = y1.y; y[6] = y1.z; y[7] = y1.w;
#else
        for (int k = 0; k < 8; ++k) y[k] = 0;
#endif
        acc += (dw*dy4[j]) * (float) T::template sub<P, TOFF>(d, j, y);
    }
}
// items it0+lane, it0+lane+L, .. < it1 of local unit lu; the first one from the prefetched registers if use_pf
template <class T, int TOFF, int P, int L, int RG, int SM = 0>
static __device__ __forceinline__ float mj_unit(const uint8_t * __restrict__ w, const uint32_t nh, const int lu, const int items,
        const int lane, const int grp, const int it0, const int it1, const uint32_t yoff,
        const bool use_pf, const typename T::template ld<P> & pf, const int U = 0) {
    float acc = 0.0f;
    int it = it0 + lane;
    if (use_pf) {
        if (it < it1) mj_item<T, TOFF, P>(acc, pf, it, yoff);
        it += L;
    }
    for (; it < it1; it += L) {
        const typename T::template ld<P> c = mj_load<T, P, RG, SM>(w, nh, lu, items, grp, it, U);
        mj_item<T, TOFF, P>(acc, c, it, yoff);
    }
    return acc;
}

template <class T0, class T1, int P, int L, int NW, int KS, int PFM = 1, int SM = 0, bool ORD = false>   // PFM: prefetch the first item under the staging; SM: step-major layout; ORD: pro_load order
__launch_bounds__(NW*64, 1)
__global__ void gemv_mj(const block_q8_1 * __restrict__ yg, const int K, const mj_args a) {
    constexpr bool HAS1 = !std::is_same<T1, none_t>::value;
    constexpr bool TWO  = HAS1 && !std::is_same<T0, T1>::value;
    constexpr int TOFF0 = 0;
    constexpr int TOFF1 = TWO ? T0::TAB_WORDS*4 : 0;
    constexpr int YOFF  = T0::TAB_WORDS*4 + (TWO ? T1::TAB_WORDS*4 : 0);
    constexpr int RG  = 64/L;
    constexpr int IPC = 4/P;
    constexpr int UPW = KS > 1 ? NW/KS : 1;       // units per workgroup round (K-split)
    static_assert(pro_split_cfg_ok<T0, T1, TWO, NW*64>(), "split prologue config");
    extern __shared__ uint32_t smem[];
#if defined(__HIP_DEVICE_COMPILE__)
    if ((uint32_t) (size_t) (__attribute__((address_space(3))) uint32_t *) smem != 0) __builtin_trap();
#endif
    const int tid  = threadIdx.y*64 + threadIdx.x;
    const int lane = threadIdx.x % L, grp = threadIdx.x / L;
    const int items = K / 128;
    const int ipr   = items*IPC;
#if defined(__HIP_DEVICE_COMPILE__)
    const int w = __builtin_amdgcn_readfirstlane(threadIdx.y);
#else
    const int w = threadIdx.y;
#endif
    int u, ustep, piece = 0, it0 = 0, it1 = ipr;
    bool wave_ok = true;
    if constexpr (KS == 1) {
        u = w*gridDim.x + blockIdx.x; ustep = gridDim.x*NW;
    } else {
        const int uw = w / KS;
        piece = w % KS;
        wave_ok = uw < UPW;
        u = blockIdx.x*UPW + uw; ustep = gridDim.x*UPW;
        it0 = (piece*ipr)/KS; it1 = ((piece + 1)*ipr)/KS;
    }

    pro_regs<T0, T1, TWO> pr;
    pro_load<T0, T1, TWO, NW*64, PFM == 0, ORD && PFM == 0>(pr, yg, K, tid);   // conditional activation loads unless a prefetch follows
    sched_fence();
    // prefetch of the first item, unconditional (clamped unit; both type paths issue 3 loads) so that the staging
    // below only waits for its own loads (vmcnt(3)) - a conditional prefetch makes it wait for everything
    typename T0::template ld<P> pf0;
    typename T1::template ld<P> pf1;
    if constexpr (PFM != 0) {
        const int uc = (wave_ok && u < a.utot) ? u : u % a.utot;   // idle waves: spread, not all on one unit
        const int j = mj_find(a, uc);
        const mj_job & J = a.job[j];
        const int itp = min(it0 + lane, it1 - 1);
        if (!HAS1 || J.tsel == 0) {
            pf0 = mj_load<T0, P, RG, SM>(J.w, J.nh, uc - J.ubeg, items, grp, itp, J.U);
        } else {
            if constexpr (HAS1) pf1 = mj_load<T1, P, RG, SM>(J.w, J.nh, uc - J.ubeg, items, grp, itp, J.U);
        }
    }
    sched_fence();
    pro_store<T0, T1, TWO, NW*64>(pr, smem, TOFF1, YOFF, K, tid);
    __syncthreads();

    bool first = true;
    if constexpr (KS == 1) {
        for (; u < a.utot; u += ustep) {
            const int j = mj_find(a, u);
            const mj_job & J = a.job[j];
            const int lu = u - J.ubeg;
            float acc;
            if (!HAS1 || J.tsel == 0) {
                acc = mj_unit<T0, TOFF0, P, L, RG, SM>(J.w, J.nh, lu, items, lane, grp, 0, ipr, YOFF, PFM && first, pf0, J.U);
            } else {
                if constexpr (HAS1) acc = mj_unit<T1, TOFF1, P, L, RG, SM>(J.w, J.nh, lu, items, lane, grp, 0, ipr, YOFF, PFM && first, pf1, J.U);
                else acc = 0.0f;
            }
            first = false;
            const float v = group_sum<L>(acc);
            if (lane == 0) {
                const int row = lu*RG + grp;
                J.dst[row] = J.res ? v + J.res[row] : v;
            }
        }
    } else {
        const uint32_t red = (uint32_t) YOFF + (uint32_t) items*(YREC_W*4);   // [NW][RG] partial sums
        for (int ub = blockIdx.x*UPW; ub < a.utot; ub += ustep) {            // workgroup-uniform trip count
            const int uu = ub + (w / KS);
            const bool act = wave_ok && uu < a.utot;
            const int j = act ? mj_find(a, uu) : 0;
            const mj_job & J = a.job[j];
            const int lu = uu - J.ubeg;
            if (act) {
                float acc;
                if (!HAS1 || J.tsel == 0) {
                    acc = mj_unit<T0, TOFF0, P, L, RG, SM>(J.w, J.nh, lu, items, lane, grp, it0, it1, YOFF, PFM && first, pf0, J.U);
                } else {
                    if constexpr (HAS1) acc = mj_unit<T1, TOFF1, P, L, RG, SM>(J.w, J.nh, lu, items, lane, grp, it0, it1, YOFF, PFM && first, pf1, J.U);
                    else acc = 0.0f;
                }
                const float v = group_sum<L>(acc);
                if (lane == 0) smem[red/4 + w*RG + grp] = __float_as_uint(v);
            }
            first = false;
            __syncthreads();
            if (act && piece == 0 && lane == 0) {
                float s = 0.0f;
#pragma unroll
                for (int q = 0; q < KS; ++q) s += __uint_as_float(smem[red/4 + (w + q)*RG + grp]);
                const int row = lu*RG + grp;
                J.dst[row] = J.res ? s + J.res[row] : s;
            }
            __syncthreads();
        }
    }
}
template <class T0, class T1>
static inline size_t lds_bytes_mj(int K, int KS, int NW, int L) {
    constexpr bool TWO = !std::is_same<T1, none_t>::value && !std::is_same<T0, T1>::value;
    return (size_t) T0::TAB_WORDS*4 + (TWO ? (size_t) T1::TAB_WORDS*4 : 0) + (size_t) (K/128)*YREC_W*4 +
        (KS > 1 ? (size_t) NW*(64/L)*4 : 0);
}

// ------------------------------------------------------------------------------------------------ batched kernel
// Same as gemv (R = 1), but each lane issues the loads of B items (it, it+L, .., it+(B-1)L, clamped) before decoding
// any of them: B x more bytes in flight per wave and one exposed memory latency per batch instead of per item.
template <class TA, class TB, int P, int L, int NW, int B>
__launch_bounds__(NW*64, 1)
__global__ void gemv_b(const uint8_t * __restrict__ wa, const uint8_t * __restrict__ wb, const block_q8_1 * __restrict__ yg,
        float * __restrict__ dst, const int K, const int N, const float * __restrict__ residual) {
    constexpr bool GLU  = !std::is_same<TB, none_t>::value;
    constexpr bool SAME = std::is_same<TA, TB>::value;
    constexpr int TOFF_A = 0;
    constexpr int TOFF_B = (GLU && !SAME) ? TA::TAB_WORDS*4 : 0;
    constexpr int YOFF   = TA::TAB_WORDS*4 + ((GLU && !SAME) ? TB::TAB_WORDS*4 : 0);
    constexpr int G  = 64/L;
    constexpr int RG = G;
    constexpr int IPC = 4/P;
    extern __shared__ uint32_t smem[];
#if defined(__HIP_DEVICE_COMPILE__)
    if ((uint32_t) (size_t) (__attribute__((address_space(3))) uint32_t *) smem != 0) __builtin_trap();
#endif
    const int tid  = threadIdx.y*64 + threadIdx.x;
    const int lane = threadIdx.x % L, grp = threadIdx.x / L;
    const int items = K / 128;
    const int ipr   = items*IPC;
    const size_t nh = (size_t) N*items;
    const int U = N / RG;
    const int ustep = gridDim.x*NW;
    int u = threadIdx.y*gridDim.x + blockIdx.x;

    prologue<TA, TB, GLU && !SAME, NW*64>(smem, TOFF_B, YOFF, yg, K, tid);
    __syncthreads();

    for (; u < U; u += ustep) {
        float acc = 0.0f, accb = 0.0f;
        for (int it0 = lane; it0 < ipr; it0 += L*B) {
            typename TA::template ld<P> la[B];
            typename TB::template ld<P> lb[GLU ? B : 1];
#pragma unroll
            for (int b = 0; b < B; ++b) {
                const int it = min(it0 + b*L, ipr - 1);
                const int h = it / IPC, p = it % IPC;
                const uint32_t c  = ((uint32_t) u*(uint32_t) items + (uint32_t) h)*RG + (uint32_t) grp;
                const uint32_t cb = ((uint32_t) u*(uint32_t) (items/2) + (uint32_t) (h >> 1))*RG + (uint32_t) grp;
                la[b] = TA::template load<P>(wa, nh, c, cb, p);
                if constexpr (GLU) lb[b] = TB::template load<P>(wb, nh, c, cb, p);
            }
#pragma unroll
            for (int b = 0; b < B; ++b) {
                const int it = it0 + b*L;
                if (it >= ipr) break;
                const int h = it / IPC, p = it % IPC;
                const uint32_t ya = YOFF + (uint32_t) h*(YREC_W*4) + (uint32_t) p*(32*P);
                const uint32_t yd_addr = YOFF + (uint32_t) h*(YREC_W*4) + 128 + (uint32_t) p*(4*P);
                float dy4[P];
#if defined(__HIP_DEVICE_COMPILE__)
                if constexpr (P == 4) {
                    const float4 v = *(__attribute__((address_space(3))) const float4 *) (size_t) yd_addr;
                    dy4[0] = v.x; dy4[1] = v.y; dy4[2] = v.z; dy4[3] = v.w;
                } else {
                    const float2 v = *(__attribute__((address_space(3))) const float2 *) (size_t) yd_addr;
                    dy4[0] = v.x; dy4[1] = v.y;
                }
#else
                for (int j = 0; j < P; ++j) dy4[j] = 0.0f;
#endif
                const typename TA::template dec<P> da = TA::template prep<P>(la[b]);
                const float dwa = TA::template dw<P>(da);
                typename TB::template dec<P> db;
                float dwb = 0.0f;
                if constexpr (GLU) { db = TB::template prep<P>(lb[b]); dwb = TB::template dw<P>(db); }
#pragma unroll
                for (int j = 0; j < P; ++j) {
                    int y[8];
#if defined(__HIP_DEVICE_COMPILE__)
                    const int4 y0 = *(__attribute__((address_space(3))) const int4 *) (size_t) (ya + 32*j);
                    const int4 y1 = *(__attribute__((address_space(3))) const int4 *) (size_t) (ya + 32*j + 16);
                    y[0] = y0.x; y[1] = y0.y; y[2] = y0.z; y[3] = y0.w; y[4] = y1.x; y[5] = y1.y; y[6] = y1.z; y[7] = y1.w;
#else
                    for (int k = 0; k < 8; ++k) y[k] = 0;
#endif
                    acc += (dwa*dy4[j]) * (float) TA::template sub<P, TOFF_A>(da, j, y);
                    if constexpr (GLU) accb += (dwb*dy4[j]) * (float) TB::template sub<P, TOFF_B>(db, j, y);
                }
            }
        }
        const int row = u*RG + grp;
        const float v = group_sum<L>(acc);
        if constexpr (GLU) {
            const float vb = group_sum<L>(accb);
            if (lane == 0) dst[row] = vb * silu(v);
        } else {
            if (lane == 0) dst[row] = residual ? v + residual[row] : v;
        }
    }
}

// ------------------------------------------------------------------------------------------------ y-in-registers kernel
// Same as gemv (R = 1, PF = 0) for shapes with exactly NI items per lane (ipr == NI*L): every lane always handles the
// same items it = lane + k*L of each row, so it keeps their activations (P sub-blocks x {8 qs words, d}) in VGPRs for
// the whole kernel, loaded once from global memory. LDS then holds only the codebook tables (no y traffic at all).
template <class TA, class TB, int P, int L, int NW, int NI>
__launch_bounds__(NW*64, 1)
__global__ void gemv_yr(const uint8_t * __restrict__ wa, const uint8_t * __restrict__ wb, const block_q8_1 * __restrict__ yg,
        float * __restrict__ dst, const int K, const int N, const float * __restrict__ residual) {
    constexpr bool GLU  = !std::is_same<TB, none_t>::value;
    constexpr bool SAME = std::is_same<TA, TB>::value;
    constexpr int TOFF_A = 0;
    constexpr int TOFF_B = (GLU && !SAME) ? TA::TAB_WORDS*4 : 0;
    constexpr int G  = 64/L;
    constexpr int RG = G;
    constexpr int IPC = 4/P;
    extern __shared__ uint32_t smem[];
#if defined(__HIP_DEVICE_COMPILE__)
    if ((uint32_t) (size_t) (__attribute__((address_space(3))) uint32_t *) smem != 0) __builtin_trap();
#endif
    const int tid  = threadIdx.y*64 + threadIdx.x;
    const int lane = threadIdx.x % L, grp = threadIdx.x / L;
    const int items = K / 128;
    const size_t nh = (size_t) N*items;
    const int U = N / RG;
    const int ustep = gridDim.x*NW;
    int u = threadIdx.y*gridDim.x + blockIdx.x;

    // activations of this lane's NI items
    int   yq[NI][P][8];
    float yd[NI][P];
#pragma unroll
    for (int k = 0; k < NI; ++k) {
        const int it = lane + k*L;
        const int h = it / IPC, p = it % IPC;
#pragma unroll
        for (int j = 0; j < P; ++j) {
            const uint8_t * b = (const uint8_t *) (yg + 4*h + P*p + j);
            const uint4 q0 = *(const uint4 *) (b + 4), q1 = *(const uint4 *) (b + 20);
            yq[k][j][0] = (int) q0.x; yq[k][j][1] = (int) q0.y; yq[k][j][2] = (int) q0.z; yq[k][j][3] = (int) q0.w;
            yq[k][j][4] = (int) q1.x; yq[k][j][5] = (int) q1.y; yq[k][j][6] = (int) q1.z; yq[k][j][7] = (int) q1.w;
            yd[k][j] = h2f(*(const uint16_t *) b);
        }
    }
    TA::stage_table(smem + TOFF_A/4, tid, NW*64);
    if constexpr (GLU && !SAME) TB::stage_table(smem + TOFF_B/4, tid, NW*64);
    __syncthreads();

    for (; u < U; u += ustep) {
        float acc = 0.0f, accb = 0.0f;
#pragma unroll
        for (int k = 0; k < NI; ++k) {
            const int it = lane + k*L;
            const int h = it / IPC, p = it % IPC;
            const uint32_t c  = ((uint32_t) u*(uint32_t) items + (uint32_t) h)*RG + (uint32_t) grp;
            const uint32_t cb = ((uint32_t) u*(uint32_t) (items/2) + (uint32_t) (h >> 1))*RG + (uint32_t) grp;
            const typename TA::template dec<P> da = TA::template prep<P>(TA::template load<P>(wa, nh, c, cb, p));
            const float dwa = TA::template dw<P>(da);
            typename TB::template dec<P> db;
            float dwb = 0.0f;
            if constexpr (GLU) { db = TB::template prep<P>(TB::template load<P>(wb, nh, c, cb, p)); dwb = TB::template dw<P>(db); }
#pragma unroll
            for (int j = 0; j < P; ++j) {
                acc += (dwa*yd[k][j]) * (float) TA::template sub<P, TOFF_A>(da, j, yq[k][j]);
                if constexpr (GLU) accb += (dwb*yd[k][j]) * (float) TB::template sub<P, TOFF_B>(db, j, yq[k][j]);
            }
        }
        const int row = u*RG + grp;
        const float v = group_sum<L>(acc);
        if constexpr (GLU) {
            const float vb = group_sum<L>(accb);
            if (lane == 0) dst[row] = vb * silu(v);
        } else {
            if (lane == 0) dst[row] = residual ? v + residual[row] : v;
        }
    }
}
template <class TA, class TB>
static inline size_t lds_bytes_tab() {
    constexpr bool GLU = !std::is_same<TB, none_t>::value, SAME = std::is_same<TA, TB>::value;
    return (size_t) TA::TAB_WORDS*4 + ((GLU && !SAME) ? (size_t) TB::TAB_WORDS*4 : 0);
}

// ------------------------------------------------------------------------------------------------ ping-pong kernel
// Same work distribution and decode as gemv, but the weight loads are software-pipelined over the wave's flat
// sequence of steps (step = one item per lane for each of the R rows of the lane group; S = ceil(ipr/L) steps per
// unit) with two register buffers and a manual 2x unroll, so no loop-carried register copies of in-flight loads
// exist and the waitcnt pass can wait precisely for the older buffer. The first load is issued before the prologue.
template <class TA, class TB, int P, int L, int R, int NW, int MINW = 1>
__launch_bounds__(NW*64, MINW)
__global__ void gemv_pp(const uint8_t * __restrict__ wa, const uint8_t * __restrict__ wb, const block_q8_1 * __restrict__ yg,
        float * __restrict__ dst, const int K, const int N, const float * __restrict__ residual) {
    constexpr bool GLU  = !std::is_same<TB, none_t>::value;
    constexpr bool SAME = std::is_same<TA, TB>::value;
    constexpr int TOFF_A = 0;
    constexpr int TOFF_B = (GLU && !SAME) ? TA::TAB_WORDS*4 : 0;
    constexpr int YOFF   = TA::TAB_WORDS*4 + ((GLU && !SAME) ? TB::TAB_WORDS*4 : 0);
    constexpr int G  = 64/L;
    constexpr int RG = G*R;
    constexpr int IPC = 4/P;
    extern __shared__ uint32_t smem[];
#if defined(__HIP_DEVICE_COMPILE__)
    if ((uint32_t) (size_t) (__attribute__((address_space(3))) uint32_t *) smem != 0) __builtin_trap();
#endif
    const int tid  = threadIdx.y*64 + threadIdx.x;
    const int lane = threadIdx.x % L, grp = threadIdx.x / L;
    const int items = K / 128;
    const int ipr   = items*IPC;
    const int S     = (ipr + L - 1) / L;
    const size_t nh = (size_t) N*items;
    const int U = N / RG;
    const int ustep = gridDim.x*NW;
    const int u0 = threadIdx.y*gridDim.x + blockIdx.x;

    typedef typename TA::template ld<P> lda_t;
    typedef typename TB::template ld<P> ldb_t;
    lda_t A[R], B[R];
    ldb_t Ab[GLU ? R : 1], Bb[GLU ? R : 1];
    auto load_step = [&](lda_t (&xa)[R], ldb_t (&xb)[GLU ? R : 1], const int uu, const int ss) {
        const int vu = uu < U ? uu : u0;                       // clamped: always a valid address
        const int it = min(ss*L + lane, ipr - 1);
        const int h = it / IPC, p = it % IPC;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint32_t rin = (uint32_t) (r*G + grp);
            const uint32_t c  = ((uint32_t) vu*(uint32_t) items + (uint32_t) h)*RG + rin;
            const uint32_t cb = ((uint32_t) vu*(uint32_t) (items/2) + (uint32_t) (h >> 1))*RG + rin;
            xa[r] = TA::template load<P>(wa, nh, c, cb, p);
            if constexpr (GLU) xb[r] = TB::template load<P>(wb, nh, c, cb, p);
        }
    };
    float acc[R], accb[GLU ? R : 1];
#pragma unroll
    for (int r = 0; r < R; ++r) { acc[r] = 0.0f; if constexpr (GLU) accb[r] = 0.0f; }
    auto compute_step = [&](const lda_t (&xa)[R], const ldb_t (&xb)[GLU ? R : 1], const int uu, const int ss) {
        const int it = ss*L + lane;
        if (it < ipr) {
            const int h = it / IPC, p = it % IPC;
            const uint32_t ya = YOFF + (uint32_t) h*(YREC_W*4) + (uint32_t) p*(32*P);
            const uint32_t yd_addr = YOFF + (uint32_t) h*(YREC_W*4) + 128 + (uint32_t) p*(4*P);
            float dy4[P];
#if defined(__HIP_DEVICE_COMPILE__)
            if constexpr (P == 4) {
                const float4 v = *(__attribute__((address_space(3))) const float4 *) (size_t) yd_addr;
                dy4[0] = v.x; dy4[1] = v.y; dy4[2] = v.z; dy4[3] = v.w;
            } else {
                const float2 v = *(__attribute__((address_space(3))) const float2 *) (size_t) yd_addr;
                dy4[0] = v.x; dy4[1] = v.y;
            }
#else
            for (int j = 0; j < P; ++j) dy4[j] = 0.0f;
#endif
            typename TA::template dec<P> da[R];
            typename TB::template dec<P> db[GLU ? R : 1];
            float dwa[R], dwb[GLU ? R : 1];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                da[r] = TA::template prep<P>(xa[r]); dwa[r] = TA::template dw<P>(da[r]);
                if constexpr (GLU) { db[r] = TB::template prep<P>(xb[r]); dwb[r] = TB::template dw<P>(db[r]); }
            }
#pragma unroll
            for (int j = 0; j < P; ++j) {
                int y[8];
#if defined(__HIP_DEVICE_COMPILE__)
                const int4 y0 = *(__attribute__((address_space(3))) const int4 *) (size_t) (ya + 32*j);
                const int4 y1 = *(__attribute__((address_space(3))) const int4 *) (size_t) (ya + 32*j + 16);
                y[0] = y0.x; y[1] = y0.y; y[2] = y0.z; y[3] = y0.w; y[4] = y1.x; y[5] = y1.y; y[6] = y1.z; y[7] = y1.w;
#else
                for (int k = 0; k < 8; ++k) y[k] = 0;
#endif
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    acc[r] += (dwa[r]*dy4[j]) * (float) TA::template sub<P, TOFF_A>(da[r], j, y);
                    if constexpr (GLU) accb[r] += (dwb[r]*dy4[j]) * (float) TB::template sub<P, TOFF_B>(db[r], j, y);
                }
            }
        }
        if (ss == S - 1) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int row = uu*RG + r*G + grp;
                const float v = group_sum<L>(acc[r]);
                if constexpr (GLU) {
                    const float vb = group_sum<L>(accb[r]);
                    if (lane == 0) dst[row] = vb * silu(v);
                    accb[r] = 0.0f;
                } else {
                    if (lane == 0) dst[row] = residual ? v + residual[row] : v;
                }
                acc[r] = 0.0f;
            }
        }
    };

    int lu = u0, ls = 0;                 // load cursor
    if (u0 < U) {
        load_step(A, Ab, lu, ls);
        if (++ls == S) { ls = 0; lu += ustep; }
    }
    prologue<TA, TB, GLU && !SAME, NW*64>(smem, TOFF_B, YOFF, yg, K, tid);
    __syncthreads();

    int cu = u0, cs = 0;                 // compute cursor
    while (cu < U) {
        load_step(B, Bb, lu, ls);
        if (++ls == S) { ls = 0; lu += ustep; }
#if defined(__HIP_DEVICE_COMPILE__)
        __builtin_amdgcn_sched_barrier(0);
#endif
        compute_step(A, Ab, cu, cs);
        if (++cs == S) { cs = 0; cu += ustep; }
        if (cu >= U) break;
        load_step(A, Ab, lu, ls);
        if (++ls == S) { ls = 0; lu += ustep; }
#if defined(__HIP_DEVICE_COMPILE__)
        __builtin_amdgcn_sched_barrier(0);
#endif
        compute_step(B, Bb, cu, cs);
        if (++cs == S) { cs = 0; cu += ustep; }
    }
}

// ------------------------------------------------------------------------------------------------ stream-K kernel
// Same decode as gemv (R = 1), but the (unit, step) space (step = one item per lane, i.e. L items of each of the RG = 64/L
// rows of a unit) is split evenly over min(W, T) waves, so every wave gets the same amount of work whatever N and K are.
// A unit that spans several waves is finished by the last arriving piece: every piece stores its partial row sums in
// part[], bumps cnt[unit]; the last one sums the partials in piece order (deterministic) and applies the epilogue, then
// resets cnt[unit] = 0. part: U*S*RG*2 floats, cnt: U ints, zero before the first launch (left at zero by every launch).
// Requires (K/128)*(4/P) % L == 0.
template <class TA, class TB, int P, int L, int NW>
__launch_bounds__(NW*64, 1)
__global__ void gemv_sk(const uint8_t * __restrict__ wa, const uint8_t * __restrict__ wb, const block_q8_1 * __restrict__ yg,
        float * __restrict__ dst, const int K, const int N, const float * __restrict__ residual,
        float * __restrict__ part, int * __restrict__ cnt) {
    constexpr bool GLU  = !std::is_same<TB, none_t>::value;
    constexpr bool SAME = std::is_same<TA, TB>::value;
    constexpr int TOFF_A = 0;
    constexpr int TOFF_B = (GLU && !SAME) ? TA::TAB_WORDS*4 : 0;
    constexpr int YOFF   = TA::TAB_WORDS*4 + ((GLU && !SAME) ? TB::TAB_WORDS*4 : 0);
    constexpr int G  = 64/L;
    constexpr int RG = G;
    constexpr int IPC = 4/P;
    extern __shared__ uint32_t smem[];
#if defined(__HIP_DEVICE_COMPILE__)
    if ((uint32_t) (size_t) (__attribute__((address_space(3))) uint32_t *) smem != 0) __builtin_trap();
#endif
    const int tid  = threadIdx.y*64 + threadIdx.x;
    const int lane = threadIdx.x % L, grp = threadIdx.x / L;
    const int items = K / 128;
    const int ipr   = items*IPC;
    const int S     = ipr / L;                   // steps per unit
    const size_t nh = (size_t) N*items;
    const int U = N / RG;
    const uint32_t T = (uint32_t) U*(uint32_t) S;          // host guarantees W*T < 2^32
    const uint32_t W = min((uint32_t) gridDim.x*NW, T);
    const int w = threadIdx.y*gridDim.x + blockIdx.x;

    prologue<TA, TB, GLU && !SAME, NW*64>(smem, TOFF_B, YOFF, yg, K, tid);
    __syncthreads();
    if ((uint32_t) w >= W) return;

    auto wstart = [&](const uint32_t x) -> uint32_t { return x*T/W; };
    uint32_t t = wstart(w);
    const uint32_t tend = wstart(w + 1);
    int u = (int) (t / (uint32_t) S), s0 = (int) (t - (uint32_t) u*S);   // one division; then incremental
    int sp_u[4], sp_n[4], nsplit = 0;          // split pieces of this wave (at most 2 in practice)
    while (t < tend) {
        const int s1 = (int) min((uint32_t) S, (uint32_t) s0 + (tend - t));
        float acc = 0.0f, accb = 0.0f;
        for (int s = s0; s < s1; ++s) {
            const int it = s*L + lane;
            const int h = it / IPC, p = it % IPC;
            const uint32_t c  = ((uint32_t) u*(uint32_t) items + (uint32_t) h)*RG + (uint32_t) grp;
            const uint32_t cb = ((uint32_t) u*(uint32_t) (items/2) + (uint32_t) (h >> 1))*RG + (uint32_t) grp;
            const typename TA::template dec<P> ca = TA::template prep<P>(TA::template load<P>(wa, nh, c, cb, p));
            typename TB::template dec<P> cbw;
            if constexpr (GLU) cbw = TB::template prep<P>(TB::template load<P>(wb, nh, c, cb, p));
            const uint32_t ya = YOFF + (uint32_t) h*(YREC_W*4) + (uint32_t) p*(32*P);
            const uint32_t yd_addr = YOFF + (uint32_t) h*(YREC_W*4) + 128 + (uint32_t) p*(4*P);
            float dy4[P];
#if defined(__HIP_DEVICE_COMPILE__)
            if constexpr (P == 4) {
                const float4 v = *(__attribute__((address_space(3))) const float4 *) (size_t) yd_addr;
                dy4[0] = v.x; dy4[1] = v.y; dy4[2] = v.z; dy4[3] = v.w;
            } else {
                const float2 v = *(__attribute__((address_space(3))) const float2 *) (size_t) yd_addr;
                dy4[0] = v.x; dy4[1] = v.y;
            }
#else
            for (int j = 0; j < P; ++j) dy4[j] = 0.0f;
#endif
            const float dwa = TA::template dw<P>(ca);
            float dwb = 0.0f;
            if constexpr (GLU) dwb = TB::template dw<P>(cbw);
#pragma unroll
            for (int j = 0; j < P; ++j) {
                int y[8];
#if defined(__HIP_DEVICE_COMPILE__)
                const int4 y0 = *(__attribute__((address_space(3))) const int4 *) (size_t) (ya + 32*j);
                const int4 y1 = *(__attribute__((address_space(3))) const int4 *) (size_t) (ya + 32*j + 16);
                y[0] = y0.x; y[1] = y0.y; y[2] = y0.z; y[3] = y0.w; y[4] = y1.x; y[5] = y1.y; y[6] = y1.z; y[7] = y1.w;
#else
                for (int k = 0; k < 8; ++k) y[k] = 0;
#endif
                acc += (dwa*dy4[j]) * (float) TA::template sub<P, TOFF_A>(ca, j, y);
                if constexpr (GLU) accb += (dwb*dy4[j]) * (float) TB::template sub<P, TOFF_B>(cbw, j, y);
            }
        }
        t += (uint32_t) (s1 - s0);
        const int cu = u, cs0 = s0;
        u += 1; s0 = 0;                          // the next piece starts at the next unit
        const int row = cu*RG + grp;
        const float v  = group_sum<L>(acc);
        const float vb = GLU ? group_sum<L>(accb) : 0.0f;
        if (cs0 == 0 && s1 == S) {
            if (lane == 0) dst[row] = GLU ? vb * silu(v) : (residual ? v + residual[row] : v);
            continue;
        }
        // split unit: pieces = waves whose (non-empty) ranges intersect [ub, ue). Store the partial now, do the
        // counter / fix-up after the wave's last piece (one fence per wave instead of one per split piece).
        const uint32_t ub = (uint32_t) cu*(uint32_t) S, ue = ub + (uint32_t) S;
        uint32_t wf = w; while (wf > 0 && wstart(wf) > ub) --wf;
        uint32_t wl = w; while (wl + 1 < W && wstart(wl + 1) < ue) ++wl;
        const int slot = (int) (w - wf), npieces = (int) (wl - wf + 1);
        float * pp = part + ((size_t) cu*S + slot)*(RG*2);
        if (lane == 0) { pp[2*grp] = v; pp[2*grp + 1] = vb; }
        if (nsplit < 4) { sp_u[nsplit] = cu; sp_n[nsplit] = npieces; ++nsplit; }
    }
    if (nsplit == 0) return;
    __threadfence();
    for (int k = 0; k < nsplit; ++k) {
        const int cu = sp_u[k], npieces = sp_n[k];
        int old = 0;
        if (threadIdx.x == 0) old = atomicAdd(cnt + cu, 1);
        old = __shfl(old, 0, 64);
        if (old == npieces - 1) {
            __threadfence();
            if (lane == 0) {
                const int row = cu*RG + grp;
                float sv = 0.0f, svb = 0.0f;
                for (int q = 0; q < npieces; ++q) {
                    const float * pq = part + ((size_t) cu*S + q)*(RG*2);
                    sv += __builtin_nontemporal_load(pq + 2*grp);
                    if constexpr (GLU) svb += __builtin_nontemporal_load(pq + 2*grp + 1);
                }
                dst[row] = GLU ? svb * silu(sv) : (residual ? sv + residual[row] : sv);
            }
            if (threadIdx.x == 0) cnt[cu] = 0;
        }
    }
}
static inline size_t sk_part_floats(int N, int K, int P, int L) { const int RG = 64/L; return (size_t) (N/RG)*((K/128)*(4/P)/L)*RG*2; }
static inline size_t sk_cnt_ints(int N, int L) { return (size_t) (N/(64/L)); }

template <class TA, class TB>
static inline size_t lds_bytes(int K) {
    constexpr bool GLU = !std::is_same<TB, none_t>::value, SAME = std::is_same<TA, TB>::value;
    return (size_t) TA::TAB_WORDS*4 + ((GLU && !SAME) ? (size_t) TB::TAB_WORDS*4 : 0) + (size_t) (K/128)*YREC_W*4;
}

// ------------------------------------------------------------------------------------------------ host side
static inline bool supported(ggml_type t) { return t == GGML_TYPE_IQ3_S || t == GGML_TYPE_IQ3_XXS; }

// src = original ggml blocks (N rows x K), dst = separate buffer of soa_bytes(t, N*K/256) bytes, RG = row interleave
static inline bool repack(ggml_type t, const void * src, void * dst, int N, int K, int RG, hipStream_t st, int CPS = 0) {
    if (K % 256 != 0 || N % RG != 0 || (CPS && ((K/128) % CPS != 0 || CPS % 2 != 0))) return false;
    const size_t nh = (size_t) N*(K/128);
    const int nthr = 256; const int nblk = (int) ((nh + nthr - 1)/nthr);
    switch (t) {
        case GGML_TYPE_IQ3_S:   hipLaunchKernelGGL(repack_iq3_s,   dim3(nblk), dim3(nthr), 0, st, (const block_iq3_s *)   src, (uint8_t *) dst, N, K, RG, CPS); return true;
        case GGML_TYPE_IQ3_XXS: hipLaunchKernelGGL(repack_iq3_xxs, dim3(nblk), dim3(nthr), 0, st, (const block_iq3_xxs *) src, (uint8_t *) dst, N, K, RG, CPS); return true;
        default: return false;
    }
}

} // namespace iq3g
