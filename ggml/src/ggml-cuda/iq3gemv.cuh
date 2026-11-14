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

// repack: one thread per (row, half-block). dst must not alias src. N % RG == 0, K % 256 == 0.
static __global__ void repack_iq3_s(const block_iq3_s * __restrict__ src, uint8_t * __restrict__ dst, const int N, const int K, const int RG) {
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
    const uint32_t c = chunk_index(row, h, items, (uint32_t) RG);
    uint4 * Q = (uint4 *) dst; uint4 * FA = (uint4 *) (dst + nh*32); uint2 * FB = (uint2 *) (dst + nh*48);
    Q[2*(size_t) c]     = make_uint4(q[0], q[1], q[2], q[3]);
    Q[2*(size_t) c + 1] = make_uint4(q[4], q[5], q[6], q[7]);
    FA[c] = make_uint4(F[0], F[1], F[2], F[3]);
    FB[c] = make_uint2(F[4], F[5]);
}

static __global__ void repack_iq3_xxs(const block_iq3_xxs * __restrict__ src, uint8_t * __restrict__ dst, const int N, const int K, const int RG) {
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
    const uint32_t c = chunk_index(row, h, items, (uint32_t) RG);
    uint4 * Q = (uint4 *) dst; uint4 * AA = (uint4 *) (dst + nh*32); uint16_t * D = (uint16_t *) (dst + nh*48);
    Q[2*(size_t) c]     = make_uint4(q[0], q[1], q[2], q[3]);
    Q[2*(size_t) c + 1] = make_uint4(q[4], q[5], q[6], q[7]);
    AA[c] = make_uint4(A[0], A[1], A[2], A[3]);
    if (half == 0) { uint16_t d; memcpy(&d, &b.d, 2); D[chunk_index(row, h/2, items/2, (uint32_t) RG)] = d; }
}

// ------------------------------------------------------------------------------------------------ type traits
// item = P sub-blocks (P = 4: whole chunk, P = 2: part p of the chunk) of one row.
// TOFF = byte offset of the type's table in LDS (absolute address).
struct iq3s_t {
    static constexpr ggml_type type = GGML_TYPE_IQ3_S;
    static constexpr int TAB_WORDS = 8192;
    template <int P> struct raw { uint32_t q[2*P]; uint32_t W[P == 4 ? 6 : 3]; uint32_t d, sc; };
    static __device__ __forceinline__ void stage_table(uint32_t * tab, const int tid, const int nt) {
        for (int e = tid; e < 8192; e += nt) tab[e] = apply_signs(iq3s_grid[e & 511], (uint32_t) e >> 9);
    }
    template <int P>
    static __device__ __forceinline__ raw<P> load(const uint8_t * __restrict__ base, const size_t nh, const uint32_t c, const uint32_t cb, const int p) {
        (void) cb;
        raw<P> r;
        const uint4 fa = *(const uint4 *) (base + nh*32 + c*16u);
        const uint2 fb = *(const uint2 *) (base + nh*48 + c*8u);
        if constexpr (P == 4) {
            const uint4 q0 = *(const uint4 *) (base + c*32u), q1 = *(const uint4 *) (base + c*32u + 16u);
            r.q[0] = q0.x; r.q[1] = q0.y; r.q[2] = q0.z; r.q[3] = q0.w; r.q[4] = q1.x; r.q[5] = q1.y; r.q[6] = q1.z; r.q[7] = q1.w;
            r.W[0] = fa.x; r.W[1] = fa.y; r.W[2] = fa.z; r.W[3] = fa.w; r.W[4] = fb.x; r.W[5] = fb.y;
            r.sc = fb.y >> 16;
        } else {
            const uint4 q0 = *(const uint4 *) (base + c*32u + 16u*p);
            r.q[0] = q0.x; r.q[1] = q0.y; r.q[2] = q0.z; r.q[3] = q0.w;
            r.W[0] = p ? fa.w : fa.x; r.W[1] = p ? fb.x : fa.y; r.W[2] = p ? fb.y : fa.z;
            r.sc = fb.y >> (16 + 8*p);
        }
        r.d = fa.z >> 16;
        return r;
    }
    template <int P> static __device__ __forceinline__ float dw(const raw<P> & r) { return h2f(r.d); }
    template <int P> static __device__ __forceinline__ uint32_t fold(const raw<P> & r) {
        uint32_t x = r.d ^ r.sc;
        for (int i = 0; i < 2*P; ++i) x ^= r.q[i];
        for (int i = 0; i < (P == 4 ? 6 : 3); ++i) x ^= r.W[i];
        return x;
    }
    // scaled integer sum of sub-block j (0..P-1 of the item) against the 8 y words
    template <int P, int TOFF>
    static __device__ __forceinline__ int sub(const raw<P> & r, const int j, const int (&y)[8]) {
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
    template <int P> struct raw { uint32_t q[2*P]; uint32_t a[P]; uint32_t d; };
    static __device__ __forceinline__ void stage_table(uint32_t * tab, const int tid, const int nt) {
        for (int e = tid; e < 4096; e += nt) tab[e] = apply_signs(iq3xxs_grid[e & 255], (uint32_t) e >> 8);
    }
    template <int P>
    static __device__ __forceinline__ raw<P> load(const uint8_t * __restrict__ base, const size_t nh, const uint32_t c, const uint32_t cb, const int p) {
        raw<P> r;
        const uint4 a = *(const uint4 *) (base + nh*32 + c*16u);
        if constexpr (P == 4) {
            const uint4 q0 = *(const uint4 *) (base + c*32u), q1 = *(const uint4 *) (base + c*32u + 16u);
            r.q[0] = q0.x; r.q[1] = q0.y; r.q[2] = q0.z; r.q[3] = q0.w; r.q[4] = q1.x; r.q[5] = q1.y; r.q[6] = q1.z; r.q[7] = q1.w;
            r.a[0] = a.x; r.a[1] = a.y; r.a[2] = a.z; r.a[3] = a.w;
        } else {
            const uint4 q0 = *(const uint4 *) (base + c*32u + 16u*p);
            r.q[0] = q0.x; r.q[1] = q0.y; r.q[2] = q0.z; r.q[3] = q0.w;
            r.a[0] = p ? a.z : a.x; r.a[1] = p ? a.w : a.y;
        }
        r.d = *(const uint16_t *) (base + nh*48 + cb*2u);
        return r;
    }
    template <int P> static __device__ __forceinline__ float dw(const raw<P> & r) { return h2f(r.d); }
    template <int P> static __device__ __forceinline__ uint32_t fold(const raw<P> & r) {
        uint32_t x = r.d;
        for (int i = 0; i < 2*P; ++i) x ^= r.q[i];
        for (int i = 0; i < P; ++i) x ^= r.a[i];
        return x;
    }
    template <int P, int TOFF>
    static __device__ __forceinline__ int sub(const raw<P> & r, const int j, const int (&y)[8]) {
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
    template <int P> struct raw {};
    template <int P> static __device__ uint32_t fold(const raw<P> &) { return 0; }
};

template <class TA, class TB, int P, int L, int R, int NW, int PF, int DBG = 0, int MINW = 1>   // DBG: 1 = loads only, 2 = compute only
__launch_bounds__(NW*64, MINW)   /* MINW = min waves per SIMD (8 with NW=16: two workgroups per CU) */
__global__ void gemv(const uint8_t * __restrict__ wa, const uint8_t * __restrict__ wb, const block_q8_1 * __restrict__ yg,
        float * __restrict__ dst, const int K, const int N, const float * __restrict__ residual) {
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

    typename TA::template raw<P> ra[R];
    typename TB::template raw<P> rb[GLU ? R : 1];
    auto load_item = [&](const int uu, const int it) {
        const int h = it / IPC, p = it % IPC;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint32_t rin = (uint32_t) (r*G + grp);
            const uint32_t c  = ((uint32_t) uu*(uint32_t) items + (uint32_t) h)*RG + rin;
            const uint32_t cb = ((uint32_t) uu*(uint32_t) (items/2) + (uint32_t) (h >> 1))*RG + rin;
            ra[r] = TA::template load<P>(wa, nh, c, cb, p);
            if constexpr (GLU) rb[r] = TB::template load<P>(wb, nh, c, cb, p);
        }
    };
    if ((PF || DBG == 2) && u < U) load_item(u, lane);

    TA::stage_table(smem + TOFF_A/4, tid, NW*64);
    if constexpr (GLU && !SAME) TB::stage_table(smem + TOFF_B/4, tid, NW*64);
    stage_y((int *) (smem + YOFF/4), yg, K, tid, NW*64);
    __syncthreads();

    for (; u < U; u += ustep) {
        float acc[R], accb[GLU ? R : 1];
#pragma unroll
        for (int r = 0; r < R; ++r) { acc[r] = 0.0f; if constexpr (GLU) accb[r] = 0.0f; }
        for (int it = lane; it < ipr; it += L) {
            if (!PF && DBG != 2) load_item(u, it);
            typename TA::template raw<P> ca[R];
            typename TB::template raw<P> cb[GLU ? R : 1];
#pragma unroll
            for (int r = 0; r < R; ++r) { ca[r] = ra[r]; if constexpr (GLU) cb[r] = rb[r]; }
            if (PF && DBG != 2) {
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
            float dwa[R], dwb[GLU ? R : 1];
#pragma unroll
            for (int r = 0; r < R; ++r) { dwa[r] = TA::template dw<P>(ca[r]); if constexpr (GLU) dwb[r] = TB::template dw<P>(cb[r]); }
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
                    acc[r] += (dwa[r]*dy4[j]) * (float) TA::template sub<P, TOFF_A>(ca[r], j, y);
                    if constexpr (GLU) accb[r] += (dwb[r]*dy4[j]) * (float) TB::template sub<P, TOFF_B>(cb[r], j, y);
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

template <class TA, class TB>
static inline size_t lds_bytes(int K) {
    constexpr bool GLU = !std::is_same<TB, none_t>::value, SAME = std::is_same<TA, TB>::value;
    return (size_t) TA::TAB_WORDS*4 + ((GLU && !SAME) ? (size_t) TB::TAB_WORDS*4 : 0) + (size_t) (K/128)*YREC_W*4;
}

// ------------------------------------------------------------------------------------------------ host side
static inline bool supported(ggml_type t) { return t == GGML_TYPE_IQ3_S || t == GGML_TYPE_IQ3_XXS; }

// src = original ggml blocks (N rows x K), dst = separate buffer of soa_bytes(t, N*K/256) bytes, RG = row interleave
static inline bool repack(ggml_type t, const void * src, void * dst, int N, int K, int RG, hipStream_t st) {
    if (K % 256 != 0 || N % RG != 0) return false;
    const size_t nh = (size_t) N*(K/128);
    const int nthr = 256; const int nblk = (int) ((nh + nthr - 1)/nthr);
    switch (t) {
        case GGML_TYPE_IQ3_S:   hipLaunchKernelGGL(repack_iq3_s,   dim3(nblk), dim3(nthr), 0, st, (const block_iq3_s *)   src, (uint8_t *) dst, N, K, RG); return true;
        case GGML_TYPE_IQ3_XXS: hipLaunchKernelGGL(repack_iq3_xxs, dim3(nblk), dim3(nthr), 0, st, (const block_iq3_xxs *) src, (uint8_t *) dst, N, K, RG); return true;
        default: return false;
    }
}

} // namespace iq3g
