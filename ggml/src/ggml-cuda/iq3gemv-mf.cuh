// iq3gemv-mf.cuh - N-column (NC = 2..16) IQ3_S / IQ3_XXS GEMVs on the int8 matrix cores (MMVQ-NC A, MFMA form).
// Same SoA copies as the single-token kernels (iq3gemv.cuh, unchanged); the weights are read and decoded once for all
// columns. Per column the result is bit-identical to the single-token kernel run on that column alone:
//   - integer stage: the per-(row, column, 32-weight sub-block) integer sum of the single-token dp4a chain comes from 8
//     chained v_mfma_i32_4x4x4i8 (int32 sums are exact, any order). Lane = 4*l + q: item lane l (0..15) of the
//     single-token kernel (items l, l + 16, .. of a row, in increasing order), q = row of the 4-row unit for the A operand
//     (the sign-expanded codebook words of that row, nc::look) and = column within a group of 4 for B (that column's q8
//     words); D gives the lane its column's sums of the 4 rows. CG = ceil(NC/4) column groups (one chain each) share the
//     decoded A words.
//   - the integer scale step, done in fp32 where it is exact: IQ3_S (float) mul24(s, 1+2sc) == (float) s * (float) (1+2sc)
//     (|s*(1+2sc)| <= 32*15*128*31 < 2^24, both are the rounding of the same exact integer); IQ3_XXS
//     sdiv<1>(mul24(ls, s) + sdiv<1>(s)) == trunc(s*(2ls+1)/4) for every integer s (both are sign-symmetric; for s >= 0
//     floor((ls*s + floor(s/2))/2) and floor((2ls+1)s/4) differ by at most s%2/4 below the same integer), evaluated as
//     truncf((float) s * ((2ls+1)/4)): |s*(2ls+1)| <= 32*62*128*31 < 2^24, so the product is exact.
//   - float stage, per (row, column) the single-token sequence: dwdy = dw*dy (rounded), term = dwdy*sv (rounded),
//     acc = acc + term (single / residual / multi-job kernels: not contracted) or acc = fma(dwdy, sv, acc) (gate/up
//     kernels), per item in the lane's item order. Packed fp32 (v_pk_*) rounds per element like the scalar op.
//   - reduction: group_sum<16> over the item lanes (butterfly l^1, l^2, l^7, l^15). After the first two steps a value is
//     equal on the 4 lanes of a quad (fp add commutes), so l^7 / l^15 can be replaced by l^4 / l^8: the tree is the xor
//     butterfly 1, 2, 4, 8 on l = lanes ^4, ^8, ^16, ^32. Each step halves the values a lane keeps (transposed
//     reduction: the lane with the lane bit clear keeps the low half, its partner the high half; both compute
//     a[l] + a[l^m] for the half they keep), so every (row, column) is summed by exactly the single-token tree.
//   - epilogues as the single-token kernels: v + residual, vb * silu(v) (gate v, up vb).
// K phases: the records of all columns rarely fit next to the tables (NC = 16, K = 5120: 92 KB), so the kernel stages
// chunk ranges [h0, h1) and every wave keeps the accumulators of its (up to MAXU) units across the phases. A lane whose
// item of a step lies outside the phase (phase ends that are not multiples of 16 items, K % 1024 != 0) contributes
// dw = 0: dwdy = +-0, term = +-0, and acc + (+-0) == acc (acc starts at +0 and a round-to-nearest sum is never -0
// unless both operands are), so the lane's sequence of items is exactly l, l + 16, .. of the row.
// LDS: codebook tables at address 0 (iq3gemv.cuh contract), then the column records of the phase (nc::lay layout with a
// runtime column count). The next phase's records are loaded into registers while the current phase is computed.
#pragma once
#include "iq3gemv-nc.cuh"

namespace iq3g {
namespace mf {

typedef int   i32x4 __attribute__((ext_vector_type(4)));
typedef float f32x2 __attribute__((ext_vector_type(2)));

static __device__ __forceinline__ f32x2 pmul(const f32x2 a, const f32x2 b) {
#pragma clang fp contract(off)
    return a*b;
}
static __device__ __forceinline__ f32x2 padd(const f32x2 a, const f32x2 b) {
#pragma clang fp contract(off)
    return a + b;
}
static __device__ __forceinline__ f32x2 pfma(const f32x2 a, const f32x2 b, const f32x2 c) {
    return __builtin_elementwise_fma(a, b, c);
}
template <int i> static __device__ __forceinline__ uint32_t qb(const uint32_t v) {   // value of lane 4*(lane/4) + i
#if defined(__HIP_DEVICE_COMPILE__)
    return (uint32_t) __builtin_amdgcn_mov_dpp((int) v, i | (i << 2) | (i << 4) | (i << 6), 0xF, 0xF, false);
#else
    return v;
#endif
}
template <int i> static __device__ __forceinline__ float qbf(const float v) { return __uint_as_float(qb<i>(__float_as_uint(v))); }

// raw weight registers of one P = 2 item, common to both types (the third load: IQ3_S fb words, IQ3_XXS block d)
struct raw { uint4 q, f; uint2 g; };
template <class T>
static __device__ __forceinline__ raw load(const uint8_t * __restrict__ base, const uint32_t nh, const uint32_t c, const uint32_t cb,
        const int p) {
    raw r;
    r.q = *(const uint4 *) (base + c*32u + 16u*(uint32_t) p);
    r.f = *(const uint4 *) (base + (size_t) nh*32 + c*16u);
    if constexpr (T::TAB_WORDS == 8192) {
        r.g = *(const uint2 *) (base + (size_t) nh*48 + c*8u);
    } else {
        r.g.x = *(const uint16_t *) (base + (size_t) nh*48 + cb*2u);
        r.g.y = 0;
    }
    return r;
}
template <class T>
static __device__ __forceinline__ typename T::template dec<2> prep(const raw & r, const int p) {
    typename T::template ld<2> l;
    l.q0 = r.q; l.q1 = r.q; l.p = p;
    if constexpr (T::TAB_WORDS == 8192) {
        l.fa = r.f; l.fb = r.g;
    } else {
        l.a = r.f; l.d = r.g.x;
    }
    return T::template prep<2>(l);
}
// chunk / block-chunk index of item it of unit uu, row q of the unit (= nc::cidx with RG = 4)
template <int SM>
static __device__ __forceinline__ void cidx(const int uu, const int it, const int items, const int U, const int q, uint32_t & c,
        uint32_t & cb, int & p) {
    nc::cidx<2, 4, SM>(uu, it, items, U, q, c, cb, p);
}

// per-matrix factors of one decoded item: dw (0 for a masked lane) and the scale factors of sub-blocks 0 / 1 of the lane's
// row, broadcast to the 4 rows of the quad (rows 0,1 and 2,3 as packed pairs)
struct rowf { f32x2 dw01, dw23, ma01, ma23, mb01, mb23; };
template <class T>
static __device__ __forceinline__ rowf row_factors(const typename T::template dec<2> & d, const float dwv) {
    float m0, m1;
    if constexpr (T::TAB_WORDS == 8192) {
        m0 = (float) (1 + 2*(int) (d.sc & 0xFu));
        m1 = (float) (1 + 2*(int) ((d.sc >> 4) & 0xFu));
    } else {
        const int ls0 = dp4a((int) ((d.a[0] >> 7) & 0x01010101u), 0x08040201, 0);
        const int ls1 = dp4a((int) ((d.a[1] >> 7) & 0x01010101u), 0x08040201, 0);
        m0 = (float) (2*ls0 + 1) * 0.25f;
        m1 = (float) (2*ls1 + 1) * 0.25f;
    }
    rowf f;
    f.dw01 = { qbf<0>(dwv), qbf<1>(dwv) }; f.dw23 = { qbf<2>(dwv), qbf<3>(dwv) };
    f.ma01 = { qbf<0>(m0),  qbf<1>(m0)  }; f.ma23 = { qbf<2>(m0),  qbf<3>(m0)  };
    f.mb01 = { qbf<0>(m1),  qbf<1>(m1)  }; f.mb23 = { qbf<2>(m1),  qbf<3>(m1)  };
    return f;
}
static __device__ __forceinline__ i32x4 mfma8(const int (&t)[8], const int4 y0, const int4 y1) {
    i32x4 s = {0, 0, 0, 0};
#if defined(__HIP_DEVICE_COMPILE__)
    s = __builtin_amdgcn_mfma_i32_4x4x4i8(t[0], y0.x, s, 0, 0, 0);
    s = __builtin_amdgcn_mfma_i32_4x4x4i8(t[1], y0.y, s, 0, 0, 0);
    s = __builtin_amdgcn_mfma_i32_4x4x4i8(t[2], y0.z, s, 0, 0, 0);
    s = __builtin_amdgcn_mfma_i32_4x4x4i8(t[3], y0.w, s, 0, 0, 0);
    s = __builtin_amdgcn_mfma_i32_4x4x4i8(t[4], y1.x, s, 0, 0, 0);
    s = __builtin_amdgcn_mfma_i32_4x4x4i8(t[5], y1.y, s, 0, 0, 0);
    s = __builtin_amdgcn_mfma_i32_4x4x4i8(t[6], y1.z, s, 0, 0, 0);
    s = __builtin_amdgcn_mfma_i32_4x4x4i8(t[7], y1.w, s, 0, 0, 0);
#else
    (void) t; (void) y0; (void) y1;
#endif
    return s;
}
// float stage of one (sub-block, column group) for the 4 rows of the quad: acc[i] for rows i
template <bool XXS, bool FMA>
static __device__ __forceinline__ void fstage(float (&acc)[4], const i32x4 s, const f32x2 m01, const f32x2 m23, const f32x2 dw01,
        const f32x2 dw23, const float dy) {
    const f32x2 dyy = { dy, dy };
    f32x2 sv01 = pmul((f32x2) { (float) s[0], (float) s[1] }, m01);
    f32x2 sv23 = pmul((f32x2) { (float) s[2], (float) s[3] }, m23);
    if constexpr (XXS) {
        sv01 = (f32x2) { truncf(sv01.x), truncf(sv01.y) };
        sv23 = (f32x2) { truncf(sv23.x), truncf(sv23.y) };
    }
    const f32x2 dd01 = pmul(dw01, dyy), dd23 = pmul(dw23, dyy);
    f32x2 a01 = { acc[0], acc[1] }, a23 = { acc[2], acc[3] };
    if constexpr (FMA) {
        a01 = pfma(dd01, sv01, a01);
        a23 = pfma(dd23, sv23, a23);
    } else {
        a01 = padd(a01, pmul(dd01, sv01));
        a23 = padd(a23, pmul(dd23, sv23));
    }
    acc[0] = a01.x; acc[1] = a01.y; acc[2] = a23.x; acc[3] = a23.y;
}

// one item (2 sub-blocks) of M matrices (M = 1, or 2: rows of two units / gate and up) that read the same activation
// item, for CG column groups. ych: LDS address of the item's chunk record of column 0; coff[g]: byte offset of the lane's
// column of group g; p: the item's half of the chunk.
template <class TA, int TOA, class TB, int TOB, int M, int CG, bool FMA>
static __device__ __forceinline__ void item(float (&acc)[M][CG][4], const typename TA::template dec<2> & da,
        const typename TB::template dec<2> & db, const float dwa, const float dwb, const int p, const uint32_t ych,
        const uint32_t (&coff)[CG]) {
    const rowf fa = row_factors<TA>(da, dwa);
    rowf fb;
    if constexpr (M == 2) fb = row_factors<TB>(db, dwb);
    f32x2 dyv[CG];
#pragma unroll
    for (int g = 0; g < CG; ++g) {
#if defined(__HIP_DEVICE_COMPILE__)
        const float2 v = *(__attribute__((address_space(3))) const float2 *) (size_t) (ych + coff[g] + 128 + 8*p);
        dyv[g] = { v.x, v.y };
#else
        dyv[g] = { 0.0f, 0.0f };
#endif
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        int ta[8], tb[8];
        nc::look<TA, 2, TOA>(da, j, ta);
        if constexpr (M == 2) nc::look<TB, 2, TOB>(db, j, tb);
#pragma unroll
        for (int g = 0; g < CG; ++g) {
            const uint32_t ya = ych + coff[g] + 32*(2*p + j);
            const int4 y0 = nc::lds_i4(ya), y1 = nc::lds_i4(ya + 16);
            const float dy = j ? dyv[g].y : dyv[g].x;
            const i32x4 sa = mfma8(ta, y0, y1);
            i32x4 sb;
            if constexpr (M == 2) sb = mfma8(tb, y0, y1);
            fstage<TA::TAB_WORDS != 8192, FMA>(acc[0][g], sa, j ? fa.mb01 : fa.ma01, j ? fa.mb23 : fa.ma23, fa.dw01, fa.dw23, dy);
            if constexpr (M == 2) {
                fstage<TB::TAB_WORDS != 8192, FMA>(acc[M - 1][g], sb, j ? fb.mb01 : fb.ma01, j ? fb.mb23 : fb.ma23, fb.dw01, fb.dw23, dy);
            }
        }
    }
}

// transposed group_sum<16> over the item lanes (lane bits 2..5) of the 4*CG values v[g*4 + i]; afterwards the lane holds
// rcnt<CG>() values, value k = original index ridx(lane) + k.
template <int CG> static constexpr int rcnt() {
    int n = 4*CG;
    for (int s = 0; s < 4; ++s) if (n % 2 == 0 && n > 1) n /= 2;
    return n;
}
template <int N, int ST>
static __device__ __forceinline__ void rstep(float * v, const int lane) {
    if constexpr (ST < 4) {
        constexpr int m = 4 << ST;
        if constexpr (N % 2 == 0 && N > 1) {
            constexpr int H = N/2;
            const bool hi = (lane & m) != 0;
#pragma unroll
            for (int i = 0; i < H; ++i) {
                const float snd = hi ? v[i] : v[i + H];
                const float rcv = __shfl_xor(snd, m, 64);
                v[i] = (hi ? v[i + H] : v[i]) + rcv;
            }
            rstep<H, ST + 1>(v, lane);
        } else {
#pragma unroll
            for (int i = 0; i < N; ++i) v[i] += __shfl_xor(v[i], m, 64);
            rstep<N, ST + 1>(v, lane);
        }
    }
}
template <int CG>
static __device__ __forceinline__ int ridx(const int lane) {   // original value index of the lane's first kept value
    int n = 4*CG, base = 0;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        if (n % 2 == 0 && n > 1) { n /= 2; if (lane & (4 << s)) base += n; }
    }
    return base;
}
template <int CG>
static __device__ __forceinline__ bool rstore(const int lane) {   // lane bits of the plain (non-halving) steps all clear
    int n = 4*CG, mask = 0;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        if (n % 2 == 0 && n > 1) n /= 2; else mask |= 4 << s;
    }
    return (lane & mask) == 0;
}

// ------------------------------------------------------------------------------------------------ kernel
// MODE 0: plain (single / multi-job): non-contracted accumulation, out = v (+ aux residual). T1 != none_t: per job tsel.
// MODE 1: gate/up SwiGLU (T0 = gate type, weights w; T1 = up type, weights w2): fma accumulation, out = vb * silu(v).
// MODE 2: gate pass of a split gate/up (fma accumulation, out = v, raw), MODE 3: up pass (fma, out = v * silu(aux)).
// R: units per wave step (MODE 0/2/3 with one type): their items share the activation reads.
// Work: rounds of R units per wave (unit of round r, slot i: u0 + (r*R + i)*ustep), each round over the K phases; the
// records are restaged per round only when there is more than one phase. One accumulator set per (unit, matrix).
static constexpr int MJ = 3;
struct job {
    const uint8_t * w;
    const uint8_t * w2;
    float         * dst;
    const float   * aux;
    uint32_t        nh;
    int             ubeg;   // unused jobs: INT_MAX
    int             tsel;
    int             U;
    int             dcs;
    int             acs;
};
struct args {
    job                job[MJ];
    const block_q8_1 * y;
    int                ycb, nc, K, hph, utot;
};

static __host__ __device__ constexpr int csb(const int nc) { return nc*nc::lay<1>::RECB + (nc % 2 == 0 ? 16 : 0); }
static constexpr int SPT = 2;   // q8 blocks staged per thread and phase (host: hph*nc*4 <= SPT*NT)

template <class T0, class T1, int CG, int SM, int MODE, int NW, int R>
__launch_bounds__(NW*64, 1)
__global__ void gemv(const args a) {
    constexpr bool HAS1 = !std::is_same<T1, none_t>::value;
    constexpr bool TWO  = HAS1 && !std::is_same<T0, T1>::value;
    constexpr bool GLU  = MODE == 1;
    constexpr bool FMA  = MODE != 0;
    constexpr bool SEL  = HAS1 && !GLU;            // per-job type
    constexpr int M     = GLU ? 2 : R;             // matrices per step
    static_assert(R == 1 || (!HAS1 && !GLU), "R > 1: one type, no gate/up");
    constexpr int TOFF0 = 0;
    constexpr int TOFF1 = TWO ? T0::TAB_WORDS*4 : 0;
    constexpr int YOFF  = T0::TAB_WORDS*4 + (TWO ? T1::TAB_WORDS*4 : 0);
    constexpr int NT    = NW*64;
    using TB = typename std::conditional<GLU, T1, T0>::type;   // type of the second matrix (M == 2)
    constexpr int TOB   = GLU ? TOFF1 : TOFF0;
    extern __shared__ uint32_t smem[];
#if defined(__HIP_DEVICE_COMPILE__)
    if ((uint32_t) (size_t) (__attribute__((address_space(3))) uint32_t *) smem != 0) __builtin_trap();
    const int w = __builtin_amdgcn_readfirstlane(threadIdx.y);
#else
    const int w = threadIdx.y;
#endif
    const int tid   = threadIdx.y*64 + threadIdx.x;
    const int lane  = threadIdx.x;
    const int l     = lane >> 2, q = lane & 3;
    const int nc    = a.nc;
    const int items = a.K / 128;
    const int ipr   = 2*items;
    const int hph   = a.hph;
    const int cs    = csb(nc);
    const int ustep = gridDim.x*NW;
    const int u0    = w*gridDim.x + blockIdx.x;
    const int nph   = (items + hph - 1)/hph;
    const int nrd   = (a.utot + ustep*R - 1)/(ustep*R);
    const int nq    = nph*nrd;
    const int p     = l & 1;                        // half of the chunk of every item of this lane

    uint32_t coff[CG];
#pragma unroll
    for (int g = 0; g < CG; ++g) coff[g] = (uint32_t) min(4*g + q, nc - 1)*nc::lay<1>::RECB;

    // ---- staging of a phase's records: loads into registers, LDS writes later
    uint4 sq0[SPT], sq1[SPT];
    uint32_t sds[SPT];
    auto stage_load = [&](const int h0, const int h1) {
        const int nb = (h1 - h0)*4;
#pragma unroll
        for (int k = 0; k < SPT; ++k) {
            const int idx = min(tid + k*NT, nc*nb - 1);   // clamped: unconditional loads
            const int c = idx / nb, s = idx - c*nb;
            const uint8_t * b = (const uint8_t *) (a.y + (size_t) c*a.ycb + 4*h0 + s);
            sq0[k] = *(const uint4 *) (b + 4);
            sq1[k] = *(const uint4 *) (b + 20);
            sds[k] = ld_u32_opaque(b);
        }
    };
    auto stage_store = [&](const int h0, const int h1) {
        const int nb = (h1 - h0)*4;
#pragma unroll
        for (int k = 0; k < SPT; ++k) {
            const int idx = tid + k*NT;
            if (idx < nc*nb) {
                const int c = idx / nb, s = idx - c*nb;
                int * rec = (int *) (smem + (YOFF + (uint32_t) (s >> 2)*cs + (uint32_t) c*nc::lay<1>::RECB)/4);
                int * r = rec + 8*(s & 3);
                *(int4 *) r       = make_int4((int) sq0[k].x, (int) sq0[k].y, (int) sq0[k].z, (int) sq0[k].w);
                *(int4 *) (r + 4) = make_int4((int) sq1[k].x, (int) sq1[k].y, (int) sq1[k].z, (int) sq1[k].w);
                ((float *) rec)[32 + (s & 3)] = h2f(sds[k] & 0xFFFFu);
            }
        }
    };

    // ---- unit bookkeeping (wave-uniform)
    struct ustate { int j, lu; bool ok; };
    auto unit_of = [&](const int r, const int i) -> ustate {
        const int u = u0 + (r*R + i)*ustep;
        ustate s;
        s.ok = u < a.utot;
        const int uc = s.ok ? u : u0 % a.utot;     // invalid: any valid unit (loads stay in bounds, results unused)
        s.j  = (uc >= a.job[1].ubeg) + (uc >= a.job[2].ubeg);
        s.lu = uc - a.job[s.j].ubeg;
        return s;
    };
    // weight loads of item it (step s) of a unit: A (T0 or T1 by tsel; GLU: T0 from w) and, GLU, B (T1 from w2)
    auto wload = [&](const ustate & us, const int s, raw & ra, raw & rb) {
        const job & J = a.job[us.j];
        uint32_t c, cb;
        if constexpr (SM == 8) {
            const uint32_t sb = (uint32_t) s*(uint32_t) J.U + (uint32_t) us.lu;
            c  = sb*32u + (uint32_t) (l >> 1)*4u + (uint32_t) q;
            cb = sb*16u + (uint32_t) (l >> 2)*4u + (uint32_t) q;
        } else {
            int pp;
            cidx<SM>(us.lu, min(16*s + l, ipr - 1), items, J.U, q, c, cb, pp);
        }
        if constexpr (GLU) {
            ra = load<T0>(J.w, J.nh, c, cb, p);
            rb = load<T1>(J.w2, J.nh, c, cb, p);
        } else if constexpr (SEL) {
            if (J.tsel == 0) ra = load<T0>(J.w, J.nh, c, cb, p);
            else             ra = load<T1>(J.w, J.nh, c, cb, p);
        } else {
            ra = load<T0>(J.w, J.nh, c, cb, p);
        }
    };

    float acc[M][CG][4];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
        for (int g = 0; g < CG; ++g)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[m][g][i] = 0.0f;

    // current / next item registers: [unit slot or gate/up]
    raw cur[M], nxt[M];
    ustate us[R];
#pragma unroll
    for (int i = 0; i < R; ++i) us[i] = unit_of(0, i);
    {   // first step of round 0, phase 0
        if constexpr (GLU) wload(us[0], 0, cur[0], cur[1]);
        else {
#pragma unroll
            for (int i = 0; i < R; ++i) { raw dummy; wload(us[i], 0, cur[i], dummy); }
        }
    }
    // prologue: tables + first phase records
    {
        const int h1 = min(items, hph);
        stage_load(0, h1);
        nc::stage_tables<T0, T1, TWO, NT>(smem, TOFF1, tid);
        stage_store(0, h1);
    }
    __syncthreads();

    for (int qi = 0; qi < nq; ++qi) {
        const int r  = qi / nph, ph = qi - r*nph;
        const int h0 = ph*hph, h1 = min(items, h0 + hph);
        const bool more = qi + 1 < nq;
        const int rn = (qi + 1)/nph, phn = (qi + 1) - rn*nph;          // next phase
        const int h0n = phn*hph, h1n = min(items, h0n + hph);
        if (nph > 1 && more) stage_load(h0n, h1n);
        ustate usn[R];
#pragma unroll
        for (int i = 0; i < R; ++i) usn[i] = unit_of(rn, i);
        const int it0 = 2*h0, it1 = 2*h1;
        const int s0 = it0 >> 4, s1 = (it1 - 1) >> 4;
        const int s0n = (2*h0n) >> 4;
        if (us[0].ok) {
            for (int s = s0; s <= s1; ++s) {
                // prefetch: next step of this phase, else the first step of the next phase (unconditional loads)
                const bool last = s == s1;
                if constexpr (GLU) {
                    wload(last ? usn[0] : us[0], last ? s0n : s + 1, nxt[0], nxt[1]);
                } else {
#pragma unroll
                    for (int i = 0; i < R; ++i) { raw dummy; wload(last ? usn[i] : us[i], last ? s0n : s + 1, nxt[i], dummy); }
                }
                const int it = 16*s + l;
                const bool valid = it >= it0 && it < it1;
                const int itc = min(max(it, it0), it1 - 1);
                const uint32_t ych = YOFF + (uint32_t) ((itc >> 1) - h0)*cs;
                if constexpr (GLU) {
                    const typename T0::template dec<2> da = prep<T0>(cur[0], p);
                    const typename T1::template dec<2> db = prep<T1>(cur[1], p);
                    item<T0, TOFF0, T1, TOFF1, 2, CG, true>(acc, da, db, valid ? T0::template dw<2>(da) : 0.0f,
                            valid ? T1::template dw<2>(db) : 0.0f, p, ych, coff);
                } else if constexpr (R == 2) {
                    const typename T0::template dec<2> da = prep<T0>(cur[0], p);
                    const typename T0::template dec<2> db = prep<T0>(cur[1], p);
                    item<T0, TOFF0, T0, TOFF0, 2, CG, FMA>(acc, da, db, valid ? T0::template dw<2>(da) : 0.0f,
                            valid && us[1].ok ? T0::template dw<2>(db) : 0.0f, p, ych, coff);
                } else {
                    if (!SEL || a.job[us[0].j].tsel == 0) {
                        const typename T0::template dec<2> da = prep<T0>(cur[0], p);
                        item<T0, TOFF0, T0, TOFF0, 1, CG, FMA>(acc, da, da, valid ? T0::template dw<2>(da) : 0.0f, 0.0f, p, ych, coff);
                    } else {
                        if constexpr (SEL) {
                            const typename T1::template dec<2> da = prep<T1>(cur[0], p);
                            item<T1, TOFF1, T1, TOFF1, 1, CG, FMA>(acc, da, da, valid ? T1::template dw<2>(da) : 0.0f, 0.0f, p, ych, coff);
                        }
                    }
                }
#pragma unroll
                for (int m = 0; m < M; ++m) cur[m] = nxt[m];
            }
        }
        if (ph == nph - 1) {
            // ---- end of the round: reduction and epilogue of its units
            constexpr int RC = rcnt<CG>();
            const int rb0 = ridx<CG>(lane);
            const bool st = rstore<CG>(lane);
#pragma unroll
            for (int i = 0; i < (GLU ? 1 : R); ++i) {
                if (!us[i].ok) continue;
                const job & J = a.job[us[i].j];
                float v[4*CG], vb[GLU ? 4*CG : 1];
#pragma unroll
                for (int g = 0; g < CG; ++g) {
#pragma unroll
                    for (int k = 0; k < 4; ++k) { v[4*g + k] = acc[i][g][k]; if constexpr (GLU) vb[4*g + k] = acc[1][g][k]; }
                }
                rstep<4*CG, 0>(v, lane);
                if constexpr (GLU) rstep<4*CG, 0>(vb, lane);
                if (st) {
#pragma unroll
                    for (int k = 0; k < RC; ++k) {
                        const int idx = rb0 + k;
                        const int col = 4*(idx >> 2) + q, row = us[i].lu*4 + (idx & 3);
                        if (col < nc) {
                            float o;
                            if constexpr (GLU) {
                                o = vb[k] * silu(v[k]);
                            } else if constexpr (MODE == 3) {
                                o = v[k] * silu(J.aux[(size_t) col*J.acs + row]);
                            } else if constexpr (MODE == 0) {
                                o = J.aux ? v[k] + J.aux[(size_t) col*J.acs + row] : v[k];
                            } else {
                                o = v[k];
                            }
                            J.dst[(size_t) col*J.dcs + row] = o;
                        }
                    }
                }
            }
#pragma unroll
            for (int m = 0; m < M; ++m)
#pragma unroll
                for (int g = 0; g < CG; ++g)
#pragma unroll
                    for (int k = 0; k < 4; ++k) acc[m][g][k] = 0.0f;
        }
        if (nph > 1 && more) {
            __syncthreads();
            stage_store(h0n, h1n);
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < R; ++i) us[i] = usn[i];
    }
}

template <class T0, class T1>
static inline size_t tab_bytes() { return nc::tab_bytes<T0, T1>(); }

} // namespace mf
} // namespace iq3g
