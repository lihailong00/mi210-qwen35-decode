// iq3gemv-mf2.cuh - N-column (NC = 2..16) IQ3_S / IQ3_XXS GEMVs on the int8 matrix cores, second form (MMVQ-NC A).
// Same work split (unit = 4 rows, 16 item lanes, one unit per wave and step), item order, integer and float arithmetic
// and reduction tree as iq3gemv-mf.cuh (see there for the per-column bit-identity argument); what differs is where the
// values live and how the loop is driven:
//   - transposed MFMA operands: the q8 words are the A operand (lane q of a quad = column q of the group) and the
//     sign-expanded codebook words the B operand (lane q = row q of the unit), so D gives lane q the sums of its own row
//     for the 4 columns of the group. The row factors (dw, sub-block scales) are the lane's own values: no quad
//     broadcasts. The dy of the 4 columns of a sub-block come as one float4 from a transposed d slot (below).
//     Per (row, column, sub-block) the operations are the same: sv = (float) isum * m (XXS: truncf), dd = dw*dy,
//     term = dd*sv, acc + term (single / residual / multi-job, not contracted) or fma(dd, sv, acc) (gate/up passes).
//   - weight loads through a buffer resource over the matrix: lane offsets fixed for the whole kernel (VGPRs) plus
//     wave-uniform offsets of the step (SGPRs, set up once per phase and advanced by additions), so the step loop does
//     no address arithmetic on the VALU and no scalar loads (their lgkmcnt wait would also wait for the LDS reads).
//   - one compact step loop for every phase shape (the instruction cache is cold in the model: code size is time), the
//     next step's weights prefetched into a second register set; lanes outside a phase get dw = 0 and a clamped
//     record address.
//   - prologue in the order of the single-token kernel (PF 7): the grid words of the tables first (no kernel argument
//     needed), all kernel arguments in one scalar round trip, then the phase-0 records (owner threads only), then the
//     first weight step; loads return in order, so the tables wait only for the grid words.
// LDS: tables at 0 (iq3gemv.cuh contract), then per chunk the NC column records (q8 words at column stride 144 B, the
// 16 B pads unused) and after them 4*CG d slots of 16 B: slot 4g + b holds the d of sub-block b of columns 4g..4g+3.
// cs2() rounds the chunk stride to an odd multiple of 16 B (bank spread).
// Padding columns (nc % 4 != 0): their d slot entries are never written, so their accumulators hold garbage (possibly
// NaN / Inf). That is harmless only because every operation on the accumulators is per element (per column) and padding
// columns are never stored (col < nc); never combine values across columns in this kernel.
#pragma once
#include "iq3gemv-mf.cuh"

namespace iq3g {
namespace mf2 {

using mf::i32x4;
using mf::f32x2;
using mf::pmul;
using mf::padd;
using mf::pfma;
using mf::raw;
using mf::args;
using mf::job;
using mf::MJ;
using mf::SPT;

static __host__ __device__ constexpr int cgs(const int nc) { return (nc + 3)/4; }
static __host__ __device__ constexpr int cs2(const int nc) {
    const int b = nc*144 + 64*cgs(nc);
    return (b/16) % 2 == 0 ? b + 16 : b;
}
static __device__ __forceinline__ uint32_t slot_off(const int k, const int nc) { return (uint32_t) nc*144u + 16u*(uint32_t) k; }

static __device__ __forceinline__ float4 lds_f4(const uint32_t a) {   // one ds_read_b128 (16-byte aligned slot)
    const int4 v = nc::lds_i4(a);
    return make_float4(__int_as_float(v.x), __int_as_float(v.y), __int_as_float(v.z), __int_as_float(v.w));
}

// weight loads: buffer resource over the matrix (lane offsets o*, wave-uniform offsets s*)
#if __has_builtin(__builtin_amdgcn_make_buffer_rsrc) && __has_builtin(__builtin_amdgcn_raw_buffer_load_b128) && \
    __has_builtin(__builtin_amdgcn_raw_buffer_load_b64) && __has_builtin(__builtin_amdgcn_raw_buffer_load_b16)
typedef __amdgpu_buffer_rsrc_t wrsrc;
static __device__ __forceinline__ wrsrc mk_rsrc(const uint8_t * w, const uint32_t nh) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<uint8_t *>(w), 0, (int) (nh*56u), 0x00020000);
}
template <class T>
static __device__ __forceinline__ raw ldb(const wrsrc rs, const uint32_t oq, const uint32_t of, const uint32_t og,
        const uint32_t sq, const uint32_t sf, const uint32_t sg) {
    raw r;
    r.q = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(rs, (int) oq, (int) sq, 0));
    r.f = __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(rs, (int) of, (int) sf, 0));
    if constexpr (T::TAB_WORDS == 8192) {
        r.g = __builtin_bit_cast(uint2, __builtin_amdgcn_raw_buffer_load_b64(rs, (int) og, (int) sg, 0));
    } else {
        r.g.x = (uint32_t) __builtin_amdgcn_raw_buffer_load_b16(rs, (int) og, (int) sg, 0);
        r.g.y = 0;
    }
    return r;
}
#else
struct wrsrc { const uint8_t * w; };
static __device__ __forceinline__ wrsrc mk_rsrc(const uint8_t * w, const uint32_t nh) { (void) nh; return wrsrc{ w }; }
template <class T>
static __device__ __forceinline__ raw ldb(const wrsrc rs, const uint32_t oq, const uint32_t of, const uint32_t og,
        const uint32_t sq, const uint32_t sf, const uint32_t sg) {
    raw r;
    r.q = *(const uint4 *) (rs.w + sq + oq);
    r.f = *(const uint4 *) (rs.w + sf + of);
    if constexpr (T::TAB_WORDS == 8192) {
        r.g = *(const uint2 *) (rs.w + sg + og);
    } else {
        r.g.x = *(const uint16_t *) (rs.w + sg + og);
        r.g.y = 0;
    }
    return r;
}
#endif

// sub-block scale factors of the lane's own row (as mf::row_factors, without the quad broadcast)
template <class T>
static __device__ __forceinline__ void row_m(const typename T::template dec<2> & d, float & m0, float & m1) {
    if constexpr (T::TAB_WORDS == 8192) {
        m0 = (float) (1 + 2*(int) (d.sc & 0xFu));
        m1 = (float) (1 + 2*(int) ((d.sc >> 4) & 0xFu));
    } else {
        const int ls0 = dp4a((int) ((d.a[0] >> 7) & 0x01010101u), 0x08040201, 0);
        const int ls1 = dp4a((int) ((d.a[1] >> 7) & 0x01010101u), 0x08040201, 0);
        m0 = (float) (2*ls0 + 1) * 0.25f;
        m1 = (float) (2*ls1 + 1) * 0.25f;
    }
}
// the 8 chained MFMAs of one sub-block, q8 words as A, codebook words as B (exact int32 sums)
static __device__ __forceinline__ i32x4 mfma8t(const int4 y0, const int4 y1, const int (&t)[8]) {
    i32x4 s = {0, 0, 0, 0};
#if defined(__HIP_DEVICE_COMPILE__)
    s = __builtin_amdgcn_mfma_i32_4x4x4i8(y0.x, t[0], s, 0, 0, 0);
    s = __builtin_amdgcn_mfma_i32_4x4x4i8(y0.y, t[1], s, 0, 0, 0);
    s = __builtin_amdgcn_mfma_i32_4x4x4i8(y0.z, t[2], s, 0, 0, 0);
    s = __builtin_amdgcn_mfma_i32_4x4x4i8(y0.w, t[3], s, 0, 0, 0);
    s = __builtin_amdgcn_mfma_i32_4x4x4i8(y1.x, t[4], s, 0, 0, 0);
    s = __builtin_amdgcn_mfma_i32_4x4x4i8(y1.y, t[5], s, 0, 0, 0);
    s = __builtin_amdgcn_mfma_i32_4x4x4i8(y1.z, t[6], s, 0, 0, 0);
    s = __builtin_amdgcn_mfma_i32_4x4x4i8(y1.w, t[7], s, 0, 0, 0);
#else
    (void) t; (void) y0; (void) y1;
#endif
    return s;
}
// float stage of one (sub-block, column group): s[i] = isum of the lane's row for column 4g + i
template <bool XXS, bool FMA>
static __device__ __forceinline__ void fstage(float (&acc)[4], const i32x4 s, const float m, const float dw, const float4 dy) {
    const f32x2 mm = { m, m }, ww = { dw, dw };
    f32x2 sv01 = pmul((f32x2) { (float) s[0], (float) s[1] }, mm);
    f32x2 sv23 = pmul((f32x2) { (float) s[2], (float) s[3] }, mm);
    if constexpr (XXS) {
        sv01 = (f32x2) { truncf(sv01.x), truncf(sv01.y) };
        sv23 = (f32x2) { truncf(sv23.x), truncf(sv23.y) };
    }
    const f32x2 dd01 = pmul(ww, (f32x2) { dy.x, dy.y }), dd23 = pmul(ww, (f32x2) { dy.z, dy.w });
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

// one item (2 sub-blocks) of M matrices (M = 2: gate and up) on the same activation item, CG column groups.
// yb[g]: LDS address of the lane's A column record of group g (item chunk, incl. the item half); ydb: of the d slot of
// sub-block 2p of group 0 (group g, sub-block 2p + j: + 64g + 16j).
template <class TA, int TOA, class TB, int TOB, int M, int CG, bool FMA>
static __device__ __forceinline__ void item(float (&acc)[M][CG][4], const typename TA::template dec<2> & da,
        const typename TB::template dec<2> & db, const float dwa, const float dwb, const uint32_t (&yb)[CG],
        const uint32_t ydb) {
    float ma[2], mb[2] = { 0.0f, 0.0f };
    row_m<TA>(da, ma[0], ma[1]);
    if constexpr (M == 2) row_m<TB>(db, mb[0], mb[1]);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        if constexpr (M == 2) {
            if (j == 1) __builtin_amdgcn_sched_barrier(0);   // keep the two sub-blocks apart (register budget)
        }
        float4 dy[CG];
#pragma unroll
        for (int g = 0; g < CG; ++g) dy[g] = lds_f4(ydb + 64*g + 16*j);
        int ta[8], tb[8];
        nc::look<TA, 2, TOA>(da, j, ta);
        if constexpr (M == 2) nc::look<TB, 2, TOB>(db, j, tb);
#pragma unroll
        for (int g = 0; g < CG; ++g) {
            const uint32_t ya = yb[g] + 32*j;
            const int4 y0 = nc::lds_i4(ya), y1 = nc::lds_i4(ya + 16);
            const i32x4 sa = mfma8t(y0, y1, ta);
            fstage<TA::TAB_WORDS != 8192, FMA>(acc[0][g], sa, ma[j], dwa, dy[g]);
            if constexpr (M == 2) {
                const i32x4 sb = mfma8t(y0, y1, tb);
                fstage<TB::TAB_WORDS != 8192, FMA>(acc[M - 1][g], sb, mb[j], dwb, dy[g]);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ kernel
// MODE as mf::gemv: 0 plain / residual / multi-job (T1 != none_t: per-job type), 1 gate/up SwiGLU (T0 gate, T1 up),
// 2 gate pass, 3 up pass. One unit per wave and step; rounds of units (u0 + r*ustep), each round over the K phases.
template <class T0, class T1, int CG, int SM, int MODE, int NW>
__launch_bounds__(NW*64, 1)
__global__ void gemv(const args a) {
    constexpr bool HAS1 = !std::is_same<T1, none_t>::value;
    constexpr bool TWO  = HAS1 && !std::is_same<T0, T1>::value;
    constexpr bool GLU  = MODE == 1;
    constexpr bool FMA  = MODE != 0;
    constexpr bool SEL  = HAS1 && !GLU;            // per-job type
    constexpr int M     = GLU ? 2 : 1;             // matrices per step
    constexpr int TOFF0 = 0;
    constexpr int TOFF1 = TWO ? T0::TAB_WORDS*4 : 0;
    constexpr int YOFF  = T0::TAB_WORDS*4 + (TWO ? T1::TAB_WORDS*4 : 0);
    constexpr int NT    = NW*64;
    constexpr uint32_t RECB = nc::lay<1>::RECB;
    extern __shared__ uint32_t smem[];
#if defined(__HIP_DEVICE_COMPILE__)
    if ((uint32_t) (size_t) (__attribute__((address_space(3))) uint32_t *) smem != 0) __builtin_trap();
    const int w = __builtin_amdgcn_readfirstlane(threadIdx.y);
#else
    const int w = threadIdx.y;
#endif
    const int tid   = threadIdx.y*64 + threadIdx.x;
    const int lane  = threadIdx.x;
    // grid words of the tables first (their address needs no kernel argument)
    const uint32_t ga = grid_word<T0>(tid % (T0::TAB_WORDS/16));
    uint32_t gb = 0;
    if constexpr (TWO) gb = grid_word<T1>(tid % (T1::TAB_WORDS/16));
    pro_fence();
    // the job fields as opaque values, requested together with the other kernel arguments (one scalar round trip; the
    // selects below then cannot become loads from a selected address)
    const uint8_t * pjw[3]  = { a.job[0].w,  a.job[1].w,  a.job[2].w  };
    const uint8_t * pjw2[3] = { a.job[0].w2, a.job[1].w2, a.job[2].w2 };
    uint32_t pjnh[3] = { a.job[0].nh, a.job[1].nh, a.job[2].nh };
    int pjU[3]  = { a.job[0].U, a.job[1].U, a.job[2].U };
    int pjub[3] = { a.job[0].ubeg, a.job[1].ubeg, a.job[2].ubeg };
    int pjts[3] = { a.job[0].tsel, a.job[1].tsel, a.job[2].tsel };
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("" : "+s"(pjw[0]), "+s"(pjw[1]), "+s"(pjw[2]), "+s"(pjnh[0]), "+s"(pjnh[1]), "+s"(pjnh[2]),
            "+s"(pjU[0]), "+s"(pjU[1]), "+s"(pjU[2]), "+s"(pjub[1]), "+s"(pjub[2]) : "s"(a.y), "s"(a.ycb), "s"(a.nc),
            "s"(a.K), "s"(a.hph), "s"(a.utot), "s"(gridDim.x));
    if constexpr (GLU) asm volatile("" : "+s"(pjw2[0]), "+s"(pjw2[1]), "+s"(pjw2[2]));
    if constexpr (SEL) asm volatile("" : "+s"(pjts[0]), "+s"(pjts[1]), "+s"(pjts[2]));
#endif
    const int l     = lane >> 2, q = lane & 3;
    const int nc    = a.nc;
    const int items = a.K / 128;
    const int ipr   = 2*items;
    const int hph   = a.hph;
    const int cs    = cs2(nc);
    const int ustep = gridDim.x*NW;
    const int u0    = w*gridDim.x + blockIdx.x;
    const int nph   = (items + hph - 1)/hph;
    const int nrd   = (a.utot + ustep - 1)/ustep;
    const int nq    = nph*nrd;
    const int p     = l & 1;                        // half of the chunk of every item of this lane

    uint32_t yq[CG];
#pragma unroll
    for (int g = 0; g < CG; ++g) yq[g] = (uint32_t) min(4*g + q, nc - 1)*RECB + 64u*(uint32_t) p;
    const uint32_t yd = slot_off(2*p, nc);
    // lane offsets of the weight loads (SM = 8: chunk c = (s*U + lu)*32 + cl, block chunk cb = (s*U + lu)*16 + cbl)
    const uint32_t cl  = (uint32_t) (l >> 1)*4u + (uint32_t) q;
    const uint32_t cbl = (uint32_t) (l >> 2)*4u + (uint32_t) q;
    const uint32_t oq  = cl*32u + 16u*(uint32_t) p;
    const uint32_t of  = cl*16u;

    // ---- staging of a phase's records (owner threads only): loads into registers, LDS writes later
    uint4 sq0[SPT], sq1[SPT];
    uint32_t sds[SPT];
    auto stage_load = [&](const int h0, const int h1) {
        const int nb = (h1 - h0)*4;
#pragma unroll
        for (int k = 0; k < SPT; ++k) {
            const int idx = tid + k*NT;
            if (idx < nc*nb) {
                const int c = idx / nb, s = idx - c*nb;
                const uint8_t * b = (const uint8_t *) (a.y + (size_t) c*a.ycb + 4*h0 + s);
                sq0[k] = *(const uint4 *) (b + 4);
                sq1[k] = *(const uint4 *) (b + 20);
                sds[k] = ld_u32_opaque(b);
            }
        }
    };
    auto stage_store = [&](const int h0, const int h1) {
        const int nb = (h1 - h0)*4;
#pragma unroll
        for (int k = 0; k < SPT; ++k) {
            const int idx = tid + k*NT;
            if (idx < nc*nb) {
                const int c = idx / nb, s = idx - c*nb;
                const uint32_t ch = YOFF + (uint32_t) (s >> 2)*cs;
                int * r = (int *) (smem + (ch + (uint32_t) c*RECB)/4) + 8*(s & 3);
                *(int4 *) r       = make_int4((int) sq0[k].x, (int) sq0[k].y, (int) sq0[k].z, (int) sq0[k].w);
                *(int4 *) (r + 4) = make_int4((int) sq1[k].x, (int) sq1[k].y, (int) sq1[k].z, (int) sq1[k].w);
                ((float *) smem)[(ch + slot_off(4*(c >> 2) + (s & 3), nc))/4 + (c & 3)] = h2f(sds[k] & 0xFFFFu);
            }
        }
    };

    // ---- units (wave-uniform): job, local unit, valid
    struct ust { int j, lu, ok; };
    auto unit_of = [&](const int r) -> ust {
        const int u = u0 + r*ustep;
        ust s;
        s.ok = u < a.utot;
        const int uc = s.ok ? u : u0 % a.utot;     // invalid: any valid unit (loads stay in bounds, results unused)
        s.j  = (uc >= pjub[1]) + (uc >= pjub[2]);
        s.lu = uc - (s.j == 0 ? 0 : s.j == 1 ? pjub[1] : pjub[2]);   // (job 0 starts at unit 0)
        return s;
    };
    // load state of one step of a unit: buffer resources and the wave-uniform offsets (SM = 8)
    struct lst { wrsrc rs, rs2; uint32_t sq, sf, sg, sg2; int tsel; };   // sg2: third load of the up matrix (GLU)
    auto lstate = [&](const ust & us, const int s) -> lst {
        const int jj = us.j;
        const uint8_t * jw  = jj == 0 ? pjw[0]  : jj == 1 ? pjw[1]  : pjw[2];
        const uint8_t * jw2 = jj == 0 ? pjw2[0] : jj == 1 ? pjw2[1] : pjw2[2];
        const uint32_t  jnh = jj == 0 ? pjnh[0] : jj == 1 ? pjnh[1] : pjnh[2];
        const int       jU  = jj == 0 ? pjU[0]  : jj == 1 ? pjU[1]  : pjU[2];
        lst t;
        t.tsel = SEL ? (jj == 0 ? pjts[0] : jj == 1 ? pjts[1] : pjts[2]) : 0;
        const uint32_t sb = (uint32_t) s*(uint32_t) jU + (uint32_t) us.lu;
        // Q at c*32 (+16p), FA at nh*32 + c*16, third load: IQ3_S FB at nh*48 + c*8, IQ3_XXS d at nh*48 + cb*2
        const bool xxs = SEL ? t.tsel != 0 : T0::TAB_WORDS != 8192;
        t.rs  = mk_rsrc(jw, jnh);
        t.rs2 = mk_rsrc(GLU ? jw2 : jw, jnh);
        t.sq  = sb*1024u;
        t.sf  = jnh*32u + sb*512u;
        t.sg  = jnh*48u + sb*(xxs ? 32u : 256u);
        t.sg2 = jnh*48u + sb*(T1::TAB_WORDS == 4096 ? 32u : 256u);
        return t;
    };
    // weight loads of one step (A: T0 or T1 by tsel; GLU: B = T1 from w2). SM = 0 (unit-major layout, small test
    // shapes): per-lane chunk indices as mf::gemv
    auto wload = [&](const ust & us, const int s, const lst & t, raw & ra, raw & rb) {
        if constexpr (SM == 8) {
            (void) us; (void) s;
            auto ld1 = [&](const wrsrc rs, auto tt, const uint32_t sg) -> raw {
                using T = decltype(tt);
                return ldb<T>(rs, oq, of, T::TAB_WORDS == 8192 ? cl*8u : cbl*2u, t.sq, t.sf, sg);
            };
            if constexpr (GLU) {
                ra = ld1(t.rs, T0{}, t.sg);
                rb = ld1(t.rs2, T1{}, t.sg2);
            } else if constexpr (SEL) {
                if (t.tsel == 0) ra = ld1(t.rs, T0{}, t.sg);
                else             ra = ld1(t.rs, T1{}, t.sg);
            } else {
                ra = ld1(t.rs, T0{}, t.sg);
            }
        } else {
            const job & J = a.job[us.j];
            uint32_t c, cb;
            int pp;
            mf::cidx<SM>(us.lu, min(16*s + l, ipr - 1), items, J.U, q, c, cb, pp);
            if constexpr (GLU) {
                ra = mf::load<T0>(J.w, J.nh, c, cb, p);
                rb = mf::load<T1>(J.w2, J.nh, c, cb, p);
            } else if constexpr (SEL) {
                if (J.tsel == 0) ra = mf::load<T0>(J.w, J.nh, c, cb, p);
                else             ra = mf::load<T1>(J.w, J.nh, c, cb, p);
            } else {
                ra = mf::load<T0>(J.w, J.nh, c, cb, p);
            }
        }
    };

    float acc[M][CG][4];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
        for (int g = 0; g < CG; ++g)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[m][g][i] = 0.0f;

    // one step: compute with the registers in `use` (type by tsel), prefetch the step of `tn` into `fill`.
    // ych: LDS address of the lane's item chunk; valid: the lane's item is inside the phase
    auto step = [&](raw (&use)[M], raw (&fill)[M], const int tsel, const ust & un, const int sn, const lst & tn,
            const uint32_t ych, const bool valid) {
        uint32_t yb[CG];
#pragma unroll
        for (int g = 0; g < CG; ++g) yb[g] = ych + yq[g];
        const uint32_t ydb = ych + yd;
        {
            raw dummy;
            wload(un, sn, tn, fill[0], GLU ? fill[M - 1] : dummy);
        }
        if constexpr (GLU) {
            const typename T0::template dec<2> da = mf::prep<T0>(use[0], p);
            const typename T1::template dec<2> db = mf::prep<T1>(use[1], p);
            item<T0, TOFF0, T1, TOFF1, 2, CG, true>(acc, da, db, valid ? T0::template dw<2>(da) : 0.0f,
                    valid ? T1::template dw<2>(db) : 0.0f, yb, ydb);
        } else {
            if (!SEL || tsel == 0) {
                const typename T0::template dec<2> da = mf::prep<T0>(use[0], p);
                item<T0, TOFF0, T0, TOFF0, 1, CG, FMA>(acc, da, da, valid ? T0::template dw<2>(da) : 0.0f, 0.0f, yb, ydb);
            } else {
                if constexpr (SEL) {
                    const typename T1::template dec<2> da = mf::prep<T1>(use[0], p);
                    item<T1, TOFF1, T1, TOFF1, 1, CG, FMA>(acc, da, da, valid ? T1::template dw<2>(da) : 0.0f, 0.0f, yb, ydb);
                }
            }
        }
    };

    // ---- prologue: records of phase 0 (the grid words are in flight), the first weight step, tables, records
    raw ra[M], rb[M];
    ust us = unit_of(0);
    {
        const int h1 = min(items, hph);
        stage_load(0, h1);
        pro_fence();
        wload(us, 0, lstate(us, 0), ra[0], ra[M - 1]);
        pro_fence();
        tab_store<T0, NT>(smem, ga, tid);
        if constexpr (TWO) tab_store<T1, NT>(smem + TOFF1/4, gb, tid);
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
        const ust usn = phn == 0 ? unit_of(rn) : us;
        const int it0 = 2*h0, it1 = 2*h1;
        const int s0 = it0 >> 4, s1 = (it1 - 1) >> 4;
        const int s0n = (2*h0n) >> 4;
        if (us.ok) {
            // load states of this phase (advanced by additions) and of the next phase's first step
            lst tc = lstate(us, s0 + 1);
            const lst tn = lstate(usn, s0n);
            const uint32_t pjUc = us.j == 0 ? pjU[0] : us.j == 1 ? pjU[1] : pjU[2];
            const bool xxs = SEL ? tc.tsel != 0 : T0::TAB_WORDS != 8192;
            const uint32_t dq = pjUc*1024u, df = pjUc*512u, dg = pjUc*(xxs ? 32u : 256u);
            const uint32_t dg2 = pjUc*(T1::TAB_WORDS == 4096 ? 32u : 256u);
            // the chunk address advances by 8 chunks per step; lanes outside the phase get dw = 0 and an address
            // clamped into the staged records
            uint32_t ych = YOFF + (uint32_t) ((l >> 1) - (h0 & 7))*cs;
#if defined(__HIP_DEVICE_COMPILE__)
            asm volatile("" : "+v"(ych));   // keep YOFF in the base register (ds_read offsets stay small)
#endif
            const uint32_t dst8 = 8u*(uint32_t) cs, ymax = YOFF + (uint32_t) (h1 - h0 - 1)*cs;
            uint32_t itr = (uint32_t) (16*s0 + l - it0);   // item index relative to the phase (wraps below 0)
            const uint32_t nit = (uint32_t) (it1 - it0);
            for (int s = s0; s <= s1; ++s) {
                const bool last = s == s1;
                const bool valid = itr < nit;
                step(ra, rb, tc.tsel, last ? usn : us, last ? s0n : s + 1, last ? tn : tc, min(ych, ymax), valid);
#pragma unroll
                for (int m = 0; m < M; ++m) ra[m] = rb[m];
                ych += dst8; itr += 16u;
                tc.sq += dq; tc.sf += df; tc.sg += dg; tc.sg2 += dg2;
            }
        }
        if (ph == nph - 1) {
            // ---- end of the round: reduction and epilogue of its unit
            constexpr int RC = mf::rcnt<CG>();
            const int rb0 = mf::ridx<CG>(lane);
            const bool st = mf::rstore<CG>(lane);
            if (us.ok) {
                const job & J = a.job[us.j];
                float v[4*CG], vb[GLU ? 4*CG : 1];
#pragma unroll
                for (int g = 0; g < CG; ++g) {
#pragma unroll
                    for (int k = 0; k < 4; ++k) { v[4*g + k] = acc[0][g][k]; if constexpr (GLU) vb[4*g + k] = acc[1][g][k]; }
                }
                mf::rstep<4*CG, 0>(v, lane);
                if constexpr (GLU) mf::rstep<4*CG, 0>(vb, lane);
                if (st) {
                    const int row = us.lu*4 + q;
#pragma unroll
                    for (int k = 0; k < RC; ++k) {
                        const int col = rb0 + k;
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
        us = usn;
    }
}

} // namespace mf2
} // namespace iq3g
