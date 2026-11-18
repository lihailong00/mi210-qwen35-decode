// iq3gemv-nc.cuh - N-column (NC = 2..6 tokens, speculative-decoding verification) variants of the iq3gemv kernels
// (MMVQ-NC A). Same SoA copies as the single-token kernels (iq3gemv.cuh, unchanged); the weights of a row are read
// and decoded once for all NC activation columns.
//
// Per column, the result is bit-identical to the single-token kernel run on that column alone:
//   - same q8_1 activation records (one LDS record set per column, built exactly like stage_y / pro_store);
//   - same lanes per row (L = 16), same item -> lane assignment (lane l: items l, l+L, ... in increasing order);
//   - per item and sub-block the same integer sum (the codebook words are looked up once and dotted with every column;
//     integer sums are exact, so the dp4a order does not matter) and the same float term (dw*dy[j]) * (float) isum
//     accumulated into the column's fp32 accumulator in the same order;
//   - the same group_sum<L> butterfly and the same epilogues (v + residual, vb*silu(v)).
// Only the row -> wave ownership differs (allowed: it does not change any per-row arithmetic).
//
// LDS: the codebook tables at address 0 (as in iq3gemv.cuh) + the NC activation record sets (column-interleaved). When all
// of K does not fit (ffn_down K = 17408, mixed IQ3_S/IQ3_XXS tables with NC >= 3, NC = 6) the kernel runs in K phases
// (PH): every wave owns up to PH_MAXU units, keeps their accumulators in registers, and the records of the chunk range
// [h0, h0 + hph) are staged per phase. A lane still visits its items in increasing order, so the arithmetic is the same.
#pragma once
#include "iq3gemv.cuh"

namespace iq3g {
namespace nc {

// the 8 sign-expanded codebook words of sub-block j of a decoded item (the table lookups of T::sub)
template <class T, int P, int TOFF>
static __device__ __forceinline__ void look(const typename T::template dec<P> & r, const int j, int (&t)[8]) {
    if constexpr (T::TAB_WORDS == 8192) {   // IQ3_S
        const int wb = P == 4 ? 3*(j >> 1) : 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int bit = 5*(8*(j & 1) + k), w = wb + (bit >> 5), o = bit & 31;
            const uint32_t f = o <= 27 ? (r.W[w] >> o) : __builtin_amdgcn_alignbit(r.W[w + 1], r.W[w], o);
            const uint32_t addr = ((f & 31u) << 10) | bx4(r.q[2*j + (k >> 2)], k & 3);
            t[k] = tab_ld(addr + TOFF);
        }
    } else {                                // IQ3_XXS
        const uint32_t W = r.a[j];
        const uint32_t x = W & 0x7F7F7F7Fu;
        uint32_t p = x ^ (x >> 4);
        p ^= p >> 2;
        p ^= p >> 1;
        const uint32_t Wf = x | ((p & 0x01010101u) << 7);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint32_t addr = (((Wf >> (4*k)) & 15u) << 10) | bx4(r.q[2*j + (k >> 2)], k & 3);
            t[k] = tab_ld(addr + TOFF);
        }
    }
}
// scaled integer sum of sub-block j from the raw dp4a sum s (the tail of T::sub)
template <class T, int P>
static __device__ __forceinline__ int fin(const typename T::template dec<P> & r, const int j, const int s) {
    if constexpr (T::TAB_WORDS == 8192) {
        const int sc = (int) ((r.sc >> (4*j)) & 0xFu);
        return __mul24(s, 1 + 2*sc);
    } else {
        const uint32_t W = r.a[j];
        const int ls = dp4a((int) ((W >> 7) & 0x01010101u), 0x08040201, 0);
        return sdiv<1>(__mul24(ls, s) + sdiv<1>(s));
    }
}

static __device__ __forceinline__ int4 lds_i4(const uint32_t a) {
#if defined(__HIP_DEVICE_COMPILE__)
    return *(__attribute__((address_space(3))) const int4 *) (size_t) a;
#else
    (void) a; return make_int4(0, 0, 0, 0);
#endif
}
template <int P> struct dyv { float v[P]; };
template <int P>
static __device__ __forceinline__ dyv<P> lds_dy(const uint32_t a) {
    dyv<P> r;
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr (P == 4) {
        const float4 v = *(__attribute__((address_space(3))) const float4 *) (size_t) a;
        r.v[0] = v.x; r.v[1] = v.y; r.v[2] = v.z; r.v[3] = v.w;
    } else {
        const float2 v = *(__attribute__((address_space(3))) const float2 *) (size_t) a;
        r.v[0] = v.x; r.v[1] = v.y;
    }
#else
    (void) a;
    for (int j = 0; j < P; ++j) r.v[j] = 0.0f;
#endif
    return r;
}

// The accumulation step acc += (dw*dy)*isum, with the rounding the single-token kernels are compiled to (gfx90a, clang
// fp-contract=fast-honor-pragmas, checked in their ISA): the single / residual (gemv<T, none_t>) and multi-job (gemv_mj)
// kernels are not contracted (the SLP vectorizer pairs the two products of an item: v_pk_mul_f32, then two v_add_f32),
// the gate/up kernels (gemv<TA, TB>) are (acc and accb paired: v_pk_fma_f32). The N-column kernels pin both forms
// explicitly, independent of how their own code is vectorized; GGML_MMVQ_NC_VERIFY (in the model and in test-backend-ops
// GEMM_FUSION) checks the equality.
static __device__ __forceinline__ float term_one(const float acc, const float dwdy, const float s) {
#pragma clang fp contract(off)
    const float t = dwdy*s;
    return acc + t;
}
static __device__ __forceinline__ float term_glu(const float acc, const float dwdy, const float s) {
    return __builtin_fmaf(dwdy, s, acc);
}

// Activation records in LDS, column-interleaved: chunk h (128 activations) of column c at h*CSB<NC> + c*YREC_W*4 (relative
// to the phase base), each record exactly as stage_y / pro_store build it ({qs[32 words], d[4] floats}). Chunk stride
// CSB = NC*144 (+16 for even NC): CSB/16 odd keeps the ds_read_b128 of the 16 lanes of a row group (items lane, lane+16,
// .. -> chunks lane/2) conflict-free, as the 144-byte stride of the single-token kernels.
template <int NC> struct lay {
    static constexpr int RECB = YREC_W*4;
    static constexpr int CSB  = NC*RECB + (NC % 2 == 0 ? 16 : 0);
};

// one item (P sub-blocks) of one row for NC columns; TB != none_t: second weight matrix (gate/up) into accb.
// ya: LDS byte address of the item's qs words in column 0, yd: of its d values (column c: + c*RECB)
template <class TA, class TB, int P, int TOFF_A, int TOFF_B, int NC>
static __device__ __forceinline__ void item(float (&acc)[NC], float (&accb)[NC], const typename TA::template ld<P> & ca,
        const typename TB::template ld<P> & cb, const uint32_t ya, const uint32_t yd) {
    constexpr bool GLU = !std::is_same<TB, none_t>::value;
    constexpr uint32_t RECB = lay<NC>::RECB;
    const typename TA::template dec<P> da = TA::template prep<P>(ca);
    const float dwa = TA::template dw<P>(da);
    typename TB::template dec<P> db;
    float dwb = 0.0f;
    if constexpr (GLU) { db = TB::template prep<P>(cb); dwb = TB::template dw<P>(db); }
    dyv<P> dy[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) dy[c] = lds_dy<P>(yd + c*RECB);
#pragma unroll
    for (int j = 0; j < P; ++j) {
        int ta[8], tb[8];
        look<TA, P, TOFF_A>(da, j, ta);
        if constexpr (GLU) look<TB, P, TOFF_B>(db, j, tb);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int4 y0 = lds_i4(ya + c*RECB + 32*j), y1 = lds_i4(ya + c*RECB + 32*j + 16);
            const int y[8] = { y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w };
            int s = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) s = dp4a(ta[k], y[k], s);
            if constexpr (GLU) {
                int sb = 0;
#pragma unroll
                for (int k = 0; k < 8; ++k) sb = dp4a(tb[k], y[k], sb);
                acc[c]  = term_glu(acc[c],  dwa*dy[c].v[j], (float) fin<TA, P>(da, j, s));
                accb[c] = term_glu(accb[c], dwb*dy[c].v[j], (float) fin<TB, P>(db, j, sb));
            } else {
                acc[c] = term_one(acc[c], dwa*dy[c].v[j], (float) fin<TA, P>(da, j, s));
            }
        }
    }
}

// chunk / block indices of item it of local unit uu (identical to iq3g::gemv load_item with R = 1 and iq3g::mj_load)
template <int P, int RG, int SM>
static __device__ __forceinline__ void cidx(const int uu, const int it, const int items, const int U, const int grp, uint32_t & c,
        uint32_t & cb, int & p) {
    constexpr int IPC = 4/P;
    const int h = it / IPC;
    p = it % IPC;
    if constexpr (SM == 0) {
        c  = ((uint32_t) uu*(uint32_t) items + (uint32_t) h)*RG + (uint32_t) grp;
        cb = ((uint32_t) uu*(uint32_t) (items/2) + (uint32_t) (h >> 1))*RG + (uint32_t) grp;
    } else {
        const uint32_t hb = (uint32_t) h >> 1;
        c  = ((((uint32_t) h / SM)*(uint32_t) U + (uint32_t) uu)*SM + (uint32_t) h % SM)*RG + (uint32_t) grp;
        cb = (((hb / (SM/2))*(uint32_t) U + (uint32_t) uu)*(SM/2) + hb % (SM/2))*RG + (uint32_t) grp;
    }
}

// items it_beg, it_beg + L, .. < it_end of local unit uu (it_beg = this lane's first item of the range); records of the
// staged chunk range starting at chunk h0 at yoff (layout lay<NC>). Exposed for the mixed-pair kernel (family D).
template <class TA, class TB, int P, int L, int RG, int SM, int TOFF_A, int TOFF_B, int NC>
static __device__ __forceinline__ void unit(float (&acc)[NC], float (&accb)[NC], const uint8_t * __restrict__ wa,
        const uint8_t * __restrict__ wb, const uint32_t nh, const int uu, const int U, const int items, const int grp,
        const int it_beg, const int it_end, const int h0, const uint32_t yoff) {
    constexpr bool GLU = !std::is_same<TB, none_t>::value;
    constexpr int IPC = 4/P;
    for (int it = it_beg; it < it_end; it += L) {
        uint32_t c, cb; int p;
        cidx<P, RG, SM>(uu, it, items, U, grp, c, cb, p);
        const typename TA::template ld<P> la = TA::template load<P>(wa, nh, c, cb, p);
        typename TB::template ld<P> lb;
        if constexpr (GLU) lb = TB::template load<P>(wb, nh, c, cb, p);
        const int h = it / IPC;
        const uint32_t base = yoff + (uint32_t) (h - h0)*lay<NC>::CSB;
        item<TA, TB, P, TOFF_A, TOFF_B, NC>(acc, accb, la, lb, base + (uint32_t) p*(32*P), base + 128 + (uint32_t) p*(4*P));
    }
}

// activation records of chunks [h0, h1) for NC columns (column c: q8_1 blocks at yg + c*ycb)
template <int NC, int NT>
static __device__ __forceinline__ void stage(uint32_t * smem, const uint32_t yoff, const block_q8_1 * __restrict__ yg,
        const int ycb, const int h0, const int h1, const int tid) {
    const int nb = (h1 - h0)*4;
    for (int idx = tid; idx < NC*nb; idx += NT) {
        const int c = idx / nb, s = idx - c*nb;
        const uint8_t * b = (const uint8_t *) (yg + (size_t) c*ycb + 4*h0 + s);
        const uint4 q0 = *(const uint4 *) (b + 4), q1 = *(const uint4 *) (b + 20);
        const uint32_t ds = *(const uint32_t *) b;
        int * rec = (int *) (smem + (yoff + (uint32_t) (s >> 2)*lay<NC>::CSB + (uint32_t) c*lay<NC>::RECB)/4);
        int * r = rec + 8*(s & 3);
        *(int4 *) r       = make_int4((int) q0.x, (int) q0.y, (int) q0.z, (int) q0.w);
        *(int4 *) (r + 4) = make_int4((int) q1.x, (int) q1.y, (int) q1.z, (int) q1.w);
        ((float *) rec)[32 + (s & 3)] = h2f(ds & 0xFFFFu);
    }
}
template <class TA, class TB, bool TWO, int NT>
static __device__ __forceinline__ void stage_tables(uint32_t * smem, const int toff_b, const int tid) {
    const uint32_t ga = grid_word<TA>(tid % (TA::TAB_WORDS/16));
    uint32_t gb = 0;
    if constexpr (TWO) gb = grid_word<TB>(tid % (TB::TAB_WORDS/16));
    tab_store<TA, NT>(smem, ga, tid);
    if constexpr (TWO) tab_store<TB, NT>(smem + toff_b/4, gb, tid);
}
// first item >= it0 of the lane's sequence lane, lane + L, ...
template <int L>
static __device__ __forceinline__ int first_item(const int it0, const int lane) {
    return it0 + (((lane - it0) % L) + L) % L;
}

// phased mode: up to PH_MAXU units per wave (accumulators of all of them in registers across the K phases)
static constexpr int PH_MAXU = 3;

// ------------------------------------------------------------------------------------------------ single / GLU / residual
// dst column c at dst + c*dcs, residual column c at residual + c*rcs, activation column c at yg + c*ycb (q8_1 blocks).
// PH = false: all of K staged once, persistent unit loop. PH = true: K phases of hph chunks, units u0 + k*ustep
// (k < PH_MAXU; host: grid*NW*PH_MAXU >= units) with their accumulators kept across the phases.
template <class TA, class TB, int P, int L, int NW, int NC, int SM, bool PH>
__launch_bounds__(NW*64, 1)
__global__ void gemv(const uint8_t * __restrict__ wa, const uint8_t * __restrict__ wb, const block_q8_1 * __restrict__ yg, const int ycb,
        float * __restrict__ dst, const int dcs, const int K, const int N, const float * __restrict__ residual, const int rcs, const int hph) {
    constexpr bool GLU  = !std::is_same<TB, none_t>::value;
    constexpr bool SAME = std::is_same<TA, TB>::value;
    constexpr bool TWO  = GLU && !SAME;
    constexpr int TOFF_A = 0;
    constexpr int TOFF_B = TWO ? TA::TAB_WORDS*4 : 0;
    constexpr int YOFF   = TA::TAB_WORDS*4 + (TWO ? TB::TAB_WORDS*4 : 0);
    constexpr int RG  = 64/L;
    constexpr int IPC = 4/P;
    constexpr int NT  = NW*64;
    extern __shared__ uint32_t smem[];
#if defined(__HIP_DEVICE_COMPILE__)
    if ((uint32_t) (size_t) (__attribute__((address_space(3))) uint32_t *) smem != 0) __builtin_trap();
    const int w = __builtin_amdgcn_readfirstlane(threadIdx.y);
#else
    const int w = threadIdx.y;
#endif
    const int tid  = threadIdx.y*64 + threadIdx.x;
    const int lane = threadIdx.x % L, grp = threadIdx.x / L;
    const int items = K / 128;
    const int ipr   = items*IPC;
    const uint32_t nh = (uint32_t) N*(uint32_t) items;
    const int U = N / RG;
    const int ustep = gridDim.x*NW;
    const int u0 = w*gridDim.x + blockIdx.x;

    stage_tables<TA, TB, TWO, NT>(smem, TOFF_B, tid);

    auto epilogue = [&](const int uu, const float (&acc)[NC], const float (&accb)[NC]) {
        const int row = uu*RG + grp;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float v = group_sum<L>(acc[c]);
            if constexpr (GLU) {
                const float vb = group_sum<L>(accb[c]);
                if (lane == 0) dst[(size_t) c*dcs + row] = vb * silu(v);
            } else {
                if (lane == 0) dst[(size_t) c*dcs + row] = residual ? v + residual[(size_t) c*rcs + row] : v;
            }
        }
    };
    if constexpr (!PH) {
        stage<NC, NT>(smem, YOFF, yg, ycb, 0, items, tid);
        __syncthreads();
        for (int u = u0; u < U; u += ustep) {
            float acc[NC], accb[NC];
#pragma unroll
            for (int c = 0; c < NC; ++c) { acc[c] = 0.0f; accb[c] = 0.0f; }
            unit<TA, TB, P, L, RG, SM, TOFF_A, TOFF_B, NC>(acc, accb, wa, wb, nh, u, U, items, grp, lane, ipr, 0, YOFF);
            epilogue(u, acc, accb);
        }
    } else {
        float acc[PH_MAXU][NC], accb[PH_MAXU][NC];
#pragma unroll
        for (int k = 0; k < PH_MAXU; ++k) {
#pragma unroll
            for (int c = 0; c < NC; ++c) { acc[k][c] = 0.0f; accb[k][c] = 0.0f; }
        }
        for (int h0 = 0; h0 < items; h0 += hph) {
            const int h1 = min(items, h0 + hph);
            if (h0 > 0) __syncthreads();
            stage<NC, NT>(smem, YOFF, yg, ycb, h0, h1, tid);
            __syncthreads();
            const int itb = first_item<L>(h0*IPC, lane);
#pragma unroll
            for (int k = 0; k < PH_MAXU; ++k) {
                const int u = u0 + k*ustep;
                if (u < U) {
                    unit<TA, TB, P, L, RG, SM, TOFF_A, TOFF_B, NC>(acc[k], accb[k], wa, wb, nh, u, U, items, grp, itb, h1*IPC, h0, YOFF);
                }
            }
        }
#pragma unroll
        for (int k = 0; k < PH_MAXU; ++k) {
            const int u = u0 + k*ustep;
            if (u < U) epilogue(u, acc[k], accb[k]);
        }
    }
}

// ------------------------------------------------------------------------------------------------ multi-job
// N-column version of iq3g::gemv_mj (KS = 1): 2..3 GEMVs on the same activation, units concatenated, each job with
// its own weights (type T0 or T1), rows and output (column c of job j at dst + c*dcs). No residual (as at N = 1).
struct mjnc_job {
    const uint8_t * w;
    float         * dst;
    uint32_t        nh;
    int             ubeg;   // unused jobs: INT_MAX
    int             tsel;
    int             U;
    int             dcs;
};
struct mjnc_args {
    mjnc_job job[MJ_MAX];
    int      njob;
    int      utot;
};
template <class T0, class T1, int P, int L, int NW, int NC, int SM, bool PH>
__launch_bounds__(NW*64, 1)
__global__ void gemv_mj(const block_q8_1 * __restrict__ yg, const int ycb, const int K, const mjnc_args a, const int hph) {
    constexpr bool HAS1 = !std::is_same<T1, none_t>::value;
    constexpr bool TWO  = HAS1 && !std::is_same<T0, T1>::value;
    constexpr int TOFF0 = 0;
    constexpr int TOFF1 = TWO ? T0::TAB_WORDS*4 : 0;
    constexpr int YOFF  = T0::TAB_WORDS*4 + (TWO ? T1::TAB_WORDS*4 : 0);
    constexpr int RG  = 64/L;
    constexpr int IPC = 4/P;
    constexpr int NT  = NW*64;
    extern __shared__ uint32_t smem[];
#if defined(__HIP_DEVICE_COMPILE__)
    if ((uint32_t) (size_t) (__attribute__((address_space(3))) uint32_t *) smem != 0) __builtin_trap();
    const int w = __builtin_amdgcn_readfirstlane(threadIdx.y);
#else
    const int w = threadIdx.y;
#endif
    const int tid  = threadIdx.y*64 + threadIdx.x;
    const int lane = threadIdx.x % L, grp = threadIdx.x / L;
    const int items = K / 128;
    const int ipr   = items*IPC;
    const int ustep = gridDim.x*NW;
    const int u0 = w*gridDim.x + blockIdx.x;

    stage_tables<T0, T1, TWO, NT>(smem, TOFF1, tid);

    float dummy[NC];
    auto run = [&](float (&acc)[NC], const mjnc_job & J, const int lu, const int it_beg, const int it_end, const int h0) {
        if (!HAS1 || J.tsel == 0) {
            unit<T0, none_t, P, L, RG, SM, TOFF0, 0, NC>(acc, dummy, J.w, nullptr, J.nh, lu, J.U, items, grp, it_beg, it_end, h0, YOFF);
        } else {
            if constexpr (HAS1) unit<T1, none_t, P, L, RG, SM, TOFF1, 0, NC>(acc, dummy, J.w, nullptr, J.nh, lu, J.U, items, grp, it_beg, it_end, h0, YOFF);
        }
    };
    auto epilogue = [&](const mjnc_job & J, const int lu, const float (&acc)[NC]) {
        const int row = lu*RG + grp;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float v = group_sum<L>(acc[c]);
            if (lane == 0) J.dst[(size_t) c*J.dcs + row] = v;
        }
    };
    auto find = [&](const int u) -> int { return (u >= a.job[1].ubeg) + (u >= a.job[2].ubeg); };
    if constexpr (!PH) {
        stage<NC, NT>(smem, YOFF, yg, ycb, 0, items, tid);
        __syncthreads();
        for (int u = u0; u < a.utot; u += ustep) {
            const mjnc_job & J = a.job[find(u)];
            const int lu = u - J.ubeg;
            float acc[NC];
#pragma unroll
            for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
            run(acc, J, lu, lane, ipr, 0);
            epilogue(J, lu, acc);
        }
    } else {
        float acc[PH_MAXU][NC];
#pragma unroll
        for (int k = 0; k < PH_MAXU; ++k) {
#pragma unroll
            for (int c = 0; c < NC; ++c) acc[k][c] = 0.0f;
        }
        for (int h0 = 0; h0 < items; h0 += hph) {
            const int h1 = min(items, h0 + hph);
            if (h0 > 0) __syncthreads();
            stage<NC, NT>(smem, YOFF, yg, ycb, h0, h1, tid);
            __syncthreads();
            const int itb = first_item<L>(h0*IPC, lane);
#pragma unroll
            for (int k = 0; k < PH_MAXU; ++k) {
                const int u = u0 + k*ustep;
                if (u < a.utot) {
                    const mjnc_job & J = a.job[find(u)];
                    run(acc[k], J, u - J.ubeg, itb, h1*IPC, h0);
                }
            }
        }
#pragma unroll
        for (int k = 0; k < PH_MAXU; ++k) {
            const int u = u0 + k*ustep;
            if (u < a.utot) {
                const mjnc_job & J = a.job[find(u)];
                epilogue(J, u - J.ubeg, acc[k]);
            }
        }
    }
}
template <class T0, class T1>
static inline size_t tab_bytes() {
    constexpr bool TWO = !std::is_same<T1, none_t>::value && !std::is_same<T0, T1>::value;
    return (size_t) T0::TAB_WORDS*4 + (TWO ? (size_t) T1::TAB_WORDS*4 : 0);
}
// LDS bytes of the activation records of nch chunks
template <int NC> static inline size_t rec_bytes(const int nch) { return (size_t) nch*lay<NC>::CSB; }

} // namespace nc
} // namespace iq3g
