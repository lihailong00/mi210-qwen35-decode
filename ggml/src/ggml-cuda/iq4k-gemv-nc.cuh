// iq4k-gemv-nc.cuh - N-column (ne11 = NC = 2..6) variants of the iq4k IQ4_XS / Q4_K decode GEMVs (MMVQ-NC, family B).
//
// Contract: column j of an NC-column launch is bit-identical to the single-token (NC = 1) launch of the same tensor on
// column j alone. The single-token kernels (iq4k::gemv, iq4k::gemv_sn in iq4k-gemv.cuh) are not modified; these kernels
// reproduce their per-row fp32 accumulation tree exactly:
//   * gemv_nc reads the same SoA copy (same cache kind, same launch geometry: L lanes per row, grid x NW waves, round-robin
//     units, row-interleaved or round-tiled layout) and per lane accumulates item l, l + L, ... in the same order with
//     the same per-item expression as iq4xs_t::compute / q4k_t::compute (the weight decode is shared by the columns,
//     the integer dot products, the scale math and the fp32 terms are per column and identical); then group_sum<L> and
//     the same epilogue (GLU: up * silu(gate); residual: + res[row]).
//   * The activation of all NC columns is staged in LDS. When NC columns of K do not fit (ffn_down, K = 17408) the K
//     range is staged in phases: every wave keeps the per-lane partial sums of its (at most MU) units in registers
//     across phases, so the sequence of fp32 adds per lane is unchanged.
//   * gemv_sn_nc (rows < 2048: attn_k / attn_v) = gemv_sn / stock mul_mat_vec_q ncols_dst = 1 structure per column.
#pragma once
#include "iq4k-gemv.cuh"

namespace iq4k {

static constexpr int NC_MAX_INST = 6;     // instantiated column counts: 2..NC_MAX_INST

// ---------------------------------------------------------------- per-item compute for NC columns
// y records of column j at yr + j*cst (words); same record layout as the single-column kernel. accum adds the item term
// of every column to acc[j] with the fp32 operation sequence of the compiled single-token kernel (gfx90a): which a*b + c
// are contracted into an fma is fixed explicitly (contract off + __builtin_fmaf), so the column loop / SLP vectorizer
// cannot change it. IQ4_XS: acc = fma(h2f(d)*d8, (float) (s*(ls - 32)), acc). Q4_K: sumf_d = fma(db, B, da*A),
// sumf_m = fma(da, C, db*D), acc = acc + (dm.x*sumf_d - dm.y*sumf_m), A..D = float of the integer products; the up matrix
// of the SwiGLU kernel (ALT) is compiled differently there: sumf_d = fma(da, A, db*B), acc = acc + fma(sumf_d, dm.x,
// -(dm.y*sumf_m)). (Read off the gfx90a ISA of the single-token kernels; GGML_MMVQ_NC_VERIFY checks it.)
template <class T, int NC> struct ncc;

template <int NC> struct ncc<iq4xs_t, NC> {
    typedef iq4xs_t T;
    struct wdec { uint32_t lo[4], hi[4]; int ls; float dw; };
    static __device__ __forceinline__ wdec dec(const T::raw & r, const T::lc & c) {
        wdec w;
        lookup8(r.q.x, c.T, c.m4, c.sb, w.lo[0], w.hi[0]);
        lookup8(r.q.y, c.T, c.m4, c.sb, w.lo[1], w.hi[1]);
        lookup8(r.q.z, c.T, c.m4, c.sb, w.lo[2], w.hi[2]);
        lookup8(r.q.w, c.T, c.m4, c.sb, w.lo[3], w.hi[3]);
        w.ls = (int) (((r.h.y >> c.shl) & 0xFu) | (((r.h.x >> c.shh) & 3u) << 4));
        w.dw = h2f(r.h.x);
        return w;
    }
    // acc + the item term of one column (record y)
    template <bool ALT>
    static __device__ __forceinline__ float step(const wdec & w, const uint32_t * y, const float acc) {
#pragma clang fp contract(off)
        const uint4 y0 = *(const uint4 *) y;
        const uint4 y1 = *(const uint4 *) (y + 4);
        const float d8 = __uint_as_float(y[8]);
        int s;
        s = dp4a((int) w.lo[0], (int) y0.x, 0); s = dp4a((int) w.hi[0], (int) y1.x, s);
        s = dp4a((int) w.lo[1], (int) y0.y, s); s = dp4a((int) w.hi[1], (int) y1.y, s);
        s = dp4a((int) w.lo[2], (int) y0.z, s); s = dp4a((int) w.hi[2], (int) y1.z, s);
        s = dp4a((int) w.lo[3], (int) y0.w, s); s = dp4a((int) w.hi[3], (int) y1.w, s);
        const float d = w.dw * d8;
        return __builtin_fmaf(d, (float) __mul24(s, w.ls - 32), acc);
    }
    template <bool ALT = false>
    static __device__ __forceinline__ void accum(const T::raw & r, const uint32_t * yr, const int cst, const T::lc & c, float (&acc)[NC]) {
        const wdec w = dec(r, c);
#pragma unroll
        for (int j = 0; j < NC; ++j) acc[j] = step<ALT>(w, yr + j*cst, acc[j]);
    }
    // SwiGLU: up (ALT) and gate per column on the same y record
    static __device__ __forceinline__ void accum2(const T::raw & ru, const T::raw & rg, const uint32_t * yr, const int cst, const T::lc & c,
            float (&accu)[NC], float (&accg)[NC]) {
        const wdec wu = dec(ru, c), wg = dec(rg, c);
#pragma unroll
        for (int j = 0; j < NC; ++j) { accu[j] = step<true>(wu, yr + j*cst, accu[j]); accg[j] = step<false>(wg, yr + j*cst, accg[j]); }
    }
};

template <int NC> struct ncc<q4k_t, NC> {
    typedef q4k_t T;
    struct wdec { uint32_t q[8]; int sca, scb, ma, mb; float2 dm; };
    static __device__ __forceinline__ wdec dec(const T::raw & r, const T::lc & c) {
        wdec w;
        w.q[0] = r.a.x; w.q[1] = r.a.y; w.q[2] = r.a.z; w.q[3] = r.a.w; w.q[4] = r.b.x; w.q[5] = r.b.y; w.q[6] = r.b.z; w.q[7] = r.b.w;
        const uint32_t t1 = r.h.y >> c.sh, t2 = r.h.z >> c.sh, t3 = r.h.w >> c.sh;
        const uint32_t SC = c.hi ? ((t3 & 0x0F0Fu) | ((t1 >> 2) & 0x3030u)) : (t1 & 0x3F3Fu);
        const uint32_t MM = c.hi ? (((t3 >> 4) & 0x0F0Fu) | ((t2 >> 2) & 0x3030u)) : (t2 & 0x3F3Fu);
        w.sca = (int) (SC & 0xFFu); w.scb = (int) (SC >> 8);
        w.ma  = (int) (MM & 0xFFu); w.mb  = (int) (MM >> 8);
        w.dm = __half22float2(*(const half2 *) &r.h.x);
        return w;
    }
    template <bool ALT>
    static __device__ __forceinline__ float step(const wdec & w, const uint32_t * y, const float acc) {
#pragma clang fp contract(off)
        const uint4 ya0 = *(const uint4 *) y,       ya1 = *(const uint4 *) (y + 4);
        const uint4 yb0 = *(const uint4 *) (y + 8), yb1 = *(const uint4 *) (y + 12);
        const uint4 dd  = *(const uint4 *) (y + 16);
        const uint32_t ua[8] = {ya0.x, ya0.y, ya0.z, ya0.w, ya1.x, ya1.y, ya1.z, ya1.w};
        const uint32_t ub[8] = {yb0.x, yb0.y, yb0.z, yb0.w, yb1.x, yb1.y, yb1.z, yb1.w};
        int dota = 0, dotb = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            dota = dp4a((int) (w.q[k] & 0x0F0F0F0Fu), (int) ua[k], dota);
            dotb = dp4a((int) ((w.q[k] >> 4) & 0x0F0F0F0Fu), (int) ub[k], dotb);
        }
        const float da = __uint_as_float(dd.x), db = __uint_as_float(dd.y);
        const float A = (float) __mul24(dota, w.sca), B = (float) __mul24(dotb, w.scb);
        const float C = (float) __mul24((int) dd.z, w.ma), D = (float) __mul24((int) dd.w, w.mb);
        float v;
        if constexpr (ALT) {   // the up matrix of the SwiGLU kernel
            const float sumf_d = __builtin_fmaf(da, A, db*B);
            const float sumf_m = __builtin_fmaf(da, C, db*D);
            v = __builtin_fmaf(sumf_d, w.dm.x, -(w.dm.y*sumf_m));
        } else {
            const float sumf_d = __builtin_fmaf(db, B, da*A);
            const float sumf_m = __builtin_fmaf(da, C, db*D);
            v = w.dm.x*sumf_d - w.dm.y*sumf_m;
        }
        return acc + v;
    }
    template <bool ALT = false>
    static __device__ __forceinline__ void accum(const T::raw & r, const uint32_t * yr, const int cst, const T::lc & c, float (&acc)[NC]) {
        const wdec w = dec(r, c);
#pragma unroll
        for (int j = 0; j < NC; ++j) acc[j] = step<ALT>(w, yr + j*cst, acc[j]);
    }
    static __device__ __forceinline__ void accum2(const T::raw & ru, const T::raw & rg, const uint32_t * yr, const int cst, const T::lc & c,
            float (&accu)[NC], float (&accg)[NC]) {
        const wdec wu = dec(ru, c), wg = dec(rg, c);
#pragma unroll
        for (int j = 0; j < NC; ++j) { accu[j] = step<true>(wu, yr + j*cst, accu[j]); accg[j] = step<false>(wg, yr + j*cst, accg[j]); }
    }
};

// stage q8_1 blocks [b0, b1) of NC columns (column j at yg + j*ys) into LDS records; record of block b of column j at
// ys_ + j*cst + (b/YPI - b0/YPI)*YW (b0 % YPI == 0)
template <class T, int NC>
static __device__ __forceinline__ void stage_y_nc(uint32_t * __restrict__ ysm, const int cst, const block_q8_1 * __restrict__ yg, const int ys,
        const int b0, const int b1, const int tid, const int nt) {
    const int nb = b1 - b0;
    for (int i = tid; i < NC*nb; i += nt) {
        const int j = i / nb, b = b0 + (i - j*nb);
        const uint8_t * p = (const uint8_t *) (yg + (size_t) j*ys + b);
        const uint32_t ds = ld_u32_opaque(p);
        const uint4 q0 = *(const uint4 *) (p + 4);
        const uint4 q1 = *(const uint4 *) (p + 20);
        T::put_y(ysm + j*cst, b - b0, q0, q1, h2f(ds));
    }
}

// ---------------------------------------------------------------- the NC-column GEMV kernel (round-robin units, exact)
// Same geometry as gemv<T, L, 1, GLU, false, RT, true> (the single-token configuration of every model shape): wave
// w = wy*grid + blockIdx.x owns units w, w + nwaves, ... (unit = G = 64/L rows, lane group grp = row grp of the unit),
// q = ipr/L steps per unit, step k: lane reads item lane + L*k. The step range [0, q) is cut into phases of qph steps
// (nph = ceil(q/qph)); phase p stages the records of items [p*qph*L, ...) of all NC columns. MU = 0: single phase, the
// accumulators are reset at every unit end; MU > 0: up to MU units per wave, accumulators kept across the phases.
// Loads are ping-pong pipelined inside a phase (clamped at the last step, no conditional memory operations); the first
// load of a phase is issued before its y staging.
// Q4_K SwiGLU (2 x 12 registers of weights per buffer) runs with 12 waves (pick_cfg_glu): 768 threads leave it ~168 VGPRs
template <class T, bool GLU> struct nc_lb { static constexpr int v = GLU && std::is_same<T, q4k_t>::value ? 768 : 1024; };
template <class T, int L, bool GLU, int NC, int MU, bool RT>
__launch_bounds__((nc_lb<T, GLU>::v), 1)
__global__ void gemv_nc(const soa s, const soa sg, const block_q8_1 * __restrict__ yg, const int ys,
        float * __restrict__ dst, const int dstr, const float * __restrict__ residual, const int rstr,
        const int K, const int N, const int qph) {
    extern __shared__ __align__(16) uint32_t ysm[];
    constexpr int G = 64/L, GR = G, MUX = MU > 0 ? MU : 1;
    const int tid  = threadIdx.y*64 + threadIdx.x;
    const int nt   = blockDim.y*64;
    const int lane = threadIdx.x % L;
    const int grp  = threadIdx.x / L;
    const uint32_t bpr = K / QK_K;
    const int ipr    = bpr*T::IPB;
    const int q      = ipr / L;
    const int nunits = N / GR;
    const int wy     = __builtin_amdgcn_readfirstlane(threadIdx.y);
    const int nwaves = gridDim.x*blockDim.y;
    const int w      = wy*gridDim.x + blockIdx.x;
    const int myu    = w < nunits ? (int) ((uint32_t) (nunits - 1 - w)/(uint32_t) nwaves) + 1 : 0;
    const int nph    = (q + qph - 1)/qph;
    const int cst    = qph*L*T::YW;                 // LDS words per column
    const typename T::lc c = T::lane_consts(lane);
    const size_t nbk = (size_t) N*bpr;
    const uint32_t qlane = lane*GR + grp, slane = (lane/T::IPB)*GR + grp;
    const uint32_t boff = RT ? (uint32_t) (nbk*4) : 4*bpr*GR;

    typedef typename T::raw raw;
    raw A, B, Ag, Bg;
    float acc[MUX][NC], accg[GLU ? MUX : 1][NC];
#pragma unroll
    for (int m = 0; m < MUX; ++m)
#pragma unroll
    for (int j = 0; j < NC; ++j) { acc[m][j] = 0.0f; if constexpr (GLU) accg[m][j] = 0.0f; }

    for (int ph = 0; ph < nph; ++ph) {
        const int k0 = ph*qph, k1 = min(q, k0 + qph), qp = k1 - k0;
        const int nst = myu*qp;
        // load cursor: unit lu, step lk (clamped at the last step of the phase)
        int ls = 0, lu = 0, lk = k0;
        auto load_step = [&](raw & bb, raw & bg) {
            const bool past = ls >= nst;
            const int kr = past ? myu - 1 : lu, kk = past ? k1 - 1 : lk;
            uint32_t qs_, ss_;
            if constexpr (RT) {
                const int nur = min(nwaves, nunits - kr*nwaves);
                const uint32_t ch = (uint32_t) (kr*nwaves*q + kk*nur + w);
                qs_ = ch*(L*GR); ss_ = ch*((L/T::IPB)*GR);
            } else {
                const uint32_t un = (uint32_t) (w + kr*nwaves);
                qs_ = (un*bpr*8 + kk*L)*GR; ss_ = (un*bpr + kk*(L/T::IPB))*GR;
            }
            bb = T::load_c(s, qs_ + qlane, ss_ + slane, boff);
            if constexpr (GLU) bg = T::load_c(sg, qs_ + qlane, ss_ + slane, boff);
            ++ls; if (++lk == k1) { lk = k0; ++lu; }
        };
        if (nst > 0) load_step(A, Ag);
        if (ph > 0) __syncthreads();                // all waves done with the previous phase's records
        stage_y_nc<T, NC>(ysm, cst, yg, ys, k0*L*T::YPI, k1*L*T::YPI, tid, nt);
        __syncthreads();

        int cs = 0, cu = 0, ck = k0;
        auto compute_step = [&](const raw & bb, const raw & bg) {
            const uint32_t * yr = ysm + lane*T::YW + (ck - k0)*(L*T::YW);
            if constexpr (MU == 0) {
                if constexpr (GLU) ncc<T, NC>::accum2(bb, bg, yr, cst, c, acc[0], accg[0]);
                else ncc<T, NC>::template accum<false>(bb, yr, cst, c, acc[0]);
            } else {                                // accumulators of unit cu: exact moves in and out
                float a[NC], ag[NC];
#pragma unroll
                for (int j = 0; j < NC; ++j) { a[j] = acc[0][j]; if constexpr (GLU) ag[j] = accg[0][j]; }
#pragma unroll
                for (int m = 1; m < MUX; ++m) {
                    if (m == cu) {
#pragma unroll
                        for (int j = 0; j < NC; ++j) { a[j] = acc[m][j]; if constexpr (GLU) ag[j] = accg[m][j]; }
                    }
                }
                if constexpr (GLU) ncc<T, NC>::accum2(bb, bg, yr, cst, c, a, ag);
                else ncc<T, NC>::template accum<false>(bb, yr, cst, c, a);
#pragma unroll
                for (int m = 0; m < MUX; ++m) {
                    if (m == cu) {
#pragma unroll
                        for (int j = 0; j < NC; ++j) { acc[m][j] = a[j]; if constexpr (GLU) accg[m][j] = ag[j]; }
                    }
                }
            }
            if (ck == q - 1 && cs < nst) {          // end of a unit (last phase): reduce and store
                const int row = (w + cu*nwaves)*GR + grp;
#pragma unroll
                for (int m = 0; m < MUX; ++m) {
                    if (MU == 0 || m == cu) {
#pragma unroll
                        for (int j = 0; j < NC; ++j) {
                            float o = group_sum<L>(acc[m][j]);
                            if constexpr (GLU) { const float g = group_sum<L>(accg[m][j]); o *= silu_f(g); }
                            if (residual) o += residual[(size_t) j*rstr + row];
                            if (lane == 0) dst[(size_t) j*dstr + row] = o;
                        }
                    }
                }
                if constexpr (MU == 0) {
#pragma unroll
                    for (int j = 0; j < NC; ++j) { acc[0][j] = 0.0f; if constexpr (GLU) accg[0][j] = 0.0f; }
                }
            }
            ++cs; if (++ck == k1) { ck = k0; ++cu; }
        };
        for (int st = 0; st < nst; st += 2) {
            load_step(B, Bg);       // step st+1 (clamped)
            compute_step(A, Ag);    // step st
            load_step(A, Ag);       // step st+2 (clamped)
            compute_step(B, Bg);    // step st+1 (past the end: unit index myu, never stored)
        }
    }
}

// ---------------------------------------------------------------- host side
struct ncplan { int qph, mu; size_t lds; };
static constexpr size_t NC_LDS_MAX = 65536;

// phases for NC columns under the LDS limit; mu = units per wave the kernel must hold (0: single phase)
static inline bool nc_plan(ggml_type t, const cfg & c, int K, int N, int NC, ncplan & p) {
    const int ipb = t == GGML_TYPE_Q4_K ? 4 : 8, yw = t == GGML_TYPE_Q4_K ? 20 : 12;
    const int ipr = (K/QK_K)*ipb, q = ipr/c.L;
    const int GR = 64/c.L, nunits = N/GR, nwaves = c.grid*c.NW;
    for (int nph = 1; nph <= q; ++nph) {
        const int qph = (q + nph - 1)/nph;
        const size_t lds = (size_t) NC*qph*c.L*yw*4;
        if (lds > NC_LDS_MAX) continue;
        p.qph = qph; p.lds = lds;
        const int maxu = (nunits + nwaves - 1)/nwaves;
        p.mu = (q + qph - 1)/qph == 1 ? 0 : maxu;
        return true;
    }
    return false;
}

// is the NC kernel instantiated for this single-token configuration (exact round-robin, R = 1, U = 1)
static inline bool nc_cfg_ok(const cfg & c, ggml_type t, int K, int N) {
    const int ipb = t == GGML_TYPE_Q4_K ? 4 : 8;
    return shape_ok(c, t, K, N) && !c.sk && c.R == 1 && c.U == 1 && (c.L == 16 || c.L == 32) &&
        ((K/QK_K)*ipb) % c.L == 0;
}
static inline bool nc_mu_ok(int mu) { return mu == 0 || mu == 1 || mu == 2 || mu == 3; }

template <class T, int L, bool GLU, int NC, int MU>
static inline void launch_nc_k(const cfg & c, const ncplan & p, const soa & s, const soa & sg, const block_q8_1 * y, int ys,
        float * dst, int dstr, const float * res, int rstr, int K, int N, hipStream_t st) {
    if (c.rt) hipLaunchKernelGGL((gemv_nc<T, L, GLU, NC, MU, true>),  dim3(c.grid), dim3(64, c.NW), p.lds, st, s, sg, y, ys, dst, dstr, res, rstr, K, N, p.qph);
    else      hipLaunchKernelGGL((gemv_nc<T, L, GLU, NC, MU, false>), dim3(c.grid), dim3(64, c.NW), p.lds, st, s, sg, y, ys, dst, dstr, res, rstr, K, N, p.qph);
}
template <class T, int L, bool GLU, int NC>
static inline bool launch_nc_mu(const cfg & c, const ncplan & p, const soa & s, const soa & sg, const block_q8_1 * y, int ys,
        float * dst, int dstr, const float * res, int rstr, int K, int N, hipStream_t st) {
    if (c.NW*64 > nc_lb<T, GLU>::v) return false;
    if constexpr (GLU) {   // gate/up: K <= 8192, single phase
        if (p.mu != 0) return false;
        launch_nc_k<T, L, GLU, NC, 0>(c, p, s, sg, y, ys, dst, dstr, res, rstr, K, N, st); return true;
    } else {
        switch (p.mu) {
        case 0: launch_nc_k<T, L, GLU, NC, 0>(c, p, s, sg, y, ys, dst, dstr, res, rstr, K, N, st); return true;
        case 1: launch_nc_k<T, L, GLU, NC, 1>(c, p, s, sg, y, ys, dst, dstr, res, rstr, K, N, st); return true;
        case 2: launch_nc_k<T, L, GLU, NC, 2>(c, p, s, sg, y, ys, dst, dstr, res, rstr, K, N, st); return true;
        case 3: launch_nc_k<T, L, GLU, NC, 3>(c, p, s, sg, y, ys, dst, dstr, res, rstr, K, N, st); return true;
        default: return false;
        }
    }
}
template <class T, bool GLU, int NC>
static inline bool launch_nc_l(const cfg & c, const ncplan & p, const soa & s, const soa & sg, const block_q8_1 * y, int ys,
        float * dst, int dstr, const float * res, int rstr, int K, int N, hipStream_t st) {
    if (c.L == 16) return launch_nc_mu<T, 16, GLU, NC>(c, p, s, sg, y, ys, dst, dstr, res, rstr, K, N, st);
    if (c.L == 32) return launch_nc_mu<T, 32, GLU, NC>(c, p, s, sg, y, ys, dst, dstr, res, rstr, K, N, st);
    return false;
}
// GLU: s = up, sg = gate. GLU with mu > 0 is not instantiated (gate/up have K <= 8192: single phase).
template <class T, bool GLU>
static inline bool launch_nc(const cfg & c, const ncplan & p, int NC, const soa & s, const soa & sg, const block_q8_1 * y, int ys,
        float * dst, int dstr, const float * res, int rstr, int K, int N, hipStream_t st) {
    if (GLU && p.mu != 0) return false;
    switch (NC) {
        case 2: return launch_nc_l<T, GLU, 2>(c, p, s, sg, y, ys, dst, dstr, res, rstr, K, N, st);
        case 3: return launch_nc_l<T, GLU, 3>(c, p, s, sg, y, ys, dst, dstr, res, rstr, K, N, st);
        case 4: return launch_nc_l<T, GLU, 4>(c, p, s, sg, y, ys, dst, dstr, res, rstr, K, N, st);
        case 5: return launch_nc_l<T, GLU, 5>(c, p, s, sg, y, ys, dst, dstr, res, rstr, K, N, st);
        case 6: return launch_nc_l<T, GLU, 6>(c, p, s, sg, y, ys, dst, dstr, res, rstr, K, N, st);
        default: return false;
    }
}

// ---------------------------------------------------------------- small-N kernel (rows < 2048), NC columns
// Per column exactly gemv_sn / mul_mat_vec_q<type, 1> on GCN: 2 waves per row, item kbx = tid/(qi/vdr) + k*bpi,
// the integer math of vec_dot_*_q8_1, wave 1 -> shared memory, wave 0 adds and warp_reduce_sum<64>. The fp32 sequence of
// the compiled stock kernel is fixed explicitly: IQ4_XS tmp = fma(h2f(d)*d8, (float) sumi, tmp); Q4_K
// sumf = fma(d8[1], float(dot_1*s_1), fma(d8[0], float(dot_0*s_0), 0)) for sumf_d / sumf_m, tmp = tmp + (dm.x*sumf_d -
// dm.y*sumf_m).
template <int NC>
static __device__ __forceinline__ void sn_accum_iq4_xs(const void * __restrict__ vbq, const block_q8_1 * __restrict__ y, const int ys,
        const int kbx, const int iqs, float (&tmp)[NC]) {
#pragma clang fp contract(off)
    const block_iq4_xs * bq4 = (const block_iq4_xs *) vbq + kbx;
    int2 v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = get_int_from_table_16(get_int_b4(bq4->qs, iqs + j), kvalues_iq4nl);
    }
    const int ls = ((bq4->scales_l[iqs/8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs/2)) & 0x03) << 4);
    const float dw = __half2float(bq4->d);
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const block_q8_1 * b8 = y + (size_t) c*ys + iqs/4;
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = ggml_cuda_dp4a(v[j].x, get_int_b4(b8->qs, j + 0), sumi);
            sumi = ggml_cuda_dp4a(v[j].y, get_int_b4(b8->qs, j + 4), sumi);
        }
        sumi *= ls - 32;
        const float d = dw * __low2float(b8->ds);
        tmp[c] = __builtin_fmaf(d, (float) sumi, tmp[c]);
    }
}
template <int NC>
static __device__ __forceinline__ void sn_accum_q4_K(const void * __restrict__ vbq, const block_q8_1 * __restrict__ y, const int ys,
        const int kbx, const int iqs, float (&tmp)[NC]) {
#pragma clang fp contract(off)
    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int v0 = q4[0], v1 = q4[4];
    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    const int j  = bq8_offset/2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);
    uint16_t aux[2];
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;
    const float2 dm4f = __half22float2(bq4_K->dm);
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
        for (int i = 0; i < QR4_K; ++i) {
            const block_q8_1 * bq8i = y + (size_t) c*ys + bq8_offset + i;
            const float d8 = __low2float(bq8i->ds);
            const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
            const int u0 = q8[0], u1 = q8[4];
            const int v0i = (v0 >> (4*i)) & 0x0F0F0F0F;
            const int v1i = (v1 >> (4*i)) & 0x0F0F0F0F;
            const int dot1 = ggml_cuda_dp4a(v1i, u1, ggml_cuda_dp4a(v0i, u0, 0));
            const int dot2 = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, 0));
            sumf_d = __builtin_fmaf(d8, (float) (dot1 * sc[i]), sumf_d);
            sumf_m = __builtin_fmaf(d8, (float) (dot2 * m[i]), sumf_m);
        }
        const float v = dm4f.x*sumf_d - dm4f.y*sumf_m;
        tmp[c] = tmp[c] + v;
    }
}
template <ggml_type type, int NC, int R>
__launch_bounds__(128, 1)
static __global__ void gemv_sn_nc(const void * __restrict__ vx, const block_q8_1 * __restrict__ y, const int ys,
        float * __restrict__ dst, const int dstr, const int bpr) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk, qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = type == GGML_TYPE_Q4_K ? VDR_Q4_K_Q8_1_MMVQ : VDR_IQ4_XS_Q8_1_MMVQ;
    constexpr int bpi = vdr*2*64/qi;
    const int row0 = R*blockIdx.x;   // R rows per workgroup: the per-row thread work and reduction are those of 1 row
    const int tid = threadIdx.y*64 + threadIdx.x;
    float tmp[R][NC];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
    for (int j = 0; j < NC; ++j) tmp[r][j] = 0.0f;
    for (int kbx = tid/(qi/vdr); kbx < bpr; kbx += bpi) {
        const int kby = kbx*(qk/QK8_1);
        const int kqs = vdr*(tid % (qi/vdr));
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if constexpr (type == GGML_TYPE_Q4_K) {
                sn_accum_q4_K<NC>(vx, y + kby, ys, (row0 + r)*bpr + kbx, kqs, tmp[r]);
            } else {
                sn_accum_iq4_xs<NC>(vx, y + kby, ys, (row0 + r)*bpr + kbx, kqs, tmp[r]);
            }
        }
    }
    __shared__ float sh[R][NC][64];
    if (threadIdx.y == 1) {
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < NC; ++j) sh[r][j][threadIdx.x] = tmp[r][j];
    }
    __syncthreads();
    if (threadIdx.y == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            float v = tmp[r][j] + sh[r][j][threadIdx.x];
            v = warp_reduce_sum<64>(v);
            if (threadIdx.x == 0) {
                dst[(size_t) j*dstr + row0 + r] = v;
            }
        }
    }
}

template <ggml_type type, int R>
static inline bool launch_sn_nc_t(int NC, const void * w, const block_q8_1 * y, int ys, float * dst, int dstr, int bpr, int N, hipStream_t st) {
    const dim3 g(N/R), b(64, 2);
    switch (NC) {
        case 2: hipLaunchKernelGGL((gemv_sn_nc<type, 2, R>), g, b, 0, st, w, y, ys, dst, dstr, bpr); return true;
        case 3: hipLaunchKernelGGL((gemv_sn_nc<type, 3, R>), g, b, 0, st, w, y, ys, dst, dstr, bpr); return true;
        case 4: hipLaunchKernelGGL((gemv_sn_nc<type, 4, R>), g, b, 0, st, w, y, ys, dst, dstr, bpr); return true;
        case 5: hipLaunchKernelGGL((gemv_sn_nc<type, 5, R>), g, b, 0, st, w, y, ys, dst, dstr, bpr); return true;
        case 6: hipLaunchKernelGGL((gemv_sn_nc<type, 6, R>), g, b, 0, st, w, y, ys, dst, dstr, bpr); return true;
        default: return false;
    }
}
// rows per workgroup R: 1 (2 rows measured no faster: the column loop of the decode is the limit, not the y reads)
static inline bool launch_sn_nc(ggml_type t, int NC, const void * w, const block_q8_1 * y, int ys, float * dst, int dstr, int K, int N, hipStream_t st,
        int R = 0) {
    if (K % QK_K != 0 || N <= 0 || (int64_t) N*(K/QK_K) >= (1ll << 31)) return false;
    if (R == 0) R = 1;
    if (R != 1) return false;
    if (t == GGML_TYPE_Q4_K)   return launch_sn_nc_t<GGML_TYPE_Q4_K, 1>(NC, w, y, ys, dst, dstr, K/QK_K, N, st);
    if (t == GGML_TYPE_IQ4_XS) return launch_sn_nc_t<GGML_TYPE_IQ4_XS, 1>(NC, w, y, ys, dst, dstr, K/QK_K, N, st);
    return false;
}

} // namespace iq4k
