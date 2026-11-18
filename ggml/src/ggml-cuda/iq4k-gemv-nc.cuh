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

// ================================================================ MTP2-B: int8 matrix-core form (NC up to 16)
// gemv_mf = gemv_nc with the integer sub-block sums of 4 rows x 4 columns from v_mfma_i32_4x4x4i8 (16 blocks of 4
// lanes). Lane t = 4*blk + q: item lane l16 = mf_l16(blk) (the single-token lane l = l16 + 16*h, h < L/16), q = row of the 4-row
// tile (L = 16: the unit's row; L = 32: unit q/2 of a pair of units, row q%2). The lane loads exactly the item the
// single-token lane (l, row) loads (same SoA copy, same chunk / header indices) and supplies its decoded int8 weights
// as the B operand; the A operand is the q8 activation of column 4g + q (chain group g). D[i][j] is the int32 sum of
// column 4g + i and row j at lane j, register i: every lane holds its own row for all columns, like the dp4a form.
// Exactness (the contract of MMVQ-NC): the int32 per (row, column, sub-block) equals the dp4a chain of the single-token
// kernel (int32 addition, any order). The fp32 steps per (row, column) are those of ncc<T>::step (contraction pinned):
//   IQ4_XS: acc = fma(dw*d8, (float) s * (float) (ls - 32), acc). |s*(ls - 32)| <= 32*127*127*32 < 2^24, so the float
//           product is exact and equals (float) __mul24(s, ls - 32).
//   Q4_K:   A = (float) dota * (float) sca, B, C = (float) suma * (float) ma, D: all |.| < 2^24, exact, equal to
//           (float) __mul24(...); then sumf_d / sumf_m / the dm epilogue exactly as ncc<q4k_t>::step (ALT for the up
//           matrix of the SwiGLU kernel).
// Each lane keeps the items l, l + L, ... of its row in order (also across K phases); the group sum is the single-token
// tree (group_sum<L>: pairs l ^ 8, ^ 4, ^ 2, ^ 1 within 16 lanes, then + the other 16 for L = 32), see mf_rsum.
// Geometry (free under the contract): the single-token launch of nwv waves owns units w, w + nwv, ... per wave w; a
// tile = the UPT = 4/G units j*UPT .. j*UPT + UPT - 1 of the list of wave w (4 rows). Virtual tile vt: L = 16 (UPT = 1)
// vt = unit index; L = 32 (UPT = 2) vt = t*nwv + w. The loads of a tile use the chunk / row indices of the single-token
// lane, so any assignment of virtual tiles to physical waves reads the same bytes. Physical wave p of a column group
// takes the tiles p, p + P, ... (P waves per group). Column groups (ncg > 1: workgroups 0..ncg-1 of the same rows, run
// side by side, so the second weight read is mostly an L2 hit) take ncw columns each (fewer LDS bytes per group).
// MU = 0: single K phase (all items of the group's columns in LDS), flat pipelined (tile, step) stream over the tiles.
// MU > 0: K phases of qph steps; every wave holds at most MU tiles, accumulators kept across the phases.
// LDS layout per phase of ni items (item = one q8_1 block for IQ4_XS, a pair for Q4_K), item record of IS words at i*IS:
//   QS: column c: QW words (the q8 qs of the item) at c*CS (CS = QW = 8 / 16 words)
//   S:  at SO = CS*NC4: column group g, field f: 4 words (the 4 columns) at SO + (g*F + f)*4: IQ4_XS F = 1 {d8};
//       Q4_K F = 4 {da, db, (float) suma, (float) sumb} (block sums as in q4k_t::put_y, exact as float)
//   IS = (CS + F)*NC4 rounded up to 4 mod 8: a ds_read_b128 of the wave (16 items x 4 columns, or 16 items x 1 S record)
//   spreads evenly over the eight 4-bank groups; all offsets inside a record are immediates.
// Columns ncl .. 4*CG - 1 of the last chain group read zeroed records: accumulated, never stored.
typedef int mf_i32x4 __attribute__((ext_vector_type(4)));
template <class T> struct mfl;
template <> struct mfl<iq4xs_t> { static constexpr int QW = 8,  CS = 8,  F = 1; };
template <> struct mfl<q4k_t>   { static constexpr int QW = 16, CS = 16, F = 4; };
static __host__ __device__ constexpr int mf_rec(const int cs, const int f, const int nc4) { return (cs + f)*nc4 + ((cs + f)*nc4 % 8 == 4 ? 0 : 4); }
template <class T> static __host__ __device__ constexpr int mf_is(const int nc4) { return mf_rec(mfl<T>::CS, mfl<T>::F, nc4); }

static __device__ __forceinline__ mf_i32x4 mf_mma(const uint32_t a, const uint32_t b, const mf_i32x4 c) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_amdgcn_mfma_i32_4x4x4i8((int) a, (int) b, c, 0, 0, 0);
#else
    (void) a; (void) b; return c;
#endif
}
static __device__ __forceinline__ float mf_mul(const float a, const float b) {
#pragma clang fp contract(off)
    return a*b;
}
static __device__ __forceinline__ float mf_add(const float a, const float b) {
#pragma clang fp contract(off)
    return a + b;
}
static __device__ __forceinline__ float mf_sub(const float a, const float b) {
#pragma clang fp contract(off)
    return a - b;
}
// int32 -> float of the MFMA sums without v_cvt: the chains start from MF_MAGIC = bits of 1.5*2^23, so the sum s' = MAGIC + s
// read as a float is exactly 1.5*2^23 + s (|s| < 2^22: IQ4_XS |s| <= 32*127*127, Q4_K <= 32*15*127). mf_ix(s', f, -1.5*2^23*f)
// = fma(1.5*2^23 + s, f, -1.5*2^23*f) = the exact s*f (|s*f| < 2^24 and 1.5*2^23*f exact for |f| <= 63) = (float) s * f =
// (float) __mul24(s, f) of the single-token kernel, in one packable fma instead of v_cvt + v_mul.
static constexpr int   MF_MAGIC   = 0x4B400000;
static constexpr float MF_MAGIC_F = 12582912.0f;
static __device__ __forceinline__ float mf_ix(const int sm, const float f, const float nf) {
    return __builtin_fmaf(__int_as_float(sm), f, nf);
}
// m ? a : b with an all-ones / zero lane mask (bit select: keeps the compiler from turning the per-lane choice of
// register-array elements into dynamic register indexing)
static __device__ __forceinline__ float mf_sel(const uint32_t m, const float a, const float b) {
    return __uint_as_float((__float_as_uint(a) & m) | (__float_as_uint(b) & ~m));
}
// item lane of MFMA block blk (lanes 4*blk .. 4*blk + 3): l16 = bit reverse of blk, so the reduction tree (l16 bit 3, 2,
// 1, 0 = pairs l ^ 8, ^ 4, ^ 2, ^ 1 of group_sum<16>) runs over lane bits 2, 3, 4, 5: the steps with the most values
// use DPP inside a row of 16 lanes, the last one (one value) a cross-row shuffle
static __device__ __forceinline__ int mf_l16(const int blk) {
    return ((blk & 1) << 3) | ((blk & 2) << 1) | ((blk & 4) >> 1) | ((blk & 8) >> 3);
}
// value of the lane whose item lane is l16 ^ (1 << b) (same q)
template <int b>
static __device__ __forceinline__ float mf_xfer(const float v) {
#if defined(__HIP_DEVICE_COMPILE__)
    if constexpr (b == 3) {          // lane ^ 4: row_ror:4 brings lane i - 4, row_ror:12 lane i + 4
        const float lo = __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x124, 0xF, 0xF, false));
        const float up = __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x12C, 0xF, 0xF, false));
        return (threadIdx.x & 4) ? lo : up;
    } else if constexpr (b == 2) {   // lane ^ 8: row_ror:8
        return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x128, 0xF, 0xF, false));
    } else if constexpr (b == 1) {   // lane ^ 16: ds_swizzle bit mode (and 0x1F, xor 0x10)
        return __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F));
    } else {                         // lane ^ 32
        return __shfl_xor(v, 32, 64);
    }
#else
    return v;
#endif
}
// the group sums of all NC4 columns at once, scattered: lane l16 returns the sum of column l16 % C (C = NC4 rounded up
// to 4, 8, 16). Same tree per column as group_sum<16> (l16 bit 3, 2, 1, 0): a bit >= log2(C) is a butterfly step on
// every column, a lower bit keeps the half of the columns whose bit equals the lane's (fp add is commutative, so each
// kept partial is the butterfly value of that column).
template <int b, int C>
static __device__ __forceinline__ void mf_rstep(float (&v)[C], const int l16) {
    if constexpr ((1 << b) >= C) {
#pragma unroll
        for (int k = 0; k < C; ++k) v[k] = mf_add(v[k], mf_xfer<b>(v[k]));
    } else {
        constexpr int h = 1 << b;   // columns left: 2h
        uint32_t m = 0u - (uint32_t) ((l16 >> b) & 1);
#if defined(__HIP_DEVICE_COMPILE__)
        asm volatile("" : "+v"(m));
#endif
#pragma unroll
        for (int k = 0; k < h; ++k) {
            const float send = mf_sel(m, v[k], v[k + h]);
            const float keep = mf_sel(m, v[k + h], v[k]);
            v[k] = mf_add(keep, mf_xfer<b>(send));
        }
    }
}
template <int NC4>
static __device__ __forceinline__ float mf_rsum(const float (&a)[NC4], const int l16) {
    constexpr int C = NC4 <= 4 ? 4 : (NC4 <= 8 ? 8 : 16);
    float v[C];
#pragma unroll
    for (int j = 0; j < C; ++j) v[j] = j < NC4 ? a[j < NC4 ? j : 0] : 0.0f;
    mf_rstep<3, C>(v, l16);
    mf_rstep<2, C>(v, l16);
    mf_rstep<1, C>(v, l16);
    mf_rstep<0, C>(v, l16);
    return v[0];
}

// stage items [i0, i1) of ncl columns (column c at yg + c*ys blocks) into the LDS layout above
template <class T>
static __device__ __forceinline__ void mf_stage(uint32_t * __restrict__ ysm, const int ncl, const int nc4,
        const block_q8_1 * __restrict__ yg, const int ys, const int i0, const int i1, const int tid, const int nt) {
    constexpr int CS = mfl<T>::CS, F = mfl<T>::F, YPI = T::YPI;
    const int IS = mf_is<T>(nc4), SO = CS*nc4;
    const int nb = (i1 - i0)*YPI, b0 = i0*YPI;
    for (int x = tid; x < ncl*nb; x += nt) {
        const int c = x / nb, bb = x - c*nb;
        const uint8_t * p = (const uint8_t *) (yg + (size_t) c*ys + b0 + bb);
        const uint32_t ds = ld_u32_opaque(p);
        const uint4 q0 = *(const uint4 *) (p + 4);
        const uint4 q1 = *(const uint4 *) (p + 20);
        const int it = YPI == 1 ? bb : bb >> 1, h = YPI == 1 ? 0 : bb & 1;
        uint32_t * rec = ysm + it*IS;
        uint32_t * qd = rec + c*CS + 8*h;
        *(uint4 *) qd = q0; *(uint4 *) (qd + 4) = q1;
        uint32_t * sd = rec + SO + (c >> 2)*F*4 + (c & 3);
        const float d = h2f(ds);
        if constexpr (YPI == 1) {
            sd[0] = __float_as_uint(d);
        } else {
            int s = dp4a((int) q0.x, 0x01010101, 0); s = dp4a((int) q0.y, 0x01010101, s); s = dp4a((int) q0.z, 0x01010101, s); s = dp4a((int) q0.w, 0x01010101, s);
            s = dp4a((int) q1.x, 0x01010101, s); s = dp4a((int) q1.y, 0x01010101, s); s = dp4a((int) q1.z, 0x01010101, s); s = dp4a((int) q1.w, 0x01010101, s);
            sd[4*h]       = __float_as_uint(d);
            sd[4*(2 + h)] = __float_as_uint((float) s);
        }
    }
    // columns ncl .. 4*ceil(ncl/4) - 1 of the last chain group are read (never stored): zero their records
    const int ncx = (ncl + 3) & ~3;
    for (int x = tid; x < (ncx - ncl)*(i1 - i0); x += nt) {
        const int c = ncl + x / (i1 - i0), it = x % (i1 - i0);
        uint32_t * rec = ysm + it*IS;
#pragma unroll
        for (int k = 0; k < mfl<T>::QW; ++k) rec[c*CS + k] = 0u;
#pragma unroll
        for (int f = 0; f < F; ++f) rec[SO + (c >> 2)*F*4 + 4*f + (c & 3)] = 0u;
    }
}

// one item of the lane's row for all chain groups: a[col] += item term (all 4*CG columns; the padded ones unused)
template <class T, int CG, bool ALT> struct mfc;
template <int CG, bool ALT> struct mfc<iq4xs_t, CG, ALT> {
    static __device__ __forceinline__ void item(const iq4xs_t::raw & r, const iq4xs_t::lc & c, const uint32_t * __restrict__ yq0,
            const uint32_t * __restrict__ sp, float (&a)[4*CG]) {
        const typename ncc<iq4xs_t, 1>::wdec w = ncc<iq4xs_t, 1>::dec(r, c);
        const float fm = (float) (w.ls - 32), nm = mf_mul(-MF_MAGIC_F, fm);
#pragma unroll
        for (int g = 0; g < CG; ++g) {
            const uint32_t * yq = yq0 + g*4*mfl<iq4xs_t>::CS;
            const uint4 y0 = *(const uint4 *) yq, y1 = *(const uint4 *) (yq + 4);
            const float4 d8 = *(const float4 *) (sp + 4*g);
            mf_i32x4 s = {MF_MAGIC, MF_MAGIC, MF_MAGIC, MF_MAGIC};
            s = mf_mma(y0.x, w.lo[0], s); s = mf_mma(y0.y, w.lo[1], s); s = mf_mma(y0.z, w.lo[2], s); s = mf_mma(y0.w, w.lo[3], s);
            s = mf_mma(y1.x, w.hi[0], s); s = mf_mma(y1.y, w.hi[1], s); s = mf_mma(y1.z, w.hi[2], s); s = mf_mma(y1.w, w.hi[3], s);
            const float dv[4] = {d8.x, d8.y, d8.z, d8.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float d = mf_mul(w.dw, dv[i]);
                a[4*g + i] = __builtin_fmaf(d, mf_ix(s[i], fm, nm), a[4*g + i]);
            }
        }
    }
};
template <int CG, bool ALT> struct mfc<q4k_t, CG, ALT> {
    static __device__ __forceinline__ void item(const q4k_t::raw & r, const q4k_t::lc & c, const uint32_t * __restrict__ yq0,
            const uint32_t * __restrict__ sp, float (&a)[4*CG]) {
        const typename ncc<q4k_t, 1>::wdec w = ncc<q4k_t, 1>::dec(r, c);
        const float fsa = (float) w.sca, fsb = (float) w.scb, fma_ = (float) w.ma, fmb = (float) w.mb;
        const float nsa = mf_mul(-MF_MAGIC_F, fsa), nsb = mf_mul(-MF_MAGIC_F, fsb);
        uint32_t lo[8], hi[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) { lo[k] = w.q[k] & 0x0F0F0F0Fu; hi[k] = (w.q[k] >> 4) & 0x0F0F0F0Fu; }
#pragma unroll
        for (int g = 0; g < CG; ++g) {
            const uint32_t * yq = yq0 + g*4*mfl<q4k_t>::CS;
            const uint4 ya0 = *(const uint4 *) yq,       ya1 = *(const uint4 *) (yq + 4);
            const uint4 yb0 = *(const uint4 *) (yq + 8), yb1 = *(const uint4 *) (yq + 12);
            const float * sr = (const float *) (sp + 16*g);
            const float4 da4 = *(const float4 *) sr, db4 = *(const float4 *) (sr + 4);
            const float4 sa4 = *(const float4 *) (sr + 8), sb4 = *(const float4 *) (sr + 12);
            mf_i32x4 xa = {MF_MAGIC, MF_MAGIC, MF_MAGIC, MF_MAGIC}, xb = {MF_MAGIC, MF_MAGIC, MF_MAGIC, MF_MAGIC};
            xa = mf_mma(ya0.x, lo[0], xa); xb = mf_mma(yb0.x, hi[0], xb);
            xa = mf_mma(ya0.y, lo[1], xa); xb = mf_mma(yb0.y, hi[1], xb);
            xa = mf_mma(ya0.z, lo[2], xa); xb = mf_mma(yb0.z, hi[2], xb);
            xa = mf_mma(ya0.w, lo[3], xa); xb = mf_mma(yb0.w, hi[3], xb);
            xa = mf_mma(ya1.x, lo[4], xa); xb = mf_mma(yb1.x, hi[4], xb);
            xa = mf_mma(ya1.y, lo[5], xa); xb = mf_mma(yb1.y, hi[5], xb);
            xa = mf_mma(ya1.z, lo[6], xa); xb = mf_mma(yb1.z, hi[6], xb);
            xa = mf_mma(ya1.w, lo[7], xa); xb = mf_mma(yb1.w, hi[7], xb);
            const float dav[4] = {da4.x, da4.y, da4.z, da4.w}, dbv[4] = {db4.x, db4.y, db4.z, db4.w};
            const float sav[4] = {sa4.x, sa4.y, sa4.z, sa4.w}, sbv[4] = {sb4.x, sb4.y, sb4.z, sb4.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float da = dav[i], db = dbv[i];
                const float A = mf_ix(xa[i], fsa, nsa), B = mf_ix(xb[i], fsb, nsb);
                const float C = mf_mul(sav[i], fma_), D = mf_mul(sbv[i], fmb);
                float v;
                if constexpr (ALT) {
                    const float sumf_d = __builtin_fmaf(da, A, mf_mul(db, B));
                    const float sumf_m = __builtin_fmaf(da, C, mf_mul(db, D));
                    v = __builtin_fmaf(sumf_d, w.dm.x, -mf_mul(w.dm.y, sumf_m));
                } else {
                    const float sumf_d = __builtin_fmaf(db, B, mf_mul(da, A));
                    const float sumf_m = __builtin_fmaf(da, C, mf_mul(db, D));
                    v = mf_sub(mf_mul(w.dm.x, sumf_d), mf_mul(w.dm.y, sumf_m));
                }
                a[4*g + i] = mf_add(a[4*g + i], v);
            }
        }
    }
};

// load geometry of one tile for this lane: chunk index of step kk = q0 + kk*dq (Q stream, 16-byte pieces) and
// s0 + kk*dsx (header entries); valid = the lane's row exists (else a clamped duplicate: loaded, never stored)
struct mfgeo { uint32_t q0, dq, s0, dsx; int row; bool valid; };
template <class T, int L, bool RT>
static __device__ __forceinline__ mfgeo mf_geo(const int vt, const int pu, const int rr, const int nwv, const int nunits,
        const int qn, const int bpr) {
    constexpr int G = 64/L, UPT = 4/G;
    mfgeo g;
    int w, j, myu;
    if constexpr (UPT == 1) {
        w = RT ? vt % nwv : vt; j = RT ? vt / nwv : 0; myu = j + 1;
    } else {
        w = vt % nwv; const int t = vt / nwv;
        myu = w < nunits ? (nunits - 1 - w)/nwv + 1 : 0;
        j = UPT*t + pu;
    }
    g.valid = j < myu;
    const int jj = min(j, myu - 1);
    const int un = RT ? w + jj*nwv : w;
    g.row = un*G + rr;
    if constexpr (RT) {
        const int nur = min(nwv, nunits - jj*nwv);
        const uint32_t ch = (uint32_t) (jj*nwv*qn + w);
        g.q0 = ch*(L*G); g.dq = (uint32_t) nur*(L*G);
        g.s0 = ch*((L/T::IPB)*G); g.dsx = (uint32_t) nur*((L/T::IPB)*G);
    } else {
        g.q0 = (uint32_t) un*bpr*8*G; g.dq = L*G;
        g.s0 = (uint32_t) un*bpr*G;   g.dsx = (L/T::IPB)*G;
    }
    return g;
}

// launch bounds: the configurations that need more than 128 (Q4_K SwiGLU: 168) registers per lane get 8 waves per
// workgroup (3-4 chain groups; 2 chain groups held across K phases with 2 tiles, 2 item halves or the SwiGLU pair)
static __host__ __device__ constexpr bool mf_big(const int L, const bool glu, const int cg, const int mu, const bool q4k) {
    return cg >= 3 || (cg == 2 && mu >= 1 && (glu || q4k || mu >= 2 || L == 32)) || (cg == 1 && mu >= 2 && (L == 32 || (glu && q4k)));
}
template <class T, int L, bool GLU, int CG, int MU> struct mf_lb { static constexpr int v = mf_big(L, GLU, CG, MU, std::is_same<T, q4k_t>::value) ? 512 : nc_lb<T, GLU>::v; };
template <class T, int L, bool GLU, int CG, int MU, bool RT>
__launch_bounds__((mf_lb<T, L, GLU, CG, MU>::v), 1)
__global__ void gemv_mf(const soa s, const soa sg, const block_q8_1 * __restrict__ yg, const int ys,
        float * __restrict__ dst, const int dstr, const float * __restrict__ residual, const int rstr,
        const int K, const int N, const int qph, const int nc, const int ncw, const int ncg, const int nwv, const int ntl) {
    extern __shared__ __align__(16) uint32_t ysm[];
    constexpr int G = 64/L, H = L/16, GH = GLU ? H : 1, NC4 = 4*CG, MUX = MU > 0 ? MU : 1, GM = GLU ? MUX : 1;
    static_assert(L == 16 || L == 32, "L");
    const int tid  = threadIdx.y*64 + threadIdx.x;
    const int nt   = blockDim.y*64;
    const int l16  = mf_l16((int) threadIdx.x >> 2), q = threadIdx.x & 3;
    const int pu   = q / G, rr = q % G;
    const int bpr  = K / QK_K;
    const int ipr  = bpr*T::IPB;
    const int qn   = ipr / L;
    const int nunits = N / G;
    const int cgi  = (int) blockIdx.x % ncg, wg = (int) blockIdx.x / ncg, nwg = (int) gridDim.x / ncg;
    const int c0   = cgi*ncw, ncl = min(ncw, nc - c0);
    const int wy   = __builtin_amdgcn_readfirstlane(threadIdx.y);
    const int P    = nwg*(int) blockDim.y;
    const int p    = wy*nwg + wg;
    const int myt  = p < ntl ? (ntl - 1 - p)/P + 1 : 0;   // tiles of this wave: p, p + P, ...
    constexpr int IS = mf_is<T>(NC4);
    const uint32_t * yq0 = ysm + q*mfl<T>::CS;           // lane's column q of chain group 0, item 0
    const uint32_t * sq0 = ysm + mfl<T>::CS*NC4;         // S records of item 0
    const typename T::lc c = T::lane_consts(l16);
    const uint32_t boff = RT ? (uint32_t) ((size_t) N*bpr*4) : 4*bpr*G;
    yg += (size_t) c0*ys;
    dst += (size_t) c0*dstr;
    if (residual) residual += (size_t) c0*rstr;
    uint32_t qoff[H], soff[H];
#pragma unroll
    for (int h = 0; h < H; ++h) { const int l = l16 + 16*h; qoff[h] = l*G + rr; soff[h] = (l/T::IPB)*G + rr; }

    typedef typename T::raw raw;
    raw A[H], B[H], Ag[GH], Bg[GH];
    auto ld = [&](raw (&bb)[H], raw (&bg)[GH], const mfgeo & g, const int kk) {
        const uint32_t qs_ = g.q0 + kk*g.dq, ss_ = g.s0 + kk*g.dsx;
#pragma unroll
        for (int h = 0; h < H; ++h) {
            bb[h] = T::load_c(s, qs_ + qoff[h], ss_ + soff[h], boff);
            if constexpr (GLU) bg[h] = T::load_c(sg, qs_ + qoff[h], ss_ + soff[h], boff);
        }
    };
    // step (item lanes l16 + 16h of local step kl) into the accumulators
    auto comp = [&](const raw (&bb)[H], const raw (&bg)[GH], float (&a)[H][NC4], float (&ag)[GH][NC4], const int kl) {
#pragma unroll
        for (int h = 0; h < H; ++h) {
            const int it = kl*L + l16 + 16*h;
            const uint32_t * yq = yq0 + it*IS;
            const uint32_t * sq = sq0 + it*IS;
            if constexpr (GLU) {
                mfc<T, CG, true >::item(bb[h], c, yq, sq, a[h]);
                mfc<T, CG, false>::item(bg[h], c, yq, sq, ag[h]);
            } else {
                mfc<T, CG, false>::item(bb[h], c, yq, sq, a[h]);
            }
        }
    };
    // end of a tile: group sums, epilogue; lane l16 stores column l16 of its row
    auto finish = [&](const float (&a)[H][NC4], const float (&ag)[GH][NC4], const mfgeo & g) {
        float o = mf_rsum<NC4>(a[0], l16);
        if constexpr (H == 2) o = mf_add(o, mf_rsum<NC4>(a[1], l16));   // group_sum<32>: + the other 16 item lanes
        if constexpr (GLU) {
            float og = mf_rsum<NC4>(ag[0], l16);
            if constexpr (H == 2) og = mf_add(og, mf_rsum<NC4>(ag[1], l16));
            o *= silu_f(og);
        }
        if (g.valid && l16 < ncl) {
            if (residual) o += residual[(size_t) l16*rstr + g.row];
            dst[(size_t) l16*dstr + g.row] = o;
        }
    };

    if constexpr (MU == 0) {
        // single phase: flat (tile, step) stream, ping-pong pipelined (the load cursor runs one step ahead, clamped)
        const int nst = myt*qn;
        int li = 0, lk = 0;
        mfgeo lg = mf_geo<T, L, RT>(p, pu, rr, nwv, nunits, qn, bpr);
        auto adv = [&]() {   // load cursor -> next step (stays on the last step at the end)
            if (++lk == qn) {
                if (li + 1 < myt) { lk = 0; ++li; lg = mf_geo<T, L, RT>(p + li*P, pu, rr, nwv, nunits, qn, bpr); }
                else lk = qn - 1;
            }
        };
        mfgeo cg = lg;
        if (nst > 0) { ld(A, Ag, lg, 0); adv(); }
        mf_stage<T>(ysm, ncl, NC4, yg, ys, 0, qn*L, tid, nt);
        __syncthreads();
        float acc[H][NC4], accg[GH][NC4];
#pragma unroll
        for (int h = 0; h < H; ++h)
#pragma unroll
        for (int j = 0; j < NC4; ++j) { acc[h][j] = 0.0f; if constexpr (GLU) accg[h % GH][j] = 0.0f; }
        // (cg: geometry of the tile being computed, taken before the first adv(): with qn = 1 that already moves lg on)
        int ck = 0, ci = 0;
        auto cstep = [&](const raw (&bb)[H], const raw (&bg)[GH]) {
            comp(bb, bg, acc, accg, ck);
            if (++ck == qn) {
                finish(acc, accg, cg);
#pragma unroll
                for (int h = 0; h < H; ++h)
#pragma unroll
                for (int j = 0; j < NC4; ++j) { acc[h][j] = 0.0f; if constexpr (GLU) accg[h % GH][j] = 0.0f; }
                ck = 0;
                if (++ci < myt) cg = mf_geo<T, L, RT>(p + ci*P, pu, rr, nwv, nunits, qn, bpr);
            }
        };
        for (int st = 0; st < nst; st += 2) {
            ld(B, Bg, lg, lk); adv();
            cstep(A, Ag);
            if (st + 1 >= nst) break;
            ld(A, Ag, lg, lk); adv();
            cstep(B, Bg);
        }
    } else {
        // K phases; tiles m < MU of this wave held across them
        mfgeo tg[MUX];
#pragma unroll
        for (int m = 0; m < MUX; ++m) tg[m] = mf_geo<T, L, RT>(min(p + m*P, ntl - 1), pu, rr, nwv, nunits, qn, bpr);
        float acc[MUX][H][NC4], accg[GM][GH][NC4];
#pragma unroll
        for (int m = 0; m < MUX; ++m)
#pragma unroll
        for (int h = 0; h < H; ++h)
#pragma unroll
        for (int j = 0; j < NC4; ++j) { acc[m][h][j] = 0.0f; if constexpr (GLU) accg[m % GM][h % GH][j] = 0.0f; }
        const int nph = (qn + qph - 1)/qph;
        for (int ph = 0; ph < nph; ++ph) {
            const int k0 = ph*qph, k1 = min(qn, k0 + qph), qp = k1 - k0;
            if (myt > 0) ld(A, Ag, tg[0], k0);
            if (ph > 0) __syncthreads();
            mf_stage<T>(ysm, ncl, NC4, yg, ys, k0*L, k1*L, tid, nt);
            __syncthreads();
#pragma unroll
            for (int m = 0; m < MUX; ++m) {
                if (m < myt) {
                    // tile m; for even qp the last load of tile m - 1 already fetched step k0 of tile m into A
                    if (m > 0 && (qp & 1)) ld(A, Ag, tg[m], k0);
                    for (int k = k0; k < k1; k += 2) {
                        ld(B, Bg, tg[m], min(k + 1, k1 - 1));
                        comp(A, Ag, acc[m], accg[GLU ? m : 0], k - k0);
                        if (k + 2 < k1)                       ld(A, Ag, tg[m], k + 2);
                        else if (!(qp & 1) && m + 1 < myt)    ld(A, Ag, tg[m + 1], k0);
                        if (k + 1 < k1) comp(B, Bg, acc[m], accg[GLU ? m : 0], k + 1 - k0);
                    }
                    if (k1 == qn) finish(acc[m], accg[GLU ? m : 0], tg[m]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- host side of gemv_mf
// plan: ncg column groups of ncw columns (chain groups cg), K phases of qph steps, mu = tiles held per wave (0: single
// phase), launch grid (ncg*nwg workgroups of nw waves), ntl virtual tiles, nwv = single-token waves (layout)
struct mfplan { int ncg, ncw, cg, mu, qph, nw, nwg, ntl, nwv; size_t lds; };
static constexpr int MF_MU_MAX = 2;
static constexpr int MF_ACC_MAX = 32;   // accumulator registers per lane held across K phases
static inline size_t mf_lds(ggml_type t, int L, int qph, int cg) {
    const int nc4 = 4*cg;
    const int is = t == GGML_TYPE_Q4_K ? mf_is<q4k_t>(nc4) : mf_is<iq4xs_t>(nc4);
    return 4*(size_t) qph*L*is;
}
// configurations the MFMA kernel is instantiated for: L = 16 (row-interleaved or round-tiled), L = 32 round-tiled
// (IQ4_XS K > 8192), GLU on L = 16 row-interleaved only
static inline bool mf_cfg_ok(const cfg & c, ggml_type t, bool glu) {
    if (c.L == 16) return !glu || !c.rt;
    return c.L == 32 && c.rt && !glu && t == GGML_TYPE_IQ4_XS;
}
// ncg_force > 0: that many column groups (benchmarking); lds_max: LDS budget per workgroup
static inline bool mf_plan(ggml_type t, const cfg & c, bool glu, int K, int N, int nc, int ncu, mfplan & p, int ncg_force = 0,
        size_t lds_max = NC_LDS_MAX) {
    if (!mf_cfg_ok(c, t, glu) || nc < 1 || nc > 16) return false;
    const int ipb = t == GGML_TYPE_Q4_K ? 4 : 8;
    const int ipr = (K/QK_K)*ipb, qn = ipr/c.L;
    const int G = 64/c.L, UPT = 4/G, H = c.L/16, nunits = N/G;
    const int nwv = c.grid*c.NW;
    if (ipr % c.L != 0 || N % G != 0) return false;
    const int maxu = (nunits + nwv - 1)/nwv;
    // UPT = 2: tile t of wave w (w < nwv; waves w >= nunits own no unit when nwv > nunits)
    const int ntl = UPT == 1 ? nunits : ((maxu + UPT - 1)/UPT)*std::min(nwv, nunits);
    const int nwmax0 = (glu && t == GGML_TYPE_Q4_K) ? 12 : 16;
    for (int ncg = 1; ncg <= 4; ++ncg) {
        if (ncg_force > 0 && ncg != ncg_force) continue;
        const int ncw = (nc + ncg - 1)/ncg;
        if ((nc + ncw - 1)/ncw != ncg) continue;
        const int cg = (ncw + 3)/4;
        const int nwmax = mf_big(c.L, glu, cg, 0, t == GGML_TYPE_Q4_K) ? 8 : nwmax0;   // = mf_lb/64
        int nph = 1;
        for (; nph <= qn; ++nph) if (mf_lds(t, c.L, (qn + nph - 1)/nph, cg) <= lds_max) break;
        if (nph > qn) continue;
        const int qph = (qn + nph - 1)/nph;
        p.ncg = ncg; p.ncw = ncw; p.cg = cg; p.qph = qph; p.ntl = ntl; p.nwv = nwv;
        p.lds = mf_lds(t, c.L, qph, cg);
        const int wpg = std::max(1, ncu/ncg);   // workgroups per column group in one round
        if (nph == 1) {
            p.mu = 0;
            const int mt = (ntl + wpg*nwmax - 1)/(wpg*nwmax);   // tiles per wave
            const int P  = (ntl + mt - 1)/mt;
            p.nw  = std::min(nwmax, (P + wpg - 1)/wpg);
            p.nwg = (P + p.nw - 1)/p.nw;
            return true;
        }
        if (ncg_force == 0 && ncg < 4) {
            // several phases: prefer more column groups if those fit in one phase
            const int ncw2 = (nc + ncg)/(ncg + 1), cg2 = (ncw2 + 3)/4;
            // (measured: SwiGLU and IQ4_XS K = 6144 stay faster with one group: the weight decode is not duplicated)
            if (ncw2 >= 1 && cg2 < cg && mf_lds(t, c.L, qn, cg2) <= lds_max && !glu && (N >= 8192 || t == GGML_TYPE_Q4_K)) continue;
        }
        const int per = H*4*cg*(glu ? 2 : 1);
        const int mur = std::max(1, std::min(MF_MU_MAX, MF_ACC_MAX/per));   // register limit
        // fewest tiles per wave that still cover all tiles in one round (most waves: latency hiding), at most mur
        const int nwr = mf_big(c.L, glu, cg, 1, t == GGML_TYPE_Q4_K) ? 8 : nwmax0;
        const int mu = std::min(mur, std::max(1, (ntl + wpg*nwr - 1)/(wpg*nwr)));
        const int Pmin = (ntl + mu - 1)/mu;
        p.mu  = (ntl + Pmin - 1)/Pmin;
        const int P = (ntl + p.mu - 1)/p.mu;
        p.nw  = std::min(mf_big(c.L, glu, cg, p.mu, t == GGML_TYPE_Q4_K) ? 8 : nwmax0, (P + wpg - 1)/wpg);
        p.nwg = (P + p.nw - 1)/p.nw;
        return true;
    }
    return false;
}

template <class T, int L, bool GLU, int CG, int MU, bool RT>
static inline void launch_mf_k(const mfplan & p, int nc, const soa & s, const soa & sg, const block_q8_1 * y, int ys,
        float * dst, int dstr, const float * res, int rstr, int K, int N, hipStream_t st) {
    hipLaunchKernelGGL((gemv_mf<T, L, GLU, CG, MU, RT>), dim3(p.nwg*p.ncg), dim3(64, p.nw), p.lds, st, s, sg, y, ys, dst, dstr, res, rstr,
        K, N, p.qph, nc, p.ncw, p.ncg, p.nwv, p.ntl);
}
template <class T, int L, bool GLU, int CG, bool RT>
static inline bool launch_mf_mu(const mfplan & p, int nc, const soa & s, const soa & sg, const block_q8_1 * y, int ys,
        float * dst, int dstr, const float * res, int rstr, int K, int N, hipStream_t st) {
    switch (p.mu) {
        case 0: launch_mf_k<T, L, GLU, CG, 0, RT>(p, nc, s, sg, y, ys, dst, dstr, res, rstr, K, N, st); return true;
        case 1: launch_mf_k<T, L, GLU, CG, 1, RT>(p, nc, s, sg, y, ys, dst, dstr, res, rstr, K, N, st); return true;
        case 2: launch_mf_k<T, L, GLU, CG, 2, RT>(p, nc, s, sg, y, ys, dst, dstr, res, rstr, K, N, st); return true;
        default: return false;
    }
}
template <class T, bool GLU, int CG>
static inline bool launch_mf_l(const cfg & c, const mfplan & p, int nc, const soa & s, const soa & sg, const block_q8_1 * y, int ys,
        float * dst, int dstr, const float * res, int rstr, int K, int N, hipStream_t st) {
    if (p.nw*64 > (mf_big(c.L, GLU, CG, p.mu, std::is_same<T, q4k_t>::value) ? 512 : nc_lb<T, GLU>::v)) return false;
    if constexpr (GLU) {
        if (c.L != 16 || c.rt) return false;
        return launch_mf_mu<T, 16, true, CG, false>(p, nc, s, sg, y, ys, dst, dstr, res, rstr, K, N, st);
    } else {
        if (c.L == 16) {
            return c.rt ? launch_mf_mu<T, 16, false, CG, true >(p, nc, s, sg, y, ys, dst, dstr, res, rstr, K, N, st)
                        : launch_mf_mu<T, 16, false, CG, false>(p, nc, s, sg, y, ys, dst, dstr, res, rstr, K, N, st);
        }
        if constexpr (std::is_same<T, iq4xs_t>::value) {
            if (c.L == 32 && c.rt) return launch_mf_mu<T, 32, false, CG, true>(p, nc, s, sg, y, ys, dst, dstr, res, rstr, K, N, st);
        }
        return false;
    }
}
template <class T, bool GLU>
static inline bool launch_mf(const cfg & c, const mfplan & p, int nc, const soa & s, const soa & sg, const block_q8_1 * y, int ys,
        float * dst, int dstr, const float * res, int rstr, int K, int N, hipStream_t st) {
    switch (p.cg) {
        case 1: return launch_mf_l<T, GLU, 1>(c, p, nc, s, sg, y, ys, dst, dstr, res, rstr, K, N, st);
        case 2: return launch_mf_l<T, GLU, 2>(c, p, nc, s, sg, y, ys, dst, dstr, res, rstr, K, N, st);
        case 3: return launch_mf_l<T, GLU, 3>(c, p, nc, s, sg, y, ys, dst, dstr, res, rstr, K, N, st);
        case 4: return launch_mf_l<T, GLU, 4>(c, p, nc, s, sg, y, ys, dst, dstr, res, rstr, K, N, st);
        default: return false;
    }
}

// ---------------------------------------------------------------- MTP2-B: small-N kernel with R rows per workgroup
// gemv_sn_ncr = gemv_sn_nc with the loops turned around: per item the R rows are decoded once, then per column the q8
// words are read once (global / L2) and used for all R rows, so the activation traffic is 1/R of the per-row kernel
// (the limit beyond 4 columns). Per (row, column) the thread's work, the fp32 sequence of the stock kernel and the
// reduction (wave 1 -> LDS, wave 0 adds, warp_reduce_sum<64>) are exactly those of gemv_sn_nc / mul_mat_vec_q.
template <ggml_type type, int NC, int R>
__launch_bounds__(128, 1)
static __global__ void gemv_sn_ncr(const void * __restrict__ vx, const block_q8_1 * __restrict__ y, const int ys,
        float * __restrict__ dst, const int dstr, const int bpr) {
#pragma clang fp contract(off)
    constexpr int qk  = ggml_cuda_type_traits<type>::qk, qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = type == GGML_TYPE_Q4_K ? VDR_Q4_K_Q8_1_MMVQ : VDR_IQ4_XS_Q8_1_MMVQ;
    constexpr int bpi = vdr*2*64/qi;
    const int row0 = R*blockIdx.x;
    const int tid = threadIdx.y*64 + threadIdx.x;
    float tmp[R][NC];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
    for (int j = 0; j < NC; ++j) tmp[r][j] = 0.0f;
    for (int kbx = tid/(qi/vdr); kbx < bpr; kbx += bpi) {
        const int kby = kbx*(qk/QK8_1);
        const int iqs = vdr*(tid % (qi/vdr));
        if constexpr (type == GGML_TYPE_IQ4_XS) {
            int2 v[R][4]; int ls[R]; float dw[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const block_iq4_xs * bq4 = (const block_iq4_xs *) vx + (size_t) (row0 + r)*bpr + kbx;
#pragma unroll
                for (int j = 0; j < 4; ++j) v[r][j] = get_int_from_table_16(get_int_b4(bq4->qs, iqs + j), kvalues_iq4nl);
                ls[r] = ((bq4->scales_l[iqs/8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs/2)) & 0x03) << 4);
                dw[r] = __half2float(bq4->d);
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const block_q8_1 * b8 = y + (size_t) c*ys + kby + iqs/4;
                const uint8_t * pb = (const uint8_t *) b8;   // the 32 qs as two 16-byte loads (4-byte aligned)
                const uint4 ua = *(const uint4 *) (pb + 4), ub = *(const uint4 *) (pb + 20);
                const int u[8] = {(int) ua.x, (int) ua.y, (int) ua.z, (int) ua.w, (int) ub.x, (int) ub.y, (int) ub.z, (int) ub.w};
                const float d8 = __low2float(b8->ds);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    int sumi = 0;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        sumi = ggml_cuda_dp4a(v[r][j].x, u[j], sumi);
                        sumi = ggml_cuda_dp4a(v[r][j].y, u[j + 4], sumi);
                    }
                    sumi *= ls[r] - 32;
                    const float d = dw[r] * d8;
                    tmp[r][c] = __builtin_fmaf(d, (float) sumi, tmp[r][c]);
                }
            }
        } else {
            const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
            const int j  = bq8_offset/2;
            const int jm = j & 1;
            const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);
            int v0[R], v1[R]; uint32_t scm[R]; float2 dm[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const block_q4_K * bq4_K = (const block_q4_K *) vx + (size_t) (row0 + r)*bpr + kbx;
                const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
                v0[r] = q4[0]; v1[r] = q4[4];
                const uint16_t * scales = (const uint16_t *)bq4_K->scales;
                const uint32_t s0 = scales[jm + 0], s2 = scales[jm + 2], s4 = scales[jm + 4];
                const uint32_t a0 = (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi)) & 0xFFFFu;
                const uint32_t a1 = (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi)) & 0xFFFFu;
                scm[r] = a0 | (a1 << 16);   // bytes: sc[0], sc[1], m[0], m[1]
                dm[r] = __half22float2(bq4_K->dm);
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                float d8[QR4_K]; int u0[QR4_K], u1[QR4_K];
#pragma unroll
                for (int i = 0; i < QR4_K; ++i) {
                    const block_q8_1 * bq8i = y + (size_t) c*ys + kby + bq8_offset + i;
                    d8[i] = __low2float(bq8i->ds);
                    const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
                    u0[i] = q8[0]; u1[i] = q8[4];
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
                    for (int i = 0; i < QR4_K; ++i) {
                        const int v0i = (v0[r] >> (4*i)) & 0x0F0F0F0F;
                        const int v1i = (v1[r] >> (4*i)) & 0x0F0F0F0F;
                        const int dot1 = ggml_cuda_dp4a(v1i, u1[i], ggml_cuda_dp4a(v0i, u0[i], 0));
                        const int dot2 = ggml_cuda_dp4a(0x01010101, u1[i], ggml_cuda_dp4a(0x01010101, u0[i], 0));
                        const int sci = (int) ((scm[r] >> (8*i)) & 0xFFu), mi = (int) ((scm[r] >> (16 + 8*i)) & 0xFFu);
                        sumf_d = __builtin_fmaf(d8[i], (float) (dot1 * sci), sumf_d);
                        sumf_m = __builtin_fmaf(d8[i], (float) (dot2 * mi), sumf_m);
                    }
                    const float vv = dm[r].x*sumf_d - dm[r].y*sumf_m;
                    tmp[r][c] = tmp[r][c] + vv;
                }
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
template <ggml_type type, int NC>
static inline bool launch_sn_ncr_r(int R, const void * w, const block_q8_1 * y, int ys, float * dst, int dstr, int bpr, int N, hipStream_t st) {
    const dim3 b(64, 2);
    switch (R) {
        case 1: hipLaunchKernelGGL((gemv_sn_ncr<type, NC, 1>), dim3(N),   b, 0, st, w, y, ys, dst, dstr, bpr); return true;
        case 2: hipLaunchKernelGGL((gemv_sn_ncr<type, NC, 2>), dim3(N/2), b, 0, st, w, y, ys, dst, dstr, bpr); return true;
        case 4: hipLaunchKernelGGL((gemv_sn_ncr<type, NC, 4>), dim3(N/4), b, 0, st, w, y, ys, dst, dstr, bpr); return true;
        case 8: hipLaunchKernelGGL((gemv_sn_ncr<type, NC, 8>), dim3(N/8), b, 0, st, w, y, ys, dst, dstr, bpr); return true;
        default: return false;
    }
}
template <ggml_type type>
static inline bool launch_sn_ncr_t(int NC, int R, const void * w, const block_q8_1 * y, int ys, float * dst, int dstr, int bpr, int N, hipStream_t st) {
    switch (NC) {
#define IQ4K_SNR(n) case n: return launch_sn_ncr_r<type, n>(R, w, y, ys, dst, dstr, bpr, N, st);
        IQ4K_SNR(2) IQ4K_SNR(3) IQ4K_SNR(4) IQ4K_SNR(5) IQ4K_SNR(6) IQ4K_SNR(7) IQ4K_SNR(8) IQ4K_SNR(9)
        IQ4K_SNR(10) IQ4K_SNR(11) IQ4K_SNR(12) IQ4K_SNR(13) IQ4K_SNR(14) IQ4K_SNR(15) IQ4K_SNR(16)
#undef IQ4K_SNR
        default: return false;
    }
}
// R = 0: the default rows per workgroup for NC columns
// (measured, K = 5120, 1024 rows: IQ4_XS 2 rows from 5 columns, Q4_K 1 row: more rows leave too few waves)
static inline int sn_ncr_rows(ggml_type t, int NC) { return t == GGML_TYPE_IQ4_XS && NC > 4 ? 2 : 1; }
static inline bool launch_sn_ncr(ggml_type t, int NC, const void * w, const block_q8_1 * y, int ys, float * dst, int dstr, int K, int N,
        hipStream_t st, int R = 0) {
    if (K % QK_K != 0 || N <= 0 || (int64_t) N*(K/QK_K) >= (1ll << 31)) return false;
    if (R == 0) R = sn_ncr_rows(t, NC);
    if (N % R != 0) R = 1;
    if (t == GGML_TYPE_Q4_K)   return launch_sn_ncr_t<GGML_TYPE_Q4_K>(NC, R, w, y, ys, dst, dstr, K/QK_K, N, st);
    if (t == GGML_TYPE_IQ4_XS) return launch_sn_ncr_t<GGML_TYPE_IQ4_XS>(NC, R, w, y, ys, dst, dstr, K/QK_K, N, st);
    return false;
}

} // namespace iq4k
