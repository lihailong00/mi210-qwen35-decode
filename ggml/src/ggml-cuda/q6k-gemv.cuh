// q6k-gemv.cuh - single-token (ne11 == 1) GEMV for Q6_K on gfx90a (MI210): the MTP draft layer of the qwen35 model.
//
// Same scheme as iq4k-gemv.cuh (persistent workgroups, q8_1 activation staged once per workgroup in LDS, one 64-weight
// item per lane per step, ping-pong pipelined loads, round-robin or stream-K balanced units), with a lossless,
// same-size SoA copy of the weights:
//   plane A | plane B | plane C | S | D      (nb = rows * K/256 blocks)
//   A, B, C: nb*64 B each, one 16-byte chunk per item; item j = 2n + h (n, h in {0, 1}) of a block holds
//            A = ql[64n + 16h, +16), B = ql[64n + 32 + 16h, +16), C = qh[32n + 16h, +16)
//            i.e. the 64 weights y[128n + 32g + 16h + i], g = 0..3, i = 0..15 (g = 0, 1: low nibbles of A, B;
//            g = 2, 3: high nibbles of A, B; bits 2g, 2g+1 of C)
//   S:       nb*16 B, one 16-byte entry per block: word j = the four int8 scales of item j, sc[8n + h + 2g] in byte g
//   D:       nb*2 B, the fp16 block scale d, same entry index as S
// Chunk / entry placement follows the launch configuration exactly as for Q4_K (row interleave IL = (64/L)*R, the
// round-tiled layout, or the stream-K round-tiled layout).
// The activation record of an item (LDS): 16 words of q8 values (segment g = half h of q8 block 4n + g of the
// 256-block), 4 floats d8_g, 4 ints -32*sum(segment g), padded to YW words (bank spread).
// Numerics: per item and segment the integer dot product of (q6 - 32) with the q8 values is exact; the float part is
// d * sum_g d8_g * (sumi_g * sc_g) as in vec_dot_q6_K_q8_1, but the accumulation order over items differs from the stock
// kernel (deterministic: fixed lane mapping, fixed reduction tree). Used only for the MTP draft head.
#pragma once
#include "iq4k-gemv.cuh"

namespace q6k {

using iq4k::dp4a;
using iq4k::h2f;
using iq4k::silu_f;
using iq4k::group_sum;
using iq4k::skgeo;
using iq4k::sk_geo;
using iq4k::sk_chunk;
using iq4k::sk_chunk_of_step;
using iq4k::ld_u32_opaque;

static constexpr int IPB = 4;     // items per 256-weight block
#ifndef Q6K_YW
#define Q6K_YW 28
#endif
static constexpr int YW = Q6K_YW; // LDS words per item record of a single-column launch (24 used)
static constexpr int YWN = 24;    // ... of a multi-column launch (LDS size)
static __host__ __device__ constexpr int yw_of(const int ncols) { return ncols == 1 ? YW : YWN; }

struct soa { const uint8_t * A; const uint8_t * B; const uint8_t * C; const uint8_t * S; const uint8_t * D; };
static __host__ __device__ inline soa soa_make(const void * base, size_t nb) {
    soa s;
    s.A = (const uint8_t *) base; s.B = s.A + nb*64; s.C = s.B + nb*64; s.S = s.C + nb*64; s.D = s.S + nb*16;
    return s;
}

// ---------------------------------------------------------------- layouts
// mode 0: row-interleaved (IL rows, units of IL rows, all items of a unit contiguous)
// mode 1: round-tiled (round-robin units over NWV waves, step-major within a round)
// mode 2: stream-K round-tiled (the flat step stream cut into equal ranges, one per wave)
// q = items per row / L steps per unit. Returns the chunk index of item `it` of row `row`, and (for it % 4 == 0) the
// header entry index of its block.
struct lay { int mode, IL, L, NWV, q, nunits, bpr; skgeo geo; };
static __host__ __device__ __forceinline__ void lay_index(const lay & g, const int64_t row, const int it, size_t & qc, size_t & se) {
    const int64_t u = row / g.IL; const int ri = (int) (row % g.IL);
    const int blk = it / IPB;
    if (g.mode == 0) {
        qc = ((size_t) u*g.bpr*IPB + it)*g.IL + ri;
        se = ((size_t) u*g.bpr + blk)*g.IL + ri;
        return;
    }
    const int kk = it / g.L, l = it % g.L;
    size_t ch;
    if (g.mode == 1) {
        ch = iq4k::rt_chunk((size_t) u, kk, g.q, g.NWV, g.nunits);
    } else {
        ch = sk_chunk_of_step(u*g.q + kk, g.geo);
    }
    qc = ch*g.L*g.IL + (size_t) l*g.IL + ri;
    se = ch*(g.L/IPB)*g.IL + (size_t) (l/IPB)*g.IL + ri;
}

// one thread per block; src = block_q6_K (210 B, 2-byte aligned) rows of sbpr blocks, the copy takes the g.bpr blocks
// starting at block b0 of every row (a K range of the matrix; b0 = 0, sbpr = g.bpr: the whole matrix)
static __global__ void repack(const uint8_t * __restrict__ src, uint8_t * __restrict__ dst, const int64_t nb, const lay g,
        const int sbpr, const int b0) {
    const int64_t i = blockIdx.x*(int64_t) blockDim.x + threadIdx.x;
    if (i >= nb) return;
    const int64_t row = i / g.bpr; const int blk = (int) (i % g.bpr);
    const uint8_t * b = src + (row*sbpr + b0 + blk)*210;   // ql[128] qh[64] scales[16] d
    uint8_t * A = dst, * B = dst + nb*64, * C = dst + nb*128, * S = dst + nb*192, * D = dst + nb*208;
    size_t qc, se = 0;
    for (int j = 0; j < IPB; ++j) {
        size_t qj, sj;
        lay_index(g, row, IPB*blk + j, qj, sj);
        if (j == 0) se = sj;
        qc = qj;
        const int n = j >> 1, h = j & 1;
        for (int k = 0; k < 16; ++k) {
            A[qc*16 + k] = b[64*n + 16*h + k];
            B[qc*16 + k] = b[64*n + 32 + 16*h + k];
            C[qc*16 + k] = b[128 + 32*n + 16*h + k];
        }
        for (int gg = 0; gg < 4; ++gg) {
            S[se*16 + 4*j + gg] = b[192 + 8*n + h + 2*gg];
        }
    }
    D[se*2 + 0] = b[208];
    D[se*2 + 1] = b[209];
}

// ---------------------------------------------------------------- per-item data and math
struct raw { uint4 a, b, c; uint32_t s, d; };
static __device__ __forceinline__ raw load(const soa & s, const uint32_t qc, const uint32_t se, const uint32_t j) {
    raw r;
    r.a = *(const uint4 *) (s.A + qc*16u);
    r.b = *(const uint4 *) (s.B + qc*16u);
    r.c = *(const uint4 *) (s.C + qc*16u);
    r.s = *(const uint32_t *) (s.S + se*16u + j*4u);
    r.d = *(const unsigned short *) (s.D + se*2u);
    return r;
}

// q8 block b (32 values) -> half 0 to item record 2*(b/4)... : 256-block k = b/8, n = (b/4)&1, g = b&3
template <int YWT>
static __device__ __forceinline__ void put_y(uint32_t * ys, const int b, const uint4 q0, const uint4 q1, const float d) {
    const int it0 = 4*(b >> 3) + 2*((b >> 2) & 1), g = b & 3;
    uint32_t * r0 = ys + it0*YWT, * r1 = r0 + YWT;
    *(uint4 *) (r0 + 4*g) = q0;
    *(uint4 *) (r1 + 4*g) = q1;
    int s0 = dp4a((int) q0.x, 0x01010101, 0); s0 = dp4a((int) q0.y, 0x01010101, s0); s0 = dp4a((int) q0.z, 0x01010101, s0); s0 = dp4a((int) q0.w, 0x01010101, s0);
    int s1 = dp4a((int) q1.x, 0x01010101, 0); s1 = dp4a((int) q1.y, 0x01010101, s1); s1 = dp4a((int) q1.z, 0x01010101, s1); s1 = dp4a((int) q1.w, 0x01010101, s1);
    r0[16 + g] = __float_as_uint(d); r1[16 + g] = __float_as_uint(d);
    r0[20 + g] = (uint32_t) (-32*s0); r1[20 + g] = (uint32_t) (-32*s1);
}

// column c of the activation at yg + c*ys (ys = q8_1 blocks per column), records of column c at ysl + c*(K/64)*yw_of(NC)
template <int NC>
static __device__ __forceinline__ void stage_y(uint32_t * __restrict__ ysl, const block_q8_1 * __restrict__ yg, const int K,
        const int ys, const int tid, const int nt) {
    for (int i = tid; i < NC*(K/32); i += nt) {
        const int c = NC == 1 ? 0 : i / (K/32), b = NC == 1 ? i : i - c*(K/32);
        const uint8_t * p = (const uint8_t *) (yg + (size_t) c*ys + b);
        const uint32_t ds = ld_u32_opaque(p);
        const uint4 q0 = *(const uint4 *) (p + 4);
        const uint4 q1 = *(const uint4 *) (p + 20);
        put_y<yw_of(NC)>(ysl + c*(K/64)*yw_of(NC), b, q0, q1, h2f(ds));
    }
}

static __device__ __forceinline__ uint32_t hbits(const uint32_t c, const int g) {
    // bits 2g, 2g+1 of every byte of c, moved to bits 4, 5
    return g == 0 ? (c << 4) & 0x30303030u : g == 1 ? (c << 2) & 0x30303030u : g == 2 ? c & 0x30303030u : (c >> 2) & 0x30303030u;
}

// the 64 unsigned 6-bit codes of an item: word 4g + w = weights 4w..4w+3 of segment g
static __device__ __forceinline__ void decode(const raw & r, uint32_t (&Q)[16]) {
    const uint32_t A[4] = {r.a.x, r.a.y, r.a.z, r.a.w};
    const uint32_t B[4] = {r.b.x, r.b.y, r.b.z, r.b.w};
    const uint32_t C[4] = {r.c.x, r.c.y, r.c.z, r.c.w};
#pragma unroll
    for (int g = 0; g < 4; ++g) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t src = (g & 1) ? B[w] : A[w];
            const uint32_t lo  = (g >> 1) ? (src >> 4) & 0x0F0F0F0Fu : src & 0x0F0F0F0Fu;
            Q[4*g + w] = lo | hbits(C[w], g);
        }
    }
}
// sum_g d8_g * (sumi_g * sc_g) of one column (record yr); the caller multiplies by d
static __device__ __forceinline__ float dotcol(const uint32_t (&Q)[16], const int (&sc)[4], const uint32_t * yr) {
    const uint4 dd = *(const uint4 *) (yr + 16);
    const uint4 cc = *(const uint4 *) (yr + 20);
    const float d8[4] = {__uint_as_float(dd.x), __uint_as_float(dd.y), __uint_as_float(dd.z), __uint_as_float(dd.w)};
    const int   c0[4] = {(int) cc.x, (int) cc.y, (int) cc.z, (int) cc.w};
    float sumf = 0.0f;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        const uint4 y = *(const uint4 *) (yr + 4*g);
        int s = c0[g];
        s = dp4a((int) Q[4*g + 0], (int) y.x, s);
        s = dp4a((int) Q[4*g + 1], (int) y.y, s);
        s = dp4a((int) Q[4*g + 2], (int) y.z, s);
        s = dp4a((int) Q[4*g + 3], (int) y.w, s);
        sumf += d8[g]*(float) (s*sc[g]);
    }
    return sumf;
}
template <int NC>
static __device__ __forceinline__ void compute(const raw & r, const uint32_t * yr, const int cstride, float (&acc)[NC]) {
    uint32_t Q[16];
    decode(r, Q);
    int sc[4];
#pragma unroll
    for (int g = 0; g < 4; ++g) sc[g] = (int) (int8_t) ((r.s >> (8*g)) & 0xFFu);
    const float d = h2f(r.d);
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] += d*dotcol(Q, sc, yr + c*cstride);
}

// ---------------------------------------------------------------- the GEMV kernel
// Structure and invariants as iq4k::gemv with EX (exact: items per row % L == 0): wave w of the grid owns either the
// round-robin units w, w + nwaves, ... (SK = false; RT: round-tiled layout, else row-interleaved) or the contiguous
// range [b, e) of the flat step stream (SK = true, round-tiled stream-K layout; split units combined by two waves
// through scratch/cnt in fixed order).
// GLU: dst = (W_up . y) * silu(W_gate . y) (s = up, sg = gate); residual (optional): dst = W . y + residual.
// NC columns (NC > 1: no GLU): column c of y at yg + c*ys blocks, of dst / residual at + c*ds floats; the weights are
// read once for all columns (each column has its own LDS records, the codes are decoded once per item).
template <int L, int R, bool GLU, bool SK, bool RT, int NC = 1>
__launch_bounds__(1024, 1)
__global__ void gemv(const soa s, const soa sg, const block_q8_1 * __restrict__ yg, float * __restrict__ dst,
        const int K, const int N, const float * __restrict__ residual, float * __restrict__ scratch, unsigned int * __restrict__ cnt,
        const int ys, const int ds) {
    static_assert(NC == 1 || !GLU, "GLU: one column");
    extern __shared__ __align__(16) uint32_t ysm[];
    constexpr int G = 64/L, GR = G*R, NV = (GLU ? 2 : 1)*NC;   // NV result vectors per unit: [nv][GR]
    const int tid  = threadIdx.y*64 + threadIdx.x;
    const int lane = threadIdx.x % L;
    const int grp  = threadIdx.x / L;
    const uint32_t bpr = K / QK_K;
    const int ipr    = bpr*IPB;
    const int q      = ipr / L;
    const int nunits = N / GR;
    const int wy     = __builtin_amdgcn_readfirstlane(threadIdx.y);
    const int nwaves = gridDim.x*blockDim.y;
    const int w      = wy*gridDim.x + blockIdx.x;
    const int64_t W  = (int64_t) nunits*q;
    const int myu    = SK ? 0 : (w < nunits ? (int) ((uint32_t) (nunits - 1 - w)/(uint32_t) nwaves) + 1 : 0);
    const skgeo geo  = sk_geo(W, nwaves);
    const int b      = SK ? (w < geo.nact ? w*geo.T : (int) W) : 0;
    const int e      = SK ? (w < geo.nact - 1 ? b + geo.T : (int) W) : myu*q;
    const int maxu   = SK ? (int) (((W + nwaves - 1)/nwaves + q - 1)/q) + 1 : myu + (myu*nwaves < nunits ? 1 : 0);
    const int u0     = b / q;
    constexpr int YWK = yw_of(NC);
    const int cst    = (K/64)*YWK;  // LDS words per column
    auto gunit = [&](const int cu) -> int { return SK ? cu : w + cu*nwaves; };
    float * ow = (float *) (ysm + NC*cst) + wy*maxu*NV*GR;
    const uint32_t jl = lane & 3;

    raw A[R], B[R];
    raw Ag[GLU ? R : 1], Bg[GLU ? R : 1];

    const int ue = SK ? (e - 1)/q : myu - 1, ke = SK ? (e - 1) % q : q - 1;
    int ls = b, lu = u0, lk = SK ? b % q : 0;
    uint32_t qlane[R], slane[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { qlane[r] = lane*GR + grp*R + r; slane[r] = (lane/IPB)*GR + grp*R + r; }
    auto load_step = [&](raw (&bb)[R], raw (&bg)[GLU ? R : 1]) {
        const bool past = ls >= e;
        const int kr = past ? ue : lu, kk = past ? ke : lk;
        uint32_t qs_, ss_;
        if constexpr (SK) {
            const int j = (past ? e - 1 : ls) - b;
            const uint32_t ch = (uint32_t) sk_chunk(w, j, geo);
            qs_ = ch*(L*GR); ss_ = ch*((L/IPB)*GR);
        } else if constexpr (RT) {
            const int nur = min(nwaves, nunits - kr*nwaves);
            const uint32_t ch = (uint32_t) (kr*nwaves*q + kk*nur + w);
            qs_ = ch*(L*GR); ss_ = ch*((L/IPB)*GR);
        } else {
            const uint32_t un = (uint32_t) gunit(kr);
            qs_ = (un*bpr*IPB + kk*L)*GR; ss_ = (un*bpr + kk*(L/IPB))*GR;
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint32_t qo = qs_ + qlane[r], so = ss_ + slane[r];
            bb[r] = load(s, qo, so, jl);
            if constexpr (GLU) bg[r] = load(sg, qo, so, jl);
        }
        ++ls; if (++lk == q) { lk = 0; ++lu; }
    };

    if (e > b) { load_step(A, Ag); }
    stage_y<NC>(ysm, yg, K, ys, tid, blockDim.y*64);
    __syncthreads();
    if (e <= b) return;

    float acc[R][NC], accg[GLU ? R : 1][1];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int c = 0; c < NC; ++c) acc[r][c] = 0.0f;
        if constexpr (GLU) accg[r][0] = 0.0f;
    }
    int cs = b, cu = u0, ck = b % q;
    auto compute_step = [&](raw (&bb)[R], raw (&bg)[GLU ? R : 1]) {
        const uint32_t * yr = ysm + lane*YWK + ck*(L*YWK);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            compute<NC>(bb[r], yr, cst, acc[r]);
            if constexpr (GLU) compute<1>(bg[r], yr, cst, accg[r]);
        }
        if (ck == q - 1 || cs == e - 1) {
            if (cs < e) {
                float * o = ow + (cu - u0)*NV*GR + grp*R;
#pragma unroll
                for (int r = 0; r < R; ++r) {
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        const float v = group_sum<L>(acc[r][c]);
                        if (lane == 0) o[c*GR + r] = v;
                    }
                    if constexpr (GLU) { const float g = group_sum<L>(accg[r][0]); if (lane == 0) o[GR + r] = g; }
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
#pragma unroll
                for (int c = 0; c < NC; ++c) acc[r][c] = 0.0f;
                if constexpr (GLU) accg[r][0] = 0.0f;
            }
        }
        ++cs; if (++ck == q) { ck = 0; ++cu; }
    };

    const int nst = e - b;
    for (int st = 0; st < nst; st += 2) {
        load_step(B, Bg);
        compute_step(A, Ag);
        load_step(A, Ag);
        compute_step(B, Bg);
    }

    // epilogue: element (column c, row rr of unit u); GLU: v = up, vg = gate
    auto out = [&](const int u, const int rr, const int c, float v, const float vg) {
        if constexpr (GLU) v *= silu_f(vg);
        if (residual) v += residual[(size_t) c*ds + u*GR + rr];
        dst[(size_t) c*ds + u*GR + rr] = v;
    };
    const bool hs = SK && (b % q) != 0, ts = SK && (e % q) != 0;
    const int c0 = u0 + (hs ? 1 : 0), c1 = ue - (ts ? 1 : 0);
    for (int i = threadIdx.x; i < (c1 - c0 + 1)*GR*NC; i += 64) {
        const int c = NC == 1 ? 0 : i % NC, ii = NC == 1 ? i : i / NC;
        const int j = ii / GR, rr = ii % GR, cu2 = c0 + j, u = gunit(cu2);
        const float * o = ow + (cu2 - u0)*NV*GR;
        out(u, rr, c, o[c*GR + rr], GLU ? o[GR + rr] : 0.0f);
    }
    auto combine = [&](const int p, const int role, const int u) {
        float * sc = scratch + (size_t) p*2*NV*GR;
        const float * o = ow + (u - u0)*NV*GR;
        for (int i = threadIdx.x; i < NV*GR; i += 64) {
            __hip_atomic_store(sc + role*NV*GR + i, o[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __threadfence();
        unsigned int old = 0;
        if (threadIdx.x == 0) old = __hip_atomic_fetch_add(cnt + p, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        old = __shfl(old, 0, 64);
        if (old == 1) {
            __threadfence();
            auto tot = [&](const int i) {
                const float mine  = o[i];
                const float other = __hip_atomic_load(sc + (1 - role)*NV*GR + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                return role == 0 ? mine + other : other + mine;   // first part + second part
            };
            for (int i = threadIdx.x; i < NC*GR; i += 64) {
                const int c = i / GR, rr = i % GR;
                out(u, rr, c, tot(c*GR + rr), GLU ? tot(GR + rr) : 0.0f);
            }
            if (threadIdx.x == 0) __hip_atomic_store(cnt + p, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    };
    if constexpr (SK) {
        if (hs) combine(w, 1, u0);
        if (ts) combine(w + 1, 0, ue);
    }
}

// ================================================================================================
// Host side
struct cfg { int L, R, NW, grid; bool sk, rt; };

static inline lay make_lay(const cfg & c, int N, int K) {
    lay g;
    g.mode = c.sk ? 2 : (c.rt ? 1 : 0);
    g.IL = (64/c.L)*c.R; g.L = c.L; g.NWV = c.grid*c.NW; g.bpr = K/QK_K;
    g.q = (g.bpr*IPB)/c.L; g.nunits = N/g.IL;
    g.geo = sk_geo((int64_t) g.nunits*g.q, g.NWV);
    return g;
}

// nv = result vectors per unit (GLU: 2, else the number of columns)
static inline size_t lds_bytes(const cfg & c, int K, int N, int nv, int ncols = 1) {
    const int GR = (64/c.L)*c.R, q = (K/256)*IPB/c.L, nunits = N/GR, nwaves = c.grid*c.NW;
    const int64_t W = (int64_t) nunits*q;
    const int maxu = c.sk ? (int) (((W + nwaves - 1)/nwaves + q - 1)/q) + 1 : (nunits + nwaves - 1)/nwaves;
    return (size_t) ncols*(K/64)*yw_of(ncols)*4 + (size_t) c.NW*maxu*nv*GR*4;
}

static constexpr size_t SCRATCH_FLOATS = iq4k::SCRATCH_FLOATS;
static constexpr size_t CNT_UINTS      = iq4k::CNT_UINTS;
static constexpr int    NC_MAX         = 4;

static inline bool shape_ok(const cfg & c, int K, int N, bool glu, int ncols = 1) {
    const int GR = (64/c.L)*c.R;
    const int ipr = (K/256)*IPB;
    const int nv = glu ? 2 : ncols;
    return K % 256 == 0 && ipr % c.L == 0 && N % GR == 0 && N/GR >= 1 && ncols >= 1 && ncols <= NC_MAX && (!glu || ncols == 1) &&
        (ncols == 1 || (c.L == 16 && c.R == 1)) &&
        (!c.sk || N/GR >= c.grid*c.NW) && c.grid*c.NW + 1 <= (int) CNT_UINTS &&
        (size_t) (c.grid*c.NW + 1)*2*nv*GR <= SCRATCH_FLOATS && (size_t) N*(K/256)*210 < (1ull << 32) &&
        lds_bytes(c, K, N, nv, ncols) <= 65536;
}

// tuned on MI210 (q6k_bench, real blk.64 weights)
static inline cfg pick_cfg(int N, int K, int ncu, bool glu) {
    (void) N; (void) glu;
    if (K > 8192) return cfg{16, 1, 10, ncu, false, true};
    return cfg{16, 1, 12, ncu, false, false};
}

template <int L, int R, bool GLU, int NC>
static inline void launch_k(const cfg & c, const soa & s, const soa & sg, const block_q8_1 * y, float * dst, int K, int N,
        hipStream_t st, const float * residual, float * scratch, unsigned int * cnt, int ys, int ds) {
    const size_t lds = lds_bytes(c, K, N, GLU ? 2 : NC, NC);
    if (c.sk)      hipLaunchKernelGGL((gemv<L, R, GLU, true,  true,  NC>), dim3(c.grid), dim3(64, c.NW), lds, st, s, sg, y, dst, K, N, residual, scratch, cnt, ys, ds);
    else if (c.rt) hipLaunchKernelGGL((gemv<L, R, GLU, false, true,  NC>), dim3(c.grid), dim3(64, c.NW), lds, st, s, sg, y, dst, K, N, residual, scratch, cnt, ys, ds);
    else           hipLaunchKernelGGL((gemv<L, R, GLU, false, false, NC>), dim3(c.grid), dim3(64, c.NW), lds, st, s, sg, y, dst, K, N, residual, scratch, cnt, ys, ds);
}

// ncols columns (GLU: 1): y column c at y + c*ys blocks, dst / residual column c at + c*ds floats
template <bool GLU>
static inline bool dispatch(const cfg & c, const soa & s, const soa & sg, const block_q8_1 * y, float * dst, int K, int N,
        hipStream_t st, const float * residual, float * scratch, unsigned int * cnt, int ncols = 1, int ys = 0, int ds = 0) {
#define Q6K_TRY(L_, R_, NC_) if (c.L == L_ && c.R == R_ && ncols == NC_) { launch_k<L_, R_, GLU, NC_>(c, s, sg, y, dst, K, N, st, residual, scratch, cnt, ys, ds); return true; }
    Q6K_TRY(16, 1, 1) Q6K_TRY(32, 1, 1) Q6K_TRY(16, 2, 1)
    if constexpr (!GLU) {
        Q6K_TRY(16, 1, 2) Q6K_TRY(16, 1, 3) Q6K_TRY(16, 1, 4)
    }
#undef Q6K_TRY
    return false;
}

// copy of the K range [k0, k0 + K) of a [nrows, Ktot] Q6_K matrix for launch configuration c
static inline bool repack_cfg(const void * src, void * dst, int64_t nrows, int K, const cfg & c, hipStream_t st, int Ktot = 0, int k0 = 0) {
    if (Ktot == 0) Ktot = K;
    const lay g = make_lay(c, (int) nrows, K);
    const int64_t nb = nrows*(K/QK_K);
    const int nthr = 256; const int nblk = (int) ((nb + nthr - 1)/nthr);
    hipLaunchKernelGGL(repack, dim3(nblk), dim3(nthr), 0, st, (const uint8_t *) src, (uint8_t *) dst, nb, g, Ktot/QK_K, k0/QK_K);
    return true;
}

} // namespace q6k
