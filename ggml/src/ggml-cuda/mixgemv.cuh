// mixgemv.cuh - one launch for a single-token GEMV pair that reads the same activation but mixes module types
// (r4-mixgroup): one IQ3_S / IQ3_XXS job (iq3gemv step-major SoA copy, LDS codebook) + one IQ4_XS / Q4_K job (iq4k
// row-interleaved IL = 4 copy, the same copy as the single-matrix launcher at K <= 8192).
// Linear-attention qkv + z and FFN gate + up pairs of mixed types (gfx90a / MI210).
//
// Structure ("flowing" pair): persistent workgroups of 16 waves (grid = number of CUs). One prologue stages both the
// IQ3 codebook + activation records and the IQ4 activation records (from the same q8_1 loads). Then every wave runs
// its units of the first job and continues with its units of the second job without any barrier, so waves that run
// out of first-job work start on the second job right away (no launch boundary, no second prologue, no per-workgroup
// tail between the two GEMVs). The second job's round robin is rotated by (units of the first job) % (waves), so the
// waves that had one unit less in the first job get the extra units of the second one.
// Per unit, both parts run exactly the code of the single kernels (iq3g::gemv PF 5 / iq3g::mj_unit, iq4k::gemv
// EX round robin): outputs are bit-identical to the two separate launches.
#pragma once
#include "iq3gemv.cuh"
#include "iq4k-gemv.cuh"

namespace mixg {

static constexpr int NW = 16;   // waves per workgroup
static constexpr int L  = 16;   // lanes per row (both parts)
static constexpr int RG = 4;    // rows per unit (= iq3g repack interleave = iq4k IL)

// IQ3 part: units u = w, w + nwv, ... of the IQ3 job (same per-unit code as iq3g::gemv<T3, none_t, 2, 16, 1, 16, 5, 0, 1, SM>)
template <class T3, int SM>
static __device__ __forceinline__ void part3(const uint8_t * __restrict__ w3, float * __restrict__ d3, const int N3, const int K,
        const int w, const int nwv, const int lane, const int grp) {
    constexpr int P = 2;
    constexpr int YOFF3 = T3::TAB_WORDS*4;
    const int items = K/128;
    const int ipr   = items*(4/P);
    const uint32_t nh = (uint32_t) N3*(uint32_t) items;
    const int U3 = N3/RG;
    typename T3::template ld<P> nopf{};
    for (int u = w; u < U3; u += nwv) {
        const float acc = iq3g::mj_unit<T3, 0, P, L, RG, SM>(w3, nh, u, items, lane, grp, 0, ipr, YOFF3, false, nopf, U3);
        const float v = iq3g::group_sum<L>(acc);
        if (lane == 0) d3[u*RG + grp] = v;
    }
}

// IQ4 part: round-robin units w, w + nwv, ... of the IQ4 job (same code as iq4k::gemv<T4, 16, 1, false, false, false, true>:
// ping-pong pipelined steps, unit results through this wave's LDS slots, written after the loop)
template <class T4>
static __device__ __forceinline__ void part4(const iq4k::soa s4, float * __restrict__ d4, const int N4, const int K,
        const uint32_t * ys4, float * ow, const int w, const int nwv, const int lane, const int grp) {
    constexpr int GR = RG;
    const uint32_t bpr = K / QK_K;
    const int ipr = bpr*T4::IPB;
    const int q   = ipr / L;                       // steps per unit (host: ipr % L == 0)
    const int nunits = N4 / GR;
    const int myu = w < nunits ? (int) ((uint32_t) (nunits - 1 - w)/(uint32_t) nwv) + 1 : 0;   // operands >= 0
    if (myu == 0) {
        return;
    }
    const int e = myu*q;
    const typename T4::lc c = T4::lane_consts(lane);
    typedef typename T4::raw raw;
    raw A, B;
    const int ue = myu - 1, ke = q - 1;           // last real step (= (e - 1)/q, (e - 1) % q with e = myu*q, myu >= 1)
    int ls = 0, lu = 0, lk = 0;                   // load cursor
    const uint32_t qlane = lane*GR + grp, slane = (lane/T4::IPB)*GR + grp;
    const uint32_t boff = 4*bpr*GR;
    auto load_step = [&](raw & bb) {
        const bool past = ls >= e;
        const int kr = past ? ue : lu, kk = past ? ke : lk;
        const uint32_t un = (uint32_t) (w + kr*nwv);
        const uint32_t qs_ = (un*bpr*8 + kk*L)*GR, ss_ = (un*bpr + kk*(L/T4::IPB))*GR;
        bb = T4::load_c(s4, qs_ + qlane, ss_ + slane, boff);
        ++ls; if (++lk == q) { lk = 0; ++lu; }
    };
    load_step(A);
    float acc = 0.0f;
    int cs = 0, cu = 0, ck = 0;                   // compute cursor
    auto compute_step = [&](raw & bb) {
        const uint32_t * yr = ys4 + lane*T4::YW + ck*(L*T4::YW);
        acc += T4::compute(bb, yr, c);
        if (ck == q - 1 || cs == e - 1) {         // end of a unit or of the range (uniform)
            if (cs < e) {
                const float v = iq4k::group_sum<L>(acc);
                if (lane == 0) ow[cu*GR + grp] = v;
            }
            acc = 0.0f;
        }
        ++cs; if (++ck == q) { ck = 0; ++cu; }
    };
    for (int st = 0; st < e; st += 2) {
        load_step(B);       // step st+1 (clamped past the end)
        compute_step(A);    // step st
        load_step(A);       // step st+2
        compute_step(B);    // step st+1
    }
    for (int i = threadIdx.x; i < myu*GR; i += 64) {
        const int j = i / GR, rr = i % GR, u = w + j*nwv;
        d4[u*GR + rr] = ow[j*GR + rr];
    }
}

// LDS: [IQ3 table | IQ3 activation records] (iq3g layout, table at address 0) | IQ4 activation records (y4off) |
// per-wave IQ4 unit results [NW][maxu4][RG]
template <class T3, class T4>
static inline size_t y4_offset(const int K) {
    return (iq3g::lds_bytes<T3, iq3g::none_t>(K) + 15) & ~(size_t) 15;
}
// calibration probe (GGML_MMVQ_MIX_EMPTY=n): n empty launches after every pair, to measure what a launch costs in graph replay
static __global__ void empty_kernel() {}

// ---- fused SwiGLU epilogue for gate/up pairs (equal N): dst = silu(gate) * up and its q8_1 blocks (+ int sums) for the
// following ffn_down GEMV, with exactly the arithmetic of the swiglu_q8 producer (q8-producers.cu: gated_q8_kernel<0>).
// Per 32-row block a counter collects the 16 unit results (8 units of 4 rows per job); after all of its work every wave
// publishes its units (release fence + one atomic per unit, in parallel over lanes) and the wave that completes a block
// computes it. The q8 blocks go to the per-stream activation cache, which also holds this kernel's own input: they are
// only written after every workgroup has staged the input (pro counter; all workgroups are co-resident, grid = #CUs).
// The last block resets the counters (graph replay safe; zero-initialized once).
struct glu_args {
    const float  * g;      // gate results (one of d3 / d4)
    const float  * u;      // up results
    float        * dst;    // GLU output (f32)
    block_q8_1   * y;      // q8_1 of the GLU output
    int          * ysum;   // int sums per q8_1 block
    unsigned int * cnt;    // [N/32] arrivals per block, then [N/32] = done blocks, [N/32 + 1] = staged workgroups
    int            nb;     // N/32
};
// (int8_t) roundf(v / d) without the IEEE division in the common case (bit-exact, = q8p_round_div)
static __device__ __forceinline__ int q8_round_div(const float v, const float d, const float inv_d) {
    const float t  = v * inv_d;
    const float fr = __builtin_amdgcn_fractf(fabsf(t));
    if (fabsf(fr - 0.5f) > 3.0e-5f) {
        return (int) rintf(t);
    }
    return (int) roundf(v / d);
}
// a3 / a4: this wave's first IQ3 / IQ4 unit (round robin step nwv)
static __device__ __forceinline__ void glu_epilogue(const glu_args & ga, const int a3, const int U3, const int a4, const int U4,
        const int nwv) {
    const int n3 = a3 < U3 ? (U3 - 1 - a3)/nwv + 1 : 0;
    const int n4 = a4 < U4 ? (U4 - 1 - a4)/nwv + 1 : 0;
    if (n3 + n4 == 0) {
        return;
    }
    __threadfence();   // release: this wave's d3 / d4 stores
    const int l = threadIdx.x;
    int b = -1;
    if (l < n3) {
        b = (a3 + l*nwv) / 8;
    } else if (l < n3 + n4) {
        b = (a4 + (l - n3)*nwv) / 8;
    }
    bool done = false;
    if (b >= 0) {
        done = atomicAdd(ga.cnt + b, 1u) == 15u;
    }
    unsigned long long m = __ballot(done);
    if (m == 0) {
        return;
    }
    // the activation cache buffer is also this kernel's input: wait until every workgroup has staged it (normally long done)
    while (__hip_atomic_load(ga.cnt + ga.nb + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < gridDim.x) {
        __builtin_amdgcn_s_sleep(1);
    }
    __threadfence();   // acquire: the other waves' results of the completed blocks
    const int half = threadIdx.x >> 5, li = threadIdx.x & 31;
    while (m) {
        const int l0 = __ffsll((long long) m) - 1; m &= m - 1;
        int l1 = -1;
        if (m) { l1 = __ffsll((long long) m) - 1; m &= m - 1; }
        const int b0 = __shfl(b, l0, 64);
        const int b1 = __shfl(b, l1 < 0 ? 0 : l1, 64);
        const int bb = half ? (l1 < 0 ? -1 : b1) : b0;
        if (bb >= 0) {
            const int i = bb*32 + li;
            const float gv = ga.g[i];
            const float r  = (gv / (1.0f + expf(-gv))) * ga.u[i];   // ggml_cuda_op_silu_single(gate) * up
            ga.dst[i] = r;
            float amax = fabsf(r);
            amax = warp_reduce_max<32>(amax);
            const float  d = amax / 127.0f;
            const int8_t q = amax == 0.0f ? 0 : q8_round_div(r, d, 1.0f / d);
            const int sum = warp_reduce_sum<32>((int) q);
            ga.y[bb].qs[li] = q;
            if (li == 0) {
                ga.y[bb].ds = make_half2(d, d * sum);
                ga.ysum[bb] = sum;
                ga.cnt[bb] = 0u;
                if (atomicAdd(ga.cnt + ga.nb, 1u) == (unsigned int) ga.nb - 1u) {   // last block of the launch: reset
                    ga.cnt[ga.nb] = 0u;
                    ga.cnt[ga.nb + 1] = 0u;
                }
            }
        }
    }
}

// nwv4 = waves that run IQ4 units (all waves in the flowing mode, (NW - na)*grid in the split mode)
template <class T3, class T4>
static inline size_t lds_bytes(const int K, const int N4, const int nwv4) {
    const int nunits = N4/RG, maxu = (nunits + nwv4 - 1)/nwv4;
    return y4_offset<T3, T4>(K) + iq4k::lds_bytes<T4>(K) + (size_t) NW*maxu*RG*4;
}

// na == 0 ("flowing"): every wave runs its units of both jobs; FIRST3: IQ3 job first (then the IQ4 job), else the IQ4 job
// first; rot = (units of the first job) % (waves).
// na > 0 ("split"): waves 0..na-1 of every workgroup run only IQ3 units, waves na..NW-1 only IQ4 units (both jobs run
// concurrently on every SIMD: the IQ3 decode is VALU/LDS bound, the IQ4 stream mostly waits for memory).
template <class T3, class T4, int SM, bool FIRST3, bool ORD = false>   // ORD: iq3g::pro_load load order (grid words first)
__launch_bounds__(NW*64, 1)
__global__ void gemv_pair(const uint8_t * __restrict__ w3, float * __restrict__ d3, const int N3,
        const iq4k::soa s4, float * __restrict__ d4, const int N4,
        const block_q8_1 * __restrict__ yg, const int K, const int y4off, const int rot, const int na, const int maxu4, const glu_args ga) {
    extern __shared__ __align__(16) uint32_t smem[];
#if defined(__HIP_DEVICE_COMPILE__)
    if ((uint32_t) (size_t) (__attribute__((address_space(3))) uint32_t *) smem != 0) __builtin_trap();
#endif
    const int tid  = threadIdx.y*64 + threadIdx.x;
    const int lane = threadIdx.x % L, grp = threadIdx.x / L;
#if defined(__HIP_DEVICE_COMPILE__)
    const int wy = __builtin_amdgcn_readfirstlane(threadIdx.y);
#else
    const int wy = threadIdx.y;
#endif
    const int nwv = gridDim.x*NW;
    const int w   = wy*gridDim.x + blockIdx.x;   // global wave index (same order as the single kernels)
    uint32_t * ys4 = smem + y4off/4;

    // prologue: owner threads load their q8_1 block once and write it in both record layouts; IQ3 codebook table
    {
        iq3g::pro_regs<T3, iq3g::none_t, false> pr;
        iq3g::pro_load<T3, iq3g::none_t, false, NW*64, true, ORD>(pr, yg, K, tid);
        iq3g::pro_store<T3, iq3g::none_t, false, NW*64>(pr, smem, 0, T3::TAB_WORDS*4, K, tid);
        if (tid < K/32) {
            T4::put_y(ys4, tid, pr.q0, pr.q1, iq4k::h2f(pr.ds));
        }
    }
    __syncthreads();
    if (ga.dst && tid == 0) {
        atomicAdd(ga.cnt + ga.nb + 1, 1u);   // this workgroup no longer reads the activation (see glu_epilogue)
    }

    const int nwv4 = na > 0 ? (NW - na)*gridDim.x : nwv;
    // maxu4 = (N4/RG + nwv4 - 1)/nwv4 (IQ4 units per wave = the per-wave LDS slot stride) comes from the host (no division here)
    float * ow = (float *) (ys4 + (K/32/T4::YPI)*T4::YW) + wy*maxu4*RG;
    if (na > 0) {
        if (wy < na) {
            part3<T3, SM>(w3, d3, N3, K, wy*gridDim.x + blockIdx.x, na*gridDim.x, lane, grp);
        } else {
            part4<T4>(s4, d4, N4, K, ys4, ow, (wy - na)*gridDim.x + blockIdx.x, nwv4, lane, grp);
        }
        return;
    }
    const int w2 = w >= rot ? w - rot : w + nwv - rot;   // = (w + nwv - rot) % nwv for 0 <= w, rot < nwv (host: rot = x % nwv)
    if (na < 0) {
        // alternating: the wave's IQ3 units (w, w + nwv, ..) and IQ4 units (w2, w2 + nwv, ..) one after the other, so that
        // compute-heavy IQ3 units and memory-bound IQ4 units are mixed on every SIMD all the time
        const int U3 = N3/RG, U4 = N4/RG, big = 1 << 30;
        int u3 = w, u4 = w2;
        while (u3 < U3 || u4 < U4) {
            if (u3 < U3) { part3<T3, SM>(w3, d3, N3, K, u3, big, lane, grp); u3 += nwv; }
            if (u4 < U4) { part4<T4>(s4, d4, N4, K, ys4, ow, u4, big, lane, grp); u4 += nwv; }
        }
        if (ga.dst) {
            glu_epilogue(ga, w, U3, w2, U4, nwv);
        }
        return;
    }
    if constexpr (FIRST3) {
        part3<T3, SM>(w3, d3, N3, K, w, nwv, lane, grp);
        part4<T4>(s4, d4, N4, K, ys4, ow, w2, nwv, lane, grp);
    } else {
        part4<T4>(s4, d4, N4, K, ys4, ow, w, nwv, lane, grp);
        part3<T3, SM>(w3, d3, N3, K, w2, nwv, lane, grp);
    }
    if (ga.dst) {   // gate/up pair: SwiGLU + q8_1 epilogue
        glu_epilogue(ga, FIRST3 ? w : w2, N3/RG, FIRST3 ? w2 : w, N4/RG, nwv);
    }
}

// MMVQ-NC mix begin ----------------------------------------------------------------------------------------------------
// N-column pair kernel (NC = 2..6 activation columns, e.g. the tokens of an MTP verification batch). Every column is
// bit-identical to gemv_pair on that column alone: same q8_1 blocks, same lanes per row (L = 16), same item-to-lane
// order (lane l: items l, l + L, ...), same per-item fp32 expressions, same group reductions. Only the weight decode is
// shared by the columns (IQ3 codebook lookups, IQ4_XS lookup8, Q4_K nibble masks): weights are read once per launch.
// Activation records: one set per column (yrec below); the IQ3 and the IQ4 part read the same q8 values and scales from
// them (identical bits). Unit results go straight to dst (lane 0 of a group).
// LDS = IQ3 table + NC*(K/64)*80 bytes (K = 5120: IQ3_S 32 KB + NC*6.25 KB -> NC <= 5, IQ3_XXS 16 KB -> NC <= 7).
// The single-column kernel's fp32 steps, spelled out: the pair kernel as compiled rounds some products separately and
// fuses others (SLP-packed v_pk_mul / v_pk_fma), and the compiler would contract or pack a multi-column loop differently.
// nc_mul / nc_add / nc_sub are never contracted; __builtin_fmaf marks the fused steps. (GGML_MMVQ_NC_VERIFY checks it.)
static __device__ __forceinline__ float nc_mul(const float a, const float b) {
#pragma clang fp contract(off)
    return a*b;
}
static __device__ __forceinline__ float nc_add(const float a, const float b) {
#pragma clang fp contract(off)
    return a + b;
}
static __device__ __forceinline__ float nc_sub(const float a, const float b) {
#pragma clang fp contract(off)
    return a - b;
}
// Activation records of the N-column kernel (all type pairs): the Q4_K pair layout {qs_a[8], qs_b[8], d_a, d_b, sum_a, sum_b}
// per two q8_1 blocks (iq4k::q4k_t::put_y). 16 lanes reading block pairs (IQ3 items, Q4_K items) or single blocks (IQ4_XS
// items) then hit every LDS bank group exactly twice; the IQ4_XS record layout (12 words per block) gives the IQ3 part
// 4-way bank conflicts.
typedef iq4k::q4k_t yrt;
struct yrec {
    static __device__ __forceinline__ const uint32_t * q(const uint32_t * ys, const int b) { return ys + 20*(b >> 1) + 8*(b & 1); }
    static __device__ __forceinline__ float d(const uint32_t * ys, const int b) { return __uint_as_float(ys[20*(b >> 1) + 16 + (b & 1)]); }
};

// the 8 sign-expanded table words of sub-block j of a decoded P = 2 item (= the lookups of T::sub<2, TOFF>)
template <class T, int TOFF>
static __device__ __forceinline__ void nc_tab8(const typename T::template dec<2> & r, const int j, int (&t)[8]) {
    if constexpr (T::TAB_WORDS == 8192) {   // IQ3_S
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int bit = 5*(8*(j & 1) + k), w = bit >> 5, o = bit & 31;
            const uint32_t f = o <= 27 ? (r.W[w] >> o) : __builtin_amdgcn_alignbit(r.W[w + 1], r.W[w], o);
            const uint32_t addr = ((f & 31u) << 10) | iq3g::bx4(r.q[2*j + (k >> 2)], k & 3);
            t[k] = iq3g::tab_ld(addr + TOFF);
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
            const uint32_t addr = (((Wf >> (4*k)) & 15u) << 10) | iq3g::bx4(r.q[2*j + (k >> 2)], k & 3);
            t[k] = iq3g::tab_ld(addr + TOFF);
        }
    }
}
// integer scale step of T::sub<2> (s = the sub-block's dp4a sum)
template <class T>
static __device__ __forceinline__ int nc_scale(const typename T::template dec<2> & r, const int j, const int s) {
    if constexpr (T::TAB_WORDS == 8192) {
        const int sc = (int) ((r.sc >> (4*j)) & 0xFu);
        return __mul24(s, 1 + 2*sc);
    } else {
        const uint32_t W = r.a[j];
        const int ls = iq3g::dp4a((int) ((W >> 7) & 0x01010101u), 0x08040201, 0);
        return iq3g::sdiv<1>(__mul24(ls, s) + iq3g::sdiv<1>(s));
    }
}
// ---- IQ3 part on the matrix cores (MIXG_NC_MFMA): the integer per-sub-block sums of 4 rows x 4 columns come from
// v_mfma_i32_4x4x4i8 (16 blocks of 4 lanes). Lane = 4*l + q: item lane l (0..15, items l, l + 16, .. as in the single
// kernel), q = row of the unit (A operand: the decoded table words of that row) and = column (B operand: that column's
// q8 words; result: the column's sums of the 4 rows). The integer sums are exact (identical to the dp4a chains), the
// fp32 steps per (row, column) are those of the single kernel, and the group reduction runs the same butterfly
// (l ^ 1, l ^ 2, l ^ 7, l ^ 15) on the item lanes, so every output is bit-identical. Needs K/64 % 16 == 0 (all lanes
// run the same items: EXEC stays full for the MFMAs). Used at NC = 4 only: the float steps cost 4 rows per lane whatever NC
// is, so the dp4a form is faster at NC <= 3, and at NC > 4 two MFMA column groups are slower than the dp4a form (MI210).
#ifndef MIXG_NC_MFMA
#define MIXG_NC_MFMA 1
#endif
typedef int mixg_i32x4 __attribute__((ext_vector_type(4)));
template <int i> static __device__ __forceinline__ uint32_t quad_bcast(const uint32_t v) {   // value of lane 4*(lane/4) + i
#if defined(__HIP_DEVICE_COMPILE__)
    return (uint32_t) __builtin_amdgcn_mov_dpp((int) v, i | (i << 2) | (i << 4) | (i << 6), 0xF, 0xF, false);
#else
    return v;
#endif
}
static __device__ __forceinline__ float group_sum_mf(float v) {   // = iq3g::group_sum<16> over the item lanes l
    v += __shfl_xor(v, 4*1, 64);
    v += __shfl_xor(v, 4*2, 64);
    v += __shfl_xor(v, 4*7, 64);
    v += __shfl_xor(v, 4*15, 64);
    return v;
}
template <class T3, int SM, int NC>
static __device__ __forceinline__ void part3_mf(const uint8_t * __restrict__ w3, float * __restrict__ d3, const int N3, const int ld3,
        const int K, const int w, const int nwv, const uint32_t * ys, const int cw) {
    constexpr int P  = 2;
    constexpr int CG = (NC + 3)/4;                         // column groups of 4 (one MFMA chain each)
    typedef typename T3::template ld<P> ldt;
    const int items = K/128;
    const int ipr   = items*(4/P);
    const uint32_t nh = (uint32_t) N3*(uint32_t) items;
    const int U3 = N3/RG;
    const int l = threadIdx.x >> 2, q = threadIdx.x & 3;
    if (w >= U3) {
        return;
    }
    const uint32_t * yc[CG];
#pragma unroll
    for (int g = 0; g < CG; ++g) {
        yc[g] = ys + min(4*g + q, NC - 1)*cw;              // this lane's column of group g (>= NC: a duplicate, not stored)
    }
    float acc[CG][4];
#pragma unroll
    for (int g = 0; g < CG; ++g) for (int i = 0; i < 4; ++i) acc[g][i] = 0.0f;
    // software pipelined over the wave's (unit, item) sequence: the next item's weights are loaded before the current
    // one is decoded (all lanes wrap to the next unit together: ipr % 16 == 0)
    int u = w, it = l;
    ldt cur = iq3g::mj_load<T3, P, RG, SM>(w3, nh, u, items, q, it, U3);
    while (true) {
        int nit = it + L, nu = u;
        if (nit >= ipr) { nit = l; nu = u + nwv; }
        const bool more = nu < U3;
        const ldt nxt = iq3g::mj_load<T3, P, RG, SM>(w3, nh, more ? nu : u, items, q, nit, U3);   // clamped, unconditional
        {
            const typename T3::template dec<P> d = T3::template prep<P>(cur);
            const int b0 = 4*(it >> 1) + 2*(it & 1);
            // per-row data of the 4 rows (quad broadcast): dw and the scale bits of both sub-blocks
            const float dw = T3::template dw<P>(d);
            uint32_t sb;
            if constexpr (T3::TAB_WORDS == 8192) {
                sb = d.sc;                                                   // nibble j: scale of sub-block j
            } else {
                const int ls0 = iq3g::dp4a((int) ((d.a[0] >> 7) & 0x01010101u), 0x08040201, 0);
                const int ls1 = iq3g::dp4a((int) ((d.a[1] >> 7) & 0x01010101u), 0x08040201, 0);
                sb = (uint32_t) ls0 | ((uint32_t) ls1 << 8);
            }
            float dwr[4];
            uint32_t sbr[4];
            dwr[0] = __uint_as_float(quad_bcast<0>(__float_as_uint(dw))); sbr[0] = quad_bcast<0>(sb);
            dwr[1] = __uint_as_float(quad_bcast<1>(__float_as_uint(dw))); sbr[1] = quad_bcast<1>(sb);
            dwr[2] = __uint_as_float(quad_bcast<2>(__float_as_uint(dw))); sbr[2] = quad_bcast<2>(sb);
            dwr[3] = __uint_as_float(quad_bcast<3>(__float_as_uint(dw))); sbr[3] = quad_bcast<3>(sb);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                int t[8];
                nc_tab8<T3, 0>(d, j, t);
#pragma unroll
                for (int g = 0; g < CG; ++g) {
                    const uint32_t * yq = yrec::q(yc[g], b0 + j);
                    const uint4 y0 = *(const uint4 *) yq, y1 = *(const uint4 *) (yq + 4);
                    const float dy = yrec::d(yc[g], b0 + j);
                    mixg_i32x4 s = {0, 0, 0, 0};
#if defined(__HIP_DEVICE_COMPILE__)
                    s = __builtin_amdgcn_mfma_i32_4x4x4i8(t[0], (int) y0.x, s, 0, 0, 0);
                    s = __builtin_amdgcn_mfma_i32_4x4x4i8(t[1], (int) y0.y, s, 0, 0, 0);
                    s = __builtin_amdgcn_mfma_i32_4x4x4i8(t[2], (int) y0.z, s, 0, 0, 0);
                    s = __builtin_amdgcn_mfma_i32_4x4x4i8(t[3], (int) y0.w, s, 0, 0, 0);
                    s = __builtin_amdgcn_mfma_i32_4x4x4i8(t[4], (int) y1.x, s, 0, 0, 0);
                    s = __builtin_amdgcn_mfma_i32_4x4x4i8(t[5], (int) y1.y, s, 0, 0, 0);
                    s = __builtin_amdgcn_mfma_i32_4x4x4i8(t[6], (int) y1.z, s, 0, 0, 0);
                    s = __builtin_amdgcn_mfma_i32_4x4x4i8(t[7], (int) y1.w, s, 0, 0, 0);
#endif
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        int sv;
                        if constexpr (T3::TAB_WORDS == 8192) {
                            sv = __mul24(s[i], 1 + 2*(int) ((sbr[i] >> (4*j)) & 0xFu));
                        } else {
                            const int ls = (int) ((sbr[i] >> (8*j)) & 0xFFu);
                            sv = iq3g::sdiv<1>(__mul24(ls, s[i]) + iq3g::sdiv<1>(s[i]));
                        }
                        acc[g][i] = nc_add(acc[g][i], nc_mul(nc_mul(dwr[i], dy), (float) sv));   // N=1: mul, mul, add
                    }
                }
            }
        }
        if (nu != u) {   // end of the unit (uniform)
#pragma unroll
            for (int g = 0; g < CG; ++g) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float v = group_sum_mf(acc[g][i]);
                    if (l == 0 && 4*g + q < NC) d3[(4*g + q)*ld3 + u*RG + i] = v;
                    acc[g][i] = 0.0f;
                }
            }
        }
        if (!more) {
            break;
        }
        u = nu; it = nit; cur = nxt;
    }
}

// one IQ3 item (P = 2, = iq3g::mj_item) for NC columns; cw = words per column record set
template <class T3, class T4, int NC>
static __device__ __forceinline__ void nc_item3(float (&acc)[NC], const typename T3::template ld<2> & c, const int it,
        const uint32_t * ys, const int cw) {
    const int b0 = 4*(it >> 1) + 2*(it & 1);   // q8_1 blocks of the item's sub-blocks: b0, b0 + 1
    const typename T3::template dec<2> d = T3::template prep<2>(c);
    const float dw = T3::template dw<2>(d);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        int t[8];
        nc_tab8<T3, 0>(d, j, t);
#pragma unroll
        for (int col = 0; col < NC; ++col) {
            const uint32_t * yq = yrec::q(ys + col*cw, b0 + j);
            const uint4 y0 = *(const uint4 *) yq, y1 = *(const uint4 *) (yq + 4);
            const float dy = yrec::d(ys + col*cw, b0 + j);
            int s = 0;
            s = iq3g::dp4a(t[0], (int) y0.x, s); s = iq3g::dp4a(t[1], (int) y0.y, s);
            s = iq3g::dp4a(t[2], (int) y0.z, s); s = iq3g::dp4a(t[3], (int) y0.w, s);
            s = iq3g::dp4a(t[4], (int) y1.x, s); s = iq3g::dp4a(t[5], (int) y1.y, s);
            s = iq3g::dp4a(t[6], (int) y1.z, s); s = iq3g::dp4a(t[7], (int) y1.w, s);
            acc[col] = nc_add(acc[col], nc_mul(nc_mul(dw, dy), (float) nc_scale<T3>(d, j, s)));   // N=1: mul, mul, add
        }
    }
}
// IQ3 part (= part3) for NC columns; column col of the output at d3 + col*ld3
template <class T3, class T4, int SM, int NC>
static __device__ __forceinline__ void part3_nc(const uint8_t * __restrict__ w3, float * __restrict__ d3, const int N3, const int ld3,
        const int K, const int w, const int nwv, const int lane, const int grp, const uint32_t * ys, const int cw) {
    constexpr int P = 2;
    const int items = K/128;
    const int ipr   = items*(4/P);
    const uint32_t nh = (uint32_t) N3*(uint32_t) items;
    const int U3 = N3/RG;
    for (int u = w; u < U3; u += nwv) {
        float acc[NC];
#pragma unroll
        for (int col = 0; col < NC; ++col) acc[col] = 0.0f;
        for (int it = lane; it < ipr; it += L) {
            const typename T3::template ld<P> c = iq3g::mj_load<T3, P, RG, SM>(w3, nh, u, items, grp, it, U3);
            nc_item3<T3, T4, NC>(acc, c, it, ys, cw);
        }
#pragma unroll
        for (int col = 0; col < NC; ++col) {
            const float v = iq3g::group_sum<L>(acc[col]);
            if (lane == 0) d3[col*ld3 + u*RG + grp] = v;
        }
    }
}
// one IQ4 step (item b of the lane) for NC columns: acc[col] += T4::compute(r, record of item b, c) with the decode shared
template <class T4, int NC>
static __device__ __forceinline__ void nc_compute4(float (&acc)[NC], const typename T4::raw & r, const uint32_t * ys, const int b, const int cw,
        const typename T4::lc & c) {
    if constexpr (std::is_same<T4, iq4k::iq4xs_t>::value) {
        uint32_t lo[4], hi[4];
        iq4k::lookup8(r.q.x, c.T, c.m4, c.sb, lo[0], hi[0]);
        iq4k::lookup8(r.q.y, c.T, c.m4, c.sb, lo[1], hi[1]);
        iq4k::lookup8(r.q.z, c.T, c.m4, c.sb, lo[2], hi[2]);
        iq4k::lookup8(r.q.w, c.T, c.m4, c.sb, lo[3], hi[3]);
        const int ls = (int) (((r.h.y >> c.shl) & 0xFu) | (((r.h.x >> c.shh) & 3u) << 4));
        const float dh = iq4k::h2f(r.h.x);
#pragma unroll
        for (int col = 0; col < NC; ++col) {
            const uint32_t * yq = yrec::q(ys + col*cw, b);
            const uint4 y0 = *(const uint4 *) yq;
            const uint4 y1 = *(const uint4 *) (yq + 4);
            const float d8 = yrec::d(ys + col*cw, b);
            int s;
            s = iq4k::dp4a((int) lo[0], (int) y0.x, 0); s = iq4k::dp4a((int) hi[0], (int) y1.x, s);
            s = iq4k::dp4a((int) lo[1], (int) y0.y, s); s = iq4k::dp4a((int) hi[1], (int) y1.y, s);
            s = iq4k::dp4a((int) lo[2], (int) y0.z, s); s = iq4k::dp4a((int) hi[2], (int) y1.z, s);
            s = iq4k::dp4a((int) lo[3], (int) y0.w, s); s = iq4k::dp4a((int) hi[3], (int) y1.w, s);
            const float d = dh * d8;
            acc[col] = __builtin_fmaf(d, (float) __mul24(s, ls - 32), acc[col]);                  // N=1: fma
        }
    } else {   // Q4_K: nibble masks and scales shared
        const uint32_t q[8] = {r.a.x, r.a.y, r.a.z, r.a.w, r.b.x, r.b.y, r.b.z, r.b.w};
        uint32_t ql[8], qh[8];
#pragma unroll
        for (int w = 0; w < 8; ++w) { ql[w] = q[w] & 0x0F0F0F0Fu; qh[w] = (q[w] >> 4) & 0x0F0F0F0Fu; }
        const uint32_t t1 = r.h.y >> c.sh, t2 = r.h.z >> c.sh, t3 = r.h.w >> c.sh;
        const uint32_t SC = c.hi ? ((t3 & 0x0F0Fu) | ((t1 >> 2) & 0x3030u)) : (t1 & 0x3F3Fu);
        const uint32_t MM = c.hi ? (((t3 >> 4) & 0x0F0Fu) | ((t2 >> 2) & 0x3030u)) : (t2 & 0x3F3Fu);
        const int sca = (int) (SC & 0xFFu), scb = (int) (SC >> 8);
        const int ma  = (int) (MM & 0xFFu), mb  = (int) (MM >> 8);
        const float2 dm = __half22float2(*(const half2 *) &r.h.x);
#pragma unroll
        for (int col = 0; col < NC; ++col) {
            const uint32_t * yr = ys + col*cw + 20*b;   // item b = record b
            const uint4 ya0 = *(const uint4 *) yr,       ya1 = *(const uint4 *) (yr + 4);
            const uint4 yb0 = *(const uint4 *) (yr + 8), yb1 = *(const uint4 *) (yr + 12);
            const uint4 dd  = *(const uint4 *) (yr + 16);
            const uint32_t ua[8] = {ya0.x, ya0.y, ya0.z, ya0.w, ya1.x, ya1.y, ya1.z, ya1.w};
            const uint32_t ub[8] = {yb0.x, yb0.y, yb0.z, yb0.w, yb1.x, yb1.y, yb1.z, yb1.w};
            int dota = 0, dotb = 0;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                dota = iq4k::dp4a((int) ql[w], (int) ua[w], dota);
                dotb = iq4k::dp4a((int) qh[w], (int) ub[w], dotb);
            }
            const float da = __uint_as_float(dd.x), db = __uint_as_float(dd.y);
            // N=1 (gemv_pair, compiled): sumf_d = fma(db, Xb, da*Xa), sumf_m = fma(da, Ma, db*Mb), r = dm.x*sumf_d - dm.y*sumf_m
            // (two products, one subtraction), acc + r
            const float sumf_d = __builtin_fmaf(db, (float) __mul24(dotb, scb), nc_mul(da, (float) __mul24(dota, sca)));
            const float sumf_m = __builtin_fmaf(da, (float) __mul24((int) dd.z, ma), nc_mul(db, (float) __mul24((int) dd.w, mb)));
            acc[col] = nc_add(acc[col], nc_sub(nc_mul(dm.x, sumf_d), nc_mul(dm.y, sumf_m)));
        }
    }
}
// IQ4 part (= part4) for NC columns
template <class T4, int NC>
static __device__ __forceinline__ void part4_nc(const iq4k::soa s4, float * __restrict__ d4, const int N4, const int ld4, const int K,
        const uint32_t * ys4, const int cw, const int w, const int nwv, const int lane, const int grp) {
    constexpr int GR = RG;
    const uint32_t bpr = K / QK_K;
    const int ipr = bpr*T4::IPB;
    const int q   = ipr / L;
    const int nunits = N4 / GR;
    const int myu = w < nunits ? (int) ((uint32_t) (nunits - 1 - w)/(uint32_t) nwv) + 1 : 0;
    if (myu == 0) {
        return;
    }
    const int e = myu*q;
    const typename T4::lc c = T4::lane_consts(lane);
    typedef typename T4::raw raw;
    raw A, B;
    const int ue = myu - 1, ke = q - 1;
    int ls = 0, lu = 0, lk = 0;
    const uint32_t qlane = lane*GR + grp, slane = (lane/T4::IPB)*GR + grp;
    const uint32_t boff = 4*bpr*GR;
    auto load_step = [&](raw & bb) {
        const bool past = ls >= e;
        const int kr = past ? ue : lu, kk = past ? ke : lk;
        const uint32_t un = (uint32_t) (w + kr*nwv);
        const uint32_t qs_ = (un*bpr*8 + kk*L)*GR, ss_ = (un*bpr + kk*(L/T4::IPB))*GR;
        bb = T4::load_c(s4, qs_ + qlane, ss_ + slane, boff);
        ++ls; if (++lk == q) { lk = 0; ++lu; }
    };
    load_step(A);
    float acc[NC];
#pragma unroll
    for (int col = 0; col < NC; ++col) acc[col] = 0.0f;
    int cs = 0, cu = 0, ck = 0;
    auto compute_step = [&](raw & bb) {
        nc_compute4<T4, NC>(acc, bb, ys4, lane + ck*L, cw, c);
        if (ck == q - 1 || cs == e - 1) {
            if (cs < e) {
                const int row = (w + cu*nwv)*GR + grp;
#pragma unroll
                for (int col = 0; col < NC; ++col) {
                    const float v = iq4k::group_sum<L>(acc[col]);
                    if (lane == 0) d4[col*ld4 + row] = v;
                }
            }
#pragma unroll
            for (int col = 0; col < NC; ++col) acc[col] = 0.0f;
        }
        ++cs; if (++ck == q) { ck = 0; ++cu; }
    };
    for (int st = 0; st < e; st += 2) {
        load_step(B);
        compute_step(A);
        load_step(A);
        compute_step(B);
    }
}

template <class T3, class T4>
static inline size_t lds_bytes_nc(const int K, const int nc) {
    return (size_t) T3::TAB_WORDS*4 + (size_t) nc*(K/64)*yrt::YW*4;
}

// yg: NC*K/32 q8_1 blocks (column col at yg + col*K/32); outputs d3 / d4 with column strides ld3 / ld4 (floats)
template <class T3, class T4, int SM, bool FIRST3, int NC>
__launch_bounds__(NW*64, 1)
__global__ void gemv_pair_nc(const uint8_t * __restrict__ w3, float * __restrict__ d3, const int N3, const int ld3,
        const iq4k::soa s4, float * __restrict__ d4, const int N4, const int ld4,
        const block_q8_1 * __restrict__ yg, const int K, const int rot) {
    extern __shared__ __align__(16) uint32_t smem[];
#if defined(__HIP_DEVICE_COMPILE__)
    if ((uint32_t) (size_t) (__attribute__((address_space(3))) uint32_t *) smem != 0) __builtin_trap();
#endif
    constexpr int NT = NW*64;
    const int tid  = threadIdx.y*64 + threadIdx.x;
    const int lane = threadIdx.x % L, grp = threadIdx.x / L;
#if defined(__HIP_DEVICE_COMPILE__)
    const int wy = __builtin_amdgcn_readfirstlane(threadIdx.y);
#else
    const int wy = threadIdx.y;
#endif
    const int nwv = gridDim.x*NW;
    const int w   = wy*gridDim.x + blockIdx.x;
    uint32_t * ys = smem + T3::TAB_WORDS;
    const int cw  = (K/64)*yrt::YW;

    // prologue: grid word (no kernel argument needed) first, then the activation blocks of all columns, then LDS writes
    {
        constexpr int GA = T3::TAB_WORDS/16;
        const uint32_t gw = iq3g::grid_word<T3>(tid % GA);
        const int nb = K/32, nbt = NC*nb;
        for (int b = tid; b < nbt; b += NT) {
            const uint8_t * p = (const uint8_t *) (yg + b);
            const uint32_t ds = iq4k::ld_u32_opaque(p);
            const uint4 q0 = *(const uint4 *) (p + 4);
            const uint4 q1 = *(const uint4 *) (p + 20);
            const int col = b / nb;
            yrt::put_y(ys + col*cw, b - col*nb, q0, q1, iq4k::h2f(ds));
        }
        iq3g::tab_store<T3, NT>(smem, gw, tid);
    }
    __syncthreads();

    const int w2 = w >= rot ? w - rot : w + nwv - rot;
    if constexpr (FIRST3) {
        if constexpr (MIXG_NC_MFMA && NC == 4) part3_mf<T3, SM, NC>(w3, d3, N3, ld3, K, w, nwv, ys, cw);
        else part3_nc<T3, T4, SM, NC>(w3, d3, N3, ld3, K, w,  nwv, lane, grp, ys, cw);
        part4_nc<T4, NC>(s4, d4, N4, ld4, K, ys, cw, w2, nwv, lane, grp);
    } else {
        part4_nc<T4, NC>(s4, d4, N4, ld4, K, ys, cw, w,  nwv, lane, grp);
        if constexpr (MIXG_NC_MFMA && NC == 4) part3_mf<T3, SM, NC>(w3, d3, N3, ld3, K, w2, nwv, ys, cw);
        else part3_nc<T3, T4, SM, NC>(w3, d3, N3, ld3, K, w2, nwv, lane, grp, ys, cw);
    }
}

// ---- MTP2-D: pair kernel with both parts on the matrix cores, nc = 2..16 activation columns (gemv_pair_mf).
// Every (row, column) output is bit-identical to gemv_pair on that column alone (= to gemv_pair_nc):
//  * integer stage: the per-sub-block int32 sums of 4 rows x 4 columns come from v_mfma_i32_4x4x4i8 chains (16 blocks of
//    4 lanes, lane = 4*l + q: l = the N=1 item lane, q = row of the unit for the A operand, column for B / D). Integer
//    sums are exact in any order. A = the decoded int8 weights of row q (IQ3: sign-expanded table words, IQ4_XS: lookup8
//    values, Q4_K: nibbles), B = column q's q8 words. CG = ceil(nc/4) chains share the decoded A words.
//  * float stage: per (row, column) exactly the fp32 steps of the single-column kernel in the same item order per item
//    lane (items l, l + 16, ...) with the same group reduction tree (IQ3: l^1, l^2, l^7, l^15 = iq3g::group_sum<16>;
//    IQ4: l^8, l^4, l^2, l^1 = iq4k::group_sum<16>). Rows (0, 1) and (2, 3) of a lane run as packed fp32 pairs
//    (v_pk_mul / v_pk_add / v_pk_fma_f32 round every element like the scalar instruction); the per-row data of the 4 rows
//    come by quad broadcast. Integer products of the N=1 code are evaluated as exact fp32 products (all operands and
//    products are integers below 2^24, so (float) (a*b) == (float) a * (float) b):
//      IQ3_S:   (float) (s*(1 + 2*sc))                 = (float) s * (float) (1 + 2*sc)
//      IQ3_XXS: (float) sdiv<1>(ls*s + sdiv<1>(s))    = truncf((float) s * ((2*ls + 1)/4))   (|s*(2*ls + 1)| < 2^23)
//      IQ4_XS:  (float) (s*(ls - 32))                  = (float) s * (float) (ls - 32)        (|s*(ls - 32)| <= 516128*32 < 2^24)
//      Q4_K:    (float) (dot*sc), (float) (sum*m)      = (float) dot * (float) sc, ...        (< 2^22)
// Schedule: persistent, one workgroup of NWM waves per CU. Workgroup b owns the IQ3 units [U3*b/G, U3*(b+1)/G) and the
// IQ4 units [U4*b/G, U4*(b+1)/G) (U = rows/4); its waves take them round robin (IQ3 first, then IQ4) in passes of S
// slots. The activation is staged in LDS in steps of 1024 weights (one IQ3 step-major chunk step = one Q4_K step = two
// IQ4_XS steps): all K at once when it fits (NB = 0), else a ring of NB = 2 / 1 one-step buffers (workgroup barriers per
// step). A wave keeps the accumulators of its S slots across the steps of a pass. LDS records = yrec (Q4_K pair records),
// column stride cs = steps*320 + 8 words (16-byte reads of the 4 columns of a quad hit different banks).
// MIXG_MF_FENCE (MIXG_MF_SCHED, default on): scheduling fence after every column group and slot (keeps the live ranges of the
// per-group operands short: no spills)
#ifndef MIXG_MF_PF
#define MIXG_MF_PF 0
#endif
#ifndef MIXG_MF_SCHED
#define MIXG_MF_SCHED 1
#endif
#if MIXG_MF_SCHED && defined(__HIP_DEVICE_COMPILE__)
#define MIXG_MF_FENCE() __builtin_amdgcn_sched_barrier(0)
#else
#define MIXG_MF_FENCE() do {} while (0)
#endif
typedef float mixg_f2 __attribute__((ext_vector_type(2)));
static __device__ __forceinline__ mixg_f2 mf_mul(const mixg_f2 a, const mixg_f2 b) {
#pragma clang fp contract(off)
    return a*b;
}
static __device__ __forceinline__ mixg_f2 mf_add(const mixg_f2 a, const mixg_f2 b) {
#pragma clang fp contract(off)
    return a + b;
}
static __device__ __forceinline__ mixg_f2 mf_sub(const mixg_f2 a, const mixg_f2 b) {
#pragma clang fp contract(off)
    return a - b;
}
static __device__ __forceinline__ mixg_f2 mf_fma(const mixg_f2 a, const mixg_f2 b, const mixg_f2 c) {
    return __builtin_elementwise_fma(a, b, c);
}
static __device__ __forceinline__ mixg_f2 mf_sp(const float v) { return mixg_f2{v, v}; }
static __device__ __forceinline__ mixg_f2 mf_cvt(const int a, const int b) { return mixg_f2{(float) a, (float) b}; }
// the value of rows 0, 1 / 2, 3 of the quad (quad broadcast)
static __device__ __forceinline__ void mf_q2(const float v, mixg_f2 & r01, mixg_f2 & r23) {
    const uint32_t u = __float_as_uint(v);
    r01 = mixg_f2{__uint_as_float(quad_bcast<0>(u)), __uint_as_float(quad_bcast<1>(u))};
    r23 = mixg_f2{__uint_as_float(quad_bcast<2>(u)), __uint_as_float(quad_bcast<3>(u))};
}
// v + v[lane ^ m] for the four lane masks m given as bpermute byte addresses (in order)
static __device__ __forceinline__ float mf_gsum(float v, const int b0, const int b1, const int b2, const int b3) {
#if defined(__HIP_DEVICE_COMPILE__)
    v += __int_as_float(__builtin_amdgcn_ds_bpermute(b0, __float_as_int(v)));
    v += __int_as_float(__builtin_amdgcn_ds_bpermute(b1, __float_as_int(v)));
    v += __int_as_float(__builtin_amdgcn_ds_bpermute(b2, __float_as_int(v)));
    v += __int_as_float(__builtin_amdgcn_ds_bpermute(b3, __float_as_int(v)));
#else
    (void) b0; (void) b1; (void) b2; (void) b3;
#endif
    return v;
}
// weights of one slot and step (12 words): IQ3 item (P = 2) {q0 | fa (S) / a (XXS) | fb.x, fb.y (S) / d (XXS), p},
// IQ4_XS items k = 0, 1 {q(0) | q(1) | h(0), h(1)}, Q4_K item {a | b | h}
struct mfw { uint4 a, b, c; };
// one 8-MFMA chain: int sums of 4 rows (A words a) x this lane's column (B words b0 | b1) over 32 weights
static __device__ __forceinline__ mixg_i32x4 mf_chain(const int (&a)[8], const uint4 b0, const uint4 b1) {
    mixg_i32x4 c = {0, 0, 0, 0};
#if defined(__HIP_DEVICE_COMPILE__)
    const uint32_t b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        c = __builtin_amdgcn_mfma_i32_4x4x4i8(a[k], (int) b[k], c, 0, 0, 0);
    }
#else
    (void) a; (void) b0; (void) b1;
#endif
    return c;
}

// IQ3 slot-step: item 16*st + l of this lane's row (q8 blocks 2*rec, 2*rec + 1: record yr = 20*rec words, column 0).
// Sub-block j outer, column group g inner: every accumulator still adds j = 0 before j = 1 (the N=1 order).
template <class T3, int CG>
static __device__ __forceinline__ void mf_s3(mixg_f2 (&acc)[CG][2], const mfw & w, const uint32_t * yr, const int (&co)[CG]) {
    typename T3::template ld<2> c;
    c.q0 = w.a; c.q1 = w.a;
    if constexpr (T3::TAB_WORDS == 8192) {
        c.fa = w.b; c.fb = make_uint2(w.c.x, w.c.y);
    } else {
        c.a = w.b; c.d = w.c.x;
    }
    c.p = (int) w.c.z;
    const typename T3::template dec<2> d = T3::template prep<2>(c);
    mixg_f2 dw01, dw23;
    mf_q2(T3::template dw<2>(d), dw01, dw23);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        int t[8];
        nc_tab8<T3, 0>(d, j, t);
        float f;
        if constexpr (T3::TAB_WORDS == 8192) {
            f = (float) (1 + 2*(int) ((d.sc >> (4*j)) & 0xFu));
        } else {
            f = (float) (2*iq3g::dp4a((int) ((d.a[j] >> 7) & 0x01010101u), 0x08040201, 0) + 1) * 0.25f;
        }
        mixg_f2 f01, f23;
        mf_q2(f, f01, f23);
#pragma unroll
        for (int g = 0; g < CG; ++g) {
            const uint32_t * yc = yr + co[g];   // co[g] = column min(4*g + q, nc - 1) (>= nc: a duplicate, never stored)
            const uint4 y0 = *(const uint4 *) (yc + 8*j), y1 = *(const uint4 *) (yc + 8*j + 4);
            const mixg_f2 dy = mf_sp(__uint_as_float(yc[16 + j]));
            const mixg_i32x4 s = mf_chain(t, y0, y1);
            mixg_f2 s01 = mf_mul(mf_cvt(s[0], s[1]), f01), s23 = mf_mul(mf_cvt(s[2], s[3]), f23);
            if constexpr (T3::TAB_WORDS != 8192) {
                s01 = mixg_f2{truncf(s01.x), truncf(s01.y)}; s23 = mixg_f2{truncf(s23.x), truncf(s23.y)};
            }
            // N=1: acc + (dw*dy)*sv
            acc[g][0] = mf_add(acc[g][0], mf_mul(mf_mul(dw01, dy), s01));
            acc[g][1] = mf_add(acc[g][1], mf_mul(mf_mul(dw23, dy), s23));
            MIXG_MF_FENCE();
        }
    }
}

// IQ4_XS slot-step: items k = 0, 1 (q8 blocks b0 + 16*k of the step buffer, b0 = 32*ls + l); item k outer, group g inner
template <int CG>
static __device__ __forceinline__ void mf_s4x(mixg_f2 (&acc)[CG][2], const mfw & w, const iq4k::iq4xs_t::lc & c,
        const uint32_t * buf, const int b0, const int (&co)[CG]) {
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const uint4 qq = k ? w.b : w.a;
        const uint32_t hx = k ? w.c.z : w.c.x, hy = k ? w.c.w : w.c.y;
        uint32_t lo[4], hi[4];
        iq4k::lookup8(qq.x, c.T, c.m4, c.sb, lo[0], hi[0]);
        iq4k::lookup8(qq.y, c.T, c.m4, c.sb, lo[1], hi[1]);
        iq4k::lookup8(qq.z, c.T, c.m4, c.sb, lo[2], hi[2]);
        iq4k::lookup8(qq.w, c.T, c.m4, c.sb, lo[3], hi[3]);
        const int a[8] = {(int) lo[0], (int) lo[1], (int) lo[2], (int) lo[3], (int) hi[0], (int) hi[1], (int) hi[2], (int) hi[3]};
        const int ls = (int) (((hy >> c.shl) & 0xFu) | (((hx >> c.shh) & 3u) << 4));
        mixg_f2 dh01, dh23, fl01, fl23;
        mf_q2(iq4k::h2f(hx), dh01, dh23);
        mf_q2((float) (ls - 32), fl01, fl23);
        const int b = b0 + 16*k;
        const int o = 20*(b >> 1) + 8*(b & 1), e = 20*(b >> 1) + 16 + (b & 1);
#pragma unroll
        for (int g = 0; g < CG; ++g) {
            const uint32_t * yc = buf + co[g];
            const uint4 y0 = *(const uint4 *) (yc + o), y1 = *(const uint4 *) (yc + o + 4);
            const mixg_f2 d8 = mf_sp(__uint_as_float(yc[e]));
            const mixg_i32x4 s = mf_chain(a, y0, y1);
            // N=1: acc = fma(dh*d8, (float) (s*(ls - 32)), acc)
            acc[g][0] = mf_fma(mf_mul(dh01, d8), mf_mul(mf_cvt(s[0], s[1]), fl01), acc[g][0]);
            acc[g][1] = mf_fma(mf_mul(dh23, d8), mf_mul(mf_cvt(s[2], s[3]), fl23), acc[g][1]);
            MIXG_MF_FENCE();
        }
    }
}

// Q4_K slot-step: item rec = 16*ls + l (sub-block pair = q8 blocks 2*rec, 2*rec + 1: record yr, column 0)
template <int CG>
static __device__ __forceinline__ void mf_s4k(mixg_f2 (&acc)[CG][2], const mfw & w, const iq4k::q4k_t::lc & c,
        const uint32_t * yr, const int (&co)[CG]) {
    const uint32_t qw[8] = {w.a.x, w.a.y, w.a.z, w.a.w, w.b.x, w.b.y, w.b.z, w.b.w};
    int ql[8], qh[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) { ql[k] = (int) (qw[k] & 0x0F0F0F0Fu); qh[k] = (int) ((qw[k] >> 4) & 0x0F0F0F0Fu); }
    const uint32_t t1 = w.c.y >> c.sh, t2 = w.c.z >> c.sh, t3 = w.c.w >> c.sh;
    const uint32_t SC = c.hi ? ((t3 & 0x0F0Fu) | ((t1 >> 2) & 0x3030u)) : (t1 & 0x3F3Fu);
    const uint32_t MM = c.hi ? (((t3 >> 4) & 0x0F0Fu) | ((t2 >> 2) & 0x3030u)) : (t2 & 0x3F3Fu);
    const float2 dm = __half22float2(*(const half2 *) &w.c.x);
    mixg_f2 sa01, sa23, sb01, sb23, ma01, ma23, mb01, mb23, dx01, dx23, dy01, dy23;
    mf_q2((float) (SC & 0xFFu), sa01, sa23);
    mf_q2((float) (SC >> 8),    sb01, sb23);
    mf_q2((float) (MM & 0xFFu), ma01, ma23);
    mf_q2((float) (MM >> 8),    mb01, mb23);
    mf_q2(dm.x, dx01, dx23);
    mf_q2(dm.y, dy01, dy23);
#pragma unroll
    for (int g = 0; g < CG; ++g) {
        const uint32_t * yc = yr + co[g];
        const uint4 dd = *(const uint4 *) (yc + 16);
        const mixg_i32x4 sa = mf_chain(ql, *(const uint4 *) yc,       *(const uint4 *) (yc + 4));
        const mixg_i32x4 sb = mf_chain(qh, *(const uint4 *) (yc + 8), *(const uint4 *) (yc + 12));
        const mixg_f2 da = mf_sp(__uint_as_float(dd.x)), db = mf_sp(__uint_as_float(dd.y));
        const mixg_f2 fz = mf_sp((float) (int) dd.z), fw = mf_sp((float) (int) dd.w);
        // N=1 (gemv_pair, compiled): sumf_d = fma(db, B, da*A), sumf_m = fma(da, Mz, db*Mw), acc + (dm.x*sumf_d - dm.y*sumf_m)
        {
            const mixg_f2 A = mf_mul(mf_cvt(sa[0], sa[1]), sa01), B = mf_mul(mf_cvt(sb[0], sb[1]), sb01);
            const mixg_f2 sd = mf_fma(db, B, mf_mul(da, A));
            const mixg_f2 sm = mf_fma(da, mf_mul(fz, ma01), mf_mul(db, mf_mul(fw, mb01)));
            acc[g][0] = mf_add(acc[g][0], mf_sub(mf_mul(dx01, sd), mf_mul(dy01, sm)));
        }
        {
            const mixg_f2 A = mf_mul(mf_cvt(sa[2], sa[3]), sa23), B = mf_mul(mf_cvt(sb[2], sb[3]), sb23);
            const mixg_f2 sd = mf_fma(db, B, mf_mul(da, A));
            const mixg_f2 sm = mf_fma(da, mf_mul(fz, ma23), mf_mul(db, mf_mul(fw, mb23)));
            acc[g][1] = mf_add(acc[g][1], mf_sub(mf_mul(dx23, sd), mf_mul(dy23, sm)));
        }
        MIXG_MF_FENCE();
    }
}

// activation staging. Ring: one q8_1 block per thread and step (nc*32 <= threads), loaded one step ahead.
struct mf_pre { uint32_t ds; uint4 q0, q1; };
static __device__ __forceinline__ void mf_pre_load(mf_pre & p, const block_q8_1 * __restrict__ yg, const int K, const int step,
        const int tid, const int nc) {
    if (tid < nc*32) {
        const int col = tid >> 5, b = tid & 31;
        const uint8_t * a = (const uint8_t *) (yg + (size_t) col*(K/32) + step*32 + b);
        p.ds = iq4k::ld_u32_opaque(a);
        p.q0 = *(const uint4 *) (a + 4);
        p.q1 = *(const uint4 *) (a + 20);
    }
}
static __device__ __forceinline__ void mf_pre_store(const mf_pre & p, uint32_t * buf, const int cs, const int tid, const int nc) {
    if (tid < nc*32) {
        yrt::put_y(buf + (tid >> 5)*cs, tid & 31, p.q0, p.q1, iq4k::h2f(p.ds));
    }
}
// ring cursor: gsi = step sequence number (all passes); buffer gsi % NB holds step gsi % nst
template <int NB>
struct mf_ring {
    mf_pre pre;
    int gsi, total, nst, cs, bw, K, nc;
    uint32_t * base;
    const block_q8_1 * yg;
    // called by every thread of the workgroup at the start of each step; returns that step's buffer
    __device__ __forceinline__ const uint32_t * enter(const int tid) {
        if constexpr (NB == 0) {
            (void) tid;
            return base;
        } else {
            uint32_t * buf = base + (NB == 2 ? (gsi & 1)*bw : 0);
            if (gsi > 0) {
                if constexpr (NB == 1) __syncthreads();         // every wave is done with the previous step's records
                mf_pre_store(pre, buf, cs, tid, nc);
                __syncthreads();
            }
            if (gsi + 1 < total) {
                mf_pre_load(pre, yg, K, (gsi + 1) % nst, tid, nc);
            }
            ++gsi;
            return buf;
        }
    }
};

// column stride (words) and LDS bytes; NB = 0: all K staged, NB = 1 / 2: ring of one-step buffers
static __host__ __device__ inline int mf_cs(const int K, const int NB) { return (NB == 0 ? K/1024 : 1)*320 + 8; }
template <class T3>
static inline size_t lds_bytes_mf(const int K, const int nc, const int NB) {
    return (size_t) T3::TAB_WORDS*4 + (size_t) (NB == 2 ? 2 : 1)*nc*mf_cs(K, NB)*4;
}

// yg: nc*K/32 q8_1 blocks (column col at yg + col*K/32); outputs d3 / d4 with column strides ld3 / ld4 (floats).
// nc = 4*(CG - 1) + 1 .. 4*CG; S slots per wave and pass; NWM waves per workgroup (one workgroup per CU).
template <class T3, class T4, int SM, int CG, int NB, int S, int NWM>
__launch_bounds__(NWM*64, 1)
__global__ void gemv_pair_mf(const uint8_t * __restrict__ w3, float * __restrict__ d3, const int N3, const int ld3,
        const iq4k::soa s4, float * __restrict__ d4, const int N4, const int ld4,
        const block_q8_1 * __restrict__ yg, const int K, const int nc) {
    extern __shared__ __align__(16) uint32_t smem[];
#if defined(__HIP_DEVICE_COMPILE__)
    if ((uint32_t) (size_t) (__attribute__((address_space(3))) uint32_t *) smem != 0) __builtin_trap();
#endif
    constexpr bool XS = std::is_same<T4, iq4k::iq4xs_t>::value;
    constexpr int NT = NWM*64;
    typedef typename T3::template ld<2> ld3t;
    typedef typename T4::raw raw4;
    const int tid = threadIdx.y*64 + threadIdx.x;
#if defined(__HIP_DEVICE_COMPILE__)
    const int wy = __builtin_amdgcn_readfirstlane(threadIdx.y);
#else
    const int wy = threadIdx.y;
#endif
    const int l = threadIdx.x >> 2, q = threadIdx.x & 3;
    const int nst = K/1024;
    // this workgroup's units: IQ3 [a3, a3 + n3), IQ4 [a4, a4 + cnt - n3)
    const int G = gridDim.x, bk = blockIdx.x;
    const int U3 = N3/RG, U4 = N4/RG;
    const int a3 = (int) (((int64_t) U3*bk)/G), a4 = (int) (((int64_t) U4*bk)/G);
    const int n3 = (int) (((int64_t) U3*(bk + 1))/G) - a3;
    const int cnt = n3 + (int) (((int64_t) U4*(bk + 1))/G) - a4;
    const int nsl = (cnt + NWM - 1)/NWM;                      // slots of the busiest wave
    const int np  = (nsl + S - 1)/S;                          // passes (uniform in the workgroup: ring barriers)
    const int myn = cnt > wy ? (cnt - 1 - wy)/NWM + 1 : 0;    // this wave's slots

    mf_ring<NB> rg;
    rg.base  = smem + T3::TAB_WORDS;
    rg.cs    = mf_cs(K, NB);
    rg.bw    = nc*rg.cs;
    rg.nst   = nst;
    rg.total = np*nst;
    rg.gsi   = 0;
    rg.yg    = yg;
    rg.K     = K;
    rg.nc    = nc;
    {
        constexpr int GA = T3::TAB_WORDS/16;
        const uint32_t gw = iq3g::grid_word<T3>(tid % GA);
        if constexpr (NB == 0) {
            const int nb = K/32, nbt = nc*nb;
            for (int b = tid; b < nbt; b += NT) {
                const uint8_t * p = (const uint8_t *) (yg + b);
                const uint32_t ds = iq4k::ld_u32_opaque(p);
                const uint4 q0 = *(const uint4 *) (p + 4);
                const uint4 q1 = *(const uint4 *) (p + 20);
                const int col = b / nb;
                yrt::put_y(rg.base + col*rg.cs, b - col*nb, q0, q1, iq4k::h2f(ds));
            }
        } else {
            mf_pre_load(rg.pre, yg, K, 0, tid, nc);
            mf_pre_store(rg.pre, rg.base, rg.cs, tid, nc);
        }
        iq3g::tab_store<T3, NT>(smem, gw, tid);
    }
    __syncthreads();

    const int items = K/128;
    const uint32_t nh3 = (uint32_t) N3*(uint32_t) items;
    const uint32_t bpr = K / QK_K;
    const uint32_t boff = 4*bpr*RG;
    const uint32_t slane = (l/T4::IPB)*RG + q;
    const typename T4::lc c4 = T4::lane_consts(l);
    // group sums by ds_bpermute with fixed byte addresses. IQ3 (iq3g::group_sum<16>: l ^ 1, ^ 2, ^ 7, ^ 15): lane ^ 4, ^ 8,
    // ^ 28, ^ 60; IQ4 (iq4k::group_sum<16>: l ^ 8, ^ 4, ^ 2, ^ 1): lane ^ 32, ^ 16, ^ 8, ^ 4
    const int bx4 = (int) ((threadIdx.x ^ 4) << 2), bx8 = (int) ((threadIdx.x ^ 8) << 2);
    const int bx28 = (int) ((threadIdx.x ^ 28) << 2), bx60 = (int) ((threadIdx.x ^ 60) << 2);
    const int bx16 = (int) ((threadIdx.x ^ 16) << 2), bx32 = (int) ((threadIdx.x ^ 32) << 2);
    int co[CG];                                               // word offset of this lane's column of group g
#pragma unroll
    for (int g = 0; g < CG; ++g) co[g] = min(4*g + q, nc - 1)*rg.cs;
    auto load4 = [&](const int un, const int kk) -> raw4 {
        const uint32_t qs_ = ((uint32_t) un*bpr*8 + kk*L)*RG, ss_ = ((uint32_t) un*bpr + kk*(L/T4::IPB))*RG;
        return T4::load_c(s4, qs_ + threadIdx.x, ss_ + slane, boff);
    };

    for (int ps = 0; ps < np; ++ps) {
        const int nv = min(max(myn - ps*S, 0), S);              // valid slots of this wave in this pass (a prefix)
        bool is3[S];
        int  un[S];
#pragma unroll
        for (int s = 0; s < S; ++s) {
            const int cc = wy + (ps*S + s)*NWM;
            is3[s] = cc < n3;
            un[s]  = cc < n3 ? a3 + cc : a4 + cc - n3;
        }
        auto load = [&](const int s, const int st0) -> mfw {
            // opaque unit and step: the per-slot load addresses are formed here, not hoisted out of the step loop
            // (S slots x 64-bit lane addresses would cost ~20 VGPRs per slot)
            int us = un[s], st = st0;
#if defined(__HIP_DEVICE_COMPILE__)
            asm volatile("" : "+s"(us), "+s"(st));
#endif
            mfw r;
            if (is3[s]) {
                const ld3t x = iq3g::mj_load<T3, 2, RG, SM>(w3, nh3, us, items, q, 16*st + l, U3);
                r.a = x.q0;
                if constexpr (T3::TAB_WORDS == 8192) {
                    r.b = x.fa; r.c = make_uint4(x.fb.x, x.fb.y, (uint32_t) x.p, 0u);
                } else {
                    r.b = x.a;  r.c = make_uint4(x.d, 0u, (uint32_t) x.p, 0u);
                }
            } else if constexpr (XS) {
                const raw4 x0 = load4(us, 2*st), x1 = load4(us, 2*st + 1);
                r.a = x0.q; r.b = x1.q; r.c = make_uint4(x0.h.x, x0.h.y, x1.h.x, x1.h.y);
            } else {
                const raw4 x = load4(us, st);
                r.a = x.a; r.b = x.b; r.c = x.h;
            }
            return r;
        };
        mixg_f2 acc[S][CG][2];
#pragma unroll
        for (int s = 0; s < S; ++s) for (int g = 0; g < CG; ++g) { acc[s][g][0] = mf_sp(0.0f); acc[s][g][1] = mf_sp(0.0f); }
        // weights: MIXG_MF_PF = 1: one step ahead per slot (pre[s] = slot s's next step); 0: one (slot, step) ahead
        mfw pre[MIXG_MF_PF ? S : 1];
#pragma unroll
        for (int s = 0; s < (MIXG_MF_PF ? S : 1); ++s) {
            if (s < nv) pre[s] = load(s, 0);
        }
        for (int st = 0; st < nst; ++st) {
            const uint32_t * buf = rg.enter(tid);
            const int ls  = NB == 0 ? st : 0;                      // step within the buffer
#pragma unroll
            for (int s = 0; s < S; ++s) {
                if (s < nv) {
                    mfw cur;
                    if constexpr (MIXG_MF_PF) {
                        cur = pre[s];
                        if (st + 1 < nst) pre[s] = load(s, st + 1);
                    } else {
                        cur = pre[0];
                        if (s + 1 < nv) pre[0] = load(s + 1 < S ? s + 1 : s, st);
                        else if (st + 1 < nst) pre[0] = load(0, st + 1);
                    }
                    if (is3[s]) {
                        mf_s3<T3, CG>(acc[s], cur, buf + 20*(16*ls + l), co);
                    } else if constexpr (XS) {
                        mf_s4x<CG>(acc[s], cur, c4, buf, 32*ls + l, co);
                    } else {
                        mf_s4k<CG>(acc[s], cur, c4, buf + 20*(16*ls + l), co);
                    }
                    MIXG_MF_FENCE();
                }
            }
        }
#pragma unroll
        for (int s = 0; s < S; ++s) {
            if (s < nv) {
                float * dst = is3[s] ? d3 : d4;
                const int ld = is3[s] ? ld3 : ld4;
#pragma unroll
                for (int g = 0; g < CG; ++g) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float a = i < 2 ? acc[s][g][0][i] : acc[s][g][1][i - 2];
                        const float v = is3[s] ? mf_gsum(a, bx4, bx8, bx28, bx60) : mf_gsum(a, bx32, bx16, bx8, bx4);
                        if (l == 0 && 4*g + q < nc) dst[(size_t) (4*g + q)*ld + un[s]*RG + i] = v;
                    }
                }
            }
        }
    }
}

// MMVQ-NC mix end ------------------------------------------------------------------------------------------------------

} // namespace mixg
