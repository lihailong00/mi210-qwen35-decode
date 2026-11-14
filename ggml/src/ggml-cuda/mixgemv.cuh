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
    const int myu = w < nunits ? (nunits - 1 - w)/nwv + 1 : 0;
    if (myu == 0) {
        return;
    }
    const int e = myu*q;
    const typename T4::lc c = T4::lane_consts(lane);
    typedef typename T4::raw raw;
    raw A, B;
    const int ue = (e - 1)/q, ke = (e - 1) % q;   // last real step
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
        const block_q8_1 * __restrict__ yg, const int K, const int y4off, const int rot, const int na, const glu_args ga) {
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
    const int nunits4 = N4/RG, maxu4 = (nunits4 + nwv4 - 1)/nwv4;
    float * ow = (float *) (ys4 + (K/32/T4::YPI)*T4::YW) + wy*maxu4*RG;
    if (na > 0) {
        if (wy < na) {
            part3<T3, SM>(w3, d3, N3, K, wy*gridDim.x + blockIdx.x, na*gridDim.x, lane, grp);
        } else {
            part4<T4>(s4, d4, N4, K, ys4, ow, (wy - na)*gridDim.x + blockIdx.x, nwv4, lane, grp);
        }
        return;
    }
    const int w2 = (w + nwv - rot) % nwv;
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

} // namespace mixg
