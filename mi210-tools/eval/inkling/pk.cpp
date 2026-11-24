// pk.cpp - feasibility microbenchmarks for a persistent decode kernel on MI210 (gfx90a)
//
// Test 1: cost of synchronisation primitives
//   1a  graph of N empty kernels (1 block / 385 blocks)            -> us per launch inside a HIP graph
//   1b  graph of N dependent single-block "chain step" kernels      -> us per small dependent kernel
//   1c  flag ping-pong between two blocks                           -> one-way flag latency
//   1d  hand-written grid barrier (atomic counter + generation)     -> us per grid-wide barrier
//   1e  cooperative-groups grid.sync()                              -> us per grid-wide barrier
// Test 2: one "segment" of a decode layer pass, repeated P times:
//   S dependent single-block small ops (RMS-norm-like) -> BF16 GEMV 3072 x 2304 (14.2 MB, new weights each pass)
//   -> the next pass's chain reads the GEMV output.
//   A  graph: S chain kernels + 1 GEMV kernel per pass       (today's structure)
//   B  graph: 1 fused chain kernel + 1 GEMV kernel per pass  (perfect fusion of the small ops, still launches)
//   C  graph: GEMV kernels only                              (bandwidth reference)
//   D  graph: chain kernels only
//   E  persistent kernel, flags instead of launches, GEMV weights loaded after the flag
//   F  persistent kernel, GEMV weights prefetched into registers BEFORE waiting for the chain
//   A/B/E/F must give bit-identical results (same device functions, same order).
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace cg = cooperative_groups;

#define CK(x)                                                                                           \
    do {                                                                                                \
        hipError_t e_ = (x);                                                                            \
        if (e_ != hipSuccess) {                                                                         \
            fprintf(stderr, "HIP error %s at %s:%d: %s\n", hipGetErrorString(e_), __FILE__, __LINE__, #x); \
            exit(1);                                                                                    \
        }                                                                                               \
    } while (0)

constexpr int K       = 2304;              // GEMV input length (n_embd)
constexpr int N       = 3072;              // GEMV rows (q projection size)
constexpr int TPB     = 256;               // threads per block
constexpr int RPW     = 2;                 // GEMV rows per wave
constexpr int NB_GEMV = N / (RPW * 4);     // 384 GEMV blocks
constexpr int JW      = K / (64 * 4);      // 9 uint2 (4 bf16 each) per lane per row
constexpr int SMAX    = 8;
constexpr long SPIN_LIMIT = 1L << 26;

__device__ __forceinline__ float bf(uint32_t h) { return __uint_as_float(h << 16); }

__device__ __forceinline__ unsigned ld_acq(unsigned * p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}

// deterministic block sum over 256 threads (4 waves of 64)
__device__ __forceinline__ float block_sum(float v, float * red) {
    for (int o = 32; o > 0; o >>= 1) {
        v += __shfl_xor(v, o);
    }
    if ((threadIdx.x & 63) == 0) {
        red[threadIdx.x >> 6] = v;
    }
    __syncthreads();
    const float s = (red[0] + red[1]) + (red[2] + red[3]);
    __syncthreads();
    return s;
}

// one small op of the chain: out = in * rsqrt(mean(in^2) + eps) * g   (one 256-thread block)
__device__ __forceinline__ void chain_step(const float * in, const float * g, float * out, float * red) {
    float v[9];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        v[i] = in[threadIdx.x + i * TPB];
        ss   = fmaf(v[i], v[i], ss);
    }
    const float tot = block_sum(ss, red);
    const float sc  = rsqrtf(tot / K + 1e-6f);
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        out[threadIdx.x + i * TPB] = v[i] * sc * g[threadIdx.x + i * TPB];
    }
}

__device__ __forceinline__ void chain_step_reg(float (&v)[9], const float (&g)[9], float * red) {
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        ss = fmaf(v[i], v[i], ss);
    }
    const float tot = block_sum(ss, red);
    const float sc  = rsqrtf(tot / K + 1e-6f);
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        v[i] = v[i] * sc * g[i];
    }
}

__device__ __forceinline__ float * chain_dst(int s, int S, float * xa, float * xb, float * xi) {
    return s == S - 1 ? xi : ((s & 1) ? xb : xa);
}

__device__ __forceinline__ void load_w(const uint16_t * Wl, int r0, int lane, uint2 (&w)[RPW][JW]) {
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        const uint2 * row = (const uint2 *) (Wl + (size_t) (r0 + r) * K);
#pragma unroll
        for (int j = 0; j < JW; ++j) {
            w[r][j] = row[j * 64 + lane];
        }
    }
}

__device__ __forceinline__ void stage_x(const float * x, float * xs) {
    for (int i = threadIdx.x; i < K / 4; i += TPB) {
        ((float4 *) xs)[i] = ((const float4 *) x)[i];
    }
}

__device__ __forceinline__ void gemv_compute(const uint2 (&w)[RPW][JW], const float * xs, float * y, int r0, int lane) {
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < JW; ++j) {
            const int    e  = (j * 64 + lane) * 4;
            const float4 xv = *(const float4 *) (xs + e);
            acc = fmaf(bf(w[r][j].x & 0xFFFF), xv.x, acc);
            acc = fmaf(bf(w[r][j].x >> 16), xv.y, acc);
            acc = fmaf(bf(w[r][j].y & 0xFFFF), xv.z, acc);
            acc = fmaf(bf(w[r][j].y >> 16), xv.w, acc);
        }
        for (int o = 32; o > 0; o >>= 1) {
            acc += __shfl_xor(acc, o);
        }
        if (lane == 0) {
            y[r0 + r] = acc;
        }
    }
}

// ------------------------------------------------------------------------------------------------
// kernels, launch-based
// ------------------------------------------------------------------------------------------------

__global__ void k_empty() {}

__global__ void __launch_bounds__(TPB) k_chain_step(const float * in, const float * g, float * out) {
    __shared__ float red[4];
    chain_step(in, g, out, red);
}

__global__ void __launch_bounds__(TPB) k_chain_fused(const float * in, const float * G, float * xa, float * xb, float * xi, int S) {
    __shared__ float red[4];
    const float * src = in;
    for (int s = 0; s < S; ++s) {
        float * dst = chain_dst(s, S, xa, xb, xi);
        chain_step(src, G + s * K, dst, red);
        __syncthreads();
        src = dst;
    }
}

__global__ void __launch_bounds__(TPB) k_chain_fused_reg(const float * in, const float * G, float * xi, int S) {
    __shared__ float red[4];
    float g[SMAX][9];
    float v[9];
#pragma unroll
    for (int s = 0; s < SMAX; ++s) {
        if (s < S) {
#pragma unroll
            for (int i = 0; i < 9; ++i) g[s][i] = G[s * K + threadIdx.x + i * TPB];
        }
    }
#pragma unroll
    for (int i = 0; i < 9; ++i) v[i] = in[threadIdx.x + i * TPB];
#pragma unroll
    for (int s = 0; s < SMAX; ++s) {
        if (s < S) chain_step_reg(v, g[s], red);
    }
#pragma unroll
    for (int i = 0; i < 9; ++i) xi[threadIdx.x + i * TPB] = v[i];
}

__global__ void __launch_bounds__(TPB) k_gemv(const uint16_t * Wl, const float * x, float * y) {
    __shared__ float4 xs4[K / 4];
    float *   xs   = (float *) xs4;
    const int lane = threadIdx.x & 63;
    const int r0   = (blockIdx.x * 4 + (threadIdx.x >> 6)) * RPW;
    uint2     w[RPW][JW];
    load_w(Wl, r0, lane, w);
    asm volatile("" ::: "memory");
    stage_x(x, xs);
    __syncthreads();
    gemv_compute(w, xs, y, r0, lane);
}

// ------------------------------------------------------------------------------------------------
// persistent kernel
// ------------------------------------------------------------------------------------------------

struct Sync {
    unsigned ready;
    unsigned pad0[31];
    unsigned done;
    unsigned pad1[31];
    int      err;
};

template <int POLL>
__device__ __forceinline__ bool wait_ge(unsigned * p, unsigned target, int * err) {
    long n = 0;
    if (POLL == 1) {
        // relaxed polling (no L1 invalidate per poll), one acquire fence at the end
        while (__hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target) {
            __builtin_amdgcn_s_sleep(2);
            if (++n > SPIN_LIMIT) {
                atomicExch(err, 1);
                return false;
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        return true;
    }
    while (ld_acq(p) < target) {
        __builtin_amdgcn_s_sleep(1);
        if (++n > SPIN_LIMIT) {
            atomicExch(err, 1);
            return false;
        }
    }
    return true;
}

template <bool PREFETCH, int POLL, bool REGCHAIN>
__global__ void __launch_bounds__(TPB) k_persist(const uint16_t * W, const float * G, const float * x0, float * xa, float * xb,
                                                  float * xi, float * y, int P, int L, int S, Sync * sy, unsigned long long * ts) {
    __shared__ float4 xs4[K / 4];
    __shared__ float  red[4];
    __shared__ int    abort_;
    float *           xs = (float *) xs4;

    if (blockIdx.x == NB_GEMV) {
        // chain block
        float g[SMAX][9];
        if (REGCHAIN) {
#pragma unroll
            for (int s = 0; s < SMAX; ++s) {
                if (s < S) {
#pragma unroll
                    for (int i = 0; i < 9; ++i) g[s][i] = G[s * K + threadIdx.x + i * TPB];
                }
            }
        }
        for (int p = 0; p < P; ++p) {
            if (p > 0) {
                if (threadIdx.x == 0) {
                    abort_ = !wait_ge<POLL>(&sy->done, (unsigned) (NB_GEMV * p), &sy->err);
                }
                __syncthreads();
                if (abort_) {
                    return;
                }
            }
            if (threadIdx.x == 0) ts[4 * p + 0] = wall_clock64();
            const float * src = p == 0 ? x0 : y;
            if (REGCHAIN) {
                float v[9];
#pragma unroll
                for (int i = 0; i < 9; ++i) v[i] = src[threadIdx.x + i * TPB];
#pragma unroll
                for (int s = 0; s < SMAX; ++s) {
                    if (s < S) chain_step_reg(v, g[s], red);
                }
#pragma unroll
                for (int i = 0; i < 9; ++i) xi[threadIdx.x + i * TPB] = v[i];
            } else {
                for (int s = 0; s < S; ++s) {
                    float * dst = chain_dst(s, S, xa, xb, xi);
                    chain_step(src, G + s * K, dst, red);
                    __syncthreads();
                    src = dst;
                }
            }
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) {
                __hip_atomic_store(&sy->ready, (unsigned) (p + 1), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
                ts[4 * p + 1] = wall_clock64();
            }
        }
        return;
    }

    const int lane = threadIdx.x & 63;
    const int r0   = (blockIdx.x * 4 + (threadIdx.x >> 6)) * RPW;
    for (int p = 0; p < P; ++p) {
        const uint16_t * Wl = W + (size_t) (p % L) * N * K;
        uint2            w[RPW][JW];
        if (PREFETCH) {
            load_w(Wl, r0, lane, w);
            asm volatile("" ::: "memory");
        }
        if (threadIdx.x == 0) {
            abort_ = !wait_ge<POLL>(&sy->ready, (unsigned) (p + 1), &sy->err);
            if (blockIdx.x == 0) ts[4 * p + 2] = wall_clock64();
        }
        __syncthreads();
        if (abort_) {
            return;
        }
        if (!PREFETCH) {
            load_w(Wl, r0, lane, w);
            asm volatile("" ::: "memory");
        }
        stage_x(xi, xs);
        __syncthreads();
        gemv_compute(w, xs, y, r0, lane);
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            __hip_atomic_fetch_add(&sy->done, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            if (blockIdx.x == 0) ts[4 * p + 3] = wall_clock64();
        }
    }
}

// ------------------------------------------------------------------------------------------------
// sync primitive kernels
// ------------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(TPB) k_bar(unsigned * cnt, unsigned * gen, int iters, int nb, int * err) {
    unsigned g = 0;
    for (int i = 0; i < iters; ++i) {
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();
            const unsigned a = __hip_atomic_fetch_add(cnt, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
            if (a == (unsigned) nb - 1) {
                __hip_atomic_store(cnt, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                __hip_atomic_store(gen, g + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            } else {
                long n = 0;
                while (ld_acq(gen) == g) {
                    __builtin_amdgcn_s_sleep(1);
                    if (++n > SPIN_LIMIT) {
                        atomicExch(err, 1);
                        break;
                    }
                }
            }
            g = g + 1;
        }
        __syncthreads();
    }
}

__global__ void __launch_bounds__(TPB) k_cg(int iters) {
    cg::grid_group gg = cg::this_grid();
    for (int i = 0; i < iters; ++i) {
        gg.sync();
    }
}

__global__ void k_pingpong(unsigned * f, int iters, int * err) {
    if (threadIdx.x != 0) {
        return;
    }
    for (int i = 0; i < iters; ++i) {
        long n = 0;
        if (blockIdx.x == 0) {
            __hip_atomic_store(f, 2u * i + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            while (ld_acq(f) != 2u * i + 2) {
                if (++n > SPIN_LIMIT) { atomicExch(err, 1); return; }
            }
        } else {
            while (ld_acq(f) != 2u * i + 1) {
                if (++n > SPIN_LIMIT) { atomicExch(err, 1); return; }
            }
            __hip_atomic_store(f, 2u * i + 2, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

// ------------------------------------------------------------------------------------------------
// host
// ------------------------------------------------------------------------------------------------

static hipStream_t st;
static int *       d_err;

template <class PRE, class F> static double time_med(int reps, PRE pre, F f) {
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    std::vector<float> v;
    pre();
    f();  // warm-up
    CK(hipStreamSynchronize(st));
    for (int i = 0; i < reps; ++i) {
        pre();
        CK(hipEventRecord(a, st));
        f();
        CK(hipEventRecord(b, st));
        CK(hipEventSynchronize(b));
        float ms;
        CK(hipEventElapsedTime(&ms, a, b));
        v.push_back(ms);
    }
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
    std::sort(v.begin(), v.end());
    return v[v.size() / 2] * 1000.0;  // us
}

static bool check_err(const char * what) {
    int e = 0;
    CK(hipMemcpy(&e, d_err, sizeof(int), hipMemcpyDeviceToHost));
    if (e) {
        printf("  %s: TIMEOUT in a spin loop (co-residency?)\n", what);
        CK(hipMemset(d_err, 0, sizeof(int)));
        return false;
    }
    return true;
}

template <class BODY> static hipGraphExec_t capture(BODY body) {
    hipGraph_t     g;
    hipGraphExec_t ge;
    CK(hipStreamBeginCapture(st, hipStreamCaptureModeGlobal));
    body();
    CK(hipStreamEndCapture(st, &g));
    CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    CK(hipGraphDestroy(g));
    return ge;
}

static int max_coresident(const void * f, int nsm) {
    int n = 0;
    CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, f, TPB, 0));
    return n * nsm;
}

static auto nop = [] {};

int main(int argc, char ** argv) {
    const int P    = 48;     // passes per run
    const int L    = 16;     // distinct weight sets (no L2 reuse between passes)
    const int REPS = argc > 1 ? atoi(argv[1]) : 20;

    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, 0));
    const int nsm = prop.multiProcessorCount;
    printf("device %s, CUs %d, cooperativeLaunch %d, L2 %d KB\n", prop.gcnArchName, nsm, prop.cooperativeLaunch,
           prop.l2CacheSize / 1024);
    CK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    CK(hipMalloc(&d_err, sizeof(int)));
    CK(hipMemset(d_err, 0, sizeof(int)));

    // ---------------- buffers ----------------
    uint16_t * d_W;
    float *    d_G, *d_x0, *d_xa, *d_xb, *d_xi, *d_y;
    const size_t wbytes = (size_t) L * N * K * sizeof(uint16_t);
    CK(hipMalloc(&d_W, wbytes));
    CK(hipMalloc(&d_G, SMAX * K * sizeof(float)));
    CK(hipMalloc(&d_x0, K * sizeof(float)));
    CK(hipMalloc(&d_xa, K * sizeof(float)));
    CK(hipMalloc(&d_xb, K * sizeof(float)));
    CK(hipMalloc(&d_xi, K * sizeof(float)));
    CK(hipMalloc(&d_y, N * sizeof(float)));
    {
        std::vector<uint16_t> hw((size_t) L * N * K);
        uint32_t              s = 12345;
        auto                  rnd = [&] { s = s * 1664525u + 1013904223u; return (s >> 8) * (1.0f / 16777216.0f); };
        for (auto & v : hw) {
            const float f = (rnd() - 0.5f) * 0.1f;
            uint32_t    u;
            memcpy(&u, &f, 4);
            v = (uint16_t) (u >> 16);
        }
        CK(hipMemcpy(d_W, hw.data(), wbytes, hipMemcpyHostToDevice));
        std::vector<float> hg(SMAX * K), hx(K);
        for (auto & v : hg) v = 0.5f + rnd();
        for (auto & v : hx) v = rnd() - 0.5f;
        CK(hipMemcpy(d_G, hg.data(), hg.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_x0, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_xi, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemset(d_y, 0, N * 4));
    }

    // ---------------- test 1 ----------------
    printf("\n== test 1: synchronisation cost ==\n");
    const int NK = 1000;
    for (int nb : { 1, NB_GEMV + 1 }) {
        hipGraphExec_t ge = capture([&] { for (int i = 0; i < NK; ++i) k_empty<<<nb, TPB, 0, st>>>(); });
        const double   us = time_med(REPS, nop, [&] { CK(hipGraphLaunch(ge, st)); });
        printf("1a graph of %d empty kernels, %3d blocks: %.3f us per kernel\n", NK, nb, us / NK);
        CK(hipGraphExecDestroy(ge));
    }
    {
        hipGraphExec_t ge = capture([&] {
            for (int i = 0; i < NK; ++i) k_chain_step<<<1, TPB, 0, st>>>((i & 1) ? d_xb : d_xa, d_G, (i & 1) ? d_xa : d_xb);
        });
        const double us = time_med(REPS, nop, [&] { CK(hipGraphLaunch(ge, st)); });
        printf("1b graph of %d dependent 1-block chain-step kernels: %.3f us per kernel\n", NK, us / NK);
        CK(hipGraphExecDestroy(ge));
    }
    {
        unsigned * d_f;
        CK(hipMalloc(&d_f, 4));
        const int iters = 50000;
        const double us = time_med(REPS, [&] { CK(hipMemsetAsync(d_f, 0, 4, st)); },
                                   [&] { k_pingpong<<<2, 64, 0, st>>>(d_f, iters, d_err); });
        if (check_err("1c")) {
            printf("1c flag ping-pong between 2 blocks: %.3f us one-way\n", us / (2.0 * iters));
        }
        CK(hipFree(d_f));
    }
    {
        unsigned * d_bar;
        CK(hipMalloc(&d_bar, 2 * 64 * sizeof(unsigned)));
        const int cap = max_coresident(reinterpret_cast<const void *>(&k_bar), nsm);
        printf("   k_bar max co-resident blocks: %d\n", cap);
        for (int nb : { nsm, 2 * nsm, NB_GEMV + 1, 4 * nsm }) {
            if (nb > cap) {
                printf("1d grid barrier, %d blocks: skipped (> co-resident)\n", nb);
                continue;
            }
            int        iters = 20000;
            unsigned * cnt   = d_bar;
            unsigned * gen   = d_bar + 64;
            void *     args[] = { &cnt, &gen, &iters, &nb, &d_err };
            const double us = time_med(REPS, [&] { CK(hipMemsetAsync(d_bar, 0, 2 * 64 * sizeof(unsigned), st)); },
                                       [&] {
                                           CK(hipLaunchCooperativeKernel(reinterpret_cast<const void *>(&k_bar), dim3(nb),
                                                                         dim3(TPB), args, 0, st));
                                       });
            if (check_err("1d")) {
                printf("1d grid barrier (atomic + generation), %3d blocks: %.3f us per barrier\n", nb, us / iters);
            }
        }
        CK(hipFree(d_bar));
    }
    {
        const int cap = max_coresident(reinterpret_cast<const void *>(&k_cg), nsm);
        for (int nb : { nsm, NB_GEMV + 1 }) {
            if (nb > cap) {
                printf("1e cg grid.sync, %d blocks: skipped\n", nb);
                continue;
            }
            int    iters  = 5000;
            void * args[] = { &iters };
            const double us = time_med(REPS, nop, [&] {
                CK(hipLaunchCooperativeKernel(reinterpret_cast<const void *>(&k_cg), dim3(nb), dim3(TPB), args, 0, st));
            });
            printf("1e cooperative-groups grid.sync, %3d blocks: %.3f us per sync\n", nb, us / iters);
        }
    }

    // ---------------- test 2 ----------------
    printf("\n== test 2: segment = S small dependent ops -> BF16 GEMV %d x %d (%.1f MB), %d passes ==\n", N, K,
           N * K * 2 / 1e6, P);
    Sync * d_sync;
    CK(hipMalloc(&d_sync, sizeof(Sync)));
    std::vector<float> yref(N), yt(N);
    unsigned long long * d_ts;
    CK(hipMalloc(&d_ts, 4 * P * sizeof(unsigned long long)));
    int wc_khz = 0;
    CK(hipDeviceGetAttribute(&wc_khz, hipDeviceAttributeWallClockRate, 0));
    printf("wall clock %d kHz\n", wc_khz);
    struct PV { const char * name; const void * f; };
    const PV pvs[] = {
        { "E  persistent, load after flag, acquire polling    ", reinterpret_cast<const void *>(&k_persist<false, 0, false>) },
        { "F  persistent, prefetch, acquire polling           ", reinterpret_cast<const void *>(&k_persist<true, 0, false>) },
        { "E1 persistent, load after flag, relaxed polling    ", reinterpret_cast<const void *>(&k_persist<false, 1, false>) },
        { "F1 persistent, prefetch, relaxed polling           ", reinterpret_cast<const void *>(&k_persist<true, 1, false>) },
        { "E2 persistent, no prefetch, relaxed, register chain", reinterpret_cast<const void *>(&k_persist<false, 1, true>) },
        { "F2 persistent, prefetch, relaxed, register chain   ", reinterpret_cast<const void *>(&k_persist<true, 1, true>) },
    };
    for (const auto & pv : pvs) {
        printf("%s co-resident max %d\n", pv.name, max_coresident(pv.f, nsm));
    }
    auto read_y = [&](std::vector<float> & v) { CK(hipMemcpy(v.data(), d_y, N * 4, hipMemcpyDeviceToHost)); };
    auto same   = [&](const std::vector<float> & a, const std::vector<float> & b) { return memcmp(a.data(), b.data(), N * 4) == 0; };

    {
        hipGraphExec_t ge = capture([&] {
            for (int p = 0; p < P; ++p) k_gemv<<<NB_GEMV, TPB, 0, st>>>(d_W + (size_t) (p % L) * N * K, d_xi, d_y);
        });
        const double us = time_med(REPS, nop, [&] { CK(hipGraphLaunch(ge, st)); });
        printf("C  GEMV only (graph)                  : %7.2f us/pass  (%.3f TB/s)\n", us / P, N * K * 2.0 / (us / P) / 1e6);
        CK(hipGraphExecDestroy(ge));
    }

    for (int S : { 1, 2, 4, 8 }) {
        printf("-- S = %d small ops per segment\n", S);
        double tA, tB, tD;
        // A
        {
            hipGraphExec_t ge = capture([&] {
                for (int p = 0; p < P; ++p) {
                    const float * src = p == 0 ? d_x0 : d_y;
                    for (int s = 0; s < S; ++s) {
                        float * dst = s == S - 1 ? d_xi : ((s & 1) ? d_xb : d_xa);
                        k_chain_step<<<1, TPB, 0, st>>>(src, d_G + s * K, dst);
                        src = dst;
                    }
                    k_gemv<<<NB_GEMV, TPB, 0, st>>>(d_W + (size_t) (p % L) * N * K, d_xi, d_y);
                }
            });
            tA = time_med(REPS, nop, [&] { CK(hipGraphLaunch(ge, st)); });
            CK(hipStreamSynchronize(st));
            read_y(yref);
            CK(hipGraphExecDestroy(ge));
        }
        // B
        {
            hipGraphExec_t ge = capture([&] {
                for (int p = 0; p < P; ++p) {
                    k_chain_fused<<<1, TPB, 0, st>>>(p == 0 ? d_x0 : d_y, d_G, d_xa, d_xb, d_xi, S);
                    k_gemv<<<NB_GEMV, TPB, 0, st>>>(d_W + (size_t) (p % L) * N * K, d_xi, d_y);
                }
            });
            tB = time_med(REPS, nop, [&] { CK(hipGraphLaunch(ge, st)); });
            CK(hipStreamSynchronize(st));
            read_y(yt);
            printf("   B result %s A\n", same(yt, yref) ? "==" : "!=");
            CK(hipGraphExecDestroy(ge));
        }
        // D
        {
            hipGraphExec_t ge = capture([&] {
                for (int p = 0; p < P; ++p) {
                    const float * src = p == 0 ? d_x0 : d_y;
                    for (int s = 0; s < S; ++s) {
                        float * dst = s == S - 1 ? d_xi : ((s & 1) ? d_xb : d_xa);
                        k_chain_step<<<1, TPB, 0, st>>>(src, d_G + s * K, dst);
                        src = dst;
                    }
                }
            });
            tD = time_med(REPS, nop, [&] { CK(hipGraphLaunch(ge, st)); });
            CK(hipGraphExecDestroy(ge));
        }
        // B2: launch-based, fused register chain
        double tB2;
        {
            hipGraphExec_t ge = capture([&] {
                for (int p = 0; p < P; ++p) {
                    k_chain_fused_reg<<<1, TPB, 0, st>>>(p == 0 ? d_x0 : d_y, d_G, d_xi, S);
                    k_gemv<<<NB_GEMV, TPB, 0, st>>>(d_W + (size_t) (p % L) * N * K, d_xi, d_y);
                }
            });
            tB2 = time_med(REPS, nop, [&] { CK(hipGraphLaunch(ge, st)); });
            CK(hipStreamSynchronize(st));
            read_y(yt);
            printf("   B2 result %s A\n", same(yt, yref) ? "==" : "!=");
            CK(hipGraphExecDestroy(ge));
        }
        std::vector<std::string> notes;
        for (const auto & pv : pvs) {
            const uint16_t * W = d_W;
            const float *    G = d_G, *x0 = d_x0;
            float *          xa = d_xa, *xb = d_xb, *xi = d_xi, *y = d_y;
            int              Pv = P, Lv = L, Sv = S;
            Sync *           sy = d_sync;
            unsigned long long * ts = d_ts;
            void *           args[] = { &W, &G, &x0, &xa, &xb, &xi, &y, &Pv, &Lv, &Sv, &sy, &ts };
            if (NB_GEMV + 1 > max_coresident(pv.f, nsm)) {
                notes.push_back(std::string(pv.name) + ": skipped (not co-resident)");
                continue;
            }
            const double us = time_med(REPS, [&] { CK(hipMemsetAsync(d_sync, 0, sizeof(Sync), st)); },
                                       [&] { CK(hipLaunchCooperativeKernel(pv.f, dim3(NB_GEMV + 1), dim3(TPB), args, 0, st)); });
            CK(hipStreamSynchronize(st));
            int e = 0;
            CK(hipMemcpy(&e, &d_sync->err, 4, hipMemcpyDeviceToHost));
            read_y(yt);
            std::vector<unsigned long long> h(4 * P);
            CK(hipMemcpy(h.data(), d_ts, h.size() * 8, hipMemcpyDeviceToHost));
            double ch = 0, fl = 0, gm = 0, hd = 0;
            int    n  = 0;
            const double tick = 1000.0 / wc_khz;  // us per tick
            for (int p = 1; p + 1 < P; ++p) {
                ch += ((long long) h[4 * p + 1] - (long long) h[4 * p + 0]) * tick;
                fl += ((long long) h[4 * p + 2] - (long long) h[4 * p + 1]) * tick;
                gm += ((long long) h[4 * p + 3] - (long long) h[4 * p + 2]) * tick;
                hd += ((long long) h[4 * (p + 1) + 0] - (long long) h[4 * p + 3]) * tick;
                ++n;
            }
            char buf[320];
            snprintf(buf, sizeof buf, "%s: %7.2f us/pass  [chain %.2f | ready->gemv blk0 sees it %.2f | blk0 after flag %.2f | blk0 done->next chain start %.2f]%s%s",
                     pv.name, us / P, ch / n, fl / n, gm / n, hd / n, same(yt, yref) ? "" : "  RESULT DIFFERS", e ? "  TIMEOUT" : "");
            notes.push_back(buf);
        }
        printf("A  graph, %d chain kernels + GEMV      : %7.2f us/pass\n", S, tA / P);
        printf("B  graph, fused chain kernel + GEMV   : %7.2f us/pass\n", tB / P);
        printf("B2 graph, fused register chain + GEMV : %7.2f us/pass\n", tB2 / P);
        printf("D  graph, chain kernels only          : %7.2f us/pass\n", tD / P);
        for (auto & x : notes) printf("%s\n", x.c_str());
    }
    return 0;
}
