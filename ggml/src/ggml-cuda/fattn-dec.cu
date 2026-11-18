// Single-query flash attention decode for short KV caches with the qwen35 sigmoid-gate + q8_1 epilogue in one launch.
//
// Why: the generic path (tile kernel: 2 KV tiles x 12 workgroups of 128 KV rows x 2 heads, + combine kernel with the
// gate epilogue) costs ~28 us per attention layer in the model at KV length 256 although a warm microbenchmark shows
// ~12 us. In the model every dispatch starts with a cold instruction cache (code fetch ~13 ns per instruction on the
// critical path), K/V are cold in HBM, and every kernel boundary is a dependent round trip.
// Here: grid (8 splits, n_kv_heads) of 256 threads; each workgroup issues all loads of its 32-row K/V chunk at once,
// all GQA heads of a KV head share the K/V loads, scores/softmax/PV run from LDS with rolled loops (small code), the 8
// workgroups of a KV head meet at an in-kernel barrier (atomic arrive counter; the whole grid is co-resident, checked
// on the host) and then every workgroup combines its own slice of outputs (all partial loads at once) and writes
// gated = sigmoid(gate) * attn and its q8_1 (32 consecutive lanes = one block, same formula as quantize_q8_1).
// In the model: ~17.5 us per layer instead of ~28 us (in-graph timestamps, KV length 256).
//
// Math: fp32 throughout (Q f32, K/V f16 -> f32, f32 accumulators); the split partials are combined with the usual
// max-rescaling (reassociation only, checked with the KLD gate).
//
// Env: GGML_CUDA_FA_DEC=0 disables (A/B switch), GGML_CUDA_FA_DEC_MAX_KV (default 1024) = largest KV length handled,
// GGML_CUDA_FA_DEC_S (32 or 16) = KV rows per chunk, GGML_CUDA_FA_DEC_DBG=1..4 = early exits for timing ablations only
// (wrong results).

#include "fattn-dec.cuh"
#include "fattn.cuh"             // MTP2-O2: the single-token generic path per query (long KV)
#include "mmvq-fused-host.cuh"  // MTP2-O2

#include <map>
#include <mutex>
#include <utility>
#include <vector>

// AMD only (wave64 layout, __hip_atomic_load / s_sleep in the in-kernel barrier): non-HIP builds get the stub below
#if defined(GGML_USE_HIP)

bool ggml_cuda_mmvq_cache_produce(ggml_backend_cuda_context & ctx, const ggml_tensor * t, block_q8_1 ** q8, int ** sums);

#define FDEC_NT 256 // threads per workgroup (4 waves of 64)

// (int8_t) roundf(v / d) without the IEEE division in the common case (bit-exact, as fattn_q8_round_div)
static __device__ __forceinline__ int fdec_q8_round_div(const float v, const float d, const float inv_d) {
#if defined(GGML_USE_HIP)
    const float t  = v * inv_d;
    const float fr = __builtin_amdgcn_fractf(fabsf(t));
    if (fabsf(fr - 0.5f) > 3.0e-5f) {
        return (int) rintf(t);
    }
#else
    GGML_UNUSED(inv_d);
#endif // defined(GGML_USE_HIP)
    return (int) roundf(v / d);
}

// grid: (FDEC_MAX_SPLIT, n_kv_heads), FDEC_NT threads. Workgroup (split, kvh) handles the KV chunks
// [split*cpb, (split+1)*cpb) of S rows each for the G query heads kvh*G .. kvh*G+G-1.
//
// Code size is the main cost of this kernel in the model: every dispatch starts with a cold instruction cache and each
// instruction on the critical path costs ~13 ns of code fetch (measured: a fully unrolled first version with ~4700
// instructions took ~65 us, early-exit ablations of a ~1900-instruction version showed ~10 us for the combine and ~7 us
// for the q8 epilogue). So: rolled loops everywhere except for the batched loads, few serial cross-lane reductions.
#define FDEC_MAX_SPLIT 8
#define FDEC_MAX_KVH   32 // counters: [0, 32) arrivals, [32, 64) departures
// MTP2-O2: the N-query kernel's arrival counters follow these 64 in the same allocation
#define FDEC_NQ_MAX 16
#define FDEC_NQ_COUNTERS (FDEC_MAX_KVH*FDEC_NQ_MAX)
template <int D, int G, int S>
__launch_bounds__(FDEC_NT, 1)
static __global__ void fattn_dec_gate_kernel(
        const float * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        const half * __restrict__ mask,
        float * __restrict__ parts, float2 * __restrict__ meta, int * __restrict__ counters,
        const float * __restrict__ gate, float * __restrict__ gated, block_q8_1 * __restrict__ q8, int * __restrict__ q8_sums,
        const float scale, const int n_chunks, const int chunks_per_block,
        const int nbq2, const int nb11, const int nb12, const int nb21, const int nb22, const int dbg) {
    static_assert(D == 256, "head size 256 only");
    static_assert(G % 2 == 0 && G <= 8, "even GQA ratio <= 8 only");
    static_assert(S == 16 || S == 32, "S must be 16 or 32");
    constexpr int RB  = D * 2 / 16;                 // 16-byte pieces per K/V row
    constexpr int NL  = S * RB / FDEC_NT;           // 16-byte K (and V) loads per thread per chunk
    constexpr int KST = D + 8;                      // K row stride in LDS (halves), padded against bank conflicts
    constexpr int GH  = G / 2;                      // heads per PV thread
    constexpr int GS  = G * S;                      // (head, row) pairs per chunk, one per thread
    constexpr int NQ  = (G * D / 4 + FDEC_NT - 1) / FDEC_NT; // 16-byte Q loads per thread
    constexpr int NB  = G * D / QK8_1;              // q8_1 blocks of the output of one KV head
    static_assert(D == FDEC_NT && GS <= FDEC_NT && NB * 4 <= FDEC_NT, "thread mapping");
    static_assert(G * D * 4 <= S * KST * 2, "s_out aliases the K tile");
    static_assert(G * D * 4 <= S * D * 2, "s_gate aliases the V tile");

    __shared__ __align__(16) half  s_k[S * KST];
    __shared__ __align__(16) half  s_v[S * D];
    __shared__ __align__(16) float s_q[G * D];
    __shared__ __align__(16) float s_p[G][S];
    __shared__ float  s_m[2][G];

    const int tid   = threadIdx.x;
    const int split = blockIdx.x;
    const int kvh   = blockIdx.y;
    const int c_begin = split * chunks_per_block;
    const int c_end   = min(n_chunks, c_begin + chunks_per_block);

    const char * Qh = (const char *) Q + kvh*G*nbq2;
    const char * Kh = K + kvh*nb12;
    const char * Vh = V + kvh*nb22;

    float4 qr[NQ];
#pragma unroll
    for (int i = 0; i < NQ; ++i) {
        const int f = i*FDEC_NT + tid; // 16-byte piece f of the G x D floats
        qr[i] = f < G*D/4 ? *(const float4 *) (Qh + (f / (D/4))*nbq2 + 16*(f % (D/4))) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    if (tid < G) {
        s_m[0][tid] = -INFINITY;
    }
#pragma unroll
    for (int i = 0; i < NQ; ++i) {
        const int f = i*FDEC_NT + tid;
        if (f < G*D/4) {
            *(float4 *) (s_q + 4*f) = qr[i];
        }
    }

    const int sg = tid / S;       // scores/softmax: head
    const int sr = tid % S;       // scores/softmax: row of the chunk
    const int dp = tid % (D/2);   // PV: dims 2*dp, 2*dp+1
    const int gh = tid / (D/2);   // PV: heads gh*GH .. gh*GH+GH-1
    float2 acc[GH];
    float  lsum[GH];
#pragma unroll
    for (int i = 0; i < GH; ++i) {
        acc[i]  = make_float2(0.0f, 0.0f);
        lsum[i] = 0.0f;
    }

    int par = 0; // s_m[par] = running max before the chunk, s_m[par^1] = after
#pragma unroll 1
    for (int c = c_begin; c < c_end; ++c) {
        // all loads of the chunk at once (one memory round trip), then into LDS
        uint4 kr[NL];
        uint4 vr[NL];
#pragma unroll
        for (int i = 0; i < NL; ++i) {
            const int f = i*FDEC_NT + tid;
            kr[i] = *(const uint4 *) (Kh + (c*S + f/RB)*nb11 + 16*(f % RB));
        }
#pragma unroll
        for (int i = 0; i < NL; ++i) {
            const int f = i*FDEC_NT + tid;
            vr[i] = *(const uint4 *) (Vh + (c*S + f/RB)*nb21 + 16*(f % RB));
        }
        const float mrow = __half2float(mask[c*S + sr]);
#pragma unroll
        for (int i = 0; i < NL; ++i) {
            const int f = i*FDEC_NT + tid;
            *(uint4 *) (s_k + (f/RB)*KST + 8*(f % RB)) = kr[i];
            *(uint4 *) (s_v + 8*f) = vr[i];
        }
        __syncthreads();
        if (dbg == 1) { return; }

        // scores: one (head, row) pair per thread, full dot product from LDS. The S rows of a head are S consecutive
        // lanes of one wave, so the chunk max is a short in-wave reduction and the numerators follow without a barrier.
        if (tid < GS) {
            const half  * kp = s_k + sr*KST;
            const float * qp = s_q + sg*D;
            float a = 0.0f;
#pragma unroll 1
            for (int d = 0; d < D; d += 8) {
                const uint4  k4 = *(const uint4 *) (kp + d);
                const float4 q0 = *(const float4 *) (qp + d);
                const float4 q1 = *(const float4 *) (qp + d + 4);
                const half2 * kh = (const half2 *) &k4;
                const float2 k0 = __half22float2(kh[0]);
                const float2 k1 = __half22float2(kh[1]);
                const float2 k2 = __half22float2(kh[2]);
                const float2 k3 = __half22float2(kh[3]);
                a = fmaf(q0.x, k0.x, a); a = fmaf(q0.y, k0.y, a);
                a = fmaf(q0.z, k1.x, a); a = fmaf(q0.w, k1.y, a);
                a = fmaf(q1.x, k2.x, a); a = fmaf(q1.y, k2.y, a);
                a = fmaf(q1.z, k3.x, a); a = fmaf(q1.w, k3.y, a);
            }
            const float sc = mrow == -INFINITY ? -INFINITY : fmaf(a, scale, mrow);
            const float mc = warp_reduce_max<S>(sc);
            const float mn = fmaxf(s_m[par][sg], mc);
            s_p[sg][sr] = sc == -INFINITY ? 0.0f : __expf(sc - mn);
            if (sr == 0) {
                s_m[par ^ 1][sg] = mn;
            }
        }
        __syncthreads();

        // P*V (+ row sums), rescaled by the change of the running max
#pragma unroll
        for (int i = 0; i < GH; ++i) {
            const float mo = s_m[par][gh*GH + i];
            const float mn = s_m[par ^ 1][gh*GH + i];
            const float al = mn == -INFINITY ? 1.0f : __expf(mo - mn);
            acc[i].x *= al;
            acc[i].y *= al;
            lsum[i]  *= al;
        }
#pragma unroll 2
        for (int r = 0; r < S; ++r) {
            const float2 v = __half22float2(((const half2 *) s_v)[r*(D/2) + dp]);
#pragma unroll
            for (int i = 0; i < GH; ++i) {
                const float pr = s_p[gh*GH + i][r];
                acc[i].x = fmaf(pr, v.x, acc[i].x);
                acc[i].y = fmaf(pr, v.y, acc[i].y);
                lsum[i] += pr;
            }
        }
        par ^= 1;
        __syncthreads();
    }
    if (dbg == 2) { return; }

    // partial results of this workgroup
    {
        const int base = (kvh*FDEC_MAX_SPLIT + split)*G;
#pragma unroll
        for (int i = 0; i < GH; ++i) {
            ((float2 *) (parts + (base + gh*GH + i)*D))[dp] = acc[i];
            if (dp == 0) {
                meta[base + gh*GH + i] = make_float2(s_m[par][gh*GH + i], lsum[i]);
            }
        }
    }

    // barrier over the FDEC_MAX_SPLIT workgroups of this KV head (all workgroups of the grid are co-resident: checked on the
    // host), then every workgroup combines and finishes its own slice of q8_1 blocks
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        atomicAdd(counters + kvh, 1);
        while (__hip_atomic_load(counters + kvh, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < FDEC_MAX_SPLIT) {
            __builtin_amdgcn_s_sleep(1);
        }
        // the last workgroup past the barrier resets both counters for the next launch
        if (atomicAdd(counters + FDEC_MAX_KVH + kvh, 1) == FDEC_MAX_SPLIT - 1) {
            counters[kvh] = 0;
            counters[FDEC_MAX_KVH + kvh] = 0;
        }
    }
    __syncthreads();
    __threadfence();
    if (dbg == 3) { return; }

    // combine + epilogue of the slice of this workgroup, one output per thread: every thread loads the FDEC_MAX_SPLIT
    // partials and (m, l) pairs of its head at once, no further barrier; the q8_1 block of 32 consecutive outputs is
    // formed by 32 consecutive lanes (same arithmetic as quantize_q8_1)
    constexpr int NSG = FDEC_MAX_SPLIT*G;
    constexpr int ON  = NB / FDEC_MAX_SPLIT * QK8_1;
    static_assert(NB % FDEC_MAX_SPLIT == 0 && ON <= FDEC_NT && ON % 64 == 0, "slice mapping");
    if (tid < ON) {
        const int e = split*ON + tid; // = g*D + d
        const int g = e / D;
        const int j = kvh*G*D + e;    // global output index
        const float  * pb = parts + kvh*NSG*D + e;
        const float2 * mb = meta  + kvh*NSG + g;
        const float gt = gate[j];
        float  pv[FDEC_MAX_SPLIT];
        float2 mv[FDEC_MAX_SPLIT];
#pragma unroll
        for (int s = 0; s < FDEC_MAX_SPLIT; ++s) {
            pv[s] = pb[s*G*D];
            mv[s] = mb[s*G];
        }
        float M = -INFINITY;
#pragma unroll
        for (int s = 0; s < FDEC_MAX_SPLIT; ++s) {
            M = fmaxf(M, mv[s].x);
        }
        float num = 0.0f;
        float den = 0.0f;
#pragma unroll
        for (int s = 0; s < FDEC_MAX_SPLIT; ++s) {
            const float w = mv[s].x == -INFINITY ? 0.0f : __expf(mv[s].x - M);
            num = fmaf(w, pv[s], num);
            den = fmaf(w, mv[s].y, den);
        }
        const float r = num / (den * (1.0f + __expf(-gt))); // sigmoid(gate) * attn
        gated[j] = r;
        if (dbg == 4) { return; }

        if (q8 != nullptr) {
            const float amax = warp_reduce_max<QK8_1>(fabsf(r));
            const float d    = amax / 127.0f;
            const int   qv   = amax == 0.0f ? 0 : (int) roundf(r / d);
            const int   sum  = warp_reduce_sum<QK8_1>(qv);
            q8[j/QK8_1].qs[j%QK8_1] = (int8_t) qv;
            if (j % QK8_1 == 0) {
                q8[j/QK8_1].ds = make_half2(d, d * sum);
                q8_sums[j/QK8_1] = sum;
            }
        }
    }
}

// persistent zero-initialized barrier counters per (device, stream); the kernel resets them after use
static int * fdec_counters(ggml_backend_cuda_context & ctx) {
    static std::mutex mtx;
    static std::map<std::pair<int, cudaStream_t>, int *> counters;
    std::lock_guard<std::mutex> lock(mtx);
    const auto key = std::make_pair(ctx.device, ctx.stream());
    const auto it  = counters.find(key);
    if (it != counters.end()) {
        return it->second;
    }
#if defined(GGML_USE_HIP)
    hipStreamCaptureStatus status;
    CUDA_CHECK(hipStreamIsCapturing(ctx.stream(), &status));
    if (status != hipStreamCaptureStatusNone) {
        return nullptr;
    }
#else
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &status));
    if (status != cudaStreamCaptureStatusNone) {
        return nullptr;
    }
#endif // defined(GGML_USE_HIP)
    int * p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, (64 + FDEC_NQ_COUNTERS)*sizeof(int)));   // MTP2-O2: + the N-query counters
    CUDA_CHECK(cudaMemsetAsync(p, 0, (64 + FDEC_NQ_COUNTERS)*sizeof(int), ctx.stream()));
    counters[key] = p;
    return p;
}

static int fdec_env_int(const char * name, const int def) {
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}

template <int G, int S>
static void fdec_launch(ggml_backend_cuda_context & ctx, const dim3 grid,
        const float * Q, const char * K, const char * V, const half * mask, float * parts, float2 * meta, int * counters,
        const float * gate, float * gated, block_q8_1 * q8, int * q8_sums,
        const float scale, const int n_chunks, const int cpb,
        const int nbq2, const int nb11, const int nb12, const int nb21, const int nb22) {
    static const int dbg = fdec_env_int("GGML_CUDA_FA_DEC_DBG", 0); // timing ablations only (wrong results if != 0)
    fattn_dec_gate_kernel<256, G, S><<<grid, FDEC_NT, 0, ctx.stream()>>>(
        Q, K, V, mask, parts, meta, counters, gate, gated, q8, q8_sums, scale, n_chunks, cpb, nbq2, nb11, nb12, nb21, nb22, dbg);
}

template <int S>
static bool fdec_launch_g(ggml_backend_cuda_context & ctx, const int G, const dim3 grid,
        const float * Q, const char * K, const char * V, const half * mask, float * parts, float2 * meta, int * counters,
        const float * gate, float * gated, block_q8_1 * q8, int * q8_sums,
        const float scale, const int n_chunks, const int cpb,
        const int nbq2, const int nb11, const int nb12, const int nb21, const int nb22) {
    switch (G) {
        case 4: fdec_launch<4, S>(ctx, grid, Q, K, V, mask, parts, meta, counters, gate, gated, q8, q8_sums, scale, n_chunks, cpb, nbq2, nb11, nb12, nb21, nb22); return true;
        case 6: fdec_launch<6, S>(ctx, grid, Q, K, V, mask, parts, meta, counters, gate, gated, q8, q8_sums, scale, n_chunks, cpb, nbq2, nb11, nb12, nb21, nb22); return true;
        case 8: fdec_launch<8, S>(ctx, grid, Q, K, V, mask, parts, meta, counters, gate, gated, q8, q8_sums, scale, n_chunks, cpb, nbq2, nb11, nb12, nb21, nb22); return true;
        default: return false;
    }
}

bool ggml_cuda_fattn_dec_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * fa, const float * gate, ggml_tensor * mul_node) {
    // GGML_CUDA_FA_DEC=0 disables; _MAX_KV: largest KV length handled (default 1024); _S: KV rows per chunk (16/32);
    // _SPLIT: max. workgroups per KV head
    static const int enabled   = fdec_env_int("GGML_CUDA_FA_DEC", 1);
    static const int max_kv    = fdec_env_int("GGML_CUDA_FA_DEC_MAX_KV", 1024);
    static const int S         = fdec_env_int("GGML_CUDA_FA_DEC_S", 32);
    if (!enabled || (S != 16 && S != 32)) {
        return false;
    }

    const ggml_tensor * Q    = fa->src[0];
    const ggml_tensor * K    = fa->src[1];
    const ggml_tensor * V    = fa->src[2];
    const ggml_tensor * mask = fa->src[3];
    if (fa->src[4] != nullptr || fa->src[5] != nullptr || mask == nullptr) {
        return false; // sinks / kv_rows / no mask
    }

    float scale, max_bias, logit_softcap;
    memcpy(&scale,         (const float *) fa->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) fa->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) fa->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    const int64_t D    = Q->ne[0];
    const int64_t n_kv = K->ne[1];
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 ||
        fa->type != GGML_TYPE_F32 || !ggml_is_contiguous(fa) ||
        D != 256 || K->ne[0] != D || V->ne[0] != D || V->ne[1] != n_kv ||
        Q->ne[1] != 1 || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 || V->ne[2] != K->ne[2] ||
        Q->ne[2] % K->ne[2] != 0 || K->ne[2] > 64 ||
        mask->ne[0] < n_kv || mask->ne[2] != 1 || mask->ne[3] != 1 ||
        n_kv > max_kv || n_kv % S != 0 || ggml_nelements(fa) != D*Q->ne[2]) {
        return false;
    }
    // 16-byte vector loads
    if ((uintptr_t) Q->data % 16 || Q->nb[0] != sizeof(float) || Q->nb[2] % 16 ||
        (uintptr_t) K->data % 16 || K->nb[0] != sizeof(half) || K->nb[1] % 16 || K->nb[2] % 16 ||
        (uintptr_t) V->data % 16 || V->nb[0] != sizeof(half) || V->nb[1] % 16 || V->nb[2] % 16 ||
        mask->nb[0] != sizeof(half)) {
        return false;
    }
    if (K->nb[2]*K->ne[2] > INT_MAX/2 || V->nb[2]*V->ne[2] > INT_MAX/2 || (int64_t) n_kv*std::max(K->nb[1], V->nb[1]) > INT_MAX/2) {
        return false; // 32-bit offsets in the kernel
    }
    const int G = Q->ne[2] / K->ne[2];
    if (G != 4 && G != 6 && G != 8) {
        return false;
    }

    const int n_kvh    = K->ne[2];
    const int n_chunks = n_kv / S;
    if (n_chunks % FDEC_MAX_SPLIT != 0) {
        return false; // exactly FDEC_MAX_SPLIT workgroups per KV head
    }
    const int n_split = FDEC_MAX_SPLIT;
    const int cpb     = n_chunks / FDEC_MAX_SPLIT;
    if (n_kvh > FDEC_MAX_KVH) {
        return false;
    }
    // the in-kernel barrier over the workgroups of a KV head needs the whole grid co-resident (1 workgroup per CU fits)
    static const bool concurrent_streams = getenv("GGML_CUDA_GRAPH_OPT") != nullptr && atoi(getenv("GGML_CUDA_GRAPH_OPT")) != 0;
    if (concurrent_streams || n_split*n_kvh > ggml_cuda_info().devices[ctx.device].nsm) {
        return false;
    }

    int * counters = fdec_counters(ctx);
    if (counters == nullptr) {
        return false;
    }

    block_q8_1 * q8      = nullptr;
    int        * q8_sums = nullptr;
    if (!ggml_cuda_mmvq_cache_produce(ctx, mul_node, &q8, &q8_sums)) {
        q8      = nullptr;
        q8_sums = nullptr;
    }

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float>  parts(pool, (size_t) n_kvh*n_split*G*D);
    ggml_cuda_pool_alloc<float2> meta (pool, (size_t) n_kvh*n_split*G);

    const dim3 grid(n_split, n_kvh, 1);
    const float * Qd = (const float *) Q->data;
    const char  * Kd = (const char *) K->data;
    const char  * Vd = (const char *) V->data;
    const half  * Md = (const half *) mask->data; // row 0 = the single query
    float       * gated = (float *) mul_node->data;

    bool ok = false;
    switch (S) {
        case 16: ok = fdec_launch_g<16>(ctx, G, grid, Qd, Kd, Vd, Md, parts.ptr, meta.ptr, counters, gate, gated, q8, q8_sums,
                                        scale, n_chunks, cpb, (int) Q->nb[2], (int) K->nb[1], (int) K->nb[2], (int) V->nb[1], (int) V->nb[2]); break;
        case 32: ok = fdec_launch_g<32>(ctx, G, grid, Qd, Kd, Vd, Md, parts.ptr, meta.ptr, counters, gate, gated, q8, q8_sums,
                                        scale, n_chunks, cpb, (int) Q->nb[2], (int) K->nb[1], (int) K->nb[2], (int) V->nb[1], (int) V->nb[2]); break;
        default: break;
    }
    CUDA_CHECK(cudaGetLastError());
    return ok;
}


// ------------------------------------------------------------------------------------------------- MTP2-O2: N queries
// N-query variant for multi-token verification (2..FDEC_NQ_MAX consecutive tokens of one sequence, causal mask): each
// query is computed exactly like a single-token call of fattn_dec_gate_kernel at its own position, so every token's
// gated output and q8_1 blocks are bit-identical to the N=1 path.
//  - The single-token graph of token t has n_kv_t = max(256, pad256(last visible cell + 1)) (llama_kv_cache::get_n_kv,
//    clamped to the batch n_kv). It is derived on the device from mask row t (graph capture safe): the split partition
//    (n_kv_t / S chunks over FDEC_MAX_SPLIT workgroups) and the mask range [0, n_kv_t) are then those of the N=1 call.
//  - Per workgroup (split, kvh, t): the statements of the N=1 kernel (scores, chunk max, exp, P*V order, partials).
//  - Combine: the last of the FDEC_MAX_SPLIT workgroups of (kvh, t) to arrive (atomic counter, no spinning, so the grid
//    need not be co-resident) combines all G*D outputs of the head group with the per-output code of the N=1 kernel
//    and writes gated = sigmoid(gate) * attn and its q8_1 (blocks of 32 consecutive outputs; max/sum are exact).
//  - Batch n_kv > GGML_CUDA_FA_DEC_MAX_KV: fdec_nq_long below (the single-token generic call per query).
// Env: GGML_CUDA_FA_DEC_NQ=0 disables (the generic FLASH_ATTN_EXT + separate gate, as before);
//      GGML_CUDA_FA_DEC_NQ_VERIFY=<n>: for the first n calls, also run the N=1 path per query (sync, graphs off) and
//      compare gated output and q8 bytes (debug); GGML_CUDA_FA_DEC_NQ_SELECT=1: long KV always in the select form (debug);
//      GGML_CUDA_FA_DEC_NQ_LONG=0: long KV through the generic N-query path as before (faster above 1024, not per-token exact).

template <int D, int G, int S>
__launch_bounds__(FDEC_NT, 1)
static __global__ void fattn_dec_gate_nq_kernel(
        const float * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        const half * __restrict__ mask,
        float * __restrict__ parts, float2 * __restrict__ meta, int * __restrict__ counters,
        const float * __restrict__ gate, float * __restrict__ gated, block_q8_1 * __restrict__ q8, int * __restrict__ q8_sums,
        const float scale, const int n_kv, const int lim, const int mask_s1,
        const int nbq1, const int nbq2, const int nb11, const int nb12, const int nb21, const int nb22) {
    static_assert(D == 256, "head size 256 only");
    static_assert(G % 2 == 0 && G <= 8, "even GQA ratio <= 8 only");
    static_assert(S == 16 || S == 32, "S must be 16 or 32");
    constexpr int RB  = D * 2 / 16;
    constexpr int NL  = S * RB / FDEC_NT;
    constexpr int KST = D + 8;
    constexpr int GH  = G / 2;
    constexpr int GS  = G * S;
    constexpr int NQ  = (G * D / 4 + FDEC_NT - 1) / FDEC_NT;
    static_assert(D == FDEC_NT && GS <= FDEC_NT, "thread mapping");
    static_assert((G * D) % FDEC_NT == 0 && FDEC_NT % QK8_1 == 0, "combine mapping");

    __shared__ __align__(16) half  s_k[S * KST];
    __shared__ __align__(16) half  s_v[S * D];
    __shared__ __align__(16) float s_q[G * D];
    __shared__ __align__(16) float s_p[G][S];
    __shared__ float  s_m[2][G];
    __shared__ int    s_nb;
    __shared__ int    s_last;

    const int tid   = threadIdx.x;
    const int split = blockIdx.x;
    const int kvh   = blockIdx.y;
    const int t     = blockIdx.z;
    const int n_kvh = gridDim.y;

    const half * mrow_t = mask + (int64_t) t*mask_s1;
    const char * Qh = (const char *) Q + t*nbq1 + kvh*G*nbq2;
    const char * Kh = K + kvh*nb12;
    const char * Vh = V + kvh*nb22;

    float4 qr[NQ];
#pragma unroll
    for (int i = 0; i < NQ; ++i) {
        const int f = i*FDEC_NT + tid;
        qr[i] = f < G*D/4 ? *(const float4 *) (Qh + (f / (D/4))*nbq2 + 16*(f % (D/4))) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    // n_kv of the single-token graph of this query: 256-blocks up to the last visible cell of mask row t
    int nb = 0;
#pragma unroll 1
    for (int i = tid; i < n_kv; i += FDEC_NT) {
        if (__half2float(mrow_t[i]) != -INFINITY) {
            nb = i/256 + 1;
        }
    }
    if (tid == 0) {
        s_nb = 0;
    }
    if (tid < G) {
        s_m[0][tid] = -INFINITY;
    }
#pragma unroll
    for (int i = 0; i < NQ; ++i) {
        const int f = i*FDEC_NT + tid;
        if (f < G*D/4) {
            *(float4 *) (s_q + 4*f) = qr[i];
        }
    }
    __syncthreads();
    if (nb > 0) {
        atomicMax(&s_nb, nb);
    }
    __syncthreads();
    const int n_kv_t           = min(n_kv, max(256, s_nb*256));
    if (n_kv_t > lim) {
        return;   // (long-KV mode) this query takes the single-token generic path: nothing to do here
    }
    const int n_chunks         = n_kv_t / S;
    const int chunks_per_block = n_chunks / FDEC_MAX_SPLIT;
    const int c_begin = split * chunks_per_block;
    const int c_end   = min(n_chunks, c_begin + chunks_per_block);

    const int sg = tid / S;
    const int sr = tid % S;
    const int dp = tid % (D/2);
    const int gh = tid / (D/2);
    float2 acc[GH];
    float  lsum[GH];
#pragma unroll
    for (int i = 0; i < GH; ++i) {
        acc[i]  = make_float2(0.0f, 0.0f);
        lsum[i] = 0.0f;
    }

    int par = 0;
#pragma unroll 1
    for (int c = c_begin; c < c_end; ++c) {
        uint4 kr[NL];
        uint4 vr[NL];
#pragma unroll
        for (int i = 0; i < NL; ++i) {
            const int f = i*FDEC_NT + tid;
            kr[i] = *(const uint4 *) (Kh + (c*S + f/RB)*nb11 + 16*(f % RB));
        }
#pragma unroll
        for (int i = 0; i < NL; ++i) {
            const int f = i*FDEC_NT + tid;
            vr[i] = *(const uint4 *) (Vh + (c*S + f/RB)*nb21 + 16*(f % RB));
        }
        const float mrow = __half2float(mrow_t[c*S + sr]);
#pragma unroll
        for (int i = 0; i < NL; ++i) {
            const int f = i*FDEC_NT + tid;
            *(uint4 *) (s_k + (f/RB)*KST + 8*(f % RB)) = kr[i];
            *(uint4 *) (s_v + 8*f) = vr[i];
        }
        __syncthreads();

        if (tid < GS) {
            const half  * kp = s_k + sr*KST;
            const float * qp = s_q + sg*D;
            float a = 0.0f;
#pragma unroll 1
            for (int d = 0; d < D; d += 8) {
                const uint4  k4 = *(const uint4 *) (kp + d);
                const float4 q0 = *(const float4 *) (qp + d);
                const float4 q1 = *(const float4 *) (qp + d + 4);
                const half2 * kh = (const half2 *) &k4;
                const float2 k0 = __half22float2(kh[0]);
                const float2 k1 = __half22float2(kh[1]);
                const float2 k2 = __half22float2(kh[2]);
                const float2 k3 = __half22float2(kh[3]);
                a = fmaf(q0.x, k0.x, a); a = fmaf(q0.y, k0.y, a);
                a = fmaf(q0.z, k1.x, a); a = fmaf(q0.w, k1.y, a);
                a = fmaf(q1.x, k2.x, a); a = fmaf(q1.y, k2.y, a);
                a = fmaf(q1.z, k3.x, a); a = fmaf(q1.w, k3.y, a);
            }
            const float sc = mrow == -INFINITY ? -INFINITY : fmaf(a, scale, mrow);
            const float mc = warp_reduce_max<S>(sc);
            const float mn = fmaxf(s_m[par][sg], mc);
            s_p[sg][sr] = sc == -INFINITY ? 0.0f : __expf(sc - mn);
            if (sr == 0) {
                s_m[par ^ 1][sg] = mn;
            }
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < GH; ++i) {
            const float mo = s_m[par][gh*GH + i];
            const float mn = s_m[par ^ 1][gh*GH + i];
            const float al = mn == -INFINITY ? 1.0f : __expf(mo - mn);
            acc[i].x *= al;
            acc[i].y *= al;
            lsum[i]  *= al;
        }
#pragma unroll 2
        for (int r = 0; r < S; ++r) {
            const float2 v = __half22float2(((const half2 *) s_v)[r*(D/2) + dp]);
#pragma unroll
            for (int i = 0; i < GH; ++i) {
                const float pr = s_p[gh*GH + i][r];
                acc[i].x = fmaf(pr, v.x, acc[i].x);
                acc[i].y = fmaf(pr, v.y, acc[i].y);
                lsum[i] += pr;
            }
        }
        par ^= 1;
        __syncthreads();
    }

    const int grp = t*n_kvh + kvh;
    {
        const int base = (grp*FDEC_MAX_SPLIT + split)*G;
#pragma unroll
        for (int i = 0; i < GH; ++i) {
            ((float2 *) (parts + (base + gh*GH + i)*D))[dp] = acc[i];
            if (dp == 0) {
                meta[base + gh*GH + i] = make_float2(s_m[par][gh*GH + i], lsum[i]);
            }
        }
    }

    // the last workgroup of (kvh, t) to arrive combines (no waiting: any number of workgroups, co-resident or not)
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const int old = atomicAdd(counters + grp, 1);
        s_last = old == FDEC_MAX_SPLIT - 1;
        if (old == FDEC_MAX_SPLIT - 1) {
            counters[grp] = 0; // all arrivals of this launch are counted: reset for the next launch
        }
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    __threadfence();

    constexpr int NSG = FDEC_MAX_SPLIT*G;
    const int64_t tok = (int64_t) t*n_kvh*G*D; // offset of token t in gate / gated (contiguous [n_head*D, N])
#pragma unroll 1
    for (int e = tid; e < G*D; e += FDEC_NT) {
        const int g = e / D;
        const int64_t j = tok + kvh*G*D + e;
        const float  * pb = parts + grp*NSG*D + e;
        const float2 * mb = meta  + grp*NSG + g;
        const float gt = gate[j];
        float  pv[FDEC_MAX_SPLIT];
        float2 mv[FDEC_MAX_SPLIT];
#pragma unroll
        for (int s = 0; s < FDEC_MAX_SPLIT; ++s) {
            pv[s] = pb[s*G*D];
            mv[s] = mb[s*G];
        }
        float M = -INFINITY;
#pragma unroll
        for (int s = 0; s < FDEC_MAX_SPLIT; ++s) {
            M = fmaxf(M, mv[s].x);
        }
        float num = 0.0f;
        float den = 0.0f;
#pragma unroll
        for (int s = 0; s < FDEC_MAX_SPLIT; ++s) {
            const float w = mv[s].x == -INFINITY ? 0.0f : __expf(mv[s].x - M);
            num = fmaf(w, pv[s], num);
            den = fmaf(w, mv[s].y, den);
        }
        const float r = num / (den * (1.0f + __expf(-gt)));
        gated[j] = r;

        if (q8 != nullptr) {
            const float amax = warp_reduce_max<QK8_1>(fabsf(r));
            const float d    = amax / 127.0f;
            const int   qv   = amax == 0.0f ? 0 : (int) roundf(r / d);
            const int   sum  = warp_reduce_sum<QK8_1>(qv);
            q8[j/QK8_1].qs[j%QK8_1] = (int8_t) qv;
            if (j % QK8_1 == 0) {
                q8[j/QK8_1].ds = make_half2(d, d * sum);
                q8_sums[j/QK8_1] = sum;
            }
        }
    }
}

// zero-initialized arrival counters per (device, stream) for the N-query kernel; reset by the kernel after use
// arrival counters of the N-query kernel (zero, reset by the kernel after use): allocated with the N=1 kernel's
// counters by its first call outside stream capture; nullptr if that has not happened yet
static int * fdec_nq_counters(ggml_backend_cuda_context & ctx) {
    int * c = fdec_counters(ctx);
    return c ? c + 64 : nullptr;
}

bool ggml_cuda_fattn_dec_nq_wanted(const int64_t n_tokens) {
    static const int enabled = fdec_env_int("GGML_CUDA_FA_DEC_NQ", 1) && fdec_env_int("GGML_CUDA_FA_DEC", 1);
    return enabled && n_tokens >= 2 && n_tokens <= FDEC_NQ_MAX;
}

int ggml_cuda_mmvq_nc_max();

struct fdec_nq_args {
    const float * Q; const char * K; const char * V; const half * mask;
    float * gate_out; const float * gate;
    int n_kvh, G, S, N, n_kv, lim, mask_s1, nbq1, nbq2, nb11, nb12, nb21, nb22;
    float scale;
};

template <int G, int S>
static void fdec_nq_launch(ggml_backend_cuda_context & ctx, const fdec_nq_args & a, int * counters, block_q8_1 * q8, int * q8_sums) {
    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float>  parts(pool, (size_t) a.N*a.n_kvh*FDEC_MAX_SPLIT*G*256);
    ggml_cuda_pool_alloc<float2> meta (pool, (size_t) a.N*a.n_kvh*FDEC_MAX_SPLIT*G);
    const dim3 grid(FDEC_MAX_SPLIT, a.n_kvh, a.N);
    fattn_dec_gate_nq_kernel<256, G, S><<<grid, FDEC_NT, 0, ctx.stream()>>>(
        a.Q, a.K, a.V, a.mask, parts.ptr, meta.ptr, counters, a.gate, a.gate_out, q8, q8_sums,
        a.scale, a.n_kv, a.lim, a.mask_s1, a.nbq1, a.nbq2, a.nb11, a.nb12, a.nb21, a.nb22);
    CUDA_CHECK(cudaGetLastError());
}

template <int S>
static bool fdec_nq_launch_g(ggml_backend_cuda_context & ctx, const fdec_nq_args & a, int * counters, block_q8_1 * q8, int * q8_sums) {
    switch (a.G) {
        case 4: fdec_nq_launch<4, S>(ctx, a, counters, q8, q8_sums); return true;
        case 6: fdec_nq_launch<6, S>(ctx, a, counters, q8, q8_sums); return true;
        case 8: fdec_nq_launch<8, S>(ctx, a, counters, q8, q8_sums); return true;
        default: return false;
    }
}

static bool fdec_nq_run(ggml_backend_cuda_context & ctx, const fdec_nq_args & a, int * counters, block_q8_1 * q8, int * q8_sums) {
    return a.S == 16 ? fdec_nq_launch_g<16>(ctx, a, counters, q8, q8_sums) : fdec_nq_launch_g<32>(ctx, a, counters, q8, q8_sums);
}

// debug (GGML_CUDA_FA_DEC_NQ_VERIFY): every query once more through the N=1 kernel on single-query views (n_kv_t from
// the mask on the host), compare gated output and q8 bytes bitwise, then re-run the N-query kernel (the N=1 calls reuse
// the activation cache buffer)
static void fdec_nq_verify(ggml_backend_cuda_context & ctx, const ggml_tensor * fa, const float * gate, ggml_tensor * mul_node,
        const fdec_nq_args & a, int * counters, const bool with_q8) {
    const ggml_tensor * Q = fa->src[0];
    const ggml_tensor * K = fa->src[1];
    const ggml_tensor * V = fa->src[2];
    const ggml_tensor * mask = fa->src[3];
    const int64_t HD = Q->ne[2]*256;
    const int64_t nblk = HD/QK8_1;
    cudaStream_t st = ctx.stream();
    CUDA_CHECK(cudaStreamSynchronize(st));
    std::vector<float> got((size_t) a.N*HD);
    std::vector<block_q8_1> got_q8(with_q8 ? (size_t) a.N*nblk : 0);
    std::vector<int> got_s(with_q8 ? (size_t) a.N*nblk : 0);
    CUDA_CHECK(cudaMemcpy(got.data(), mul_node->data, got.size()*sizeof(float), cudaMemcpyDeviceToHost));
    if (with_q8) {
        block_q8_1 * q8 = nullptr; int * sums = nullptr;
        GGML_ASSERT(ggml_cuda_mmvq_cache_produce(ctx, mul_node, &q8, &sums));
        CUDA_CHECK(cudaMemcpy(got_q8.data(), q8, got_q8.size()*sizeof(block_q8_1), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(got_s.data(), sums, got_s.size()*sizeof(int), cudaMemcpyDeviceToHost));
    }
    std::vector<ggml_fp16_t> mrow((size_t) a.n_kv);
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), (size_t) HD);
    std::vector<float> ref((size_t) HD);
    std::vector<block_q8_1> ref_q8((size_t) nblk);
    std::vector<int> ref_s((size_t) nblk);
    for (int t = 0; t < a.N; ++t) {
        CUDA_CHECK(cudaMemcpy(mrow.data(), (const char *) mask->data + t*mask->nb[1], a.n_kv*sizeof(ggml_fp16_t), cudaMemcpyDeviceToHost));
        int last = -1;
        for (int i = 0; i < a.n_kv; ++i) {
            if (GGML_FP16_TO_FP32(mrow[i]) != -INFINITY) {
                last = i;
            }
        }
        const int n_kv_t = std::min(a.n_kv, std::max(256, (int) GGML_PAD(last + 1, 256)));
        ggml_tensor Q1 = *Q, K1 = *K, V1 = *V, M1 = *mask, F1 = *fa, U1 = *mul_node;
        Q1.ne[1] = 1; Q1.data = (char *) Q->data + t*Q->nb[1];
        K1.ne[1] = n_kv_t; V1.ne[1] = n_kv_t;
        M1.ne[1] = 1; M1.data = (char *) mask->data + t*mask->nb[1];
        F1.src[0] = &Q1; F1.src[1] = &K1; F1.src[2] = &V1; F1.src[3] = &M1;
        F1.ne[2] = 1; F1.ne[3] = 1; F1.nb[3] = F1.nb[2]; F1.data = scratch.ptr; F1.view_src = nullptr;
        U1.ne[0] = HD; U1.ne[1] = 1; U1.ne[2] = 1; U1.ne[3] = 1;
        U1.nb[1] = HD*sizeof(float); U1.nb[2] = U1.nb[1]; U1.nb[3] = U1.nb[1];
        U1.data = scratch.ptr; U1.view_src = nullptr;
        if (!ggml_cuda_fattn_dec_gate(ctx, &F1, gate + t*HD, &U1)) {
            fprintf(stderr, "fa_nq verify %s: t=%d n_kv_t=%d: N=1 path not taken (generic), skipped\n", fa->name, t, n_kv_t);
            continue;
        }
        CUDA_CHECK(cudaStreamSynchronize(st));
        CUDA_CHECK(cudaMemcpy(ref.data(), scratch.ptr, HD*sizeof(float), cudaMemcpyDeviceToHost));
        size_t d_out = 0, d_q8 = 0;
        for (int64_t e = 0; e < HD; ++e) {
            d_out += memcmp(&ref[e], &got[t*HD + e], sizeof(float)) != 0;
        }
        if (with_q8) {
            block_q8_1 * q8 = nullptr; int * sums = nullptr;
            GGML_ASSERT(ggml_cuda_mmvq_cache_produce(ctx, &U1, &q8, &sums));
            CUDA_CHECK(cudaMemcpy(ref_q8.data(), q8, nblk*sizeof(block_q8_1), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(ref_s.data(), sums, nblk*sizeof(int), cudaMemcpyDeviceToHost));
            for (int64_t b = 0; b < nblk; ++b) {
                d_q8 += memcmp(&ref_q8[b], &got_q8[t*nblk + b], sizeof(block_q8_1)) != 0 || ref_s[b] != got_s[t*nblk + b];
            }
        }
        fprintf(stderr, "fa_nq verify %s: N=%d t=%d n_kv=%d n_kv_t=%d: out %zu/%lld differ, q8 %zu/%lld differ%s\n",
            fa->name, a.N, t, a.n_kv, n_kv_t, d_out, (long long) HD, d_q8, (long long) (with_q8 ? nblk : 0),
            d_out || d_q8 ? "  <-- MISMATCH" : "");
    }
    // restore the N-query results (output and the cached q8 of mul_node)
    block_q8_1 * q8 = nullptr; int * sums = nullptr;
    if (!with_q8 || !ggml_cuda_mmvq_cache_produce(ctx, mul_node, &q8, &sums)) {
        q8 = nullptr; sums = nullptr;
    }
    fdec_nq_args a2 = a;
    a2.gate = gate;   // the saved gate (the output may overwrite the gate tensor in place)
    fdec_nq_run(ctx, a2, counters, q8, sums);
    CUDA_CHECK(cudaStreamSynchronize(st));
}

// Long KV (batch n_kv > GGML_CUDA_FA_DEC_MAX_KV): the single-token graph of query t runs the generic flash attention
// (one query, K/V views of length n_kv_t) with the gate + q8 epilogue in its combine kernel. Here every query runs
// exactly that call on single-query views. Its n_kv_t is only known on the device (mask row t; graph capture), but for
// N <= 16 consecutive cells it is n_kv or n_kv - 256 (c_hi / c_lo):
//  - the K/V view length only matters through the launch configuration (cells past the query are masked: their scores
//    are -inf, P = 0, the running max and the scale factor 1 are unchanged, and the KV tiles of a parallel block are
//    interleaved, so the visible tiles of each block are the same): if a dry run gives the same configuration for both
//    lengths, one call per query with length n_kv is the single-token call of either length;
//  - otherwise both lengths run into scratch rows (and c_lo <= max_kv queries through the N-query kernel above with
//    lim = c_lo) and a select kernel copies each query's row from its own n_kv_t.
// The epilogue's q8_1 goes to the per-stream cache under single-row views and is dropped afterwards: the following
// N-column GEMV quantizes the gated rows itself (bit-identical to the producer's q8, see q8-producers.cu).
static __global__ void fdec_nq_select_kernel(const half * __restrict__ mask, const int mask_s1, const int n_kv, const int lim,
        const int c_lo, const float * __restrict__ src_hi, const float * __restrict__ src_lo, float * __restrict__ out, const int HD) {
    const int t = blockIdx.x;
    const half * m = mask + (int64_t) t*mask_s1;
    __shared__ int s_nb;
    int nb = 0;
    for (int i = threadIdx.x; i < n_kv; i += blockDim.x) {
        if (__half2float(m[i]) != -INFINITY) {
            nb = i/256 + 1;
        }
    }
    if (threadIdx.x == 0) {
        s_nb = 0;
    }
    __syncthreads();
    if (nb > 0) {
        atomicMax(&s_nb, nb);
    }
    __syncthreads();
    const int n_kv_t = min(n_kv, max(256, s_nb*256));
    if (n_kv_t <= lim) {
        return;   // written by the N-query kernel
    }
    // c_hi, or (non-consecutive cells, never in a verification batch) the closest available length
    const float * src = n_kv_t == c_lo && src_lo != nullptr ? src_lo : src_hi;
    for (int e = threadIdx.x; e < HD; e += blockDim.x) {
        out[(int64_t) t*HD + e] = src[(int64_t) t*HD + e];
    }
}

struct fdec_q1_views {
    ggml_tensor Q1, K1, V1, M1, F1, U1;
};

// single-query views of query t with K/V length L: FA output (raw, unused afterwards) into fa_out, gated row into out
static void fdec_q1_make(const ggml_tensor * fa, const ggml_tensor * mul_node, const int t, const int L, float * fa_out,
        float * out, fdec_q1_views & v) {
    const ggml_tensor * Q    = fa->src[0];
    const ggml_tensor * mask = fa->src[3];
    const int64_t HD = Q->ne[0]*Q->ne[2];
    v.Q1 = *Q; v.K1 = *fa->src[1]; v.V1 = *fa->src[2]; v.M1 = *mask; v.F1 = *fa; v.U1 = *mul_node;
    v.Q1.ne[1] = 1; v.Q1.data = (char *) Q->data + t*Q->nb[1]; v.Q1.view_src = nullptr;
    v.K1.ne[1] = L; v.V1.ne[1] = L; v.K1.view_src = nullptr; v.V1.view_src = nullptr;
    v.M1.ne[0] = L; v.M1.ne[1] = 1; v.M1.data = (char *) mask->data + t*mask->nb[1]; v.M1.view_src = nullptr;
    v.M1.nb[2] = v.M1.nb[1]; v.M1.nb[3] = v.M1.nb[1];
    v.F1.src[0] = &v.Q1; v.F1.src[1] = &v.K1; v.F1.src[2] = &v.V1; v.F1.src[3] = &v.M1;
    v.F1.ne[2] = 1; v.F1.ne[3] = 1; v.F1.nb[3] = v.F1.nb[2]; v.F1.data = fa_out; v.F1.view_src = nullptr;
    v.U1.ne[0] = HD; v.U1.ne[1] = 1; v.U1.ne[2] = 1; v.U1.ne[3] = 1;
    v.U1.nb[1] = HD*sizeof(float); v.U1.nb[2] = v.U1.nb[1]; v.U1.nb[3] = v.U1.nb[1];
    v.U1.data = out; v.U1.view_src = nullptr;
}

// the generic single-token call (FA + combine with the gate epilogue) of query t at length L; dry: configuration only.
// Returns false if that call would not apply the gate in its combine (then the caller declines before any launch).
static bool fdec_q1_generic(ggml_backend_cuda_context & ctx, const ggml_tensor * fa, const float * gate, const ggml_tensor * mul_node,
        const int t, const int L, float * fa_out, float * out, const bool dry, uint64_t * cfg) {
    const int64_t HD = fa->src[0]->ne[0]*fa->src[0]->ne[2];
    fdec_q1_views v;
    fdec_q1_make(fa, mul_node, t, L, fa_out, out, v);
    ggml_cuda_fattn_gate_epilogue ep;
    ep.ctx      = &ctx;
    ep.gate     = gate + t*HD;
    ep.mul_node = &v.U1;
    ep.dry_run  = dry;
    ggml_cuda_fattn_gate_epilogue * prev = ggml_cuda_fattn_epilogue;
    ggml_cuda_fattn_epilogue = &ep;
    ggml_cuda_flash_attn_ext(ctx, &v.F1);
    ggml_cuda_fattn_epilogue = prev;
    if (dry) {
        memcpy(cfg, ep.cfg, sizeof(ep.cfg));
        // recorded, combine step with the epilogue: parallel blocks > 1, no stream-k / mask scan / sparse / kv_rows
        return ep.cfg[6] == 1 && (ep.cfg[2] & 0xffffffffu) > 1 && ep.cfg[5] == 0;
    }
    return ep.applied;
}

static bool fdec_nq_long(ggml_backend_cuda_context & ctx, const ggml_tensor * fa, const float * gate, ggml_tensor * mul_node,
        const fdec_nq_args & a0, int * counters, const int max_kv, const bool verify) {
    const ggml_tensor * Q = fa->src[0];
    const int     N    = a0.N;
    const int64_t HD   = Q->ne[0]*Q->ne[2];
    const int     c_hi = a0.n_kv;
    const int     c_lo = a0.n_kv - 256;
    const bool lo_dec  = c_lo <= max_kv;   // the c_lo queries run fattn-dec in their single-token graph
    cudaStream_t st = ctx.stream();

    ggml_cuda_pool_alloc<float> fa_tmp(ctx.pool(), (size_t) HD);
    uint64_t cfg_hi[8], cfg_lo[8];
    if (!fdec_q1_generic(ctx, fa, gate, mul_node, 0, c_hi, fa_tmp.ptr, fa_tmp.ptr, true, cfg_hi)) {
        return false;
    }
    bool single = false;
    if (!lo_dec) {
        if (!fdec_q1_generic(ctx, fa, gate, mul_node, 0, c_lo, fa_tmp.ptr, fa_tmp.ptr, true, cfg_lo)) {
            return false;
        }
        // GGML_CUDA_FA_DEC_NQ_SELECT=1 (debug): always the two-length + select form
        static const bool force_select = fdec_env_int("GGML_CUDA_FA_DEC_NQ_SELECT", 0) != 0;
        single = !force_select && memcmp(cfg_hi, cfg_lo, 6*sizeof(uint64_t)) == 0;
    }
    // verify: keep the gate (the gated output may overwrite it in place)
    ggml_cuda_pool_alloc<float> gsave(ctx.pool());
    if (verify) {
        gsave.alloc((size_t) N*HD);
        CUDA_CHECK(cudaMemcpyAsync(gsave.ptr, gate, (size_t) N*HD*sizeof(float), cudaMemcpyDeviceToDevice, st));
    }
    float * out = (float *) mul_node->data;
    if (single) {
        for (int t = 0; t < N; ++t) {
            GGML_ASSERT(fdec_q1_generic(ctx, fa, gate, mul_node, t, c_hi, fa_tmp.ptr, out + t*HD, false, nullptr));
        }
    } else {
        // every query reads its gate row before any gated row is written (the select / N-query writes come last)
        ggml_cuda_pool_alloc<float> s_hi(ctx.pool(), (size_t) N*HD);
        ggml_cuda_pool_alloc<float> s_lo(ctx.pool());
        if (!lo_dec) {
            s_lo.alloc((size_t) N*HD);
        }
        for (int t = 0; t < N; ++t) {
            GGML_ASSERT(fdec_q1_generic(ctx, fa, gate, mul_node, t, c_hi, fa_tmp.ptr, s_hi.ptr + t*HD, false, nullptr));
            if (!lo_dec) {
                GGML_ASSERT(fdec_q1_generic(ctx, fa, gate, mul_node, t, c_lo, fa_tmp.ptr, s_lo.ptr + t*HD, false, nullptr));
            }
        }
        int lim = 0;
        if (lo_dec) {
            fdec_nq_args a = a0;
            a.lim = c_lo;
            GGML_ASSERT(fdec_nq_run(ctx, a, counters, nullptr, nullptr));
            lim = c_lo;
        }
        fdec_nq_select_kernel<<<N, 256, 0, st>>>(a0.mask, a0.mask_s1, a0.n_kv, lim, c_lo, s_hi.ptr, lo_dec ? nullptr : s_lo.ptr,
            out, (int) HD);
        CUDA_CHECK(cudaGetLastError());
    }
    // the epilogue's q8_1 blocks of the single-row views are not the N-row activation: drop them
    ggml_cuda_mmvq_cache_reset(ctx);

    if (verify) {
        // reference per query: its single-token call at its own n_kv_t (host, from the mask row)
        const ggml_tensor * mask = fa->src[3];
        CUDA_CHECK(cudaStreamSynchronize(st));
        std::vector<float> got((size_t) N*HD), ref((size_t) HD);
        CUDA_CHECK(cudaMemcpy(got.data(), out, got.size()*sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<ggml_fp16_t> mrow((size_t) a0.n_kv);
        ggml_cuda_pool_alloc<float> r1(ctx.pool(), (size_t) HD);
        for (int t = 0; t < N; ++t) {
            CUDA_CHECK(cudaMemcpy(mrow.data(), (const char *) mask->data + t*mask->nb[1], a0.n_kv*sizeof(ggml_fp16_t), cudaMemcpyDeviceToHost));
            int last = -1;
            for (int i = 0; i < a0.n_kv; ++i) {
                if (GGML_FP16_TO_FP32(mrow[i]) != -INFINITY) {
                    last = i;
                }
            }
            const int n_kv_t = std::min(a0.n_kv, std::max(256, (int) GGML_PAD(last + 1, 256)));
            const char * how = "generic";
            if (n_kv_t <= max_kv) {
                fdec_q1_views v;
                fdec_q1_make(fa, mul_node, t, n_kv_t, fa_tmp.ptr, r1.ptr, v);
                how = "fattn-dec";
                if (!ggml_cuda_fattn_dec_gate(ctx, &v.F1, gsave.ptr + t*HD, &v.U1)) {
                    fprintf(stderr, "fa_nq verify %s: t=%d n_kv_t=%d: N=1 path not taken, skipped\n", fa->name, t, n_kv_t);
                    continue;
                }
            } else if (!fdec_q1_generic(ctx, fa, gsave.ptr, mul_node, t, n_kv_t, fa_tmp.ptr, r1.ptr, false, nullptr)) {
                fprintf(stderr, "fa_nq verify %s: t=%d n_kv_t=%d: no gate epilogue, skipped\n", fa->name, t, n_kv_t);
                continue;
            }
            CUDA_CHECK(cudaStreamSynchronize(st));
            CUDA_CHECK(cudaMemcpy(ref.data(), r1.ptr, HD*sizeof(float), cudaMemcpyDeviceToHost));
            size_t d_out = 0;
            for (int64_t e = 0; e < HD; ++e) {
                d_out += memcmp(&ref[e], &got[t*HD + e], sizeof(float)) != 0;
            }
            fprintf(stderr, "fa_nq verify %s: long N=%d t=%d n_kv=%d n_kv_t=%d (%s, %s): out %zu/%lld differ%s\n",
                fa->name, N, t, a0.n_kv, n_kv_t, how, single ? "single" : "select", d_out, (long long) HD,
                d_out ? "  <-- MISMATCH" : "");
        }
        ggml_cuda_mmvq_cache_reset(ctx);
    }
    return true;
}

bool ggml_cuda_fattn_dec_gate_nq(ggml_backend_cuda_context & ctx, const ggml_tensor * fa, const float * gate, ggml_tensor * mul_node) {
    // the same switches as the N=1 kernel: the partition of query t must be the one its single-token call uses
    static const int max_kv = fdec_env_int("GGML_CUDA_FA_DEC_MAX_KV", 1024);
    static const int S      = fdec_env_int("GGML_CUDA_FA_DEC_S", 32);
    static int verify_left  = fdec_env_int("GGML_CUDA_FA_DEC_NQ_VERIFY", 0);
    if (S != 16 && S != 32) {
        return false;
    }
    const ggml_tensor * Q    = fa->src[0];
    const ggml_tensor * K    = fa->src[1];
    const ggml_tensor * V    = fa->src[2];
    const ggml_tensor * mask = fa->src[3];
    if (fa->src[4] != nullptr || fa->src[5] != nullptr || mask == nullptr) {
        return false;
    }
    float scale, max_bias, logit_softcap;
    memcpy(&scale,         (const float *) fa->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) fa->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) fa->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    const int64_t D    = Q->ne[0];
    const int64_t N    = Q->ne[1];
    const int64_t n_kv = K->ne[1];
    // the conditions of ggml_cuda_fattn_dec_gate for every query (its n_kv_t <= n_kv is a multiple of 256), plus N
    if (!ggml_cuda_fattn_dec_nq_wanted(N) ||
        Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 ||
        fa->type != GGML_TYPE_F32 || !ggml_is_contiguous(fa) || !ggml_is_contiguous(mul_node) ||
        D != 256 || K->ne[0] != D || V->ne[0] != D || V->ne[1] != n_kv ||
        Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 || V->ne[2] != K->ne[2] ||
        Q->ne[2] % K->ne[2] != 0 || K->ne[2] > 64 ||
        mask->ne[0] < n_kv || mask->ne[1] < N || mask->ne[2] != 1 || mask->ne[3] != 1 ||
        n_kv % 256 != 0 || n_kv < 256 ||
        ggml_nelements(fa) != D*Q->ne[2]*N || ggml_nelements(mul_node) != D*Q->ne[2]*N) {
        return false;
    }
    if ((uintptr_t) Q->data % 16 || Q->nb[0] != sizeof(float) || Q->nb[1] % 16 || Q->nb[2] % 16 ||
        (uintptr_t) K->data % 16 || K->nb[0] != sizeof(half) || K->nb[1] % 16 || K->nb[2] % 16 ||
        (uintptr_t) V->data % 16 || V->nb[0] != sizeof(half) || V->nb[1] % 16 || V->nb[2] % 16 ||
        mask->nb[0] != sizeof(half) || mask->nb[1] % sizeof(half)) {
        return false;
    }
    // MTP2-merge: the combine of one (kv head, query) group writes its gated rows while workgroups of other groups may still
    // read Q, K, V, the mask and the gate (the grid is not co-resident): the output may only coincide exactly with the gate
    // or with a contiguous Q when N == 1 or H == 1 (the kernel reads Q as [D, N, H]; only then does that match the output
    // layout [D, H, N], so only the workgroups of that group read those elements), never overlap otherwise
    {
        const auto ovl = [](const void * a, size_t na, const void * b, size_t nb) {
            const char * a0 = (const char *) a; const char * b0 = (const char *) b;
            return a0 < b0 + nb && b0 < a0 + na;
        };
        const void * md = mul_node->data;
        const size_t mn = ggml_nbytes(mul_node);
        const size_t gn = (size_t) ggml_nelements(mul_node)*sizeof(float);
        if (ovl(md, mn, K->data, ggml_nbytes(K)) || ovl(md, mn, V->data, ggml_nbytes(V)) ||
            ovl(md, mn, mask->data, ggml_nbytes(mask)) ||
            (ovl(md, mn, gate, gn) && md != (const void *) gate) ||
            (ovl(md, mn, Q->data, ggml_nbytes(Q)) && (md != Q->data || !ggml_is_contiguous(Q) || ggml_nbytes(Q) != mn || (N != 1 && Q->ne[2] != 1)))) {
            return false;
        }
    }
    if (K->nb[2]*K->ne[2] > INT_MAX/2 || V->nb[2]*V->ne[2] > INT_MAX/2 || (int64_t) n_kv*std::max(K->nb[1], V->nb[1]) > INT_MAX/2 ||
        Q->nb[1]*N > INT_MAX/2 || Q->nb[2]*Q->ne[2] > INT_MAX/2 || mask->nb[1]*N > INT_MAX/2) {
        return false;
    }
    const int G = Q->ne[2] / K->ne[2];
    if (G != 4 && G != 6 && G != 8) {
        return false;
    }
    const int n_kvh = K->ne[2];
    // the N=1 kernel needs its whole grid co-resident; where it declines, the single-token graph takes the generic path
    static const bool concurrent_streams = getenv("GGML_CUDA_GRAPH_OPT") != nullptr && atoi(getenv("GGML_CUDA_GRAPH_OPT")) != 0;
    if (n_kvh > FDEC_MAX_KVH || concurrent_streams || FDEC_MAX_SPLIT*n_kvh > ggml_cuda_info().devices[ctx.device].nsm) {
        return false;
    }
    int * counters = fdec_nq_counters(ctx);
    if (counters == nullptr) {
        return false;
    }
    fdec_nq_args a;
    a.Q = (const float *) Q->data; a.K = (const char *) K->data; a.V = (const char *) V->data; a.mask = (const half *) mask->data;
    a.gate_out = (float *) mul_node->data; a.gate = gate;
    a.n_kvh = n_kvh; a.G = G; a.S = S; a.N = (int) N; a.n_kv = (int) n_kv; a.lim = (int) n_kv;
    a.mask_s1 = (int) (mask->nb[1] / sizeof(half));
    a.nbq1 = (int) Q->nb[1]; a.nbq2 = (int) Q->nb[2];
    a.nb11 = (int) K->nb[1]; a.nb12 = (int) K->nb[2]; a.nb21 = (int) V->nb[1]; a.nb22 = (int) V->nb[2];
    a.scale = scale;
    if (n_kv > max_kv) {
        // GGML_CUDA_FA_DEC_NQ_LONG=0: long KV not taken (the generic N-query FLASH_ATTN_EXT + separate gate, as before)
        static const bool long_on = fdec_env_int("GGML_CUDA_FA_DEC_NQ_LONG", 1) != 0;
        if (!long_on) {
            return false;
        }
        // long KV: the queries of the single-token generic path (and, at the 256-boundary above max_kv, this kernel)
        static int verify_long_left = fdec_env_int("GGML_CUDA_FA_DEC_NQ_VERIFY", 0);
        const bool vl = verify_long_left > 0;
        verify_long_left -= vl;
        return fdec_nq_long(ctx, fa, gate, mul_node, a, counters, max_kv, vl);
    }
    // q8_1 of the N gated rows for the following N-column GEMV (only if that GEMV takes the N-column route)
    const bool want_q8 = N <= ggml_cuda_mmvq_nc_max();
    block_q8_1 * q8      = nullptr;
    int        * q8_sums = nullptr;
    if (!want_q8 || !ggml_cuda_mmvq_cache_produce(ctx, mul_node, &q8, &q8_sums)) {
        q8      = nullptr;
        q8_sums = nullptr;
    }
    // verify: the gated output may be allocated in place over the gate tensor: keep a copy of the gate first
    const bool verify = verify_left > 0;
    ggml_cuda_pool_alloc<float> gsave(ctx.pool());
    if (verify) {
        verify_left--;
        gsave.alloc((size_t) N*Q->ne[2]*D);
        CUDA_CHECK(cudaMemcpyAsync(gsave.ptr, gate, (size_t) N*Q->ne[2]*D*sizeof(float), cudaMemcpyDeviceToDevice, ctx.stream()));
    }
    if (!fdec_nq_run(ctx, a, counters, q8, q8_sums)) {
        return false;
    }
    if (verify) {
        fdec_nq_verify(ctx, fa, gsave.ptr, mul_node, a, counters, q8 != nullptr);
    }
    return true;
}
// ------------------------------------------------------------------------------------------------- MTP2-O2 end

#else

bool ggml_cuda_fattn_dec_gate(ggml_backend_cuda_context &, const ggml_tensor *, const float *, ggml_tensor *) {
    return false;
}

bool ggml_cuda_fattn_dec_nq_wanted(const int64_t) {
    return false;
}

bool ggml_cuda_fattn_dec_gate_nq(ggml_backend_cuda_context &, const ggml_tensor *, const float *, ggml_tensor *) {
    return false;
}

#endif // defined(GGML_USE_HIP)
