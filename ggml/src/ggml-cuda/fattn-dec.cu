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

#include <map>
#include <mutex>
#include <utility>

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
    CUDA_CHECK(cudaMalloc(&p, 64*sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(p, 0, 64*sizeof(int), ctx.stream()));
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

#else

bool ggml_cuda_fattn_dec_gate(ggml_backend_cuda_context &, const ggml_tensor *, const float *, ggml_tensor *) {
    return false;
}

#endif // defined(GGML_USE_HIP)
