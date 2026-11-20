#include "argsort.cuh"
#include "top-k.cuh"

// Adjusted implementation thresholds from #28547, can be overridden at build time
#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
// not measured on HIP/MUSA, keep the old split
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 1024
#    else
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 512
#    endif
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC

#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT
#    define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT 4096
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT

// bitonic up to this width while nrows fits in one wave of SMs, 0 disables
#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 0
#    else
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 1024
#    endif
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
// DeviceTopK has a race condition before CCCL 3.4.3.
// https://github.com/NVIDIA/cccl/pull/10627
#    if (CCCL_MAJOR_VERSION > 3 || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION > 4) || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION == 4 && CCCL_PATCH_VERSION >= 3))
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL >= 3.4.3
#endif      // GGML_CUDA_USE_CUB

// max rows for the per-row DeviceTopK / CUB argsort path before switching to radix / bitonic
#ifndef GGML_CUDA_TOP_K_NROWS_THRESHOLD
#    ifdef CUB_TOP_K_AVAILABLE
#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 2
#    else
#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 1
#    endif
#endif // GGML_CUDA_TOP_K_NROWS_THRESHOLD

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#endif                            // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int64_t col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int64_t col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = (int) std::min<int64_t>(((int64_t) ncols + 1023) / 1024, 64);

    // chunk the rows to bound the histogram memory to 64 MB
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) blocks_per_row * NBINS * sizeof(int), nrows);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, chunk_nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) chunk_nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        top_k_radix_init<<<(iter_nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, iter_nrows, k);

        const dim3 row_grid(blocks_per_row * iter_nrows);
        for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
            top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
                <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                    src, states, histograms, ncols, blocks_per_row, shift);
            top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
                <<<iter_nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
        }

        top_k_radix_reset_counters
            <<<(iter_nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, iter_nrows);
        top_k_radix_gather<BLOCK_SIZE>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, dst, states, ncols, k, blocks_per_row);

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
}

// ---- TOPK-FAST: one launch for a few long rows and a small k (the backend top-k sampler of an MTP draft head: one row
// of 248320 logits, k = 10). Total order on (value, index): the larger value first (ordered-integer key), equal values:
// the lower index first, so the result is deterministic; the k indices are written in that order (a valid "unsorted"
// top-k). Every wave reads 1024 consecutive elements and writes its maximum (64-bit key = value key << 32 | ~index).
// The last workgroup of a row (atomic counter, reset afterwards: graph replay safe) takes the k largest wave maxima: the
// k-th of them, T, is <= the k-th largest element, so the top k are among the elements >= T of those k waves. They are
// gathered into LDS and the top k selected among them by k rounds of a wave arg-max. If more than TOPK_FAST_CMAX
// elements pass (many equal values) each of the k waves is reduced to its own top k first (exact as well).
// Replaces the 10-launch radix path (~95 us on MI210 for 248320 x 1). GGML_CUDA_TOP_K_FAST=0 disables.
// Note: this applies to every TOP_K with ncols >= 8192, k <= 32, nrows <= 64 (not only the MTP draft head). The set of
// indices is the same, but their order (value desc, index asc) and the choice among equal values at the k-th place can
// differ from the radix path, so a backend sampler chain that walks the candidates in order (e.g. top_k then dist
// with temperature > 0) can draw other tokens than before for the same seed. Greedy results are unaffected.
#if defined(GGML_USE_HIP)
#include <mutex>
#include <map>
#define TOPK_FAST_WS    64                     // wave64 (CDNA / GCN) only
#define TOPK_FAST_EPL   16                     // elements per lane
#define TOPK_FAST_EPW   (TOPK_FAST_EPL*TOPK_FAST_WS)
#define TOPK_FAST_WPB   4                      // waves per workgroup
#define TOPK_FAST_NT    (TOPK_FAST_WPB*TOPK_FAST_WS)
#define TOPK_FAST_KMAX  32
#define TOPK_FAST_WMAXL 8                      // wave maxima per lane in the last workgroup: waves per row <= 512
#define TOPK_FAST_CMAX  1024                   // gathered candidates (>= TOPK_FAST_KMAX^2 for the fallback)
#define TOPK_FAST_RMAX  64                     // rows (counters)

static __device__ __forceinline__ unsigned long long top_k_fast_key(const float v, const int idx) {
    return ((unsigned long long) top_k_float_to_ordered(v) << 32) | (unsigned long long) (uint32_t) ~(uint32_t) idx;
}
static __device__ __forceinline__ int top_k_fast_idx(const unsigned long long key) {
    return (int) ~(uint32_t) (key & 0xFFFFFFFFull);
}
// wave maximum of a 32-bit value (DPP within rows of 16 lanes, then the four row results by readlane), uniform result
static __device__ __forceinline__ uint32_t top_k_fast_wmax32(uint32_t v) {
    v = max(v, (uint32_t) __builtin_amdgcn_mov_dpp((int) v, 0xB1,  0xF, 0xF, false));   // quad_perm 1,0,3,2
    v = max(v, (uint32_t) __builtin_amdgcn_mov_dpp((int) v, 0x4E,  0xF, 0xF, false));   // quad_perm 2,3,0,1
    v = max(v, (uint32_t) __builtin_amdgcn_mov_dpp((int) v, 0x141, 0xF, 0xF, false));   // row_half_mirror
    v = max(v, (uint32_t) __builtin_amdgcn_mov_dpp((int) v, 0x140, 0xF, 0xF, false));   // row_mirror
    const uint32_t a = (uint32_t) __builtin_amdgcn_readlane((int) v, 0),  b = (uint32_t) __builtin_amdgcn_readlane((int) v, 16);
    const uint32_t c = (uint32_t) __builtin_amdgcn_readlane((int) v, 32), d = (uint32_t) __builtin_amdgcn_readlane((int) v, 48);
    return max(max(a, b), max(c, d));
}
// wave maximum of the 64-bit keys: the largest value key, then among those the largest low word (= the lowest index)
static __device__ __forceinline__ unsigned long long top_k_fast_wave_max(const unsigned long long x) {
    const uint32_t hi = (uint32_t) (x >> 32);
    const uint32_t mh = top_k_fast_wmax32(hi);
    const uint32_t ml = top_k_fast_wmax32(hi == mh ? (uint32_t) x : 0u);
    return ((unsigned long long) mh << 32) | ml;
}
// the k largest of the keys v[] of all lanes (0 = empty; keys are unique): lane r < k returns the r-th, in order
template <int NPL>
static __device__ __forceinline__ unsigned long long top_k_fast_wave_select(unsigned long long (&v)[NPL], const int k, const int lane) {
    auto lmax = [&]() {
        unsigned long long m = v[0];
#pragma unroll
        for (int j = 1; j < NPL; ++j) m = v[j] > m ? v[j] : m;
        return m;
    };
    unsigned long long lb = lmax(), mine = 0;
    for (int r = 0; r < k; ++r) {
        const unsigned long long wb = top_k_fast_wave_max(lb);
        if (lane == r) mine = wb;
        if (lb == wb) {   // the owner lane drops it
#pragma unroll
            for (int j = 0; j < NPL; ++j) v[j] = v[j] == wb ? 0ull : v[j];
            lb = lmax();
        }
    }
    return mine;
}
// the 16 keys of lane `lane` of the wave chunk w (padding beyond the row: key 0x80000000, below every real element)
static __device__ __forceinline__ void top_k_fast_load(const float * __restrict__ rs, const int ncols, const int w, const int lane,
        unsigned long long (&v)[TOPK_FAST_EPL]) {
#pragma unroll
    for (int j = 0; j < TOPK_FAST_EPL; ++j) {
        const int i = w*TOPK_FAST_EPW + j*TOPK_FAST_WS + lane;
        v[j] = i < ncols ? top_k_fast_key(rs[i], i) : (unsigned long long) 0x80000000u;
    }
}
// select from the C keys in LDS list[] with one wave; lane r < k writes the r-th index
template <int NPL>
static __device__ __forceinline__ void top_k_fast_final(const unsigned long long * list, const int C, const int k, const int lane, int * __restrict__ out) {
    unsigned long long v[NPL];
#pragma unroll
    for (int j = 0; j < NPL; ++j) v[j] = lane + TOPK_FAST_WS*j < C ? list[lane + TOPK_FAST_WS*j] : 0ull;
    const unsigned long long m = top_k_fast_wave_select<NPL>(v, k, lane);
    if (lane < k) out[lane] = top_k_fast_idx(m);
}

static __global__ void __launch_bounds__(TOPK_FAST_NT) top_k_fast_kernel(
        const float * __restrict__ src, int * __restrict__ dst, unsigned long long * __restrict__ wmax,
        unsigned int * __restrict__ cnt, const int ncols, const int k, const int wpr) {
    const int row  = blockIdx.y;
    const int lane = threadIdx.x % TOPK_FAST_WS;
    const int wl   = threadIdx.x / TOPK_FAST_WS;
    const int wave = blockIdx.x*TOPK_FAST_WPB + wl;
    const float * rs = src + (size_t) row*ncols;
    unsigned long long * rw = wmax + (size_t) row*wpr;

    if (wave < wpr) {
        unsigned long long v[TOPK_FAST_EPL];
        top_k_fast_load(rs, ncols, wave, lane, v);
        unsigned long long m = v[0];
#pragma unroll
        for (int j = 1; j < TOPK_FAST_EPL; ++j) m = v[j] > m ? v[j] : m;
        m = top_k_fast_wave_max(m);
        if (lane == 0) rw[wave] = m;
    }
    __threadfence();
    __syncthreads();
    __shared__ unsigned int last;
    if (threadIdx.x == 0) {
        last = atomicAdd(cnt + row, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) {
        return;
    }
    __threadfence();

    // (a) the k largest wave maxima (every wave computes the same, no barrier); lane r of each wave holds the r-th
    unsigned long long sel;
    {
        unsigned long long v[TOPK_FAST_WMAXL];
#pragma unroll
        for (int j = 0; j < TOPK_FAST_WMAXL; ++j) {
            const int i = lane + TOPK_FAST_WS*j;
            v[j] = i < wpr ? __hip_atomic_load(rw + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) : 0ull;
        }
        sel = top_k_fast_wave_select<TOPK_FAST_WMAXL>(v, k, lane);
    }
    const unsigned long long T = __shfl(sel, k - 1, TOPK_FAST_WS);

    // (b) gather the elements >= T of the k selected waves (all k*1024 loads of a thread independent)
    __shared__ unsigned long long list[TOPK_FAST_CMAX];
    __shared__ unsigned int lcnt;
    __shared__ int selw[TOPK_FAST_KMAX];
    if (threadIdx.x == 0) lcnt = 0;
    if (wl == 0 && lane < k) selw[lane] = top_k_fast_idx(sel) / TOPK_FAST_EPW;
    __syncthreads();
#pragma unroll 8
    for (int j = (int) threadIdx.x; j < k*TOPK_FAST_EPW; j += TOPK_FAST_NT) {
        const int e = selw[j / TOPK_FAST_EPW]*TOPK_FAST_EPW + j % TOPK_FAST_EPW;
        if (e < ncols) {
            const unsigned long long key = top_k_fast_key(rs[e], e);
            if (key >= T) {
                const unsigned int p = atomicAdd(&lcnt, 1u);
                if (p < TOPK_FAST_CMAX) list[p] = key;
            }
        }
    }
    __syncthreads();
    const int C = (int) lcnt;
    int * out = dst + (size_t) row*k;
    if (C > TOPK_FAST_CMAX) {
        // fallback (many equal values): the top k of each selected wave, then the top k of those k*k
        __syncthreads();
        for (int r = wl; r < k; r += TOPK_FAST_WPB) {
            const int w = selw[r];
            unsigned long long v[TOPK_FAST_EPL];
            top_k_fast_load(rs, ncols, w, lane, v);
            const unsigned long long m = top_k_fast_wave_select<TOPK_FAST_EPL>(v, k, lane);
            if (lane < k) list[r*k + lane] = m;
        }
        __syncthreads();
        if (wl == 0) top_k_fast_final<TOPK_FAST_CMAX/TOPK_FAST_WS>(list, k*k, k, lane, out);
    } else if (wl == 0) {
        if      (C <= 1*TOPK_FAST_WS) top_k_fast_final<1>(list, C, k, lane, out);
        else if (C <= 2*TOPK_FAST_WS) top_k_fast_final<2>(list, C, k, lane, out);
        else if (C <= 4*TOPK_FAST_WS) top_k_fast_final<4>(list, C, k, lane, out);
        else                          top_k_fast_final<TOPK_FAST_CMAX/TOPK_FAST_WS>(list, C, k, lane, out);
    }
    if (threadIdx.x == 0) {
        cnt[row] = 0;
    }
}

static bool top_k_fast_enabled() {
    static const bool on = [] {
        const char * v = getenv("GGML_CUDA_TOP_K_FAST");
        return v == nullptr || v[0] != '0';
    }();
    return on;
}

// zeroed row counters per (device, stream), allocated outside stream capture: launches on different streams (two
// contexts, concurrent graph branches) never share a counter; launches on one stream are ordered, and each kernel
// leaves its counters at zero
static unsigned int * top_k_fast_counters(const int device, cudaStream_t stream) {
    static std::mutex mtx;
    static std::map<std::pair<int, cudaStream_t>, unsigned int *> bufs;
    std::lock_guard<std::mutex> lock(mtx);
    auto it = bufs.find({device, stream});
    if (it == bufs.end()) {
        hipStreamCaptureStatus status;
        CUDA_CHECK(hipStreamIsCapturing(stream, &status));
        if (status != hipStreamCaptureStatusNone) {
            return nullptr;
        }
        unsigned int * p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, TOPK_FAST_RMAX*sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(p, 0, TOPK_FAST_RMAX*sizeof(unsigned int)));
        CUDA_CHECK(cudaDeviceSynchronize());
        it = bufs.emplace(std::make_pair(device, stream), p).first;
    }
    return it->second;
}

static bool top_k_fast_cuda(ggml_cuda_pool & pool, const float * src, int * dst, const int64_t ncols, const int64_t nrows, const int k,
        cudaStream_t stream) {
    if (!top_k_fast_enabled() || ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != TOPK_FAST_WS || k < 1 || k > TOPK_FAST_KMAX ||
            ncols < 8192 || nrows < 1 || nrows > TOPK_FAST_RMAX) {
        return false;
    }
    const int wpr = (int) ((ncols + TOPK_FAST_EPW - 1)/TOPK_FAST_EPW);
    if (wpr > TOPK_FAST_WMAXL*TOPK_FAST_WS || k > wpr) {
        return false;
    }
    unsigned int * cnt = top_k_fast_counters(ggml_cuda_get_device(), stream);
    if (!cnt) {
        return false;
    }
    ggml_cuda_pool_alloc<unsigned long long> wmax(pool, (size_t) nrows*wpr);
    const dim3 grid((wpr + TOPK_FAST_WPB - 1)/TOPK_FAST_WPB, (unsigned int) nrows);
    top_k_fast_kernel<<<grid, TOPK_FAST_NT, 0, stream>>>(src, dst, wmax.get(), cnt, (int) ncols, k, wpr);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
#else
static bool top_k_fast_cuda(ggml_cuda_pool &, const float *, int *, const int64_t, const int64_t, const int, cudaStream_t) {
    return false;
}
#endif // defined(GGML_USE_HIP)

static void top_k_argsort_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, bool use_cub, cudaStream_t stream) {
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) ncols * sizeof(int), nrows);

    ggml_cuda_pool_alloc<int> tmp_alloc(pool, (size_t) ncols * chunk_nrows);
    int * tmp = tmp_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        if (use_cub) {
#ifdef GGML_CUDA_USE_CUB
            argsort_f32_i32_cuda_cub(pool, src, tmp, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
#else
            GGML_ABORT("CUB is not available");
#endif // GGML_CUDA_USE_CUB
        } else {
            argsort_f32_i32_cuda_bitonic(src, tmp, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst, k * sizeof(int), tmp, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();

    const int device = ggml_cuda_get_device();

    if (top_k_fast_cuda(pool, src0_d, dst_d, ncols, nrows, (int) k, stream)) {   // TOPK-FAST
        return;
    }

#ifdef CUB_TOP_K_AVAILABLE
    // a single row always uses DeviceTopK if available
    const bool bitonic_short    = nrows > 1 && ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
#else
    const bool bitonic_short    = ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
#endif // CUB_TOP_K_AVAILABLE
    const bool bitonic_few_rows = nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD &&
                                  ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS &&
                                  nrows <= ggml_cuda_info().devices[device].nsm;

    if (bitonic_short || bitonic_few_rows) {
        // the padded row must fit in shared memory
        const int ncols_pad = next_power_of_2(ncols);
        if (ncols_pad * sizeof(int) <= ggml_cuda_info().devices[device].smpb) {
            top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, false, stream);
            return;
        }
    }

    if (nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }

#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Assess perf of `DeviceBatchedTopK` for multi-row TopK & CCCL >= 3.5.0, re-running perf sweep of https://github.com/ggml-org/llama.cpp/pull/28713
    for (int64_t i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    if (ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT) {
        top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, true, stream);
    } else {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    }
#else                             // GGML_CUDA_USE_CUB
    top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
#endif                            // CUB_TOP_K_AVAILABLE
}
