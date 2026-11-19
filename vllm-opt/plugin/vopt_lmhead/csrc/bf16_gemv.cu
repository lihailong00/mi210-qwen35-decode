// BF16 GEMV for small M on gfx90a (MI210): y[m, n] = sum_k x[m, k] * w[n, k]
// x [M, K] bf16 (row stride ldx), w [N, K] bf16 (row stride ldw), y [M, N] bf16 (contiguous).
// fp32 accumulation, bf16 output with round-to-nearest-even. K = 5120 (compile time J = K/512).
// M = 1..8 (big N: 7..8 rows run as two launches). K = 10240 (MTP fc) with the small-N family only.
//
// Two kernel families:
//  * big N (lm_head 248320 x 5120): persistent grid; one wave owns R full rows at a time, each lane
//    loads J 16-byte chunks per row (all issued together, optionally one row group ahead).
//    x lives either in registers (XREG, loaded once per wave) or in LDS (staged once per block; the
//    LDS address is made opaque per iteration so the compiler cannot hoist the x loads into
//    registers and spill).
//  * small N (in_proj_ba 96 x 5120): S waves split K of one row group; partial sums reduced through
//    LDS; one row group per wave group, no persistence.
// Optional greedy epilogue (ARGMAX): per-wave (value, index) of the bf16-rounded logits, reduced to
// one candidate per block in a workspace; the last block to finish reduces the block candidates to
// the argmax per m (first index on ties, like torch.argmax / vLLM's gumbel argmax) and resets the
// counter, so the op is graph-safe and needs no host sync.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float bflo(uint32_t u) { return __builtin_bit_cast(float, u << 16); }
__device__ __forceinline__ float bfhi(uint32_t u) { return __builtin_bit_cast(float, u & 0xffff0000u); }
__device__ __forceinline__ uint16_t f2bf(float f) {
  uint32_t u = __builtin_bit_cast(uint32_t, f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);  // quiet NaN
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

template <bool NT>
__device__ __forceinline__ u32x4 ldw16(const u32x4* p) {
  if constexpr (NT) return __builtin_nontemporal_load(p);
  else return *p;
}

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int off = 32; off >= 1; off >>= 1) v += __shfl_xor(v, off);
  return v;
}

template <int CTRL>
__device__ __forceinline__ float dppf(float v) {
  return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), CTRL, 0xf, 0xf, true));
}
// full-wave sum, result uniform (scalar): DPP within rows of 16, then 4 readlanes
__device__ __forceinline__ float wave_sum_u(float v) {
  v += dppf<0xb1>(v);   // quad_perm [1,0,3,2]
  v += dppf<0x4e>(v);   // quad_perm [2,3,0,1]
  v += dppf<0x141>(v);  // row_half_mirror
  v += dppf<0x140>(v);  // row_mirror
  const int b = __builtin_bit_cast(int, v);
  return (__builtin_bit_cast(float, __builtin_amdgcn_readlane(b, 0)) +
          __builtin_bit_cast(float, __builtin_amdgcn_readlane(b, 16))) +
         (__builtin_bit_cast(float, __builtin_amdgcn_readlane(b, 32)) +
          __builtin_bit_cast(float, __builtin_amdgcn_readlane(b, 48)));
}

typedef float f2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ f2 up2(uint32_t u) { return (f2){bflo(u), bfhi(u)}; }
// packed-fp32 dot of one 16-byte chunk (8 bf16), started from zero (short dependency chains)
__device__ __forceinline__ f2 dot8p(const u32x4 w, const u32x4 x) {
  f2 a = up2(w.x) * up2(x.x);
  a = __builtin_elementwise_fma(up2(w.y), up2(x.y), a);
  a = __builtin_elementwise_fma(up2(w.z), up2(x.z), a);
  a = __builtin_elementwise_fma(up2(w.w), up2(x.w), a);
  return a;
}

__device__ __forceinline__ float dot8(const u32x4 w, const u32x4 x, float a) {
  a = fmaf(bflo(w.x), bflo(x.x), a); a = fmaf(bfhi(w.x), bfhi(x.x), a);
  a = fmaf(bflo(w.y), bflo(x.y), a); a = fmaf(bfhi(w.y), bfhi(x.y), a);
  a = fmaf(bflo(w.z), bflo(x.z), a); a = fmaf(bfhi(w.z), bfhi(x.z), a);
  a = fmaf(bflo(w.w), bflo(x.w), a); a = fmaf(bfhi(w.w), bfhi(x.w), a);
  return a;
}

// argmax combine: larger value wins, ties -> smaller index; NaN counts as the largest value.
__device__ __forceinline__ void amax_comb(float& v, int& i, float v2, int i2) {
  const bool n1 = v != v, n2 = v2 != v2;
  bool take;
  if (n1 || n2) take = n2 && (!n1 || i2 < i);
  else take = (v2 > v) || (v2 == v && i2 < i);
  if (take) { v = v2; i = i2; }
}

__device__ __forceinline__ void wave_amax(float& v, int& i) {
#pragma unroll
  for (int off = 32; off >= 1; off >>= 1) {
    const float v2 = __shfl_xor(v, off);
    const int i2 = __shfl_xor(i, off);
    amax_comb(v, i, v2, i2);
  }
}

// ---------------------------------------------------------------------------------------------
template <int M, int R, int J, int WPB, bool NT, bool PF, bool XREG, bool ARGMAX>
__global__ __launch_bounds__(64 * WPB) void gemv_bigN(
    const uint16_t* __restrict__ x, int ldx, const uint16_t* __restrict__ w, int64_t ldw,
    uint16_t* __restrict__ y, int N, int ngroups,
    float* __restrict__ ws_val, int* __restrict__ ws_idx, unsigned* __restrict__ counter,
    int64_t* __restrict__ out_idx) {
  constexpr int KC = 64 * J;  // 16-byte chunks per row
  __shared__ u32x4 xs[XREG ? 1 : M * KC];
  const int lane = threadIdx.x & 63;
  const int wv = threadIdx.x >> 6;
  const int gstride = gridDim.x * WPB;
  int g = blockIdx.x * WPB + wv;

  const u32x4* wb = reinterpret_cast<const u32x4*>(w);
  const int64_t ldw4 = ldw / 8;

  u32x4 cur[R][J];
  auto issue = [&](int gg, u32x4 (&dst)[R][J]) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      int row = gg * R + r;
      row = row < N ? row : N - 1;
      const u32x4* p = wb + (int64_t)row * ldw4 + lane;
#pragma unroll
      for (int j = 0; j < J; ++j) dst[r][j] = ldw16<NT>(p + j * 64);
    }
  };
  if (g < ngroups) issue(g, cur);

  u32x4 xr[XREG ? M : 1][XREG ? J : 1];
  if constexpr (XREG) {
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int j = 0; j < J; ++j)
        xr[m][j] = reinterpret_cast<const u32x4*>(x + (int64_t)m * ldx)[j * 64 + lane];
  } else {
    for (int i = threadIdx.x; i < M * KC; i += 64 * WPB) {
      const int m = i / KC, c = i - m * KC;
      xs[i] = reinterpret_cast<const u32x4*>(x + (int64_t)m * ldx)[c];
    }
    __syncthreads();
  }

  float bv[M];
  int bi[M];
#pragma unroll
  for (int m = 0; m < M; ++m) { bv[m] = -__builtin_inff(); bi[m] = 0x7fffffff; }

  for (; g < ngroups; g += gstride) {
    u32x4 nxt[PF ? R : 1][PF ? J : 1];
    const int gn = g + gstride;
    if constexpr (PF) {
      if (gn < ngroups) issue(gn, nxt);
    }
    f2 acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int m = 0; m < M; ++m) acc[r][m] = (f2){0.f, 0.f};
    int xo = lane;
    if constexpr (!XREG) asm volatile("" : "+v"(xo));  // opaque: keep x loads inside the loop
#pragma unroll
    for (int j = 0; j < J; ++j) {
#pragma unroll
      for (int m = 0; m < M; ++m) {
        u32x4 xv;
        if constexpr (XREG) xv = xr[m][j];
        else xv = xs[m * KC + j * 64 + xo];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r][m] += dot8p(cur[r][j], xv);
      }
      // LDS mode: keep only one j-step of x live (otherwise the scheduler issues all M*J LDS
      // loads up front and spills)
      // pin the accumulators here so their FMAs are not sunk past later j steps
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m) asm volatile("" : "+v"(acc[r][m]));
      if constexpr (!XREG) __builtin_amdgcn_sched_barrier(0);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int row = g * R + r;
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float s = wave_sum_u(acc[r][m].x + acc[r][m].y);
        const uint16_t o = f2bf(s);
        if (lane == 0 && row < N) y[(int64_t)m * N + row] = o;
        if constexpr (ARGMAX) {
          if (row < N) amax_comb(bv[m], bi[m], __builtin_bit_cast(float, (uint32_t)o << 16), row);
        }
      }
    }
    if constexpr (PF) {
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < J; ++j) cur[r][j] = nxt[r][j];
    } else {
      if (gn < ngroups) issue(gn, cur);
    }
  }

  if constexpr (ARGMAX) {
    __shared__ float sv[WPB][M];
    __shared__ int si[WPB][M];
    __shared__ bool last;
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < M; ++m) { sv[wv][m] = bv[m]; si[wv][m] = bi[m]; }
    }
    __syncthreads();
    if (threadIdx.x < M) {
      const int m = threadIdx.x;
      float v = sv[0][m];
      int i = si[0][m];
      for (int q = 1; q < WPB; ++q) amax_comb(v, i, sv[q][m], si[q][m]);
      ws_val[(int64_t)blockIdx.x * M + m] = v;
      ws_idx[(int64_t)blockIdx.x * M + m] = i;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(counter, 1u) == gridDim.x - 1;
    __syncthreads();
    if (last) {
      __threadfence();
      if (wv < M) {
        const int m = wv;
        float v = -__builtin_inff();
        int i = 0x7fffffff;
        for (int b = lane; b < (int)gridDim.x; b += 64) {
          const float v2 = __builtin_nontemporal_load(ws_val + (int64_t)b * M + m);
          const int i2 = __builtin_nontemporal_load(ws_idx + (int64_t)b * M + m);
          amax_comb(v, i, v2, i2);
        }
        wave_amax(v, i);
        if (lane == 0) out_idx[m] = i;
      }
      if (threadIdx.x == 0) *counter = 0u;  // self-reset for the next call / graph replay
    }
  }
}

// ---------------------------------------------------------------------------------------------
template <int M, int R, int S, int J, int G, bool NT>
__global__ __launch_bounds__(64 * S * G) void gemv_smallN(
    const uint16_t* __restrict__ x, int ldx, const uint16_t* __restrict__ w, int64_t ldw,
    uint16_t* __restrict__ y, int N, int ngroups) {
  const int lane = threadIdx.x & 63;
  const int wv = threadIdx.x >> 6;
  const int gi = wv / S, s = wv - gi * S;
  const int g = blockIdx.x * G + gi;
  const bool valid = g < ngroups;
  const u32x4* wb = reinterpret_cast<const u32x4*>(w);
  const int64_t ldw4 = ldw / 8;
  u32x4 wr[R][J];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    int row = (valid ? g : 0) * R + r;
    row = row < N ? row : N - 1;
    const u32x4* p = wb + (int64_t)row * ldw4 + s * 64 + lane;
#pragma unroll
    for (int j = 0; j < J; ++j) wr[r][j] = ldw16<NT>(p + j * 64 * S);
  }
  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
#pragma unroll
  for (int m = 0; m < M; ++m) {
    const u32x4* xp = reinterpret_cast<const u32x4*>(x + (int64_t)m * ldx) + s * 64 + lane;
#pragma unroll
    for (int j = 0; j < J; ++j) {
      const u32x4 xv = xp[j * 64 * S];
#pragma unroll
      for (int r = 0; r < R; ++r) acc[r][m] = dot8(wr[r][j], xv, acc[r][m]);
    }
  }
  __shared__ float red[G][S][R * M];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float v = wave_sum(acc[r][m]);
      if (lane == 0) red[gi][s][r * M + m] = v;
    }
  __syncthreads();
  if (s == 0 && lane < R * M && valid) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q < S; ++q) v += red[gi][q][lane];
    const int r = lane / M, m = lane - r * M;
    const int row = g * R + r;
    if (row < N) y[(int64_t)m * N + row] = f2bf(v);
  }
}

// ---------------------------------------------------------------------------------------------
// host side. Config ids: big-N configs are (R, WPB, NT, PF, XREG); XREG configs are only valid
// for M <= 2.
struct BigCfg { int R, WPB; bool NT, PF, XREG; };
static const BigCfg kBig[] = {
    {1, 4, true, false, true},    // 0
    {1, 4, true, true, true},     // 1
    {2, 4, true, false, true},    // 2
    {1, 8, true, false, true},    // 3
    {1, 4, false, false, true},   // 4
    {1, 16, true, false, false},  // 5  LDS x
    {1, 16, true, true, false},   // 6
    {2, 16, true, false, false},  // 7
    {1, 8, true, false, false},   // 8
    {2, 8, true, false, false},   // 9
    {1, 8, true, true, false},    // 10
    {1, 8, true, true, true},     // 11
    {1, 4, false, true, true},    // 12
    {1, 4, true, true, true},     // 13 (2 blocks/CU)
};
static constexpr int kNumBig = sizeof(kBig) / sizeof(kBig[0]);
struct SmallCfg { int R, S, G; };
static const SmallCfg kSmall[] = {{1, 10, 1}, {1, 5, 3}, {2, 10, 1}, {2, 5, 3}};  // ids 100..103

#define BIGN_ARGS xp, ldx, wp, ldw, yp, N, ngroups, ws_val, ws_idx, counter, out_idx
#define BIGN_PARAMS const uint16_t *xp, int ldx, const uint16_t *wp, int64_t ldw, uint16_t *yp, int N, \
  int ngroups, float *ws_val, int *ws_idx, unsigned *counter, int64_t *out_idx

template <typename Kern>
static int occ_blocks(Kern k, int threads) {
  int nb = 0;
  hipOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, threads, 0);
  return nb > 0 ? nb : 1;
}

static int g_num_cus = 0;
static int num_cus() {
  if (!g_num_cus) {
    int dev;
    hipGetDevice(&dev);
    hipDeviceGetAttribute(&g_num_cus, hipDeviceAttributeMultiprocessorCount, dev);
  }
  return g_num_cus;
}

template <int M, int R, int WPB, bool NT, bool PF, bool XREG, bool AM, int BPC>
static int launch_big(int grid_override, hipStream_t st, BIGN_PARAMS) {
  auto k = gemv_bigN<M, R, 10, WPB, NT, PF, XREG, AM>;
  static int bpc = BPC > 0 ? BPC : occ_blocks(k, 64 * WPB);
  int grid = grid_override > 0 ? grid_override : num_cus() * bpc;
  grid = std::min(grid, (ngroups + WPB - 1) / WPB);
  if (AM) grid = std::min(grid, 1024);  // the sampler reads one candidate per block (<= 1024)
  hipLaunchKernelGGL(k, dim3(grid), dim3(64 * WPB), 0, st, BIGN_ARGS);
  return grid;
}

// returns grid used (for workspace sizing) via launch; argmax workspace sized for the max grid
template <int M, bool AM>
static int dispatch_big(int cid, int grid_override, hipStream_t st, BIGN_PARAMS) {
  switch (cid) {
#define C(ID, R_, W_, NT_, PF_, XR_, BPC_)                                               \
  case ID:                                                                               \
    if constexpr (!XR_ || M <= 2) return launch_big<M, R_, W_, NT_, PF_, XR_, AM, BPC_>(grid_override, st, BIGN_ARGS); \
    break;
    // id, rows/wave, waves/block, nontemporal, prefetch, x-in-registers, blocks/CU (0 = occupancy)
    C(0, 1, 4, true, false, true, 0) C(1, 1, 4, true, true, true, 1) C(2, 2, 4, true, false, true, 0)
    C(3, 1, 8, true, false, true, 0) C(4, 1, 4, false, false, true, 0) C(5, 1, 16, true, false, false, 0)
    C(6, 1, 16, true, true, false, 0) C(7, 2, 16, true, false, false, 0) C(8, 1, 8, true, false, false, 0)
    C(9, 2, 8, true, false, false, 0) C(10, 1, 8, true, true, false, 0)
    C(11, 1, 8, true, true, true, 1) C(12, 1, 4, false, true, true, 1) C(13, 1, 4, true, true, true, 2)
#undef C
  }
  TORCH_CHECK(false, "bf16_gemv: config ", cid, " not available for M=", M);
  return 0;
}

template <int M>
static void dispatch_small(int cid, hipStream_t st, const uint16_t* xp, int ldx, const uint16_t* wp,
                           int64_t ldw, uint16_t* yp, int N, int K) {
#define B(ID, R_, S_, G_)                                                                       \
  if (cid == ID) {                                                                              \
    const int ngroups = (N + R_ - 1) / R_;                                                      \
    const int grid = (ngroups + G_ - 1) / G_;                                                   \
    if (K == 10240)                                                                             \
      hipLaunchKernelGGL((gemv_smallN<M, R_, S_, 20 / S_, G_, true>), dim3(grid), dim3(64 * S_ * G_), 0, st, \
                         xp, ldx, wp, ldw, yp, N, ngroups);                                     \
    else                                                                                        \
      hipLaunchKernelGGL((gemv_smallN<M, R_, S_, 10 / S_, G_, true>), dim3(grid), dim3(64 * S_ * G_), 0, st, \
                         xp, ldx, wp, ldw, yp, N, ngroups);                                     \
    return;                                                                                     \
  }
  B(100, 1, 10, 1) B(101, 1, 5, 3) B(102, 2, 10, 1) B(103, 2, 5, 3)
#undef B
  TORCH_CHECK(false, "bf16_gemv: bad smallN config ", cid);
}

// default configs (from the micro-bench, see test_bf16_gemv.py)
static int default_cfg(int N, int M) {
  if (N <= 4096) return 100;
  if (M <= 2) return 1;   // x in registers, prefetch, 256-thread blocks
  if (M <= 4) return 10;  // x in LDS, prefetch, 512-thread blocks
  return 9;               // x in LDS, 2 rows per wave
}

static torch::Tensor g_ws;  // argmax workspace: counter + per-block candidates (zeroed once)
static constexpr int kMaxGrid = 4096;
// last argmax GEMV launch (consumed by the fused sampler): logits pointer, grid, M
static const void* g_last_logits = nullptr;
static int g_last_grid = 0, g_last_m = 0;

static void run(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y, int64_t cfg_id,
                int64_t grid_override, torch::Tensor* idx_out) {
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1);
  TORCH_CHECK(w.dim() == 2 && w.scalar_type() == at::kBFloat16 && w.stride(1) == 1);
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(w.size(1) == K && (K == 5120 || K == 10240), "bf16_gemv: K must be 5120 or 10240");
  // K = 10240 (MTP fc 5120x10240): small-N split-K family only (configs 100..103), no argmax
  TORCH_CHECK(K == 5120 || (idx_out == nullptr && (cfg_id < 0 || cfg_id >= 100)), "bf16_gemv: K=10240 needs a small-N config");
  TORCH_CHECK(M >= 1 && M <= 8, "bf16_gemv: M must be 1..8");
  TORCH_CHECK(x.stride(0) % 8 == 0 && w.stride(0) % 8 == 0);
  TORCH_CHECK(((uintptr_t)x.data_ptr() & 15) == 0 && ((uintptr_t)w.data_ptr() & 15) == 0);
  if (M > 6 && K == 5120 && (cfg_id >= 0 ? cfg_id < 100 : N > 4096)) {
    // big N, 7..8 rows: two launches of 4 and M-4 rows (x of 7..8 rows does not fit in LDS). Every
    // output row is computed by the same per-row arithmetic for any M, so rows stay bitwise equal
    // to the M=1 call. The fused top-k sampler needs one launch's block candidates: not offered.
    auto x0 = x.narrow(0, 0, 4), x1 = x.narrow(0, 4, M - 4);
    auto y0 = y.narrow(0, 0, 4), y1 = y.narrow(0, 4, M - 4);
    if (idx_out) {
      auto i0 = idx_out->narrow(0, 0, 4), i1 = idx_out->narrow(0, 4, M - 4);
      run(x0, w, y0, cfg_id, grid_override, &i0);
      run(x1, w, y1, cfg_id, grid_override, &i1);
      g_last_logits = nullptr; g_last_grid = 0; g_last_m = 0;
    } else {
      run(x0, w, y0, cfg_id, grid_override, nullptr);
      run(x1, w, y1, cfg_id, grid_override, nullptr);
    }
    return;
  }
  const int cid = cfg_id < 0 ? (K == 10240 ? 100 : default_cfg(N, M)) : (int)cfg_id;
  auto st = at::cuda::getCurrentCUDAStream().stream();
  const uint16_t* xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
  const uint16_t* wp = reinterpret_cast<const uint16_t*>(w.data_ptr());
  uint16_t* yp = reinterpret_cast<uint16_t*>(y.data_ptr());
  const int ldx = x.stride(0);
  const int64_t ldw = w.stride(0);
  if (cid >= 100) {
    TORCH_CHECK(idx_out == nullptr, "argmax only for big-N configs");
    switch (M) {
      case 1: return dispatch_small<1>(cid, st, xp, ldx, wp, ldw, yp, N, K);
      case 2: return dispatch_small<2>(cid, st, xp, ldx, wp, ldw, yp, N, K);
      case 3: return dispatch_small<3>(cid, st, xp, ldx, wp, ldw, yp, N, K);
      case 4: return dispatch_small<4>(cid, st, xp, ldx, wp, ldw, yp, N, K);
      case 5: return dispatch_small<5>(cid, st, xp, ldx, wp, ldw, yp, N, K);
      case 6: return dispatch_small<6>(cid, st, xp, ldx, wp, ldw, yp, N, K);
      case 7: return dispatch_small<7>(cid, st, xp, ldx, wp, ldw, yp, N, K);
      case 8: return dispatch_small<8>(cid, st, xp, ldx, wp, ldw, yp, N, K);
    }
  }
  TORCH_CHECK(cid >= 0 && cid < kNumBig, "bad config");
  const int ngroups = (N + kBig[cid].R - 1) / kBig[cid].R;
  TORCH_CHECK(grid_override <= kMaxGrid);
  float* ws_val = nullptr;
  int* ws_idx = nullptr;
  unsigned* counter = nullptr;
  int64_t* out_idx = nullptr;
  if (idx_out) {
    const int64_t need = 64 + (int64_t)kMaxGrid * 8 * 2;
    if (!g_ws.defined() || g_ws.device() != x.device()) {
      g_ws = torch::zeros({need}, x.options().dtype(at::kInt));  // allocated outside graph capture
    }
    int* base = g_ws.data_ptr<int>();
    counter = reinterpret_cast<unsigned*>(base);
    ws_val = reinterpret_cast<float*>(base + 64);
    ws_idx = base + 64 + kMaxGrid * 8;
    out_idx = idx_out->data_ptr<int64_t>();
  }
  int grid = 0;
#define D(MM)                                                                           case MM:                                                                                if (idx_out) grid = dispatch_big<MM, true>(cid, (int)grid_override, st, BIGN_ARGS);     else grid = dispatch_big<MM, false>(cid, (int)grid_override, st, BIGN_ARGS);          break;
  switch (M) { D(1) D(2) D(3) D(4) D(5) D(6) }
#undef D
  if (idx_out) { g_last_logits = y.data_ptr(); g_last_grid = grid; g_last_m = M; }
}

torch::Tensor bf16_gemv(torch::Tensor x, torch::Tensor w, int64_t cfg, int64_t grid) {
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto y = torch::empty({x.size(0), w.size(0)}, x.options());
  run(x, w, y, cfg, grid, nullptr);
  return y;
}

// logits (bf16 [M, N]) plus greedy argmax of the bf16 logits per row (int64 [M]).
std::vector<torch::Tensor> bf16_gemv_argmax(torch::Tensor x, torch::Tensor w, int64_t cfg, int64_t grid) {
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto y = torch::empty({x.size(0), w.size(0)}, x.options());
  auto idx = torch::empty({x.size(0)}, x.options().dtype(at::kLong));
  run(x, w, y, cfg, grid, &idx);
  return {y, idx};
}

// ---------------------------------------------------------------------------------------------
// Fused top-k / top-p / temperature sampler for small batches (kSNB blocks of kSB threads per row).
// Semantics follow vLLM's sampler for requests that only use temperature, top_k (<= 64) and top_p:
//   * temperature 0 -> greedy: the argmax of the bf16 logits from the GEMV epilogue;
//   * keep logits >= the k-th largest value (ties at the boundary kept, like the torch reference);
//   * top-p on softmax(logits / T) of the kept set: in ascending order, drop tokens whose cumulative
//     probability is <= 1 - p (the largest token is always kept);
//   * sample from softmax(logits / T) restricted to the kept set by Gumbel-max with counter-based
//     Philox noise keyed by (request seed, position, token id).
// Candidate pruning: the k-th largest of the per-block maxima of the GEMV epilogue (one per block,
// each an actual logit) is <= the true k-th largest logit, so all elements >= it contain the top-k.
// Typically a few dozen candidates; if more than CAP, an exact threshold is found by a 16-step
// binary search over the bf16 order keys of the row.
constexpr int kCap = 1024;

__device__ __forceinline__ void philox10(uint32_t (&c)[4], uint32_t k0, uint32_t k1) {
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    const uint32_t hi0 = __umulhi(0xD2511F53u, c[0]), lo0 = 0xD2511F53u * c[0];
    const uint32_t hi1 = __umulhi(0xCD9E8D57u, c[2]), lo1 = 0xCD9E8D57u * c[2];
    const uint32_t n0 = hi1 ^ c[1] ^ k0, n2 = hi0 ^ c[3] ^ k1;
    c[0] = n0; c[1] = lo1; c[2] = n2; c[3] = lo0;
    k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
  }
}

__device__ __forceinline__ float gumbel(int64_t seed, int64_t pos, int token) {
  uint32_t c[4] = {(uint32_t)pos, (uint32_t)((uint64_t)pos >> 32), (uint32_t)token, 0x5eed5a3du};
  philox10(c, (uint32_t)seed, (uint32_t)((uint64_t)seed >> 32));
  // u in (0, 1) from 32 random bits (53-bit double for the tails)
  const double u = ((double)c[0] + 0.5) * (1.0 / 4294967296.0);
  return (float)(-log(-log(u)));
}

__device__ __forceinline__ uint32_t bf_key(uint32_t b) {  // order-preserving key of a bf16
  return (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u);
}

constexpr int kSB = 256;  // threads per sampler block
constexpr int kSNB = 32;  // sampler blocks per row (the row scan is split over them)

__device__ __forceinline__ int block_sum_int(int v, int* sbuf) {  // kSB threads
#pragma unroll
  for (int off = 32; off >= 1; off >>= 1) v += __shfl_xor(v, off);
  const int lane = threadIdx.x & 63, wv = threadIdx.x >> 6;
  __syncthreads();
  if (lane == 0) sbuf[wv] = v;
  __syncthreads();
  int t = 0;
#pragma unroll
  for (int i = 0; i < kSB / 64; ++i) t += sbuf[i];
  return t;
}

// grid (kSNB, M): every block derives the threshold, scans 1/kSNB of the row and appends candidates
// to a global per-row list; the last block of a row (atomic ticket) ranks the candidates, applies
// top-k / top-p and samples, then resets the row's counters (no host sync, re-launchable).
template <typename IdxT>
__global__ __launch_bounds__(kSB) void topk_topp_sample_kernel(
    const uint16_t* __restrict__ logits, int64_t ldl, int V,
    const float* __restrict__ bmax, int nb, int M,
    const int64_t* __restrict__ amax, const IdxT* __restrict__ idx_map,
    const float* __restrict__ temp_a, const int* __restrict__ topk_a, const float* __restrict__ topp_a,
    const int64_t* __restrict__ seed_a, const int64_t* __restrict__ pos_a, int64_t* __restrict__ out,
    int* __restrict__ g_cnt, unsigned* __restrict__ g_done, float* __restrict__ g_val, int* __restrict__ g_idx) {
  const int m = blockIdx.y;
  const int tid = threadIdx.x, lane = tid & 63;
  __shared__ float s_bm[1024];
  __shared__ float cv[kCap];
  __shared__ int ci[kCap];
  __shared__ float sv[kCap];
  __shared__ int si[kCap];
  __shared__ int s_cnt, s_nk;
  __shared__ float s_thr;
  __shared__ bool s_last;
  __shared__ int s_red[kSB / 64];

  const int req = (int)idx_map[m];
  // all per-request parameters in one round trip (they may live in host-mapped UVA memory)
  const float T = temp_a[req];
  const int k = topk_a[req];
  const float p = topp_a[req];
  const int64_t seed = seed_a[req];
  const int64_t ps = pos_a[m];
  if (T == 0.f) {  // greedy row: GEMV argmax
    if (blockIdx.x == 0 && tid == 0) out[m] = amax[m];
    return;
  }

  // 1. threshold = k-th largest per-block maximum of the GEMV epilogue
  for (int i = tid; i < nb; i += kSB) s_bm[i] = bmax[(int64_t)i * M + m];
  if (tid == 0) s_thr = -__builtin_inff();
  __syncthreads();
  for (int i = tid; i < nb; i += kSB) {
    const float v = s_bm[i];
    int r = 0;
    for (int j = 0; j < nb; ++j) {
      const float w = s_bm[j];
      r += (w > v) || (w == v && j < i);
    }
    if (r == k - 1) s_thr = v;
  }
  __syncthreads();
  float thr = s_thr;

  // 2. scan this block's slice, append candidates >= thr to the row's global list
  const u32x4* row = reinterpret_cast<const u32x4*>(logits + (int64_t)m * ldl);
  const int nch = V / 8;
  const int per = (nch + gridDim.x - 1) / gridDim.x;
  const int c0 = blockIdx.x * per, c1 = min(nch, c0 + per);
  int* cnt = g_cnt + m;
  float* gv = g_val + (int64_t)m * kCap;
  int* gi = g_idx + (int64_t)m * kCap;
  for (int c = c0 + tid; c < c1; c += kSB) {
    const u32x4 u = row[c];
    const uint32_t wds[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float f0 = bflo(wds[q]), f1 = bfhi(wds[q]);
      if (f0 >= thr) { const int ps = atomicAdd(cnt, 1); if (ps < kCap) { gv[ps] = f0; gi[ps] = c * 8 + 2 * q; } }
      if (f1 >= thr) { const int ps = atomicAdd(cnt, 1); if (ps < kCap) { gv[ps] = f1; gi[ps] = c * 8 + 2 * q + 1; } }
    }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = atomicAdd(g_done + m, 1u) == gridDim.x - 1;
  __syncthreads();
  if (!s_last) return;
  __threadfence();

  // 3. last block of the row: gather candidates (or exact threshold on overflow)
  int n = atomicAdd(cnt, 0);
  if (n <= kCap) {
    for (int i = tid; i < n; i += kSB) {
      cv[i] = __builtin_nontemporal_load(gv + i);
      ci[i] = __builtin_nontemporal_load(gi + i);
    }
  } else {
    uint32_t lo = 0, hi = 0xffffu;  // invariant: count(key >= lo) >= k
    while (lo < hi) {
      const uint32_t mid = (lo + hi + 1) >> 1;
      int cn = 0;
      for (int c = tid; c < nch; c += kSB) {
        const u32x4 u = row[c];
        const uint32_t wds[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          cn += bf_key(wds[q] & 0xffffu) >= mid;
          cn += bf_key(wds[q] >> 16) >= mid;
        }
      }
      cn = block_sum_int(cn, s_red);
      if (cn >= k) lo = mid; else hi = mid - 1;
    }
    const uint32_t b = (lo & 0x8000u) ? (lo & 0x7fffu) : (~lo & 0xffffu);
    thr = __builtin_bit_cast(float, b << 16);
    if (tid == 0) s_cnt = 0;
    __syncthreads();
    for (int c = tid; c < nch; c += kSB) {
      const u32x4 u = row[c];
      const uint32_t wds[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float f0 = bflo(wds[q]), f1 = bfhi(wds[q]);
        if (f0 >= thr) { const int ps = atomicAdd(&s_cnt, 1); if (ps < kCap) { cv[ps] = f0; ci[ps] = c * 8 + 2 * q; } }
        if (f1 >= thr) { const int ps = atomicAdd(&s_cnt, 1); if (ps < kCap) { cv[ps] = f1; ci[ps] = c * 8 + 2 * q + 1; } }
      }
    }
    __syncthreads();
    n = min(s_cnt, kCap);  // > kCap only with > kCap ties at the k-th value (then truncated)
  }
  __syncthreads();
  if (tid == 0) { g_cnt[m] = 0; g_done[m] = 0u; }  // reset for the next launch

  // 4. exact descending order of the candidates (value desc, index asc)
  for (int i = tid; i < n; i += kSB) {
    const float v = cv[i];
    const int id = ci[i];
    int r = 0;
    for (int j = 0; j < n; ++j) {
      const float w = cv[j];
      r += (w > v) || (w == v && ci[j] < id);
    }
    sv[r] = v;
    si[r] = id;
  }
  __syncthreads();
  const int kk = min(k, n);
  const float kth = sv[kk - 1];
  for (int i = tid; i < n; i += kSB)
    if (sv[i] >= kth && (i + 1 == n || sv[i + 1] < kth)) s_nk = i + 1;
  __syncthreads();
  const int nk = s_nk;

  // 5. top-p + Gumbel-max sampling (wave 0)
  if (tid >= 64) return;
  const float invT = 1.f / T;
  const float v0 = sv[0];
  int np = nk;
  if (p < 1.f) {
    float z = 0.f;
    for (int r = lane; r < nk; r += 64) {
      const float e = __expf((sv[r] - v0) * invT);
      cv[r] = e;
      z += e;
    }
    z = wave_sum(z);
    int keep = nk;
    if (lane == 0) {
      // ascending cumulative probability from the smallest kept token
      float cum = 0.f;
      for (int r = nk - 1; r >= 1; --r) {
        cum += cv[r] / z;
        if (cum <= 1.f - p) keep = r; else break;
      }
    }
    np = __shfl(keep, 0);
  }
  float best = -__builtin_inff();
  int bidx = 0x7fffffff;
  for (int r = lane; r < np; r += 64) {
    const float sc = sv[r] * invT + gumbel(seed, ps, si[r]);
    amax_comb(best, bidx, sc, si[r]);
  }
  wave_amax(best, bidx);
  if (lane == 0) out[m] = bidx;
}

static torch::Tensor g_sws;  // sampler workspace: per-row counters + candidate lists (zeroed once)

torch::Tensor sample_topk_topp(torch::Tensor logits, torch::Tensor amax, torch::Tensor idx_mapping,
                               torch::Tensor temp, torch::Tensor top_k, torch::Tensor top_p,
                               torch::Tensor seeds, torch::Tensor pos) {
  const at::cuda::OptionalCUDAGuard guard(logits.device());
  TORCH_CHECK(logits.dim() == 2 && logits.scalar_type() == at::kBFloat16 && logits.stride(1) == 1);
  const int M = logits.size(0), V = logits.size(1);
  TORCH_CHECK(logits.data_ptr() == g_last_logits && M == g_last_m && g_ws.defined() && M <= 8,
              "sample_topk_topp: logits must come from the last bf16_gemv_argmax call");
  TORCH_CHECK(V % 8 == 0 && logits.stride(0) % 8 == 0);
  TORCH_CHECK(amax.scalar_type() == at::kLong && amax.numel() == M);
  TORCH_CHECK(temp.scalar_type() == at::kFloat && top_k.scalar_type() == at::kInt &&
              top_p.scalar_type() == at::kFloat && seeds.scalar_type() == at::kLong &&
              pos.scalar_type() == at::kLong && pos.numel() >= M && idx_mapping.numel() >= M);
  if (!g_sws.defined() || g_sws.device() != logits.device())
    g_sws = torch::zeros({64 + 8 * kCap * 2}, logits.options().dtype(at::kInt));
  int* wsb = g_sws.data_ptr<int>();
  auto out = torch::empty({M}, amax.options());
  auto st = at::cuda::getCurrentCUDAStream().stream();
  const float* bmax = reinterpret_cast<const float*>(g_ws.data_ptr<int>() + 64);
  auto pos_c = pos.contiguous();
  auto im = idx_mapping.contiguous();
#define L(T_)                                                                                      \
  hipLaunchKernelGGL((topk_topp_sample_kernel<T_>), dim3(kSNB, M), dim3(kSB), 0, st,               \
                     reinterpret_cast<const uint16_t*>(logits.data_ptr()), logits.stride(0), V,    \
                     bmax, g_last_grid, M, amax.data_ptr<int64_t>(), im.data_ptr<T_>(),            \
                     temp.data_ptr<float>(), top_k.data_ptr<int>(), top_p.data_ptr<float>(),       \
                     seeds.data_ptr<int64_t>(), pos_c.data_ptr<int64_t>(), out.data_ptr<int64_t>(), \
                     wsb, reinterpret_cast<unsigned*>(wsb + 8),                                    \
                     reinterpret_cast<float*>(wsb + 64), wsb + 64 + 8 * kCap)
  if (im.scalar_type() == at::kInt) L(int);
  else { TORCH_CHECK(im.scalar_type() == at::kLong); L(int64_t); }
#undef L
  return out;
}

TORCH_LIBRARY(vopt_lmhead, m) {
  m.def("bf16_gemv(Tensor x, Tensor w, int cfg=-1, int grid=0) -> Tensor");
  m.def("bf16_gemv_argmax(Tensor x, Tensor w, int cfg=-1, int grid=0) -> Tensor[]");
  m.def("sample_topk_topp(Tensor logits, Tensor amax, Tensor idx_mapping, Tensor temp, Tensor top_k, "
        "Tensor top_p, Tensor seeds, Tensor pos) -> Tensor");
}
TORCH_LIBRARY_IMPL(vopt_lmhead, CUDA, m) {
  m.impl("bf16_gemv", &bf16_gemv);
  m.impl("bf16_gemv_argmax", &bf16_gemv_argmax);
  m.impl("sample_topk_topp", &sample_topk_topp);
}
