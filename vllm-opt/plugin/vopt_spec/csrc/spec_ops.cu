// vopt_spec kernels (gfx90a / MI210) for the MTP draft side and the greedy verify fast path.
//
// 1. Draft head (argmax only; the draft never needs logits) over a row subset of lm_head, in one
//    of two draft-only weight copies made at load time:
//    * fp8: e4m3fn bytes with one fp32 scale per row. e4m3 -> f16 is exact: per 16-bit half,
//      h = (b + (b & 0x80)) << 7 moves the sign to bit 15 and exponent/mantissa into f16 position
//      (value * 2^-8, folded into the row scale; e4m3 subnormals map to f16 subnormals exactly).
//    * int4: 4-bit codes with an asymmetric (scale, min) per row and 128-column group; rows are
//      stored in interleaved pairs (one 16-byte load = 16 columns of two rows). A code n becomes the
//      f16 1 + n/16 (0x3C00 | n << 6, two codes per op), so sum_k w x = a * sum_k h x + b * sum_k x
//      with a = 16 * scale, b = min - 16 * scale per group (stored as f16).
//    x (bf16) is converted once per block to f16 and staged in LDS in the order the weight decode
//    produces; dot products use v_dot2_f32_f16 (fp32 accumulate).
//    Each wave keeps its top-4 rows; the last block finds the global max and (optionally) rescores
//    every row within `delta` of it with the bf16 lm_head (fp32 dot, bf16-rounded logit, lowest
//    vocab id on ties), so the draft equals the bf16-subset argmax whenever the quantization error
//    of the competing logits is below delta / 2.
//    Draft-only: the result decides which tokens are proposed, never which tokens are output.
// 2. quantize_rows_fp8: bf16 [N, K] -> e4m3fn [N, K] + fp32 scale [N] (per-row absmax / 448, RNE).
// 3. greedy_verify: rejection sampling for all-greedy batches from the target argmax that the
//    lm_head GEMV epilogue already computed (same kernel/order as the K=0 greedy path).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));
typedef _Float16 h2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float bf2f(uint16_t b) { return __builtin_bit_cast(float, (uint32_t)b << 16); }

template <int CTRL>
__device__ __forceinline__ float dppf(float v) {
  return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), CTRL, 0xf, 0xf, true));
}
__device__ __forceinline__ float wave_sum_u(float v) {
  v += dppf<0xb1>(v);
  v += dppf<0x4e>(v);
  v += dppf<0x141>(v);
  v += dppf<0x140>(v);
  const int b = __builtin_bit_cast(int, v);
  return (__builtin_bit_cast(float, __builtin_amdgcn_readlane(b, 0)) +
          __builtin_bit_cast(float, __builtin_amdgcn_readlane(b, 16))) +
         (__builtin_bit_cast(float, __builtin_amdgcn_readlane(b, 32)) +
          __builtin_bit_cast(float, __builtin_amdgcn_readlane(b, 48)));
}

// larger value wins, ties -> smaller index; NaN counts as the largest value
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

__device__ __forceinline__ uint32_t f16pair(uint32_t t) {  // t = two bytes in the low byte of each half
  return (t + (t & 0x00800080u)) << 7;
}
__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
  return __builtin_amdgcn_fdot2(__builtin_bit_cast(h2, a), __builtin_bit_cast(h2, b), c, false);
}

// one 16-byte fp8 weight chunk (16 values) against the matching 2 x 16 bytes of staged f16 x
__device__ __forceinline__ float dot16(const u32x4 w, const u32x4 xa, const u32x4 xb, float acc) {
  const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
  const uint32_t xs[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint32_t ev = f16pair(ws[q] & 0x00ff00ffu);           // bytes 0, 2
    const uint32_t od = f16pair((ws[q] >> 8) & 0x00ff00ffu);    // bytes 1, 3
    acc = dot2(ev, xs[2 * q], acc);
    acc = dot2(od, xs[2 * q + 1], acc);
  }
  return acc;
}

// 8 int4 codes (nibble i = column i of 8) against x pairs (x_i, x_{i+4}), i = 0..3 -> sum_k (1 + n_k/16) x_k
__device__ __forceinline__ float dot8i4(uint32_t w, const uint32_t (&xp)[4], float acc) {
  const uint32_t w8 = w >> 8;
  acc = dot2(((w & 0x000F000Fu) << 6) | 0x3C003C00u, xp[0], acc);
  acc = dot2(((w & 0x00F000F0u) << 2) | 0x3C003C00u, xp[1], acc);
  acc = dot2(((w8 & 0x000F000Fu) << 6) | 0x3C003C00u, xp[2], acc);
  acc = dot2(((w8 & 0x00F000F0u) << 2) | 0x3C003C00u, xp[3], acc);
  return acc;
}

constexpr int KD = 5120;           // hidden size
constexpr int KCH = KD / 16;       // 16-column chunks per row (320)
constexpr int JD = KCH / 64;       // chunks per lane (5)
constexpr int GQ = 128;            // int4 group (columns)
constexpr int NG = KD / GQ;        // groups per row (40)

__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
  const _Float16 ha = (_Float16)fminf(fmaxf(a, -65504.f), 65504.f), hb = (_Float16)fminf(fmaxf(b, -65504.f), 65504.f);
  return (uint32_t)__builtin_bit_cast(uint16_t, ha) | ((uint32_t)__builtin_bit_cast(uint16_t, hb) << 16);
}
__device__ __forceinline__ float h2f(uint16_t h) { return (float)__builtin_bit_cast(_Float16, h); }

// stage x[m] (bf16) as f16 in decode order. Chunk c (16 columns) -> 8 words (2 x u32x4).
//   fp8 : word 2q = (x[16c+4q], x[16c+4q+2]), word 2q+1 = (x[16c+4q+1], x[16c+4q+3])
//   int4: word 4u+i = (x[16c+8u+i], x[16c+8u+i+4]), u = 0, 1, i = 0..3; plus sx[m][c] = sum of the
//         16 f16-rounded x values (fp32)
template <int M, bool I4>
__device__ __forceinline__ void stage_x(const uint16_t* __restrict__ x, int ldx, u32x4* xs, float* sx) {
  for (int i = threadIdx.x; i < M * KCH * 2; i += blockDim.x) {
    const int m = i / (KCH * 2), r = i - m * KCH * 2;
    const int c = r >> 1, half = r & 1;
    const uint16_t* xp = x + (int64_t)m * ldx + c * 16 + half * 8;
    float v[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) v[t] = (float)(_Float16)fminf(fmaxf(bf2f(xp[t]), -65504.f), 65504.f);
    uint32_t o[4];
    if constexpr (I4) {
#pragma unroll
      for (int t = 0; t < 4; ++t) o[t] = pack_h2(v[t], v[t + 4]);
      float s = 0.f;
#pragma unroll
      for (int t = 0; t < 8; ++t) s += v[t];
      // the two halves of a chunk add up in sx (two threads): use an atomic on LDS
      atomicAdd(&sx[m * KCH + c], s);
    } else {
#pragma unroll
      for (int qq = 0; qq < 2; ++qq) {
        o[2 * qq] = pack_h2(v[qq * 4], v[qq * 4 + 2]);
        o[2 * qq + 1] = pack_h2(v[qq * 4 + 1], v[qq * 4 + 3]);
      }
    }
    xs[(m * KCH + c) * 2 + half] = (u32x4){o[0], o[1], o[2], o[3]};
  }
}

// candidate order: larger value first, ties -> smaller index (values are finite here)
__device__ __forceinline__ bool better(float v2, int i2, float v, int i) { return v2 > v || (v2 == v && i2 < i); }

constexpr int TOPT = 4;   // candidates kept per wave / block (rescoring window)
constexpr int CAP = 32;   // max candidates rescored in bf16

template <int T>
__device__ __forceinline__ void top_insert(float (&tv)[T], int (&ti)[T], float v, int i) {
  if (!better(v, i, tv[T - 1], ti[T - 1])) return;
  tv[T - 1] = v; ti[T - 1] = i;
#pragma unroll
  for (int t = T - 1; t > 0; --t) {
    if (better(tv[t], ti[t], tv[t - 1], ti[t - 1])) {
      const float a = tv[t]; tv[t] = tv[t - 1]; tv[t - 1] = a;
      const int b = ti[t]; ti[t] = ti[t - 1]; ti[t - 1] = b;
    }
  }
}

__device__ __forceinline__ uint16_t f2bf(float f) {
  uint32_t u = __builtin_bit_cast(uint32_t, f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

struct HeadArgs {
  const uint16_t* x; int ldx;
  int N, ngroups;
  float* ws_val; int* ws_idx; unsigned* counter;
  int64_t* out_idx; const int64_t* idmap;
  const uint16_t* wbf; int64_t ldw; float delta;
};

// Block reduction of the per-wave top lists, cross-block reduction in the last block, optional
// bf16 rescoring of the window (all waves of the last block share the rescoring work).
template <int M, int WPB>
__device__ __forceinline__ void head_finish(const HeadArgs& A, const float (&tv)[M][TOPT], const int (&ti)[M][TOPT]) {
  const int lane = threadIdx.x & 63;
  const int wv = threadIdx.x >> 6;
  __shared__ float sv[WPB][M][TOPT];
  __shared__ int si[WPB][M][TOPT];
  __shared__ bool last;
  __shared__ int cand[M][CAP];
  __shared__ float clog[M][CAP];
  __shared__ int ccount[M];
  __shared__ bool cok[M];
  __shared__ int64_t cbest[M];
  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int t = 0; t < TOPT; ++t) { sv[wv][m][t] = tv[m][t]; si[wv][m][t] = ti[m][t]; }
  }
  __syncthreads();
  if (threadIdx.x < M) {
    const int m = threadIdx.x;
    float bv[TOPT];
    int bi[TOPT];
#pragma unroll
    for (int t = 0; t < TOPT; ++t) { bv[t] = sv[0][m][t]; bi[t] = si[0][m][t]; }
    for (int w = 1; w < WPB; ++w)
#pragma unroll
      for (int t = 0; t < TOPT; ++t) top_insert<TOPT>(bv, bi, sv[w][m][t], si[w][m][t]);
#pragma unroll
    for (int t = 0; t < TOPT; ++t) {
      A.ws_val[((int64_t)blockIdx.x * M + m) * TOPT + t] = bv[t];
      A.ws_idx[((int64_t)blockIdx.x * M + m) * TOPT + t] = bi[t];
    }
    ccount[m] = 0;
    cok[m] = false;
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(A.counter, 1u) == gridDim.x - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  const int nc = (int)gridDim.x * TOPT;
  if (wv < M) {
    const int m = wv;
    float v = -__builtin_inff();
    int i = 0x7fffffff;
    for (int c = lane; c < nc; c += 64) {
      const int b = c / TOPT, t = c - b * TOPT;
      const float v2 = __builtin_nontemporal_load(A.ws_val + ((int64_t)b * M + m) * TOPT + t);
      const int i2 = __builtin_nontemporal_load(A.ws_idx + ((int64_t)b * M + m) * TOPT + t);
      if (better(v2, i2, v, i)) { v = v2; i = i2; }
    }
    wave_amax(v, i);
    // garbage rows (e.g. padded requests of a captured batch: NaN/inf hidden states) leave no valid
    // candidate: fall back to row 0 so no out-of-range vocab id is ever produced
    if (i < 0 || i >= A.N || !(v == v) || v == -__builtin_inff()) i = 0;
    if (lane == 0) cbest[m] = A.idmap ? A.idmap[i] : (int64_t)i;
    if (A.wbf != nullptr) {
      const float thr = v - A.delta;
      bool ovf = false;
      for (int c = lane; c < nc; c += 64) {
        const int b = c / TOPT, t = c - b * TOPT;
        const float v2 = __builtin_nontemporal_load(A.ws_val + ((int64_t)b * M + m) * TOPT + t);
        const int i2 = __builtin_nontemporal_load(A.ws_idx + ((int64_t)b * M + m) * TOPT + t);
        if (v2 >= thr && i2 >= 0 && i2 < A.N) {
          if (t == TOPT - 1) ovf = true;  // this block may hold more rows inside the window
          const int p = atomicAdd(&ccount[m], 1);
          if (p < CAP) cand[m][p] = i2;
        }
      }
      ovf = __any(ovf);
      if (lane == 0) cok[m] = !ovf;
    }
  }
  __syncthreads();
  if (A.wbf == nullptr) {
    if (threadIdx.x < M) A.out_idx[threadIdx.x] = cbest[threadIdx.x];
    if (threadIdx.x == 0) *A.counter = 0u;
    return;
  }
  // rescoring: work item (m, c) -> wave (m * CAP + c) % WPB
  for (int it = wv; it < M * CAP; it += WPB) {
    const int m = it / CAP, c = it - m * CAP;
    const int n = ccount[m];
    if (!cok[m] || n > CAP || c >= n) continue;
    const int row = cand[m][c];
    const int64_t vid = A.idmap ? A.idmap[row] : (int64_t)row;
    const u32x4* wr = reinterpret_cast<const u32x4*>(A.wbf + vid * A.ldw);
    const u32x4* xr = reinterpret_cast<const u32x4*>(A.x + (int64_t)m * A.ldx);
    float a = 0.f;
#pragma unroll
    for (int j = 0; j < KD / 512; ++j) {
      const u32x4 w8 = wr[j * 64 + lane], x8 = xr[j * 64 + lane];
      const uint32_t ww[4] = {w8.x, w8.y, w8.z, w8.w}, xx[4] = {x8.x, x8.y, x8.z, x8.w};
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        a = fmaf(__builtin_bit_cast(float, ww[h] << 16), __builtin_bit_cast(float, xx[h] << 16), a);
        a = fmaf(__builtin_bit_cast(float, ww[h] & 0xffff0000u), __builtin_bit_cast(float, xx[h] & 0xffff0000u), a);
      }
    }
    const float lg = __builtin_bit_cast(float, (uint32_t)f2bf(wave_sum_u(a)) << 16);
    if (lane == 0) clog[m][c] = lg;
  }
  __syncthreads();
  if (threadIdx.x < M) {
    const int m = threadIdx.x;
    const int n = ccount[m];
    int64_t best = cbest[m];
    if (cok[m] && n <= CAP && n > 0) {
      float rv = -__builtin_inff();
      int64_t ri = 0x7fffffffffffLL;
      for (int c = 0; c < n; ++c) {
        const int64_t vid = A.idmap ? A.idmap[cand[m][c]] : (int64_t)cand[m][c];
        const float lg = clog[m][c];
        if (lg > rv || (lg == rv && vid < ri)) { rv = lg; ri = vid; }
      }
      if (ri >= 0 && ri < 0x7fffffffffffLL) best = ri;  // all rescored logits NaN: keep the quantized argmax
    }
    A.out_idx[m] = best;
  }
  if (threadIdx.x == 0) *A.counter = 0u;
}

template <int M>
__device__ __forceinline__ void init_top(float (&tv)[M][TOPT], int (&ti)[M][TOPT]) {
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int t = 0; t < TOPT; ++t) { tv[m][t] = -__builtin_inff(); ti[m][t] = 0x7fffffff; }
}

// ---------------------------------------------------------------------------------------------
template <int M, int R, int WPB, bool PF>
__global__ __launch_bounds__(64 * WPB) void fp8_head_argmax(HeadArgs A, const uint8_t* __restrict__ q,
                                                            const float* __restrict__ scale) {
  extern __shared__ u32x4 xs[];  // M * KCH * 2
  const int lane = threadIdx.x & 63;
  const int wv = threadIdx.x >> 6;
  const int gstride = gridDim.x * WPB;
  const int N = A.N, ngroups = A.ngroups;
  int g = blockIdx.x * WPB + wv;
  const u32x4* wb = reinterpret_cast<const u32x4*>(q);

  u32x4 cur[R][JD];
  auto issue = [&](int gg, u32x4 (&dst)[R][JD]) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      int row = gg * R + r;
      row = row < N ? row : N - 1;
      const u32x4* p = wb + (int64_t)row * KCH + lane;
#pragma unroll
      for (int j = 0; j < JD; ++j) dst[r][j] = __builtin_nontemporal_load(p + j * 64);
    }
  };
  if (g < ngroups) issue(g, cur);
  stage_x<M, false>(A.x, A.ldx, xs, nullptr);
  __syncthreads();

  float tv[M][TOPT];
  int ti[M][TOPT];
  init_top<M>(tv, ti);

  for (; g < ngroups; g += gstride) {
    u32x4 nxt[PF ? R : 1][PF ? JD : 1];
    const int gn = g + gstride;
    if constexpr (PF) {
      if (gn < ngroups) issue(gn, nxt);
    }
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int m = 0; m < M; ++m) acc[r][m] = 0.f;
    int xo = lane;
    asm volatile("" : "+v"(xo));
#pragma unroll
    for (int j = 0; j < JD; ++j) {
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const u32x4 xa = xs[(m * KCH + j * 64 + xo) * 2];
        const u32x4 xb = xs[(m * KCH + j * 64 + xo) * 2 + 1];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r][m] = dot16(cur[r][j], xa, xb, acc[r][m]);
      }
      __builtin_amdgcn_sched_barrier(0);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int row = g * R + r;
      const float sc = scale[row < N ? row : N - 1];
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float s = wave_sum_u(acc[r][m]) * sc;
        if (row < N) top_insert<TOPT>(tv[m], ti[m], s, row);
      }
    }
    if constexpr (PF) {
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int j = 0; j < JD; ++j) cur[r][j] = nxt[r][j];
    } else {
      if (gn < ngroups) issue(gn, cur);
    }
  }
  head_finish<M, WPB>(A, tv, ti);
}

// int4: rows in pairs; qp = u32x4 [N/2][KCH] (words: rowA cols 0-7, rowA 8-15, rowB 0-7, rowB 8-15 of the
// chunk), ab = u32x2 [N/2][NG] (half2 (a, b) of rowA, of rowB). One wave owns one row pair at a time.
template <int M, int WPB, bool PF>
__global__ __launch_bounds__(64 * WPB) void i4_head_argmax(HeadArgs A, const u32x4* __restrict__ qp,
                                                           const u32x2* __restrict__ ab) {
  extern __shared__ u32x4 xs[];  // M * KCH * 2 u32x4, then M * KCH floats (sx)
  float* sx = reinterpret_cast<float*>(xs + M * KCH * 2);
  const int lane = threadIdx.x & 63;
  const int wv = threadIdx.x >> 6;
  const int gstride = gridDim.x * WPB;
  const int N = A.N, ngroups = A.ngroups;  // ngroups = row pairs
  int g = blockIdx.x * WPB + wv;

  u32x4 cur[JD];
  u32x2 cab[JD];
  auto issue = [&](int gg, u32x4 (&dq)[JD], u32x2 (&da)[JD]) {
    const u32x4* p = qp + (int64_t)gg * KCH + lane;
    const u32x2* pa = ab + (int64_t)gg * NG + (lane >> 3);
#pragma unroll
    for (int j = 0; j < JD; ++j) {
      dq[j] = __builtin_nontemporal_load(p + j * 64);
      da[j] = pa[j * 8];
    }
  };
  if (g < ngroups) issue(g, cur, cab);
  for (int i = threadIdx.x; i < M * KCH; i += blockDim.x) sx[i] = 0.f;
  __syncthreads();
  stage_x<M, true>(A.x, A.ldx, xs, sx);
  __syncthreads();

  float tv[M][TOPT];
  int ti[M][TOPT];
  init_top<M>(tv, ti);

  for (; g < ngroups; g += gstride) {
    u32x4 nq[PF ? JD : 1];
    u32x2 na[PF ? JD : 1];
    const int gn = g + gstride;
    if constexpr (PF) {
      if (gn < ngroups) issue(gn, nq, na);
    }
    float acc[2][M];
#pragma unroll
    for (int m = 0; m < M; ++m) { acc[0][m] = 0.f; acc[1][m] = 0.f; }
    int xo = lane;
    asm volatile("" : "+v"(xo));
#pragma unroll
    for (int j = 0; j < JD; ++j) {
      const float aA = h2f((uint16_t)(cab[j].x & 0xffffu)), bA = h2f((uint16_t)(cab[j].x >> 16));
      const float aB = h2f((uint16_t)(cab[j].y & 0xffffu)), bB = h2f((uint16_t)(cab[j].y >> 16));
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const u32x4 xa = xs[(m * KCH + j * 64 + xo) * 2];
        const u32x4 xb = xs[(m * KCH + j * 64 + xo) * 2 + 1];
        const float s = sx[m * KCH + j * 64 + xo];
        const uint32_t x0[4] = {xa.x, xa.y, xa.z, xa.w}, x1[4] = {xb.x, xb.y, xb.z, xb.w};
        const float dA = dot8i4(cur[j].y, x1, dot8i4(cur[j].x, x0, 0.f));
        const float dB = dot8i4(cur[j].w, x1, dot8i4(cur[j].z, x0, 0.f));
        acc[0][m] = fmaf(aA, dA, fmaf(bA, s, acc[0][m]));
        acc[1][m] = fmaf(aB, dB, fmaf(bB, s, acc[1][m]));
      }
      __builtin_amdgcn_sched_barrier(0);
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int row = g * 2 + r;
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float s = wave_sum_u(acc[r][m]);
        if (row < N) top_insert<TOPT>(tv[m], ti[m], s, row);
      }
    }
    if constexpr (PF) {
#pragma unroll
      for (int j = 0; j < JD; ++j) { cur[j] = nq[j]; cab[j] = na[j]; }
    } else {
      if (gn < ngroups) issue(gn, cur, cab);
    }
  }
  head_finish<M, WPB>(A, tv, ti);
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

constexpr int kMaxGrid = 1024;
constexpr int kMaxM = 4;
// workspace: [0..63] counters, then per-block top-TOPT (val, idx) candidates per m
static torch::Tensor g_ws;

static void ensure_ws(const torch::Tensor& like) {
  if (!g_ws.defined() || g_ws.device() != like.device())
    g_ws = torch::zeros({64 + kMaxGrid * kMaxM * TOPT * 2}, like.options().dtype(at::kInt));
}

static HeadArgs make_args(const torch::Tensor& x, int N, int64_t* out, const int64_t* idmap,
                          const uint16_t* wbf, int64_t ldw, float delta) {
  int* base = g_ws.data_ptr<int>();
  HeadArgs A;
  A.x = reinterpret_cast<const uint16_t*>(x.data_ptr());
  A.ldx = x.stride(0);
  A.N = N; A.ngroups = 0;
  A.counter = reinterpret_cast<unsigned*>(base);
  A.ws_val = reinterpret_cast<float*>(base + 64);
  A.ws_idx = base + 64 + kMaxGrid * kMaxM * TOPT;
  A.out_idx = out; A.idmap = idmap; A.wbf = wbf; A.ldw = ldw; A.delta = delta;
  return A;
}

template <typename K>
static int blocks_per_cu(K k, int threads, size_t lds) {
  int nb = 0;
  hipOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, threads, lds);
  return nb > 0 ? nb : 1;
}

// fp8 configs: 0 = (R1, 8 waves, prefetch), 1 = (R1, 8, no PF), 2 = (R2, 8, PF), 3 = (R1, 16, PF),
// 4 = (R1, 4, PF), 5 = (R2, 4, PF)
template <int M>
static void launch_fp8(int cfg, int grid_override, hipStream_t st, HeadArgs A, const uint8_t* qp, const float* sp) {
  const size_t lds = (size_t)M * KCH * 2 * 16;
#define L(ID, R_, W_, PF_)                                                                          \
  if (cfg == ID) {                                                                                  \
    auto k = fp8_head_argmax<M, R_, W_, PF_>;                                                       \
    static int bpc = blocks_per_cu(k, 64 * W_, lds);                                                \
    A.ngroups = (A.N + R_ - 1) / R_;                                                                \
    int grid = grid_override > 0 ? grid_override : num_cus() * bpc;                                \
    grid = std::min(std::min(grid, (A.ngroups + W_ - 1) / W_), kMaxGrid);                           \
    hipLaunchKernelGGL(k, dim3(grid), dim3(64 * W_), lds, st, A, qp, sp);                           \
    return;                                                                                         \
  }
  L(0, 1, 8, true) L(1, 1, 8, false) L(2, 2, 8, true) L(3, 1, 16, true) L(4, 1, 4, true) L(5, 2, 4, true)
#undef L
  TORCH_CHECK(false, "fp8_head_argmax: bad config ", cfg);
}

// int4 configs: 0 = (8 waves, PF), 1 = (4 waves, PF), 2 = (8, no PF), 3 = (16, PF)
template <int M>
static void launch_i4(int cfg, int grid_override, hipStream_t st, HeadArgs A, const u32x4* qp, const u32x2* abp) {
  const size_t lds = (size_t)M * KCH * (2 * 16 + 4);
#define L(ID, W_, PF_)                                                                              \
  if (cfg == ID) {                                                                                  \
    auto k = i4_head_argmax<M, W_, PF_>;                                                            \
    static int bpc = blocks_per_cu(k, 64 * W_, lds);                                                \
    A.ngroups = (A.N + 1) / 2;                                                                      \
    int grid = grid_override > 0 ? grid_override : num_cus() * bpc;                                \
    grid = std::min(std::min(grid, (A.ngroups + W_ - 1) / W_), kMaxGrid);                           \
    hipLaunchKernelGGL(k, dim3(grid), dim3(64 * W_), lds, st, A, qp, abp);                          \
    return;                                                                                         \
  }
  L(0, 8, true) L(1, 4, true) L(2, 8, false) L(3, 16, true)
#undef L
  TORCH_CHECK(false, "i4_head_argmax: bad config ", cfg);
}

static void check_common(const torch::Tensor& x, const torch::Tensor& idmap, int N, const torch::Tensor& wbf,
                         double delta, const int64_t** im, const uint16_t** wp, int64_t* ldw) {
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1 && x.size(1) == KD);
  TORCH_CHECK(x.stride(0) % 8 == 0 && ((uintptr_t)x.data_ptr() & 15) == 0);
  TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= kMaxM, "draft head: M must be 1..4");
  *im = nullptr;
  if (idmap.numel() > 0) {
    TORCH_CHECK(idmap.scalar_type() == at::kLong && idmap.is_contiguous() && idmap.numel() >= N);
    *im = idmap.data_ptr<int64_t>();
  }
  *wp = nullptr;
  *ldw = 0;
  if (wbf.numel() > 0 && delta > 0) {
    TORCH_CHECK(wbf.dim() == 2 && wbf.scalar_type() == at::kBFloat16 && wbf.stride(1) == 1 && wbf.size(1) == KD);
    TORCH_CHECK(wbf.stride(0) % 8 == 0 && ((uintptr_t)wbf.data_ptr() & 15) == 0);
    TORCH_CHECK(*im != nullptr || wbf.size(0) >= N);
    *wp = reinterpret_cast<const uint16_t*>(wbf.data_ptr());
    *ldw = wbf.stride(0);
  }
}

// x bf16 [M, 5120] (M <= 4), q uint8 [N, 5120] e4m3fn, scale fp32 [N]; idmap optional int64 [N]
// (subset row -> vocab id; empty = identity); wbf optional bf16 [V, 5120] (full lm_head, rows by
// vocab id) enables bf16 rescoring of the rows within `delta` of the max. Returns int64 [M] vocab ids.
torch::Tensor fp8_head_argmax_op(torch::Tensor x, torch::Tensor q, torch::Tensor scale, torch::Tensor idmap,
                                 torch::Tensor wbf, double delta, int64_t cfg, int64_t grid) {
  const at::cuda::OptionalCUDAGuard guard(x.device());
  TORCH_CHECK(q.dim() == 2 && q.scalar_type() == at::kByte && q.is_contiguous() && q.size(1) == KD);
  TORCH_CHECK(scale.scalar_type() == at::kFloat && scale.is_contiguous() && scale.numel() == q.size(0));
  TORCH_CHECK(((uintptr_t)q.data_ptr() & 15) == 0);
  const int M = x.size(0), N = q.size(0);
  const int64_t* im; const uint16_t* wp; int64_t ldw;
  check_common(x, idmap, N, wbf, delta, &im, &wp, &ldw);
  ensure_ws(x);
  auto out = torch::empty({M}, x.options().dtype(at::kLong));
  auto st = at::cuda::getCurrentCUDAStream().stream();
  HeadArgs A = make_args(x, N, out.data_ptr<int64_t>(), im, wp, ldw, (float)delta);
  const uint8_t* qp = q.data_ptr<uint8_t>();
  const float* sp = scale.data_ptr<float>();
  const int c = cfg < 0 ? (M <= 2 ? 5 : 0) : (int)cfg;  // micro-bench: R2/4 waves for M<=2, R1/8 waves for M=3,4
  switch (M) {
    case 1: launch_fp8<1>(c, grid, st, A, qp, sp); break;
    case 2: launch_fp8<2>(c, grid, st, A, qp, sp); break;
    case 3: launch_fp8<3>(c, grid, st, A, qp, sp); break;
    case 4: launch_fp8<4>(c, grid, st, A, qp, sp); break;
  }
  return out;
}

// int4 draft head. q int32 [ceil(N/2), 320, 4] (packed pairs), ab int32 [ceil(N/2), 40, 2] (half2 a, b),
// n = number of real rows (N); the rest as for fp8_head_argmax.
torch::Tensor i4_head_argmax_op(torch::Tensor x, torch::Tensor q, torch::Tensor ab, int64_t n, torch::Tensor idmap,
                                torch::Tensor wbf, double delta, int64_t cfg, int64_t grid) {
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const int N = (int)n;
  TORCH_CHECK(q.scalar_type() == at::kInt && q.is_contiguous() && q.numel() == (int64_t)((N + 1) / 2) * KCH * 4);
  TORCH_CHECK(ab.scalar_type() == at::kInt && ab.is_contiguous() && ab.numel() == (int64_t)((N + 1) / 2) * NG * 2);
  TORCH_CHECK(((uintptr_t)q.data_ptr() & 15) == 0 && ((uintptr_t)ab.data_ptr() & 7) == 0);
  const int M = x.size(0);
  const int64_t* im; const uint16_t* wp; int64_t ldw;
  check_common(x, idmap, N, wbf, delta, &im, &wp, &ldw);
  ensure_ws(x);
  auto out = torch::empty({M}, x.options().dtype(at::kLong));
  auto st = at::cuda::getCurrentCUDAStream().stream();
  HeadArgs A = make_args(x, N, out.data_ptr<int64_t>(), im, wp, ldw, (float)delta);
  const u32x4* qp = reinterpret_cast<const u32x4*>(q.data_ptr<int>());
  const u32x2* abp = reinterpret_cast<const u32x2*>(ab.data_ptr<int>());
  const int c = cfg < 0 ? (M <= 2 ? 1 : 0) : (int)cfg;
  switch (M) {
    case 1: launch_i4<1>(c, grid, st, A, qp, abp); break;
    case 2: launch_i4<2>(c, grid, st, A, qp, abp); break;
    case 3: launch_i4<3>(c, grid, st, A, qp, abp); break;
    case 4: launch_i4<4>(c, grid, st, A, qp, abp); break;
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// e4m3fn encode with round-to-nearest-even; |a| already scaled to <= 448
__device__ __forceinline__ uint8_t enc_e4m3(float f) {
  const uint32_t sgn = (__builtin_bit_cast(uint32_t, f) >> 24) & 0x80u;
  float a = fabsf(f);
  if (!(a == a)) return 0x7f;
  a = fminf(a, 448.f);
  uint32_t code;
  if (a < 0.015625f) {  // subnormal: m * 2^-9, m = 0..8 (8 -> smallest normal, same code)
    code = (uint32_t)rintf(a * 512.f);
  } else {
    int e;
    const float fr = frexpf(a, &e);  // a = fr * 2^e, fr in [0.5, 1)
    // a = 2^(e-1) * (1 + mant), mant = 2*fr - 1
    uint32_t m = (uint32_t)rintf((2.f * fr - 1.f) * 8.f);
    int E = e - 1 + 7;
    if (m == 8) { m = 0; ++E; }
    code = ((uint32_t)E << 3) | m;
    if (code > 0x7eu) code = 0x7eu;
  }
  return (uint8_t)(sgn | code);
}

__global__ __launch_bounds__(256) void quant_rows_kernel(const uint16_t* __restrict__ w, int64_t ldw,
                                                         const int64_t* __restrict__ rows, int K,
                                                         uint8_t* __restrict__ q, float* __restrict__ scale) {
  const int n = blockIdx.x;
  const int64_t src = rows ? rows[n] : n;
  const uint16_t* wr = w + src * ldw;
  float mx = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) mx = fmaxf(mx, fabsf(bf2f(wr[k])));
  __shared__ float red[256];
  red[threadIdx.x] = mx;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if ((int)threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  mx = red[0];
  const float sc = mx > 0.f ? mx / 448.f : 1.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) q[(int64_t)n * K + k] = enc_e4m3(bf2f(wr[k]) / sc);
  if (threadIdx.x == 0) scale[n] = sc * 256.f;  // * 2^8: the kernel's f16 view of e4m3 is value * 2^-8
}

// w bf16 [V, K]; rows int64 [N] (empty = first N=V rows) -> (q uint8 [N, K], scale fp32 [N])
std::vector<torch::Tensor> quantize_rows_fp8(torch::Tensor w, torch::Tensor rows, int64_t n) {
  const at::cuda::OptionalCUDAGuard guard(w.device());
  TORCH_CHECK(w.dim() == 2 && w.scalar_type() == at::kBFloat16 && w.stride(1) == 1);
  const int K = w.size(1);
  const int64_t* rp = nullptr;
  int N = (int)n;
  if (rows.numel() > 0) {
    TORCH_CHECK(rows.scalar_type() == at::kLong && rows.is_contiguous());
    rp = rows.data_ptr<int64_t>();
    N = rows.numel();
  }
  TORCH_CHECK(N >= 1 && N <= w.size(0));
  auto q = torch::empty({N, K}, w.options().dtype(at::kByte));
  auto s = torch::empty({N}, w.options().dtype(at::kFloat));
  hipLaunchKernelGGL(quant_rows_kernel, dim3(N), dim3(256), 0, at::cuda::getCurrentCUDAStream().stream(),
                     reinterpret_cast<const uint16_t*>(w.data_ptr()), w.stride(0), rp, K, q.data_ptr<uint8_t>(),
                     s.data_ptr<float>());
  return {q, s};
}

// ---------------------------------------------------------------------------------------------
// Greedy verify (one thread per request). For request r with logits rows [s, e):
//   drafts d_i = input_ids[logits_indices[s + i + 1]], i < e - s - 1; target t_i = amax[s + i].
//   accept while d_i == t_i (and d_i >= 0); sampled[r, i] = t_i for every emitted position (an
//   accepted draft equals t_i), i.e. tokens t_0 .. t_a where a = number of accepted drafts; the last
//   one is the correction (first mismatch) or the bonus token (all accepted). num_sampled = a + 1.
// Identical to vLLM's greedy rejection (accept iff draft == target argmax, store the argmax on
// rejection, bonus = argmax of the last row), with the argmax taken from the lm_head GEMV epilogue.
template <typename IdT>
__global__ void greedy_verify_kernel(const int64_t* __restrict__ amax, const IdT* __restrict__ input_ids,
                                     const int64_t* __restrict__ logits_indices, const int* __restrict__ cu,
                                     int num_reqs, int64_t* __restrict__ sampled, int sstride,
                                     int* __restrict__ num_sampled) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= num_reqs) return;
  const int s = cu[r], e = cu[r + 1];
  const int nd = e - s - 1;
  int a = 0;
  for (; a < nd; ++a) {
    const int64_t d = (int64_t)input_ids[logits_indices[s + a + 1]];
    const int64_t t = amax[s + a];
    sampled[(int64_t)r * sstride + a] = t;
    if (d < 0 || d != t) break;
  }
  if (a == nd) sampled[(int64_t)r * sstride + nd] = amax[s + nd];
  num_sampled[r] = a + 1;
}

std::vector<torch::Tensor> greedy_verify(torch::Tensor amax, torch::Tensor input_ids, torch::Tensor logits_indices,
                                         torch::Tensor cu_num_logits, int64_t num_spec) {
  const at::cuda::OptionalCUDAGuard guard(amax.device());
  TORCH_CHECK(amax.scalar_type() == at::kLong && amax.is_contiguous());
  TORCH_CHECK(logits_indices.scalar_type() == at::kLong && logits_indices.is_contiguous());
  TORCH_CHECK(cu_num_logits.scalar_type() == at::kInt && cu_num_logits.is_contiguous());
  TORCH_CHECK(input_ids.is_contiguous());
  const int R = cu_num_logits.numel() - 1;
  auto sampled = torch::empty({R, num_spec + 1}, amax.options());
  auto ns = torch::empty({R}, amax.options().dtype(at::kInt));
  if (R <= 0) return {sampled, ns};
  auto st = at::cuda::getCurrentCUDAStream().stream();
  const int tb = 64, nb = (R + tb - 1) / tb;
  if (input_ids.scalar_type() == at::kInt)
    hipLaunchKernelGGL(greedy_verify_kernel<int>, dim3(nb), dim3(tb), 0, st, amax.data_ptr<int64_t>(),
                       input_ids.data_ptr<int>(), logits_indices.data_ptr<int64_t>(), cu_num_logits.data_ptr<int>(),
                       R, sampled.data_ptr<int64_t>(), (int)sampled.stride(0), ns.data_ptr<int>());
  else {
    TORCH_CHECK(input_ids.scalar_type() == at::kLong);
    hipLaunchKernelGGL(greedy_verify_kernel<int64_t>, dim3(nb), dim3(tb), 0, st, amax.data_ptr<int64_t>(),
                       input_ids.data_ptr<int64_t>(), logits_indices.data_ptr<int64_t>(),
                       cu_num_logits.data_ptr<int>(), R, sampled.data_ptr<int64_t>(), (int)sampled.stride(0),
                       ns.data_ptr<int>());
  }
  return {sampled, ns};
}

TORCH_LIBRARY(vopt_spec, m) {
  m.def("fp8_head_argmax(Tensor x, Tensor q, Tensor scale, Tensor idmap, Tensor wbf, float delta=0.0, int cfg=-1, int grid=0) -> Tensor");
  m.def("i4_head_argmax(Tensor x, Tensor q, Tensor ab, int n, Tensor idmap, Tensor wbf, float delta=0.0, int cfg=-1, int grid=0) -> Tensor");
  m.def("quantize_rows_fp8(Tensor w, Tensor rows, int n) -> Tensor[]");
  m.def("greedy_verify(Tensor amax, Tensor input_ids, Tensor logits_indices, Tensor cu_num_logits, int num_spec) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(vopt_spec, CUDA, m) {
  m.impl("fp8_head_argmax", &fp8_head_argmax_op);
  m.impl("i4_head_argmax", &i4_head_argmax_op);
  m.impl("quantize_rows_fp8", &quantize_rows_fp8);
  m.impl("greedy_verify", &greedy_verify);
}
