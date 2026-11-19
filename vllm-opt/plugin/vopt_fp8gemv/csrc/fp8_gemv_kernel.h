// Weight-only FP8 (e4m3fn, 128x128 block scales) GEMV for small M on gfx90a (MI210).
//   y[m, n] = sum_k x[m, k] * w[n, k] * s[n / 128, k / 128],   x / y bf16, fp32 accumulation.
//
// Weight layout (lossless load-time re-layout, see fp8_gemv.py:relayout):
//   tiled[N/64][K/16][64][16] bytes: tile (g, c) = rows 64g..64g+63, k 16c..16c+15, row-major
//   inside the tile. A wave reads one tile (1 KB, perfectly coalesced) per 16-byte load; lane l
//   owns row 64g + l, so every lane accumulates its own row: no cross-lane reduction, and the
//   activation chunk x[m, 16c..16c+15] is wave-uniform -> scalar loads into SGPRs.
//   The 128x128 scale is wave-uniform too (a wave covers 64 or 128 rows of one 128-row block).
//
// Exact e4m3 -> f16: put byte b in the high byte of a 16-bit half (b << 8), arithmetic shift
//   right by 1 and clear bit 14: f16 bits = sign << 15 | (b & 0x7f) << 7, i.e. value * 2^-8
//   for normals AND subnormals (f16 subnormal range covers e4m3 subnormals exactly). The 2^8
//   is folded into the block scale. NaN (0x7f / 0xff) is not reproduced (checked at load).
//   f16 * x(f32) accumulation uses v_fma_mix_f32 (f16 operand converted exactly in the FMA).
//
// Work split: a block = S waves on the same 64*R rows; wave w handles a contiguous range of
//   128-wide k blocks ("sb"); partials are reduced through LDS. Optionally B blocks share a row
//   group (split-K across blocks): fp32 partials go to a persistent workspace and the last
//   arriving block (atomic counter, self-resetting -> graph-replay safe) sums them in a fixed
//   order (deterministic) and writes bf16.
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include <type_traits>

template <int I, int N, typename F>
__device__ __forceinline__ void static_for(F&& f) {
  if constexpr (I < N) {
    f(std::integral_constant<int, I>{});
    static_for<I + 1, N>(f);
  }
}

typedef short v2i16 __attribute__((ext_vector_type(2)));
typedef unsigned int v4u __attribute__((ext_vector_type(4)));

__device__ __forceinline__ uint32_t dec2(uint32_t u, uint32_t sel) {
  // sel picks two bytes of u into the high byte of each 16-bit half (low byte = 0)
  const uint32_t t = __builtin_amdgcn_perm(0u, u, sel);
  v2i16 v = __builtin_bit_cast(v2i16, t);
  v = v >> (v2i16){1, 1};
  return __builtin_bit_cast(uint32_t, v) & 0xbfffbfffu;
}
__device__ __forceinline__ float lo16(uint32_t f) {
  return (float)__builtin_bit_cast(_Float16, (unsigned short)(f & 0xffffu));
}
__device__ __forceinline__ float hi16(uint32_t f) {
  return (float)__builtin_bit_cast(_Float16, (unsigned short)(f >> 16));
}
__device__ __forceinline__ uint16_t f2bf(float f) {
  uint32_t u = __builtin_bit_cast(uint32_t, f);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

#ifndef VOPT_LAUNDER
#define VOPT_LAUNDER 0
#endif
#ifndef VOPT_SB_MAX_M
#define VOPT_SB_MAX_M 2  // M <= this: sched_barrier between chunks (keeps loads/compute in order)
#endif
#ifndef VOPT_BMAJOR
#define VOPT_BMAJOR 0
#endif
#ifndef VOPT_NT
#define VOPT_NT 1
#endif
template <int R, int K>
__device__ __forceinline__ void load_sb(v4u (&buf)[R][8], const uint8_t* __restrict__ wg, int lane,
                                        int sb) {
  // wg: wave-uniform base of this wave's first 64-row group; tile (g, c) at (g*K/16 + c) KB
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const uint8_t* wu = wg + ((size_t)r * (K / 16) + sb * 8) * 1024;  // uniform
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const v4u* p = reinterpret_cast<const v4u*>(wu + j * 1024 + lane * 16);
#if VOPT_NT
      buf[r][j] = __builtin_nontemporal_load(p);
#else
      buf[r][j] = *p;
#endif
    }
  }
}

// M <= 2: v_fma_mix_f32 (f16 weight half selected by op_sel, converted inside the FMA),
//         x as f32 in SGPRs: 1.5 (decode) + M VALU per weight.
// M >= 3: weights converted to f32 once (v_cvt_f32_f16), then v_pk_fma_f32 over k pairs
//         with the x pair (x[2d], x[2d+1]) from an SGPR pair: 2.5 + M/2 VALU per weight.
typedef float v2f __attribute__((ext_vector_type(2)));
typedef unsigned int v8u __attribute__((ext_vector_type(8)));
#ifndef VOPT_PK_MIN_M
#define VOPT_PK_MIN_M 3  // M >= this: cvt + v_pk_fma_f32 path, else v_fma_mix_f32
#endif
#ifndef VOPT_ASM_XLOAD_MIN_M
#define VOPT_ASM_XLOAD_MIN_M 3  // M >= this: x chunk loads via inline asm (no hoisting / spills)
#endif
// s_load_dwordx8 of x[row m0 (+1)] chunk at byte offset OFF (+OFF2), waited inside the asm:
// the compiler cannot hoist or batch these loads, so only one chunk of x is live in SGPRs.
template <int OFF>
__device__ __forceinline__ v8u sload1(const uint32_t* p) {
  v8u r;
  asm volatile("s_load_dwordx8 %0, %1, %2\n\ts_waitcnt lgkmcnt(0)" : "=s"(r) : "s"(p), "i"(OFF) : "memory");
  return r;
}
template <int OFF, int OFF2>
__device__ __forceinline__ void sload2(const uint32_t* p, v8u& a, v8u& b) {
  asm volatile("s_load_dwordx8 %0, %2, %3\n\ts_load_dwordx8 %1, %2, %4\n\ts_waitcnt lgkmcnt(0)"
               : "=&s"(a), "=&s"(b) : "s"(p), "i"(OFF), "i"(OFF2) : "memory");
}

template <int M, int R, int K>
__device__ __forceinline__ void compute_sb(const v4u (&buf)[R][8], const uint32_t* __restrict__ x,
                                           const float* __restrict__ srow, int sb,
                                           float (&acc)[R][M]) {
  constexpr bool PK = M >= VOPT_PK_MIN_M;
  constexpr int MG = M <= 2 ? M : 2;  // rows of x live in SGPRs at a time
  float p[R][M];
  v2f p2[R][M];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) { p[r][m] = 0.f; p2[r][m] = (v2f){0.f, 0.f}; }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t* xc = x + sb * 64 + j * 8;  // uniform; rows at immediate offsets m*K/2
#if VOPT_LAUNDER
    asm volatile("" : "+s"(xc));  // opaque per chunk: x loads cannot be hoisted across chunks
#endif
    uint32_t f[R][8];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const uint32_t q[4] = {buf[r][j].x, buf[r][j].y, buf[r][j].z, buf[r][j].w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        f[r][2 * i] = dec2(q[i], 0x010c000cu);      // bytes 0,1 -> k 4i, 4i+1
        f[r][2 * i + 1] = dec2(q[i], 0x030c020cu);  // bytes 2,3 -> k 4i+2, 4i+3
      }
    }
    v2f wf[R][8];
    if constexpr (PK) {
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int d = 0; d < 8; ++d) wf[r][d] = (v2f){lo16(f[r][d]), hi16(f[r][d])};
    }
    static_for<0, (M + MG - 1) / MG>([&](auto gi) {
      constexpr int m0 = decltype(gi)::value * MG;
      uint32_t xs[MG][8];
      if constexpr (M >= VOPT_ASM_XLOAD_MIN_M) {
        v8u a, b;
        if constexpr (MG == 2) {
          if constexpr (m0 + 1 < M) sload2<m0 * K * 2, (m0 + 1) * K * 2>(xc, a, b);
          else { a = sload1<m0 * K * 2>(xc); b = a; }
        } else {
          a = sload1<m0 * K * 2>(xc); b = a;
        }
#pragma unroll
        for (int d = 0; d < 8; ++d) { xs[0][d] = a[d]; if constexpr (MG > 1) xs[MG - 1][d] = b[d]; }
      } else {
#pragma unroll
        for (int mm = 0; mm < MG; ++mm)
#pragma unroll
          for (int d = 0; d < 8; ++d)
            xs[mm][d] = (m0 + mm < M) ? xc[(m0 + mm) * (K / 2) + d] : 0u;
      }
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int mm = 0; mm < MG; ++mm) {
          if (m0 + mm >= M) continue;  // (compile-time)
          if constexpr (PK) {
            v2f a = p2[r][m0 + mm];
#pragma unroll
            for (int d = 0; d < 8; ++d) {
              const v2f xv = {__builtin_bit_cast(float, xs[mm][d] << 16),
                              __builtin_bit_cast(float, xs[mm][d] & 0xffff0000u)};
              a = __builtin_elementwise_fma(wf[r][d], xv, a);
            }
            p2[r][m0 + mm] = a;
          } else {
            float a = p[r][m0 + mm];
#pragma unroll
            for (int d = 0; d < 8; ++d) {
              a = fmaf(lo16(f[r][d]), __builtin_bit_cast(float, xs[mm][d] << 16), a);
              a = fmaf(hi16(f[r][d]), __builtin_bit_cast(float, xs[mm][d] & 0xffff0000u), a);
            }
            p[r][m0 + mm] = a;
          }
        }
    });
    if constexpr (M <= VOPT_SB_MAX_M) __builtin_amdgcn_sched_barrier(0);
  }
  const float sc = srow[sb] * 256.f;
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float v = PK ? (p2[r][m].x + p2[r][m].y) : p[r][m];
      acc[r][m] = fmaf(v, sc, acc[r][m]);
    }
}

// ---------------------------------------------------------------------------------------------
// Multi-column path (M >= VOPT_MFMA_MIN_M), per column bitwise identical to the M=1 kernel.
//
// The M=1 kernel computes, per output row n and 128-wide k block sb (k = 128 sb + t):
//   p = 0;  for t = 0..127: p = fma(f32(w16[n, k]), x[k], p);   acc = fma(p, s[sb] * 256, acc)
// followed by the fixed (S, B) split-K reductions. v_mfma_f32_4x4x1f32 computes every output
// element as a single correctly rounded fma (D = A*B + C, one k), bitwise equal to fmaf incl.
// denormals (probe/mfma_fma.hip), so a chain of 4x4x1 MFMAs over t = 0..127 reproduces p exactly
// for 4 columns at once:
//   srcA (the 4 "rows" of each 4x4 block) = x[4c + i][k]: broadcast to all 16 blocks with
//        CBSZ=4 from block ABID. Lane 4b+i holds x[4c+i][128 sb + 8b + v] in register v (0..7),
//        so (v, ABID) = (k % 8, k / 8): one 16-byte x load + 8 bf16->f32 shifts per sb and group.
//   srcB (the 4 "columns" of block b) = the weight of row 4b+j, i.e. lane l supplies its own
//        row l: the same tile layout / decode as M=1 (f16 -> f32 conversion is exact).
//   D: VGPR i, lane l = (row l, column 4c+i) -> the per-lane acc[r][m] layout of the M=1 kernel.
// The (S, B) split and every reduction are unchanged, so with the M=1 (S, B) of a shape every
// output column equals the M=1 call on that row of x. R (rows per wave) does not enter the
// per-row arithmetic and may differ per M.
typedef float v4f __attribute__((ext_vector_type(4)));
#ifndef VOPT_MFMA_MIN_M
#define VOPT_MFMA_MIN_M 3  // M = 2: the M=1 fma_mix chain kernel is faster (and exact with the M=1 (S, B))
#endif

template <int M, int K>
__device__ __forceinline__ void load_x_mfma(v4u (&xr)[(M + 3) / 4], const uint32_t* __restrict__ x,
                                            int lane, int sb) {
#pragma unroll
  for (int c = 0; c < (M + 3) / 4; ++c) {
    int m = c * 4 + (lane & 3);
    m = m < M ? m : M - 1;  // padding columns read a valid row; their outputs are dropped
    xr[c] = *reinterpret_cast<const v4u*>(x + (size_t)m * (K / 2) + sb * 64 + (lane >> 2) * 4);
  }
}

#ifndef VOPT_MFMA_XPF
#define VOPT_MFMA_XPF 1  // 1: x chunk of the next sb prefetched with its weights; 0: loaded in compute
#endif
template <int M, int R, int K>
__device__ __forceinline__ void compute_sb_mfma(const v4u (&buf)[R][8], v4u (&xr)[(M + 3) / 4],
                                                const uint32_t* __restrict__ x, int lane,
                                                const float* __restrict__ srow, int sb,
                                                v4f (&acc)[R][(M + 3) / 4]) {
  constexpr int NCG = (M + 3) / 4;
  if constexpr (!VOPT_MFMA_XPF) load_x_mfma<M, K>(xr, x, lane, sb);
  float xa[NCG][8];
#pragma unroll
  for (int c = 0; c < NCG; ++c) {
    const uint32_t q[4] = {xr[c].x, xr[c].y, xr[c].z, xr[c].w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      xa[c][2 * i] = __builtin_bit_cast(float, q[i] << 16);
      xa[c][2 * i + 1] = __builtin_bit_cast(float, q[i] & 0xffff0000u);
    }
  }
  v4f p[R][NCG];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int c = 0; c < NCG; ++c) p[r][c] = (v4f){0.f, 0.f, 0.f, 0.f};
  static_for<0, 8>([&](auto jj) {
    constexpr int j = decltype(jj)::value;
    float wf[R][16];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const uint32_t q[4] = {buf[r][j].x, buf[r][j].y, buf[r][j].z, buf[r][j].w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t f0 = dec2(q[i], 0x010c000cu);  // k 4i, 4i+1
        const uint32_t f1 = dec2(q[i], 0x030c020cu);  // k 4i+2, 4i+3
        wf[r][4 * i] = lo16(f0);
        wf[r][4 * i + 1] = hi16(f0);
        wf[r][4 * i + 2] = lo16(f1);
        wf[r][4 * i + 3] = hi16(f1);
      }
    }
    static_for<0, 16>([&](auto tt) {
      constexpr int t = decltype(tt)::value;
      constexpr int k = 16 * j + t;
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NCG; ++c)
#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
          p[r][c] = __builtin_amdgcn_mfma_f32_4x4x1f32(xa[c][k & 7], wf[r][t], p[r][c], 4, k >> 3, 0);
#else
          __builtin_trap();  // no MFMA on this arch (the op is only installed on gfx90a)
#endif
    });
  });
  const float sc = srow[sb] * 256.f;
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int c = 0; c < NCG; ++c)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[r][c][i] = fmaf(p[r][c][i], sc, acc[r][c][i]);
}

#ifndef VOPT_WPE
#define VOPT_WPE 4
#endif
#ifndef VOPT_WPE_MFMA
#define VOPT_WPE_MFMA 4  // occupancy hint of the multi-column MFMA instantiations (M=1 keeps VOPT_WPE)
#endif
// ONE (MFMA path only): every wave has at most one 128-k block (K/128/B <= S) -> single weight
// buffer, no loop (fewer VGPRs, higher occupancy). Same arithmetic.
template <int M, int R, int K, bool ONE = false>
__global__ __launch_bounds__(512) __attribute__((amdgpu_waves_per_eu(M >= VOPT_MFMA_MIN_M ? VOPT_WPE_MFMA : VOPT_WPE))) void fp8_gemv_kernel(
    const uint32_t* __restrict__ x,                  // [M, K] bf16 contiguous
    const uint8_t* __restrict__ w,                   // tiled [N/64][K/16][64][16]
    const float* __restrict__ s,                     // [N/128, K/128] fp32
    uint16_t* __restrict__ y, int ldy,               // [M, N] bf16
    float* __restrict__ ws, int* __restrict__ cnt,   // split-K workspace / counters (B > 1)
    int B, int S) {
  extern __shared__ float red[];
  __shared__ int s_last;
  constexpr int RW = 64 * R;  // rows per block
  const int lane = threadIdx.x & 63;
  const int wv = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
  // block order: rg-major (default) or b-major (blocks resident together share a K slice)
#if VOPT_BMAJOR
  const int ngroups = gridDim.x / B;
  const int b = blockIdx.x / ngroups, rg = blockIdx.x - b * ngroups;
#else
  const int rg = blockIdx.x / B, b = blockIdx.x - rg * B;
#endif
  constexpr int nsb = K >> 7;
  const int sbb = nsb / B;
  const int sb0 = b * sbb + (wv * sbb) / S, sb1 = b * sbb + ((wv + 1) * sbb) / S;
  const int rb = (rg * RW) >> 7;
  const float* srow = s + (size_t)rb * nsb;
  const uint8_t* wg = w + (size_t)(rg * R) * (K / 16) * 1024;

  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;

  if constexpr (M >= VOPT_MFMA_MIN_M) {
    // multi-column MFMA path: bitwise the same fp32 fma chain per (row, column) as M=1
    constexpr int NCG = (M + 3) / 4;
    v4f pacc[R][NCG];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int c = 0; c < NCG; ++c) pacc[r][c] = (v4f){0.f, 0.f, 0.f, 0.f};
    if constexpr (ONE) {
      if (sb0 < sb1) {
        v4u A[R][8];
        v4u XA[NCG];
        load_sb<R, K>(A, wg, lane, sb0);
        if constexpr (VOPT_MFMA_XPF) load_x_mfma<M, K>(XA, x, lane, sb0);
        compute_sb_mfma<M, R, K>(A, XA, x, lane, srow, sb0, pacc);
      }
    } else if (sb0 < sb1) {
      v4u A[R][8], Bf[R][8];
#if VOPT_MFMA_XPF
      v4u XA[NCG], XB[NCG];
      load_x_mfma<M, K>(XA, x, lane, sb0);
#define XLD(buf, sbx) load_x_mfma<M, K>(buf, x, lane, sbx)
#else
      v4u XA[NCG], XB[NCG];  // loaded inside compute (x is L2-resident): fewer live VGPRs
#define XLD(buf, sbx) (void)0
#endif
      load_sb<R, K>(A, wg, lane, sb0);
      for (int sb = sb0; sb < sb1; sb += 2) {
        if (sb + 1 < sb1) { load_sb<R, K>(Bf, wg, lane, sb + 1); XLD(XB, sb + 1); }
        compute_sb_mfma<M, R, K>(A, XA, x, lane, srow, sb, pacc);
        if (sb + 1 >= sb1) break;
        if (sb + 2 < sb1) { load_sb<R, K>(A, wg, lane, sb + 2); XLD(XA, sb + 2); }
        compute_sb_mfma<M, R, K>(Bf, XB, x, lane, srow, sb + 1, pacc);
      }
#undef XLD
    }
    // MFMA D layout: VGPR i of column group c, lane l = (row l, column 4c+i) -> same as acc[r][m]
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int m = 0; m < M; ++m) acc[r][m] = pacc[r][m >> 2][m & 3];
  } else if constexpr (ONE) {
    if (sb0 < sb1) {
      v4u A[R][8];
      load_sb<R, K>(A, wg, lane, sb0);
      compute_sb<M, R, K>(A, x, srow, sb0, acc);
    }
  } else if (sb0 < sb1) {
    v4u A[R][8], Bf[R][8];
    load_sb<R, K>(A, wg, lane, sb0);
    for (int sb = sb0; sb < sb1; sb += 2) {
      if (sb + 1 < sb1) load_sb<R, K>(Bf, wg, lane, sb + 1);
      compute_sb<M, R, K>(A, x, srow, sb, acc);
      if (sb + 1 >= sb1) break;
      if (sb + 2 < sb1) load_sb<R, K>(A, wg, lane, sb + 2);
      compute_sb<M, R, K>(Bf, x, srow, sb + 1, acc);
    }
  }

  const int row0 = rg * RW;
  float* wsb = ws + ((size_t)rg * B + b) * (M * RW);
  if (S == 1) {
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int m = 0; m < M; ++m) {
        if (B == 1) y[(size_t)m * ldy + row0 + r * 64 + lane] = f2bf(acc[r][m]);
        else wsb[m * RW + r * 64 + lane] = acc[r][m];
      }
  } else {
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int m = 0; m < M; ++m) red[(wv * M + m) * RW + r * 64 + lane] = acc[r][m];
    __syncthreads();
    for (int o = threadIdx.x; o < M * RW; o += S * 64) {
      float v = 0.f;
      for (int i = 0; i < S; ++i) v += red[i * (M * RW) + o];
      const int m = o / RW, rr = o - m * RW;
      if (B == 1) y[(size_t)m * ldy + row0 + rr] = f2bf(v);
      else wsb[o] = v;
    }
  }
  if (B == 1) return;
  // split-K across blocks: last block of this row group reduces in fixed order
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const int prev = atomicAdd(&cnt[rg], 1);
    s_last = (prev == B - 1);
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  const volatile float* wsr = ws + (size_t)rg * B * (M * RW);
  for (int o = threadIdx.x; o < M * RW; o += S * 64) {
    float v = 0.f;
    for (int i = 0; i < B; ++i) v += wsr[(size_t)i * (M * RW) + o];
    const int m = o / RW, rr = o - m * RW;
    y[(size_t)m * ldy + row0 + rr] = f2bf(v);
  }
  if (threadIdx.x == 0) cnt[rg] = 0;
}


// host-side launcher for one M (defined in fp8_gemv_m<M>.cu so the 8 M variants compile in parallel)
template <int M>
void launch_fp8_gemv(dim3 grid, dim3 block, size_t shm, hipStream_t st, const uint32_t* x,
                     const uint8_t* w, const float* s, uint16_t* y, int N, float* ws, int* cnt,
                     int K, int R, int B, int S) {
#define L(RR, KK, O)   hipLaunchKernelGGL((fp8_gemv_kernel<M, RR, KK, O>), grid, block, shm, st, x, w, s, y, N, ws, cnt, B, S)
#define LK(RR, O)   if (K == 5120) L(RR, 5120, O); else if (K == 6144) L(RR, 6144, O); else L(RR, 17408, O);
#ifndef VOPT_MFMA_ONE
#define VOPT_MFMA_ONE 0  // 1: single-buffer MFMA variant when K/128/B <= S (45 VGPRs; measured slower on K=5120 shapes)
#endif
#ifndef VOPT_CHAIN_ONE
#define VOPT_CHAIN_ONE 0  // same for the fma_mix path (M = 1, 2); 0 = deploy-v1 kernel as is
#endif
  constexpr bool mf = M >= VOPT_MFMA_MIN_M;
  const bool one = (mf ? VOPT_MFMA_ONE : VOPT_CHAIN_ONE) && (K / 128 / B) <= S;
  if (R == 1) { if (one) { LK(1, true) } else { LK(1, false) } }
  else { LK(2, false) }
#undef LK
#undef L
}
