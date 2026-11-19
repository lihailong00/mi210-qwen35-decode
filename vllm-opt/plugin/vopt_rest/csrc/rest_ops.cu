// vopt_rest: fused small-op kernels for Qwen3.5/3.8 decode on gfx90a (MI210).
//   gemma_rms_norm      : y = bf16(x * rsqrt(mean(x^2)+eps) * (1+w))           (GemmaRMSNorm)
//   gemma_add_rms_norm  : s = x + r (fp32); r' = bf16(s); y = bf16(norm(s)*(1+w)) (fused_add)
//   silu_mul            : y = bf16(bf16(silu(g)) * u)                           (SiluAndMul)
//   sigmoid_mul         : y = bf16(a * bf16(sigmoid(g)))                        (attn output gate)
//   gdn_decode          : causal conv1d update (width 4) + SiLU + q/k L2 norm + gating
//                         + gated delta rule on the fp32 state + gated RMSNorm (silu(z)),
//                         one launch per GDN layer, single decode token per request.
//   gdn_spec            : the same per token for R requests x T speculative-verify tokens (MTP),
//                         bitwise equal to gdn_decode run token by token; writes the state after
//                         every token into vLLM's per-position slots (rollback contract)
//   attn_pseudo_meta    : per-token pseudo-sequence metadata for exact multi-query attention
// All numerics follow the vLLM reference kernels (fp32 math, bf16 rounding at the same
// points as the reference stores); only reduction order differs.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdint.h>

typedef uint16_t bf16_t;

__device__ __forceinline__ float bf2f(uint32_t b) { return __builtin_bit_cast(float, b << 16); }
// round-to-nearest-even to bf16; result in the upper 16 bits (branchless; a NaN input whose
// payload sits only in the low 16 bits could round to inf - never produced by finite math here)
__device__ __forceinline__ uint32_t rne_hi(float f) {
  const uint32_t u = __builtin_bit_cast(uint32_t, f);
  return u + 0x7fffu + ((u >> 16) & 1u);
}
__device__ __forceinline__ uint32_t f2bf(float f) { return rne_hi(f) >> 16; }
__device__ __forceinline__ float rbf(float f) { return __builtin_bit_cast(float, rne_hi(f) & 0xffff0000u); }
__device__ __forceinline__ float lo(uint32_t w) { return __builtin_bit_cast(float, w << 16); }
__device__ __forceinline__ float hi(uint32_t w) { return __builtin_bit_cast(float, w & 0xffff0000u); }
// pack bf16(a) into the low half and bf16(b) into the high half with one v_perm_b32
__device__ __forceinline__ uint32_t pack(float a, float b) {
  return __builtin_amdgcn_perm(rne_hi(b), rne_hi(a), 0x07060302u);
}

template <int CTRL, int ROWMASK = 0xf>
__device__ __forceinline__ float dpp(float v) {
  return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), CTRL,
                                                                ROWMASK, 0xf, false));
}
// sum over each group of 8 consecutive lanes, result in all 8 lanes (DPP, no LDS)
__device__ __forceinline__ float sum8(float v) {
  v += dpp<0xb1>(v);   // quad_perm [1,0,3,2]
  v += dpp<0x4e>(v);   // quad_perm [2,3,0,1]
  v += dpp<0x141>(v);  // row_half_mirror
  return v;
}
// full wave (64 lanes) sum, returned uniformly (gfx9 DPP row broadcasts + readlane 63)
__device__ __forceinline__ float wave_sum(float v) {
  v = sum8(v);
  v += dpp<0x140>(v);         // row_mirror: 16-lane sums
  v += dpp<0x142, 0xa>(v);    // row_bcast15 -> rows 1,3 hold 32-lane sums
  v += dpp<0x143, 0xc>(v);    // row_bcast31 -> row 3 holds the total
  return __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), 63));
}

// ------------------------------------------------------------------------------------
// Gemma RMSNorm (optionally fused residual add). One block per row, 8 elements / thread
// per chunk, NCH chunks per thread.
template <int NCH, bool RES>
__global__ __launch_bounds__(1024) void gemma_rms_k(const bf16_t* __restrict__ x, int64_t ldx,
                                                    const bf16_t* __restrict__ r, int64_t ldr,
                                                    const bf16_t* __restrict__ w,
                                                    bf16_t* __restrict__ y, int64_t ldy,
                                                    bf16_t* __restrict__ ro, int64_t ldro, int D,
                                                    float eps) {
  __shared__ float sh[16];
  const int row = blockIdx.x, T = blockDim.x, nc = D >> 3;
  const bf16_t* xr = x + (int64_t)row * ldx;
  float v[NCH][8];
  uint4 xa[NCH], ra[NCH], wa[NCH];
#pragma unroll
  for (int i = 0; i < NCH; ++i) {
    const int c = threadIdx.x + i * T;
    if (c < nc) {
      xa[i] = *reinterpret_cast<const uint4*>(xr + c * 8);
      wa[i] = *reinterpret_cast<const uint4*>(w + c * 8);
      if (RES) ra[i] = *reinterpret_cast<const uint4*>(r + (int64_t)row * ldr + c * 8);
    }
  }
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < NCH; ++i) {
    const int c = threadIdx.x + i * T;
    if (c < nc) {
      const uint32_t xw[4] = {xa[i].x, xa[i].y, xa[i].z, xa[i].w};
#pragma unroll
      for (int e = 0; e < 4; ++e) { v[i][2 * e] = lo(xw[e]); v[i][2 * e + 1] = hi(xw[e]); }
      if (RES) {
        const uint32_t rw[4] = {ra[i].x, ra[i].y, ra[i].z, ra[i].w};
#pragma unroll
        for (int e = 0; e < 4; ++e) { v[i][2 * e] += lo(rw[e]); v[i][2 * e + 1] += hi(rw[e]); }
        uint4 o;
        o.x = pack(v[i][0], v[i][1]); o.y = pack(v[i][2], v[i][3]);
        o.z = pack(v[i][4], v[i][5]); o.w = pack(v[i][6], v[i][7]);
        *reinterpret_cast<uint4*>(ro + (int64_t)row * ldro + c * 8) = o;
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += v[i][e] * v[i][e];
    }
  }
  ss = wave_sum(ss);
  if ((threadIdx.x & 63) == 0) sh[threadIdx.x >> 6] = ss;
  __syncthreads();
  float tot = 0.f;
  for (int i = 0; i < (T >> 6); ++i) tot += sh[i];
  const float rs = rsqrtf(tot / (float)D + eps);
#pragma unroll
  for (int i = 0; i < NCH; ++i) {
    const int c = threadIdx.x + i * T;
    if (c < nc) {
      const uint32_t ww[4] = {wa[i].x, wa[i].y, wa[i].z, wa[i].w};
      float o[8];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        o[2 * e] = (v[i][2 * e] * rs) * (lo(ww[e]) + 1.0f);
        o[2 * e + 1] = (v[i][2 * e + 1] * rs) * (hi(ww[e]) + 1.0f);
      }
      uint4 p;
      p.x = pack(o[0], o[1]); p.y = pack(o[2], o[3]); p.z = pack(o[4], o[5]); p.w = pack(o[6], o[7]);
      *reinterpret_cast<uint4*>(y + (int64_t)row * ldy + c * 8) = p;
    }
  }
}

// Specialised for a fixed hidden size: T = D/8 threads (D=5120 -> 640 = 10 waves), 8 elements
// per thread, same math and rounding points as gemma_rms_k.
template <int D, bool RES>
__global__ __launch_bounds__(D / 8) void gemma_rms_fixed_k(const bf16_t* __restrict__ x, int ldx,
                                                         const bf16_t* __restrict__ r, int ldr,
                                                         const bf16_t* __restrict__ w,
                                                         bf16_t* __restrict__ y, int ldy,
                                                         bf16_t* __restrict__ ro, int ldro,
                                                         float eps) {
  constexpr int NW = D / 8 / 64;
  __shared__ float sh[NW];
  const int row = blockIdx.x, c = threadIdx.x;
  const uint4 xa = *reinterpret_cast<const uint4*>(x + row * ldx + c * 8);
  const uint4 wa = *reinterpret_cast<const uint4*>(w + c * 8);
  uint4 ra;
  if (RES) ra = *reinterpret_cast<const uint4*>(r + row * ldr + c * 8);
  const uint32_t xw[4] = {xa.x, xa.y, xa.z, xa.w};
  float v[8];
#pragma unroll
  for (int e = 0; e < 4; ++e) { v[2 * e] = lo(xw[e]); v[2 * e + 1] = hi(xw[e]); }
  if (RES) {
    const uint32_t rw[4] = {ra.x, ra.y, ra.z, ra.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) { v[2 * e] += lo(rw[e]); v[2 * e + 1] += hi(rw[e]); }
    *reinterpret_cast<uint4*>(ro + row * ldro + c * 8) =
        make_uint4(pack(v[0], v[1]), pack(v[2], v[3]), pack(v[4], v[5]), pack(v[6], v[7]));
  }
  float ss = 0.f;
#pragma unroll
  for (int e = 0; e < 8; ++e) ss += v[e] * v[e];
  ss = wave_sum(ss);
  if ((c & 63) == 0) sh[c >> 6] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < NW; ++i) tot += sh[i];
  const float rs = rsqrtf(tot / (float)D + eps);
  const uint32_t ww[4] = {wa.x, wa.y, wa.z, wa.w};
  float o[8];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    o[2 * e] = (v[2 * e] * rs) * (lo(ww[e]) + 1.0f);
    o[2 * e + 1] = (v[2 * e + 1] * rs) * (hi(ww[e]) + 1.0f);
  }
  *reinterpret_cast<uint4*>(y + row * ldy + c * 8) =
      make_uint4(pack(o[0], o[1]), pack(o[2], o[3]), pack(o[4], o[5]), pack(o[6], o[7]));
}

static void check_rows(const torch::Tensor& t, int64_t D, const char* n) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16, n, ": bf16 cuda");
  TORCH_CHECK(t.dim() == 2 && t.size(1) == D && t.stride(1) == 1 && t.stride(0) % 8 == 0, n,
              ": [M,D] rows contiguous, 16B aligned");
  TORCH_CHECK(((uintptr_t)t.data_ptr() & 15) == 0, n, ": 16B aligned");
}

static void launch_rms(const torch::Tensor& x, const torch::Tensor* r, const torch::Tensor& w,
                       torch::Tensor& y, torch::Tensor* ro, double eps) {
  const int64_t M = x.size(0), D = x.size(1);
  TORCH_CHECK(D % 8 == 0 && D <= 32768, "D");
  check_rows(x, D, "x");
  check_rows(y, D, "y");
  TORCH_CHECK(w.is_contiguous() && w.numel() == D && w.scalar_type() == at::kBFloat16, "w");
  if (M == 0) return;
  const int nc = (int)(D / 8);
  int T = ((nc + 63) / 64) * 64;
  if (T > 1024) T = 1024;
  const int nch = (nc + T - 1) / T;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto P = [](const torch::Tensor& t) { return reinterpret_cast<bf16_t*>(t.data_ptr()); };
  if (D == 5120 && M * x.stride(0) < (1ll << 31) && (!ro || M * ro->stride(0) < (1ll << 31)) &&
      (!r || M * r->stride(0) < (1ll << 31)) && M * y.stride(0) < (1ll << 31)) {
    if (r) {
      check_rows(*r, D, "residual");
      check_rows(*ro, D, "residual_out");
      gemma_rms_fixed_k<5120, true><<<dim3(M), dim3(640), 0, stream>>>(
          P(x), (int)x.stride(0), P(*r), (int)r->stride(0), P(w), P(y), (int)y.stride(0), P(*ro),
          (int)ro->stride(0), (float)eps);
    } else {
      gemma_rms_fixed_k<5120, false><<<dim3(M), dim3(640), 0, stream>>>(
          P(x), (int)x.stride(0), nullptr, 0, P(w), P(y), (int)y.stride(0), nullptr, 0, (float)eps);
    }
    return;
  }
#define RMS_L(N, R)                                                                          \
  gemma_rms_k<N, R><<<dim3(M), dim3(T), 0, stream>>>(                                        \
      P(x), x.stride(0), r ? P(*r) : nullptr, r ? r->stride(0) : 0, P(w), P(y), y.stride(0), \
      ro ? P(*ro) : nullptr, ro ? ro->stride(0) : 0, (int)D, (float)eps)
  if (r) {
    check_rows(*r, D, "residual");
    check_rows(*ro, D, "residual_out");
    if (nch == 1) RMS_L(1, true); else if (nch == 2) RMS_L(2, true); else RMS_L(4, true);
  } else {
    if (nch == 1) RMS_L(1, false); else if (nch == 2) RMS_L(2, false); else RMS_L(4, false);
  }
#undef RMS_L
}

torch::Tensor gemma_rms_norm(torch::Tensor x, torch::Tensor w, double eps) {
  auto y = torch::empty_like(x, x.options().memory_format(at::MemoryFormat::Contiguous));
  launch_rms(x, nullptr, w, y, nullptr, eps);
  return y;
}

std::vector<torch::Tensor> gemma_add_rms_norm(torch::Tensor x, torch::Tensor r, torch::Tensor w,
                                              double eps) {
  auto y = torch::empty({x.size(0), x.size(1)}, x.options());
  auto ro = torch::empty({x.size(0), x.size(1)}, x.options());
  launch_rms(x, &r, w, y, &ro, eps);
  return {y, ro};
}

// ------------------------------------------------------------------------------------
__global__ __launch_bounds__(256) void silu_mul_k(const bf16_t* __restrict__ x, int64_t ldx,
                                                  bf16_t* __restrict__ y, int64_t ldy, int d,
                                                  int64_t total) {
  // 2 elements per thread
  const int64_t gid = (int64_t)blockIdx.x * 256 + threadIdx.x;
  if (gid >= total) return;
  const int nc = d >> 1;
  const int64_t row = gid / nc;
  const int c = (int)(gid - row * nc);
  const uint32_t g = *reinterpret_cast<const uint32_t*>(x + row * ldx + c * 2);
  const uint32_t u = *reinterpret_cast<const uint32_t*>(x + row * ldx + d + c * 2);
  const float g0 = lo(g), g1 = hi(g);
  const float s0 = rbf(g0 / (1.0f + expf(-g0))), s1 = rbf(g1 / (1.0f + expf(-g1)));
  *reinterpret_cast<uint32_t*>(y + row * ldy + c * 2) = pack(s0 * lo(u), s1 * hi(u));
}

torch::Tensor silu_mul(torch::Tensor x) {
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.stride(1) == 1, "x");
  const int64_t M = x.size(0), d = x.size(1) / 2;
  TORCH_CHECK(d % 8 == 0 && x.stride(0) % 8 == 0 && ((uintptr_t)x.data_ptr() & 15) == 0, "align");
  auto y = torch::empty({M, d}, x.options());
  const int64_t total = M * (d / 2);
  if (total == 0) return y;
  auto stream = at::cuda::getCurrentCUDAStream();
  silu_mul_k<<<dim3((total + 255) / 256), dim3(256), 0, stream>>>(
      reinterpret_cast<const bf16_t*>(x.data_ptr()), x.stride(0),
      reinterpret_cast<bf16_t*>(y.data_ptr()), d, (int)d, total);
  return y;
}

// ------------------------------------------------------------------------------------
__global__ __launch_bounds__(256) void sigmoid_mul_k(const bf16_t* __restrict__ a, int64_t lda,
                                                     const bf16_t* __restrict__ g, int64_t ldg,
                                                     bf16_t* __restrict__ y, int n,
                                                     int64_t total) {
  const int64_t gid = (int64_t)blockIdx.x * 256 + threadIdx.x;
  if (gid >= total) return;
  const int nc = n >> 1;
  const int64_t row = gid / nc;
  const int c = (int)(gid - row * nc);
  const uint32_t av = *reinterpret_cast<const uint32_t*>(a + row * lda + c * 2);
  const uint32_t gv = *reinterpret_cast<const uint32_t*>(g + row * ldg + c * 2);
  const float s0 = rbf(1.0f / (1.0f + expf(-lo(gv))));
  const float s1 = rbf(1.0f / (1.0f + expf(-hi(gv))));
  *reinterpret_cast<uint32_t*>(y + row * (int64_t)n + c * 2) = pack(lo(av) * s0, hi(av) * s1);
}

torch::Tensor sigmoid_mul(torch::Tensor a, torch::Tensor g) {
  TORCH_CHECK(a.dim() == 2 && g.dim() == 2 && a.sizes() == g.sizes(), "shape");
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && g.scalar_type() == at::kBFloat16, "dtype");
  const int64_t M = a.size(0), n = a.size(1);
  TORCH_CHECK(a.stride(1) == 1 && g.stride(1) == 1 && n % 8 == 0 && a.stride(0) % 8 == 0 &&
                  g.stride(0) % 8 == 0 && ((uintptr_t)a.data_ptr() & 15) == 0 &&
                  ((uintptr_t)g.data_ptr() & 15) == 0,
              "align");
  auto y = torch::empty({M, n}, a.options());
  const int64_t total = M * (n / 2);
  if (total == 0) return y;
  auto stream = at::cuda::getCurrentCUDAStream();
  sigmoid_mul_k<<<dim3((total + 255) / 256), dim3(256), 0, stream>>>(
      reinterpret_cast<const bf16_t*>(a.data_ptr()), a.stride(0),
      reinterpret_cast<const bf16_t*>(g.data_ptr()), g.stride(0),
      reinterpret_cast<bf16_t*>(y.data_ptr()), (int)n, total);
  return y;
}

// ------------------------------------------------------------------------------------
// Gated DeltaNet decode, Qwen3.5 layout: H=16 k-heads, HV=48 v-heads, K=V=128, conv width 4.
// mixed row: [q (16x128) | k (16x128) | v (48x128)] (conv input, bf16), z row: 48x128.
// Grid (48 v-heads, tokens); 1024 threads; one block owns a full v-head (128x128 fp32 state),
// 8 threads per state row (16 k each). The gated RMSNorm over the head is done in-block.
// q/k conv state is shared by the 3 v-heads of a k-head: all 3 blocks read the OLD state; the
// last one to arrive (self-resetting counter cnt_qk[t*16+h], no spinning) writes the NEW one.
// The arrival atomic is issued early and its result is only consumed at the end.
#define GDN_H 16
#define GDN_HV 48
#define GDN_D 128
#define GDN_QKV 10240
#define GDN_VOFF 4096

__device__ __forceinline__ float ldf(const void* p, int i, int f32) {
  return f32 ? reinterpret_cast<const float*>(p)[i] : bf2f(reinterpret_cast<const bf16_t*>(p)[i]);
}

// LDS-only barrier (global stores stay in flight; __syncthreads would also wait for them)
__device__ __forceinline__ void lds_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}
typedef float vopt_f4v __attribute__((ext_vector_type(4)));
template <bool NT>
__device__ __forceinline__ void st_state4(float* p, float a, float b, float c, float d) {
  if (NT) {
    vopt_f4v v = {a, b, c, d};
    __builtin_nontemporal_store(v, reinterpret_cast<vopt_f4v*>(p));
  } else {
    *reinterpret_cast<float4*>(p) = make_float4(a, b, c, d);
  }
}

// bit2: non-temporal state stores (streams to HBM instead of leaving dirty L2 lines whose write-back
// slows the next kernel); values are identical either way
template <int DBG>  // debug/profiling only: bit0 skip state load, bit1 skip state store, bit2 NT
__global__ __launch_bounds__(1024) void gdn_decode_k(
    const bf16_t* __restrict__ qkv, int64_t ld_qkv, const bf16_t* __restrict__ z, int64_t ld_z,
    const bf16_t* __restrict__ bb, int64_t ld_b, const bf16_t* __restrict__ aa, int64_t ld_a,
    bf16_t* __restrict__ cst, int64_t cs_seq, int64_t cs_dim, int64_t cs_tok,
    const bf16_t* __restrict__ cw, int64_t cw_dim, int64_t cw_k, const void* A_log, int A_f32,
    const void* dt_bias, int dt_f32, const void* norm_w, int nw_f32, float* __restrict__ ssm,
    int64_t ssm_seq, const void* idx_ptr, int idx64, int n_act, bf16_t* __restrict__ out,
    int* __restrict__ cnt_qk, float scale, float eps) {
  __shared__ float qn[GDN_D], kn[GDN_D], vsh[GDN_D], osh[GDN_D], red[8], gate_sh[2];
  __shared__ int flag;
  const int hv = blockIdx.x, t = blockIdx.y, h = hv / 3, tid = threadIdx.x;
  const int r = tid >> 3, kseg = tid & 7;
  // independent loads first (input row, conv weights, z, gating params)
  const bf16_t* xrow = qkv + (int64_t)t * ld_qkv;
  const int ch = (tid < 128) ? (h * GDN_D + tid)
               : (tid < 256) ? (GDN_H * GDN_D + h * GDN_D + tid - 128)
                             : (GDN_VOFF + hv * GDN_D + (tid & 127));
  const bool conv_thr = tid < 384;
  bf16_t xin = 0, w0 = 0, w1 = 0, w2 = 0, w3 = 0;
  if (conv_thr) {
    xin = xrow[ch];
    const bf16_t* cwr = cw + (int64_t)ch * cw_dim;
    w0 = cwr[0]; w1 = cwr[cw_k]; w2 = cwr[2 * cw_k]; w3 = cwr[3 * cw_k];
  }
  float zz = 0.f, nw = 0.f;
  if (tid < GDN_D) {
    zz = bf2f(z[(int64_t)t * ld_z + hv * GDN_D + tid]);
    nw = ldf(norm_w, tid, nw_f32);
  }
  const bool gate_wave = (tid >> 6) == 6;  // idle during the conv phase
  float av = 0.f, bv = 0.f, dtb = 0.f, alog = 0.f;
  if (gate_wave) {
    av = bf2f(aa[(int64_t)t * ld_a + hv]); bv = bf2f(bb[(int64_t)t * ld_b + hv]);
    dtb = ldf(dt_bias, hv, dt_f32); alog = ldf(A_log, hv, A_f32);
  }
  int64_t idx = 0;
  if (t < n_act)
    idx = idx64 ? reinterpret_cast<const int64_t*>(idx_ptr)[t]
                : (int64_t)reinterpret_cast<const int*>(idx_ptr)[t];
  bf16_t* orow = out + (int64_t)t * (GDN_HV * GDN_D) + hv * GDN_D;
  if (idx <= 0) {  // padded / null slot: zero output, no state update (as the reference)
    if (tid < GDN_D) orow[tid] = 0;
    return;
  }
  // state loads (depend on idx only)
  float* hrow = ssm + idx * ssm_seq + ((int64_t)hv * GDN_D + r) * GDN_D;
  float4 hs[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    hs[j] = (DBG & 1) ? make_float4(0.f, 0.f, 0.f, 0.f)
                      : *reinterpret_cast<const float4*>(hrow + (j * 8 + kseg) * 4);
  // conv state + causal conv (width 4) + SiLU, bf16 product rounding as the Triton reference
  bf16_t* cs = cst + idx * cs_seq;
  bf16_t c0 = 0, c1 = 0, c2 = 0;
  float acc = 0.f;
  if (conv_thr) {
    c0 = cs[ch * cs_dim]; c1 = cs[ch * cs_dim + cs_tok]; c2 = cs[ch * cs_dim + 2 * cs_tok];
    acc += rbf(bf2f(c0) * bf2f(w0));
    acc += rbf(bf2f(c1) * bf2f(w1));
    acc += rbf(bf2f(c2) * bf2f(w2));
    acc += rbf(bf2f(xin) * bf2f(w3));
    acc = rbf(acc / (1.0f + expf(-acc)));
    if (tid >= 256) {  // v channels: this block is their only reader/writer
      vsh[tid - 256] = acc;
      cs[ch * cs_dim] = c1; cs[ch * cs_dim + cs_tok] = c2; cs[ch * cs_dim + 2 * cs_tok] = xin;
    }
  }
  if (gate_wave && (tid & 63) == 0) {  // gating scalars (as the reference kernel)
    const float xg = av + dtb;
    const float sp = (xg <= 20.0f) ? logf(1.0f + expf(xg)) : xg;
    const float g = -expf(alog) * sp;
    gate_sh[0] = expf(g);
    gate_sh[1] = 1.0f / (1.0f + expf(-bv));
  }
  // q/k L2 norm partial sums (waves 0,1 = q; waves 2,3 = k)
  const float ss = wave_sum(acc * acc);
  if ((tid & 63) == 0 && tid < 256) red[tid >> 6] = ss;
  __syncthreads();  // all q/k conv-state reads of this block are complete here
  int old = 0;
  if (tid == 0) old = atomicAdd(&cnt_qk[t * GDN_H + h], 1);  // result consumed at the end
  if (tid < 256) {
    const float tot = (tid < 128) ? (red[0] + red[1]) : (red[2] + red[3]);
    const float vn = acc / sqrtf(tot + 1e-6f);
    if (tid < 128) qn[tid] = vn * scale; else kn[tid - 128] = vn;
  }
  __syncthreads();
  const float decay = gate_sh[0], beta = gate_sh[1];
  // delta rule: row r, k = (j*8+kseg)*4 .. +3
  float kk[16], qq[16], hh[16];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float4 k4 = reinterpret_cast<const float4*>(kn)[j * 8 + kseg];
    const float4 q4 = reinterpret_cast<const float4*>(qn)[j * 8 + kseg];
    kk[4 * j] = k4.x; kk[4 * j + 1] = k4.y; kk[4 * j + 2] = k4.z; kk[4 * j + 3] = k4.w;
    qq[4 * j] = q4.x; qq[4 * j + 1] = q4.y; qq[4 * j + 2] = q4.z; qq[4 * j + 3] = q4.w;
    hh[4 * j] = hs[j].x * decay; hh[4 * j + 1] = hs[j].y * decay;
    hh[4 * j + 2] = hs[j].z * decay; hh[4 * j + 3] = hs[j].w * decay;
  }
  float hk = 0.f;
#pragma unroll
  for (int e = 0; e < 16; ++e) hk += hh[e] * kk[e];
  hk = sum8(hk);
  const float vv = (vsh[r] - hk) * beta;
  float o = 0.f;
#pragma unroll
  for (int e = 0; e < 16; ++e) { hh[e] += vv * kk[e]; o += hh[e] * qq[e]; }
  o = sum8(o);
#pragma unroll
  for (int j = 0; j < 4; ++j)
    if (!(DBG & 2)) st_state4<(DBG & 4) != 0>(hrow + (j * 8 + kseg) * 4, hh[4 * j], hh[4 * j + 1],
                                              hh[4 * j + 2], hh[4 * j + 3]);
  if (kseg == 0) osh[r] = rbf(o);  // reference stores o as bf16 before the norm
  if (tid == 0) {
    flag = (old == 2);
    if (flag) cnt_qk[t * GDN_H + h] = 0;
  }
  __syncthreads();
  // gated RMSNorm over the head: y = o * rstd * w * silu(z)
  float ov = 0.f;
  if (tid < GDN_D) {
    ov = osh[tid];
    const float s2 = wave_sum(ov * ov);
    if ((tid & 63) == 0) red[4 + (tid >> 6)] = s2;
  }
  if (flag && tid < 256) {  // last of the 3 blocks sharing this k-head: new q/k conv state
    cs[ch * cs_dim] = c1; cs[ch * cs_dim + cs_tok] = c2; cs[ch * cs_dim + 2 * cs_tok] = xin;
  }
  __syncthreads();
  if (tid < GDN_D) {
    const float rstd = rsqrtf((red[4] + red[5]) / (float)GDN_D + eps);
    float yv = ov * rstd * nw;
    yv *= zz * (1.0f / (1.0f + expf(-zz)));
    orow[tid] = (bf16_t)f2bf(yv);
  }
}

void gdn_decode(torch::Tensor qkv, torch::Tensor z, torch::Tensor b, torch::Tensor a,
                torch::Tensor conv_state, int64_t cs_seq, int64_t cs_dim, int64_t cs_tok,
                torch::Tensor conv_w, torch::Tensor A_log, torch::Tensor dt_bias,
                torch::Tensor norm_w, torch::Tensor ssm, torch::Tensor idx, torch::Tensor out,
                torch::Tensor cnt_qk, double scale, double eps, int64_t dbg) {
  const int64_t M = out.size(0);
  const int n_act = (int)idx.size(0);
  TORCH_CHECK(out.is_contiguous() && out.size(1) == GDN_HV * GDN_D, "out");
  TORCH_CHECK(qkv.stride(1) == 1 && z.stride(1) == 1 && b.stride(1) == 1 && a.stride(1) == 1, "rows");
  TORCH_CHECK(qkv.size(0) >= n_act && n_act <= M, "tokens");
  TORCH_CHECK(ssm.scalar_type() == at::kFloat && ssm.dim() == 4 && ssm.size(1) == GDN_HV &&
                  ssm.size(2) == GDN_D && ssm.size(3) == GDN_D && ssm.stride(3) == 1 &&
                  ssm.stride(2) == GDN_D && ssm.stride(1) == GDN_D * GDN_D,
              "ssm state layout");
  TORCH_CHECK(conv_state.scalar_type() == at::kBFloat16 && conv_w.scalar_type() == at::kBFloat16, "conv dtype");
  TORCH_CHECK(conv_w.size(0) == GDN_QKV && conv_w.size(-1) == 4, "conv_w");
  TORCH_CHECK(cnt_qk.numel() >= M * GDN_H && cnt_qk.scalar_type() == at::kInt, "counters");
  TORCH_CHECK(idx.scalar_type() == at::kInt || idx.scalar_type() == at::kLong, "idx");
  TORCH_CHECK(idx.is_contiguous(), "idx contiguous");
  if (M == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto P = [](const torch::Tensor& t) { return reinterpret_cast<bf16_t*>(t.data_ptr()); };
  decltype(&gdn_decode_k<0>) kerns[8] = {gdn_decode_k<0>, gdn_decode_k<1>, gdn_decode_k<2>, gdn_decode_k<3>,
                                         gdn_decode_k<4>, gdn_decode_k<5>, gdn_decode_k<6>, gdn_decode_k<7>};
  auto kern = kerns[dbg & 7];
  kern<<<dim3(GDN_HV, M), dim3(1024), 0, stream>>>(
      P(qkv), qkv.stride(0), P(z), z.stride(0), P(b), b.stride(0), P(a), a.stride(0),
      P(conv_state), cs_seq, cs_dim, cs_tok, P(conv_w), conv_w.stride(0), conv_w.stride(-1),
      A_log.data_ptr(), A_log.scalar_type() == at::kFloat, dt_bias.data_ptr(),
      dt_bias.scalar_type() == at::kFloat, norm_w.data_ptr(), norm_w.scalar_type() == at::kFloat,
      ssm.data_ptr<float>(), ssm.stride(0), idx.data_ptr(), idx.scalar_type() == at::kLong, n_act,
      P(out), cnt_qk.data_ptr<int>(), (float)scale, (float)eps);
}

// ------------------------------------------------------------------------------------
// Speculative verify (MTP) GDN: R requests x T tokens in one launch. Per token the math is the
// gdn_decode_k math (same expressions, same lane -> channel/row mapping for every wave reduction,
// same rounding points), so every token's output and every written state are bitwise equal to
// running gdn_decode token by token.
// vLLM spec contract (causal_conv1d_update IS_SPEC_DECODING + fused_sigmoid_gating_delta_rule_update):
//   conv state: one slot sidx[i,0]; window = cols o..o+2 with o = num_accepted[i]-1; afterwards the
//               slot holds [old[o+1], old[o+2], x_0 .. x_{T-1}] in cols 0..T+1.
//   ssm state : initial state from slot sidx[i,o]; the state after token t is written to sidx[i,t]
//               (slots <= 0 skipped). Output rows of padded / null requests are zero.
// Grid (48 v-heads, nb+1), 1024 threads; block (hv, nb) zero-fills output rows [qsl[nb], n_tok).
// TT = max tokens per request (template, = sidx.size(1)). All inputs of all tokens are loaded
// before the first barrier; the per-token barriers order LDS only, so the state stores of token t
// stream out while token t+1 is computed (the T state write-outs, 3.1 MB each per layer, are the
// floor of this kernel; phased variants that burst them were slower).
template <int TT>
__global__ __launch_bounds__(1024) void gdn_spec_k(
    const bf16_t* __restrict__ qkv, int64_t ld_qkv, const bf16_t* __restrict__ z, int64_t ld_z,
    const bf16_t* __restrict__ bb, int64_t ld_b, const bf16_t* __restrict__ aa, int64_t ld_a,
    bf16_t* __restrict__ cst, int64_t cs_seq, int64_t cs_dim, int64_t cs_tok, int cs_width,
    const bf16_t* __restrict__ cw, int64_t cw_dim, int64_t cw_k, const void* A_log, int A_f32,
    const void* dt_bias, int dt_f32, const void* norm_w, int nw_f32, float* __restrict__ ssm,
    int64_t ssm_seq, const int* __restrict__ sidx, int64_t ld_sidx, const int* __restrict__ nacc,
    const int* __restrict__ qsl, int nb, int n_tok, bf16_t* __restrict__ out,
    int* __restrict__ cnt_qk, float scale, float eps, int dbg) {
  __shared__ float qn[GDN_D], kn[GDN_D], vsh[GDN_D], osh[GDN_D], red[8], gate_sh[2];
  __shared__ int flag;
  const int hv = blockIdx.x, i = blockIdx.y, h = hv / 3, tid = threadIdx.x;
  const int r = tid >> 3, kseg = tid & 7;
  if (i == nb) {  // tail rows not owned by any request
    const int s0 = qsl[nb];
    for (int row = s0 + (tid >> 7); row < n_tok; row += 8)
      out[(int64_t)row * (GDN_HV * GDN_D) + hv * GDN_D + (tid & 127)] = 0;
    return;
  }
  // metadata: one level of independent loads (query range, accepted count, all TT slot ids)
  const int* srow = sidx + (int64_t)i * ld_sidx;
  int sv[TT];
#pragma unroll
  for (int t = 0; t < TT; ++t) sv[t] = srow[t];
  const int bos = qsl[i];
  int T = qsl[i + 1] - bos;
  const int o = nacc[i] - 1;
  if (T <= 0) return;
  if (T > TT) T = TT;
  if (bos + T > n_tok) T = n_tok - bos;
  const int cslot = sv[0];
  int rslot = 0;
#pragma unroll
  for (int t = 0; t < TT; ++t)
    if (t == o) rslot = sv[t];
  if (o + 2 >= cs_width) rslot = 0;
  if (cslot <= 0 || rslot <= 0 || T + 2 > cs_width) {  // null / padded request: zero output
    for (int t = tid >> 7; t < T; t += 8)
      out[(int64_t)(bos + t) * (GDN_HV * GDN_D) + hv * GDN_D + (tid & 127)] = 0;
    return;
  }
  const int ch = (tid < 128) ? (h * GDN_D + tid)
               : (tid < 256) ? (GDN_H * GDN_D + h * GDN_D + tid - 128)
                             : (GDN_VOFF + hv * GDN_D + (tid & 127));
  const bool conv_thr = tid < 384;
  const bool gate_wave = (tid >> 6) == 6;
  // ---- every input of every token, loaded up front
  bf16_t w0 = 0, w1 = 0, w2 = 0, w3 = 0, c0 = 0, c1 = 0, c2 = 0;
  bf16_t xs[TT];
  float zs[TT], as_[TT], bs_[TT];
  int ws[TT];
  bf16_t* cs = cst + (int64_t)cslot * cs_seq;
#pragma unroll
  for (int t = 0; t < TT; ++t) {
    xs[t] = 0; zs[t] = 0.f; as_[t] = 0.f; bs_[t] = 0.f;
    ws[t] = (t < T) ? sv[t] : 0;
  }
  if (conv_thr) {
    const bf16_t* cwr = cw + (int64_t)ch * cw_dim;
    w0 = cwr[0]; w1 = cwr[cw_k]; w2 = cwr[2 * cw_k]; w3 = cwr[3 * cw_k];
    c0 = cs[ch * cs_dim + o * cs_tok]; c1 = cs[ch * cs_dim + (o + 1) * cs_tok];
    c2 = cs[ch * cs_dim + (o + 2) * cs_tok];
#pragma unroll
    for (int t = 0; t < TT; ++t)
      if (t < T) xs[t] = qkv[(int64_t)(bos + t) * ld_qkv + ch];
  }
  const bf16_t s1 = c1, s2 = c2;
  float nw = 0.f, dtb = 0.f, alog = 0.f;
  if (tid < GDN_D) {
    nw = ldf(norm_w, tid, nw_f32);
#pragma unroll
    for (int t = 0; t < TT; ++t)
      if (t < T) zs[t] = bf2f(z[(int64_t)(bos + t) * ld_z + hv * GDN_D + tid]);
  }
  if (gate_wave) {
    dtb = ldf(dt_bias, hv, dt_f32); alog = ldf(A_log, hv, A_f32);
#pragma unroll
    for (int t = 0; t < TT; ++t)
      if (t < T) {
        as_[t] = bf2f(aa[(int64_t)(bos + t) * ld_a + hv]);
        bs_[t] = bf2f(bb[(int64_t)(bos + t) * ld_b + hv]);
      }
  }
  const int64_t hoff = ((int64_t)hv * GDN_D + r) * GDN_D;
  float hs[16];
  {
    const float* hrow = ssm + (int64_t)rslot * ssm_seq + hoff;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float4 v4 = (dbg & 1) ? make_float4(0.f, 0.f, 0.f, 0.f)
                                  : *reinterpret_cast<const float4*>(hrow + (j * 8 + kseg) * 4);
      hs[4 * j] = v4.x; hs[4 * j + 1] = v4.y; hs[4 * j + 2] = v4.z; hs[4 * j + 3] = v4.w;
    }
  }
  __syncthreads();  // all conv-state reads of this block are done
  int old = 0;
  if (tid == 0) old = atomicAdd(&cnt_qk[i * GDN_H + h], 1);  // consumed after the token loop
#pragma unroll
  for (int t = 0; t < TT; ++t) {
    if (t >= T) break;
    const int64_t row = bos + t;
    float acc = 0.f;
    if (conv_thr) {
      const bf16_t xin = xs[t];
      acc += rbf(bf2f(c0) * bf2f(w0));
      acc += rbf(bf2f(c1) * bf2f(w1));
      acc += rbf(bf2f(c2) * bf2f(w2));
      acc += rbf(bf2f(xin) * bf2f(w3));
      acc = rbf(acc / (1.0f + expf(-acc)));
      if (tid >= 256) vsh[tid - 256] = acc;
      c0 = c1; c1 = c2; c2 = xin;
    }
    if (gate_wave && (tid & 63) == 0) {
      const float xg = as_[t] + dtb;
      const float sp = (xg <= 20.0f) ? logf(1.0f + expf(xg)) : xg;
      const float g = -expf(alog) * sp;
      gate_sh[0] = expf(g);
      gate_sh[1] = 1.0f / (1.0f + expf(-bs_[t]));
    }
    const float ss = wave_sum(acc * acc);
    if ((tid & 63) == 0 && tid < 256) red[tid >> 6] = ss;
    lds_barrier();
    if (tid < 256) {
      const float tot = (tid < 128) ? (red[0] + red[1]) : (red[2] + red[3]);
      const float vn = acc / sqrtf(tot + 1e-6f);
      if (tid < 128) qn[tid] = vn * scale; else kn[tid - 128] = vn;
    }
    lds_barrier();
    const float decay = gate_sh[0], beta = gate_sh[1];
    float kk[16], qq[16], hh[16];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float4 k4 = reinterpret_cast<const float4*>(kn)[j * 8 + kseg];
      const float4 q4 = reinterpret_cast<const float4*>(qn)[j * 8 + kseg];
      kk[4 * j] = k4.x; kk[4 * j + 1] = k4.y; kk[4 * j + 2] = k4.z; kk[4 * j + 3] = k4.w;
      qq[4 * j] = q4.x; qq[4 * j + 1] = q4.y; qq[4 * j + 2] = q4.z; qq[4 * j + 3] = q4.w;
      hh[4 * j] = hs[4 * j] * decay; hh[4 * j + 1] = hs[4 * j + 1] * decay;
      hh[4 * j + 2] = hs[4 * j + 2] * decay; hh[4 * j + 3] = hs[4 * j + 3] * decay;
    }
    float hk = 0.f;
#pragma unroll
    for (int e = 0; e < 16; ++e) hk += hh[e] * kk[e];
    hk = sum8(hk);
    const float vv = (vsh[r] - hk) * beta;
    float ov_ = 0.f;
#pragma unroll
    for (int e = 0; e < 16; ++e) { hh[e] += vv * kk[e]; ov_ += hh[e] * qq[e]; }
    ov_ = sum8(ov_);
    if (!(dbg & 2) && ws[t] > 0) {
      float* wrow = ssm + (int64_t)ws[t] * ssm_seq + hoff;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (dbg & 4)
          st_state4<true>(wrow + (j * 8 + kseg) * 4, hh[4 * j], hh[4 * j + 1], hh[4 * j + 2], hh[4 * j + 3]);
        else
          st_state4<false>(wrow + (j * 8 + kseg) * 4, hh[4 * j], hh[4 * j + 1], hh[4 * j + 2], hh[4 * j + 3]);
      }
    }
#pragma unroll
    for (int e = 0; e < 16; ++e) hs[e] = hh[e];
    if (kseg == 0) osh[r] = rbf(ov_);
    lds_barrier();
    float ov = 0.f;
    if (tid < GDN_D) {
      ov = osh[tid];
      const float s2_ = wave_sum(ov * ov);
      if ((tid & 63) == 0) red[4 + (tid >> 6)] = s2_;
    }
    lds_barrier();
    if (tid < GDN_D) {
      const float rstd = rsqrtf((red[4] + red[5]) / (float)GDN_D + eps);
      float yv = ov * rstd * nw;
      const float zz = zs[t];
      yv *= zz * (1.0f / (1.0f + expf(-zz)));
      out[row * (GDN_HV * GDN_D) + hv * GDN_D + tid] = (bf16_t)f2bf(yv);
    }
  }
  // conv state: [old[o+1], old[o+2], x_0 .. x_{T-1}] -> cols 0..T+1. v channels: this block is the
  // only owner; q/k channels: the last of the 3 blocks sharing the k-head (all 3 read at the start).
  if (tid == 0) {
    flag = (old == 2);
    if (flag) cnt_qk[i * GDN_H + h] = 0;
  }
  __syncthreads();
  if (conv_thr && (tid >= 256 || flag)) {
    bf16_t* cc = cs + ch * cs_dim;
    cc[0] = s1;
    cc[cs_tok] = s2;
#pragma unroll
    for (int t = 0; t < TT; ++t)
      if (t < T) cc[(2 + t) * cs_tok] = xs[t];
  }
}

void gdn_spec(torch::Tensor qkv, torch::Tensor z, torch::Tensor b, torch::Tensor a,
              torch::Tensor conv_state, int64_t cs_seq, int64_t cs_dim, int64_t cs_tok,
              int64_t cs_width, torch::Tensor conv_w, torch::Tensor A_log, torch::Tensor dt_bias,
              torch::Tensor norm_w, torch::Tensor ssm, torch::Tensor sidx, torch::Tensor nacc,
              torch::Tensor qsl, torch::Tensor out, torch::Tensor cnt_qk, double scale, double eps,
              int64_t dbg) {
  const int64_t n_tok = out.size(0);
  const int nb = (int)sidx.size(0);
  const int TT = (int)sidx.size(1);
  TORCH_CHECK(TT >= 1 && TT <= 8, "gdn_spec: 1..8 tokens per request");
  TORCH_CHECK(out.is_contiguous() && out.size(1) == GDN_HV * GDN_D, "out");
  TORCH_CHECK(qkv.stride(1) == 1 && z.stride(1) == 1 && b.stride(1) == 1 && a.stride(1) == 1, "rows");
  TORCH_CHECK(qkv.size(0) >= n_tok, "tokens");
  TORCH_CHECK(ssm.scalar_type() == at::kFloat && ssm.dim() == 4 && ssm.size(1) == GDN_HV &&
                  ssm.size(2) == GDN_D && ssm.size(3) == GDN_D && ssm.stride(3) == 1 &&
                  ssm.stride(2) == GDN_D && ssm.stride(1) == GDN_D * GDN_D,
              "ssm state layout");
  TORCH_CHECK(conv_state.scalar_type() == at::kBFloat16 && conv_w.scalar_type() == at::kBFloat16, "conv dtype");
  TORCH_CHECK(conv_w.size(0) == GDN_QKV && conv_w.size(-1) == 4, "conv_w");
  TORCH_CHECK(sidx.scalar_type() == at::kInt && sidx.dim() == 2 && sidx.stride(1) == 1, "sidx");
  TORCH_CHECK(nacc.scalar_type() == at::kInt && nacc.numel() >= nb && nacc.stride(0) == 1, "nacc");
  TORCH_CHECK(qsl.scalar_type() == at::kInt && qsl.numel() >= nb + 1 && qsl.stride(0) == 1, "qsl");
  TORCH_CHECK(cnt_qk.numel() >= (int64_t)nb * GDN_H && cnt_qk.scalar_type() == at::kInt, "counters");
  if (n_tok == 0 && nb == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto P = [](const torch::Tensor& t) { return reinterpret_cast<bf16_t*>(t.data_ptr()); };
#define GDN_SPEC_L(N)                                                                             \
  gdn_spec_k<N><<<dim3(GDN_HV, nb + 1), dim3(1024), 0, stream>>>(                                \
      P(qkv), qkv.stride(0), P(z), z.stride(0), P(b), b.stride(0), P(a), a.stride(0),            \
      P(conv_state), cs_seq, cs_dim, cs_tok, (int)cs_width, P(conv_w), conv_w.stride(0),          \
      conv_w.stride(-1), A_log.data_ptr(), A_log.scalar_type() == at::kFloat, dt_bias.data_ptr(), \
      dt_bias.scalar_type() == at::kFloat, norm_w.data_ptr(), norm_w.scalar_type() == at::kFloat, \
      ssm.data_ptr<float>(), ssm.stride(0), sidx.data_ptr<int>(), sidx.stride(0),                 \
      nacc.data_ptr<int>(), qsl.data_ptr<int>(), nb, (int)n_tok, P(out),                          \
      cnt_qk.data_ptr<int>(), (float)scale, (float)eps, (int)dbg)
  switch (TT) {
    case 1: GDN_SPEC_L(1); break;
    case 2: GDN_SPEC_L(2); break;
    case 3: GDN_SPEC_L(3); break;
    case 4: GDN_SPEC_L(4); break;
    case 5: GDN_SPEC_L(5); break;
    case 6: GDN_SPEC_L(6); break;
    case 7: GDN_SPEC_L(7); break;
    default: GDN_SPEC_L(8); break;
  }
#undef GDN_SPEC_L
}

// ------------------------------------------------------------------------------------
// Pseudo-sequence metadata for per-token-exact multi-query attention: every query token s
// becomes its own 1-query sequence over the keys [0, pos(s)] of its request, so the Triton
// unified attention takes the same 3D split-softmax decode path as a 1-token decode step.
//   cu[s] = s, su[s] = seq_len[r] - qlen[r] + (s - qsl[r]) + 1, bt_out[s] = bt[r]
// Tokens not covered by any request (graph padding) get su = 1 and request 0's block row.
__global__ __launch_bounds__(256) void attn_pseudo_meta_k(
    const int* __restrict__ qsl, const int* __restrict__ seq_lens, const int* __restrict__ bt,
    int64_t ld_bt, int nb_cols, int R, int n_tok, int* __restrict__ cu, int* __restrict__ su,
    int* __restrict__ bto) {
  __shared__ int rq;
  const int s = blockIdx.x;
  if (threadIdx.x == 0) {
    int rr = -1;
    for (int k = 0; k < R; ++k)
      if (qsl[k] <= s && s < qsl[k + 1]) { rr = k; break; }
    rq = rr;
    cu[s] = s;
    if (s == n_tok - 1) cu[n_tok] = n_tok;
    int v = 1;
    if (rr >= 0) {
      v = seq_lens[rr] - (qsl[rr + 1] - qsl[rr]) + (s - qsl[rr]) + 1;
      if (v < 1) v = 1;
    }
    su[s] = v;
  }
  __syncthreads();
  const int src = rq < 0 ? 0 : rq;
  for (int c = threadIdx.x; c < nb_cols; c += 256)
    bto[(int64_t)s * nb_cols + c] = bt[(int64_t)src * ld_bt + c];
}

void attn_pseudo_meta(torch::Tensor qsl, torch::Tensor seq_lens, torch::Tensor bt, int64_t R,
                      torch::Tensor cu, torch::Tensor su, torch::Tensor bto) {
  const int64_t n_tok = su.size(0);
  TORCH_CHECK(qsl.scalar_type() == at::kInt && seq_lens.scalar_type() == at::kInt &&
                  bt.scalar_type() == at::kInt && cu.scalar_type() == at::kInt &&
                  su.scalar_type() == at::kInt && bto.scalar_type() == at::kInt, "int32");
  TORCH_CHECK(qsl.stride(0) == 1 && seq_lens.stride(0) == 1 && bt.stride(1) == 1, "strides");
  TORCH_CHECK(bto.is_contiguous() && bto.size(0) == n_tok && bto.size(1) == bt.size(1), "bto");
  TORCH_CHECK(cu.numel() >= n_tok + 1 && qsl.numel() >= R + 1 && seq_lens.numel() >= R, "sizes");
  if (n_tok == 0) return;
  attn_pseudo_meta_k<<<dim3(n_tok), dim3(256), 0, at::cuda::getCurrentCUDAStream()>>>(
      qsl.data_ptr<int>(), seq_lens.data_ptr<int>(), bt.data_ptr<int>(), bt.stride(0),
      (int)bt.size(1), (int)R, (int)n_tok, cu.data_ptr<int>(), su.data_ptr<int>(),
      bto.data_ptr<int>());
}

__global__ void noop_k(int* p) {
  if (p && threadIdx.x == 1024) p[0] = 0;
}
void noop(torch::Tensor x, int64_t nblk, int64_t nthr) {
  noop_k<<<dim3(nblk), dim3(nthr), 0, at::cuda::getCurrentCUDAStream()>>>(nullptr);
}

TORCH_LIBRARY(vopt_rest, m) {
  m.def("gemma_rms_norm(Tensor x, Tensor w, float eps) -> Tensor");
  m.def("gemma_add_rms_norm(Tensor x, Tensor r, Tensor w, float eps) -> Tensor[]");
  m.def("silu_mul(Tensor x) -> Tensor");
  m.def("sigmoid_mul(Tensor a, Tensor g) -> Tensor");
  m.def(
      "gdn_decode(Tensor qkv, Tensor z, Tensor b, Tensor a, Tensor(a!) conv_state, int cs_seq, "
      "int cs_dim, int cs_tok, Tensor conv_w, Tensor A_log, Tensor dt_bias, Tensor norm_w, "
      "Tensor(b!) ssm, Tensor idx, Tensor(c!) out, Tensor(d!) cnt_qk, float scale, "
      "float eps, int dbg=0) -> ()");
  m.def(
      "gdn_spec(Tensor qkv, Tensor z, Tensor b, Tensor a, Tensor(a!) conv_state, int cs_seq, "
      "int cs_dim, int cs_tok, int cs_width, Tensor conv_w, Tensor A_log, Tensor dt_bias, "
      "Tensor norm_w, Tensor(b!) ssm, Tensor sidx, Tensor nacc, Tensor qsl, Tensor(c!) out, "
      "Tensor(d!) cnt_qk, float scale, float eps, int dbg=0) -> ()");
  m.def(
      "attn_pseudo_meta(Tensor qsl, Tensor seq_lens, Tensor bt, int R, Tensor(a!) cu, "
      "Tensor(b!) su, Tensor(c!) bto) -> ()");
  m.def("noop(Tensor x, int nblk, int nthr) -> ()");
}
TORCH_LIBRARY_IMPL(vopt_rest, CUDA, m) {
  m.impl("gemma_rms_norm", &gemma_rms_norm);
  m.impl("gemma_add_rms_norm", &gemma_add_rms_norm);
  m.impl("silu_mul", &silu_mul);
  m.impl("sigmoid_mul", &sigmoid_mul);
  m.impl("gdn_decode", &gdn_decode);
  m.impl("gdn_spec", &gdn_spec);
  m.impl("attn_pseudo_meta", &attn_pseudo_meta);
  m.impl("noop", &noop);
}
