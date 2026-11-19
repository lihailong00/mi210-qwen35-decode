// Host entry + torch op registration for the weight-only FP8 GEMV (kernels: fp8_gemv_kernel.h).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <stdint.h>
// VOPT_WS_NSLOT: split-K workspace slots (see below). VOPT_WS_FULL: shapes with at most this many row
// groups get one slot per row group, i.e. no slot reuse; then a call without the designated reducer
// (down, N=5120: 80 row groups) has no waits at all and keeps the static block order (no work ticket,
// whose round trip at block start costs down 3-10%). 64 vs 80 slots measured equal there. Default 96
// with the default slot count; 0 (off) when VOPT_WS_NSLOT is set explicitly, so a forced slot count
// is honoured on every shape (edge-value tests).
#ifndef VOPT_WS_NSLOT
#define VOPT_WS_NSLOT 64
#ifndef VOPT_WS_FULL
#define VOPT_WS_FULL 96
#endif
#endif
#ifndef VOPT_WS_FULL
#define VOPT_WS_FULL 0
#endif
// slot counters cnt[0 .. nslot), use counters cnt[2048 .. 2048 + nslot), ticket / error flag at
// cnt[4064], cnt[4065] (fp8_gemv_kernel.h)
static_assert(VOPT_WS_NSLOT >= 1 && VOPT_WS_NSLOT <= 1024, "VOPT_WS_NSLOT must be in 1..1024");
static_assert(VOPT_WS_FULL >= 0 && VOPT_WS_FULL <= 1024, "VOPT_WS_FULL must be in 0..1024");
void vopt_launch_m1(dim3, dim3, size_t, hipStream_t, const uint32_t*, const uint8_t*, const float*, uint16_t*, int, float*, int*, int, int, int, int, int);
void vopt_launch_m2(dim3, dim3, size_t, hipStream_t, const uint32_t*, const uint8_t*, const float*, uint16_t*, int, float*, int*, int, int, int, int, int);
void vopt_launch_m3(dim3, dim3, size_t, hipStream_t, const uint32_t*, const uint8_t*, const float*, uint16_t*, int, float*, int*, int, int, int, int, int);
void vopt_launch_m4(dim3, dim3, size_t, hipStream_t, const uint32_t*, const uint8_t*, const float*, uint16_t*, int, float*, int*, int, int, int, int, int);
void vopt_launch_m5(dim3, dim3, size_t, hipStream_t, const uint32_t*, const uint8_t*, const float*, uint16_t*, int, float*, int*, int, int, int, int, int);
void vopt_launch_m6(dim3, dim3, size_t, hipStream_t, const uint32_t*, const uint8_t*, const float*, uint16_t*, int, float*, int*, int, int, int, int, int);
void vopt_launch_m7(dim3, dim3, size_t, hipStream_t, const uint32_t*, const uint8_t*, const float*, uint16_t*, int, float*, int*, int, int, int, int, int);
void vopt_launch_m8(dim3, dim3, size_t, hipStream_t, const uint32_t*, const uint8_t*, const float*, uint16_t*, int, float*, int*, int, int, int, int, int);

// x [M,K] bf16 (stride(1)==1), w tiled uint8 [N,K] (contiguous), s fp32 [N/128,K/128],
// ws fp32 workspace, cnt int32 zeroed counters (4096 entries, layout above), cfg (R, S, B)
torch::Tensor fp8_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor ws,
                       torch::Tensor cnt, int64_t R, int64_t S, int64_t B) {
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.is_contiguous());
  TORCH_CHECK(w.dim() == 2 && w.scalar_type() == at::kByte && w.is_contiguous());
  TORCH_CHECK(s.dim() == 2 && s.scalar_type() == at::kFloat && s.is_contiguous());
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(w.size(1) == K && K % 128 == 0 && N % 128 == 0 && M >= 1 && M <= 8);
  TORCH_CHECK(s.size(0) == N / 128 && s.size(1) == K / 128);
  TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) & 3) == 0);
  TORCH_CHECK(K == 5120 || K == 6144 || K == 17408, "K not instantiated");
  TORCH_CHECK((R == 1 || R == 2) && S >= 1 && S <= 8 && B >= 1 && B <= 17 && (K / 128) % B == 0);
  const int RW = 64 * R;
  const int ngroups = N / RW;
  if (B > 1) {
    TORCH_CHECK(ws.scalar_type() == at::kFloat && ws.numel() >= (int64_t)ngroups * B * M * RW);
    TORCH_CHECK(cnt.scalar_type() == at::kInt && cnt.numel() >= ngroups);
  }
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto y = torch::empty({M, N}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(ngroups * B), block(S * 64);
  const size_t shm = S > 1 ? (size_t)S * M * RW * sizeof(float) : 0;
  TORCH_CHECK(shm <= 64 * 1024 - 64, "LDS too large for config");
  auto xp = reinterpret_cast<const uint32_t*>(x.data_ptr());
  auto wp = w.data_ptr<uint8_t>();
  auto sp = s.data_ptr<float>();
  auto yp = reinterpret_cast<uint16_t*>(y.data_ptr());
  float* wsp = B > 1 ? ws.data_ptr<float>() : nullptr;
  int* cp = B > 1 ? cnt.data_ptr<int>() : nullptr;
  using LF = void (*)(dim3, dim3, size_t, hipStream_t, const uint32_t*, const uint8_t*, const float*,
                      uint16_t*, int, float*, int*, int, int, int, int, int);
  static const LF tab[8] = {vopt_launch_m1, vopt_launch_m2, vopt_launch_m3, vopt_launch_m4,
                            vopt_launch_m5, vopt_launch_m6, vopt_launch_m7, vopt_launch_m8};
  // split-K workspace slots (row group rg uses slot rg % nslot): ~45 row groups are in flight at a
  // time; 64 slots keep waits rare and the workspace L2-resident (one slot per row group meant MBs
  // of dirty lines per call written back to HBM in the middle of the weight stream)
  const int nslot = ngroups <= VOPT_WS_FULL || ngroups < VOPT_WS_NSLOT ? ngroups : VOPT_WS_NSLOT;
  TORCH_CHECK(nslot >= 1 && nslot <= 1024, "nslot out of range");
  TORCH_CHECK(B == 1 || cnt.numel() >= 4096, "cnt must hold 4096 int32 (slot/use counters, ticket)");
  tab[M - 1](grid, block, shm, stream, xp, wp, sp, yp, N, wsp, cp, K, (int)R, (int)B, (int)S, nslot);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

TORCH_LIBRARY(vopt_fp8gemv, m) {
  m.def("fp8_gemv(Tensor x, Tensor w, Tensor s, Tensor ws, Tensor cnt, int R, int S, int B) -> Tensor");
}
TORCH_LIBRARY_IMPL(vopt_fp8gemv, CUDA, m) { m.impl("fp8_gemv", &fp8_gemv); }
