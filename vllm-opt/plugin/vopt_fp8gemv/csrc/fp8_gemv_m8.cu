// instantiation unit for M=8
#include "fp8_gemv_kernel.h"
void vopt_launch_m8(dim3 g, dim3 b, size_t shm, hipStream_t st, const uint32_t* x, const uint8_t* w,
                     const float* s, uint16_t* y, int N, float* ws, int* cnt, int K, int R, int B, int S, int nslot) {
  launch_fp8_gemv<8>(g, b, shm, st, x, w, s, y, N, ws, cnt, K, R, B, S, nslot);
}
