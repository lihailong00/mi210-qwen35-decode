// MFMA f16 peak with data-dependent (non-constant) operands, to expose the power-cap effect.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef float float16v __attribute__((ext_vector_type(16)));

__global__ void mfma_f16(float * out, int iters, int rnd) {
    unsigned h = (blockIdx.x * 256u + threadIdx.x) * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    half4 a[4], b[4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) {
        h = h * 1664525u + 1013904223u;
        float fa = rnd ? ((int) (h >> 8 & 0xffff) - 32768) / 32768.0f : 0.0f;
        h = h * 1664525u + 1013904223u;
        float fb = rnd ? ((int) (h >> 8 & 0xffff) - 32768) / 32768.0f : 0.0f;
        a[i][j] = (_Float16) fa; b[i][j] = (_Float16) fb;
    }
    const half4 a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    float16v c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
    for (int i = 0; i < iters; ++i) {
        c0 = __builtin_amdgcn_mfma_f32_32x32x8f16(a0, b0, c0, 0, 0, 0);
        c1 = __builtin_amdgcn_mfma_f32_32x32x8f16(a1, b1, c1, 0, 0, 0);
        c2 = __builtin_amdgcn_mfma_f32_32x32x8f16(a2, b2, c2, 0, 0, 0);
        c3 = __builtin_amdgcn_mfma_f32_32x32x8f16(a3, b3, c3, 0, 0, 0);
    }
    float s = 0; for (int k = 0; k < 16; ++k) s += c0[k] + c1[k] + c2[k] + c3[k];
    if (s == 12345.678f) out[0] = s;
}

int main() {
    float * dout; CK(hipMalloc(&dout, 64));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    const int blocks = 104 * 8, th = 256, iters = 1000000;   // ~1.7 s per launch so the clock settles
    for (int rnd : {0, 1, 0, 1}) {
        hipLaunchKernelGGL(mfma_f16, dim3(blocks), dim3(th), 0, 0, dout, 20000, rnd); CK(hipDeviceSynchronize());
        CK(hipEventRecord(e0));
        for (int r = 0; r < 3; ++r) hipLaunchKernelGGL(mfma_f16, dim3(blocks), dim3(th), 0, 0, dout, iters, rnd);
        CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
        float ms; CK(hipEventElapsedTime(&ms, e0, e1));
        double ops = 3.0 * blocks * (th / 64.0) * iters * 4.0 * (32 * 32 * 8 * 2);
        printf("MFMA f16 %s operands: %.1f TFLOPS (%.2f s)\n", rnd ? "random" : "zero  ", ops / (ms * 1e-3) / 1e12, ms * 1e-3);
    }
    return 0;
}
