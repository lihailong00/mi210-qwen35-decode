// Peak HBM read bandwidth, peak MFMA throughput (f16, i8) and rocBLAS GEMM on MI210.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <vector>
#include <algorithm>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)

template <int U>
__global__ void read_kernel(const int4 * __restrict__ p, size_t n16, int * out) {
    int4 acc = make_int4(0, 0, 0, 0);
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + (U - 1) * stride < n16; i += U * stride) {
        int4 v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) v[u] = p[i + u * stride];
#pragma unroll
        for (int u = 0; u < U; ++u) { acc.x ^= v[u].x; acc.y ^= v[u].y; acc.z ^= v[u].z; acc.w ^= v[u].w; }
    }
    for (; i < n16; i += stride) { int4 v = p[i]; acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w; }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x7f3a9c1d) out[0] = 1;
}

typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef float float16v __attribute__((ext_vector_type(16)));
typedef int int16v __attribute__((ext_vector_type(16)));

__global__ void mfma_f16(float * out, int iters) {
    half4 a = {(_Float16)1.0f, (_Float16)0.5f, (_Float16)0.25f, (_Float16)0.125f};
    half4 b = a;
    float16v c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
    for (int i = 0; i < iters; ++i) {
        c0 = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c0, 0, 0, 0);
        c1 = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c1, 0, 0, 0);
        c2 = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c2, 0, 0, 0);
        c3 = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c3, 0, 0, 0);
    }
    float s = 0; for (int k = 0; k < 16; ++k) s += c0[k] + c1[k] + c2[k] + c3[k];
    if (s == 12345.678f) out[0] = s;
}

__global__ void mfma_i8(int * out, int iters) {
    int a = 0x01020304, b = 0x01010101;
    int16v c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
    for (int i = 0; i < iters; ++i) {
        c0 = __builtin_amdgcn_mfma_i32_32x32x8i8(a, b, c0, 0, 0, 0);
        c1 = __builtin_amdgcn_mfma_i32_32x32x8i8(a, b, c1, 0, 0, 0);
        c2 = __builtin_amdgcn_mfma_i32_32x32x8i8(a, b, c2, 0, 0, 0);
        c3 = __builtin_amdgcn_mfma_i32_32x32x8i8(a, b, c3, 0, 0, 0);
    }
    int s = 0; for (int k = 0; k < 16; ++k) s += c0[k] + c1[k] + c2[k] + c3[k];
    if (s == 123456789) out[0] = s;
}

template <typename F> float time_ms(F f, int reps) {
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    f(); CK(hipDeviceSynchronize());
    CK(hipEventRecord(a)); for (int r = 0; r < reps; ++r) f(); CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms; CK(hipEventElapsedTime(&ms, a, b)); return ms / reps;
}

int main() {
    int * dout; CK(hipMalloc(&dout, 64));
    // ---- bandwidth ----
    const size_t maxb = 4ull << 30;
    char * buf; CK(hipMalloc(&buf, maxb)); CK(hipMemset(buf, 1, maxb));
    size_t sizes[] = {48ull << 20, 256ull << 20, 1ull << 30, 4ull << 30};
    for (size_t sz : sizes) {
        double best = 0; char cfg[64] = "";
        for (int blocks : {104 * 4, 104 * 8, 104 * 16, 104 * 32}) for (int th : {256, 512, 1024}) {
            size_t n16 = sz / 16;
            // rotate offset across copies for small sizes so data is not L2-resident (L2 = 8 MB, sizes >= 48 MB)
            auto f4 = [&] { hipLaunchKernelGGL((read_kernel<4>), dim3(blocks), dim3(th), 0, 0, (const int4 *) buf, n16, dout); };
            auto f8 = [&] { hipLaunchKernelGGL((read_kernel<8>), dim3(blocks), dim3(th), 0, 0, (const int4 *) buf, n16, dout); };
            int reps = sz < (1ull << 30) ? 50 : 10;
            double g4 = sz / (time_ms(f4, reps) * 1e-3) / 1e9, g8 = sz / (time_ms(f8, reps) * 1e-3) / 1e9;
            if (g4 > best) { best = g4; snprintf(cfg, 64, "blocks=%d threads=%d U=4", blocks, th); }
            if (g8 > best) { best = g8; snprintf(cfg, 64, "blocks=%d threads=%d U=8", blocks, th); }
        }
        printf("READ %6zu MB : best %.1f GB/s  (%s)\n", sz >> 20, best, cfg);
    }
    CK(hipFree(buf));
    // ---- MFMA peak ----
    int iters = 20000;
    for (int bpc : {4, 8, 16}) {
        int blocks = 104 * bpc, th = 256; double waves = blocks * (th / 64.0);
        float ms16 = time_ms([&] { hipLaunchKernelGGL(mfma_f16, dim3(blocks), dim3(th), 0, 0, (float *) dout, iters); }, 5);
        float ms8  = time_ms([&] { hipLaunchKernelGGL(mfma_i8,  dim3(blocks), dim3(th), 0, 0, dout, iters); }, 5);
        double ops = waves * iters * 4.0 * (32 * 32 * 8 * 2);
        printf("MFMA blocks/CU=%2d : f16 %.1f TFLOPS   i8 %.1f TOPS\n", bpc, ops / (ms16 * 1e-3) / 1e12, ops / (ms8 * 1e-3) / 1e12);
    }
    // ---- rocBLAS GEMM (achievable) ----
    rocblas_handle h; rocblas_create_handle(&h);
    for (int n : {4096, 8192}) {
        int m = n, k = n;
        void *A, *B, *C; CK(hipMalloc(&A, (size_t)m * k * 4)); CK(hipMalloc(&B, (size_t)k * n * 4)); CK(hipMalloc(&C, (size_t)m * n * 4));
        CK(hipMemset(A, 0, (size_t)m * k * 4)); CK(hipMemset(B, 0, (size_t)k * n * 4));
        float alpha = 1, beta = 0; int ia = 1, ib = 0;
        auto g16 = [&] { rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none, m, n, k, &alpha, A, rocblas_datatype_f16_r, k, B, rocblas_datatype_f16_r, k, &beta, C, rocblas_datatype_f16_r, m, C, rocblas_datatype_f16_r, m, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0); };
        auto gi8 = [&] { rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none, m, n, k, &ia, A, rocblas_datatype_i8_r, k, B, rocblas_datatype_i8_r, k, &ib, C, rocblas_datatype_i32_r, m, C, rocblas_datatype_i32_r, m, rocblas_datatype_i32_r, rocblas_gemm_algo_standard, 0, 0); };
        double fl = 2.0 * m * n * k;
        printf("rocBLAS %d^3 : f16 %.1f TFLOPS   i8 %.1f TOPS\n", n, fl / (time_ms(g16, 10) * 1e-3) / 1e12, fl / (time_ms(gi8, 10) * 1e-3) / 1e12);
        hipFree(A); hipFree(B); hipFree(C);
    }
    return 0;
}
