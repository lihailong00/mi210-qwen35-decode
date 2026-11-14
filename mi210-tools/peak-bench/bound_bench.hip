// (1) decode floor: one read kernel per real weight tensor, captured in a HIP graph
// (2) prefill: rocBLAS f16->f32 GEMM on the real prefill shapes, default kernel vs best of all solutions
#define ROCBLAS_BETA_FEATURES_API
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

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

static float elapsed(hipEvent_t a, hipEvent_t b) { float ms; CK(hipEventElapsedTime(&ms, a, b)); return ms; }

int main(int argc, char ** argv) {
    int * dout; CK(hipMalloc(&dout, 64));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    hipStream_t s; CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    // ---------- (1) per-tensor read floor ----------
    {
        std::vector<size_t> sz; size_t tot = 0;
        FILE * f = fopen(argc > 1 ? argv[1] : "tensors.txt", "r");
        char name[256], type[64]; size_t n;
        while (fscanf(f, "%255s %zu %63s", name, &n, type) == 3) { sz.push_back(n); tot += n; }
        fclose(f);
        std::vector<char *> bufs(sz.size());
        for (size_t i = 0; i < sz.size(); ++i) { CK(hipMalloc(&bufs[i], (sz[i] + 15) & ~15ull)); CK(hipMemset(bufs[i], 1, sz[i])); }
        printf("tensors=%zu  bytes=%.3f GB\n", sz.size(), tot / 1e9);
        struct Cfg { int maxb, th, u; } cfgs[] = {{104 * 4, 256, 4}, {104 * 8, 256, 4}, {104 * 8, 512, 8}, {104 * 16, 256, 8}, {104 * 32, 256, 4}};
        for (auto c : cfgs) {
            hipGraph_t g; hipGraphExec_t ge;
            CK(hipStreamBeginCapture(s, hipStreamCaptureModeRelaxed));
            for (size_t i = 0; i < sz.size(); ++i) {
                size_t n16 = sz[i] / 16;
                long need = (long)((n16 + (size_t)c.th * c.u - 1) / ((size_t)c.th * c.u));
                int blocks = (int)std::max(1L, std::min((long)c.maxb, need));
                if (c.u == 4) hipLaunchKernelGGL((read_kernel<4>), dim3(blocks), dim3(c.th), 0, s, (const int4 *) bufs[i], n16, dout);
                else          hipLaunchKernelGGL((read_kernel<8>), dim3(blocks), dim3(c.th), 0, s, (const int4 *) bufs[i], n16, dout);
            }
            CK(hipStreamEndCapture(s, &g)); CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
            for (int w = 0; w < 3; ++w) CK(hipGraphLaunch(ge, s));
            CK(hipStreamSynchronize(s));
            const int R = 30;
            CK(hipEventRecord(e0, s)); for (int r = 0; r < R; ++r) CK(hipGraphLaunch(ge, s)); CK(hipEventRecord(e1, s)); CK(hipEventSynchronize(e1));
            double ms = elapsed(e0, e1) / R;
            printf("per-tensor graph (maxblocks=%4d th=%d U=%d): %.3f ms/token -> %.1f tok/s, %.1f GB/s\n", c.maxb, c.th, c.u, ms, 1e3 / ms, tot / (ms * 1e-3) / 1e9);
            CK(hipGraphExecDestroy(ge)); CK(hipGraphDestroy(g));
        }
        for (auto b : bufs) CK(hipFree(b));
    }
    if (argc > 2) return 0;
    // ---------- (2) prefill GEMM shapes ----------
    {
        rocblas_handle h; rocblas_create_handle(&h); rocblas_set_stream(h, s);
        struct Sh { const char * name; int m, k, per_ubatch; } shapes[] = {
            {"ffn_gate/up 17408x5120", 17408, 5120, 128}, {"ffn_down 5120x17408", 5120, 17408, 64},
            {"ssm qkv 10240x5120", 10240, 5120, 48},      {"ssm z 6144x5120", 6144, 5120, 48},
            {"ssm_out/attn_o 5120x6144", 5120, 6144, 64}, {"attn_q 12288x5120", 12288, 5120, 16},
            {"attn_k/v 1024x5120", 1024, 5120, 32}};
        const int N = 4096;
        double tot_def = 0, tot_best = 0, tot_fl = 0;
        for (auto sh : shapes) {
            int m = sh.m, k = sh.k;
            void *A, *B, *C;
            CK(hipMalloc(&A, (size_t)m * k * 2)); CK(hipMalloc(&B, (size_t)k * N * 2)); CK(hipMalloc(&C, (size_t)m * N * 4));
            CK(hipMemset(A, 0, (size_t)m * k * 2)); CK(hipMemset(B, 0, (size_t)k * N * 2));
            float alpha = 1, beta = 0;
            auto run = [&](rocblas_gemm_algo algo, int sol) {
                return rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none, m, N, k, &alpha,
                    A, rocblas_datatype_f16_r, k, B, rocblas_datatype_f16_r, k, &beta,
                    C, rocblas_datatype_f32_r, m, C, rocblas_datatype_f32_r, m, rocblas_datatype_f32_r, algo, sol, 0);
            };
            auto timeit = [&](rocblas_gemm_algo algo, int sol, int reps) {
                if (run(algo, sol) != rocblas_status_success) return 1e9;
                CK(hipStreamSynchronize(s));
                CK(hipEventRecord(e0, s)); for (int r = 0; r < reps; ++r) run(algo, sol); CK(hipEventRecord(e1, s)); CK(hipEventSynchronize(e1));
                return (double) elapsed(e0, e1) / reps;
            };
            double fl = 2.0 * m * N * k;
            double tdef = timeit(rocblas_gemm_algo_standard, 0, 10);
            rocblas_int nsol = 0;
            rocblas_gemm_ex_get_solutions(h, rocblas_operation_transpose, rocblas_operation_none, m, N, k, &alpha,
                A, rocblas_datatype_f16_r, k, B, rocblas_datatype_f16_r, k, &beta, C, rocblas_datatype_f32_r, m, C, rocblas_datatype_f32_r, m,
                rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, 0, nullptr, &nsol);
            std::vector<rocblas_int> sols(nsol);
            rocblas_gemm_ex_get_solutions(h, rocblas_operation_transpose, rocblas_operation_none, m, N, k, &alpha,
                A, rocblas_datatype_f16_r, k, B, rocblas_datatype_f16_r, k, &beta, C, rocblas_datatype_f32_r, m, C, rocblas_datatype_f32_r, m,
                rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, 0, sols.data(), &nsol);
            double tbest = tdef; int bsol = -1;
            for (int i = 0; i < nsol; ++i) {
                double t = timeit(rocblas_gemm_algo_solution_index, sols[i], 2);
                if (t < tbest * 1.05) { t = timeit(rocblas_gemm_algo_solution_index, sols[i], 6); if (t < tbest) { tbest = t; bsol = sols[i]; } }
            }
            printf("%-26s default %7.3f ms (%5.1f TFLOPS)  best of %4d: %7.3f ms (%5.1f TFLOPS) sol=%d\n", sh.name,
                   tdef, fl / (tdef * 1e-3) / 1e12, nsol, tbest, fl / (tbest * 1e-3) / 1e12, bsol);
            tot_def += tdef * sh.per_ubatch; tot_best += tbest * sh.per_ubatch; tot_fl += fl * sh.per_ubatch;
            hipFree(A); hipFree(B); hipFree(C);
        }
        printf("per 4096-token ubatch: GEMM default %.1f ms (%.1f TFLOPS), best %.1f ms (%.1f TFLOPS); GFLOP/token %.2f\n",
               tot_def, tot_fl / (tot_def * 1e-3) / 1e12, tot_best, tot_fl / (tot_best * 1e-3) / 1e12, tot_fl / N / 1e9);
    }
    return 0;
}
