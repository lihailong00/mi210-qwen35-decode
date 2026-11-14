// For each real prefill shape: default rocBLAS kernel vs best solution, across token counts N.
// Also checks whether the best solution tuned at N=4096 transfers to other N.
#define ROCBLAS_BETA_FEATURES_API
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)

int main(int argc, char ** argv) {
    bool full = argc > 1 && atoi(argv[1]) != 0;   // full tune at every N
    hipStream_t s; CK(hipStreamCreate(&s));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    rocblas_handle h; rocblas_create_handle(&h); rocblas_set_stream(h, s);
    struct Sh { const char * name; int m, k; } shapes[] = {
        {"ffn_gate/up", 17408, 5120}, {"ffn_down", 5120, 17408}, {"ssm_qkv", 10240, 5120},
        {"ssm_z", 6144, 5120}, {"ssm_out/attn_o", 5120, 6144}, {"attn_q", 12288, 5120}, {"attn_k/v", 1024, 5120}};
    int Ns[] = {4096, 512, 1024, 2048, 3001, 8192};
    const int maxN = 8192;
    void *A, *B, *C;
    CK(hipMalloc(&A, (size_t)17408 * 5120 * 2)); CK(hipMalloc(&B, (size_t)17408 * maxN * 2)); CK(hipMalloc(&C, (size_t)17408 * maxN * 4));
    CK(hipMemset(A, 0, (size_t)17408 * 5120 * 2)); CK(hipMemset(B, 0, (size_t)17408 * maxN * 2));
    float alpha = 1, beta = 0;
    std::vector<int> best4096(7, 0);
    for (int N : Ns) {
        for (int si = 0; si < 7; ++si) {
            int m = shapes[si].m, k = shapes[si].k;
            auto run = [&](rocblas_gemm_algo algo, int sol) {
                return rocblas_gemm_ex(h, rocblas_operation_transpose, rocblas_operation_none, m, N, k, &alpha,
                    A, rocblas_datatype_f16_r, k, B, rocblas_datatype_f16_r, k, &beta,
                    C, rocblas_datatype_f32_r, m, C, rocblas_datatype_f32_r, m, rocblas_datatype_f32_r, algo, sol, 0);
            };
            auto timeit = [&](rocblas_gemm_algo algo, int sol, int reps) {
                if (run(algo, sol) != rocblas_status_success) return 1e9;
                CK(hipStreamSynchronize(s));
                CK(hipEventRecord(e0, s)); for (int r = 0; r < reps; ++r) run(algo, sol); CK(hipEventRecord(e1, s)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1)); return (double) ms / reps;
            };
            double fl = 2.0 * m * N * k;
            double tdef = timeit(rocblas_gemm_algo_standard, 0, 8);
            double tx = best4096[si] ? timeit(rocblas_gemm_algo_solution_index, best4096[si], 8) : 0;
            double tbest = 0; int bsol = 0;
            if (N == 4096 || full) {
                rocblas_int nsol = 0;
                rocblas_gemm_ex_get_solutions(h, rocblas_operation_transpose, rocblas_operation_none, m, N, k, &alpha,
                    A, rocblas_datatype_f16_r, k, B, rocblas_datatype_f16_r, k, &beta, C, rocblas_datatype_f32_r, m, C, rocblas_datatype_f32_r, m,
                    rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, 0, nullptr, &nsol);
                std::vector<rocblas_int> sols(nsol);
                rocblas_gemm_ex_get_solutions(h, rocblas_operation_transpose, rocblas_operation_none, m, N, k, &alpha,
                    A, rocblas_datatype_f16_r, k, B, rocblas_datatype_f16_r, k, &beta, C, rocblas_datatype_f32_r, m, C, rocblas_datatype_f32_r, m,
                    rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, 0, sols.data(), &nsol);
                tbest = tdef;
                for (int i = 0; i < nsol; ++i) {
                    double t = timeit(rocblas_gemm_algo_solution_index, sols[i], 2);
                    if (t < tbest * 1.05) { t = timeit(rocblas_gemm_algo_solution_index, sols[i], 6); if (t < tbest) { tbest = t; bsol = sols[i]; } }
                }
                if (N == 4096) best4096[si] = bsol;
            }
            printf("N=%5d %-15s default %8.3f ms %6.1f TF | best@4096-sol %8.3f ms %6.1f TF | best %8.3f ms %6.1f TF sol=%d\n",
                   N, shapes[si].name, tdef, fl / tdef / 1e9, tx, tx ? fl / tx / 1e9 : 0, tbest, tbest ? fl / tbest / 1e9 : 0, bsol);
        }
    }
    return 0;
}
