// GEMV micro-benchmark for MI210: real GGUF weights, ggml q8_1 activations, llama.cpp vec_dot reference.
#define vec_dot_iq3_s_q8_1 vec_dot_iq3_s_q8_1_original
#define vec_dot_iq3_xxs_q8_1 vec_dot_iq3_xxs_q8_1_original
#include "common.cuh"
#include "vecdotq.cuh"
#undef vec_dot_iq3_s_q8_1
#undef vec_dot_iq3_xxs_q8_1
#include "vecdotq-iq3-sign.cuh"
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while(0)

template <ggml_type t> struct TT;
#define DEF(T, F, V) template <> struct TT<T> { static __device__ __forceinline__ float dot(const void * __restrict__ a, const block_q8_1 * __restrict__ b, const int & k, const int & i) { return F(a,b,k,i);} static constexpr int vdr = V; };
DEF(GGML_TYPE_IQ3_S,  vec_dot_iq3_s_q8_1,  VDR_IQ3_S_Q8_1_MMVQ)
DEF(GGML_TYPE_IQ3_XXS,vec_dot_iq3_xxs_q8_1,VDR_IQ3_XXS_Q8_1_MMVQ)
DEF(GGML_TYPE_IQ4_XS, vec_dot_iq4_xs_q8_1, VDR_IQ4_XS_Q8_1_MMVQ)
DEF(GGML_TYPE_Q4_K,   vec_dot_q4_K_q8_1,   VDR_Q4_K_Q8_1_MMVQ)
DEF(GGML_TYPE_Q2_K,   vec_dot_q2_K_q8_1,   VDR_Q2_K_Q8_1_MMVQ)
DEF(GGML_TYPE_IQ2_S,  vec_dot_iq2_s_q8_1,  VDR_IQ2_S_Q8_1_MMVQ)
DEF(GGML_TYPE_IQ2_XS, vec_dot_iq2_xs_q8_1, VDR_IQ2_XS_Q8_1_MMVQ)
DEF(GGML_TYPE_IQ2_XXS,vec_dot_iq2_xxs_q8_1,VDR_IQ2_XXS_Q8_1_MMVQ)
DEF(GGML_TYPE_IQ1_M,  vec_dot_iq1_m_q8_1,  VDR_IQ1_M_Q8_1_MMVQ)

__global__ void quant_q8_1(const float * x, block_q8_1 * y, int K) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= K) return;
    const float xi = x[i];
    float amax = fabsf(xi);
    amax = warp_reduce_max<QK8_1>(amax);
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    const int sum = warp_reduce_sum<QK8_1>((int) q);
    y[i/QK8_1].qs[i%QK8_1] = q;
    if (i % QK8_1 == 0) y[i/QK8_1].ds = make_half2(d, d*sum);
}

// V0: llama.cpp structure (NW waves per block, R rows per block, wave64)
template <ggml_type type, int NW, int R>
__launch_bounds__(NW*64, 1)
__global__ void gemv_v0(const void * __restrict__ vx, const block_q8_1 * __restrict__ y, float * __restrict__ dst, int K, int N) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk, qi = ggml_cuda_type_traits<type>::qi, vdr = TT<type>::vdr;
    const int tid = 64*threadIdx.y + threadIdx.x;
    const int row0 = R*blockIdx.x;
    const int bpr = K / qk;
    constexpr int bpi = vdr*NW*64/qi;
    float tmp[R] = {0.0f};
    for (int kbx = tid/(qi/vdr); kbx < bpr; kbx += bpi) {
        const int kby = kbx*(qk/QK8_1);
        const int kqs = vdr*(tid % (qi/vdr));
#pragma unroll
        for (int i = 0; i < R; ++i) tmp[i] += TT<type>::dot(vx, &y[kby], (row0+i)*bpr + kbx, kqs);
    }
    __shared__ float sh[NW][R][64];
#pragma unroll
    for (int i = 0; i < R; ++i) sh[threadIdx.y][i][threadIdx.x] = tmp[i];
    __syncthreads();
    if (threadIdx.y == 0) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < NW; ++w) v += sh[w][i][threadIdx.x];
            v = warp_reduce_sum<64>(v);
            if (threadIdx.x == 0 && row0 + i < N) dst[row0+i] = v;
        }
    }
}

// V1: L lanes per row; each wave handles (64/L)*R rows; NW waves per block
template <ggml_type type, int L, int R, int NW>
__launch_bounds__(NW*64, 1)
__global__ void gemv_v1(const void * __restrict__ vx, const block_q8_1 * __restrict__ y, float * __restrict__ dst, int K, int N) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk, qi = ggml_cuda_type_traits<type>::qi, vdr = TT<type>::vdr;
    constexpr int G = 64/L;
    const int lane = threadIdx.x % L, grp = threadIdx.x / L;
    const int wave = blockIdx.x*NW + threadIdx.y;
    const int rowbase = (wave*G + grp)*R;
    if (rowbase >= N) return;
    const int bpr = K / qk;
    constexpr int cpb = qi/vdr;
    const int C = bpr*cpb;
    float tmp[R] = {0.0f};
    for (int t = lane; t < C; t += L) {
        const int kbx = t / cpb, kqs = vdr*(t % cpb);
        const int kby = kbx*(qk/QK8_1);
#pragma unroll
        for (int i = 0; i < R; ++i) tmp[i] += TT<type>::dot(vx, &y[kby], (rowbase+i)*bpr + kbx, kqs);
    }
#pragma unroll
    for (int i = 0; i < R; ++i) {
        float v = warp_reduce_sum<L>(tmp[i]);
        if (lane == 0 && rowbase + i < N) dst[rowbase+i] = v;
    }
}

#include "lds_part.inc"
#include "soa_part.inc"

// pure read roofline: dwordx4 loads, grid-stride
__global__ void read_kernel(const int4 * __restrict__ p, size_t n16, float * out) {
    int4 acc = make_int4(0,0,0,0);
    for (size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x*blockDim.x) { int4 v = p[i]; acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w; }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678) out[0] = 1.0f;
}

struct Ctx { std::vector<char*> copies; size_t bytes; int N, K; block_q8_1 * y; float * dst; float * ref; int ncopy; };

template <typename F> double timeit(Ctx & c, F launch, int iters) {
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    for (int w = 0; w < 3; ++w) launch(c.copies[w % c.ncopy]);
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(a));
    for (int it = 0; it < iters; ++it) launch(c.copies[it % c.ncopy]);
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms; CK(hipEventElapsedTime(&ms, a, b));
    return ms*1000.0/iters;
}

static double check(Ctx & c) {
    std::vector<float> o(c.N), r(c.N);
    CK(hipMemcpy(o.data(), c.dst, c.N*4, hipMemcpyDeviceToHost)); CK(hipMemcpy(r.data(), c.ref, c.N*4, hipMemcpyDeviceToHost));
    double num = 0, den = 0; for (int i = 0; i < c.N; ++i) { num += (o[i]-r[i])*(double)(o[i]-r[i]); den += (double)r[i]*r[i]; }
    return sqrt(num/(den+1e-30));
}

template <ggml_type type> void run_all(Ctx & c, int iters) {
    const int N = c.N, K = c.K;
    auto report = [&](const char * name, double us) {
        double err = check(c);
        printf("%-28s %9.2f us  %8.1f GB/s  relerr %.2e\n", name, us, c.bytes/us/1e3, err);
        CK(hipMemset(c.dst, 0, N*4));
    };
    hipLaunchKernelGGL((gemv_v0<type,1,1>), dim3(N), dim3(64,1), 0, 0, c.copies[0], c.y, c.ref, K, N);
    CK(hipDeviceSynchronize());
#define RUNV0(NW,R) { auto L = [&](char * w){ hipLaunchKernelGGL((gemv_v0<type,NW,R>), dim3((N+R-1)/R), dim3(64,NW), 0, 0, w, c.y, c.dst, K, N); }; double us = timeit(c, L, iters); report("v0 nw=" #NW " r=" #R, us); }
#define RUNV1(Ln,R,NW) { constexpr int rpb = (64/Ln)*R*NW; auto L = [&](char * w){ hipLaunchKernelGGL((gemv_v1<type,Ln,R,NW>), dim3((N+rpb-1)/rpb), dim3(64,NW), 0, 0, w, c.y, c.dst, K, N); }; double us = timeit(c, L, iters); report("v1 L=" #Ln " r=" #R " nw=" #NW, us); }
    RUNV0(2,1) RUNV0(1,2)
    RUNV1(64,2,1) RUNV1(32,2,1) RUNV1(32,2,4)
    if constexpr (LT<type>::words > 0) {
#define RUNV2(Ln,R,NW) { constexpr int rpb = (64/Ln)*R*NW; auto L = [&](char * w){ hipLaunchKernelGGL((gemv_v2<type,Ln,R,NW>), dim3((N+rpb-1)/rpb), dim3(64,NW), 0, 0, w, c.y, c.dst, K, N); }; double us = timeit(c, L, iters); report("v2(lds) L=" #Ln " r=" #R " nw=" #NW, us); }
        RUNV2(64,2,4) RUNV2(32,1,4) RUNV2(32,2,4)
    }
    if constexpr (type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS) {
        constexpr bool xxs = type == GGML_TYPE_IQ3_XXS;
        const size_t count = (size_t)N*K/256;
        const size_t pbytes = count*(32*sizeof(uint32_t)+sizeof(pk_meta));
        std::vector<char*> pcopies;
        for (int i = 0; i < c.ncopy; ++i) {
            char * p; CK(hipMalloc(&p, pbytes));
            uint32_t * codes = (uint32_t *) p; pk_meta * meta = (pk_meta *)(codes + count*32);
            if (xxs) hipLaunchKernelGGL(pk_pack_iq3xxs, dim3((count*8+255)/256), dim3(256), 0, 0, (const block_iq3_xxs *) c.copies[i], codes, meta, count);
            else     hipLaunchKernelGGL(pk_pack_iq3s,   dim3((count*8+255)/256), dim3(256), 0, 0, (const block_iq3_s *) c.copies[i], codes, meta, count);
            pcopies.push_back(p);
        }
        CK(hipDeviceSynchronize());
        {
            const size_t nb = count;
            const size_t sbytes = nb*(xxs ? 98 : 112);
            std::vector<char*> scopies;
            for (int i = 0; i < c.ncopy; ++i) {
                char * p; CK(hipMalloc(&p, sbytes));
                if (xxs) hipLaunchKernelGGL(soa_pack_iq3xxs, dim3((nb+255)/256), dim3(256), 0, 0, (const block_iq3_xxs *) c.copies[i], (uint8_t *) p, nb);
                else     hipLaunchKernelGGL(soa_pack_iq3s,   dim3((nb+255)/256), dim3(256), 0, 0, (const block_iq3_s *) c.copies[i], (uint8_t *) p, nb);
                scopies.push_back(p);
            }
            CK(hipDeviceSynchronize());
#define RUNSOA(Ln, NWv, KM, GRIDMUL) { int it2 = 0; auto L2 = [&](char *) { char * p = scopies[it2++ % c.ncopy]; hipLaunchKernelGGL((soa_gemv_iq3<xxs, Ln, NWv, KM>), dim3(104*GRIDMUL), dim3(64, NWv), 0, 0, (const uint8_t *) p, c.y, c.dst, K, N, nb); }; double us = timeit(c, L2, iters); double err = check(c); printf("%-28s %9.2f us  %8.1f GB/s(orig) %8.1f GB/s(soa)  relerr %.2e\n", "soa L=" #Ln " nw=" #NWv " g=104x" #GRIDMUL, us, c.bytes/us/1e3, sbytes/us/1e3, err); CK(hipMemset(c.dst, 0, N*4)); }
            if (K <= 6144) {
                RUNSOA(32, 4, 6144, 2) RUNSOA(32, 4, 6144, 4) RUNSOA(32, 4, 6144, 8) RUNSOA(32, 8, 6144, 2) RUNSOA(16, 4, 6144, 4) RUNSOA(16, 4, 6144, 8)
            } else {
                RUNSOA(32, 4, 17408, 2) RUNSOA(32, 4, 17408, 4) RUNSOA(32, 8, 17408, 1) RUNSOA(32, 8, 17408, 2) RUNSOA(16, 4, 17408, 2) RUNSOA(16, 8, 17408, 2)
            }
        }
        int * sums; CK(hipMalloc(&sums, (K/32)*4));
        const int groups = K/32;
        int it = 0;
        auto L = [&](char *) {
            char * p = pcopies[it++ % c.ncopy];
            const uint32_t * codes = (const uint32_t *) p; const pk_meta * meta = (const pk_meta *)(codes + count*32);
            if (!xxs) hipLaunchKernelGGL(pk_sum_q8, dim3((groups+127)/128), dim3(128), 0, 0, c.y, sums, groups);
            hipLaunchKernelGGL((pk_gemv<xxs>), dim3(N), dim3(128), 0, 0, codes, meta, c.y, sums, c.dst, groups);
        };
        double us = timeit(c, L, iters);
        double err = check(c);
        printf("%-28s %9.2f us  %8.1f GB/s(orig-equiv) %8.1f GB/s(packed bytes)  relerr %.2e\n", "retained packed", us, c.bytes/us/1e3, pbytes/us/1e3, err);
    }
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s weights.bin TYPE N K [iters]\n", argv[0]); return 1; }
    std::string type = argv[2]; int N = atoi(argv[3]), K = atoi(argv[4]); int iters = argc > 5 ? atoi(argv[5]) : 50;
    std::ifstream f(argv[1], std::ios::binary | std::ios::ate); size_t bytes = f.tellg(); f.seekg(0);
    std::vector<char> h(bytes); f.read(h.data(), bytes);
    Ctx c; c.bytes = bytes; c.N = N; c.K = K;
    c.ncopy = (int) std::max<size_t>(1, (size_t)(256ull<<20) / bytes + 1); if (c.ncopy > 64) c.ncopy = 64;
    for (int i = 0; i < c.ncopy; ++i) { char * d; CK(hipMalloc(&d, bytes)); CK(hipMemcpy(d, h.data(), bytes, hipMemcpyHostToDevice)); c.copies.push_back(d); }
    std::vector<float> hx(K); std::mt19937 rng(1234); std::normal_distribution<float> nd(0.f, 1.f); for (auto & v : hx) v = nd(rng);
    float * dx; CK(hipMalloc(&dx, K*4)); CK(hipMemcpy(dx, hx.data(), K*4, hipMemcpyHostToDevice));
    CK(hipMalloc(&c.y, (K/32)*sizeof(block_q8_1))); CK(hipMalloc(&c.dst, N*4)); CK(hipMalloc(&c.ref, N*4));
    hipLaunchKernelGGL(quant_q8_1, dim3((K+255)/256), dim3(256), 0, 0, dx, c.y, K); CK(hipDeviceSynchronize());
    printf("== %s %s N=%d K=%d bytes=%zu (%.2f MB) copies=%d\n", argv[1], type.c_str(), N, K, bytes, bytes/1e6, c.ncopy);
    { float * o; CK(hipMalloc(&o, 4)); auto L = [&](char * w){ hipLaunchKernelGGL(read_kernel, dim3(104*8), dim3(256), 0, 0, (const int4*)w, bytes/16, o); }; double us = timeit(c, L, iters); printf("%-28s %9.2f us  %8.1f GB/s\n", "pure-read roofline", us, bytes/us/1e3); }
    if      (type == "IQ3_S")   run_all<GGML_TYPE_IQ3_S>(c, iters);
    else if (type == "IQ3_XXS") run_all<GGML_TYPE_IQ3_XXS>(c, iters);
    else if (type == "IQ4_XS")  run_all<GGML_TYPE_IQ4_XS>(c, iters);
    else if (type == "Q4_K")    run_all<GGML_TYPE_Q4_K>(c, iters);
    else if (type == "Q2_K")    run_all<GGML_TYPE_Q2_K>(c, iters);
    else if (type == "IQ2_S")   run_all<GGML_TYPE_IQ2_S>(c, iters);
    else if (type == "IQ2_XS")  run_all<GGML_TYPE_IQ2_XS>(c, iters);
    else if (type == "IQ2_XXS") run_all<GGML_TYPE_IQ2_XXS>(c, iters);
    else if (type == "IQ1_M")   run_all<GGML_TYPE_IQ1_M>(c, iters);
    else { fprintf(stderr, "unknown type\n"); return 1; }
    return 0;
}
