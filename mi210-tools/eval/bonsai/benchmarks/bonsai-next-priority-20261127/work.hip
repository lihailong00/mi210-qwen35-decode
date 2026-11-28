#include <type_traits>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#define CK(z) do { hipError_t e=(z); if(e!=hipSuccess){fprintf(stderr,"line %d: %s\n",__LINE__,hipGetErrorString(e));exit(2);} } while(0)
struct Group { int k, uses; };
template<typename T> T* buf(size_t n){T* x;CK(hipMalloc(&x,n*sizeof(T)));return x;}

using half=__half;
struct alignas(16) q2_0_words{int4 lo,hi;};
template<int W> __device__ __forceinline__ float warp_reduce_sum(float v){for(int m=W/2;m;m/=2)v+=__shfl_xor(v,m,W);return v;}
template<bool TILED>
static __device__ __forceinline__ int64_t q2_0_offset(int row, int k, int width) {
    if constexpr (TILED) {
        return (int64_t) (row / 4) * 4 * width + 4 * k + row % 4;
    }
    return (int64_t) row * width + k;
}

template<typename T>
static __device__ __forceinline__ auto q2_0_component(T v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

template<int C, bool TILED, bool RAW, int MODE=0>
__launch_bounds__(128)
static __global__ void q2_0_verify_mfma(const uint2 * codes, const half * scales,
        const q2_0_words * activations, const float4 * activation_scales,
        const int4 * sums_in, float * dst, int chunks, int rows) {
    using int4_acc = int __attribute__((ext_vector_type(4)));
    constexpr int row_groups = 2;
    constexpr int rows_per_block = 4 * row_groups;
    constexpr int slots = 32; // 128 threads in quads, one K chunk per quad
    constexpr int parts = 128 / slots;
    const int tid = threadIdx.x;
    const int quad_lane = tid % 4;
    const int slot = tid / 4;
    const int row0 = blockIdx.x * rows_per_block;
    float sums[parts][row_groups][C] = {};
    for (int base = 0; base < chunks; base += 128) {
#pragma unroll
        for (int part = 0; part < parts; ++part) {
            const int wave_start = base + part * slots + 16 * (tid / 64);
            // Tiled weights have a multiple of 32 chunks. Both waves can skip a full slot group together.
            if ((TILED ? base + part * slots : wave_start) >= chunks) {
                continue;
            }
            const int item = wave_start + (tid % 64) / 4;
            // All lanes must reach the MFMA even for a partially filled wave.
            const int k = TILED ? item : min(item, chunks - 1);
            const q2_0_words act = activations[4 * k + quad_lane];
            const float4 dy = activation_scales[k];
            const int4 sx=RAW?sums_in[k]:make_int4(0,0,0,0);
#pragma unroll
            for (int group = 0; group < row_groups; ++group) {
                const int row = row0 + 4 * group + quad_lane;
                const int64_t offset = q2_0_offset<TILED>(row, k, chunks);
                const uint2 code = codes[offset];
                const float dw = __half2float(scales[q2_0_offset<TILED>(row, k / 2, chunks / 2)]);
                int4_acc dots = {-sx.x,-sx.y,-sx.z,-sx.w}; int4_acc odd={}, third={}, fourth={};
                // Each quad forms a 4-row x 4-column tile. Eight integer matrix
                // instructions cover 32 K values; a lane owns one row of results.
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t word = j < 4 ? code.x : code.y;
                    const uint32_t sel = (word >> (2 * (j % 4))) & 0x03030303u;
                    const int w = MODE==1 ? int(word) : RAW?int(sel):__builtin_amdgcn_perm(0x020100ff, 0x020100ff, sel);
#if defined(__gfx90a__)
                    const int a = q2_0_component(j < 4 ? act.lo : act.hi, j % 4);
                    if constexpr(MODE==2 || MODE==3){
                        if (MODE==3 && (j%4)==2) third=__builtin_amdgcn_mfma_i32_4x4x4i8(a,w,third,0,0,0);
                        else if (MODE==3 && (j%4)==3) fourth=__builtin_amdgcn_mfma_i32_4x4x4i8(a,w,fourth,0,0,0);
                        else if (j%2) odd=__builtin_amdgcn_mfma_i32_4x4x4i8(a,w,odd,0,0,0);
                        else dots=__builtin_amdgcn_mfma_i32_4x4x4i8(a,w,dots,0,0,0);
                    } else dots = __builtin_amdgcn_mfma_i32_4x4x4i8(a, w, dots, 0, 0, 0);
#endif
                }
                if constexpr(MODE==2 || MODE==3) dots+=odd+third+fourth;
                if (TILED || item < chunks) {
#pragma unroll
                    for (int col = 0; col < C; ++col) {
                        sums[part][group][col] += dw * q2_0_component(dy, col) * dots[col];
                    }
                }
            }
        }
    }
    // Match GEMV's 128-slot FP32 reduction: combine slots 64 apart, then
    // 32 apart, then use the same shuffle tree on the remaining 32 slots.
    __shared__ float partial[rows_per_block * C][slots + 1];
#pragma unroll
    for (int group = 0; group < row_groups; ++group) {
#pragma unroll
        for (int col = 0; col < C; ++col) {
            const float lo = sums[0][group][col] + sums[2][group][col];
            const float hi = sums[1][group][col] + sums[3][group][col];
            partial[(4 * group + quad_lane) * C + col][slot] = lo + hi;
        }
    }
    __syncthreads();
    for (int output = tid / slots; output < rows_per_block * C; output += 128 / slots) {
        const int lane = tid % slots;
        const float sum = warp_reduce_sum<slots>(partial[output][lane]);
        if (lane == 0) {
            dst[(output % C) * rows + row0 + output / C] = sum;
        }
    }
}
template<int C, bool TILED, bool RAW, int MODE=0>
__launch_bounds__(128)
static __global__ void q2_group(const uint2 * codes, const half * scales,
        const q2_0_words * activations, const float4 * activation_scales,
        const int4 * sums_in, float * dst, int chunks, int rows,
    const uint2 * c1,const half* s1,float *out1,int n1,
    const uint2 * c2,const half* s2,float *out2,int n2) {
    using int4_acc = int __attribute__((ext_vector_type(4)));
    constexpr int row_groups = 2;
    constexpr int rows_per_block = 4 * row_groups;
    constexpr int slots = 32; // 128 threads in quads, one K chunk per quad
    constexpr int parts = 128 / slots;
    const int tid = threadIdx.x;
    const int quad_lane = tid % 4;
    const int slot = tid / 4;
    int row0 = blockIdx.x * rows_per_block;
    if(row0>=rows){
       row0-=rows;
       if(row0>=n1){row0-=n1;codes=c2;scales=s2;dst=out2;rows=n2;}
       else{codes=c1;scales=s1;dst=out1;rows=n1;}
    }
    float sums[parts][row_groups][C] = {};
    for (int base = 0; base < chunks; base += 128) {
#pragma unroll
        for (int part = 0; part < parts; ++part) {
            const int wave_start = base + part * slots + 16 * (tid / 64);
            // Tiled weights have a multiple of 32 chunks. Both waves can skip a full slot group together.
            if ((TILED ? base + part * slots : wave_start) >= chunks) {
                continue;
            }
            const int item = wave_start + (tid % 64) / 4;
            // All lanes must reach the MFMA even for a partially filled wave.
            const int k = TILED ? item : min(item, chunks - 1);
            const q2_0_words act = activations[4 * k + quad_lane];
            const float4 dy = activation_scales[k];
            const int4 sx=RAW?sums_in[k]:make_int4(0,0,0,0);
#pragma unroll
            for (int group = 0; group < row_groups; ++group) {
                const int row = row0 + 4 * group + quad_lane;
                const int64_t offset = q2_0_offset<TILED>(row, k, chunks);
                const uint2 code = codes[offset];
                const float dw = __half2float(scales[q2_0_offset<TILED>(row, k / 2, chunks / 2)]);
                int4_acc dots = {-sx.x,-sx.y,-sx.z,-sx.w}; int4_acc odd={}, third={}, fourth={};
                if constexpr(MODE==4) {
                    const int a = act.lo.x ^ act.lo.y ^ act.lo.z ^ act.lo.w ^ act.hi.x ^ act.hi.y ^ act.hi.z ^ act.hi.w;
                    dots=__builtin_amdgcn_mfma_i32_4x4x4i8(a, static_cast<int>(code.x ^ code.y), dots,0,0,0);
                } else {
                // Each quad forms a 4-row x 4-column tile. Eight integer matrix
                // instructions cover 32 K values; a lane owns one row of results.
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t word = j < 4 ? code.x : code.y;
                    const uint32_t sel = (word >> (2 * (j % 4))) & 0x03030303u;
                    const int w = MODE==1 ? int(word) : RAW?int(sel):__builtin_amdgcn_perm(0x020100ff, 0x020100ff, sel);
#if defined(__gfx90a__)
                    const int a = q2_0_component(j < 4 ? act.lo : act.hi, j % 4);
                    if constexpr(MODE==2 || MODE==3){
                        if (MODE==3 && (j%4)==2) third=__builtin_amdgcn_mfma_i32_4x4x4i8(a,w,third,0,0,0);
                        else if (MODE==3 && (j%4)==3) fourth=__builtin_amdgcn_mfma_i32_4x4x4i8(a,w,fourth,0,0,0);
                        else if (j%2) odd=__builtin_amdgcn_mfma_i32_4x4x4i8(a,w,odd,0,0,0);
                        else dots=__builtin_amdgcn_mfma_i32_4x4x4i8(a,w,dots,0,0,0);
                    } else dots = __builtin_amdgcn_mfma_i32_4x4x4i8(a, w, dots, 0, 0, 0);
#endif
                }
                }
                if constexpr(MODE==2 || MODE==3) dots+=odd+third+fourth;
                if (TILED || item < chunks) {
#pragma unroll
                    for (int col = 0; col < C; ++col) {
                        sums[part][group][col] += dw * q2_0_component(dy, col) * dots[col];
                    }
                }
            }
        }
    }
    // Match GEMV's 128-slot FP32 reduction: combine slots 64 apart, then
    // 32 apart, then use the same shuffle tree on the remaining 32 slots.
    __shared__ float partial[rows_per_block * C][slots + 1];
#pragma unroll
    for (int group = 0; group < row_groups; ++group) {
#pragma unroll
        for (int col = 0; col < C; ++col) {
            const float lo = sums[0][group][col] + sums[2][group][col];
            const float hi = sums[1][group][col] + sums[3][group][col];
            partial[(4 * group + quad_lane) * C + col][slot] = lo + hi;
        }
    }
    __syncthreads();
    for (int output = tid / slots; output < rows_per_block * C; output += 128 / slots) {
        const int lane = tid % slots;
        const float sum = warp_reduce_sum<slots>(partial[output][lane]);
        if (lane == 0) {
            dst[(output % C) * rows + row0 + output / C] = sum;
        }
    }
}
struct Shape{int k,n;size_t byte0,output0;};
__global__ void initWords(unsigned* p,size_t count){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(blockDim.x)*gridDim.x){unsigned v=i+104729;v^=v>>16;v*=0x7feb352dU;v^=v>>15;v*=0x846ca68bU;p[i]=v^(v>>16);}
}
__global__ void initScale(half* p,size_t count){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(blockDim.x)*gridDim.x)p[i]=__float2half(0.17f+(i%29)/53.f);
}
__global__ void scan(const uint4* p,size_t vcount,unsigned* dst){
 unsigned acc=0;
 for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<vcount;i+=size_t(gridDim.x)*blockDim.x){
  uint4 v=p[i];acc^=v.x^v.y^v.z^v.w;
 }
 for(int j=32;j;j/=2)acc^=__shfl_xor(acc,j,64);if(threadIdx.x%64==0)dst[blockIdx.x*4+threadIdx.x/64]=acc;
}
int main(int argc,char**argv){
 std::ifstream in(argv[1]);std::vector<Shape> ops;int k,n;size_t wb=0,on=0;
 while(in>>k>>n){ops.push_back({k,n,wb,on});wb+=size_t(k)*n*9/32;on+=size_t(4)*n;}if(ops.size()!=401)return 1;
 auto w=buf<char>(wb);auto x=buf<q2_0_words>(4*544);auto dy=buf<float4>(544);auto isum=buf<int4>(544);
 auto y=buf<float>(on);auto check=buf<unsigned>(8192);std::vector<signed char> xv(4*32*544);std::vector<float4>d(544);std::vector<int4>ss(544);std::mt19937 rng(7338);
 for(int b=0;b<544;b++){int ar[4];float dr[4];for(int c=0;c<4;c++){ar[c]=0;for(int j=0;j<32;j++){int v=int(rng()%255)-127;xv[(4*b+c)*32+j]=v;ar[c]+=v;}dr[c]=(b*7+c)%13*0.015625f;} ss[b]=make_int4(ar[0],ar[1],ar[2],ar[3]);d[b]=make_float4(dr[0],dr[1],dr[2],dr[3]);}
 CK(hipMemcpy(x,xv.data(),xv.size(),hipMemcpyHostToDevice));CK(hipMemcpy(dy,d.data(),d.size()*sizeof(float4),hipMemcpyHostToDevice));CK(hipMemcpy(isum,ss.data(),ss.size()*sizeof(int4),hipMemcpyHostToDevice));
 hipStream_t st;CK(hipStreamCreate(&st));initWords<<<4096,256,0,st>>>((unsigned*)w,wb/4);
 for(auto a:ops){size_t nc=size_t(a.n)*a.k/32;auto codes=(uint2*)(w+a.byte0);initScale<<<std::min(size_t(4096),(nc/2+255)/256),256,0,st>>>((half*)(codes+nc),nc/2);}
 CK(hipStreamSynchronize(st));
 std::ifstream gg(argv[2]); std::vector<int> count;int width,num;while(gg>>width>>num)count.push_back(num);
 auto multi=[&](auto tag,size_t i,int nn){
   Shape a=ops[i], b=nn>=2?ops[i+1]:a,c=nn==3?ops[i+2]:a;

   auto codes=[&](Shape z){return (uint2*)(w+z.byte0);};
   auto scale=[&](Shape z){return (half*)(codes(z)+size_t(z.n)*(z.k/32));};
   if(a.k!=b.k||a.k!=c.k)exit(9);
   q2_group<4,true,false,decltype(tag)::value><<<(a.n+(nn>=2?b.n:0)+(nn==3?c.n:0))/8,128,0,st>>>(codes(a),scale(a),x,dy,isum,y+a.output0,a.k/32,a.n,
       codes(b),scale(b),y+b.output0,b.n,codes(c),scale(c),y+c.output0,c.n);
 };
 hipGraph_t graphs[5];hipGraphExec_t exe[5];std::vector<float>ref(on),res(on);
 for(int m=0;m<5;m++){
  CK(hipMemset(y,0x7f,on*4));CK(hipDeviceSynchronize());
  CK(hipStreamBeginCapture(st,hipStreamCaptureModeGlobal));size_t i=0;for(int nn:count){switch(m){case 0:multi(std::integral_constant<int,0>{},i,nn);break;case 1:multi(std::integral_constant<int,1>{},i,nn);break;case 2:multi(std::integral_constant<int,2>{},i,nn);break;case 3:multi(std::integral_constant<int,3>{},i,nn);break;case 4:multi(std::integral_constant<int,4>{},i,nn);break;}i+=nn;}if(i!=ops.size())exit(7);CK(hipStreamEndCapture(st,&graphs[m]));CK(hipGraphInstantiate(&exe[m],graphs[m],nullptr,nullptr,0));CK(hipGraphLaunch(exe[m],st));CK(hipStreamSynchronize(st));
  if(m==0){CK(hipMemcpy(ref.data(),y,on*4,hipMemcpyDeviceToHost));CK(hipMemset(y,0x7f,on*4));}
  if(m==2||m==3){CK(hipMemcpy(res.data(),y,on*4,hipMemcpyDeviceToHost));size_t dif=0;for(size_t i=0;i<on;i++)dif+=memcmp(&ref[i],&res[i],4)!=0;printf("exact %zu %zu bytes %zu\n",dif,on,wb);if(dif)return 3;}
 }
 hipEvent_t a,b;CK(hipEventCreate(&a));CK(hipEventCreate(&b));
 for(int rep=0;rep<9;rep++)for(int j=0;j<5;j++){
  int m=(rep+j)%5;CK(hipGraphLaunch(exe[m],st));CK(hipEventRecord(a,st));for(int it=0;it<4;it++)CK(hipGraphLaunch(exe[m],st));CK(hipEventRecord(b,st));CK(hipEventSynchronize(b));float ms;CK(hipEventElapsedTime(&ms,a,b));printf("%d,%d,%.6f\n",rep,m,ms/4);
 }
 return 0;
}
