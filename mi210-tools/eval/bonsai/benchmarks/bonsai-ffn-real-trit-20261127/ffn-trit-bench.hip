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


#define ggml_hip_perm __builtin_amdgcn_perm
template<int C, bool TILED>
static __device__ __forceinline__ void q2_0_verify_rows(const uint2 * codes, const half * scales,
        const q2_0_words * activations, const float4 * activation_scales,
        float * dst, int chunks, int rows, int row0) {
    using int4_acc = int __attribute__((ext_vector_type(4)));
    constexpr int row_groups = 2;
    constexpr int rows_per_block = 4 * row_groups;
    constexpr int slots = 32; // 128 threads in quads, one K chunk per quad
    constexpr int parts = 128 / slots;
    const int tid = threadIdx.x;
    const int quad_lane = tid % 4;
    const int slot = tid / 4;
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
#pragma unroll
            for (int group = 0; group < row_groups; ++group) {
                const int row = row0 + 4 * group + quad_lane;
                const int64_t offset = q2_0_offset<TILED>(row, k, chunks);
                const uint2 code = codes[offset];
                const float dw = __half2float(scales[q2_0_offset<TILED>(row, k / 2, chunks / 2)]);
                int4_acc dots = {};
                // Each quad forms a 4-row x 4-column tile. Eight integer matrix
                // instructions cover 32 K values; a lane owns one row of results.
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t word = j < 4 ? code.x : code.y;
                    const uint32_t sel = (word >> (2 * (j % 4))) & 0x03030303u;
                    const int w = ggml_hip_perm(0x020100ff, 0x020100ff, sel);
#if defined(__gfx90a__)
                    const int a = q2_0_component(j < 4 ? act.lo : act.hi, j % 4);
                    dots = __builtin_amdgcn_mfma_i32_4x4x4i8(a, w, dots, 0, 0, 0);
#endif
                }
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
template<int C, bool TILED>
__launch_bounds__(128)
static __global__ void q2_0_verify_mfma(const uint2 * codes, const half * scales, const q2_0_words * activations,
        const float4 * activation_scales, float * dst, int chunks, int rows) {
    q2_0_verify_rows<C, TILED>(codes, scales, activations, activation_scales, dst, chunks, rows, blockIdx.x * 8);
}

struct q2_0_projection {
    const uint2 * codes;
    const half * scales;
    float * dst;
    int rows;
};

template<int C>
__launch_bounds__(128)
static __global__ void q2_0_verify_group(q2_0_projection first, q2_0_projection second, q2_0_projection third,
        const q2_0_words * activations, const float4 * activation_scales, int chunks) {
    int row = blockIdx.x * 8;
    q2_0_projection proj = first;
    if (row >= first.rows) {
        row -= first.rows;
        proj = second;
        if (row >= second.rows) {
            row -= second.rows;
            proj = third;
        }
    }
    q2_0_verify_rows<C, true>(proj.codes, proj.scales, activations, activation_scales, proj.dst, chunks, proj.rows, row);
}

// Three arrays hold 4, 2 and 0.5 bytes per 32 weights. Two halves share the last byte.
struct TritFfnMatrix {
    const unsigned *four;
    const unsigned short *two;
    const unsigned char *tail;
    const half *scales;
    float *out;
    int rows;
};

static TritFfnMatrix trit_matrix(const void *data, int rows, int chunks, float *out) {
    const size_t nc = size_t(rows) * chunks;
    auto *four = static_cast<const unsigned *>(data);
    auto *two = reinterpret_cast<const unsigned short *>(four + nc);
    auto *tail = reinterpret_cast<const unsigned char *>(two + nc);
    return {four, two, tail, reinterpret_cast<const half *>(tail + nc / 2), out, rows};
}

struct RawTernaryBlock { unsigned short scale; unsigned char codes[16]; };
static_assert(sizeof(RawTernaryBlock) == 18);
__device__ __forceinline__ unsigned read_code(const RawTernaryBlock &z,int i) {
    return (z.codes[i/4] >> (2*(i%4))) & 3;
}
template<bool FIXED>
static __global__ void trit_repack(const RawTernaryBlock *raw, void *packed, int rows, int blocks) {
    const size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t nb=size_t(rows)*blocks;
    if(i>=nb) return;
    const int row=i/blocks, k=i%blocks;
    const size_t bo=q2_0_offset<true>(row,k,blocks);
    auto *four=static_cast<unsigned*>(packed);
    auto *two=reinterpret_cast<unsigned short*>(four+2*nb);
    auto *last=reinterpret_cast<unsigned char*>(two+2*nb);
    auto *scale=reinterpret_cast<unsigned short*>(last+nb);
    const RawTernaryBlock z=raw[i]; unsigned both=0;
    for(int h=0;h<2;h++) {
        unsigned head=0, mid=0;
        for(int v=0;v<6;v++) {
            unsigned b=0;
            for (int j=0;j<5;j++) {
                if constexpr(FIXED) {
                    const int pos=v<4 ? 4*j+v : 20+2*j+v-4;
                    b=3*b+read_code(z,h*32+pos);
                } else {
                    const int pos=j<4 ? 4*v+j : 24+v;
                    b+=(j==0?1:j==1?3:j==2?9:j==3?27:81)*read_code(z,h*32+pos);
                }
            }
            if constexpr(FIXED) b=(256*b+242)/243;
            if(v<4) head|=b<<(8*v); else mid|=b<<(8*(v-4));
        }
        const size_t co=q2_0_offset<true>(row,2*k+h,2*blocks);
        four[co]=head;two[co]=mid;
        both|=(read_code(z,h*32+30)|(read_code(z,h*32+31)<<2))<<(4*h);
    }
    last[bo]=both;scale[bo]=z.scale;
}
// Look up the four low-order ternary digits as four signed int8 weights.
__device__ static const unsigned trit_words[256]={
    0xffffffffu, 0xffffff00u, 0xffffff01u, 0xffff00ffu, 0xffff0000u, 0xffff0001u, 0xffff01ffu, 0xffff0100u,
    0xffff0101u, 0xff00ffffu, 0xff00ff00u, 0xff00ff01u, 0xff0000ffu, 0xff000000u, 0xff000001u, 0xff0001ffu,
    0xff000100u, 0xff000101u, 0xff01ffffu, 0xff01ff00u, 0xff01ff01u, 0xff0100ffu, 0xff010000u, 0xff010001u,
    0xff0101ffu, 0xff010100u, 0xff010101u, 0x00ffffffu, 0x00ffff00u, 0x00ffff01u, 0x00ff00ffu, 0x00ff0000u,
    0x00ff0001u, 0x00ff01ffu, 0x00ff0100u, 0x00ff0101u, 0x0000ffffu, 0x0000ff00u, 0x0000ff01u, 0x000000ffu,
    0x00000000u, 0x00000001u, 0x000001ffu, 0x00000100u, 0x00000101u, 0x0001ffffu, 0x0001ff00u, 0x0001ff01u,
    0x000100ffu, 0x00010000u, 0x00010001u, 0x000101ffu, 0x00010100u, 0x00010101u, 0x01ffffffu, 0x01ffff00u,
    0x01ffff01u, 0x01ff00ffu, 0x01ff0000u, 0x01ff0001u, 0x01ff01ffu, 0x01ff0100u, 0x01ff0101u, 0x0100ffffu,
    0x0100ff00u, 0x0100ff01u, 0x010000ffu, 0x01000000u, 0x01000001u, 0x010001ffu, 0x01000100u, 0x01000101u,
    0x0101ffffu, 0x0101ff00u, 0x0101ff01u, 0x010100ffu, 0x01010000u, 0x01010001u, 0x010101ffu, 0x01010100u,
    0x01010101u, 0xffffffffu, 0xffffff00u, 0xffffff01u, 0xffff00ffu, 0xffff0000u, 0xffff0001u, 0xffff01ffu,
    0xffff0100u, 0xffff0101u, 0xff00ffffu, 0xff00ff00u, 0xff00ff01u, 0xff0000ffu, 0xff000000u, 0xff000001u,
    0xff0001ffu, 0xff000100u, 0xff000101u, 0xff01ffffu, 0xff01ff00u, 0xff01ff01u, 0xff0100ffu, 0xff010000u,
    0xff010001u, 0xff0101ffu, 0xff010100u, 0xff010101u, 0x00ffffffu, 0x00ffff00u, 0x00ffff01u, 0x00ff00ffu,
    0x00ff0000u, 0x00ff0001u, 0x00ff01ffu, 0x00ff0100u, 0x00ff0101u, 0x0000ffffu, 0x0000ff00u, 0x0000ff01u,
    0x000000ffu, 0x00000000u, 0x00000001u, 0x000001ffu, 0x00000100u, 0x00000101u, 0x0001ffffu, 0x0001ff00u,
    0x0001ff01u, 0x000100ffu, 0x00010000u, 0x00010001u, 0x000101ffu, 0x00010100u, 0x00010101u, 0x01ffffffu,
    0x01ffff00u, 0x01ffff01u, 0x01ff00ffu, 0x01ff0000u, 0x01ff0001u, 0x01ff01ffu, 0x01ff0100u, 0x01ff0101u,
    0x0100ffffu, 0x0100ff00u, 0x0100ff01u, 0x010000ffu, 0x01000000u, 0x01000001u, 0x010001ffu, 0x01000100u,
    0x01000101u, 0x0101ffffu, 0x0101ff00u, 0x0101ff01u, 0x010100ffu, 0x01010000u, 0x01010001u, 0x010101ffu,
    0x01010100u, 0x01010101u, 0xffffffffu, 0xffffff00u, 0xffffff01u, 0xffff00ffu, 0xffff0000u, 0xffff0001u,
    0xffff01ffu, 0xffff0100u, 0xffff0101u, 0xff00ffffu, 0xff00ff00u, 0xff00ff01u, 0xff0000ffu, 0xff000000u,
    0xff000001u, 0xff0001ffu, 0xff000100u, 0xff000101u, 0xff01ffffu, 0xff01ff00u, 0xff01ff01u, 0xff0100ffu,
    0xff010000u, 0xff010001u, 0xff0101ffu, 0xff010100u, 0xff010101u, 0x00ffffffu, 0x00ffff00u, 0x00ffff01u,
    0x00ff00ffu, 0x00ff0000u, 0x00ff0001u, 0x00ff01ffu, 0x00ff0100u, 0x00ff0101u, 0x0000ffffu, 0x0000ff00u,
    0x0000ff01u, 0x000000ffu, 0x00000000u, 0x00000001u, 0x000001ffu, 0x00000100u, 0x00000101u, 0x0001ffffu,
    0x0001ff00u, 0x0001ff01u, 0x000100ffu, 0x00010000u, 0x00010001u, 0x000101ffu, 0x00010100u, 0x00010101u,
    0x01ffffffu, 0x01ffff00u, 0x01ffff01u, 0x01ff00ffu, 0x01ff0000u, 0x01ff0001u, 0x01ff01ffu, 0x01ff0100u,
    0x01ff0101u, 0x0100ffffu, 0x0100ff00u, 0x0100ff01u, 0x010000ffu, 0x01000000u, 0x01000001u, 0x010001ffu,
    0x01000100u, 0x01000101u, 0x0101ffffu, 0x0101ff00u, 0x0101ff01u, 0x010100ffu, 0x01010000u, 0x01010001u,
    0x010101ffu, 0x01010100u, 0x01010101u, 0xffffffffu, 0xffffff00u, 0xffffff01u, 0xffff00ffu, 0xffff0000u,
    0xffff0001u, 0xffff01ffu, 0xffff0100u, 0xffff0101u, 0xff00ffffu, 0xff00ff00u, 0xff00ff01u, 0xff0000ffu
};
__device__ __forceinline__ unsigned high_trits(unsigned v) {
    const unsigned high=0x80808080u;
    const unsigned biased=(v&0x7f7f7f7fu)|high;
    const unsigned ge81=(biased-0x51515151u)|v;
    const unsigned ge162=(biased-0x22222222u)&v;
    return (((~ge81 >> 7)&0x01010101u)*255u) | ((ge162>>7)&0x01010101u);
}
__device__ __forceinline__ unsigned signed_tail(unsigned nib) {
    return (((nib&3u)-1u)&255u) | (((((nib>>2)&3u)-1u)&255u)<<8);
}
// For x=ceil(256*n/243), five successive base-3 fractional digits recover n.
// Each 16-bit lane holds one byte and its carry. The high byte contains the digit.
__device__ __forceinline__ unsigned trit_step(unsigned &v) {
    v=(v&0x00ff00ffu)*3u;
    return ((v&0x03000300u)+0xff00ff00u)&0xff00ff00u;
}
__device__ __forceinline__ unsigned trit_pair(unsigned &v) {
    const unsigned t=trit_step(v);
    return (t>>8&255u)|(t>>16&0xff00u);
}
template<int MODE>
__device__ __forceinline__ void unpack_chunk(unsigned head,unsigned mid,unsigned nib,const unsigned *table, unsigned (&out)[8]) {
    if constexpr(MODE==3) {
        unsigned even=head&0x00ff00ffu, odd=(head>>8)&0x00ff00ffu;
#pragma unroll
        for(int j=0;j<5;j++) {
            unsigned a=trit_step(even), b=trit_step(odd);
            out[j]=(a>>8)|b;
        }
        unsigned v=(mid&255u)|((mid&0xff00u)<<8);
#pragma unroll
        for(int j=5;j<7;j++) {
            unsigned a=trit_pair(v), b=trit_pair(v);out[j]=a|(b<<16);
        }
        out[7]=trit_pair(v)|(signed_tail(nib)<<16);
    } else {
#pragma unroll
        for(int j=0;j<4;j++) out[j]=table[(head>>(8*j))&255u];
        out[4]=table[mid&255u];out[5]=table[mid>>8];
        out[6]=high_trits(head);
        out[7]=(high_trits(mid)&0xffffu)|(signed_tail(nib)<<16);
    }
}

template<int MODE, bool GROUP>
__launch_bounds__(128)
static __global__ void trit_ffn(TritFfnMatrix first,TritFfnMatrix second,
        const q2_0_words *activations,const float4 *activation_scales,int chunks) {
    constexpr int C=4; constexpr bool TILED=true;
    int row0=blockIdx.x*8;
    TritFfnMatrix mat=first;
    if constexpr(GROUP) if(row0>=mat.rows) {row0-=mat.rows;mat=second;}
    const int rows=mat.rows;
    __shared__ unsigned shared_table[MODE==2?256:1];
    const unsigned *table=trit_words;
    if constexpr(MODE==2) {
        for(int i=threadIdx.x;i<256;i+=blockDim.x) shared_table[i]=trit_words[i];
        __syncthreads();table=shared_table;
    }
    using int4_acc = int __attribute__((ext_vector_type(4)));
    constexpr int row_groups = 2;
    constexpr int rows_per_block = 4 * row_groups;
    constexpr int slots = 32; // 128 threads in quads, one K chunk per quad
    constexpr int parts = 128 / slots;
    const int tid = threadIdx.x;
    const int quad_lane = tid % 4;
    const int slot = tid / 4;
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
#pragma unroll
            for (int group = 0; group < row_groups; ++group) {
                const int row = row0 + 4 * group + quad_lane;
                const int64_t offset = q2_0_offset<TILED>(row, k, chunks);
                unsigned decoded[8];
                const size_t bo=q2_0_offset<true>(row,k/2,chunks/2);
                unpack_chunk<MODE>(mat.four[offset],mat.two[offset],(mat.tail[bo]>>(4*(k%2)))&15,table,decoded);
                const float dw = __half2float(mat.scales[bo]);
                int4_acc dots = {};
                // Each quad forms a 4-row x 4-column tile. Eight integer matrix
                // instructions cover 32 K values; a lane owns one row of results.
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int w=static_cast<int>(decoded[j]);
#if defined(__gfx90a__)
                    const int a = q2_0_component(j < 4 ? act.lo : act.hi, j % 4);
                    dots = __builtin_amdgcn_mfma_i32_4x4x4i8(a, w, dots, 0, 0, 0);
#endif
                }
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
            mat.out[(output % C) * rows + row0 + output / C] = sum;
        }
    }
}

#include <string>
#include <cstdint>
#include <cerrno>
#include <unistd.h>
#include <fcntl.h>
#include <stdexcept>
#include <limits>
#include <array>
struct BaseBlock {half d; unsigned char qs[16];};
static_assert(sizeof(BaseBlock)==18);
template<bool TILED>
static __global__ void q2_0_repack_kernel(const BaseBlock * src, uint2 * codes, half * scales, const int64_t nblocks, int blocks_per_row) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nblocks) {
        return;
    }
    const BaseBlock & b = src[i];
    const int row = i / blocks_per_row;
    const int block = i % blocks_per_row;
    scales[TILED ? q2_0_offset<true>(row, block, blocks_per_row) : i] = b.d;
#pragma unroll
    for (int chunk = 0; chunk < 2; ++chunk) {
        uint32_t words[2] = {};
#pragma unroll
        for (int word = 0; word < 2; ++word) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int k = 32*chunk + 16*word + j;
                const uint32_t q = (b.qs[k/4] >> (2*(k%4))) & 3;
                words[word] |= q << (8*(j%4) + 2*(j/4));
            }
        }
        codes[TILED ? q2_0_offset<true>(row, 2*block + chunk, 2*blocks_per_row) : 2*i + chunk] = make_uint2(words[0], words[1]);
    }
}


template<int MODE>
__global__ void roundtrip(const RawTernaryBlock *raw,TritFfnMatrix m,int blocks,unsigned *errors) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; if(i>=size_t(m.rows)*blocks)return;
    int row=i/blocks, k=i%blocks;size_t bo=q2_0_offset<true>(row,k,blocks);
    bool bad=reinterpret_cast<const unsigned short*>(m.scales)[bo]!=raw[i].scale;
    for(int h=0;h<2;h++) {
        size_t co=q2_0_offset<true>(row,2*k+h,2*blocks);unsigned wd[8];
        unpack_chunk<MODE>(m.four[co],m.two[co],(m.tail[bo]>>(4*h))&15,trit_words,wd);
        for(int j=0;j<32;j++) {
            unsigned original=read_code(raw[i],32*h+j);
            bad|=original==3 || ((wd[j/4]>>(8*(j%4)))&255)!=((original-1u)&255u);
        }
    }
    if(bad) atomicAdd(errors,1);
}
struct LayerWeight {
    std::string name; int k,n; uint64_t file,bytes,output_offset;
    uint2 *base;void *lut,*fraction;
};
static void read_at(int fd, unsigned char *dst,size_t bytes,off_t pos){
    while(bytes){ssize_t n=pread(fd,dst,bytes,pos); if(n<0 && errno==EINTR)continue;
        if(n<=0)throw std::runtime_error("short GGUF read");dst+=n;bytes-=n;pos+=n;}
}
__global__ void check_outputs(const float *ref,const float *out,size_t size,unsigned *err){
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size;i+=size_t(gridDim.x)*blockDim.x)
        if(!isfinite(ref[i])||__float_as_uint(ref[i])!=__float_as_uint(out[i])) atomicAdd(err,1u);
}
int main(int argc,char **argv) {
    if(argc!=4) return 1;
    std::ifstream list(argv[1]);int fd=open(argv[2],O_RDONLY);
    if(fd<0||!list)throw std::runtime_error("cannot open manifest or GGUF");
    const int phase=atoi(argv[3]);
    hipStream_t st;CK(hipStreamCreate(&st));
    std::vector<LayerWeight> ops;LayerWeight op{};size_t outputs=0, bytes=0;
    while(list>>op.name>>op.k>>op.n>>op.file>>op.bytes) {
        op.output_offset=outputs;outputs+=4ull*op.n;bytes+=op.bytes;ops.push_back(op);
    }
    if(ops.size()!=192||bytes!=4812963840ull)throw std::runtime_error("unexpected FFN weights");
    std::vector<unsigned char> host(ops[0].bytes);
    auto *raw=buf<unsigned char>(host.size());auto *errors=buf<unsigned>(1);CK(hipMemset(errors,0,4));
    for(auto &o:ops) {
        size_t blocks=size_t(o.n)*o.k/64;
        if(o.bytes!=18*blocks||o.bytes>host.size())throw std::runtime_error("unsupported layout");
        read_at(fd,host.data(),o.bytes,o.file);CK(hipMemcpy(raw,host.data(),o.bytes,hipMemcpyHostToDevice));
        o.base=reinterpret_cast<uint2*>(buf<unsigned char>(o.bytes));
        o.lut=buf<unsigned char>(blocks*15);o.fraction=buf<unsigned char>(blocks*15);
        const auto shape=(blocks+255)/256;
        q2_0_repack_kernel<true><<<shape,256,0,st>>>(reinterpret_cast<const BaseBlock*>(raw),o.base,reinterpret_cast<half*>(o.base+2*blocks),blocks,o.k/64);
        trit_repack<false><<<shape,256,0,st>>>(reinterpret_cast<const RawTernaryBlock*>(raw),o.lut,o.n,o.k/64);
        trit_repack<true><<<shape,256,0,st>>>(reinterpret_cast<const RawTernaryBlock*>(raw),o.fraction,o.n,o.k/64);
        roundtrip<1><<<shape,256,0,st>>>(reinterpret_cast<const RawTernaryBlock*>(raw),trit_matrix(o.lut,o.n,o.k/32,nullptr),o.k/64,errors);
        roundtrip<3><<<shape,256,0,st>>>(reinterpret_cast<const RawTernaryBlock*>(raw),trit_matrix(o.fraction,o.n,o.k/32,nullptr),o.k/64,errors);
        CK(hipStreamSynchronize(st));
    }
    close(fd);unsigned bad=0;CK(hipMemcpy(&bad,errors,4,hipMemcpyDeviceToHost));
    printf("ROUNDTRIP blocks=%zu layouts=2 bad=%u\n",bytes/18,bad);if(bad)return 3;
    const size_t ac=64*(5120+17408)/32;
    auto *x=buf<q2_0_words>(4*ac);auto *dy=buf<float4>(ac);
    auto *y=buf<float>(outputs),*ref=buf<float>(outputs);
    std::vector<signed char> xv(ac*4*32);std::vector<float> ds(ac*4);
    auto fill=[&](int seed) {
        std::mt19937 rng(seed);
        for(auto &v:xv)v=static_cast<int>(rng()%255)-127;
        for(auto &v:ds)v=((rng()%4095)+1)*(1.0f/131072);
        CK(hipMemcpy(x,xv.data(),xv.size(),hipMemcpyHostToDevice));
        CK(hipMemcpy(dy,ds.data(),ds.size()*4,hipMemcpyHostToDevice));
    };
    hipGraphExec_t graph[3][4];hipGraph_t captures[3][4];
    for(int part=0;part<3;part++)for(int m=0;m<4;m++){
        CK(hipStreamBeginCapture(st,hipStreamCaptureModeGlobal));size_t start=0;
        for(int layer=0;layer<64;layer++)for(int pair=0;pair<2;pair++) {
            auto &a=ops[3*layer+2*pair];auto &b=ops[3*layer+(pair?2:1)];
            int chunks=a.k/32;auto *act=x+4*start;auto *sc=dy+start;start+=chunks;
            if((part==1&&pair)||(part==2&&!pair))continue;
            int rows=a.n+(pair?0:b.n);
            if(m==0) {
                auto base=[&](const LayerWeight &q){return q2_0_projection{q.base,reinterpret_cast<const half*>(q.base+size_t(q.n)*chunks),y+q.output_offset,q.n};};
                auto p=base(a),p2=base(b);
                if(pair)q2_0_verify_mfma<4,true><<<rows/8,128,0,st>>>(p.codes,p.scales,act,sc,p.dst,chunks,rows);
                else q2_0_verify_group<4><<<rows/8,128,0,st>>>(p,p2,{},act,sc,chunks);
            } else {
                auto desc=[&](LayerWeight &o){return trit_matrix(m==3?o.fraction:o.lut,o.n,chunks,y+o.output_offset);};
                auto p=desc(a),p2=desc(b);
#define RUN(M) if(pair){trit_ffn<M,false><<<rows/8,128,0,st>>>(p,p2,act,sc,chunks);}else{trit_ffn<M,true><<<rows/8,128,0,st>>>(p,p2,act,sc,chunks);}
                if(m==1){RUN(1)}else if(m==2){RUN(2)}else{RUN(3)}
#undef RUN
            }
        }
        if(start!=ac)throw std::runtime_error("activation layout");
        CK(hipStreamEndCapture(st,&captures[part][m]));CK(hipGraphInstantiate(&graph[part][m],captures[part][m],nullptr,nullptr,0));
    }
    for(int seed:{19,5721}){
        fill(seed);for(int m=0;m<4;m++){
            CK(hipMemsetAsync(y,0xff,outputs*4,st));CK(hipGraphLaunch(graph[0][m],st));
            if(m==0) CK(hipMemcpyAsync(ref,y,outputs*4,hipMemcpyDeviceToDevice,st));
            else {
                CK(hipMemsetAsync(errors,0,4,st));check_outputs<<<4096,256,0,st>>>(ref,y,outputs,errors);
                CK(hipStreamSynchronize(st));CK(hipMemcpy(&bad,errors,4,hipMemcpyDeviceToHost));
                printf("EXACT seed=%d mode=%d values=%zu bad=%u\n",seed,m,outputs,bad);if(bad)return 4;
            }
        }
    }
    hipEvent_t t0,t1;CK(hipEventCreate(&t0));CK(hipEventCreate(&t1));
    for(int part=0;part<3;part++) for(int rep=0;rep<16;rep++) for(int j=0;j<4;j++){
        int m=(rep+phase+(rep%2?3-j:j))%4;
        CK(hipGraphLaunch(graph[part][m],st));CK(hipEventRecord(t0,st));
        for(int i=0;i<4;i++)CK(hipGraphLaunch(graph[part][m],st));
        CK(hipEventRecord(t1,st));CK(hipEventSynchronize(t1));float ms;CK(hipEventElapsedTime(&ms,t0,t1));
        printf("TIME,%d,%d,%d,%.6f\n",part,rep,m,ms/4);
    }
    return 0;
}
