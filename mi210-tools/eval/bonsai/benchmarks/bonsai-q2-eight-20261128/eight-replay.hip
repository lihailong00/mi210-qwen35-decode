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
#define CK(z)                                                                                                \
    do {                                                                                                     \
        hipError_t e = (z);                                                                                  \
        if (e != hipSuccess) {                                                                               \
            fprintf(stderr, "line %d: %s\n", __LINE__, hipGetErrorString(e));                                \
            exit(2);                                                                                         \
        }                                                                                                    \
    } while (0)
struct Group {
    int k, uses;
};
template <typename T> T *buf(size_t n) {
    T *x;
    CK(hipMalloc(&x, n * sizeof(T)));
    return x;
}

using half = __half;
struct alignas(16) q2_0_words {
    int4 lo, hi;
};
template <int W> __device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int m = W / 2; m; m /= 2)
        v += __shfl_xor(v, m, W);
    return v;
}
template <bool TILED> static __device__ __forceinline__ int64_t q2_0_offset(int row, int k, int width) {
    if constexpr (TILED) {
        return (int64_t)(row / 4) * 4 * width + 4 * k + row % 4;
    }
    return (int64_t)row * width + k;
}

template <typename T> static __device__ __forceinline__ auto q2_0_component(T v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

#define ggml_hip_perm __builtin_amdgcn_perm
template <int C, bool TILED>
static __device__ __forceinline__ void
q2_0_verify_rows(const uint2 *codes, const half *scales, const q2_0_words *activations,
                 const float4 *activation_scales, float *dst, int chunks, int rows, int row0) {
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
template <int C, bool TILED>
__launch_bounds__(128) static __global__
    void q2_0_verify_mfma(const uint2 *codes, const half *scales, const q2_0_words *activations,
                          const float4 *activation_scales, float *dst, int chunks, int rows) {
    q2_0_verify_rows<C, TILED>(codes, scales, activations, activation_scales, dst, chunks, rows,
                               blockIdx.x * 8);
}

struct q2_0_projection {
    const uint2 *codes;
    const half *scales;
    float *dst;
    int rows;
};

template <int C>
__launch_bounds__(128) static __global__
    void q2_0_verify_group(q2_0_projection first, q2_0_projection second, q2_0_projection third,
                           const q2_0_words *activations, const float4 *activation_scales, int chunks) {
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
    q2_0_verify_rows<C, true>(proj.codes, proj.scales, activations, activation_scales, proj.dst, chunks,
                              proj.rows, row);
}

#define ggml_cuda_dp4a(a, b, c) __builtin_amdgcn_sdot4(a, b, c, false)
template <int C, bool TILED>
static __device__ __forceinline__ void
q2_0_verify_rows_wide(const uint2 *codes, const half *scales, const q2_0_words *activations,
                      const float4 *activation_scales, const q2_0_words *activations_hi,
                      const float4 *scales_hi, float *dst, int chunks, int rows, int row0) {
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
            const q2_0_words act_hi = activations_hi[C == 5 ? k : 4 * k + quad_lane];
            const float4 dy_hi =
                C == 5 ? make_float4(reinterpret_cast<const float *>(scales_hi)[k], 0, 0, 0) : scales_hi[k];
#pragma unroll
            for (int group = 0; group < row_groups; ++group) {
                const int row = row0 + 4 * group + quad_lane;
                const int64_t offset = q2_0_offset<TILED>(row, k, chunks);
                const uint2 code = codes[offset];
                const float dw = __half2float(scales[q2_0_offset<TILED>(row, k / 2, chunks / 2)]);
                int4_acc dots = {}, dots_hi = {};
                int fifth = 0;
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
                    const int a_hi = q2_0_component(j < 4 ? act_hi.lo : act_hi.hi, j % 4);
                    if constexpr (C == 5)
                        fifth = ggml_cuda_dp4a(a_hi, w, fifth);
                    else
                        dots_hi = __builtin_amdgcn_mfma_i32_4x4x4i8(a_hi, w, dots_hi, 0, 0, 0);
#endif
                }
                if (TILED || item < chunks) {
#pragma unroll
                    for (int col = 0; col < C; ++col) {
                        sums[part][group][col] +=
                            dw * q2_0_component(col < 4 ? dy : dy_hi, col % 4) *
                            (col < 4 ? dots[col % 4] : (C == 5 ? fifth : dots_hi[col % 4]));
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
template <int C>
__launch_bounds__(128) static __global__
    void q2_0_verify_group_wide(q2_0_projection first, q2_0_projection second, q2_0_projection third,
                                const q2_0_words *activations, const float4 *activation_scales,
                                const q2_0_words *hi, const float4 *dy_hi, int chunks) {
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
    q2_0_verify_rows_wide<C, true>(proj.codes, proj.scales, activations, activation_scales, hi, dy_hi,
                                   proj.dst, chunks, proj.rows, row);
}

#include <string>
#include <cstdint>
#include <cerrno>
#include <unistd.h>
#include <fcntl.h>
#include <stdexcept>
#include <limits>
#include <array>
struct BaseBlock {
    half d;
    unsigned char qs[16];
};
static_assert(sizeof(BaseBlock) == 18);
template <bool TILED>
static __global__ void q2_0_repack_kernel(const BaseBlock *src, uint2 *codes, half *scales,
                                          const int64_t nblocks, int blocks_per_row) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblocks) {
        return;
    }
    const BaseBlock &b = src[i];
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
                const int k = 32 * chunk + 16 * word + j;
                const uint32_t q = (b.qs[k / 4] >> (2 * (k % 4))) & 3;
                words[word] |= q << (8 * (j % 4) + 2 * (j / 4));
            }
        }
        codes[TILED ? q2_0_offset<true>(row, 2 * block + chunk, 2 * blocks_per_row) : 2 * i + chunk] =
            make_uint2(words[0], words[1]);
    }
}

struct LayerWeight {
    std::string name;
    int k, n;
    uint64_t file, bytes, output_offset;
    uint2 *base;
    void *paired;
};
static void read_at(int fd, unsigned char *dst, size_t bytes, off_t pos) {
    while (bytes) {
        ssize_t n = pread(fd, dst, bytes, pos);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            throw std::runtime_error("short GGUF read");
        dst += n;
        bytes -= n;
        pos += n;
    }
}
__global__ void check_outputs(const float *ref, const float *out, size_t size, unsigned *err) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
         i += size_t(gridDim.x) * blockDim.x)
        if (!isfinite(ref[i]) || __float_as_uint(ref[i]) != __float_as_uint(out[i]))
            atomicAdd(err, 1u);
}
// Interleave the two four-row groups for one 128-bit vector load per lane.
static __global__ void pair_pack(const uint2 *src, const half *ds, uint2 *dst, half *outscale, int rows,
                                 int chunks) {
    size_t off = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (off >= size_t(rows) * chunks)
        return;
    const int row = int(off / (4 * chunks)) * 4 + off % 4, k = int(off / 4) % chunks;
    const size_t i = (size_t(row / 8) * chunks + k) * 4 + row % 4;
    dst[2 * i + (row / 4) % 2] = src[off];
    if ((k & 1) == 0) {
        const size_t si = (size_t(row / 8) * (chunks / 2) + k / 2) * 4 + row % 4;
        outscale[2 * si + (row / 4) % 2] = ds[q2_0_offset<true>(row, k / 2, chunks / 2)];
    }
}

// A block covers eight output rows and four columns. Consecutive blocks cover the
// two halves of one eight-column verification and can reuse the same weight lines.
template <int CHUNKS, int SLOTS, bool INTERLEAVE>
__launch_bounds__(4 * SLOTS) static __global__
    void q2_0_eight_rows(q2_0_projection first, q2_0_projection second, q2_0_projection third,
                         const q2_0_words *lo, const float4 *lo_scale, const q2_0_words *hi,
                         const float4 *hi_scale) {
    using accum4 = int __attribute__((ext_vector_type(4)));
    constexpr int parts = 128 / SLOTS;
    const int thread = threadIdx.x;
    const int lane = thread % 4;
    const int slot = thread / 4;
    const int side = blockIdx.x % 2;
    int row0 = (blockIdx.x / 2) * 8;
    auto proj = first;
    if (row0 >= proj.rows) {
        row0 -= proj.rows;
        proj = second;
        if (row0 >= proj.rows) {
            row0 -= proj.rows;
            proj = third;
        }
    }
    const auto *act = side ? hi : lo;
    const auto *scales = side ? hi_scale : lo_scale;
    const auto *codes = reinterpret_cast<const uint4 *>(proj.codes) + size_t(row0 / 8) * CHUNKS * 4;
    const auto *weight_scales =
        reinterpret_cast<const half2 *>(proj.scales) + size_t(row0 / 8) * (CHUNKS / 2) * 4;
    float sums[parts][2][4] = {};
#pragma unroll
    for (int part = 0; part < parts; ++part) {
        for (int base = 0; base < CHUNKS; base += 128) {
            const int k = base + part * SLOTS + slot;
            if (k >= CHUNKS)
                continue;
            const q2_0_words x = act[4 * k + lane];
            const float4 dy = scales[k];
            const uint4 code = codes[4 * k + lane];
            const half2 ds = weight_scales[4 * (k / 2) + lane];
            accum4 dots[2] = {};
            if constexpr (INTERLEAVE) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int a = q2_0_component(j < 4 ? x.lo : x.hi, j % 4);
#pragma unroll
                    for (int g = 0; g < 2; ++g) {
                        const unsigned word = g == 0 ? (j < 4 ? code.x : code.y) : (j < 4 ? code.z : code.w);
                        const int w =
                            ggml_hip_perm(0x020100ff, 0x020100ff, (word >> (2 * (j % 4))) & 0x03030303u);
                        dots[g] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, w, dots[g], 0, 0, 0);
                    }
                }
            } else {
#pragma unroll
                for (int g = 0; g < 2; ++g) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const unsigned word = g == 0 ? (j < 4 ? code.x : code.y) : (j < 4 ? code.z : code.w);
                        const int w =
                            ggml_hip_perm(0x020100ff, 0x020100ff, (word >> (2 * (j % 4))) & 0x03030303u);
                        const int a = q2_0_component(j < 4 ? x.lo : x.hi, j % 4);
                        dots[g] = __builtin_amdgcn_mfma_i32_4x4x4i8(a, w, dots[g], 0, 0, 0);
                    }
                }
            }
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                const float dw = __half2float(g == 0 ? __low2half(ds) : __high2half(ds));
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    sums[part][g][c] += dw * q2_0_component(dy, c) * dots[g][c];
            }
        }
    }
    __shared__ float partial[32][SLOTS + 1];
#pragma unroll
    for (int g = 0; g < 2; ++g) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
#pragma unroll
            for (int delta = parts / 2; delta; delta /= 2) {
#pragma unroll
                for (int i = 0; i < delta; ++i)
                    sums[i][g][c] += sums[i + delta][g][c];
            }
            if constexpr (SLOTS == 16) {
                float v = sums[0][g][c];
#pragma unroll
                for (int delta = 8; delta; delta /= 2)
                    v += __shfl_xor(v, 4 * delta, 64);
                if (slot == 0)
                    proj.dst[(4 * side + c) * proj.rows + row0 + 4 * g + lane] = v;
            } else {
                partial[(4 * g + lane) * 4 + c][slot] = sums[0][g][c];
            }
        }
    }
    if constexpr (SLOTS != 16) {
        __syncthreads();
        for (int out = thread / SLOTS; out < 32; out += 4) {
            const float v = warp_reduce_sum<SLOTS>(partial[out][thread % SLOTS]);
            if (thread % SLOTS == 0)
                proj.dst[(4 * side + out % 4) * proj.rows + row0 + out / 4] = v;
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s weights.tsv groups.txt model.gguf repetitions phase\n", argv[0]);
        return 1;
    }
    std::ifstream list(argv[1]), groups(argv[2]);
    int fd = open(argv[3], O_RDONLY);
    if (fd < 0 || !list || !groups)
        throw std::runtime_error("input files");
    const int reps = atoi(argv[4]), phase = atoi(argv[5]);
    std::vector<std::string> modes = {"baseline", "ref4", "compact"};
    if (reps < 1 || phase < 0 || phase > 2)
        throw std::runtime_error("expected positive repetitions and a phase between 0 and 2");
    hipStream_t st;
    CK(hipStreamCreate(&st));
    std::vector<LayerWeight> ops;
    LayerWeight op{};
    size_t on = 0, wb = 0, largest = 0;
    while (list >> op.name >> op.k >> op.n >> op.file >> op.bytes) {
        op.output_offset = on;
        on += 8ull * op.n;
        wb += op.bytes;
        largest = std::max<size_t>(largest, op.bytes);
        ops.push_back(op);
    }
    std::vector<Group> sets;
    Group group{};
    size_t ac = 0, count = 0;
    while (groups >> group.k >> group.uses) {
        if ((group.k != 5120 && group.k != 6144 && group.k != 17408) || group.uses < 1 || group.uses > 3)
            throw std::runtime_error("unsupported group in manifest");
        sets.push_back(group);
        ac += group.k / 32;
        count += group.uses;
    }
    if (sets.size() != 257 || ops.size() != 401 || count != ops.size() || wb != 7199539200ull)
        throw std::runtime_error("manifest mismatch");
    std::vector<unsigned char> host(largest);
    auto *raw = buf<unsigned char>(largest);
    auto errors = buf<unsigned>(1);
    for (auto &o : ops) {
        size_t blocks = size_t(o.n) * o.k / 64;
        if (o.bytes != 18 * blocks)
            throw std::runtime_error("invalid tensor bytes");
        read_at(fd, host.data(), o.bytes, o.file);
        CK(hipMemcpy(raw, host.data(), o.bytes, hipMemcpyHostToDevice));
        o.base = reinterpret_cast<uint2 *>(buf<unsigned char>(o.bytes));
        q2_0_repack_kernel<true><<<(blocks + 255) / 256, 256, 0, st>>>(
            reinterpret_cast<const BaseBlock *>(raw), o.base, reinterpret_cast<half *>(o.base + 2 * blocks),
            blocks, o.k / 64);
        o.paired = buf<unsigned char>(o.bytes);
        auto *wp = static_cast<uint2 *>(o.paired);
        size_t nc = 2 * blocks;
        pair_pack<<<(nc + 255) / 256, 256, 0, st>>>(o.base, reinterpret_cast<const half *>(o.base + nc), wp,
                                                    reinterpret_cast<half *>(wp + nc), o.n, o.k / 32);
        CK(hipStreamSynchronize(st));
    }
    close(fd);
    CK(hipFree(raw));
    auto *x = buf<q2_0_words>(8 * ac);
    auto *dy = buf<float4>(2 * ac);
    auto *y = buf<float>(on);
    auto *ref = buf<float>(on);
    std::vector<signed char> xv(ac * 8 * 32);
    std::vector<float> ds(ac * 8);
    auto fill = [&](int seed) {
        std::mt19937 rng(seed);
        for (auto &vv : xv)
            vv = int(rng() % 255) - 127;
        for (auto &vv : ds)
            vv = ((rng() % 4095) + 1) * (1.f / 131072);
        CK(hipMemcpy(x, xv.data(), xv.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(dy, ds.data(), ds.size() * 4, hipMemcpyHostToDevice));
    };
    std::vector<hipGraphExec_t> graphs(modes.size());
    for (size_t m = 0; m < modes.size(); m++) {
        hipGraph_t cap;
        CK(hipStreamBeginCapture(st, hipStreamCaptureModeGlobal));
        size_t start = 0, i = 0;
        size_t nlaunch = 0;
        for (auto set : sets) {
            const int chunks = set.k / 32;
            auto xl = x + 4 * start;
            auto xh = x + 4 * (ac + start);
            auto sl = dy + start;
            auto sh = dy + ac + start;
            start += chunks;
            q2_0_projection pp[3] = {};
            int rows = 0;
            for (int j = 0; j < set.uses; j++) {
                auto &a = ops[i + j];
                if (a.k != set.k)
                    throw std::runtime_error("shape mismatch");
                pp[j] = {a.base, reinterpret_cast<const half *>(a.base + size_t(a.n) * chunks),
                         y + a.output_offset, a.n};
                rows += a.n;
            }
            q2_0_projection np[3] = {};
            for (int j = 0; j < set.uses; j++) {
                auto &o = ops[i + j];
                auto ptr = static_cast<uint2 *>(o.paired);
                np[j] = {ptr, reinterpret_cast<const half *>(ptr + size_t(o.n) * chunks), pp[j].dst, o.n};
            }
            i += set.uses;
            auto &mode = modes[m];
            nlaunch++;
            if (mode == "baseline")
                q2_0_verify_group_wide<8>
                    <<<rows / 8, 128, 0, st>>>(pp[0], pp[1], pp[2], xl, sl, xh, sh, chunks);
            else if (mode == "ref4") {
                q2_0_verify_group<4><<<rows / 8, 128, 0, st>>>(pp[0], pp[1], pp[2], xl, sl, chunks);
                for (int j = 0; j < set.uses; j++)
                    pp[j].dst += 4 * pp[j].rows;
                q2_0_verify_group<4><<<rows / 8, 128, 0, st>>>(pp[0], pp[1], pp[2], xh, sh, chunks);
            } else if (mode == "compact") {
                if (chunks == 160)
                    q2_0_eight_rows<160, 16, 1><<<rows / 4, 64, 0, st>>>(np[0], np[1], np[2], xl, sl, xh, sh);
                else if (chunks == 192)
                    q2_0_eight_rows<192, 32, 0>
                        <<<rows / 4, 128, 0, st>>>(np[0], np[1], np[2], xl, sl, xh, sh);
                else if (chunks == 544)
                    q2_0_eight_rows<544, 32, 0>
                        <<<rows / 4, 128, 0, st>>>(np[0], np[1], np[2], xl, sl, xh, sh);
            } else
                throw std::runtime_error("mode " + mode);
        }
        if (start != ac || i != 401 || nlaunch != 257)
            throw std::runtime_error("work count");
        CK(hipStreamEndCapture(st, &cap));
        CK(hipGraphInstantiate(&graphs[m], cap, nullptr, nullptr, 0));
    }
    for (int seed : {19, 5721}) {
        fill(seed);
        for (size_t m = 0; m < modes.size(); m++) {
            CK(hipMemsetAsync(y, 0xff, on * 4, st));
            CK(hipGraphLaunch(graphs[m], st));
            if (m == 0)
                CK(hipMemcpyAsync(ref, y, on * 4, hipMemcpyDeviceToDevice, st));
            else {
                unsigned bad;
                CK(hipMemsetAsync(errors, 0, 4, st));
                check_outputs<<<4096, 256, 0, st>>>(ref, y, on, errors);
                CK(hipStreamSynchronize(st));
                CK(hipMemcpy(&bad, errors, 4, hipMemcpyDeviceToHost));
                printf("EXACT,%d,%s,%zu,%u\n", seed, modes[m].c_str(), on, bad);
                if (bad)
                    return 4;
            }
        }
    }
    hipEvent_t ta, tb;
    CK(hipEventCreate(&ta));
    CK(hipEventCreate(&tb));
    int nc = modes.size();
    for (int rep = 0; rep < reps; rep++)
        for (int j = 0; j < nc; j++) {
            int m = (rep + phase + (rep % 2 ? nc - 1 - j : j)) % nc;
            CK(hipGraphLaunch(graphs[m], st));
            CK(hipEventRecord(ta, st));
            for (int it = 0; it < 3; it++)
                CK(hipGraphLaunch(graphs[m], st));
            CK(hipEventRecord(tb, st));
            CK(hipEventSynchronize(tb));
            float ms;
            CK(hipEventElapsedTime(&ms, ta, tb));
            printf("TIME,%d,%s,%.6f\n", rep, modes[m].c_str(), ms / 3);
        }
    return 0;
}
