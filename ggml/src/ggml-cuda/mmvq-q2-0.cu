#include "common.cuh"
#include "vecdotq.cuh"
#include "unary.cuh"
#include "fwht.cuh"
#include "iq3-repack.cuh"
#include "mmvq-fused-host.cuh"

#if defined(GGML_USE_HIP)
#include <cstdlib>

static constexpr int q2_0_repack_kind = 0x7200;

static constexpr int q2_0_tile4_kind = 0x72b4;

template<bool TILED>
static __device__ __forceinline__ int64_t q2_0_offset(int row, int k, int width) {
    if constexpr (TILED) {
        return (int64_t) (row / 4) * 4 * width + 4 * k + row % 4;
    }
    return (int64_t) row * width + k;
}

// Same-size, lossless SoA copy: eight packed bytes per 32 values followed by
// one fp16 scale per 64 values. Transpose the two-bit codes within each word
// so a shift and byte permutation feed four signed values directly to dp4a.
template<bool TILED>
static __global__ void q2_0_repack_kernel(const block_q2_0 * src, uint2 * codes, half * scales, const int64_t nblocks, int blocks_per_row) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nblocks) {
        return;
    }
    const block_q2_0 & b = src[i];
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

template<bool TILED>
static void q2_0_repack(const void * src, void * dst, const ggml_tensor * w, hipStream_t stream) {
    const int64_t nblocks = ggml_nelements(w)/QK2_0;
    auto * codes = (uint2 *) dst;
    auto * scales = (half *) (codes + 2*nblocks);
    q2_0_repack_kernel<TILED><<<(nblocks + 255)/256, 256, 0, stream>>>(
        (const block_q2_0 *) src, codes, scales, nblocks, w->ne[0] / QK2_0);
    CUDA_CHECK(cudaGetLastError());
}

static __device__ __forceinline__ float q2_0_dot(const uint2 code, const float d, const block_q8_1 & x) {
    int sum = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t a = (code.x >> (2*j)) & 0x03030303u;
        const uint32_t b = (code.y >> (2*j)) & 0x03030303u;
        const int qa = ggml_hip_perm(0x020100FF, 0x020100FF, a);
        const int qb = ggml_hip_perm(0x020100FF, 0x020100FF, b);
        sum = ggml_cuda_dp4a(qa, get_int_b4(x.qs, j), sum);
        sum = ggml_cuda_dp4a(qb, get_int_b4(x.qs, j + 4), sum);
    }
    return d * __low2float(x.ds) * sum;
}

template<int R, bool GATE, bool BIAS>
__launch_bounds__(128)
static __global__ void q2_0_gemv_kernel(
        const uint2 * codes, const half * scales, const uint2 * gate, const half * gate_scales,
        const block_q8_1 * x, float * dst, const float * bias, const int chunks) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int tid = threadIdx.x;
    const int row0 = blockIdx.x*R;
    float sums[R] = {};
    float gate_sums[R] = {};
    for (int k = tid; k < chunks; k += 128) {
        const block_q8_1 xv = x[k];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int64_t offset = (int64_t) (row0 + r)*chunks + k;
            sums[r] += q2_0_dot(codes[offset], __half2float(scales[offset/2]), xv);
            if constexpr (GATE) {
                gate_sums[r] += q2_0_dot(gate[offset], __half2float(gate_scales[offset/2]), xv);
            }
        }
    }
    // Match the generic Q2_0 GEMV: 128 lanes, then the same two-wave reduction.
    __shared__ float partial[R][64];
    __shared__ float gate_partial[GATE ? R : 1][64];
    if (tid >= 64) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            partial[r][tid - 64] = sums[r];
            if constexpr (GATE) {
                gate_partial[r][tid - 64] = gate_sums[r];
            }
        }
    }
    __syncthreads();
    if (tid >= 64) {
        return;
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        sums[r] += partial[r][tid];
        sums[r] = warp_reduce_sum<warp_size>(sums[r]);
        if constexpr (GATE) {
            gate_sums[r] += gate_partial[r][tid];
            gate_sums[r] = warp_reduce_sum<warp_size>(gate_sums[r]);
        }
        if (tid == 0) {
            float value = sums[r];
            if constexpr (GATE || BIAS) {
                value += BIAS ? bias[row0 + r] : 0.0f;
            }
            if constexpr (GATE) {
                value *= ggml_cuda_op_silu_single(gate_sums[r] + 0.0f);
            }
            dst[row0 + r] = value;
        }
    }
}

// Four lanes read the same Q8 activation and adjacent codes from four weight rows.
// Accumulate 128 logical K slots per row, as in the row-major single-token path.
template<int THREADS, bool GATE, bool BIAS>
__launch_bounds__(THREADS)
static __global__ void q2_0_tile4_gemv(const uint2 * codes, const half * scales,
        const uint2 * gate_codes, const half * gate_scales, const block_q8_1 * x,
        float * dst, const float * bias, int chunks) {
    constexpr int slots = THREADS / 4;
    constexpr int parts = 128 / slots;
    const int tid = threadIdx.x;
    const int row0 = blockIdx.x * 4;
    const int row = row0 + tid % 4;
    const int slot = tid / 4;
    float sums[parts] = {};
    float gates[parts] = {};
    for (int base = 0; base < chunks; base += 128) {
#pragma unroll
        for (int part = 0; part < parts; ++part) {
            const int k = base + part * slots + slot;
            if (k < chunks) {
                const block_q8_1 act = x[k];
                const int64_t wi = q2_0_offset<true>(row, k, chunks);
                const int64_t si = q2_0_offset<true>(row, k / 2, chunks / 2);
                sums[part] += q2_0_dot(codes[wi], __half2float(scales[si]), act);
                if constexpr (GATE) {
                    gates[part] += q2_0_dot(gate_codes[wi], __half2float(gate_scales[si]), act);
                }
            }
        }
    }
#pragma unroll
    for (int delta = parts / 2; delta > 0; delta /= 2) {
#pragma unroll
        for (int i = 0; i < delta; ++i) {
            sums[i] += sums[i + delta];
            if constexpr (GATE) {
                gates[i] += gates[i + delta];
            }
        }
    }
    __shared__ float partial[4][slots + 1];
    __shared__ float gate_partial[GATE ? 4 : 1][slots + 1];
    partial[tid % 4][slot] = sums[0];
    if constexpr (GATE) {
        gate_partial[tid % 4][slot] = gates[0];
    }
    __syncthreads();
    const int out = tid / slots;
    const int lane = tid % slots;
    float value = warp_reduce_sum<slots>(partial[out][lane]);
    if constexpr (GATE || BIAS) {
        value += BIAS ? bias[row0 + out] : 0.0f;
    }
    if constexpr (GATE) {
        const float gate = warp_reduce_sum<slots>(gate_partial[out][lane]);
        value *= ggml_cuda_op_silu_single(gate + 0.0f);
    }
    if (lane == 0) {
        dst[row0 + out] = value;
    }
}

template<bool TILED, bool GATE, bool BIAS>
static void q2_0_gemv_dispatch(const uint2 * codes, const half * scales, const uint2 * gate,
        const half * gate_scales, const block_q8_1 * x, float * dst, const float * bias,
        int K, int N, cudaStream_t stream) {
    const int chunks = K / QK8_1;
    if constexpr (!TILED) {
        q2_0_gemv_kernel<2, GATE, BIAS><<<N / 2, 128, 0, stream>>>(codes, scales, gate, gate_scales, x, dst, bias, chunks);
    } else if (K <= 5120) {
        q2_0_tile4_gemv<128, GATE, BIAS><<<N / 4, 128, 0, stream>>>(codes, scales, gate, gate_scales, x, dst, bias, chunks);
    } else {
        q2_0_tile4_gemv<256, GATE, BIAS><<<N / 4, 256, 0, stream>>>(codes, scales, gate, gate_scales, x, dst, bias, chunks);
    }
}

template<bool TILED>
static void q2_0_launch(
        const void * packed, const void * packed_gate, const block_q8_1 * x, float * dst,
        const float * bias, const int K, const int N, cudaStream_t stream) {
    const auto * codes = (const uint2 *) packed;
    const auto * scales = (const half *) (codes + (int64_t) N*K/32);
    const auto * gate = (const uint2 *) packed_gate;
    const auto * gate_scales = gate ? (const half *) (gate + (int64_t) N*K/32) : nullptr;
    if (gate) {
        q2_0_gemv_dispatch<TILED, true, false>(codes, scales, gate, gate_scales, x, dst, nullptr, K, N, stream);
    } else if (bias) {
        q2_0_gemv_dispatch<TILED, false, true>(codes, scales, nullptr, nullptr, x, dst, bias, K, N, stream);
    } else {
        q2_0_gemv_dispatch<TILED, false, false>(codes, scales, nullptr, nullptr, x, dst, nullptr, K, N, stream);
    }
}

// Verify multiple tokens with one weight load. Each column retains the
// accumulation and wave reduction order of the single-token kernel.
template<int C, bool TILED>
__launch_bounds__(128)
static __global__ void q2_0_verify_kernel(const uint2 * codes, const half * scales,
        const block_q8_1 * x, float * dst, int chunks, int x_stride, int rows) {
    constexpr int wave = ggml_cuda_get_physical_warp_size();
    constexpr int R = 2;
    const int tid = threadIdx.x;
    const int row = blockIdx.x*R;
    float sums[R][C] = {};
    for (int k = tid; k < chunks; k += 128) {
        block_q8_1 act[C];
#pragma unroll
        for (int c = 0; c < C; ++c) {
            act[c] = x[c*x_stride + k];
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int64_t offset = q2_0_offset<TILED>(row + r, k, chunks);
            const uint2 code = codes[offset];
            const float scale = __half2float(scales[q2_0_offset<TILED>(row + r, k / 2, chunks / 2)]);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                sums[r][c] += q2_0_dot(code, scale, act[c]);
            }
        }
    }
    __shared__ float partial[R][C][64];
    if (tid >= 64) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int c = 0; c < C; ++c) {
                partial[r][c][tid-64] = sums[r][c];
            }
        }
    }
    __syncthreads();
    if (tid >= 64) return;
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int c = 0; c < C; ++c) {
            const float result = warp_reduce_sum<wave>(sums[r][c] + partial[r][c][tid]);
            if (tid == 0) dst[c*rows + row + r] = result;
        }
    }
}

template<int C, bool TILED>
static void q2_0_verify(const void * packed, const block_q8_1 * x, float * dst,
        int K, int N, cudaStream_t stream) {
    const auto * codes = (const uint2 *) packed;
    const auto * scales = (const half *) (codes + (int64_t) N*K/QK8_1);
    q2_0_verify_kernel<C, TILED><<<N/2, 128, 0, stream>>>(
        codes, scales, x, dst, K/QK8_1, GGML_PAD(K, MATRIX_ROW_PADDING)/QK8_1, N);
}

template<bool TILED>
static void q2_0_dispatch(int cols, const void * packed, const void * packed_gate, const block_q8_1 * x,
        float * out, const float * residual, int K, int N, cudaStream_t stream) {
    switch (cols) {
        case 2: q2_0_verify<2, TILED>(packed, x, out, K, N, stream); break;
        case 3: q2_0_verify<3, TILED>(packed, x, out, K, N, stream); break;
        case 4: q2_0_verify<4, TILED>(packed, x, out, K, N, stream); break;
        default: q2_0_launch<TILED>(packed, packed_gate, x, out, residual, K, N, stream); break;
    }
}

// The four columns of one K chunk are contiguous. Each lane reads two aligned
// int4 vectors and shares its activation column with the other three rows via MFMA.
struct alignas(16) q2_0_words {
    int4 lo;
    int4 hi;
};
static_assert(sizeof(q2_0_words) == QK8_1);

template<int C>
static __global__ void q2_0_quantize_columns(const float * src, int64_t stride,
        q2_0_words * words, float * scales, int chunks) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int chunk = i / (4 * QK8_1);
    if (chunk >= chunks) {
        return;
    }
    const int col = (i / QK8_1) % 4;
    const int lane = i % QK8_1;
    const float x = col < C ? src[col * stride + chunk * QK8_1 + lane] : 0.0f;
    // Match quantize_q8_1: round the bytes using the FP32 scale, then store
    // the scale rounded to FP16. A three-column tile has a zero fourth column.
    const float amax = warp_reduce_max<QK8_1>(fabsf(x));
    const float d = amax / 127.0f;
    reinterpret_cast<int8_t *>(words)[i] = amax == 0.0f ? 0 : roundf(x / d);
    if (lane == 0) {
        scales[4 * chunk + col] = __half2float(__float2half(d));
    }
}

#if defined(__gfx90a__) || !defined(__HIP_DEVICE_COMPILE__)
template<typename T>
static __device__ __forceinline__ auto q2_0_component(T v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

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

#endif

static bool q2_0_weight_ok(const ggml_tensor * w) {
    return w && w->type == GGML_TYPE_Q2_0 && ggml_is_matrix(w) && ggml_is_contiguous(w) &&
        w->ne[0] >= 4096 && w->ne[0] <= 32768 && w->ne[0]%256 == 0 &&
        w->ne[1] >= 128 && w->ne[1] <= 1048576 && w->ne[1]%8 == 0 &&
        w->buffer && ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}
#endif

#if defined(GGML_USE_HIP)
static bool q2_0_mfma_enabled() {
    static const bool enabled = [] {
        for (const char * key : { "GGML_MMVQ_Q2_0", "GGML_MMVQ_Q2_0_VERIFY_BATCH", "GGML_MMVQ_Q2_0_MFMA" }) {
            const char * e = getenv(key);
            if (e && e[0] == '0') {
                return false;
            }
        }
        return true;
    }();
    return enabled;
}

static bool q2_0_use_tile4(const ggml_backend_cuda_context & ctx, const ggml_tensor * w) {
    static const bool enabled = [] {
        const char * e = getenv("GGML_MMVQ_Q2_0_TILE");
        return !e || e[0] != '0';
    }();
    return enabled && q2_0_mfma_enabled() && ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_CDNA2 &&
           w->ne[0] % (32 * QK8_1) == 0;
}

static const void * q2_0_packed(ggml_backend_cuda_context & ctx, const ggml_tensor * w, bool tiled) {
    return ggml_cuda_get_repacked(w, ctx.device, ctx.stream(), tiled ? q2_0_tile4_kind : q2_0_repack_kind,
            ggml_nbytes(w), tiled ? q2_0_repack<true> : q2_0_repack<false>);
}

template<int C, bool TILED>
static void q2_0_launch_verify_mfma(ggml_backend_cuda_context & ctx, const void * packed,
        const ggml_tensor * src, ggml_tensor * dst) {
    const int chunks = src->ne[0] / QK8_1;
    const int rows = dst->ne[0];
    const auto stream = ctx.stream();
    ggml_cuda_pool_alloc<char> scratch(ctx.pool());
    const size_t bytes = chunks * (4 * sizeof(q2_0_words) + sizeof(float4));
    auto * words = reinterpret_cast<q2_0_words *>(scratch.alloc(bytes));
    auto * dy = reinterpret_cast<float4 *>(words + 4 * chunks);
    const auto * codes = static_cast<const uint2 *>(packed);
    const auto * scales = reinterpret_cast<const half *>(codes + (int64_t) rows * chunks);
    q2_0_quantize_columns<C><<<(chunks + 1) / 2, 256, 0, stream>>>(
        static_cast<const float *>(src->data), src->nb[1] / sizeof(float), words, reinterpret_cast<float *>(dy), chunks);
#if defined(__gfx90a__) || !defined(__HIP_DEVICE_COMPILE__)
    q2_0_verify_mfma<C, TILED><<<rows / 8, 128, 0, stream>>>(codes, scales, words, dy, static_cast<float *>(dst->data), chunks, rows);
#endif
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_mmvq_q2_0_mfma(ggml_backend_cuda_context & ctx, const ggml_tensor * w,
        const ggml_tensor * src, ggml_tensor * dst) {
    const auto & device = ggml_cuda_info().devices[ctx.device];
    if (!q2_0_mfma_enabled() || device.cc != GGML_CUDA_CC_CDNA2 || device.warp_size != 64 || !q2_0_weight_ok(w) ||
            src->type != GGML_TYPE_F32 || !ggml_is_matrix(src) || src->nb[0] != sizeof(float) ||
            src->nb[1] % sizeof(float) != 0 || src->ne[0] != w->ne[0] ||
            (src->ne[1] < 2 || src->ne[1] > 4) || dst->type != GGML_TYPE_F32 || !ggml_is_matrix(dst) ||
            !ggml_is_contiguous(dst) || dst->ne[0] != w->ne[1] || dst->ne[1] != src->ne[1]) {
        return false;
    }
    const bool tiled = q2_0_use_tile4(ctx, w);
    if (!tiled && src->ne[1] == 2) {
        return false;
    }
    const void * packed = q2_0_packed(ctx, w, tiled);
    if (!packed) {
        return false;
    }
    if (src->ne[1] == 2) {
        q2_0_launch_verify_mfma<2, true>(ctx, packed, src, dst);
    } else if (src->ne[1] == 3) {
        if (tiled) q2_0_launch_verify_mfma<3, true>(ctx, packed, src, dst);
        else       q2_0_launch_verify_mfma<3, false>(ctx, packed, src, dst);
    } else {
        if (tiled) q2_0_launch_verify_mfma<4, true>(ctx, packed, src, dst);
        else       q2_0_launch_verify_mfma<4, false>(ctx, packed, src, dst);
    }
    return true;
}

template<int C>
static void q2_0_fwht_project(ggml_backend_cuda_context & ctx, const ggml_tensor * input, const ggml_tensor * signs,
        ggml_tensor * const * outputs, const void * const * packed, int count) {
    const int width = input->ne[0];
    const int chunks = width / QK8_1;
    ggml_cuda_pool_alloc<char> scratch(ctx.pool());
    auto * words = reinterpret_cast<q2_0_words *>(scratch.alloc(chunks * (4 * sizeof(q2_0_words) + sizeof(float4))));
    auto * dy = reinterpret_cast<float4 *>(words + 4 * chunks);
    ggml_cuda_fwht_q2_0_q8(ctx, static_cast<const float *>(input->data), static_cast<const float *>(signs->data),
            width, C, reinterpret_cast<int8_t *>(words), reinterpret_cast<float *>(dy));
#if defined(__gfx90a__) || !defined(__HIP_DEVICE_COMPILE__)
    q2_0_projection proj[3] = {};
    int total_rows = 0;
    for (int i = 0; i < count; ++i) {
        const auto * codes = static_cast<const uint2 *>(packed[i]);
        const int rows = outputs[i]->ne[0];
        proj[i] = {codes, reinterpret_cast<const half *>(codes + (int64_t) rows * chunks),
                   static_cast<float *>(outputs[i]->data), rows};
        total_rows += rows;
    }
    if (count == 1) {
        q2_0_verify_mfma<C, true><<<total_rows / 8, 128, 0, ctx.stream()>>>(
                proj[0].codes, proj[0].scales, words, dy, proj[0].dst, chunks, total_rows);
    } else {
        q2_0_verify_group<C><<<total_rows / 8, 128, 0, ctx.stream()>>>(proj[0], proj[1], proj[2], words, dy, chunks);
    }
    CUDA_CHECK(cudaGetLastError());
#endif
}

bool ggml_cuda_mmvq_q2_0_fwht_group(ggml_backend_cuda_context & ctx, const ggml_tensor * input,
        const ggml_tensor * signs, ggml_tensor * const * outputs, int count) {
    GGML_ASSERT(count >= 1 && count <= 3);
    if (!q2_0_use_tile4(ctx, outputs[0]->src[0]) || ggml_cuda_info().devices[ctx.device].warp_size != 64) {
        return false;
    }
    const void * packed[3] = {};
    for (int i = 0; i < count; ++i) {
        const ggml_tensor * w = outputs[i]->src[0];
        if (!q2_0_weight_ok(w) || w->ne[0] != input->ne[0] || w->ne[1] != outputs[i]->ne[0]) {
            return false;
        }
        packed[i] = q2_0_packed(ctx, w, true);
        if (!packed[i]) {
            return false;
        }
    }
    switch (input->ne[1]) {
        case 2: q2_0_fwht_project<2>(ctx, input, signs, outputs, packed, count); break;
        case 3: q2_0_fwht_project<3>(ctx, input, signs, outputs, packed, count); break;
        case 4: q2_0_fwht_project<4>(ctx, input, signs, outputs, packed, count); break;
        default: return false;
    }
    return true;
}
#endif

bool ggml_cuda_mmvq_q2_0(
        ggml_backend_cuda_context & ctx, const ggml_tensor * w, const block_q8_1 * x,
        ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
#if defined(GGML_USE_HIP)
    static const bool enabled = [] {
        const char * e = getenv("GGML_MMVQ_Q2_0");
        return e == nullptr || e[0] != '0';
    }();
    if (!enabled || !GGML_CUDA_CC_IS_CDNA2(ggml_cuda_info().devices[ctx.device].cc) ||
            ggml_cuda_info().devices[ctx.device].warp_size != 64 || !q2_0_weight_ok(w) ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
            dst->ne[0] != w->ne[1] || !ggml_is_matrix(dst) || dst->ne[1] < 1 || dst->ne[1] > 4) {
        return false;
    }
    static const bool verify_enabled = [] {
        const char * e = getenv("GGML_MMVQ_Q2_0_VERIFY_BATCH");
        return !e || e[0] != '0';
    }();
    if (dst->ne[1] > 1 && (!verify_enabled || fusion)) {
        return false;
    }
    const ggml_tensor * gate = fusion ? fusion->gate : nullptr;
    const ggml_tensor * bias = fusion ? fusion->x_bias : nullptr;
    if (fusion && (fusion->gate_bias || fusion->x_scale || fusion->gate_scale || fusion->shared_up ||
            fusion->shared_gate || fusion->shared_dst)) {
        return false;
    }
    if (gate && (bias || fusion->glu_op != GGML_GLU_OP_SWIGLU ||
            !q2_0_weight_ok(gate) || !ggml_are_same_shape(w, gate))) {
        return false;
    }
    if (bias && (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) ||
            !ggml_are_same_shape(bias, dst))) {
        return false;
    }
    const bool tiled = q2_0_use_tile4(ctx, w);
    const void * packed = q2_0_packed(ctx, w, tiled);
    if (!packed) {
        return false;
    }
    const void * packed_gate = gate ? q2_0_packed(ctx, gate, tiled) : nullptr;
    if (gate && !packed_gate) {
        return false;
    }
    const int K = (int) w->ne[0], N = (int) w->ne[1];
    const auto * residual = bias ? (const float *) bias->data : nullptr;
    auto * out = (float *) dst->data;
    if (tiled) {
        q2_0_dispatch<true>(dst->ne[1], packed, packed_gate, x, out, residual, K, N, ctx.stream());
    } else {
        q2_0_dispatch<false>(dst->ne[1], packed, packed_gate, x, out, residual, K, N, ctx.stream());
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
#else
    GGML_UNUSED(ctx); GGML_UNUSED(w); GGML_UNUSED(x); GGML_UNUSED(dst); GGML_UNUSED(fusion);
    return false;
#endif
}
