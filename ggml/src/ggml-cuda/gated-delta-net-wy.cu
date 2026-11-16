#include "gated-delta-net-mma.cuh"

#include <climits>
#include <cstdint>
#include <cstdlib>

// The kernels address memory through buffer resources (the buffer builtins of recent clang). Without them the launch
// returns false and the one-pass kernel runs.
#if defined(GGML_USE_HIP) && defined(__has_builtin)
#if __has_builtin(__builtin_amdgcn_make_buffer_rsrc) && __has_builtin(__builtin_amdgcn_raw_buffer_load_b32) && \
    __has_builtin(__builtin_amdgcn_raw_buffer_load_b128) && __has_builtin(__builtin_amdgcn_raw_buffer_store_b32)
#define GDN_WY_AVAILABLE
#endif
#endif

#if defined(GDN_WY_AVAILABLE)

typedef float gdn_f4 __attribute__((ext_vector_type(4)));

using gdn_mfma_args = ggml_cuda_gdn_mma_args;

constexpr int GDN_D = 128;  // key / value head dim
constexpr int GDN_C = 16;   // tokens per chunk
constexpr int GDN_V = 64;   // value columns per block of the serial kernel

#define GDN_MFMA(a, b, c) __builtin_amdgcn_mfma_f32_16x16x4f32((a), (b), (c), 0, 0, 0)

static __device__ __forceinline__ gdn_f4 gdn_ld4(const float * p) {
    return *(const gdn_f4 *) p;
}

// MFMA 16x16x4 f32 layouts: A[i][k] in lane i + 16*k, B[k][j] in lane j + 16*k,
// C[i][j] in register i % 4 of lane j + 16*(i / 4) (see gated-delta-net-mfma.cu for the one-pass kernel).

// Two-pass chunked gated delta rule (GGML_CUDA_GDN_WY, default on; GGML_CUDA_GDN_WY=0 runs the one-pass kernel).
//
// The one-pass kernel spends most of each chunk on work that does not depend on the state: the gate tables,
// K K^T and Q K^T, the matrices A and P and the forward substitution, all between barriers on the serial
// chunk loop. Here a fully parallel pre-pass (one wave per sequence, key head and 16-token chunk, serving all
// value heads of that key head) computes, with G_t = exp(sum_{u<=t} g_u) inside the chunk (relative to the chunk
// start) and s_tj = sum_{j<u<=t} g_u,
//     T       = (I + A)^-1,  A[t][j] = beta_t exp(s_tj) (k_t . k_j)  (j <  t)    [16 x 16, by substitution]
//     P[t][j] = scale exp(s_tj) (q_t . k_j)                         (j <= t)
//     beta_t, beta_t G_t, scale G_t, G_C / G_t = exp(sum_{u>t} g_u), G_C
// Every exponent is summed over exactly the gates it spans (s_tj from u = j + 1 up; continued to the chunk end it
// gives G_C / G_j), never formed as a difference of two prefixes: such a difference loses the small gates that
// follow a large one and can round to a positive value (exp(+1000) for prefixes near -1e10). Gates below -1e30
// (-inf) count as -1e30, so for gates <= 0 every exponent is a sum of non-positive terms, every factor lies in
// [0, 1], nothing overflows, and g = -inf is an exact reset. Exponents are not clamped: positive gates are computed
// as exp(g) like in the other kernels, and NaN gates propagate.
// Against an fp64 reference (77 shapes and gate patterns up to 64K tokens, model dumps): output / final state
// nmse about 5e-14 / 1e-14 on random inputs (one-pass kernel
// 8e-14 / 4e-14, autoregressive kernel 1.3e-14 / 6e-15) and 5e-13 / 1.4e-12 on model data (one-pass 6.6e-13 /
// 1.7e-12); unchanged and finite when 1/16 of the gates are -30 .. -inf (one-pass: up to 0.26, NaN for -inf) or for
// a -1e4 / -1 mix (one-pass 4e-7); 1.2x the one-pass error (2.2e-12) only for nearly parallel keys with beta = 1 and
// no decay (I + A ill-conditioned: an explicit inverse applied by a product loses more than a direct solve).
// The serial kernel keeps the one-pass kernel's work split (one block
// per head and 64 value columns, the 128 x 16 state slice of each MFMA wave in registers) and per chunk only does
//     delta = T (beta v - beta G (K S)),   out = scale G (Q S) + P delta,   S = G_C S + K^T (delta G_C / G_t)
// The MFMA C layout of K S, delta and the scaled delta is exactly the B operand layout of the next product
// (depth index d = 4 * (lane / 16) + step), so only the chunk inputs go through LDS.
//
// The serial kernel is wave-specialised. With one wave per SIMD, the non-MFMA instructions of that wave are not
// hidden behind its MFMAs (gfx90a, measured), while a second wave on the same SIMD runs beside them almost for
// free. So the block has four MFMA waves (one per SIMD) and four staging waves: during chunk c the staging waves
// move chunk c + 1 (K and Q in the MFMA operand layout, K^T, the pre-pass record and v) from global memory into
// the other half of a double-buffered LDS area. The MFMA waves only read LDS, run 104 MFMAs per chunk (K S 32,
// Q S 32, state update 32, T and P 4 + 4) and store the output; there is one barrier per chunk, placed between
// Q S and the state update so that the next chunk's K / Q operands are read while the state update runs, and the
// remaining LDS reads are issued in small groups between the MFMAs (a wave can have at most 15 LDS instructions
// in flight, and a burst of reads at the top of the chunk stalls its MFMAs). Measured on one MI210 (4096
// tokens, 48 heads): 2.58 us per chunk (~3800 cycles, against 104 x 32 = 3328 cycles of MFMA issue), 3.42 us for
// the 4-wave version of the same kernel (all reads up front, the staging work in the MFMA waves).
// All global accesses are buffer loads / stores with loop-invariant lane offsets and a uniform per-chunk offset:
// no address arithmetic, no clamping (the hardware bounds check returns zeros past the last token and drops
// stores there, which needs token strides of at least one row: sq2 >= 128, sv2 >= 64), no branches in the loops.

constexpr int GDN_WY_REC = 592;   // floats per (sequence, head, chunk) record
constexpr int GDN_WY_T   = 0;     // T[t*16 + j]
constexpr int GDN_WY_P   = 256;   // P[t*16 + j]
constexpr int GDN_WY_BT  = 512;   // beta_t
constexpr int GDN_WY_BG  = 528;   // beta_t G_t
constexpr int GDN_WY_GS  = 544;   // scale G_t
constexpr int GDN_WY_DE  = 560;   // G_C / G_t
constexpr int GDN_WY_DC  = 576;   // G_C, 16 copies

// buffer resource over [p, p + bytes); accesses past the end read 0 / are dropped
static __device__ __forceinline__ __amdgpu_buffer_rsrc_t gdn_rsrc(const void * p, const int bytes) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void *>(p), 0, bytes, 0x00020000);
}
static __device__ __forceinline__ gdn_f4 gdn_bld4(const __amdgpu_buffer_rsrc_t r, const int voff, const int soff) {
    return __builtin_bit_cast(gdn_f4, __builtin_amdgcn_raw_buffer_load_b128(r, voff, soff, 0));
}
static __device__ __forceinline__ float gdn_bld1(const __amdgpu_buffer_rsrc_t r, const int voff, const int soff) {
    return __builtin_bit_cast(float, __builtin_amdgcn_raw_buffer_load_b32(r, voff, soff, 0));
}

// grid (ceil(n_chunks / 4), H_k, n_seqs), one wave per chunk; requires H % H_k == 0 and H / H_k <= 4
static __global__ void __launch_bounds__(256) gdn_wy_prep_kernel(
        const gdn_mfma_args a, float * __restrict__ rec, const int n_chunks) {
#if defined(CDNA)
    constexpr int D = GDN_D, C = GDN_C;
    struct wave_lds {
        float kk[C * C];      // K K^T [t][j]
        float qk[C * C];      // Q K^T [t][j]
        float g[4][C];        // gates per value head
        float b[4][C];        // beta per value head
        float A[4][C * C];    // A per value head
    };
    __shared__ wave_lds s4[4];

    const int w   = threadIdx.x >> 6;
    const int l   = threadIdx.x & 63;
    const int l16 = l & 15;
    const int lk  = l >> 4;
    wave_lds & s = s4[w];

    const int     R     = (int) (a.H / a.H_k);       // value heads per key head
    const bool    valid = (int) blockIdx.x * 4 + w < n_chunks;
    const int     c     = valid ? blockIdx.x * 4 + w : 0;
    const int     kh    = blockIdx.y;
    const int64_t seq   = blockIdx.z;
    const int     n     = (int) a.n_tokens;
    const int     t0    = c * C;
    const bool    hok   = lk < R;
    const int     hv    = kh + (hok ? lk : 0) * (int) a.H_k;   // value head of lane group lk

    const int  sq2 = (int) a.sq2, sb2 = (int) a.sb2;
    const auto rk  = gdn_rsrc(a.k + (seq / a.rq3) * a.sq3 + kh * a.sq1, ((n - 1) * sq2 + D) * 4);
    const auto rq  = gdn_rsrc(a.q + (seq / a.rq3) * a.sq3 + kh * a.sq1, ((n - 1) * sq2 + D) * 4);
    const auto rg  = gdn_rsrc(a.g    + seq * a.sb3 + hv * a.sb1, ((n - 1) * sb2 + 1) * 4);
    const auto rb  = gdn_rsrc(a.beta + seq * a.sb3 + hv * a.sb1, ((n - 1) * sb2 + 1) * 4);

    // row l16 of the K / Q chunk, key rows 16*it + 4*lk + m (zero past the last token)
    gdn_f4 k4[D / 16], q4[D / 16];
#pragma unroll
    for (int it = 0; it < D / 16; ++it) {
        k4[it] = gdn_bld4(rk, (l16 * sq2 + 16 * it + 4 * lk) * 4, t0 * sq2 * 4);
        q4[it] = gdn_bld4(rq, (l16 * sq2 + 16 * it + 4 * lk) * 4, t0 * sq2 * 4);
    }
    // gates of token t0 + l16 for value head hv (g = beta = 0 past the last token); gates below -1e30 (-inf: a full
    // reset) count as -1e30, so that sums of up to 16 gates stay finite and no -inf - -inf = NaN can form
    float gl = gdn_bld1(rg, l16 * sb2 * 4, t0 * sb2 * 4);
    gl = gl < -1e30f ? -1e30f : gl;
    const float bl = gdn_bld1(rb, l16 * sb2 * 4, t0 * sb2 * 4);
    float p = gl;   // inclusive prefix p_t = sum_{u <= t} g_u
#pragma unroll
    for (int off = 1; off < C; off <<= 1) {
        const float x = __shfl_up(p, off, C);
        if (l16 >= off) {
            p += x;
        }
    }
    const float tot = __shfl(p, C - 1, C);

    gdn_f4 gk = { 0.0f, 0.0f, 0.0f, 0.0f }, gq = gk;
#pragma unroll
    for (int it = 0; it < D / 16; ++it) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            gk = GDN_MFMA(k4[it][m], k4[it][m], gk);
            gq = GDN_MFMA(q4[it][m], k4[it][m], gq);
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {   // element [t = 4*lk + r][j = l16]
        s.kk[(4 * lk + r) * C + l16] = gk[r];
        s.qk[(4 * lk + r) * C + l16] = gq[r];
    }
    s.g[lk][l16] = gl;
    s.b[lk][l16] = bl;

    // exponents are sums of exactly the gates they span (<= 0 for gates <= 0, in any summation order)
    float * rp = rec + ((seq * a.H + hv) * n_chunks + c) * GDN_WY_REC;
    if (valid && hok) {
        const float G = expf(p);
        rp[GDN_WY_BT + l16] = bl;
        rp[GDN_WY_BG + l16] = bl * G;
        rp[GDN_WY_GS + l16] = a.scale * G;
        rp[GDN_WY_DC + l16] = expf(tot);
    }
    __syncthreads();

    // lane group lk, column j = l16: A and P of value head hv
    {
        float seg = 0.0f;   // log G_t / G_j = sum_{j < u <= t} g_u, summed from u = j + 1 up (not p_t - p_j)
#pragma unroll
        for (int t = 0; t < C; ++t) {
            if (t > l16) {
                seg += s.g[lk][t];
            }
            const float e = expf(seg);   // only used for j <= t, where it is G_t / G_j
            s.A[lk][t * C + l16] = l16 < t ? s.b[lk][t] * e * s.kk[t * C + l16] : 0.0f;
            if (valid && hok) {
                rp[GDN_WY_P + t * C + l16] = l16 <= t ? a.scale * e * s.qk[t * C + l16] : 0.0f;
            }
        }
        if (valid && hok) {   // seg is now the suffix sum_{u > j} g_u: G_C / G_j (not tot - p_j)
            rp[GDN_WY_DE + l16] = expf(seg);
        }
    }
    __syncthreads();

    // column j = l16 of T = (I + A)^-1 by forward substitution, in fp64 (about 2% of this kernel's time; it brings the
    // error on model data below that of the one-pass kernel, which solves (I + A) delta = rhs directly)
    {
        double x[C];
#pragma unroll
        for (int t = 0; t < C; ++t) {
            x[t] = t == l16 ? 1.0 : 0.0;
        }
#pragma unroll
        for (int t = 1; t < C; ++t) {
            double acc = x[t];
#pragma unroll
            for (int jm = 0; jm * 4 < t; ++jm) {
                const gdn_f4 a4 = gdn_ld4(&s.A[lk][t * C + 4 * jm]);
#pragma unroll
                for (int cc = 0; cc < 4; ++cc) {
                    if (4 * jm + cc < t) {
                        acc -= (double) a4[cc] * x[4 * jm + cc];
                    }
                }
            }
            x[t] = acc;
        }
        if (valid && hok) {
#pragma unroll
            for (int t = 0; t < C; ++t) {
                rp[GDN_WY_T + t * C + l16] = (float) x[t];
            }
        }
    }
#else
    GGML_UNUSED(a); GGML_UNUSED(rec); GGML_UNUSED(n_chunks);
    NO_DEVICE_CODE;
#endif // defined(CDNA)
}

struct gdn_wy_lds {
    alignas(16) float Kp[2][GDN_C * GDN_D];   // float4 (pq*16 + (t ^ (pq & 7))) = K[t][4*pq .. 4*pq + 3]
    alignas(16) float Qp[2][GDN_C * GDN_D];   // same layout
    alignas(16) float KT[2][GDN_C * GDN_D];   // KT[((it*4 + k)*16 + i)*4 + kk] = K[4*k + kk][16*it + i]
    alignas(16) float R[2][GDN_WY_REC];       // pre-pass record of the chunk
    alignas(16) float Vt[2][GDN_V * GDN_C];   // Vt[col*16 + t] = v[t0 + t][v0 + col]
};

// 512 threads: waves 0..3 MFMA, waves 4..7 staging (see above).
// requires 4 * (n_tokens + 16) * max(sq2, sv2, H * 128) < 2^31, sq2 >= 128, sv2 >= 64 and H % H_k == 0 (checked by the caller)
static __global__ void __launch_bounds__(512) gdn_wy_kernel(
        const gdn_mfma_args a, const float * __restrict__ rec, const int n_chunks) {
#if defined(CDNA)
    constexpr int D = GDN_D, C = GDN_C, V = GDN_V;
    __shared__ gdn_wy_lds s;

    const int tid = threadIdx.x;
    const int w   = __builtin_amdgcn_readfirstlane(tid >> 6);   // wave, uniform
    const int l   = tid & 63;
    const int l16 = l & 15;
    const int lk  = l >> 4;

    const int     slice = blockIdx.x % (D / V);
    const int     h     = (blockIdx.x / (D / V)) % a.H;
    const int64_t seq   = blockIdx.x / ((D / V) * a.H);
    const int     v0    = slice * V;
    const int     n     = (int) a.n_tokens;

    if (w < 4) {
        // ---------------- MFMA waves: value column vw, state rows 16*it + 4*lk + r in st[it][r]
        const int     vw   = 16 * w + l16;
        const int64_t soff = (seq * a.H + h) * D * D;
        const int     so2  = (int) (a.H * D);
        const auto    ro   = gdn_rsrc(a.dst + (seq * n * a.H + h) * D + v0, ((n - 1) * so2 + V) * 4);

        gdn_f4 st[D / 16];
#pragma unroll
        for (int it = 0; it < D / 16; ++it) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                st[it][r] = a.state[soff + (int64_t) (v0 + vw) * D + it * 16 + 4 * lk + r];
            }
        }
        int oo[4];   // output byte offsets of rows 4*lk + r
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            oo[r] = ((4 * lk + r) * so2 + vw) * 4;
        }
        // K / Q operand reads (lane l: float4 l of a 256-float row, permuted inside 8-lane groups, conflict free)
        const int lde = (16 * lk + (l16 ^ lk)) * 16;          // bytes, even it
        const int ldo = (16 * lk + (l16 ^ (4 + lk))) * 16;    // odd it
        const int orT = l16 * C + 4 * lk;

        gdn_f4 ka[D / 16], qa[D / 16];   // K and Q of the current chunk, read during the previous state update
        const auto read_kq = [&](const int pb, const int it) {
            ka[it] = *(const gdn_f4 *) ((const char *) s.Kp[pb] + 1024 * it + (it & 1 ? ldo : lde));
            qa[it] = *(const gdn_f4 *) ((const char *) s.Qp[pb] + 1024 * it + (it & 1 ? ldo : lde));
        };
        __syncthreads();   // chunk 0 is in buffer 0
#pragma unroll
        for (int it = 0; it < D / 16; ++it) {
            read_kq(0, it);
        }

        for (int c = 0; c < n_chunks; ++c) {
            const int     pb = c & 1;
            const float * KT = s.KT[pb];
            const float * Rp = s.R[pb];

            // K S (on the serial path) in four groups of 8 MFMAs, each with a few LDS reads of operands used later
            gdn_f4 ta[D / 16];
            gdn_f4 T, bt, bg, v4, P, gs, de;
            float  dec;
            gdn_f4 ks0 = { 0.0f, 0.0f, 0.0f, 0.0f }, ks1 = ks0;
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                if (g == 0) {
                    T  = gdn_ld4(Rp + GDN_WY_T + orT);
                    bt = gdn_ld4(Rp + GDN_WY_BT + 4 * lk);
                    bg = gdn_ld4(Rp + GDN_WY_BG + 4 * lk);
                    v4 = gdn_ld4(s.Vt[pb] + vw * 16 + 4 * lk);
                } else if (g == 1) {
                    P   = gdn_ld4(Rp + GDN_WY_P + orT);
                    gs  = gdn_ld4(Rp + GDN_WY_GS + 4 * lk);
                    de  = gdn_ld4(Rp + GDN_WY_DE + 4 * lk);
                    dec = Rp[GDN_WY_DC];
                } else {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        ta[4 * (g - 2) + i] = gdn_ld4(KT + 256 * (4 * (g - 2) + i) + 4 * l);
                    }
                }
#pragma unroll
                for (int it = 2 * g; it < 2 * g + 2; ++it) {
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        if (it & 1) {
                            ks1 = GDN_MFMA(ka[it][m], st[it][m], ks1);
                        } else {
                            ks0 = GDN_MFMA(ka[it][m], st[it][m], ks0);
                        }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);   // keep the reads of the next group behind these MFMAs
            }

            // delta = T (beta v - beta G (K S)), interleaved with Q S (off the serial path)
            const gdn_f4 rhs = bt * v4 - bg * (ks0 + ks1);
            gdn_f4 dl = { 0.0f, 0.0f, 0.0f, 0.0f }, qs0 = dl, qs1 = dl;
#pragma unroll
            for (int it = 0; it < D / 16; ++it) {
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    if (it & 1) {
                        qs1 = GDN_MFMA(qa[it][m], st[it][m], qs1);
                    } else {
                        qs0 = GDN_MFMA(qa[it][m], st[it][m], qs0);
                    }
                }
                if (it < 4) {
                    dl = GDN_MFMA(T[it], rhs[it], dl);
                }
            }

            // All reads of buffer pb are done (the barrier waits for them); chunk c + 1 is in buffer pb ^ 1, and
            // the staging waves refill buffer pb with chunk c + 2 after this barrier.
            __syncthreads();

            // S = G_C S + K^T (delta G_C / G_t), interleaved with out = scale G (Q S) + P delta and with the
            // reads of the next chunk's K and Q
            const gdn_f4 ds = dl * de;
            gdn_f4       o  = gs * (qs0 + qs1);
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                read_kq(pb ^ 1, 2 * g);
                read_kq(pb ^ 1, 2 * g + 1);
#pragma unroll
                for (int it = 2 * g; it < 2 * g + 2; ++it) {
                    st[it] = st[it] * dec;
#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        st[it] = GDN_MFMA(ta[it][kk], ds[kk], st[it]);
                    }
                    if (it < 4) {
                        o = GDN_MFMA(P[it], dl[it], o);
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
            }

            // rows past the last token fall outside the buffer and are dropped
            const int so = c * C * so2 * 4;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const float x = o[r];   // (a bit_cast of the vector element itself reads element 0)
                __builtin_amdgcn_raw_buffer_store_b32(__builtin_bit_cast(unsigned int, x), ro, oo[r], so, 0);
            }
        }

#pragma unroll
        for (int it = 0; it < D / 16; ++it) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                a.state_out[soff + (int64_t) (v0 + vw) * D + it * 16 + 4 * lk + r] = st[it][r];
            }
        }
    } else {
        // ---------------- staging waves (thread ht): loads of chunk c + 2 are in flight while chunk c + 1 is stored
        const int  ht  = tid - 256;
        const int  sq2 = (int) a.sq2, sv2 = (int) a.sv2;
        const auto rk  = gdn_rsrc(a.k + (seq / a.rq3) * a.sq3 + (h % a.H_k) * a.sq1, ((n - 1) * sq2 + D) * 4);
        const auto rq  = gdn_rsrc(a.q + (seq / a.rq3) * a.sq3 + (h % a.H_k) * a.sq1, ((n - 1) * sq2 + D) * 4);
        const auto rv  = gdn_rsrc(a.v + seq * a.sv3 + h * a.sv1 + v0, ((n - 1) * sv2 + V) * 4);
        const auto rr  = gdn_rsrc(rec + (seq * a.H + h) * (int64_t) n_chunks * GDN_WY_REC, n_chunks * GDN_WY_REC * 4);
        // K / Q: key-row quad pq of tokens pt and pt + 8 (each row read as 512 contiguous bytes); the XOR swizzle
        // keeps these LDS writes and the operand reads free of bank conflicts
        const int pq   = ht & 31;
        const int pt   = ht >> 5;
        const int okq0 = (pt * sq2 + 4 * pq) * 4;
        const int okq1 = okq0 + 8 * sq2 * 4;
        const int ldsw = (pq * 16 + (pt ^ (pq & 7))) * 16;   // bytes; + 128 for token pt + 8
        // K^T float4s ht and ht + 256 (it = ht / 64 and + 4, k = (ht / 16) % 4, i = ht % 16), four scalar loads each
        const int okt0 = ((4 * ((ht >> 4) & 3)) * sq2 + 16 * (ht >> 6) + (ht & 15)) * 4;
        const int okt1 = okt0 + 16 * 4 * 4;
        // v: float4 of token vt, columns vc .. vc + 3, stored transposed
        const int  vt  = ht >> 4;
        const int  vc  = 4 * (ht & 15);
        const int  ovv = (vt * sv2 + vc) * 4;
        // record: float4 ht
        const bool hr  = ht < GDN_WY_REC / 4;

        gdn_f4 k[2], q[2], kt[2], vv, rr4;
        const auto fetch = [&](int c) {
            c = min(c, n_chunks - 1);   // past the end: a harmless repeat
            const int skq = c * C * sq2 * 4;
            k[0] = gdn_bld4(rk, okq0, skq);
            k[1] = gdn_bld4(rk, okq1, skq);
            q[0] = gdn_bld4(rq, okq0, skq);
            q[1] = gdn_bld4(rq, okq1, skq);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                kt[0][kk] = gdn_bld1(rk, okt0 + kk * sq2 * 4, skq);
                kt[1][kk] = gdn_bld1(rk, okt1 + kk * sq2 * 4, skq);
            }
            vv  = gdn_bld4(rv, ovv, c * C * sv2 * 4);
            rr4 = gdn_bld4(rr, hr ? 16 * ht : 0, c * GDN_WY_REC * 4);
        };
        const auto publish = [&](const int pb) {
            *(gdn_f4 *) ((char *) s.Kp[pb] + ldsw)       = k[0];
            *(gdn_f4 *) ((char *) s.Kp[pb] + ldsw + 128) = k[1];
            *(gdn_f4 *) ((char *) s.Qp[pb] + ldsw)       = q[0];
            *(gdn_f4 *) ((char *) s.Qp[pb] + ldsw + 128) = q[1];
            *(gdn_f4 *) (s.KT[pb] + 4 * ht)        = kt[0];
            *(gdn_f4 *) (s.KT[pb] + 4 * ht + 1024) = kt[1];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                s.Vt[pb][(vc + j) * 16 + vt] = vv[j];
            }
            if (hr) {
                *(gdn_f4 *) (s.R[pb] + 4 * ht) = rr4;
            }
        };

        fetch(0);
        publish(0);
        fetch(1);
        __syncthreads();
        for (int c = 0; c < n_chunks; ++c) {
            publish((c + 1) & 1);
            fetch(c + 2);
            __syncthreads();
        }
    }
#else
    GGML_UNUSED(a); GGML_UNUSED(rec); GGML_UNUSED(n_chunks);
    NO_DEVICE_CODE;
#endif // defined(CDNA)
}

#endif // defined(GDN_WY_AVAILABLE)

// Two-pass chunked gated delta rule on CDNA2. Returns false when the problem is not eligible or the compiler lacks
// the buffer builtins; the caller then tries the one-pass MFMA kernel and the autoregressive kernel.
// GGML_CUDA_GDN_WY=0 (or GGML_CUDA_GDN_MFMA=0) disables it.
bool ggml_cuda_gdn_wy_launch(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_mma_args & a, cudaStream_t stream) {
#if defined(GDN_WY_AVAILABLE)
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_GDN_WY");
        const char * m = getenv("GGML_CUDA_GDN_MFMA");
        return (e == nullptr || atoi(e) != 0) && (m == nullptr || atoi(m) != 0);
    }();
    const auto & info = ggml_cuda_info().devices[ctx.device];
    if (!enabled || !GGML_CUDA_CC_IS_CDNA2(info.cc) || info.smpbo < sizeof(gdn_wy_lds)) {
        return false;
    }
    // scalar gate, 128-dim heads (a.eligible), a single output state slot, enough tokens to chunk
    if (!a.eligible || a.state_out == nullptr || a.n_tokens < 2 * GDN_C || a.rq3 <= 0 || a.H <= 0 || a.H_k <= 0 ||
        a.n_seqs <= 0 || a.n_seqs % a.rq3 != 0) {
        return false;
    }
    // the pre-pass serves all value heads of a key head from one wave
    if (a.H % a.H_k != 0 || a.H / a.H_k > 4) {
        return false;
    }
    // padding tokens of a partial last chunk must fall outside the buffer ranges (zeros): rows may not overlap
    if ((((uintptr_t) a.q | (uintptr_t) a.k | (uintptr_t) a.v | (uintptr_t) a.dst) & 3) != 0 ||
        a.sq1 % 4 != 0 || a.sq2 % 4 != 0 || a.sq3 % 4 != 0 || a.sv1 % 4 != 0 || a.sv2 % 4 != 0 || a.sv3 % 4 != 0 ||
        a.sq2 < GDN_D || a.sv2 < GDN_V || a.sb2 <= 0) {
        return false;
    }
    // buffer offsets and sizes are 32-bit byte counts inside the kernels
    const int64_t n_chunks   = (a.n_tokens + GDN_C - 1) / GDN_C;
    const int64_t max_stride = std::max(std::max(a.sq2, a.sv2), std::max(a.sb2, a.H * GDN_D));
    if ((a.n_tokens + GDN_C) * max_stride * 4 >= INT_MAX || n_chunks * GDN_WY_REC * 4 >= INT_MAX ||
        a.H * a.n_seqs >= INT_MAX / (GDN_D / GDN_V) || a.H_k > 65535 || a.n_seqs > 65535) {
        return false;
    }

    ggml_cuda_pool_alloc<float> rec(ctx.pool(), a.n_seqs * a.H * n_chunks * GDN_WY_REC);

    const dim3 prep_grid((unsigned) ((n_chunks + 3) / 4), (unsigned) a.H_k, (unsigned) a.n_seqs);
    gdn_wy_prep_kernel<<<prep_grid, 256, 0, stream>>>(a, rec.get(), (int) n_chunks);
    CUDA_CHECK(cudaGetLastError());
    gdn_wy_kernel<<<a.H * a.n_seqs * (GDN_D / GDN_V), 512, 0, stream>>>(a, rec.get(), (int) n_chunks);
    CUDA_CHECK(cudaGetLastError());
    return true;
#else
    GGML_UNUSED_VARS(ctx, a, stream);
    return false;
#endif // defined(GDN_WY_AVAILABLE)
}
