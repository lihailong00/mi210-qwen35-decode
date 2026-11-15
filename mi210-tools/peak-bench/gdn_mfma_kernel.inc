// Chunked gated delta rule for CDNA (wave64) using fp32 MFMA. Shared by the harness and ggml.
//
// Per head and value column v the recurrence over tokens t is
//     delta_t = beta_t * (v_t - g_t * (s_{t-1} . k_t)),   s_t = g_t * s_{t-1} + k_t * delta_t,   out_t = s_t . q_t
// With G_t = prod_{u<=t} g_u inside a chunk of C tokens this becomes
//     (I + A) delta = beta * (v - G * (K S0)),     A[t][j] = beta_t * (G_t/G_j) * (k_t . k_j)   (j <  t)
//     out          = G * (Q S0) + P delta,         P[t][j] =          (G_t/G_j) * (q_t . k_j)   (j <= t)
//     S_end        = G_C * S0 + K^T (delta * G_C/G_t)
// One block owns GDN_V = 64 value columns of one head; each of its four waves owns 16 of them and keeps
// that 128 x 16 slice of the state in registers for the whole call.
//
// MFMA 16x16x4 f32 layouts: A[i][k] in lane i + 16*k, B[k][j] in lane j + 16*k,
// C[i][j] in register i % 4 of lane j + 16*(i / 4).
// The depth index of a product over the 128 key rows is split as i = 16*it + 4*k + m, so the B operand of
// step (it, m) is exactly register m of state tile it: the state never goes through LDS.
//
// A wave executes roughly one ordinary instruction per 10 cycles on this hardware (an MFMA takes ~35),
// so the kernel is written to minimise the instruction count on the critical path: the LDS copies of K
// and Q are "lane-major" (one 16-byte read feeds four MFMAs) and double-buffered by chunk parity (one
// barrier fewer), global pointers are bumped per chunk instead of recomputed, all loops over register
// arrays are fully unrolled (a real loop would index registers dynamically), and the masking of
// out-of-range tokens only exists on the path taken by a partial last chunk.
// Build with -mllvm -amdgpu-mfma-vgpr-form so the state does not shuttle between VGPRs and AccVGPRs.

typedef float gdn_f4 __attribute__((ext_vector_type(4)));

struct gdn_mfma_args {
    const float * q, * k, * v, * g, * beta, * state;
    float * dst, * state_out;
    int64_t H, H_k, n_tokens, n_seqs, rq3;
    int64_t sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3;
    float scale;
};

constexpr int GDN_D = 128;  // key / value head dim
constexpr int GDN_C = 16;   // tokens per chunk
constexpr int GDN_V = 64;   // value columns per block

struct gdn_mfma_gates {
    alignas(16) float prefix[GDN_C];
    alignas(16) float beta[GDN_C];
    alignas(16) float decay[GDN_C];        // G_t
    alignas(16) float dendp[GDN_C];        // dendp[(t%4)*4 + t/4] = G_C / G_t
    alignas(16) float idecay[GDN_C];       // 1 / G_t
    alignas(16) float tot;                 // log G_C
};

struct gdn_mfma_lds {
    alignas(16) float Kp[2][GDN_C * GDN_D];   // Kp[(t*4 + k)*32 + it*4 + m] = K[t][16*it + 4*k + m]
    alignas(16) float Qp[2][GDN_C * GDN_D];   // same layout
    alignas(16) float KT[2][GDN_C * GDN_D];   // KT[(i*4 + t%4)*4 + t/4]     = K[t][i]
    alignas(16) float R[GDN_C * GDN_V];       // R[v*16 + t]                 = rhs[t][v]
    alignas(16) float Dp[GDN_C * GDN_V];      // Dp[v*16 + (t%4)*4 + t/4]    = delta[t][v]
    alignas(16) float DPp[GDN_C * GDN_V];     // same layout, delta[t][v] * G_C/G_t
    alignas(16) float A[GDN_C * GDN_C];       // A[t*16 + j]
    alignas(16) float Pp[GDN_C * GDN_C];      // Pp[(t*4 + j%4)*4 + j/4]     = P[t][j] * scale
    gdn_mfma_gates    gt[2];
};
static_assert(sizeof(gdn_mfma_lds) <= 64 * 1024, "LDS budget");

#define GDN_MFMA(a, b, c) __builtin_amdgcn_mfma_f32_16x16x4f32((a), (b), (c), 0, 0, 0)

static __device__ __forceinline__ gdn_f4 gdn_ld4(const float * p) {
    return *(const gdn_f4 *) p;
}

// requires n_tokens * max(sq2, sv2, H * 128, sb2) < 2^31 (checked by the caller)
static __global__ void __launch_bounds__(256) gdn_mfma_kernel(const gdn_mfma_args a) {
    constexpr int D = GDN_D, C = GDN_C, V = GDN_V;
    __shared__ gdn_mfma_lds s;

    const int tid = threadIdx.x;
    const int w   = tid >> 6;   // wave
    const int l   = tid & 63;   // lane
    const int l16 = l & 15;
    const int lk  = l >> 4;

    const int     slice = blockIdx.x % (D / V);
    const int     h     = (blockIdx.x / (D / V)) % a.H;
    const int64_t seq   = blockIdx.x / ((D / V) * a.H);
    const int     v0    = slice * V;
    const int     vw    = 16 * w + l16;   // this lane's value column inside the block (MFMA j = l16)
    const int     n     = (int) a.n_tokens;

    const int64_t soff = (seq * a.H + h) * D * D;
    const float * kq_k = a.k + (seq / a.rq3) * a.sq3 + (h % a.H_k) * a.sq1;
    const float * kq_q = a.q + (seq / a.rq3) * a.sq3 + (h % a.H_k) * a.sq1;
    const float * v_in = a.v + seq * a.sv3 + h * a.sv1 + v0 + vw;
    const float * g_in = a.g + seq * a.sb3 + h * a.sb1;
    const float * b_in = a.beta + seq * a.sb3 + h * a.sb1;
    float *       out  = a.dst + (seq * n * a.H + h) * D + v0 + vw;

    const int64_t sq2 = a.sq2, sv2 = a.sv2, sb2 = a.sb2, so2 = a.H * D;

    // state rows 16*it + 4*lk + r of column vw live in st[it][r]
    gdn_f4 st[D / 16];
#pragma unroll
    for (int it = 0; it < D / 16; ++it) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            st[it][r] = a.state[soff + (int64_t) (v0 + vw) * D + it * 16 + 4 * lk + r];
        }
    }

    // per-thread positions inside a chunk
    const int pt[2] = { tid >> 5, (tid + 256) >> 5 };   // token of the two K/Q float4s this thread moves
    const int pm    = tid & 31;                         // key rows 4*pm .. 4*pm + 3: it = pm / 4, k = pm % 4
    const int kpo   = (pm & 3) * 32 + (pm >> 2) * 4;

    // Chunk inputs are fetched one chunk ahead so the global-load latency overlaps the MFMA work.
    gdn_f4 kn[2], qn[2];
    const auto fetch = [&](const int t0, const bool clamp) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int tt = clamp ? min(t0 + pt[p], n - 1) : t0 + pt[p];
            kn[p] = gdn_ld4(kq_k + tt * sq2 + 4 * pm);
            qn[p] = gdn_ld4(kq_q + tt * sq2 + 4 * pm);
        }
    };
    // write the fetched chunk into LDS buffer pb
    const auto publish = [&](const int t0, const int pb, const bool mask) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            gdn_f4 k4 = kn[p], q4 = qn[p];
            if (mask) {
                const float msk = t0 + pt[p] < n ? 1.0f : 0.0f;
                k4 = k4 * msk;
                q4 = q4 * msk;
            }
            *(gdn_f4 *) (s.Kp[pb] + pt[p] * 128 + kpo) = k4;
            *(gdn_f4 *) (s.Qp[pb] + pt[p] * 128 + kpo) = q4;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                s.KT[pb][((4 * pm + c) * 4 + (pt[p] & 3)) * 4 + (pt[p] >> 2)] = k4[c];
            }
        }
    };
    // wave 1, lanes 0..15: gate tables of the chunk starting at t0 into buffer pb
    const auto gates = [&](const int t0, const int pb, const bool mask) {
        if (w == 1 && l < C) {
            float acc = 0.0f, tot = 0.0f;
#pragma unroll
            for (int u = 0; u < C; ++u) {
                float gu;
                if (mask) {
                    gu = g_in[min(t0 + u, n - 1) * sb2];
                    gu = t0 + u < n ? gu : 0.0f;
                } else {
                    gu = g_in[(t0 + u) * sb2];
                }
                tot += gu;
                if (u <= l) {
                    acc += gu;
                }
            }
            float bt;
            if (mask) {
                bt = b_in[min(t0 + l, n - 1) * sb2];
                bt = t0 + l < n ? bt : 0.0f;
            } else {
                bt = b_in[(t0 + l) * sb2];
            }
            const float dec = expf(acc);
            s.gt[pb].prefix[l] = acc;
            s.gt[pb].decay[l]  = dec;
            s.gt[pb].dendp[(l & 3) * 4 + (l >> 2)] = expf(tot - acc);
            s.gt[pb].idecay[l] = 1.0f / dec;
            s.gt[pb].beta[l]   = bt;
            if (l == 0) {
                s.gt[pb].tot = tot;
            }
        }
    };
    {
        const bool part = C > n;
        fetch(0, part);
        publish(0, 0, part);
        gates(0, 0, part);
    }
    __syncthreads();

    const int abase = (l16 * 4 + lk) * 32;
    int       pb    = 0;
    for (int t0 = 0; t0 < n; t0 += C, pb ^= 1) {
        const gdn_mfma_gates & gt = s.gt[pb];
        const float * Kp = s.Kp[pb];
        const float * Qp = s.Qp[pb];
        const bool full      = t0 + C <= n;       // this chunk has no padding tokens
        const bool more      = t0 + C < n;
        const bool next_part = t0 + 2 * C > n;    // the next chunk is partial

        // ---- issue the loads needed later in this chunk and for the next chunk
        float vv[4];
        {
            const float * vp = v_in + (int64_t) (t0 + 4 * lk) * sv2;
            if (full) {
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    vv[r] = vp[r * sv2];
                }
            } else {
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int tt = t0 + 4 * lk + r;
                    vv[r] = v_in[min(tt, n - 1) * sv2];
                    vv[r] = tt < n ? vv[r] : 0.0f;
                }
            }
        }
        if (more) {
            fetch(t0 + C, next_part);
        }

        // ---- phase 1: every wave K S0 for its columns; waves 0/1 also Q S0, wave 2 K K^T, wave 3 Q K^T
        gdn_f4 ks = { 0.0f, 0.0f, 0.0f, 0.0f }, qs = ks, gram = ks;
#pragma unroll
        for (int it = 0; it < D / 16; ++it) {
            const gdn_f4 k4 = gdn_ld4(Kp + abase + 4 * it);
            gdn_f4       q4 = k4;
            if (w != 2) {
                q4 = gdn_ld4(Qp + abase + 4 * it);
            }
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                ks = GDN_MFMA(k4[m], st[it][m], ks);
                if (w < 2) {
                    qs = GDN_MFMA(q4[m], st[it][m], qs);
                } else {
                    gram = GDN_MFMA(q4[m], k4[m], gram);   // wave 2: q4 == k4
                }
            }
        }

        // ---- right-hand side (all waves), A (wave 2) and P (wave 3)
        const gdn_f4 dc4 = gdn_ld4(gt.decay + 4 * lk);
        const gdn_f4 bt4 = gdn_ld4(gt.beta + 4 * lk);
        {
            gdn_f4 v4 = { vv[0], vv[1], vv[2], vv[3] };
            *(gdn_f4 *) (s.R + vw * 16 + 4 * lk) = bt4 * (v4 - dc4 * ks);
        }
        if (w >= 2) {
            // G_t / G_j: a product with the reciprocal unless the chunk decays far enough to underflow
            gdn_f4 e4;
            if (gt.tot > -80.0f) {
                e4 = dc4 * gt.idecay[l16];
            } else {
                const gdn_f4 pf4 = gdn_ld4(gt.prefix + 4 * lk);
                const float  pfj = gt.prefix[l16];
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    e4[r] = expf(pf4[r] - pfj);
                }
            }
            if (w == 2) {
                e4 = e4 * bt4 * gram;
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int t = 4 * lk + r;
                    s.A[t * C + l16] = l16 < t ? e4[r] : 0.0f;
                }
            } else {
                e4 = e4 * gram * a.scale;
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int t = 4 * lk + r;
                    s.Pp[(t * 4 + (l16 & 3)) * 4 + (l16 >> 2)] = l16 <= t ? e4[r] : 0.0f;
                }
            }
        }
        __syncthreads();

        // ---- phase 2: wave 0 solves (I + A) delta = rhs by forward substitution, one lane per value
        //      column of the block; wave 1 builds the next gate tables; waves 2/3 compute their Q S0
        if (w == 0) {
            float d[C];
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const gdn_f4 r4 = gdn_ld4(s.R + l * 16 + 4 * m);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    d[4 * m + c] = r4[c];
                }
            }
#pragma unroll
            for (int t = 1; t < C; ++t) {
                float x = d[t];
#pragma unroll
                for (int jm = 0; jm * 4 < t; ++jm) {
                    const gdn_f4 a4 = gdn_ld4(s.A + t * C + 4 * jm);
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        if (4 * jm + c < t) {
                            x -= a4[c] * d[4 * jm + c];
                        }
                    }
                }
                d[t] = x;
            }
#pragma unroll
            for (int m = 0; m < 4; ++m) {   // m = t % 4
                const gdn_f4 de4 = gdn_ld4(gt.dendp + 4 * m);
                const gdn_f4 d4  = (gdn_f4) { d[m], d[m + 4], d[m + 8], d[m + 12] };
                *(gdn_f4 *) (s.Dp + l * 16 + 4 * m)  = d4;
                *(gdn_f4 *) (s.DPp + l * 16 + 4 * m) = d4 * de4;
            }
        } else if (w == 1) {
            if (more) {
                gates(t0 + C, pb ^ 1, next_part);
            }
        } else {
#pragma unroll
            for (int it = 0; it < D / 16; ++it) {
                const gdn_f4 q4 = gdn_ld4(Qp + abase + 4 * it);
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    qs = GDN_MFMA(q4[m], st[it][m], qs);
                }
            }
        }
        qs = qs * (dc4 * a.scale);
        __syncthreads();

        // ---- outputs: scale * (G * (Q S0) + P delta)
        {
            const gdn_f4 pa4 = gdn_ld4(s.Pp + (l16 * 4 + lk) * 4);
            const gdn_f4 dl4 = gdn_ld4(s.Dp + vw * 16 + lk * 4);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                qs = GDN_MFMA(pa4[c], dl4[c], qs);
            }
            float * op = out + (int64_t) (t0 + 4 * lk) * so2;
            if (full) {
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    op[r * so2] = qs[r];
                }
            } else {
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    if (t0 + 4 * lk + r < n) {
                        op[r * so2] = qs[r];
                    }
                }
            }
        }

        // ---- state update: S = G_C * S + K^T (delta * G_C/G_t)
        {
            const float   dec = gt.decay[C - 1];
            const gdn_f4  bd4 = gdn_ld4(s.DPp + vw * 16 + lk * 4);
            const float * KT  = s.KT[pb];
#pragma unroll
            for (int it = 0; it < D / 16; ++it) {
                const gdn_f4 ka4 = gdn_ld4(KT + ((it * 16 + l16) * 4 + lk) * 4);
                st[it] = st[it] * dec;
#pragma unroll
                for (int kk = 0; kk < C / 4; ++kk) {
                    st[it] = GDN_MFMA(ka4[kk], bd4[kk], st[it]);
                }
            }
        }

        // ---- publish the next chunk into the other buffer; one barrier separates the chunks
        if (more) {
            publish(t0 + C, pb ^ 1, next_part);
        }
        __syncthreads();
    }

#pragma unroll
    for (int it = 0; it < D / 16; ++it) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            a.state_out[soff + (int64_t) (v0 + vw) * D + it * 16 + 4 * lk + r] = st[it][r];
        }
    }
}
