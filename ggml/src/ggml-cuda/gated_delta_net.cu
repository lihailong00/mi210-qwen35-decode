#include "gated_delta_net.cuh"
#include "gated-delta-net-mma.cuh"
#include "ggml-cuda/common.cuh"

#include <cstdint>

constexpr int gdn_cols_per_warp = 4;

template <int S_v, bool KDA, bool keep_rs_t, int cols_per_warp = gdn_cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * lanes_per_col + lane_in_col;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // four columns per warp (see the kernel); shrink the CTA when the wider CTA would leave
    // SMs without a CTA, so small head counts keep the device filled
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int cols_per_warp = gdn_cols_per_warp;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    // one CTA covers cols_per_warp*num_warps columns (see the kernel)
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
    }
}

static ggml_cuda_gdn_mma_args ggml_cuda_gdn_mma_args_from_tensor(const ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * g     = dst->src[3];
    const ggml_tensor * beta  = dst->src[4];
    const ggml_tensor * state = dst->src[5];

    const int64_t S_v = v->ne[0];
    const int64_t H   = v->ne[1];
    return {
        (const float *) q->data,
        (const float *) k->data,
        (const float *) v->data,
        (const float *) g->data,
        (const float *) beta->data,
        (const float *) state->data,
        (float *) dst->data,
        nullptr,
        H,
        q->ne[1],
        v->ne[2],
        v->ne[3],
        v->ne[3] / q->ne[3],
        (int64_t) (q->nb[1] / sizeof(float)),
        (int64_t) (q->nb[2] / sizeof(float)),
        (int64_t) (q->nb[3] / sizeof(float)),
        (int64_t) (v->nb[1] / sizeof(float)),
        (int64_t) (v->nb[2] / sizeof(float)),
        (int64_t) (v->nb[3] / sizeof(float)),
        (int64_t) (beta->nb[1] / sizeof(float)),
        (int64_t) (beta->nb[2] / sizeof(float)),
        (int64_t) (beta->nb[3] / sizeof(float)),
        1.0f / sqrtf((float) S_v),
        g->ne[0] == 1 && ggml_get_op_params_i32(dst, 0) < v->ne[2] && S_v == 128 && q->ne[0] == 128,
    };
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    GGML_ASSERT(neq3 > 0 && nev3 >= neq3 && nev3 % neq3 == 0);
    GGML_ASSERT(nek3 == neq3);
    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    // MTP2-merge: with rollback snapshots (K > 1, e.g. a prompt evaluated while MTP drafting is on) the chunked prefill
    // kernels (WY / MFMA) are not used, so the outputs and the final state differed in the last bits from the same batch
    // without snapshots (the no-speculation run). GGML_CUDA_GDN_KEEP_RS_EXACT (default 1): the snapshot recurrence below
    // runs from a copy of the input state, then the chunked kernel the K = 1 call would take recomputes the outputs and
    // slot 0 from that copy, so both are bit-identical to the call without snapshots (slots 1..K-1 come from the
    // recurrence). 0 restores the previous behaviour exactly.
    // GGML_CUDA_GDN_KEEP_RS_TAIL (default 1): slots 1..K-1 only need the states after the last K-1 tokens, so for a long
    // batch the chunked kernel runs over the first n - K + 1 tokens (giving slot K-1) and the per-token recurrence only
    // over the last K - 1 tokens (slots K-2..1; slot 0 is rewritten by the full chunked call). The outputs and slot 0 are
    // unchanged; slots 1..K-1 now come from the chunked prefix state instead of a recurrence over the whole batch.
    static const bool keep_rs_exact = [] {
        const char * e = getenv("GGML_CUDA_GDN_KEEP_RS_EXACT");
        return e == nullptr || atoi(e) != 0;
    }();
    static const bool keep_rs_tail = [] {
        const char * e = getenv("GGML_CUDA_GDN_KEEP_RS_TAIL");
        return e == nullptr || atoi(e) != 0;
    }();
    const bool exact_k1 = keep_rs && keep_rs_exact && !kda && n_tokens >= 2;
    ggml_cuda_pool_alloc<float> s_copy(ctx.pool());
    if (exact_k1) {
        const size_t ns = (size_t) S_v*S_v*H*n_seqs;
        s_copy.alloc(ns);
        CUDA_CHECK(cudaMemcpyAsync(s_copy.get(), s_d, ns*sizeof(float), cudaMemcpyDeviceToDevice, stream));
        s_d = s_copy.get();
    }
    if (!exact_k1) {
        ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
        args.state_out = state_d;
        if (ggml_cuda_gdn_mma_launch(ctx.device, args, stream)) {
            return;
        }
        if (!keep_rs && ggml_cuda_gdn_wy_launch(ctx, args, stream)) {
            return;
        }
        if (!keep_rs && ggml_cuda_gdn_mfma_launch(ctx.device, args, stream)) {
            return;
        }
    }
    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            // tail mode: the prefix is long enough for the chunked kernels (2 chunks of 16 tokens)
            const int64_t n_pre = n_tokens - K + 1;
            const bool tail = exact_k1 && keep_rs_tail && K >= 2 && n_pre >= 64;
            // split mode (one sequence): the chunked call over the whole batch is issued as two calls split at a chunk
            // boundary n_a (a multiple of 16 with at least 64 tokens on each side and n_a <= n - K + 1), the second
            // starting from the state the first leaves. The chunks are the same as in the single call (they start at
            // token 0 in steps of 16 and the scan carries the state from chunk to chunk), so the outputs and the final
            // state are bit-identical to it (checked against GGML_CUDA_GDN_KEEP_RS_SPLIT=0); the snapshot recurrence then
            // only runs over the n - n_a tokens after n_a, from that state, instead of a second chunked call over the
            // first n - K + 1 tokens. GGML_CUDA_GDN_KEEP_RS_SPLIT=0 keeps the two chunked calls of the tail mode.
            static const bool keep_rs_split = [] {
                const char * e = getenv("GGML_CUDA_GDN_KEEP_RS_SPLIT");
                return e == nullptr || atoi(e) != 0;
            }();
            const int64_t n_a = ((n_tokens - 64) / 16) * 16 < (n_pre / 16) * 16 ? ((n_tokens - 64) / 16) * 16 : (n_pre / 16) * 16;
            bool split_done = false;
            if (tail && keep_rs_split && n_seqs == 1 && n_a >= 64 && n_tokens - n_a >= 64) {
                ggml_cuda_pool_alloc<float> s_a(ctx.pool(), (size_t) S_v*S_v*H*n_seqs);
                ggml_cuda_gdn_mma_args a1 = ggml_cuda_gdn_mma_args_from_tensor(dst);
                a1.state     = s_d;
                a1.state_out = s_a.get();
                a1.n_tokens  = n_a;
                // both halves must take the same chunked kernel as the single call (the WY kernel on CDNA2)
                ggml_cuda_gdn_mma_args a_full = ggml_cuda_gdn_mma_args_from_tensor(dst);
                a_full.state = s_d; a_full.state_out = state_d;
                ggml_cuda_gdn_mma_args a2 = a_full;
                a2.q = a_full.q + n_a*sq2; a2.k = a_full.k + n_a*sq2; a2.v = a_full.v + n_a*sv2;
                a2.g = a_full.g + n_a*sb2; a2.beta = a_full.beta + n_a*sb2;
                a2.dst = a_full.dst + n_a*S_v*H;
                a2.state = s_a.get();
                a2.n_tokens = n_tokens - n_a;
                if (ggml_cuda_gdn_wy_launch(ctx, a1, stream)) {
                    {   // slots K-1..0 from the tokens after n_a (slot 0 is rewritten by the second chunked call)
                        ggml_cuda_pool_alloc<float> scratch(ctx.pool(), (size_t) S_v*H*(n_tokens - n_a)*n_seqs);
                        launch_gated_delta_net<false, true>(q_d + n_a*sq2, k_d + n_a*sq2, v_d + n_a*sv2,
                            g_d + n_a*sb2, b_d + n_a*sb2, s_a.get(), scratch.get(), state_d,
                            S_v, H, n_tokens - n_a, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
                    }
                    if (ggml_cuda_gdn_wy_launch(ctx, a2, stream)) {
                        return;
                    }
                    GGML_ABORT("gated_delta_net: second chunked call declined");   // cannot happen: same checks as the first
                }
            }
            GGML_UNUSED(split_done);
            if (tail) {
                float * slot_last = state_d + (int64_t) (K - 1) * state_slot_stride;
                {   // slot K-1 = state after the first n_pre tokens (outputs to scratch)
                    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), (size_t) S_v*H*n_pre*n_seqs);
                    ggml_cuda_gdn_mma_args ap = ggml_cuda_gdn_mma_args_from_tensor(dst);
                    ap.state     = s_d;
                    ap.state_out = slot_last;
                    ap.dst       = scratch.get();
                    ap.n_tokens  = n_pre;
                    if (!ggml_cuda_gdn_mma_launch(ctx.device, ap, stream) && !ggml_cuda_gdn_wy_launch(ctx, ap, stream) &&
                        !ggml_cuda_gdn_mfma_launch(ctx.device, ap, stream)) {
                        launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, scratch.get(), slot_last,
                            S_v, H, n_pre, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
                    }
                }
                {   // slots K-2..0 from the last K-1 tokens, starting at slot K-1 (slot 0 is rewritten below)
                    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), (size_t) S_v*H*(K - 1)*n_seqs);
                    launch_gated_delta_net<false, true>(q_d + n_pre*sq2, k_d + n_pre*sq2, v_d + n_pre*sv2,
                        g_d + n_pre*sb2, b_d + n_pre*sb2, slot_last, scratch.get(), state_d,
                        S_v, H, K - 1, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
                }
            } else {
                launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
            }
            if (exact_k1) {   // MTP2-merge: outputs + slot 0 as the call without snapshots computes them
                ggml_cuda_gdn_mma_args args = ggml_cuda_gdn_mma_args_from_tensor(dst);
                args.state     = s_d;
                args.state_out = state_d;
                if (!ggml_cuda_gdn_mma_launch(ctx.device, args, stream) && !ggml_cuda_gdn_wy_launch(ctx, args, stream) &&
                    !ggml_cuda_gdn_mfma_launch(ctx.device, args, stream) && tail) {
                    // the K = 1 call runs the recurrence; in tail mode the outputs are not written yet
                    launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                        sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
                }
                // none taken outside tail mode: the K = 1 call runs the same recurrence, whose outputs and last state
                // are the ones above
            }
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
