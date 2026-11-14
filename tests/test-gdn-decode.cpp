// Kernel-level check of the fused single-token gated delta net decode ops (ggml_gdn_decode_prep/out) against the
// unfused op chain that llama.cpp builds for qwen3.5 linear-attention layers (src/models/qwen35.cpp +
// src/models/delta-net-base.cpp with n_rs_seq == 0 and the fused gated_delta_net op), on one backend.
//
// Both paths run on the same inputs for several decode steps, each path carrying its own recurrent states
// (conv + ssm, updated in the "cache" tensors). Reports max abs/rel differences and the number of bit-identical
// values of the layer output, the intermediate q/k/v/beta values and the updated states.
//
// usage: test-gdn-decode [backend name (default: first GPU)] [n_steps (default 8)] [weights: file prefix | -]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct cmp_stats {
    double max_abs = 0.0;
    double max_rel = 0.0;
    size_t n       = 0;
    size_t n_same  = 0;
};

static cmp_stats compare(const std::vector<float> & a, const std::vector<float> & b) {
    cmp_stats s;
    s.n = a.size();
    for (size_t i = 0; i < a.size(); ++i) {
        if (memcmp(&a[i], &b[i], sizeof(float)) == 0) {
            s.n_same++;
            continue;
        }
        const double d = fabs((double) a[i] - (double) b[i]);
        const double r = d / std::max(1e-30, std::max(fabs((double) a[i]), fabs((double) b[i])));
        s.max_abs = std::max(s.max_abs, d);
        s.max_rel = std::max(s.max_rel, r);
    }
    return s;
}

static std::vector<float> get_f32(const ggml_tensor * t, size_t offset_elems = 0, size_t n = 0) {
    if (n == 0) {
        n = ggml_nelements(t);
    }
    std::vector<float> v(n);
    ggml_backend_tensor_get(t, v.data(), offset_elems*sizeof(float), n*sizeof(float));
    return v;
}

static void fill_uniform(std::mt19937 & rng, ggml_tensor * t, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        std::vector<float> v(n);
        for (auto & x : v) {
            x = dist(rng);
        }
        ggml_backend_tensor_set(t, v.data(), 0, n*sizeof(float));
    } else if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> v(n);
        for (auto & x : v) {
            x = ggml_fp32_to_bf16(dist(rng));
        }
        ggml_backend_tensor_set(t, v.data(), 0, n*sizeof(ggml_bf16_t));
    } else {
        GGML_ABORT("unsupported type");
    }
}

int main(int argc, char ** argv) {
    const char * backend_name = argc > 1 ? argv[1] : nullptr;
    const int    n_steps      = argc > 2 ? atoi(argv[2]) : 8;

    // qwen3.5-27B linear attention shapes
    const int   H_k    = 16;
    const int   H_v    = 48;
    const int   S      = 128;
    const int   n_embd = 5120;
    const int   d_conv = 4;
    const float eps    = 1e-6f;

    const int64_t conv_dim = (int64_t) (2*H_k + H_v)*S;
    const int64_t n_embd_r = (d_conv - 1)*conv_dim;
    const int64_t n_embd_s = (int64_t) S*S*H_v;
    const int64_t n_rows   = 2;   // cache rows; the state is read from and written to dst_row
    const int     dst_row  = 1;

    ggml_backend_load_all();
    ggml_backend_t backend = nullptr;
    if (backend_name) {
        backend = ggml_backend_init_by_name(backend_name, nullptr);
    } else {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    if (!backend) {
        fprintf(stderr, "failed to init backend\n");
        return 1;
    }
    printf("backend: %s\n", ggml_backend_name(backend));

    // persistent tensors: weights, per-step inputs, two copies of the recurrent cache
    ggml_init_params ip = { 64*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_w = ggml_init(ip);

    ggml_tensor * w_alpha = ggml_new_tensor_2d(ctx_w, GGML_TYPE_BF16, n_embd, H_v);
    ggml_tensor * w_beta  = ggml_new_tensor_2d(ctx_w, GGML_TYPE_BF16, n_embd, H_v);
    ggml_tensor * dt      = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, H_v);
    ggml_tensor * a       = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, H_v);
    ggml_tensor * conv_w  = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, d_conv, conv_dim);
    ggml_tensor * norm_w  = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, S);

    ggml_tensor * qkv_in  = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, conv_dim);
    ggml_tensor * z_in    = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, (int64_t) S*H_v, 1);
    ggml_tensor * x_in    = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, n_embd, 1);
    ggml_tensor * s_copy  = ggml_new_tensor_1d(ctx_w, GGML_TYPE_I32, 1);

    ggml_tensor * cs_ref  = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, n_embd_r, n_rows);
    ggml_tensor * ss_ref  = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, n_embd_s, n_rows);
    ggml_tensor * cs_fus  = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, n_embd_r, n_rows);
    ggml_tensor * ss_fus  = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, n_embd_s, n_rows);

    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);

    std::mt19937 rng(1234);
    fill_uniform(rng, w_alpha, -0.05f, 0.05f);
    fill_uniform(rng, w_beta,  -0.05f, 0.05f);
    fill_uniform(rng, dt,      -1.0f, 1.0f);
    fill_uniform(rng, a,       -8.0f, -0.05f);
    fill_uniform(rng, conv_w,  -0.6f, 0.6f);
    fill_uniform(rng, norm_w,   0.5f, 1.5f);
    {
        // start from a non-trivial state
        std::vector<float> v(n_embd_r*n_rows);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto & x : v) x = dist(rng);
        ggml_backend_tensor_set(cs_ref, v.data(), 0, v.size()*sizeof(float));
        ggml_backend_tensor_set(cs_fus, v.data(), 0, v.size()*sizeof(float));
        std::vector<float> s(n_embd_s*n_rows);
        std::uniform_real_distribution<float> dist_s(-0.05f, 0.05f);
        for (auto & x : s) x = dist_s(rng);
        ggml_backend_tensor_set(ss_ref, s.data(), 0, s.size()*sizeof(float));
        ggml_backend_tensor_set(ss_fus, s.data(), 0, s.size()*sizeof(float));
        const int32_t row = dst_row;
        ggml_backend_tensor_set(s_copy, &row, 0, sizeof(row));
    }

    // graphs
    ggml_init_params gp = { 512*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(gp);

    // ---- reference: the unfused chain of qwen35.cpp build_layer_attn_linear (n_seqs = n_seq_tokens = 1)
    ggml_cgraph * gf_ref = ggml_new_graph(ctx);
    ggml_tensor * ref_q = nullptr, * ref_k = nullptr, * ref_beta = nullptr, * ref_out = nullptr;
    {
        ggml_tensor * qkv_mixed = ggml_reshape_3d(ctx, qkv_in, conv_dim, 1, 1);
        ggml_tensor * z = z_in;
        ggml_tensor * cur = x_in;

        ggml_tensor * beta = ggml_mul_mat(ctx, w_beta, cur);
        beta = ggml_reshape_4d(ctx, beta, 1, H_v, 1, 1);
        beta = ggml_sigmoid(ctx, beta);

        ggml_tensor * alpha = ggml_mul_mat(ctx, w_alpha, cur);
        alpha = ggml_reshape_3d(ctx, alpha, H_v, 1, 1);
        ggml_tensor * alpha_biased   = ggml_add(ctx, alpha, dt);
        ggml_tensor * alpha_softplus = ggml_softplus(ctx, alpha_biased);
        ggml_tensor * gate = ggml_mul(ctx, alpha_softplus, a);
        gate = ggml_reshape_4d(ctx, gate, 1, H_v, 1, 1);

        // build_conv_state -> build_rs (rs_z < 0: empty zeroing view)
        ggml_tensor * conv_states;
        {
            ggml_tensor * states = ggml_reshape_2d(ctx, cs_ref, n_embd_r, cs_ref->ne[1]);
            ggml_tensor * state_zero = ggml_view_1d(ctx, states, 0, 0);
            ggml_build_forward_expand(gf_ref, ggml_scale_inplace(ctx, state_zero, 0));
            ggml_tensor * states_all = ggml_get_rows(ctx, states, s_copy);
            conv_states = ggml_view_2d(ctx, states_all, n_embd_r, 1, states_all->nb[1], 0);
            ggml_build_forward_expand(gf_ref, conv_states);
        }
        conv_states = ggml_reshape_3d(ctx, conv_states, d_conv - 1, conv_dim, 1);
        ggml_tensor * qkv_t = ggml_transpose(ctx, qkv_mixed);
        ggml_tensor * conv_input = ggml_concat(ctx, conv_states, qkv_t, 0);
        {
            ggml_tensor * conv_state_last = ggml_view_3d(ctx, conv_input, d_conv - 1, conv_dim, 1,
                    conv_input->nb[1], conv_input->nb[2], ggml_row_size(conv_input->type, 1));
            ggml_tensor * conv_state_update = ggml_view_2d(ctx, cs_ref, n_embd_r, 1, cs_ref->nb[1],
                    dst_row*ggml_row_size(cs_ref->type, n_embd_r));
            ggml_build_forward_expand(gf_ref, ggml_cpy(ctx, conv_state_last, conv_state_update));
        }

        // build_rs for the ssm state
        ggml_tensor * state;
        {
            ggml_tensor * states = ggml_reshape_2d(ctx, ss_ref, n_embd_s, ss_ref->ne[1]);
            ggml_tensor * state_zero = ggml_view_1d(ctx, states, 0, 0);
            ggml_build_forward_expand(gf_ref, ggml_scale_inplace(ctx, state_zero, 0));
            ggml_tensor * states_all = ggml_get_rows(ctx, states, s_copy);
            state = ggml_view_2d(ctx, states_all, n_embd_s, 1, states_all->nb[1], 0);
            ggml_build_forward_expand(gf_ref, state);
        }
        state = ggml_reshape_4d(ctx, state, S, S, H_v, 1);

        ggml_tensor * conv_out = ggml_ssm_conv(ctx, conv_input, conv_w);
        conv_out = ggml_silu(ctx, conv_out);

        const int64_t nb1_qkv = ggml_row_size(conv_out->type, conv_dim);
        ggml_tensor * q_conv = ggml_view_4d(ctx, conv_out, S, H_k, 1, 1,
                ggml_row_size(conv_out->type, S), nb1_qkv, nb1_qkv, 0);
        ggml_tensor * k_conv = ggml_view_4d(ctx, conv_out, S, H_k, 1, 1,
                ggml_row_size(conv_out->type, S), nb1_qkv, nb1_qkv, (int64_t) S*H_k*ggml_element_size(conv_out));
        ggml_tensor * v_conv = ggml_view_4d(ctx, conv_out, S, H_v, 1, 1,
                ggml_row_size(conv_out->type, S), nb1_qkv, nb1_qkv, ggml_row_size(conv_out->type, 2*S*H_k));

        // build_gdn_l2_norm
        const float n = S;
        q_conv = ggml_scale(ctx, ggml_rms_norm(ctx, q_conv, eps/n), 1.0f/sqrtf(n));
        k_conv = ggml_scale(ctx, ggml_rms_norm(ctx, k_conv, eps/n), 1.0f/sqrtf(n));

        // build_recurrent_attn -> build_delta_net_fused (K = 1) + state write back
        ggml_tensor * result = ggml_gated_delta_net(ctx, q_conv, k_conv, v_conv, gate, beta, state, 1);
        ggml_tensor * output = ggml_view_4d(ctx, result, S, H_v, 1, 1,
                ggml_row_size(result->type, S), ggml_row_size(result->type, S*H_v),
                ggml_row_size(result->type, S*H_v), 0);
        ggml_tensor * new_state = ggml_view_4d(ctx, result, S, S, H_v, 1,
                ggml_row_size(result->type, S), ggml_row_size(result->type, S*S),
                ggml_row_size(result->type, S*S*H_v), ggml_row_size(result->type, S*H_v));
        ggml_build_forward_expand(gf_ref, ggml_cpy(ctx, new_state,
                ggml_view_2d(ctx, ss_ref, n_embd_s, 1, ss_ref->nb[1], dst_row*n_embd_s*ggml_element_size(ss_ref))));

        // build_norm_gated
        ggml_tensor * z_2d = ggml_reshape_4d(ctx, z, S, H_v, 1, 1);
        ggml_tensor * normalized = ggml_mul(ctx, ggml_rms_norm(ctx, output, eps), norm_w);
        ggml_tensor * gated_silu = ggml_silu(ctx, z_2d);
        ggml_tensor * attn_out_norm = ggml_mul(ctx, normalized, gated_silu);
        ref_out = ggml_reshape_2d(ctx, attn_out_norm, (int64_t) S*H_v, 1);
        ggml_set_output(ref_out);
        ggml_build_forward_expand(gf_ref, ref_out);

        // keep the intermediates readable (copies at the end do not change the fusion patterns above)
        ref_q    = ggml_cpy(ctx, q_conv, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) S*H_k));
        ref_k    = ggml_cpy(ctx, k_conv, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) S*H_k));
        ref_beta = ggml_cpy(ctx, beta,   ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H_v));
        for (ggml_tensor * t : { ref_q, ref_k, ref_beta }) {
            ggml_set_output(t);
            ggml_build_forward_expand(gf_ref, t);
        }
    }

    // ---- fused
    ggml_cgraph * gf_fus = ggml_new_graph(ctx);
    ggml_tensor * fus_prep = nullptr, * fus_out = nullptr;
    {
        fus_prep = ggml_gdn_decode_prep(ctx, qkv_in, ggml_reshape_1d(ctx, x_in, n_embd), w_alpha, w_beta, dt, a,
                conv_w, cs_fus, s_copy, nullptr, dst_row, H_k, H_v, S, eps/(float) S);
        fus_out = ggml_gdn_decode_out(ctx, fus_prep, qkv_in, conv_w, cs_fus,
                ggml_reshape_1d(ctx, z_in, (int64_t) S*H_v), norm_w, ss_fus, s_copy, nullptr, dst_row, H_k, H_v, S, eps);
        ggml_set_output(fus_prep);
        ggml_set_output(fus_out);
        ggml_build_forward_expand(gf_fus, fus_out);
    }

    ggml_gallocr_t galloc_ref = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_t galloc_fus = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc_ref, gf_ref) || !ggml_gallocr_alloc_graph(galloc_fus, gf_fus)) {
        fprintf(stderr, "graph allocation failed\n");
        return 1;
    }

    bool all_same = true;
    auto report = [&](const char * name, const cmp_stats & s) {
        printf("  %-12s n=%8zu  bit-identical=%8zu  max_abs=%.3e  max_rel=%.3e\n", name, s.n, s.n_same, s.max_abs, s.max_rel);
        all_same = all_same && s.n_same == s.n;
    };

    for (int step = 0; step < n_steps; ++step) {
        fill_uniform(rng, qkv_in, -2.0f, 2.0f);
        fill_uniform(rng, z_in,   -3.0f, 3.0f);
        fill_uniform(rng, x_in,   -1.0f, 1.0f);

        if (ggml_backend_graph_compute(backend, gf_ref) != GGML_STATUS_SUCCESS ||
            ggml_backend_graph_compute(backend, gf_fus) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "compute failed\n");
            return 1;
        }
        ggml_backend_synchronize(backend);

        printf("step %d\n", step);
        const std::vector<float> prep = get_f32(fus_prep);
        auto slice = [&](size_t off, size_t len) { return std::vector<float>(prep.begin() + off, prep.begin() + off + len); };
        report("q",      compare(get_f32(ref_q),    slice(0, (size_t) S*H_k)));
        report("k",      compare(get_f32(ref_k),    slice((size_t) S*H_k, (size_t) S*H_k)));
        report("beta",   compare(get_f32(ref_beta), slice((size_t) 2*S*H_k + H_v, H_v)));
        report("out",    compare(get_f32(ref_out),  get_f32(fus_out)));
        report("conv_st",compare(get_f32(cs_ref, dst_row*n_embd_r, n_embd_r), get_f32(cs_fus, dst_row*n_embd_r, n_embd_r)));
        report("ssm_st", compare(get_f32(ss_ref, dst_row*n_embd_s, n_embd_s), get_f32(ss_fus, dst_row*n_embd_s, n_embd_s)));
    }

    printf("%s\n", all_same ? "RESULT: bit-identical" : "RESULT: differences found");

    ggml_gallocr_free(galloc_ref);
    ggml_gallocr_free(galloc_fus);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
    ggml_backend_free(backend);
    return 0;
}
