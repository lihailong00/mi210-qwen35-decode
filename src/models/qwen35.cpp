#include "models.h"
#include "llama-memory-recurrent.h"

#include <cstring>

void llama_model_qwen35::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // Mark recurrent layers (linear attention layers). MTP layers are dense
    // attention-only and must be flagged non-recurrent.
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_8B : LLM_TYPE_2B; break;
        case 32: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_9B; break;
        case 64: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const auto nf = nextn_flags(ml);
    const int trunk_flags = nf.trunk;
    const int mtp_flags   = nf.mtp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // optional projection of the embeddings output
    cls_out   = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), { n_embd, hparams.n_embd_out() }, TENSOR_NOT_REQUIRED);
    cls_out_b = create_tensor(tn(LLM_TENSOR_CLS_OUT, "bias"),   { hparams.n_embd_out() },         TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        // Calculate dimensions from hyperparameters
        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);

        if (!hparams.is_recr(il)) {
            // Attention layers
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);
        } else {
            // Linear attention (gated delta net) specific tensors
            // Create tensors with calculated dimensions
            layer.wqkv           = create_tensor(tn(LLM_TENSOR_ATTN_QKV,       "weight", il), { n_embd, key_dim * 2 + value_dim }, TENSOR_NOT_REQUIRED);
            layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", il), { n_embd, value_dim }, TENSOR_NOT_REQUIRED);
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta       = create_tensor(tn(LLM_TENSOR_SSM_BETA,       "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha      = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,      "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, flags);
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, flags);
    };

    auto load_block_mtp = [&](int il) {
        auto & layer = layers[il];

        // MTP block looks like a full-attention Qwen3.5 decoder block.
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, mtp_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, mtp_flags);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, mtp_flags);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, mtp_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, mtp_flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, mtp_flags);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, mtp_flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, mtp_flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, mtp_flags);

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", il), { 2 * n_embd, n_embd }, mtp_flags);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);
    };

    for (int i = 0; i < n_layer; ++i) {
        load_block_trunk(i, trunk_flags);
    }
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block_mtp(i);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen35::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen35::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // With a large batch the residual is the first ADD operand, so that ggml-alloc computes the sum in place on the
    // residual and the backend can fold the add into the preceding GEMM (beta = 1, see ggml_cuda_match_gemm_add).
    // Small batches keep the original operand order: the single-token graph (buffer layout, fusions) is unchanged.
    // The threshold is the batch size from which the backend fuses (the `mm->ne[1] < 256` check in
    // ggml_cuda_match_gemm_add, ggml/src/ggml-cuda/ggml-cuda.cu): keep the two in sync.
    //
    // The two forms are different graphs for ggml-alloc, which applies a reservation by node and leaf index
    // (ggml_gallocr_alloc_graph). The operand order changes the order in which ggml_build_forward_expand reaches the
    // leafs: in every layer the ffn_down / ffn_gate weights trade places with the leafs below the attention output.
    // Those are pre-allocated weights, which the allocator skips. In the last layer, however, get_rows(inpSA, out_ids)
    // also moves across the tail of the attention block (5 node positions differ) and the graph input out_ids moves
    // to another leaf index, so a reservation made for one form must not be applied to a graph of the other one.
    // It is not: when the form changes between two multi-token graphs (e.g. a full ubatch followed by a rest of
    // less than 256 tokens) the graph is reserved again, because
    //  - ggml_backend_sched_alloc_splits compares the buffer type of every node and leaf index with the previous
    //    graph, and out_ids (host buffer) trades its index with a weight (verified with -ngl 999, 40 and 0), and
    //  - without any GPU device ggml_gallocr_needs_realloc finds a node source that has no allocation in the
    //    reservation (verified with --device none).
    // Both are consequences of the reordered last layer, not guarantees of the allocator: a user of a plain
    // ggml_gallocr that keeps one reservation for both forms, or a change to the tail of this graph, has to make
    // sure that the graph inputs keep their leaf index (e.g. by expanding them first) or that it reserves again.
    const bool residual_first = n_tokens >= 256;

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recr(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = residual_first ? ggml_add(ctx0, inpSA, cur) : ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // Dense FFN layer - without residual connection
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = residual_first ? ggml_add(ctx0, ffn_residual, cur) : ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (crop_after_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if (model.cls_out) {
        ggml_tensor * embd = build_lora_mm(model.cls_out, cur);
        if (model.cls_out_b) {
            embd = ggml_add(ctx0, embd, model.cls_out_b);
        }
        cb(embd, "result_embd_proj", -1);
        res->t_embd = embd;
        ggml_build_forward_expand(gf, embd);
    }

    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen35::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    auto [Qcur_full, Kcur, Vcur] = build_qkv(model.layers[il], cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "Qcur_full", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // graph order: the three projections back to back, then q norm + rope, the gate copy, k norm + rope and the
    // k/v cache stores (build_attn). The CUDA backend runs that whole chain as one kernel
    // (ggml_cuda_try_attn_prep_fusion); emitting the gate CONT right after the q rope keeps the projections alive
    // until the fused outputs are allocated, so they cannot alias the kernel inputs.
    ggml_build_forward_expand(gf, Qcur_full);
    ggml_build_forward_expand(gf, Kcur);
    ggml_build_forward_expand(gf, Vcur);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply MRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    ggml_build_forward_expand(gf, Qcur);
    ggml_build_forward_expand(gf, gate);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

// LLAMA_GDN_FUSED_DECODE=0 disables the fused single-token gated delta net decode path (default: enabled)
static bool qwen35_gdn_fused_decode_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_GDN_FUSED_DECODE");
        return e == nullptr || atoi(e) != 0;
    }();
    return enabled;
}

// LLAMA_GDN_FUSED_DECODE_N: the largest ubatch (tokens of one sequence, e.g. an MTP verification batch) that takes the
// N-token fused ops, also with rollback snapshots (n_rs_seq > 0); default 16 (the backend limit). 0 restores the
// previous graphs exactly: the fused ops only for a single token without snapshots, the unfused graph otherwise.
static int qwen35_gdn_fused_decode_n_max() {
    static const int n_max = [] {
        const char * e = getenv("LLAMA_GDN_FUSED_DECODE_N");
        const int n = e ? atoi(e) : 16;
        return n < 0 ? 0 : (n > 16 ? 16 : n);
    }();
    return n_max;
}

static bool qwen35_tensor_on_dev(const ggml_tensor * t, ggml_backend_dev_t dev) {
    return t != nullptr && t->buffer != nullptr &&
        ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer)) == dev;
}

// Fused decode of one linear-attention layer for the tokens of a single sequence (ggml_gdn_decode_prep(_ext) +
// ggml_gdn_decode_out(_ext)): one token, or up to LLAMA_GDN_FUSED_DECODE_N tokens (MTP verification), with or without
// rollback snapshots. The ops update the conv/ssm states in place in the recurrent cache, so they are only used
// when they run on the device that holds the states (and every weight that steers the op placement); otherwise,
// or if the backend does not support them, nullptr is returned and the caller builds the unfused graph.
// returns the gated, normalized attention output [head_v_dim*num_v_heads, n_tokens] (input of ssm_out)
// Every precondition that the ggml_gdn_decode_prep/_out constructors assert on (plus the contiguity the reshapes need),
// checked before any node is built, so that an unusual GGUF (e.g. ssm_alpha/ssm_beta of different types from a
// --tensor-type override) falls back to the unfused graph instead of aborting. Backend limits are left to supports_op.
static bool qwen35_gdn_fused_decode_args_ok(const llama_layer & layer, const ggml_tensor * qkv_mixed, const ggml_tensor * x,
        const ggml_tensor * z, const ggml_tensor * conv_states, const ggml_tensor * ssm_states, const ggml_tensor * s_copy,
        const int64_t H_k, const int64_t H_v, const int64_t S, const int64_t dst_row, const int64_t n_tokens,
        const int64_t last_row) {
    const ggml_tensor * wa = layer.ssm_alpha;
    const ggml_tensor * wb = layer.ssm_beta;
    const ggml_tensor * dt = layer.ssm_dt;
    const ggml_tensor * a  = layer.ssm_a;
    const ggml_tensor * cw = layer.ssm_conv1d;
    const ggml_tensor * nw = layer.ssm_norm;
    if (!wa || !wb || !dt || !a || !cw || !nw || !s_copy || H_k <= 0 || H_v <= 0 || S <= 0 || H_v % H_k != 0 ||
            n_tokens < 1) {
        return false;
    }
    const int64_t conv_dim = (2*H_k + H_v) * S;
    const int64_t n_embd   = x->ne[0];
    const int64_t d_conv   = cw->ne[0];
    return qkv_mixed->type == GGML_TYPE_F32 && ggml_is_contiguous(qkv_mixed) && qkv_mixed->ne[0] == conv_dim &&
           ggml_nelements(qkv_mixed) == conv_dim*n_tokens &&
           x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && ggml_nelements(x) == n_embd*n_tokens &&
           wa->ne[0] == n_embd && wa->ne[1] == H_v && ggml_is_contiguous_rows(wa) &&
           wb->ne[0] == n_embd && wb->ne[1] == H_v && ggml_is_contiguous_rows(wb) &&
           wa->type == wb->type &&
           dt->type == GGML_TYPE_F32 && ggml_nelements(dt) == H_v && ggml_is_contiguous(dt) &&
           a->type == GGML_TYPE_F32 && ggml_nelements(a) == H_v && ggml_is_contiguous(a) &&
           cw->type == GGML_TYPE_F32 && cw->ne[1] == conv_dim && ggml_is_contiguous(cw) && d_conv >= 2 &&
           conv_states->type == GGML_TYPE_F32 && conv_states->ne[0] == (d_conv - 1)*conv_dim &&
           conv_states->nb[0] == sizeof(float) && dst_row >= 0 && last_row < conv_states->ne[1] &&
           s_copy->type == GGML_TYPE_I32 && ggml_nelements(s_copy) >= 1 &&
           z->type == GGML_TYPE_F32 && ggml_is_contiguous(z) && ggml_nelements(z) == H_v*S*n_tokens &&
           nw->type == GGML_TYPE_F32 && ggml_is_contiguous(nw) && ggml_nelements(nw) == S &&
           ssm_states->type == GGML_TYPE_F32 && ssm_states->ne[0] == S*S*H_v &&
           ssm_states->nb[0] == sizeof(float) && last_row < ssm_states->ne[1];
}

static ggml_tensor * qwen35_build_gdn_fused_decode(
        llm_graph_context  & g,
        const llama_model  & model,
        llm_graph_input_rs * inp,
        ggml_tensor        * x,
        ggml_tensor        * qkv_mixed,
        ggml_tensor        * z,
        int                  il) {
    const auto & hparams = g.hparams;
    const auto & layer   = model.layers[il];
    const auto * mctx    = inp->mctx;

    const int64_t head_k_dim  = hparams.ssm_d_state;
    const int64_t num_k_heads = hparams.ssm_n_group;
    const int64_t num_v_heads = hparams.ssm_dt_rank;
    const int64_t head_v_dim  = hparams.ssm_d_inner / num_v_heads;

    // single token without snapshots: the single-token ops (graph unchanged); 1..LLAMA_GDN_FUSED_DECODE_N tokens or
    // rollback snapshots: the N-token ops (per token the same arithmetic, K = n_rs_seq + 1 snapshot slots)
    const int64_t n_tok  = g.ubatch.n_seq_tokens;
    const bool    single = n_tok == 1 && g.cparams.n_rs_seq == 0;
    if (!qwen35_gdn_fused_decode_enabled() ||
        !(single || (n_tok >= 1 && n_tok <= qwen35_gdn_fused_decode_n_max())) ||
        g.ubatch.n_seqs != 1 || mctx->get_n_rs() != 1 || head_k_dim != head_v_dim ||
        layer.ssm_alpha_s != nullptr || layer.ssm_beta_s != nullptr) {
        return nullptr;
    }
    const int K           = (int) g.cparams.n_rs_seq + 1;
    const int slot_stride = K > 1 ? (int) mctx->get_size() : 0;

    ggml_tensor * conv_states_all = mctx->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx->get_s_l(il);
    if (conv_states_all->type != GGML_TYPE_F32 || ssm_states_all->type != GGML_TYPE_F32) {
        return nullptr;
    }

    ggml_backend_dev_t dev = model.dev_layer(il);
    if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return nullptr;
    }
    // only the CUDA/ROCm backend implements the fused ops on a GPU device. Other devices may still claim support:
    // RPC reports GPU type and supports_op == true for every op without knowing what the remote server implements
    // (a Vulkan remote silently skips unknown ops, Metal/SYCL/OpenCL or an older server abort)
    {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
        if (reg_name == nullptr || (strcmp(reg_name, "CUDA") != 0 && strcmp(reg_name, "ROCm") != 0)) {
            return nullptr;
        }
    }
    for (const ggml_tensor * t : { conv_states_all, ssm_states_all, layer.ssm_alpha, layer.ssm_beta, layer.ssm_dt,
                                   layer.ssm_a, layer.ssm_conv1d, layer.ssm_norm }) {
        if (!qwen35_tensor_on_dev(t, dev)) {
            return nullptr;
        }
    }

    // the alpha/beta projections are computed inside the op: no Hadamard input transform, no LoRA
    if (g.hdmd && (g.hdmd->rot.count(layer.ssm_alpha) || g.hdmd->rot.count(layer.ssm_beta))) {
        return nullptr;
    }
    for (const auto & lora : *g.loras) {
        if (lora.first->get_weight(layer.ssm_alpha) || lora.first->get_weight(layer.ssm_beta)) {
            return nullptr;
        }
    }

    const int64_t last_row = (int64_t) mctx->get_head() + (int64_t) (K - 1)*slot_stride;
    if (!qwen35_gdn_fused_decode_args_ok(layer, qkv_mixed, x, z, conv_states_all, ssm_states_all, inp->s_copy,
            num_k_heads, num_v_heads, head_k_dim, (int64_t) mctx->get_head(), n_tok, last_row)) {
        return nullptr;
    }

    ggml_context * ctx0 = g.ctx0;

    // compute both input projections (same input) back to back, before the fused ops
    ggml_build_forward_expand(g.gf, qkv_mixed);
    ggml_build_forward_expand(g.gf, z);

    // zero a cleared state row first, like build_rs does (no-op view when rs_z < 0);
    // the zeroing nodes are passed as dependencies of the fused ops so they run before them
    const int32_t rs_z = mctx->get_rs_z();
    auto zero_row = [&](ggml_tensor * states) {
        ggml_tensor * row = ggml_view_1d(ctx0, states, states->ne[0]*(rs_z >= 0), rs_z*states->nb[1]*(rs_z >= 0));
        return ggml_scale_inplace(ctx0, row, 0);
    };

    const int dst_row = (int) mctx->get_head();

    ggml_tensor * prep;
    ggml_tensor * out;
    if (single) {
        ggml_tensor * qkv_1d = ggml_reshape_1d(ctx0, qkv_mixed, ggml_nelements(qkv_mixed));

        prep = ggml_gdn_decode_prep(ctx0, qkv_1d, x,
                layer.ssm_alpha, layer.ssm_beta, layer.ssm_dt, layer.ssm_a, layer.ssm_conv1d,
                conv_states_all, inp->s_copy, zero_row(conv_states_all),
                dst_row, (int) num_k_heads, (int) num_v_heads, (int) head_k_dim,
                hparams.f_norm_rms_eps / (float) head_k_dim); // == build_gdn_l2_norm eps

        out = ggml_gdn_decode_out(ctx0, prep, qkv_1d, layer.ssm_conv1d, conv_states_all,
                ggml_reshape_1d(ctx0, z, ggml_nelements(z)), layer.ssm_norm,
                ssm_states_all, inp->s_copy, zero_row(ssm_states_all),
                dst_row, (int) num_k_heads, (int) num_v_heads, (int) head_v_dim,
                hparams.f_norm_rms_eps);
    } else {
        ggml_tensor * qkv_2d = ggml_reshape_2d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_tok);

        prep = ggml_gdn_decode_prep_ext(ctx0, qkv_2d, x,
                layer.ssm_alpha, layer.ssm_beta, layer.ssm_dt, layer.ssm_a, layer.ssm_conv1d,
                conv_states_all, inp->s_copy, zero_row(conv_states_all),
                dst_row, (int) num_k_heads, (int) num_v_heads, (int) head_k_dim,
                hparams.f_norm_rms_eps / (float) head_k_dim, K, slot_stride);

        out = ggml_gdn_decode_out_ext(ctx0, prep, qkv_2d, layer.ssm_conv1d, conv_states_all,
                ggml_reshape_2d(ctx0, z, head_v_dim*num_v_heads, n_tok), layer.ssm_norm,
                ssm_states_all, inp->s_copy, zero_row(ssm_states_all),
                dst_row, (int) num_k_heads, (int) num_v_heads, (int) head_v_dim,
                hparams.f_norm_rms_eps, K, slot_stride);
    }

    if (!ggml_backend_dev_supports_op(dev, prep) || !ggml_backend_dev_supports_op(dev, out)) {
        return nullptr;
    }

    g.cb(prep, "gdn_decode_prep", il);
    g.cb(out,  "final_output",    il);

    return ggml_reshape_2d(ctx0, out, head_v_dim*num_v_heads, n_tok);
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    // single-token decode of one sequence: two fused ops (conv/norms/gates + delta rule/gated norm, states updated
    // in place in the cache) replace ~15 kernels and the two alpha/beta projections
    if (ggml_tensor * fused = qwen35_build_gdn_fused_decode(*this, model, inp, cur, qkv_mixed, z, il)) {
        cur = build_lora_mm(model.layers[il].ssm_out, fused, model.layers[il].ssm_out_s);
        cb(cur, "linear_attn_out", il);
        return cur;
    }

    // small batches (MTP verification): the qkv and z projections next to each other in the graph (otherwise z is only
    // placed before its use in the gated norm), so a backend can run the two GEMVs on the same input as one launch
    // Only for 2..GGML_MMVQ_NC_MAX tokens (never the single-token graph, never a batch the N-column GEMVs do not take,
    // where the reorder would only split the gated-norm fusion) and only while the N-column GEMVs that run the pair are
    // on: GGML_MMVQ_NC=0, or turning off one of the families of the pair (GGML_MMVQ_NC_MIX=0: mixed IQ3 + IQ4 pair,
    // GGML_MMVQ_NC_IQ3=0 / GGML_MMVQ_IQ3=0: IQ3 incl. the same-type multi-job pair) keeps the previous node order exactly.
    // Same parses as ggml_cuda_mmvq_nc_max() and the per-family switches of the HIP backend (mmvq-iq3.cu).
    static const int nc_order_max = [] {
        const char * e = getenv("GGML_MMVQ_NC");
        while (e && (*e == ' ' || *e == '\t')) {
            ++e;
        }
        if (e && (e[0] == '0' || e[0] == 'f' || e[0] == 'F' || e[0] == 'n' || e[0] == 'N')) {
            return 0;
        }
        for (const char * name : { "GGML_MMVQ_NC_MIX", "GGML_MMVQ_NC_IQ3", "GGML_MMVQ_IQ3" }) {
            const char * v = getenv(name);
            if (v && v[0] == '0') {
                return 0;
            }
        }
        const char * m = getenv("GGML_MMVQ_NC_MAX");
        const int n = m ? atoi(m) : 16;   // MTP2-P0: default 16 (= ggml_cuda_mmvq_nc_max)
        // MTP2-P0: qkv/z reorder up to 16 tokens
        return n < 2 ? 2 : (n > 16 ? 16 : n);
    }();
    if (ubatch.n_tokens >= 2 && (int) ubatch.n_tokens <= nc_order_max) {
        ggml_build_forward_expand(gf, qkv_mixed);
        ggml_build_forward_expand(gf, z);
    }

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    // note: need explicit repeat only if we are not using the fused GDN.
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * head_dim , n_tokens * n_seqs ]
    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Qwen3.5 does not use MoE FFN
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
        model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
        model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);

    return cur;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 dense series
llama_model_qwen35::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN35 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN35 MTP currently only supports a single MTP block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // hparams.n_layer includes both main model layers and MTP layers. The MTP
    // layer is stored immediately after the main layers in model.layers[].
    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    // TODO: make static using `ggml_build_forward_select()`
    //       see llm_graph_context::build_inp_embd() for reference
    ggml_tensor * tok_embd;
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);

        // a Hadamard-latent table stores rotated rows; restore the primal basis
        if (hdmd) {
            if (const auto it = hdmd->inv.find(tok_embd_w); it != hdmd->inv.end()) {
                tok_embd = llama_mul_mat_hadamard(ctx0, tok_embd, it->second.rot);
                if (it->second.signs) {
                    tok_embd = ggml_mul(ctx0, tok_embd, it->second.signs);
                }
            }
        }
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    auto [Qcur_full, Kcur, Vcur] = build_qkv(layer, cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_post_norm", il);

    cur = build_ffn(cur,
            layer.ffn_up,   nullptr, layer.ffn_up_s,
            layer.ffn_gate, nullptr, layer.ffn_gate_s,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35 MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35 MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
}
