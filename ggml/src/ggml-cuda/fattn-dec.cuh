#pragma once

#include "common.cuh"

// Single-query flash attention decode (f16 K/V, head size 256, no sinks/ALiBi/softcap) for the qwen35 gated attention:
// split-KV over 8 small workgroups per KV head (all GQA heads of one KV head share the K/V loads), the partial results
// are combined in the same launch (in-kernel barrier over the workgroups of a KV head, no separate combine kernel) and
// the epilogue writes gated = sigmoid(gate) * attn plus the q8_1 of gated for the following single-token GEMV.
// Returns false (nothing launched) if the call is not supported; then the caller runs the generic path.
bool ggml_cuda_fattn_dec_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * fa, const float * gate, ggml_tensor * mul_node);
// MTP2-O2: the same for 2..16 query tokens of one sequence (multi-token verification, causal mask). Every query is
// bit-identical to the single-token call above at its position (its own n_kv from the mask row). Returns false (nothing
// launched) if not supported; GGML_CUDA_FA_DEC_NQ=0 disables.
bool ggml_cuda_fattn_dec_nq_wanted(int64_t n_tokens);
bool ggml_cuda_fattn_dec_gate_nq(ggml_backend_cuda_context & ctx, const ggml_tensor * fa, const float * gate, ggml_tensor * mul_node);
