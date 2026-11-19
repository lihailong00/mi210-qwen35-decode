"""vLLM monkeypatches for the non-linear decode ops of Qwen3.5/3.8 (role "rest").

Patch names (env VOPT_PATCHES, comma list):
  norms : GemmaRMSNorm (+fused residual add) -> one HIP kernel (was ~11 native kernels)
  act   : SiluAndMul for M <= VOPT_ACT_MAXM -> multi-block HIP kernel (vLLM's uses 1 block/token)
  gdn   : GDN pure-decode path -> 1 fused HIP kernel (conv1d update + l2norm + gating + delta
          rule + gated RMSNorm), replaces zeros fill + conv + recurrent + norm (4 launches)
  attn  : Qwen3NextAttention: enable the (Triton) fused qk-norm + RoPE + gate split on ROCm,
          sigmoid(gate)*out in one HIP kernel
  gdn (MTP): pure speculative-verify batches -> 1 fused HIP kernel for R requests x T tokens
          (VOPT_GDN_SPEC, default 1), per token bitwise equal to the 1-token decode kernel,
          writes vLLM's per-position state slots (rollback contract of the stock spec path)
  attn (MTP): TRITON_ATTN multi-query batches of <= seq_threshold_3D tokens are run as 1-query
          pseudo-sequences (VOPT_ATTN_SPEC, default 1): per token identical to 1-token decode
Every patch falls back to the stock code for anything it does not handle (prefill, mixed
batches, other dtypes/layouts, M above the cap).
"""
import os
import sys

import torch

from .ext import load_ext

ACT_MAXM = int(os.environ.get("VOPT_ACT_MAXM", "256"))
GDN_MAXM = int(os.environ.get("VOPT_GDN_MAXM", "512"))
GDN_SPEC = os.environ.get("VOPT_GDN_SPEC", "1") == "1"          # fused multi-token GDN verify
ATTN_SPEC = os.environ.get("VOPT_ATTN_SPEC", "1") == "1"        # per-token exact verify attention
ATTN_SPEC_MAXQ = int(os.environ.get("VOPT_ATTN_SPEC_MAXQ", "8"))
# verify GDN: non-temporal state stores (bit-identical values). auto = only for 1 request (measured
# faster there, slower for 2+); the 1-token decode kernel keeps its deploy-v1 stores.
GDN_NT_MODE = os.environ.get("VOPT_GDN_NT", "auto")
GDN_DBG = int(os.environ.get("VOPT_GDN_DBG", "0"))  # profiling experiments only (1/2: skip state load/store)


def _spec_flags(nreq):
    nt = GDN_NT_MODE == "1" or (GDN_NT_MODE == "auto" and nreq == 1)
    return (4 if nt else 0) | GDN_DBG


def _log(msg):
    print(f"[vopt_rest] {msg}", file=sys.stderr)


def _bf16_rows(t, D):
    return (t.is_cuda and t.dtype == torch.bfloat16 and t.dim() == 2 and t.size(1) == D
            and t.stride(1) == 1 and t.stride(0) % 8 == 0 and t.data_ptr() % 16 == 0)


# ----------------------------------------------------------------------------- norms
def install_norms(mod):
    load_ext()
    G = mod.GemmaRMSNorm
    if getattr(G, "_vopt_rest", False):
        return
    orig_native = G.forward_native

    def forward(self, x, residual=None):
        w = self.weight
        D = x.shape[-1]
        if (x.is_cuda and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and D % 8 == 0
                and w.is_contiguous() and x.numel() > 0):
            x2 = x.reshape(-1, D)
            if residual is None:
                if _bf16_rows(x2, D):
                    return torch.ops.vopt_rest.gemma_rms_norm(x2, w, self.variance_epsilon).view(x.shape)
            else:
                r2 = residual.reshape(-1, D)
                if _bf16_rows(x2, D) and _bf16_rows(r2, D) and r2.size(0) == x2.size(0):
                    y, r = torch.ops.vopt_rest.gemma_add_rms_norm(x2, r2, w, self.variance_epsilon)
                    return y.view(x.shape), r.view(residual.shape)
        return orig_native(self, x, residual)

    G.forward_cuda = forward
    G.forward_hip = forward
    G.forward_native = forward
    G._vopt_rest = True
    _log("GemmaRMSNorm patched")


# ----------------------------------------------------------------------------- act
def install_act(mod):
    load_ext()
    S = mod.SiluAndMul
    if getattr(S, "_vopt_rest", False):
        return
    orig_cuda = S.forward_cuda

    def forward_cuda(self, x):
        d2 = x.shape[-1]
        if (x.is_cuda and x.dtype == torch.bfloat16 and d2 % 16 == 0 and x.numel() > 0):
            x2 = x.reshape(-1, d2)
            if x2.size(0) <= ACT_MAXM and _bf16_rows(x2, d2):
                return torch.ops.vopt_rest.silu_mul(x2).view(*x.shape[:-1], d2 // 2)
        return orig_cuda(self, x)

    S.forward_cuda = forward_cuda
    S.forward_hip = forward_cuda
    S._vopt_rest = True
    _log(f"SiluAndMul patched (M <= {ACT_MAXM})")


# ----------------------------------------------------------------------------- gdn
_SCR = {}


def _gdn_scratch(dev):
    s = _SCR.get(dev)
    if s is None:
        s = torch.zeros(GDN_MAXM * 16, dtype=torch.int32, device=dev)
        _SCR[dev] = s
    return s


def install_gdn(mod):
    load_ext()
    C = mod.QwenGatedDeltaNetAttention
    if getattr(C, "_vopt_rest", False):
        return
    orig = C.forward_cuda
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    def _decode_md(self):
        fc = get_forward_context()
        amd = fc.attn_metadata
        if not isinstance(amd, dict):
            return None
        m = amd.get(self.prefix)
        if m is None or getattr(m, "spec_sequence_masks", None) is not None:
            return None
        if m.num_prefills != 0 or m.num_decodes <= 0:
            return None
        if getattr(m, "non_spec_state_indices_tensor", None) is None:
            return None
        return m

    def _spec_md(self):
        """pure speculative-verify batch (every request has 1 + drafts tokens, no prefill and no
        plain decode): GDN metadata with per-position state slots, or None"""
        fc = get_forward_context()
        amd = fc.attn_metadata
        if not isinstance(amd, dict):
            return None
        m = amd.get(self.prefix)
        if m is None or getattr(m, "spec_sequence_masks", None) is None:
            return None
        if m.num_prefills != 0 or m.num_decodes != 0:
            return None
        si = m.spec_state_indices_tensor
        na = m.num_accepted_tokens
        qs = m.spec_query_start_loc
        if si is None or na is None or qs is None:
            return None
        if (si.dtype != torch.int32 or na.dtype != torch.int32 or qs.dtype != torch.int32
                or si.dim() != 2 or si.stride(1) != 1 or na.stride(0) != 1 or qs.stride(0) != 1
                or qs.size(0) < si.size(0) + 1 or na.size(0) < si.size(0)
                or si.size(0) * 16 > GDN_MAXM * 16):
            return None
        return m

    def forward_cuda(self, hidden_states):
        n_tok = hidden_states.size(0)
        ok = (hidden_states.dtype == torch.bfloat16 and n_tok <= GDN_MAXM and getattr(self, "tp_size", 1) == 1
              and not self.gqa_interleaved_layout and self.num_k_heads == 16
              and self.num_v_heads == 48 and self.head_k_dim == 128 and self.head_v_dim == 128
              and self.conv_kernel_size == 4 and self.activation in ("silu", "swish")
              and self.norm.activation in ("silu", "swish") and self.norm.group_size is None
              and self.norm.norm_before_gate and self.conv1d.bias is None)
        md = _decode_md(self) if ok else None
        smd = None
        if md is None and ok and GDN_SPEC:
            smd = _spec_md(self)
        if md is None and smd is None:
            return orig(self, hidden_states)
        cs = self.kv_cache[0]
        ssm = self.kv_cache[1]
        if is_conv_state_dim_first():
            cs_seq, cs_dim, cs_tok, cs_w = cs.stride(0), cs.stride(1), cs.stride(2), cs.size(2)
        else:
            cs_seq, cs_tok, cs_dim, cs_w = cs.stride(0), cs.stride(1), cs.stride(2), cs.size(1)
        if smd is not None and (ssm.dtype != torch.float32
                                or smd.spec_state_indices_tensor.size(1) + 2 > cs_w):
            return orig(self, hidden_states)
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
        z = mixed_qkvz[:, 10240:]
        b = ba[:, :48]
        a = ba[:, 48:]
        cw = self.conv1d.weight
        cw = cw.view(cw.size(0), cw.size(-1))
        out = torch.empty((n_tok, 48 * 128), dtype=hidden_states.dtype, device=hidden_states.device)
        c_qk = _gdn_scratch(hidden_states.device)
        if smd is not None:
            torch.ops.vopt_rest.gdn_spec(mixed_qkvz, z, b, a, cs, cs_seq, cs_dim, cs_tok, cs_w, cw,
                                         self.A_log, self.dt_bias, self.norm.weight, ssm,
                                         smd.spec_state_indices_tensor, smd.num_accepted_tokens,
                                         smd.spec_query_start_loc, out, c_qk,
                                         self.head_k_dim ** -0.5, self.norm.eps,
                                         _spec_flags(smd.spec_state_indices_tensor.size(0)))
        else:
            n_act = md.num_actual_tokens
            idx = md.non_spec_state_indices_tensor[:n_act]
            if not idx.is_contiguous():  # eager mode passes a strided block_table[:, 0] view
                idx = idx.contiguous()
            torch.ops.vopt_rest.gdn_decode(mixed_qkvz, z, b, a, cs, cs_seq, cs_dim, cs_tok, cw,
                                           self.A_log, self.dt_bias, self.norm.weight, ssm, idx, out,
                                           c_qk, self.head_k_dim ** -0.5, self.norm.eps)
        output, _ = self.out_proj(out)
        return output

    C.forward_cuda = forward_cuda
    C._vopt_rest = True
    _log(f"QwenGatedDeltaNetAttention decode patched (M <= {GDN_MAXM}; speculative verify: {GDN_SPEC})")


# ----------------------------------------------------------------------------- attn (verify)
_ATTN_BUF = {}
_ATTN_CAP_TOK, _ATTN_CAP_COLS = 64, 4096


def _attn_bufs(dev):
    b = _ATTN_BUF.get(dev)
    if b is None:
        b = (torch.zeros(_ATTN_CAP_TOK + 1, dtype=torch.int32, device=dev),
             torch.ones(_ATTN_CAP_TOK, dtype=torch.int32, device=dev),
             torch.zeros(_ATTN_CAP_TOK * _ATTN_CAP_COLS, dtype=torch.int32, device=dev))
        _ATTN_BUF[dev] = b
    return b


def install_attn_spec(mod):
    """Multi-query (speculative verify / draft prefill) attention, per-token exact: each of the
    n query tokens is presented to the stock Triton unified attention as its own 1-query sequence
    (cu_seqlens_q = 0..n, seqused_k = position+1, block-table row of its request), so it takes the
    same 3D split-softmax decode path, with the same segment split, as a 1-token decode step.
    Only for small batches (n <= seq_threshold_3D, max query len <= VOPT_ATTN_SPEC_MAXQ)."""
    if not ATTN_SPEC:
        return
    load_ext()
    import dataclasses
    I = mod.TritonAttentionImpl
    if getattr(I, "_vopt_rest", False):
        return
    orig = I.forward
    from vllm.forward_context import get_forward_context
    dec = mod.AttentionType.DECODER

    def _pseudo(md):
        fc = get_forward_context()
        cache = getattr(fc, "_vopt_attn_pseudo", None)
        if cache is None:
            cache = {}
            fc._vopt_attn_pseudo = cache
        ent = cache.get(id(md))
        if ent is not None and ent[0] is md:
            return ent[1]
        n = md.num_actual_tokens
        qsl, sl, bt = md.query_start_loc, md.seq_lens, md.block_table
        R = qsl.size(0) - 1
        if (R < 1 or sl.numel() < R or bt.dim() != 2 or bt.size(0) < R or bt.size(1) > _ATTN_CAP_COLS
                or qsl.dtype != torch.int32 or sl.dtype != torch.int32 or bt.dtype != torch.int32
                or qsl.stride(0) != 1 or sl.stride(0) != 1 or bt.stride(1) != 1):
            cache[id(md)] = (md, None)
            return None
        cu_b, su_b, bt_b = _attn_bufs(qsl.device)
        nbc = bt.size(1)
        cu, su, bto = cu_b[:n + 1], su_b[:n], bt_b[:n * nbc].view(n, nbc)
        torch.ops.vopt_rest.attn_pseudo_meta(qsl, sl, bt, R, cu, su, bto)
        pm = dataclasses.replace(md, query_start_loc=cu, seq_lens=su, block_table=bto, max_query_len=1)
        cache[id(md)] = (md, pm)
        return pm

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        md = attn_metadata
        if (md is not None and 1 < md.max_query_len <= ATTN_SPEC_MAXQ
                and md.num_actual_tokens <= min(md.seq_threshold_3D, _ATTN_CAP_TOK)
                and not md.use_cascade and md.causal is True
                and getattr(md, "mm_prefix_range_tensor", None) is None
                and getattr(md, "rswa_prefix_lens", None) is None
                and self.attn_type == dec and self.alibi_slopes is None
                and getattr(self, "sinks", None) is None and not self.logits_soft_cap
                and tuple(self.sliding_window) == (-1, -1)
                and getattr(self, "chunk_lookback", -1) in (-1, None)
                and not getattr(self, "_is_per_token_head_quant", False)
                and output_block_scale is None):
            pm = _pseudo(md)
            if pm is not None:
                attn_metadata = pm
        return orig(self, layer, query, key, value, kv_cache, attn_metadata, output,
                    output_scale, output_block_scale)

    I.forward = forward
    I._vopt_rest = True
    _log(f"TritonAttentionImpl patched: per-token exact multi-query attention (q_len <= {ATTN_SPEC_MAXQ})")


# ----------------------------------------------------------------------------- attn
def install_attn(mod):
    load_ext()
    A = mod.Qwen3NextAttention
    if getattr(A, "_vopt_rest", False):
        return
    orig_init = A.__init__
    fused_qk = os.environ.get("VOPT_FUSED_QK", "1") == "1"

    def __init__(self, *args, **kw):
        orig_init(self, *args, **kw)
        if fused_qk and not self.use_fused_qk_norm_rope_gate:
            from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
            re = self.rotary_emb
            mc = kw.get("model_config")
            mm = mc.multimodal_config if mc is not None else None
            text_only = mm is None or mm.language_model_only
            sec = getattr(re, "mrope_section", None)
            supports_mrope = bool(type(re) is MRotaryEmbedding and sec and len(sec) == 3
                                  and sum(sec) == re.rotary_dim // 2
                                  and getattr(re, "mrope_interleaved", False))
            supports_dtype = getattr(re, "dtype", None) in (torch.float16, torch.bfloat16)
            self.use_fused_qk_norm_rope_gate = bool(
                self.attn_output_gate and getattr(re, "is_neox_style", False) and supports_dtype
                and (text_only or supports_mrope))

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        attn_output = self.attn(q, k, v)
        if gate is not None:
            n = attn_output.shape[-1]
            if (attn_output.dim() == 2 and gate.dim() == 2 and _bf16_rows(attn_output, n)
                    and _bf16_rows(gate, n) and gate.size(0) == attn_output.size(0)):
                attn_output = torch.ops.vopt_rest.sigmoid_mul(attn_output, gate)
            else:
                attn_output = attn_output * torch.sigmoid(gate)
        output, _ = self.o_proj(attn_output)
        return output

    A.__init__ = __init__
    A.forward = forward
    A._vopt_rest = True
    _log(f"Qwen3NextAttention patched (fused qk-norm/rope/gate on ROCm: {fused_qk})")
