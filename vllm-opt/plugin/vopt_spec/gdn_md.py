"""Fast GDN attention-metadata build for pure speculative-decode batches (VOPT_GDN_MD, default 1).

vLLM builds GDNAttentionMetadata once per mamba KV-cache group (3 groups for this model) before
every verify step. For a batch where every real request is a spec decode (K+1 tokens), padded
requests sit at the back and the FULL cudagraph path is taken, stock build() runs ~19 small GPU
ops per group (mask H2D copy, align-mode block-table gather, boolean-mask indexing, arange, and
copies/fills into the persistent cudagraph buffers). This writes the same persistent buffers with
one Triton kernel per group and returns the same metadata object; anything else (prefills, mixed
batches, non-prefix masks, eager-size batches, cudagraph disabled) goes to the stock build().

VOPT_GDN_MD_CHECK=1: run both and compare every field (debug; prints mismatches).
"""
import os
import sys

ON = os.environ.get("VOPT_GDN_MD", "1") == "1"
CHECK = os.environ.get("VOPT_GDN_MD_CHECK", "0") == "1"


def _log(msg):
    print(f"[vopt_spec] {msg}", file=sys.stderr)


def install(mod):
    import torch
    from vllm.triton_utils import tl, triton
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    @triton.jit
    def _gdn_spec_md_kernel(bt_ptr, bt_stride, seq_lens_ptr, qsl_ptr, na_ptr,
                            st_ptr, st_stride, mask_ptr, tok_ptr, sqsl_ptr, nacc_ptr,
                            n, S, block_size, null_id,
                            ALIGN: tl.constexpr, T: tl.constexpr, TP: tl.constexpr):
        i = tl.program_id(0)
        j = tl.arange(0, TP)
        jm = j < T
        valid = i < n
        if ALIGN:
            sl = tl.load(seq_lens_ptr + i)
            start = tl.maximum((sl - 1) // block_size, 0)
        else:
            start = 0
        b = tl.load(bt_ptr + i * bt_stride + start + j, mask=jm & valid, other=0)
        tl.store(st_ptr + i * st_stride + j, tl.where(valid, b, null_id), mask=jm)
        tl.store(mask_ptr + i, valid)
        na = tl.load(na_ptr + i, mask=valid, other=1)
        tl.store(nacc_ptr + i, tl.where(valid, na, 1))
        q = tl.load(qsl_ptr + tl.minimum(i + 1, n))
        tl.store(sqsl_ptr + i + 1, q)
        if i == 0:
            tl.store(sqsl_ptr, tl.load(qsl_ptr))
        t = i * T + j
        tl.store(tok_ptr + t, t, mask=jm & (t < S))

    cls = mod.GDNAttentionMetadataBuilder
    orig = cls.build
    Meta = mod.GDNAttentionMetadata
    stat = {"fast": 0, "slow": 0, "bad": 0}

    def fast(self, m, num_accepted_tokens, ndd_cpu):
        if not (self.use_spec_decode and self.use_full_cuda_graph and ndd_cpu is not None
                and num_accepted_tokens is not None):
            return None
        mode = self.vllm_config.cache_config.mamba_cache_mode
        if mode not in ("all", "none", "align"):
            return None
        bs = m.num_reqs
        mask_cpu = ndd_cpu >= 0
        n = int(mask_cpu.sum())
        if n == 0 or n > bs or ndd_cpu.shape[0] != bs or not bool(mask_cpu[:n].all()):
            return None
        if int(ndd_cpu[:n].sum()) == 0:
            return None
        qsl_cpu = m.query_start_loc_cpu
        ql = qsl_cpu[1:bs + 1] - qsl_cpu[:bs]
        if bool((ql[n:] != 0).any()):  # non-spec rows must be zero-length padding
            return None
        nsdt = int(ql[:n].sum())
        T = self.num_spec + 1
        S = min(n * T, int(qsl_cpu[-1]))
        if n > self.decode_cudagraph_max_bs or nsdt > self.decode_cudagraph_max_bs or bs > self.decode_cudagraph_max_bs:
            return None
        bt = m.block_table_tensor
        align = mode == "align"
        if bt.dtype != torch.int32 or bt.stride(1) != 1 or (not align and bt.shape[1] < T):
            return None
        if align and (1 + self.kv_cache_spec.num_speculative_blocks) < T:
            return None
        st = self.spec_state_indices_tensor
        _gdn_spec_md_kernel[(bs,)](
            bt, bt.stride(0), m.seq_lens, m.query_start_loc, num_accepted_tokens,
            st, st.stride(0), self.spec_sequence_masks, self.spec_token_indx,
            self.spec_query_start_loc, self.num_accepted_tokens,
            n, S, self.kv_cache_spec.block_size if align else 1, NULL_BLOCK_ID,
            ALIGN=align, T=T, TP=triton.next_power_of_2(T), num_warps=1)
        return Meta(
            num_prefills=0, num_prefill_tokens=0, num_decodes=0, num_decode_tokens=0,
            num_spec_decodes=n, num_spec_decode_tokens=nsdt, num_actual_tokens=m.num_actual_tokens,
            has_initial_state=None, chunk_indices=None, chunk_offsets=None,
            prefill_query_start_loc=None, prefill_state_indices=None, prefill_has_initial_state=None,
            spec_query_start_loc=self.spec_query_start_loc[:bs + 1], non_spec_query_start_loc=None,
            spec_state_indices_tensor=st[:bs], non_spec_state_indices_tensor=None,
            spec_sequence_masks=self.spec_sequence_masks[:bs], spec_token_indx=self.spec_token_indx[:S],
            non_spec_token_indx=self.non_spec_token_indx[:0],
            num_accepted_tokens=self.num_accepted_tokens[:bs],
            nums_dict=None, batch_ptr=None, token_chunk_offset_ptr=None)

    def snapshot(md):
        import dataclasses
        out = {}
        for f in dataclasses.fields(md):
            v = getattr(md, f.name)
            out[f.name] = v.clone() if isinstance(v, torch.Tensor) else v
        return out

    def build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens=None,
              num_decode_draft_tokens_cpu=None, fast_build=False):
        md = None
        try:
            md = fast(self, common_attn_metadata, num_accepted_tokens, num_decode_draft_tokens_cpu)
        except Exception as e:  # never break the step: stock path
            if stat["slow"] < 3:
                _log(f"gdn metadata fast path error {e!r}; stock build")
            md = None
        if md is not None and CHECK:
            a = snapshot(md)
            ref = orig(self, common_prefix_len, common_attn_metadata, num_accepted_tokens,
                       num_decode_draft_tokens_cpu, fast_build)
            b = snapshot(ref)
            for k in a:
                x, y = a[k], b[k]
                same = (torch.equal(x, y) if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
                        else x == y)
                if not same:
                    stat["bad"] += 1
                    if stat["bad"] <= 20:
                        _log(f"gdn metadata CHECK mismatch {k}: fast={x} stock={y}")
            if stat["fast"] % 200 == 0:
                _log(f"gdn metadata CHECK: {stat['fast'] + 1} compared, {stat['bad']} field mismatches")
            stat["fast"] += 1
            return ref
        if md is not None:
            stat["fast"] += 1
            return md
        stat["slow"] += 1
        return orig(self, common_prefix_len, common_attn_metadata, num_accepted_tokens,
                    num_decode_draft_tokens_cpu, fast_build)

    cls.build = build
    _log(f"gdn spec metadata fast path patched (check={CHECK})")
