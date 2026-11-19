"""vopt_spec: speculative-decoding (MTP) patches for the vLLM v2 GPU model runner.

Installed by usercustomize.py when "vopt_spec" is in VOPT_PKGS (and VOPT_PATCHES is set). Every
piece only acts when MTP / spec decode is configured; plain decode is untouched.

  draft head  (VOPT_DRAFT_HEAD, default int4)
      DraftModelSpeculator.sample_draft for greedy drafting (draft_sample_method greedy, no
      local-argmax, no acceptance estimator): argmax over a row subset of the shared lm_head
      instead of full logits + torch argmax.
        int4 - subset rows [0, VOPT_DRAFT_VOCAB) plus the tokenizer's added tokens, re-quantized
               at load time to asymmetric int4 per 128-column group (draft-only copy, ~0.39 GiB
               for 151.6k rows); fused GEMV + top-4 per wave (vopt_spec i4_head_argmax), then the
               rows within VOPT_DRAFT_DELTA (default 3.0) of the int4 max are rescored with the
               bf16 lm_head, so the draft is the bf16-subset argmax unless the int4 logit error
               exceeds delta/2 (measured max 1.4). M <= 2 per launch (more rows: 2-row launches).
        fp8  - the same subset as FP8 e4m3 with one scale per row (~0.74 GiB), rescoring window
               default 1.0 (measured max fp8 logit error 0.46); M <= 4 per launch.
        bf16 - first VOPT_DRAFT_VOCAB rows of the bf16 lm_head (a view, no copy) via the
               vopt_lmhead GEMV with argmax epilogue.
        full - full bf16 lm_head via the vopt_lmhead GEMV argmax (only drops torch's argmax).
        stock - unpatched.
      More than 8 draft rows (large batches): bf16 subset through torch.matmul + argmax.
      The draft only decides which tokens are proposed; with greedy verification the output is
      the target's argmax whatever the draft, so this can change acceptance, not the output.
  greedy verify fast path (VOPT_REJECT_FAST, default 1)
      RejectionSampler.__call__ when every request is greedy and needs no logits processing,
      no logprobs / watermark / adaptive / synthetic verification, and the logits come straight
      from the vopt_lmhead GEMV with argmax epilogue: accept/reject from that argmax (same kernel
      and tie rule as the K=0 greedy path) in one tiny kernel, replacing the 5 Triton rejection
      kernels. Same semantics as vLLM's greedy rejection (tested with planted ties).

  GDN metadata fast path (VOPT_GDN_MD, default 1; see gdn_md.py)
      one Triton kernel per mamba group instead of ~19 small ops when building the GDN
      attention metadata of a pure spec-decode batch (same persistent buffers, same values).

Env: VOPT_SPEC=0 disables the package; VOPT_DRAFT_HEAD=int4|fp8|bf16|full|stock;
VOPT_DRAFT_VOCAB (default 151552); VOPT_DRAFT_EXTRA=auto|none|"a-b,c" (extra vocab ids for the
int4/fp8 subset; auto = the tokenizer's added tokens); VOPT_DRAFT_DELTA (rescoring window; 0 = off;
default 3.0 int4, 1.0 fp8); VOPT_REJECT_FAST=0|1; VOPT_GDN_MD=0|1 (VOPT_GDN_MD_CHECK=1 compares).
"""
import importlib.abc
import importlib.util
import json
import os
import sys

ON = os.environ.get("VOPT_SPEC", "1") == "1"
HEAD = os.environ.get("VOPT_DRAFT_HEAD", "int4")
VOCAB = int(os.environ.get("VOPT_DRAFT_VOCAB", "151552"))
EXTRA = os.environ.get("VOPT_DRAFT_EXTRA", "auto")
REJECT_FAST = os.environ.get("VOPT_REJECT_FAST", "1") == "1"
DELTA = float(os.environ.get("VOPT_DRAFT_DELTA", "3.0" if HEAD == "int4" else "1.0"))  # bf16 rescoring window
QMAXM = 8      # quantized heads: up to 8 rows (fp8: 4 rows per launch, int4: 2 rows per launch)
GEMV_MAXM = int(os.environ.get("VOPT_LMHEAD_MAXM", "8"))


def _log(msg):
    print(f"[vopt_spec] {msg}", file=sys.stderr)


def _extra_ids(model_config, V):
    if EXTRA == "none":
        return []
    ids = []
    if EXTRA == "auto":
        try:
            path = os.path.join(model_config.tokenizer, "tokenizer.json")
            ids = [t["id"] for t in json.load(open(path))["added_tokens"]]
        except Exception as e:  # tokenizer not a local dir: no extra ids
            _log(f"added tokens not found ({e!r}); fp8 subset = prefix only")
    else:
        for part in EXTRA.split(","):
            if "-" in part:
                a, b = part.split("-")
                ids += list(range(int(a), int(b) + 1))
            elif part.strip():
                ids.append(int(part))
    return sorted(set(i for i in ids if VOCAB <= i < V))


class _Head:
    def __init__(self, spec, w):
        import torch
        self.w = w
        V = w.shape[0]
        self.n = min(VOCAB, V)
        self.mode = HEAD
        self.q = self.s = self.idmap = None
        if self.mode in ("fp8", "int4"):
            from . import ext
            ext.load_ext()
            extra = _extra_ids(spec.vllm_config.model_config, V)
            rows = torch.cat([torch.arange(self.n, dtype=torch.long, device=w.device),
                              torch.tensor(extra, dtype=torch.long, device=w.device)])
            if self.mode == "fp8":
                self.q, self.s = torch.ops.vopt_spec.quantize_rows_fp8(w, rows, 0)
                mib = (self.q.numel() + self.s.numel() * 4) / 2**20
            else:
                from .quant import quant_int4
                self.q, self.s = quant_int4(w, rows)
                mib = (self.q.numel() + self.s.numel()) * 4 / 2**20
                torch.cuda.empty_cache()
            self.idmap = rows
            self.nrows = rows.numel()
            _log(f"draft head {self.mode}: {rows.numel()} rows ({self.n} prefix + {len(extra)} added tokens), "
                 f"{mib:.0f} MiB, rescoring window {DELTA}")
        else:
            _log(f"draft head {self.mode}: {self.n if self.mode == 'bf16' else V} rows")
        from vopt_lmhead import ext as lext  # bf16 GEMV with argmax epilogue
        lext.load_ext()

    def __call__(self, x):
        import torch
        if x.dim() != 2 or x.dtype != torch.bfloat16 or x.shape[1] != 5120 or not x.is_cuda:
            return None
        if x.stride(1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16:
            x = x.contiguous()
        M = x.shape[0]
        if M == 0:
            return None
        ops = torch.ops.vopt_spec
        if self.mode == "int4" and M <= QMAXM:
            if M <= 2:
                return ops.i4_head_argmax(x, self.q, self.s, self.nrows, self.idmap, self.w, DELTA, -1, 0)
            return torch.cat([ops.i4_head_argmax(x[i:i + 2], self.q, self.s, self.nrows, self.idmap, self.w,
                                                 DELTA, -1, 0) for i in range(0, M, 2)])
        if self.mode == "fp8" and M <= QMAXM:
            if M <= 4:
                return ops.fp8_head_argmax(x, self.q, self.s, self.idmap, self.w, DELTA, -1, 0)
            return torch.cat([ops.fp8_head_argmax(x[i:i + 4], self.q, self.s, self.idmap, self.w, DELTA, -1, 0)
                              for i in range(0, M, 4)])
        w = self.w if self.mode == "full" else self.w[: self.n]  # (int4/fp8 with > 8 rows: bf16 prefix)
        if M <= GEMV_MAXM:
            return torch.ops.vopt_lmhead.bf16_gemv_argmax(x, w, -1, 0)[1]
        if self.mode == "full":
            return None
        return torch.matmul(x, w.t()).argmax(dim=-1)


def _speculator(mod):
    cls = mod.DraftModelSpeculator
    orig_load = cls.load_model
    orig_sample = cls.sample_draft

    def load_model(self, target_model):
        orig_load(self, target_model)
        self._vopt_head = None
        try:
            if HEAD != "stock" and self.speculative_config.draft_sample_method != "probabilistic":
                w = self.model.lm_head.weight
                import torch
                if w.dtype == torch.bfloat16 and w.dim() == 2 and w.shape[1] == 5120 and w.is_contiguous():
                    self._vopt_head = _Head(self, w)
                else:
                    _log(f"draft head: unsupported lm_head {tuple(w.shape)} {w.dtype}; stock path")
        except Exception as e:
            _log(f"draft head disabled: {e!r}")
            self._vopt_head = None

    def sample_draft(self, hidden_states, sample_src_positions, idx_mapping, temperature, seeds,
                     draft_step, draft_logits):
        h = getattr(self, "_vopt_head", None)
        if (h is not None and draft_logits is None and not self.use_local_argmax_reduction
                and self.acceptance_estimator is None):
            r = h(hidden_states)
            if r is not None:
                return r
        return orig_sample(self, hidden_states, sample_src_positions, idx_mapping, temperature, seeds,
                           draft_step, draft_logits)

    cls.load_model = load_model
    cls.sample_draft = sample_draft
    _log(f"draft head patched (mode={HEAD}, vocab={VOCAB})")


def _rejection(mod):
    import numpy as np
    import torch
    from . import ext
    ext.load_ext()
    from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
    cls = mod.RejectionSampler
    orig = cls.__call__
    get_ns = mod.get_num_sampled_and_rejected
    SamplerOutput = mod.SamplerOutput
    try:
        import vopt_lmhead
        lstate = vopt_lmhead._state
    except Exception:
        lstate = None
    stat = {"fast": 0, "slow": 0}

    def __call__(self, logits, input_batch, draft_logits=None):
        st = lstate.get("stash") if lstate is not None else None
        if lstate is not None:
            lstate["stash"] = None
        smp = self.sampler
        if (st is not None and draft_logits is None and logits.dim() == 2
                and st[0] == logits.data_ptr() and st[1] == logits.shape[0]
                and not smp.compute_nans and self.watermark_key is None
                and not self.enable_adaptive_verification
                and self.synthetic_conditional_rates is None
                and input_batch.logits_indices.dtype == torch.int64
                and input_batch.cu_num_logits.dtype == torch.int32
                and input_batch.cu_num_logits.shape[0] == input_batch.num_reqs + 1
                and logits.shape[0] == input_batch.logits_indices.shape[0]):
            im = input_batch.idx_mapping_np
            if (np.all(smp.sampling_states.temperature.np[im] == 0.0)
                    and not np.any(smp.needs_logits_processing[im])
                    and smp.sampling_states.max_num_logprobs(im) == NO_LOGPROBS):
                sampled, num_sampled = torch.ops.vopt_spec.greedy_verify(
                    st[2], input_batch.input_ids, input_batch.logits_indices,
                    input_batch.cu_num_logits, self.num_speculative_steps)
                num_sampled, num_rejected = get_ns(
                    num_sampled, input_batch.seq_lens, input_batch.cu_num_logits,
                    input_batch.idx_mapping, smp.req_states.prefill_len.gpu)
                stat["fast"] += 1
                return SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None, num_nans=None,
                                     num_sampled=num_sampled, num_rejected=num_rejected)
        stat["slow"] += 1
        if stat["slow"] in (1, 10, 100, 1000):
            _log(f"greedy verify fast path: {stat['fast']} fast / {stat['slow']} stock calls")
        return orig(self, logits, input_batch, draft_logits)

    cls.__call__ = __call__
    _log("greedy verify fast path patched")


class _PostImportFinder(importlib.abc.MetaPathFinder):
    def __init__(self, hooks):
        self.hooks = hooks
        self.busy = set()

    def find_spec(self, name, path=None, target=None):
        if name not in self.hooks or name in self.busy:
            return None
        self.busy.add(name)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self.busy.discard(name)
        if spec is None or spec.loader is None:
            return spec
        orig = spec.loader.exec_module
        cbs = self.hooks[name]

        def exec_module(module, _orig=orig, _cbs=cbs):
            _orig(module)
            for cb in _cbs:
                try:
                    cb(module)
                except Exception as e:  # never break vLLM import
                    _log(f"patch {cb.__name__} failed: {e!r}")

        spec.loader.exec_module = exec_module
        return spec


def install_hooks(names):
    """names (VOPT_PATCHES) are ignored: this package is controlled by its own env switches."""
    if not ON:
        return
    hooks = {}
    if HEAD != "stock":
        hooks.setdefault("vllm.v1.worker.gpu.spec_decode.speculator", []).append(_speculator)
    if REJECT_FAST:
        hooks.setdefault("vllm.v1.worker.gpu.spec_decode.rejection_sampler", []).append(_rejection)
    if os.environ.get("VOPT_GDN_MD", "1") == "1":
        def _gdn_md(mod):
            from . import gdn_md
            gdn_md.install(mod)
        hooks.setdefault("vllm.v1.attention.backends.gdn_attn", []).append(_gdn_md)
    for tgt in list(hooks):
        if tgt in sys.modules:
            for cb in hooks.pop(tgt):
                cb(sys.modules[tgt])
    if hooks:
        sys.meta_path.insert(0, _PostImportFinder(hooks))
