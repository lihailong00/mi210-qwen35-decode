"""vopt_lmhead: post-import patch hooks for vLLM (no files inside the image are changed).

Patches (enable with VOPT_PATCHES=<comma list>, package listed in VOPT_PKGS):
  lmhead   - lm_head (UnquantizedEmbeddingMethod.apply) -> torch.ops.vopt_lmhead.bf16_gemv for
             bf16, K=5120, 1 <= M <= VOPT_LMHEAD_MAXM (default 8; 6 in deploy v1); otherwise stock path.
  greedy   - (needs lmhead) the lm_head GEMV also produces the argmax of its bf16 logits; the
             v1 GPU Sampler uses it instead of the gumbel/argmax kernels when every request in the
             batch is greedy (temperature 0) and needs no logits processing, and no grammar
             bitmask was applied.
  topk     - (needs lmhead,greedy) fused top-k (<= 64) / top-p / temperature sampler
             (torch.ops.vopt_lmhead.sample_topk_topp) for batches whose requests use only those
             parameters (no penalties, bias, bad words, min_p, explicit seed, processed logprobs).
  bf16gemv - small bf16 linears (UnquantizedLinearMethod.apply, e.g. linear_attn.in_proj_ba
             96x5120) -> split-K bf16_gemv for M <= VOPT_LMHEAD_MAXM; captured in decode graphs.
             Also the MTP draft fc (5120x10240, K=10240) unless VOPT_FC_GEMV=0 (draft-only layer).
Unknown patch names are ignored (owned by other packages).
"""
import importlib.abc
import importlib.util
import os
import sys

MAXM = int(os.environ.get("VOPT_LMHEAD_MAXM", "8"))
TOPK_MAXM = 6  # fused top-k sampler: logits of one GEMV launch (M <= 6)
CFG_BIG = int(os.environ.get("VOPT_LMHEAD_CFG", "-1"))
CFG_SMALL = int(os.environ.get("VOPT_BF16_SMALL_CFG", "-1"))
# MTP draft fc (bf16 5120 x 10240): split-K GEMV instead of wvSplitK (draft-only layer)
FC_GEMV = os.environ.get("VOPT_FC_GEMV", "1") == "1"
CFG_FC = int(os.environ.get("VOPT_FC_CFG", "100"))
FC_MAXM = int(os.environ.get("VOPT_FC_MAXM", "2"))  # measured: M<=2 71/82 us vs wvSplitK 88; M=4 no gain

_state = {"greedy": False, "stash": None, "grammar": False, "topk": False}
TOPK_MAX = 64


def _ok_x(x, K):
    return (x.dim() == 2 and x.dtype.itemsize == 2 and x.shape[1] == K and 1 <= x.shape[0] <= MAXM
            and x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0)


def _lmhead(mod):
    import torch
    from . import ext
    ext.load_ext()
    ops = torch.ops.vopt_lmhead
    cls = mod.UnquantizedEmbeddingMethod
    orig = cls.apply

    def apply(self, layer, x, bias=None):
        _state["stash"] = None
        w = layer.weight
        if (bias is None and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and x.is_cuda
                and w.dim() == 2 and w.shape[1] == 5120 and w.shape[0] > 4096 and w.stride(1) == 1
                and w.stride(0) % 8 == 0 and _ok_x(x, 5120)):
            if _state["greedy"] and not _state["grammar"]:
                y, idx = ops.bf16_gemv_argmax(x, w, CFG_BIG, 0)
                _state["stash"] = (y.data_ptr(), y.shape[0], idx)
                return y
            return ops.bf16_gemv(x, w, CFG_BIG, 0)
        return orig(self, layer, x, bias)

    cls.apply = apply
    print(f"[vopt_lmhead] lm_head patched (MAXM={MAXM}, greedy={_state['greedy']})", file=sys.stderr)


def _greedy_runner(mod):
    _state["greedy"] = True
    cls = mod.GPUModelRunner
    orig = cls.sample

    def sample(self, hidden_states, input_batch, grammar_output):
        _state["grammar"] = grammar_output is not None
        return orig(self, hidden_states, input_batch, grammar_output)

    cls.sample = sample


def _dev_params(self, device):
    """Device-resident copies of temperature / top_k / top_p / seeds (vLLM keeps them in
    host-mapped UVA memory; reading those from the sampler kernel costs PCIe round trips every
    step). Refreshed with one small H2D copy per array only when the host values change."""
    import numpy as np
    import torch
    ss = self.sampling_states
    srcs = (ss.temperature, ss.top_k, ss.top_p, ss.seeds)
    cache = getattr(self, "_vopt_dev", None)
    if cache is None or cache[0][0].device != device:
        cache = [(torch.empty(s_.np.shape, dtype=s_.dtype, device=device), np.full_like(s_.np, -1))
                 for s_ in srcs]
        self._vopt_dev = cache
    for (d, snap), s_ in zip(cache, srcs):
        if not np.array_equal(snap, s_.np):
            snap[:] = s_.np
            # fresh pinned buffer from torch's caching host allocator (kept alive until the async
            # copy has executed), so no host sync and no overwrite of an in-flight staging buffer
            d.copy_(torch.from_numpy(snap.copy()).pin_memory(), non_blocking=True)
    return tuple(d for d, _ in cache)


def _greedy_sampler(mod):
    import numpy as np
    import torch
    _state["greedy"] = True
    cls = mod.Sampler
    orig = cls.sample
    orig_add = cls.add_request
    processed_modes = getattr(mod, "PROCESSED_LOGPROBS_MODES", ())

    def add_request(self, req_idx, prompt_len, sampling_params):
        orig_add(self, req_idx, prompt_len, sampling_params)
        if not hasattr(self, "_vopt_other"):
            self._vopt_other = np.zeros(self.needs_logits_processing.shape[0], dtype=bool)
        st = self.sampling_states
        # logits processing other than temperature / top_k / top_p (the fused sampler's scope)
        self._vopt_other[req_idx] = bool(
            self.logit_bias_state.use_logit_bias[req_idx]
            or self.penalties_state.use_penalty[req_idx]
            or self.bad_words_state.num_bad_words.np[req_idx] > 0
            or (self.thinking_budget_state.enabled
                and self.thinking_budget_state.use_thinking_budget[req_idx])
            or st.min_p.np[req_idx] != 0.0
            or st.seeds_set[req_idx])

    def sample(self, logits, expanded_idx_mapping, idx_mapping, idx_mapping_np, pos, input_ids,
               expanded_local_pos, return_logprobs=False):
        st = _state["stash"]
        _state["stash"] = None
        if (st is not None and not _state["grammar"] and logits.dim() == 2
                and st[0] == logits.data_ptr() and st[1] == logits.shape[0]
                and logits.shape[0] == idx_mapping_np.shape[0]
                and self.trace_replay_state is None and not self.return_sampling_mask):
            temps = self.sampling_states.temperature.np[idx_mapping_np]
            if np.all(temps == 0.0) and not np.any(self.needs_logits_processing[idx_mapping_np]):
                return st[2], logits
            other = getattr(self, "_vopt_other", None)
            if (_state["topk"] and other is not None and logits.shape[0] <= TOPK_MAXM
                    and not np.any(other[idx_mapping_np])
                    and not (return_logprobs and self.logprobs_mode in processed_modes)):
                ks = self.sampling_states.top_k.np[idx_mapping_np]
                if np.all((temps == 0.0) | ((ks >= 1) & (ks <= TOPK_MAX))):
                    out = torch.ops.vopt_lmhead.sample_topk_topp(
                        logits, st[2], expanded_idx_mapping, *_dev_params(self, logits.device), pos)
                    return out, logits
        return orig(self, logits, expanded_idx_mapping, idx_mapping, idx_mapping_np, pos,
                    input_ids, expanded_local_pos, return_logprobs)

    cls.add_request = add_request
    cls.sample = sample


def _bf16gemv(mod):
    import torch
    from . import ext
    ext.load_ext()
    ops = torch.ops.vopt_lmhead
    cls = mod.UnquantizedLinearMethod
    orig = cls.apply

    def apply(self, layer, x, bias=None):
        w = layer.weight
        if (bias is None and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and x.is_cuda
                and w.dim() == 2 and w.shape[1] == 5120 and w.shape[0] <= 4096 and w.stride(1) == 1
                and w.stride(0) % 8 == 0 and w.data_ptr() % 16 == 0):
            x2 = x.reshape(-1, x.shape[-1])
            if _ok_x(x2, 5120):
                return ops.bf16_gemv(x2, w, CFG_SMALL, 0).view(*x.shape[:-1], w.shape[0])
        elif (FC_GEMV and bias is None and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
                and x.is_cuda and w.dim() == 2 and w.shape[1] == 10240 and w.shape[0] <= 8192
                and w.stride(1) == 1 and w.stride(0) % 8 == 0 and w.data_ptr() % 16 == 0):
            x2 = x.reshape(-1, x.shape[-1])
            if _ok_x(x2, 10240) and x2.shape[0] <= FC_MAXM:
                return ops.bf16_gemv(x2, w, CFG_FC, 0).view(*x.shape[:-1], w.shape[0])
        return orig(self, layer, x, bias)

    cls.apply = apply
    print(f"[vopt_lmhead] small bf16 linears patched (MAXM={MAXM})", file=sys.stderr)


PATCHES = {
    "lmhead": [("vllm.model_executor.layers.vocab_parallel_embedding", _lmhead)],
    "greedy": [("vllm.v1.worker.gpu.model_runner", _greedy_runner),
               ("vllm.v1.worker.gpu.sample.sampler", _greedy_sampler)],
    "topk": [],  # flag only (see install_hooks); implemented in the greedy sampler patch
    "bf16gemv": [("vllm.model_executor.layers.linear", _bf16gemv)],
}


class _PostImportFinder(importlib.abc.MetaPathFinder):
    def __init__(self, hooks):
        self.hooks = hooks  # module name -> [callables]
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
                cb(module)

        spec.loader.exec_module = exec_module
        return spec


def install_hooks(names):
    names = [x.strip() for x in names.split(",") if x.strip()]
    if "greedy" in names:
        _state["greedy"] = True  # lm_head apply checks it at call time
    if "topk" in names:
        _state["topk"] = True
    hooks = {}
    for n in names:
        if n not in PATCHES:  # owned by another patch package
            continue
        for tgt, cb in PATCHES[n]:
            if tgt in sys.modules:  # already imported: patch now
                cb(sys.modules[tgt])
            else:
                hooks.setdefault(tgt, []).append(cb)
    if hooks:
        sys.meta_path.insert(0, _PostImportFinder(hooks))
