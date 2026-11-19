"""vopt_rest: post-import patch hooks for vLLM (role "rest": norms, activation, GDN decode,
attention prep). No files inside the image are changed.

A patch = (target module name, callable(module)); the callable runs right after the target
module finished executing for the first time, in whichever process imports it. Patch names
not listed here are ignored (owned by other vopt_* packages).
"""
import importlib.abc
import importlib.util
import sys


def _norms(mod):
    from . import patches
    patches.install_norms(mod)


def _act(mod):
    from . import patches
    patches.install_act(mod)


def _gdn(mod):
    from . import patches
    patches.install_gdn(mod)


def _attn(mod):
    from . import patches
    patches.install_attn(mod)


def _attn_spec(mod):
    from . import patches
    patches.install_attn_spec(mod)


PATCHES = {
    "norms": ("vllm.model_executor.layers.layernorm", _norms),
    "act": ("vllm.model_executor.layers.activation", _act),
    "gdn": ("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn", _gdn),
    "attn": [("vllm.model_executor.models.qwen3_next", _attn),
             ("vllm.v1.attention.backends.triton_attn", _attn_spec)],
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
    hooks = {}
    for n in [x.strip() for x in names.split(",") if x.strip()]:
        if n not in PATCHES:  # owned by another patch package
            continue
        ent = PATCHES[n]
        for tgt, cb in (ent if isinstance(ent, list) else [ent]):
            if tgt in sys.modules:  # already imported: patch now
                cb(sys.modules[tgt])
            else:
                hooks.setdefault(tgt, []).append(cb)
    if hooks:
        sys.meta_path.insert(0, _PostImportFinder(hooks))
