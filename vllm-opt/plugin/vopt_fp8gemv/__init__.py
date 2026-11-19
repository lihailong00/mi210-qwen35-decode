"""vopt_fp8gemv: post-import patch hooks for vLLM (no files inside the image are changed).

A patch = (target module name, callable(module)). The callable runs right after the target
module finished executing for the first time, in whichever process imports it (API server,
EngineCore). Patch names not listed here are ignored (owned by other vopt_* packages).
"""
import importlib.abc
import importlib.util
import sys


def _fp8gemv(mod):
    from . import fp8_gemv
    fp8_gemv.install(mod)


PATCHES = {
    # FP8 block-scaled linear: weight-only HIP GEMV for M <= VOPT_FP8_MAXM on a tiled weight
    # layout, tiled-layout Triton fallback for larger M
    "fp8gemv": ("vllm.model_executor.kernels.linear", _fp8gemv),
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
        if n not in PATCHES:
            continue
        tgt, cb = PATCHES[n]
        if tgt in sys.modules:
            cb(sys.modules[tgt])
        else:
            hooks.setdefault(tgt, []).append(cb)
    if hooks:
        sys.meta_path.insert(0, _PostImportFinder(hooks))
