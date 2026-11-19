"""Build/load the vopt_spec HIP extension (torch.ops.vopt_spec.*). The source is mounted read-only;
it is copied into $TORCH_EXTENSIONS_DIR/vopt_spec and built there (cached)."""
import os
import shutil
import sys

_LOADED = False


def load_ext():
    global _LOADED
    if _LOADED:
        return
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx90a")
    from torch.utils.cpp_extension import load
    src = os.path.join(os.path.dirname(__file__), "csrc", "spec_ops.cu")
    bd = os.path.join(os.environ.get("TORCH_EXTENSIONS_DIR", "/cache/torch_ext"), "vopt_spec")
    os.makedirs(bd, exist_ok=True)
    dst = os.path.join(bd, "spec_ops.cu")
    if not os.path.exists(dst) or open(dst, "rb").read() != open(src, "rb").read():
        shutil.copyfile(src, dst)
    load(name="vopt_spec_ext", sources=[dst], build_directory=bd, is_python_module=False,
         extra_cuda_cflags=["-O3", "--offload-arch=gfx90a"], verbose=False)
    _LOADED = True
    print("[vopt_spec] extension loaded", file=sys.stderr)
