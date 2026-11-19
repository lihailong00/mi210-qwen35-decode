"""Build/load the vopt_rest HIP extension (torch.utils.cpp_extension, cached in
$TORCH_EXTENSIONS_DIR/vopt_rest_ops). Registers torch.ops.vopt_rest.*"""
import os
import shutil
import sys

_LOADED = False


def load_ext():
    global _LOADED
    if _LOADED:
        return
    import torch  # noqa: F401
    old_arch = os.environ.get("PYTORCH_ROCM_ARCH")
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx90a"  # build for this card only (DPP row bcast is gfx9)
    from torch.utils.cpp_extension import load
    src = os.path.join(os.path.dirname(__file__), "csrc", "rest_ops.cu")
    bd = os.path.join(os.environ.get("TORCH_EXTENSIONS_DIR", "/cache/torch_ext"), "vopt_rest_ops_mtp")
    os.makedirs(bd, exist_ok=True)
    dst = os.path.join(bd, "rest_ops.cu")  # hipify writes next to the source: build from a copy
    if not os.path.exists(dst) or open(dst, "rb").read() != open(src, "rb").read():
        shutil.copyfile(src, dst)
    load(name="vopt_rest_ops_mtp", sources=[dst], build_directory=bd, is_python_module=False,
         extra_cuda_cflags=["-O3", "--offload-arch=gfx90a"], verbose=False)
    if old_arch is None:
        os.environ.pop("PYTORCH_ROCM_ARCH", None)
    else:
        os.environ["PYTORCH_ROCM_ARCH"] = old_arch
    _LOADED = True
    print("[vopt_rest] rest_ops extension loaded", file=sys.stderr)
