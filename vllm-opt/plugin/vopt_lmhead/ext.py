"""Build/load the vopt_lmhead HIP extension (torch.ops.vopt_lmhead.*) with torch.utils.cpp_extension.
The source is mounted read-only; it is copied into $TORCH_EXTENSIONS_DIR/vopt_lmhead (hipify writes
next to the source) and the build is cached there."""
import os
import shutil
import sys

_LOADED = False


def load_ext():
    global _LOADED
    if _LOADED:
        return
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx90a")  # only MI210 (default builds 9 archs)
    from torch.utils.cpp_extension import load
    src = os.path.join(os.path.dirname(__file__), "csrc", "bf16_gemv.cu")
    bd = os.path.join(os.environ.get("TORCH_EXTENSIONS_DIR", "/cache/torch_ext"), "vopt_lmhead")
    os.makedirs(bd, exist_ok=True)
    dst = os.path.join(bd, "bf16_gemv.cu")
    if not os.path.exists(dst) or open(dst, "rb").read() != open(src, "rb").read():
        shutil.copyfile(src, dst)
    load(name="vopt_lmhead_ext", sources=[dst], build_directory=bd, is_python_module=False,
         extra_cuda_cflags=["-O3", "--offload-arch=gfx90a"], verbose=False)
    _LOADED = True
    print("[vopt_lmhead] extension loaded", file=sys.stderr)
