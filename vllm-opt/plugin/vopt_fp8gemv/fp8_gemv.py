"""Weight-only FP8 block-scaled GEMV for small M on MI210 (gfx90a), as a vLLM linear kernel.

torch.ops.vopt_fp8gemv.fp8_gemv(x bf16 [M,K], w_tiled uint8 [N,K], s fp32 [N/128,K/128],
                                ws, cnt, R, S, B) -> bf16 [M,N]
built with torch.utils.cpp_extension inside the vLLM image (cached in $TORCH_EXTENSIONS_DIR).

Integration: VoptFp8GemvKernel(TritonFp8BlockScaledMMKernel) is put first in vLLM's ROCm FP8
block kernel list. process_weights_after_loading re-lays the fp8 bytes out losslessly into
[N/64][K/16][64][16] tiles (original freed). apply_weights: M <= VOPT_FP8_MAXM -> HIP GEMV
(graph-capture safe); larger M -> stock dynamic activation quant + a copy of the reference
exact-decode Triton kernel that addresses the tiled layout (tiled_triton.py).
"""
import os
import shutil
import sys

import torch

_LOADED = False
MAXM = int(os.environ.get("VOPT_FP8_MAXM", "8"))
# VOPT_FP8_EXACT=1 (default): every M of a shape uses the shape's M=1 split-K (S, B) and M >= 2 runs
# the MFMA multi-column kernel, so each output column is bitwise equal to the M=1 call on that row
# of x (speculative verify batches reproduce plain decode). 0: deploy-v1 kernels and per-M table.
EXACT = os.environ.get("VOPT_FP8_EXACT", "1") == "1"

# (N, K) -> {M: (R, S, B)}; M not listed -> nearest smaller listed M. Tuned with
# test_fp8_gemv.py --sweep (graph replay, 4-way weight rotation). Env VOPT_FP8_CFG="N,K,M:R,S,B;..."
# overrides entries.
CFG = {  # measured best (sweep: graph replay, 4-way weight rotation), see runs/var2.log
    (34816, 5120): {1: (1, 8, 5), 2: (1, 8, 5), 4: (1, 4, 5), 8: (1, 4, 2)},
    (16384, 5120): {1: (1, 4, 10), 2: (1, 8, 5), 4: (1, 4, 5), 8: (1, 4, 4)},
    (14336, 5120): {1: (1, 4, 10), 2: (1, 8, 5), 4: (1, 8, 5), 8: (1, 8, 5)},
    (5120, 6144): {1: (1, 4, 12), 2: (1, 8, 6), 4: (1, 4, 12), 8: (1, 8, 6)},
    (5120, 17408): {1: (1, 8, 8), 2: (1, 8, 17), 4: (1, 4, 17), 8: (1, 8, 17)},
}


# exact mode: rows per wave R for M >= 2 (R does not change the per-row arithmetic). Tuned with
# tests/fp8_mtp.py --sweep-r (graph replay, 4-way weight rotation). Env VOPT_FP8_RTAB="N,K,M:R;..."
RTAB = {}


def _env_rtab():
    for item in os.environ.get("VOPT_FP8_RTAB", "").split(";"):
        if not item.strip():
            continue
        a, b = item.split(":")
        n, k, m = (int(v) for v in a.split(","))
        RTAB.setdefault((n, k), {})[m] = int(b)


def _env_cfg():
    for item in os.environ.get("VOPT_FP8_CFG", "").split(";"):
        if not item.strip():
            continue
        a, b = item.split(":")
        n, k, m = (int(v) for v in a.split(","))
        CFG.setdefault((n, k), {})[m] = tuple(int(v) for v in b.split(","))


_env_cfg()
_env_rtab()


def pick_cfg(N, K, M):
    if EXACT and M > 1:
        _, S, B = _pick_cfg(N, K, 1)
        t = RTAB.get((N, K), {})
        ms = [m for m in t if m <= M]
        return (t[max(ms)] if ms else 1, S, B)
    return _pick_cfg(N, K, M)


def _pick_cfg(N, K, M):
    t = CFG.get((N, K))
    if t:
        ms = [m for m in t if m <= M]
        return t[max(ms)] if ms else t[min(t)]
    # generic: 64-row groups, split K over 8 waves, split across blocks for small N
    nsb = K // 128
    groups = N // 64
    B = 1
    while groups * B < 208 and nsb % (B + 1) == 0 and B < 16:
        B += 1
    return (1, 8, B)


def load_ext():
    global _LOADED
    if _LOADED:
        return
    from torch.utils.cpp_extension import load
    here = os.path.dirname(os.path.abspath(__file__))
    sd = os.path.join(here, "csrc")
    # separate build names: the exact build (MFMA multi-column path) and the deploy-v1 build
    name = ("vopt_fp8gemv_mtp" if EXACT else "vopt_fp8gemv_ext") + os.environ.get("VOPT_FP8_EXT_SUFFIX", "")
    bd = os.path.join(os.environ.get("TORCH_EXTENSIONS_DIR", "/cache/torch_ext"), name)
    os.makedirs(bd, exist_ok=True)
    # copy sources into the (writable) build dir: hipify writes next to the sources
    srcs = []
    for fn in sorted(os.listdir(sd)):
        if not fn.endswith((".cu", ".h")):
            continue
        src, dst = os.path.join(sd, fn), os.path.join(bd, fn)
        if not os.path.exists(dst) or open(dst, "rb").read() != open(src, "rb").read():
            shutil.copyfile(src, dst)
        if fn.endswith(".cu"):
            srcs.append(dst)
    # torch's load() only hashes the .cu sources: fold the header contents into the flags so
    # header edits trigger a rebuild
    import hashlib
    h = hashlib.sha1()
    for fn in sorted(os.listdir(sd)):
        if fn.endswith((".cu", ".h")):
            h.update(open(os.path.join(sd, fn), "rb").read())
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx90a")
    flags = ["-O3", "--offload-arch=gfx90a", "-fno-slp-vectorize", f"-DVOPT_SRC_HASH=0x{h.hexdigest()[:8]}"] + ([] if EXACT else ["-DVOPT_MFMA_MIN_M=9"]) + os.environ.get("VOPT_FP8_CFLAGS", "").replace(",", " ").split()
    load(name=name, sources=srcs, build_directory=bd, is_python_module=False,
         extra_cuda_cflags=[f for f in flags if f], verbose=False)
    _LOADED = True
    print(f"[vopt_fp8gemv] extension loaded (MAXM={MAXM}, EXACT={int(EXACT)})", file=sys.stderr)


_WS = {}


def workspace(dev):
    """Persistent split-K workspace + self-resetting counters (shared by all layers; kernels
    on one stream are serialized). Allocated before graph capture (at weight processing)."""
    key = str(dev)
    if key not in _WS:
        ws = torch.empty(8 << 20, dtype=torch.float32, device=dev)  # 32 MB
        cnt = torch.zeros(4096, dtype=torch.int32, device=dev)
        _WS[key] = (ws, cnt)
    return _WS[key]


def relayout(w):
    """[N,K] fp8/uint8 (any row stride) -> contiguous uint8 [N,K] holding tiles
    [N/64][K/16][64][16]. Lossless permutation."""
    N, K = w.shape
    assert N % 128 == 0 and K % 128 == 0, (N, K)
    u = w.view(torch.uint8)
    ld = u.stride(0)
    assert u.stride(1) == 1
    t = torch.empty((N // 64, K // 16, 64, 16), dtype=torch.uint8, device=w.device)
    t.copy_(torch.as_strided(u, (N // 64, K // 16, 64, 16), (64 * ld, 16, ld, 1), u.storage_offset()))
    return t.view(N, K)


def unrelayout(t):
    N, K = t.shape
    return t.view(N // 64, K // 16, 64, 16).permute(0, 2, 1, 3).reshape(N, K)


def gemv(x2, wt, s, M_cfg=None):
    N, K = wt.shape
    R, S, B = pick_cfg(N, K, M_cfg or x2.shape[0])
    ws, cnt = workspace(x2.device)
    return torch.ops.vopt_fp8gemv.fp8_gemv(x2, wt, s, ws, cnt, R, S, B)


def install(linear_mod):
    from vllm.platforms import PlatformEnum, current_platform
    from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonFp8BlockScaledMMKernel
    from vllm.model_executor.utils import replace_parameter
    from . import tiled_triton

    class VoptFp8GemvKernel(TritonFp8BlockScaledMMKernel):
        @classmethod
        def is_supported(cls, compute_capability=None):
            if not current_platform.is_rocm():
                return False, "ROCm only"
            arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
            if arch != "gfx90a":
                return False, "gfx90a only"
            return True, None

        @classmethod
        def can_implement(cls, config):
            ok, why = super().can_implement(config)
            if not ok:
                return ok, why
            if list(config.weight_quant_key.scale.group_shape) != [128, 128]:
                return False, "128x128 weight blocks only"
            return True, None

        def process_weights_after_loading(self, layer):
            super().process_weights_after_loading(layer)  # (pads rows on ROCm)
            load_ext()
            params = self._get_layer_params(layer)
            w = params.weight
            N, K = w.shape
            sname = params.block_scale_attr
            s = getattr(layer, sname)
            if N % 128 or K % 128 or s.dtype != torch.float32 or tuple(s.shape) != (N // 128, K // 128):
                layer.vopt_tiled = False
                return
            u = w.view(torch.uint8)
            if bool((((u & 0x7F) == 0x7F).any()).item()):
                print(f"[vopt_fp8gemv] NaN fp8 weights in {N}x{K}: keep stock path", file=sys.stderr)
                layer.vopt_tiled = False
                return
            t = relayout(w)
            del u
            replace_parameter(layer, params.WEIGHT, t.view(w.dtype))
            del w, t
            replace_parameter(layer, sname, s.contiguous())
            workspace(layer.weight.device)
            layer.vopt_tiled = True

        def apply_weights(self, layer, x, bias=None, **kwargs):
            if not getattr(layer, "vopt_tiled", False):
                return super().apply_weights(layer, x, bias, **kwargs)
            x2 = x.reshape(-1, x.shape[-1])
            w = layer.weight
            if x2.shape[0] <= MAXM and x2.dtype == torch.bfloat16:
                if not x2.is_contiguous():
                    x2 = x2.contiguous()
                out = gemv(x2, w.view(torch.uint8), layer.weight_scale_inv)
                if bias is not None:
                    out = out + bias
                return out.view(*x.shape[:-1], w.shape[0])
            # large M (prefill / big batches): stock dynamic per-token-group fp8 activation
            # quant + reference exact-decode Triton GEMM addressing the tiled layout
            params = self._get_layer_params(layer)
            q, xs = self.quant_fp8(x2, params.input_scale, params.input_scale_ub,
                                   use_triton=self.use_triton)
            out = tiled_triton.block_scaled_mm_tiled(q, w, xs, layer.weight_scale_inv,
                                                     self.config.out_dtype)
            if bias is not None:
                out = out + bias
            return out.to(self.config.out_dtype).view(*x.shape[:-1], w.shape[0])

    lst = linear_mod._POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.ROCM]
    if not any(k.__name__ == "VoptFp8GemvKernel" for k in lst):
        lst.insert(0, VoptFp8GemvKernel)
    linear_mod.VoptFp8GemvKernel = VoptFp8GemvKernel
