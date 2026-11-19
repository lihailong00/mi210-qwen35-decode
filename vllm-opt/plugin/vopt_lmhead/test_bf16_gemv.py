"""Unit test + micro-bench for torch.ops.vopt_lmhead.bf16_gemv / bf16_gemv_argmax.
Runs inside the vLLM image with the GPU lock held (harness/img_run.sh).
 - correctness: fp64 reference x @ W^T (row-chunked), error in units of the bf16 ulp of the result;
   greedy argmax vs torch.argmax of the op's own bf16 logits (must be identical);
 - timing: CUDA-graph replay of NREP back-to-back calls (lm_head 2.54 GB >> L2, so no rotation
   needed; small N uses an 8-way weight rotation), us per call and GB/s of weight bytes;
 - baselines: vLLM's wvSplitK (what vLLM uses today for M<=5), torch F.linear (hipBLASLt).
usage: python3 -m vopt_lmhead.test_bf16_gemv [--cfgs 0,1,..] [--ms 1,2,4] [--real] [--small]"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vopt_lmhead import ext  # noqa: E402

OPS = None


def graph_time(fn, nrep=20, reps=5):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(nrep):
            fn(i)
    g.replay(); torch.cuda.synchronize()
    best = 1e30
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1000 / nrep)
    return best


def check(y, x, W):
    """max error in bf16 ulps of the reference + count beyond 1 ulp (+ accumulation slack)."""
    N = W.shape[0]
    worst_ulp, nbad = 0.0, 0
    x64 = x.double()
    for r0 in range(0, N, 32768):
        ref = x64 @ W[r0:r0 + 32768].double().T
        yy = y[:, r0:r0 + 32768].double()
        ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
        slack = 1e-6 * (x64.abs() @ W[r0:r0 + 32768].double().abs().T)
        e = (yy - ref).abs()
        worst_ulp = max(worst_ulp, (e / (ulp + slack)).max().item())
        nbad += int((e > ulp + slack).sum().item())
    return worst_ulp, nbad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", default="0,1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--ms", default="1,2,3,4,5,6")
    ap.add_argument("--grids", default="0")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--small", action="store_true", help="in_proj_ba 96x5120 instead of lm_head")
    ap.add_argument("--nocheck", action="store_true")
    ap.add_argument("--baselines", action="store_true")
    a = ap.parse_args()
    ext.load_ext()
    ops = torch.ops.vopt_lmhead
    dev = "cuda"
    torch.manual_seed(0)
    res = {}
    if a.small:
        N, K = 96, 5120
        Ws = [(torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16) for _ in range(8)]
    else:
        if a.real:
            from safetensors import safe_open
            with safe_open("/model/outside.safetensors", "pt", device="cuda") as h:
                W = h.get_tensor("lm_head.weight")
        else:
            W = (torch.randn(248320, 5120, device=dev) * 0.02).to(torch.bfloat16)
        N, K = W.shape
        Ws = [W]
    gb = N * K * 2 / 1e9
    print(f"N={N} K={K} weight {gb:.3f} GB", flush=True)
    cfgs = [int(c) for c in a.cfgs.split(",")]
    grids = [int(g) for g in a.grids.split(",")]
    cu = torch.cuda.get_device_properties(0).multi_processor_count
    for M in [int(m) for m in a.ms.split(",")]:
        x = (torch.randn(M, K, device=dev) * 1.5).to(torch.bfloat16)
        W0 = Ws[0]
        if not a.nocheck:
            y = ops.bf16_gemv(x, W0, -1, 0)
            ulp, nbad = check(y, x, W0)
            res[f"M{M}/check"] = {"max_err_ulp": ulp, "n_beyond_1ulp": nbad}
            if not a.small:
                y2, idx = ops.bf16_gemv_argmax(x, W0, -1, 0)
                ref_idx = torch.argmax(y2.float(), dim=-1)
                res[f"M{M}/check"]["argmax_ok"] = bool((idx == ref_idx).all().item())
                res[f"M{M}/check"]["logits_same"] = bool((y2 == y).all().item())
        for c in cfgs:
            for gr in grids:
                try:
                    us = graph_time(lambda i=0: ops.bf16_gemv(x, Ws[i % len(Ws)], c, gr))
                except Exception as e:  # noqa
                    res[f"M{M}/cfg{c}/g{gr}"] = repr(e)[:80]
                    continue
                res[f"M{M}/cfg{c}/g{gr}"] = {"us": round(us, 1), "TBps": round(gb / us * 1e3, 3)}
                print(f"M{M} cfg{c} grid{gr}: {us:.1f} us {gb / us * 1e3:.3f} TB/s", flush=True)
        if not a.small:
            us = graph_time(lambda i=0: ops.bf16_gemv_argmax(x, W0, -1, 0))
            res[f"M{M}/argmax_default"] = {"us": round(us, 1)}
            print(f"M{M} argmax default: {us:.1f} us", flush=True)
        if a.baselines:
            import vllm._custom_ops as vops  # noqa
            if M <= 5:
                us = graph_time(lambda i=0: torch.ops._rocm_C.wvSplitK(Ws[i % len(Ws)], x, None, cu))
                res[f"M{M}/wvSplitK"] = {"us": round(us, 1), "TBps": round(gb / us * 1e3, 3)}
                print(f"M{M} wvSplitK: {us:.1f} us", flush=True)
            us = graph_time(lambda i=0: F.linear(x, Ws[i % len(Ws)]))
            res[f"M{M}/F.linear"] = {"us": round(us, 1), "TBps": round(gb / us * 1e3, 3)}
            print(f"M{M} F.linear: {us:.1f} us", flush=True)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
