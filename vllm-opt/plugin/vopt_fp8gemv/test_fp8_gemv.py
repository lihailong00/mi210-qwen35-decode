"""Unit test + micro-bench for torch.ops.vopt_fp8gemv.fp8_gemv (run inside the vLLM image with
the GPU lock held, e.g. harness/img_run.sh <tag> "python3 -m vopt_fp8gemv.test_fp8_gemv ...").

Reference: fp64 x @ dequant(W)^T, W dequantized exactly from the fp8 bytes and 128x128 block
scales. Error check: |y - ref| <= 1e-2*|ref| + 1e-3*max|ref| per element (bf16 output rounding
is 2^-9 relative) plus rel L2.
Bench: CUDA graph of 50 back-to-back calls over a 4-way weight rotation (weights never L2-hot),
replayed 5x; us/call and GB/s of weight bytes (N*K). Also benches the reference Triton path
(activation quant + overlay exact-decode kernel on the original padded layout) and the tiled
Triton fallback.

usage: python3 -m vopt_fp8gemv.test_fp8_gemv [--real] [--bench] [--sweep] [--triton]
                                              [--shapes a,b] [--ms 1,2,4,8]
"""
import argparse, importlib.util, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vopt_fp8gemv import fp8_gemv as F  # noqa: E402

SHAPES = {"gate_up": (34816, 5120), "qkvz": (16384, 5120), "attn_qkv": (14336, 5120),
          "out": (5120, 6144), "down": (5120, 17408)}
REAL = {  # (file, [tensor prefixes concatenated along N])
    "gate_up": ("layers-0", ["mlp.gate_proj", "mlp.up_proj"]),
    "qkvz": ("layers-0", ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]),
    "attn_qkv": ("layers-3", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
    "out": ("layers-0", ["linear_attn.out_proj"]),
    "down": ("layers-0", ["mlp.down_proj"]),
}


def dequant(w8, s):
    N, K = w8.shape
    wf = w8.view(torch.float8_e4m3fn).to(torch.float64)
    sf = s.to(torch.float64).repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)[:, :K]
    return wf * sf


def rand_w(N, K, dev):
    b = torch.randint(0, 256, (N, K), dtype=torch.uint8, device=dev)
    b[(b & 0x7F) == 0x7F] = 0x7E  # no NaN, keep max magnitude present
    b[:, :64][(b[:, :64] & 0x78) == 0] |= 0x01  # make sure subnormals are present
    s = torch.rand(((N + 127) // 128, (K + 127) // 128), device=dev) * 1e-3 + 1e-4
    return b, s


def real_w(name):
    from safetensors import safe_open
    f, keys = REAL[name]
    ws, ss = [], []
    with safe_open(f"/model/{f}.safetensors", "pt") as h:
        for k in keys:
            ws.append(h.get_tensor(f"model.language_model.layers.{f.split('-')[1]}.{k}.weight"))
            ss.append(h.get_tensor(f"model.language_model.layers.{f.split('-')[1]}.{k}.weight_scale_inv").float())
    return torch.cat(ws).cuda().view(torch.uint8), torch.cat(ss).cuda()


def load_ref_overlay():
    p = os.environ.get("VOPT_REF_OVERLAY", "/vopt/fp8gemv/ref/fp8_utils_ref.py")
    spec = importlib.util.spec_from_file_location("vopt_ref_fp8_utils", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def bench_graph(fn_list, reps=5):
    """fn_list: 4 callables (weight rotation); returns us per call"""
    for f in fn_list:
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(50):
            fn_list[i % len(fn_list)]()
    g.replay(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(reps):
        g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / (50 * reps)


def check(y, ref):
    err = (y.to(torch.float64) - ref).abs()
    tol = 1e-2 * ref.abs() + 1e-3 * ref.abs().max()
    return {"max_abs": err.max().item(), "rel_l2": (err.norm() / ref.norm()).item(),
            "bad": int((err > tol).sum().item())}


def sweep_cfgs(N, K):
    nsb = K // 128
    out = []
    for R in (1, 2):
        groups = N // (64 * R)
        for S in (1, 2, 4, 8):
            for B in range(1, 33):
                if nsb % B:
                    continue
                sbw = nsb / B / S
                if sbw < 0.99 or sbw > 24:
                    continue
                blocks = groups * B
                if blocks < 80 or blocks > 8192:
                    continue
                out.append((R, S, B))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--triton", action="store_true")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--ms", default="1,2,4,8")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    F.load_ext()
    dev = "cuda"
    ms = [int(v) for v in a.ms.split(",")]
    res = {}
    ref_mod = load_ref_overlay() if a.triton else None
    from vopt_fp8gemv import tiled_triton as TT
    for name in a.shapes.split(","):
        N, K = SHAPES[name]
        if a.real:
            w, s = real_w(name)
            assert tuple(w.shape) == (N, K), (name, w.shape)
        else:
            w, s = rand_w(N, K, dev)
        nan = int(((w & 0x7F) == 0x7F).sum().item())
        # vLLM layout: padded row stride (K + 256), then our tiled re-layout
        wp = torch.zeros(N, K + 256, dtype=torch.uint8, device=dev)[:, :K]
        wp.copy_(w)
        wt = F.relayout(wp)
        assert torch.equal(F.unrelayout(wt), w), "relayout not lossless"
        Wd = dequant(w, s)
        for M in ms:
            x = (torch.randn(M, K, device=dev) * 2).to(torch.bfloat16)
            ref = x.to(torch.float64) @ Wd.T
            cfgs = [F.pick_cfg(N, K, M)]
            if a.sweep:
                cfgs += [c for c in sweep_cfgs(N, K) if c not in cfgs and c[1] * M * 64 * c[0] * 4 <= 65000
                         and N * c[2] * M <= (8 << 20) and not (M >= 3 and c[0] == 2)]
            ws_, cnt = F.workspace(dev)
            r = {"cfg": list(cfgs[0]), "nan_bytes": nan}
            for ci, (R, S, B) in enumerate(cfgs):
                y = torch.ops.vopt_fp8gemv.fp8_gemv(x, wt, s, ws_, cnt, R, S, B)
                c = check(y, ref)
                if ci == 0:
                    r.update(c)
                    # determinism: identical bits on repeat
                    y2 = torch.ops.vopt_fp8gemv.fp8_gemv(x, wt, s, ws_, cnt, R, S, B)
                    r["deterministic"] = bool(torch.equal(y, y2))
                elif c["bad"]:
                    print(f"  cfg {(R, S, B)} FAILED {c}", file=sys.stderr)
                    continue
                if a.bench or a.sweep:
                    wts = [wt] + [wt.clone() for _ in range(3)]
                    us = bench_graph([lambda t=t: torch.ops.vopt_fp8gemv.fp8_gemv(x, t, s, ws_, cnt, R, S, B)
                                      for t in wts])
                    del wts
                    gbps = N * K / us / 1e3
                    if ci == 0:
                        r["us"], r["GBps"] = round(us, 2), round(gbps, 1)
                    if a.sweep:
                        r.setdefault("sweep", []).append([R, S, B, round(us, 2), round(gbps, 1)])
            if a.sweep:
                r["sweep"].sort(key=lambda v: v[3])
                r["sweep"] = r["sweep"][:6]
            if a.triton and M in (1, 4):
                # reference path: activation quant + overlay exact-decode Triton (original layout)
                w8 = wp.view(torch.float8_e4m3fn)
                wps = [w8] + [wp.clone().view(torch.float8_e4m3fn) for _ in range(3)]

                def tri(wq):
                    q, qs = ref_mod.per_token_group_quant_fp8(x, 128, column_major_scales=False)
                    return ref_mod.w8a8_triton_block_scaled_mm(q, wq, qs, s, [128, 128], torch.bfloat16)
                yt = tri(w8)
                r["triton_ref"] = check(yt, ref)
                r["triton_us"] = round(bench_graph([lambda t=t: tri(t) for t in wps]), 2)
                q, qs = ref_mod.per_token_group_quant_fp8(x, 128, column_major_scales=False)
                ya = ref_mod.w8a8_triton_block_scaled_mm(q, w8, qs, s, [128, 128], torch.bfloat16)
                yb = TT.block_scaled_mm_tiled(q, wt, qs, s)
                r["tiled_triton_equal"] = bool(torch.equal(ya, yb))
                del wps
            res[f"{name}/M{M}"] = r
            print(name, M, json.dumps(r), flush=True)
        if a.triton:
            # large-M fallback parity: tiled Triton vs reference Triton on the original layout
            for M in (16, 64, 512):
                x = (torch.randn(M, K, device=dev) * 2).to(torch.bfloat16)
                q, qs = ref_mod.per_token_group_quant_fp8(x, 128, column_major_scales=False)
                w8 = wp.view(torch.float8_e4m3fn)
                ya = ref_mod.w8a8_triton_block_scaled_mm(q, w8, qs, s, [128, 128], torch.bfloat16)
                yb = TT.block_scaled_mm_tiled(q, wt, qs, s)
                ta = bench_graph([lambda: ref_mod.w8a8_triton_block_scaled_mm(q, w8, qs, s, [128, 128], torch.bfloat16)], 2)
                tb = bench_graph([lambda: TT.block_scaled_mm_tiled(q, wt, qs, s)], 2)
                r = {"equal": bool(torch.equal(ya, yb)), "ref_us": round(ta, 1), "tiled_us": round(tb, 1)}
                res[f"{name}/fallbackM{M}"] = r
                print(name, "fallback", M, json.dumps(r), flush=True)
        del w, s, wp, wt, Wd
        torch.cuda.empty_cache()
    bad = sum(v.get("bad", 0) for v in res.values())
    print("PASS" if bad == 0 else f"FAIL bad={bad}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
