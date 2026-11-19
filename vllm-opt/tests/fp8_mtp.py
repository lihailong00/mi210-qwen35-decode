"""Exactness + timing of the FP8 weight-only GEMV for multi-token verify (M = 1..8).

  python3 fp8_mtp.py [--real] [--ms 1,...,8] [--shapes ...] [--bench] [--sweep-r] [--dump F] [--ref F]

Checks per shape (random fp8 bytes incl. subnormals and max magnitude; --real: layer-0/3 checkpoint
weights):
  * bitwise: row m of an M-row call == the M=1 call on x[m] (all M, all rows, all columns)
  * fp64: |y - ref| <= 1e-2|ref| + 1e-3 max|ref| and rel-L2 vs exact fp64 dequant reference
  * graph: a captured M-row call replayed on new x equals the eager call
  * --dump F: save M=1 outputs (fixed seeds); --ref F: compare this build's M=1 outputs with a dump
    made by another build (e.g. the deploy-v1 plugin) -> M=1 numerics unchanged
  * --bench: graph replay of 50 calls over 4 weight copies (no L2 reuse), us/call per M
  * --sweep-r: time R=1/2 for every M (exact mode keeps the M=1 (S,B))
"""
import argparse, os, sys, time
import torch
from vopt_fp8gemv import fp8_gemv as F

SHAPES = {"gate_up": (34816, 5120), "qkvz": (16384, 5120), "attn_qkv": (14336, 5120),
          "out": (5120, 6144), "down": (5120, 17408)}
REAL = {"gate_up": ("layers-0", ["mlp.gate_proj", "mlp.up_proj"]),
        "qkvz": ("layers-0", ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]),
        "attn_qkv": ("layers-3", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
        "out": ("layers-0", ["linear_attn.out_proj"]), "down": ("layers-0", ["mlp.down_proj"])}


def rand_w(N, K, g):
    b = torch.randint(0, 256, (N, K), dtype=torch.uint8, device="cuda", generator=g)
    b[(b & 0x7F) == 0x7F] = 0x7E
    b[:, :64][(b[:, :64] & 0x78) == 0] |= 0x01
    s = torch.rand((N // 128, K // 128), device="cuda", generator=g) * 1e-3 + 1e-4
    return b, s


def real_w(name):
    from safetensors import safe_open
    f, keys = REAL[name]
    li = f.split("-")[1]
    ws, ss = [], []
    with safe_open(f"/model/{f}.safetensors", "pt") as h:
        for k in keys:
            ws.append(h.get_tensor(f"model.language_model.layers.{li}.{k}.weight"))
            ss.append(h.get_tensor(f"model.language_model.layers.{li}.{k}.weight_scale_inv").float())
    return torch.cat(ws).cuda().view(torch.uint8), torch.cat(ss).cuda()


def rand_x(M, K, g, scale=1.0):
    x = torch.randn(M, K, device="cuda", generator=g) * scale
    # heavy-ish tails + a few tiny / zero / negative-zero values
    x[:, ::97] *= 30.0
    x[:, 5::131] = 0.0
    x[:, 7::131] = -0.0
    x[:, 11::131] *= 1e-30
    return x.to(torch.bfloat16)


def dequant(w8, s):
    wf = w8.view(torch.float8_e4m3fn).to(torch.float64)
    return wf * s.double().repeat_interleave(128, 0).repeat_interleave(128, 1)


def call(x, wt, s, cfg):
    ws, cnt = F.workspace(x.device)
    return torch.ops.vopt_fp8gemv.fp8_gemv(x, wt, s, ws, cnt, *cfg)


def bench(fns, reps=5):
    for f in fns:
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(50):
            fns[i % len(fns)]()
    g.replay(); torch.cuda.synchronize()
    best = 1e9
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1e3 / 50)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--ms", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--sweep-r", action="store_true")
    ap.add_argument("--nocheck", action="store_true")
    ap.add_argument("--dump", default="")
    ap.add_argument("--ref", default="")
    a = ap.parse_args()
    F.load_ext()
    ms = [int(v) for v in a.ms.split(",")]
    print(f"EXACT={int(getattr(F, 'EXACT', 0))} flags={os.environ.get('VOPT_FP8_CFLAGS', '')} real={a.real}", flush=True)
    dump = {}
    ref = torch.load(a.ref) if a.ref else None
    fails = 0
    for name in a.shapes.split(","):
        N, K = SHAPES[name]
        g = torch.Generator(device="cuda"); g.manual_seed(1234 + N + K)
        if a.real:
            w8, s = real_w(name)
        else:
            w8, s = rand_w(N, K, g)
        wt = F.relayout(w8)
        x = rand_x(8, K, g)
        c1 = F.pick_cfg(N, K, 1)
        one = torch.cat([call(x[i:i + 1].contiguous(), wt, s, c1) for i in range(8)])
        line = [f"{name} {N}x{K} cfg1={c1}"]
        if a.dump:
            dump[name] = one.cpu()
        if ref is not None:
            nd = int((one.view(torch.int16) != ref[name].cuda().view(torch.int16)).sum())
            line.append(f"M1 vs ref-build: {nd} differ")
            fails += nd != 0
        if not a.nocheck:
            yr = x.double() @ dequant(w8, s).t()
            tol = 1e-2 * yr.abs() + 1e-3 * yr.abs().max()
            for M in ms:
                cm = F.pick_cfg(N, K, M)
                y = call(x[:M].contiguous(), wt, s, cm)
                nd = int((y.view(torch.int16) != one[:M].view(torch.int16)).sum())
                bad = int(((y.double() - yr[:M]).abs() > tol[:M]).sum())
                rl2 = float((y.double() - yr[:M]).norm() / yr[:M].norm())
                # graph capture: replay with new x
                xs = x[:M].clone()
                st = torch.cuda.Stream()
                with torch.cuda.stream(st):
                    call(xs, wt, s, cm); torch.cuda.synchronize()
                    gr = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gr):
                        yg = call(xs, wt, s, cm)
                xs.copy_(rand_x(M, K, g)); gr.replay(); torch.cuda.synchronize()
                ng = int((yg.view(torch.int16) != call(xs, wt, s, cm).view(torch.int16)).sum())
                fails += (nd != 0) + (bad != 0) + (ng != 0) + (rl2 > 5e-3)
                line.append(f"M={M} R{cm[0]}: diff_vs_M1 {nd} fp64_bad {bad} rl2 {rl2:.1e} graph_diff {ng}")
        print(" | ".join(line), flush=True)
        if a.bench or a.sweep_r:
            wts = [wt] + [F.relayout(rand_w(N, K, g)[0]) for _ in range(3)]
            out = []
            for M in ms:
                xm = x[:M].contiguous()
                cfgs = [F.pick_cfg(N, K, M)]
                if a.sweep_r and M > 1:
                    _, S, B = cfgs[0]
                    cfgs = [(1, S, B), (2, S, B)]
                ts = []
                for cfg in cfgs:
                    fns = [(lambda w_=w_, cfg=cfg: call(xm, w_, s, cfg)) for w_ in wts]
                    ts.append(f"R{cfg[0]}:{bench(fns):.1f}")
                out.append(f"M={M} " + "/".join(ts))
            print(f"  time {name} floor {N * K / 1446e3:.1f}us: " + " | ".join(out), flush=True)
            del wts
        del wt, w8
        torch.cuda.empty_cache()
    if a.dump:
        torch.save(dump, a.dump)
    print("FAILS", fails)


if __name__ == "__main__":
    main()
