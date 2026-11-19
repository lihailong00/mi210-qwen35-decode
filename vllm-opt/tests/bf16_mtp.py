"""Per-row exactness of the bf16 GEMVs (lm_head 248320x5120, in_proj_ba 96x5120) for M = 1..8.

  python3 bf16_mtp.py [--bench] [--real]

  * lm_head: the patched apply's dispatch (M <= 6: one launch with the default config of that M;
    M > 6: launches of <= BIG_SPLIT rows) -> row m bitwise == the M=1 call on x[m]; vs fp64
    (bf16 ulps); fused argmax (M <= 6) == torch.argmax of the op's logits, planted exact ties at
    the maximum resolve to the lowest index (== torch / the K=0 greedy path)
  * in_proj_ba: M = 1..8 bitwise vs M=1, vs fp64, graph capture + replay with new x
  * --bench: graph-replay us per call for every M
"""
import argparse, os, sys
import torch
import vopt_lmhead as L
from vopt_lmhead import ext


def lm_apply(x, w):  # same dispatch as vopt_lmhead._lmhead.apply (without the stash bookkeeping)
    ops = torch.ops.vopt_lmhead
    if x.shape[0] > 6:
        return torch.cat([ops.bf16_gemv(x[i:i + L.BIG_SPLIT], w, -1, 0)
                          for i in range(0, x.shape[0], L.BIG_SPLIT)])
    return ops.bf16_gemv(x, w, -1, 0)


def ulp_err(y, x, W):
    worst, nbad = 0.0, 0
    x64 = x.double()
    for r0 in range(0, W.shape[0], 32768):
        ref = x64 @ W[r0:r0 + 32768].double().T
        ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(1e-30))) - 7)
        slack = 1e-6 * (x64.abs() @ W[r0:r0 + 32768].double().abs().T)
        e = (y[:, r0:r0 + 32768].double() - ref).abs()
        worst = max(worst, (e / (ulp + slack)).max().item())
        nbad += int((e > ulp + slack).sum().item())
    return worst, nbad


def gtime(fn, nrep=20, reps=5):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(nrep):
            fn()
    g.replay(); torch.cuda.synchronize()
    best = 1e30
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1000 / nrep)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args()
    ext.load_ext()
    ops = torch.ops.vopt_lmhead
    g = torch.Generator(device="cuda"); g.manual_seed(7)
    fails = 0
    # ---------------- lm_head
    N, K = 248320, 5120
    if a.real:
        from safetensors import safe_open
        import json
        idx = json.load(open("/model/model.safetensors.index.json"))["weight_map"]
        key = [k for k in idx if k.endswith("lm_head.weight")][0]
        with safe_open("/model/" + idx[key], "pt") as h:
            W = h.get_tensor(key).cuda()
    else:
        W = (torch.randn(N, K, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    x = (torch.randn(8, K, device="cuda", generator=g) * 2).to(torch.bfloat16)
    one = torch.cat([ops.bf16_gemv(x[i:i + 1], W, -1, 0) for i in range(8)])
    for M in range(1, 9):
        y = lm_apply(x[:M].contiguous(), W)
        nd = int((y.view(torch.int16) != one[:M].view(torch.int16)).sum())
        wu, nb = ulp_err(y, x[:M], W) if M in (1, 4, 8) else (0.0, 0)
        line = f"lm_head M={M}: diff_vs_M1 {nd}"
        if M in (1, 4, 8):
            line += f" fp64 worst {wu:.2f} ulp, >1ulp {nb}"
        if M <= 6:
            ya, ia = ops.bf16_gemv_argmax(x[:M].contiguous(), W, -1, 0)
            na = int((ya.view(torch.int16) != y.view(torch.int16)).sum())
            ok = bool((ia == torch.argmax(y.float(), dim=1)).all())
            line += f" argmax-variant diff {na} argmax==torch {ok}"
            fails += (na != 0) + (not ok)
        fails += nd != 0 or nb != 0
        print(line, flush=True)
    # planted ties: force the top logit of row m to appear at two (and three) indices
    Wt = W.clone()
    for trial in range(3):
        xs = x[:6].contiguous()
        y = ops.bf16_gemv(xs, Wt, -1, 0)
        top = torch.argmax(y.float(), dim=1)
        for m in range(6):
            dup = [int(top[m]) + 1000 * (trial + 1) + m, int(top[m]) - 777 * (trial + 1) - m]
            for d in dup:
                if 0 <= d < N:
                    Wt[d] = Wt[top[m]]  # identical row -> identical logit for every x row
        y = ops.bf16_gemv(xs, Wt, -1, 0)
        ya, ia = ops.bf16_gemv_argmax(xs, Wt, -1, 0)
        ref = torch.argmax(y.float(), dim=1)  # torch: first maximal index
        ntie = int(sum(int((y[m].float() == y[m].float().max()).sum()) > 1 for m in range(6)))
        ok = bool((ia == ref).all())
        # per-row M=1 argmax as in K=0 greedy decode
        i1 = torch.stack([ops.bf16_gemv_argmax(xs[m:m + 1], Wt, -1, 0)[1][0] for m in range(6)])
        ok1 = bool((i1 == ia).all())
        print(f"ties trial {trial}: rows with ties {ntie}/6 argmax==torch {ok} M6 argmax==M1 argmax {ok1}")
        fails += (not ok) + (not ok1)
    del Wt
    if a.bench:
        for M in range(1, 9):
            xm = x[:M].contiguous()
            print(f"  lm_head time M={M}: {gtime(lambda: lm_apply(xm, W), nrep=5):.0f} us", flush=True)
    del W
    torch.cuda.empty_cache()
    # ---------------- small N (in_proj_ba)
    for (N, scale) in [(96, 0.05), (192, 0.05)]:
        Ws = [(torch.randn(N, K, device="cuda", generator=g) * scale).to(torch.bfloat16) for _ in range(8)]
        x = (torch.randn(8, K, device="cuda", generator=g)).to(torch.bfloat16)
        one = torch.cat([ops.bf16_gemv(x[i:i + 1], Ws[0], -1, 0) for i in range(8)])
        for M in range(1, 9):
            y = ops.bf16_gemv(x[:M].contiguous(), Ws[0], -1, 0)
            nd = int((y.view(torch.int16) != one[:M].view(torch.int16)).sum())
            wu, nb = ulp_err(y, x[:M], Ws[0])
            xs = x[:M].clone()
            st = torch.cuda.Stream()
            with torch.cuda.stream(st):
                ops.bf16_gemv(xs, Ws[0], -1, 0); torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    yg = ops.bf16_gemv(xs, Ws[0], -1, 0)
            xs.copy_(torch.randn(M, K, device="cuda", generator=g).to(torch.bfloat16))
            gr.replay(); torch.cuda.synchronize()
            ngr = int((yg.view(torch.int16) != ops.bf16_gemv(xs, Ws[0], -1, 0).view(torch.int16)).sum())
            t = ""
            if a.bench:
                xm = x[:M].contiguous()
                k = [0]

                def f():
                    k[0] = (k[0] + 1) % 8
                    ops.bf16_gemv(xm, Ws[k[0]], -1, 0)
                t = f" time {gtime(f, nrep=40):.1f} us"
            print(f"smallN {N} M={M}: diff_vs_M1 {nd} fp64 worst {wu:.2f} ulp >1ulp {nb} graph_diff {ngr}{t}",
                  flush=True)
            fails += (nd != 0) + (nb != 0) + (ngr != 0)
    print("FAILS", fails)


if __name__ == "__main__":
    main()
