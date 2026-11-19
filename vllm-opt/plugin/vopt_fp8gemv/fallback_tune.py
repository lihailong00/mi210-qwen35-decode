"""Tune the tiled-layout Triton fallback (M > MAXM) against the reference overlay kernel on the
original layout. Same inputs (dynamic fp8 activation quant), CUDA-graph timing (20 calls x 3).
usage: python3 -m vopt_fp8gemv.fallback_tune [--ms 16,64,512] [--shapes ...] [--out f.json]"""
import argparse, itertools, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vopt_fp8gemv import fp8_gemv as F  # noqa: E402
from vopt_fp8gemv import tiled_triton as TT  # noqa: E402
from vopt_fp8gemv.test_fp8_gemv import SHAPES, rand_w, load_ref_overlay  # noqa: E402


def gtime(fn, n=20, reps=3):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    g.replay(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(reps):
        g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / (n * reps)


def cfgs_for(M):
    if os.environ.get("FT_EXT") and M <= 16:
        out = []
        for bn, bk, st, wpe, gm, perm in itertools.product([64, 128], [64, 128], [1, 2, 3], [0, 1, 2], [1, 32], [0, 1]):
            c = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": gm,
                 "num_warps": 4, "num_stages": st}
            if wpe:
                c["waves_per_eu"] = wpe
            out.append((c, perm))
        return out
    bms = [16] if M <= 16 else ([16, 32] if M <= 64 else [64, 128])
    out = []
    for bm, bn, bk, nw, perm in itertools.product(bms, [32, 64, 128, 256], [64, 128, 256], [4, 8], [0, 1]):
        if bm * bn > 128 * 256 or (M <= 64 and nw == 8 and bn < 256):
            continue
        if (bn == 32 and (perm or M > 16)) or (bk == 256 and M > 16):
            continue
        out.append(({"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 32,
                     "num_warps": nw, "num_stages": 2}, perm))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="16,64,512")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ref = load_ref_overlay()
    res = {}
    for name in a.shapes.split(","):
        N, K = SHAPES[name]
        w, s = rand_w(N, K, "cuda")
        wp = torch.zeros(N, K + 256, dtype=torch.uint8, device="cuda")[:, :K]
        wp.copy_(w)
        w8 = wp.view(torch.float8_e4m3fn)
        wt = F.relayout(wp)
        for M in [int(v) for v in a.ms.split(",")]:
            x = (torch.randn(M, K, device="cuda") * 2).to(torch.bfloat16)
            q, qs = ref.per_token_group_quant_fp8(x, 128, column_major_scales=False)
            yr = ref.w8a8_triton_block_scaled_mm(q, w8, qs, s, [128, 128], torch.bfloat16)
            tr = gtime(lambda: ref.w8a8_triton_block_scaled_mm(q, w8, qs, s, [128, 128], torch.bfloat16))
            best = None
            for cfg, perm in cfgs_for(M):
                if N % cfg["BLOCK_SIZE_N"] or K % cfg["BLOCK_SIZE_K"]:
                    continue
                try:
                    yt = TT.block_scaled_mm_tiled(q, wt, qs, s, cfg=cfg, permute=perm)
                    d = (yt.float() - yr.float()).abs().max().item() / max(yr.float().abs().max().item(), 1e-30)
                    t = gtime(lambda: TT.block_scaled_mm_tiled(q, wt, qs, s, cfg=cfg, permute=perm))
                except Exception as e:  # noqa: BLE001
                    print("  fail", cfg, perm, repr(e)[:120], flush=True)
                    continue
                if d > 1e-2:
                    print("  WRONG", cfg, perm, d, flush=True)
                    continue
                if best is None or t < best[0]:
                    best = (t, cfg, perm, d)
            r = {"ref_us": round(tr, 1), "best_us": round(best[0], 1), "cfg": best[1], "permute": best[2],
                 "rel_maxdiff": best[3]}
            res[f"{name}/M{M}"] = r
            print(name, M, json.dumps(r), flush=True)
        del w, s, wp, wt
        torch.cuda.empty_cache()
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
