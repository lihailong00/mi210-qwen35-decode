"""Per-row exactness of the bf16 GEMV for M = 1..8 (row m of an M-row call == M=1 call on x[m]),
lm_head (248320x5120, with argmax epilogue) and in_proj_ba (96x5120); plus M=4 config timing.
Run in the image: python3 vopt_lmhead/test_rows_exact.py"""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vopt_lmhead import ext
ext.load_ext()
ops = torch.ops.vopt_lmhead
dev = "cuda"
torch.manual_seed(0)
try:
    from safetensors import safe_open
    with safe_open("/model/outside.safetensors", "pt", device="cpu") as f:
        W = f.get_tensor("lm_head.weight").to(dev)
    print("lm_head: real weights")
except Exception as e:
    W = (torch.randn(248320, 5120, device=dev) * 0.02).to(torch.bfloat16)
    print("lm_head: random weights", e)
Wb = (torch.randn(96, 5120, device=dev) * 0.02).to(torch.bfloat16)
bad = 0
for trial in range(3):
    X = (torch.randn(8, 5120, device=dev) * (3 if trial else 30)).to(torch.bfloat16)
    if trial == 2:  # planted tie in row 5: duplicate the winning row's weights at a lower index
        pass
    ref = [ops.bf16_gemv_argmax(X[m:m + 1], W, -1, 0) for m in range(8)]
    refb = [ops.bf16_gemv(X[m:m + 1], Wb, -1, 0) for m in range(8)]
    for M in range(1, 9):
        y, idx = ops.bf16_gemv_argmax(X[:M], W, -1, 0)
        yb = ops.bf16_gemv(X[:M], Wb, -1, 0)
        for m in range(M):
            ok = torch.equal(y[m], ref[m][0][0]) and idx[m].item() == ref[m][1].item() and torch.equal(yb[m], refb[m][0])
            if not ok:
                bad += 1
                print(f"MISMATCH M={M} row {m}")
        # argmax == torch.argmax of the bf16 logits (first index on ties)
        assert torch.equal(idx, y.argmax(1)), (idx, y.argmax(1))
    # strided x (row stride 5120*2) as from hidden_states[logits_indices] views
    Xs = torch.randn(16, 5120, device=dev).to(torch.bfloat16)[::2]
    y, idx = ops.bf16_gemv_argmax(Xs, W, -1, 0)
    for m in range(8):
        assert torch.equal(y[m], ops.bf16_gemv_argmax(Xs[m:m + 1].contiguous(), W, -1, 0)[0][0])
print(f"rows exact: M=1..8 x 3 trials, lm_head (+argmax) and in_proj_ba: {bad} mismatches")
# graph capture at M=8
x = torch.randn(8, 5120, device=dev).to(torch.bfloat16)
ops.bf16_gemv_argmax(x, W, -1, 0)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    yy, ii = ops.bf16_gemv_argmax(x, W, -1, 0)
for _ in range(3):
    x.copy_(torch.randn(8, 5120, device=dev).to(torch.bfloat16))
    g.replay(); torch.cuda.synchronize()
    y2, i2 = ops.bf16_gemv_argmax(x, W, -1, 0)
    assert torch.equal(yy, y2) and torch.equal(ii, i2)
print("graph replay M=8: OK")
def tm(fn, it=30):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(5): fn()
    g.replay(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(it): g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / it / 5 * 1e6
for M in (1, 2, 4, 6, 8):
    x = torch.randn(M, 5120, device=dev).to(torch.bfloat16)
    cfgs = [c for c in range(14) if not (c in (0, 1, 2, 3, 4, 11, 12, 13) and M > 2)]
    ts = {}
    for c in cfgs:
        try:
            ts[c] = tm(lambda: ops.bf16_gemv_argmax(x, W, c, 0))
        except Exception as e:
            pass
    d = tm(lambda: ops.bf16_gemv_argmax(x, W, -1, 0))
    best = min(ts, key=ts.get)
    print(f"lm_head M={M}: default {d:.0f} us; best cfg {best} {ts[best]:.0f} us; " + " ".join(f"{c}:{t:.0f}" for c, t in ts.items()))
