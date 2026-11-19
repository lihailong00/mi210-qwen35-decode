"""MTP draft fc (bf16 [5120, 10240]) through the split-K bf16 GEMV: accuracy vs fp64, graph replay,
timing vs the stock path (torch F.linear -> wvSplitK / hipBLASLt). python3 vopt_lmhead/test_fc.py"""
import os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vopt_lmhead import ext
ext.load_ext()
ops = torch.ops.vopt_lmhead
dev = "cuda"
torch.manual_seed(0)
try:
    from safetensors import safe_open
    with safe_open("/model/mtp.safetensors", "pt", device="cpu") as f:
        W = f.get_tensor("mtp.fc.weight").to(dev)
    print("fc: real weights", tuple(W.shape), W.dtype)
except Exception as e:
    W = (torch.randn(5120, 10240, device=dev) * 0.01).to(torch.bfloat16)
    print("fc: random weights", e)


def tm(fn, it=50):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    g.replay(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / it / 10 * 1e6


for M in (1, 2, 4, 8):
    x = torch.randn(M, 10240, device=dev).to(torch.bfloat16)
    ref = (x.double() @ W.double().T)
    for cfg in (100, 101, 102, 103):
        y = ops.bf16_gemv(x, W, cfg, 0)
        err = ((y.double() - ref).abs() / (ref.abs() + 1e-2)).max().item()
        ulp = ((y.double() - ref).abs() / (ref.abs().clamp_min(1e-30) * 2 ** -8)).max().item()
        assert err < 2e-2, (M, cfg, err)
    y_st = F.linear(x, W)
    e_st = (y_st.double() - ref).abs().max().item()
    e_me = (ops.bf16_gemv(x, W, -1, 0).double() - ref).abs().max().item()
    ts = {c: tm(lambda: ops.bf16_gemv(x, W, c, 0)) for c in (100, 101, 102, 103)}
    t_st = tm(lambda: F.linear(x, W))
    print(f"fc M={M}: max|err| vs fp64 gemv {e_me:.4g} stock {e_st:.4g} | time stock {t_st:.1f} us, gemv "
          + " ".join(f"{c}:{t:.1f}" for c, t in ts.items()) + " us (floor 72.5)")
# graph replay correctness
x = torch.randn(4, 10240, device=dev).to(torch.bfloat16)
ops.bf16_gemv(x, W, -1, 0)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    yy = ops.bf16_gemv(x, W, -1, 0)
for _ in range(3):
    x.copy_(torch.randn(4, 10240, device=dev).to(torch.bfloat16))
    g.replay(); torch.cuda.synchronize()
    assert torch.equal(yy, ops.bf16_gemv(x, W, -1, 0))
print("fc graph replay: OK")
