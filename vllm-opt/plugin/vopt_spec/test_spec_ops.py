"""Unit tests + micro-bench for vopt_spec kernels (run inside the vLLM image with the GPU).
  python3 test_spec_ops.py [quant] [head] [agree] [bench] [verify]
agree/bench use the real lm_head (/model/outside.safetensors) and real lm_head inputs
(target final-norm outputs from the scout's eager dumps, /vopt/mtp-draft/runs/hid.pt)."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vopt_spec import ext  # noqa: E402

ext.load_ext()
ops = torch.ops.vopt_spec
dev = "cuda"
ARGS = set(sys.argv[1:]) or {"quant", "head", "i4", "nan", "agree", "agree4", "bench", "verify"}
HID = os.environ.get("HID", "/vopt/mtp-draft/runs/hid.pt")
EB = torch.empty(0, dtype=torch.bfloat16, device=dev)  # no bf16 rescoring


def dequant(q, s):
    """e4m3fn bytes -> fp64 value * (scale / 256)."""
    b = q.to(torch.int64)
    sgn = torch.where(b >= 128, -1.0, 1.0).to(torch.float64)
    e = (b >> 3) & 15
    m = (b & 7).to(torch.float64)
    v = torch.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * torch.pow(2.0, (e - 7).to(torch.float64)))
    return sgn * v * (s.to(torch.float64)[:, None] / 256)


def test_quant():
    torch.manual_seed(0)
    w = (torch.randn(512, 5120, device=dev) * torch.rand(512, 1, device=dev) * 0.1).to(torch.bfloat16)
    w[3, 7] = 0
    w[5] = 0  # all-zero row
    q, s = ops.quantize_rows_fp8(w, torch.empty(0, dtype=torch.long, device=dev), w.shape[0])
    ok = True
    if hasattr(torch, "float8_e4m3fn"):
        sc = (s / 256).clone()
        ref = (w.float() / sc[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
        mism = (ref != q).sum().item()
        print(f"quant: codes vs torch float8_e4m3fn cast: {mism} mismatches of {q.numel()}")
        ok &= mism == 0
    dq = dequant(q, s)
    rel = ((dq - w.double()).norm(dim=1) / w.double().norm(dim=1).clamp_min(1e-30))
    print(f"quant: per-row rel L2 error mean {rel[w.double().norm(dim=1) > 0].mean():.4f} max {rel.max():.4f}")
    assert ok
    # row-gather variant
    rows = torch.tensor([5, 0, 511, 3], device=dev)
    q2, s2 = ops.quantize_rows_fp8(w, rows, 0)
    assert torch.equal(q2, q[rows]) and torch.equal(s2, s[rows])
    print("quant: OK")


def ref_argmax(x, q, s):
    xs = x.float().clamp(-65504, 65504).half().double()
    y = xs @ dequant(q, s).T
    return y


def test_head():
    torch.manual_seed(1)
    for N in (1, 7, 1000, 65536, 100003):
        w = (torch.randn(N, 5120, device=dev) * 0.02).to(torch.bfloat16)
        q, s = ops.quantize_rows_fp8(w, torch.empty(0, dtype=torch.long, device=dev), N)
        for M in (1, 2, 3, 4):
            x = (torch.randn(M, 5120, device=dev) * 3).to(torch.bfloat16)
            y = ref_argmax(x, q, s)
            top2 = y.topk(min(2, N), dim=1).values
            for cfg in range(6):
                idx = ops.fp8_head_argmax(x, q, s, torch.empty(0, dtype=torch.long, device=dev), EB, 0.0, cfg, 0)
                got = y.gather(1, idx[:, None])[:, 0]
                # fp32 vs fp64 accumulation: must hit the max unless the top-2 gap is ~fp32 noise
                gap = (top2[:, 0] - got).abs()
                tol = 1e-4 * y.abs().max(dim=1).values + 1e-6
                bad = (gap > tol).sum().item()
                if bad:
                    print(f"head FAIL N={N} M={M} cfg={cfg} idx={idx.tolist()} ref={y.argmax(1).tolist()} gap={gap.tolist()}")
                assert bad == 0
            # idmap
            im = torch.randperm(10 ** 6, device=dev)[:N]
            idx0 = ops.fp8_head_argmax(x, q, s, torch.empty(0, dtype=torch.long, device=dev), EB, 0.0, 0, 0)
            idx1 = ops.fp8_head_argmax(x, q, s, im, EB, 0.0, 0, 0)
            assert torch.equal(im[idx0], idx1)
            # bf16 rescoring (window 2.0): equals the argmax of the bf16-rounded exact logits
            idx2 = ops.fp8_head_argmax(x, q, s, torch.empty(0, dtype=torch.long, device=dev), w, 2.0, 0, 0)
            yb = (x.double() @ w.double().T).to(torch.bfloat16).float()
            # fp32 vs fp64 dot can flip a bf16 rounding: accept any row that hits the bf16 max
            okr = (yb.gather(1, idx2[:, None])[:, 0] == yb.max(1).values)
            assert bool(okr.all()), (idx2, yb.argmax(1))
        # tie: two identical rows -> lower index
        w2 = w.clone()
        if N >= 1000:
            w2[900] = w2[100]
            q2, s2 = ops.quantize_rows_fp8(w2, torch.empty(0, dtype=torch.long, device=dev), N)
            x = w2[100:101].float().mul(10).to(torch.bfloat16)
            idx = ops.fp8_head_argmax(x, q2, s2, torch.empty(0, dtype=torch.long, device=dev), EB, 0.0, 0, 0)
            assert idx.item() == 100, idx
        print(f"head N={N}: OK (M=1..4, cfg 0..5, idmap, ties)")
    # graph capture
    N = 65536
    w = (torch.randn(N, 5120, device=dev) * 0.02).to(torch.bfloat16)
    q, s = ops.quantize_rows_fp8(w, torch.empty(0, dtype=torch.long, device=dev), N)
    x = torch.randn(2, 5120, device=dev).to(torch.bfloat16)
    e = torch.empty(0, dtype=torch.long, device=dev)
    ops.fp8_head_argmax(x, q, s, e, w, 1.0, -1, 0)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = ops.fp8_head_argmax(x, q, s, e, w, 1.0, -1, 0)
    for _ in range(3):
        x.copy_(torch.randn(2, 5120, device=dev).to(torch.bfloat16))
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ops.fp8_head_argmax(x, q, s, e, w, 1.0, -1, 0))
    print("head: graph replay OK")


def load_lm_head():
    from safetensors import safe_open
    with safe_open("/model/outside.safetensors", "pt", device="cpu") as f:
        return f.get_tensor("lm_head.weight").to(dev)


def test_agree():
    W = load_lm_head()
    X = torch.load(HID).to(dev)
    sys.path.insert(0, "/vopt/mtp-draft/plugin")
    from vopt_lmhead import ext as lext
    lext.load_ext()
    full = []
    for i in range(X.shape[0]):
        full.append(torch.ops.vopt_lmhead.bf16_gemv_argmax(X[i:i + 1], W, -1, 0)[1])
    full = torch.cat(full)
    yref = (X.float() @ W.float().T)
    top2 = yref.topk(2, dim=1)
    margin = (top2.values[:, 0] - top2.values[:, 1])
    print(f"agree: {X.shape[0]} rows; top-1 margin median {margin.median():.3f}, <0.1: {(margin < 0.1).float().mean():.3f}")
    e = torch.empty(0, dtype=torch.long, device=dev)
    for N in (32768, 49152, 65536, 81920, 100000, 151552, 248320):
        inset = (full < N).float().mean().item()
        qN, sN = ops.quantize_rows_fp8(W, e, N)
        fp8 = torch.cat([ops.fp8_head_argmax(X[i:i + 1], qN, sN, e, EB, 0.0, 0, 0) for i in range(X.shape[0])])
        fr = torch.cat([ops.fp8_head_argmax(X[i:i + 1], qN, sN, e, W, 1.0, 0, 0) for i in range(X.shape[0])])
        bfN = torch.cat([torch.ops.vopt_lmhead.bf16_gemv_argmax(X[i:i + 1], W[:N], -1, 0)[1] for i in range(X.shape[0])])
        print(f"agree N={N:6d}: target argmax in subset {inset:.4f} | bf16 subset == full {(bfN == full).float().mean():.4f}"
              f" | fp8 == full {(fp8 == full).float().mean():.4f} (== bf16 subset {(fp8 == bfN).float().mean():.4f})"
              f" | fp8+rescore == full {(fr == full).float().mean():.4f} (== bf16 subset {(fr == bfN).float().mean():.4f})")
        del qN, sN


def bench_graph(fn, iters=50):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    g.replay()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters / 10 * 1e6


def test_bench():
    W = load_lm_head()
    sys.path.insert(0, "/vopt/mtp-draft/plugin")
    from vopt_lmhead import ext as lext
    lext.load_ext()
    e = torch.empty(0, dtype=torch.long, device=dev)
    for N in (32768, 65536, 100000, 248320):
        qN, sN = ops.quantize_rows_fp8(W, e, N)
        for M in (1, 2, 4):
            x = torch.randn(M, 5120, device=dev).to(torch.bfloat16)
            ts = {c: bench_graph(lambda: ops.fp8_head_argmax(x, qN, sN, e, W, 1.0, c, 0)) for c in range(6)}
            best = min(ts, key=ts.get)
            bf = bench_graph(lambda: torch.ops.vopt_lmhead.bf16_gemv_argmax(x, W[:N], -1, 0))
            gbs = N * 5120 / ts[best] / 1e3
            print(f"bench N={N:6d} M={M}: fp8 " + " ".join(f"c{c}:{t:.0f}" for c, t in ts.items())
                  + f" us  (best c{best} {gbs:.0f} GB/s) | bf16 subset {bf:.0f} us")
        del qN, sN


def test_i4():
    """int4 head: layout round trip, argmax vs fp64 of the dequantized weights, rescoring, ties, graph."""
    from vopt_spec.quant import quant_int4, dequant_int4
    torch.manual_seed(5)
    e = torch.empty(0, dtype=torch.long, device=dev)
    for V, N, sub in ((1001, 1001, False), (70001, 65537, True), (200000, 151585, True)):
        w = (torch.randn(V, 5120, device=dev) * 0.02).to(torch.bfloat16)
        rows = (torch.randperm(V, device=dev)[:N].sort().values if sub else torch.arange(N, device=dev))
        q, ab = quant_int4(w, rows)
        if N <= 70000:
            dq = dequant_int4(q, ab, N)
            rel = ((dq - w[rows].double()).norm(dim=1) / w[rows].double().norm(dim=1)).mean().item()
        else:
            dq, rel = None, float("nan")
        for M in (1, 2, 3, 4):
            x = (torch.randn(M, 5120, device=dev) * 3).to(torch.bfloat16)
            for cfg in range(4):
                idx = ops.i4_head_argmax(x, q, ab, N, rows, EB, 0.0, cfg, 0)
                if dq is not None:
                    y = x.float().clamp(-65504, 65504).half().double() @ dq.T
                    got = y.gather(1, torch.searchsorted(rows, idx)[:, None])[:, 0]
                    gap = y.max(1).values - got
                    tol = 1e-3 * y.abs().max(dim=1).values + 1e-5
                    if bool((gap > tol).any()):
                        print(f"i4 FAIL V={V} N={N} M={M} cfg={cfg} gap={gap.tolist()}")
                    assert not bool((gap > tol).any())
                else:
                    assert torch.equal(idx, ops.i4_head_argmax(x, q, ab, N, rows, EB, 0.0, 0, 0))
            idx2 = ops.i4_head_argmax(x, q, ab, N, rows, w, 3.0, -1, 0)
            yb = (x.double() @ w[rows].double().T).to(torch.bfloat16).float()
            okr = yb.gather(1, torch.searchsorted(rows, idx2)[:, None])[:, 0] == yb.max(1).values
            # random weights: many rows inside the window at large N can force the int4 fallback
            assert bool(okr.all()) or N > 70000, (idx2, rows[yb.argmax(1)])
        print(f"i4 V={V} N={N}: OK (per-row rel L2 err {rel:.4f}; M=1..4, cfg 0..3, rescoring)")
    # graph replay
    x = torch.randn(1, 5120, device=dev).to(torch.bfloat16)
    ops.i4_head_argmax(x, q, ab, N, rows, w, 3.0, -1, 0)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = ops.i4_head_argmax(x, q, ab, N, rows, w, 3.0, -1, 0)
    for _ in range(3):
        x.copy_(torch.randn(1, 5120, device=dev).to(torch.bfloat16))
        g.replay(); torch.cuda.synchronize()
        assert torch.equal(out, ops.i4_head_argmax(x, q, ab, N, rows, w, 3.0, -1, 0))
    print("i4: graph replay OK")


def test_agree4():
    """int4 head on the real lm_head and real inputs vs the full bf16 lm_head argmax."""
    from vopt_spec.quant import quant_int4
    W = load_lm_head()
    X = torch.load(HID).to(dev)
    sys.path.insert(0, "/vopt/mtp-draft/plugin")
    from vopt_lmhead import ext as lext
    lext.load_ext()
    full = torch.cat([torch.ops.vopt_lmhead.bf16_gemv_argmax(X[i:i + 1], W, -1, 0)[1] for i in range(X.shape[0])])
    for N in (100000, 151552, 248320):
        rows = torch.arange(N, device=dev)
        if N < 248044:
            rows = torch.cat([rows, torch.arange(248044, 248077, device=dev)])
        q, ab = quant_int4(W, rows)
        Nr = rows.numel()
        res = {}
        for d in (0.0, 2.0, 3.0, 4.0):
            r = torch.cat([ops.i4_head_argmax(X[i:i + 1], q, ab, Nr, rows, W if d > 0 else EB, d, -1, 0)
                           for i in range(X.shape[0])])
            res[d] = (r == full).float().mean().item()
        x = X[:1]
        ts = {c: bench_graph(lambda: ops.i4_head_argmax(x, q, ab, Nr, rows, W, 3.0, c, 0)) for c in range(4)}
        x2 = X[:2]
        t2 = bench_graph(lambda: ops.i4_head_argmax(x2, q, ab, Nr, rows, W, 3.0, -1, 0))
        x4 = X[:4]
        t4 = bench_graph(lambda: ops.i4_head_argmax(x4, q, ab, Nr, rows, W, 3.0, -1, 0))
        mb = (q.numel() * 4 + ab.numel() * 4) / 2 ** 20
        print(f"agree4 N={Nr}: == full argmax " + " ".join(f"d{d:.0f}:{v:.4f}" for d, v in res.items())
              + f" | {mb:.0f} MiB | M=1 " + " ".join(f"c{c}:{t:.0f}" for c, t in ts.items())
              + f" us, M=2 {t2:.0f} us, M=4 {t4:.0f} us")
        del q, ab


def test_nan():
    """Garbage rows (NaN / inf / huge, as in padded requests of a captured batch) must still give an
    in-range vocab id (it is fed to the embedding of the next draft step)."""
    from vopt_spec.quant import quant_int4
    torch.manual_seed(7)
    V, N = 5000, 3001
    w = (torch.randn(V, 5120, device=dev) * 0.02).to(torch.bfloat16)
    rows = torch.arange(N, device=dev) + 1000
    q8, s8 = ops.quantize_rows_fp8(w, rows, 0)
    q4, ab = quant_int4(w, rows)
    bad = 0
    for kind in ("nan", "inf", "-inf", "big", "mix"):
        x = torch.randn(4, 5120, device=dev)
        if kind == "nan": x[:] = float("nan")
        if kind == "inf": x[:] = float("inf")
        if kind == "-inf": x[:] = float("-inf")
        if kind == "big": x[:] = 3e38
        if kind == "mix": x[1] = float("nan"); x[3, ::7] = float("inf")
        x = x.to(torch.bfloat16)
        for d in (0.0, 3.0):
            wb = w if d > 0 else EB
            for idx in (ops.fp8_head_argmax(x, q8, s8, rows, wb, d, -1, 0),
                        ops.i4_head_argmax(x[:2], q4, ab, N, rows, wb, d, -1, 0),
                        ops.i4_head_argmax(x[2:], q4, ab, N, rows, wb, d, -1, 0)):
                ok = bool(((idx >= 1000) & (idx < 1000 + N)).all())
                bad += not ok
                if not ok:
                    print(f"nan FAIL {kind} delta={d}: {idx.tolist()}")
    assert bad == 0
    print("nan/inf rows: in-range ids OK")


def test_verify():
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
    torch.manual_seed(3)
    V = 248320
    nbad = 0
    ntie = 0
    for trial in range(200):
        R = int(torch.randint(1, 5, ()).item())
        K = int(torch.randint(1, 5, ()).item())
        T = K + 1
        L = R * T
        logits = (torch.randn(L, V, device=dev) * 3).to(torch.bfloat16)
        amax = logits.argmax(1)
        # plant exact ties at the top: copy the max into a random other column (both sides)
        for i in range(L):
            if torch.rand(()) < 0.5:
                j = int(torch.randint(0, V, ()).item())
                logits[i, j] = logits[i, amax[i]]
                ntie += 1
        amax = logits.argmax(1)  # torch: first index of the max
        # drafts: accept a random prefix
        draft = torch.randint(0, V, (L,), device=dev)
        for r in range(R):
            a = int(torch.randint(0, T, ()).item())
            for i in range(a):
                draft[r * T + i + 1] = amax[r * T + i]
            if torch.rand(()) < 0.1:
                draft[r * T + 1] = -1
        cu = torch.arange(0, R + 1, device=dev, dtype=torch.int32) * T
        idx_mapping = torch.arange(R, device=dev, dtype=torch.int32)
        exp_idx = idx_mapping.repeat_interleave(T)
        exp_pos = torch.arange(T, device=dev, dtype=torch.int32).repeat(R)
        pos = torch.arange(L, device=dev, dtype=torch.int64) + 100
        temp = torch.zeros(R, device=dev)
        seeds = torch.zeros(R, dtype=torch.int64, device=dev)
        s_ref, n_ref = rejection_sample(logits, None, draft, cu, pos, idx_mapping, exp_idx, exp_pos, temp,
                                        seeds, K)
        # mine: input_ids[logits_indices] == draft
        input_ids = draft.to(torch.int32)
        li = torch.arange(L, device=dev, dtype=torch.int64)
        s_me, n_me = ops.greedy_verify(amax, input_ids, li, cu, K)
        if not torch.equal(n_ref, n_me):
            nbad += 1
            print("verify num_sampled mismatch", n_ref.tolist(), n_me.tolist())
            continue
        for r in range(R):
            n = int(n_ref[r])
            if not torch.equal(s_ref[r, :n], s_me[r, :n]):
                nbad += 1
                print("verify tokens mismatch", r, s_ref[r, :n].tolist(), s_me[r, :n].tolist())
    print(f"verify: 200 random batches (R 1..4, K 1..4, {ntie} planted top-1 ties): {nbad} mismatches vs vLLM rejection_sample")
    assert nbad == 0


if __name__ == "__main__":
    for t in ("quant", "head", "i4", "nan", "verify", "agree", "agree4", "bench"):
        if t in ARGS:
            globals()["test_" + t]()
