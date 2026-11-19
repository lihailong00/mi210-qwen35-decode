"""Unit test + timing for torch.ops.vopt_lmhead.sample_topk_topp (fused top-k/top-p/temperature
sampler fed by the lm_head GEMV argmax epilogue). Runs inside the vLLM image with the GPU lock held.
 - distribution test: reference kept set and probabilities from vLLM's torch reference
   (apply_top_k_top_p_pytorch on fp32 logits / T, then softmax) in fp64; NDRAW draws with different
   positions; checks support (no token outside the kept set) and total-variation distance / chi2;
 - greedy rows (temperature 0) must return the GEMV argmax;
 - timing (eager, events over 200 calls): fused op vs the stock path (fp32 copy + temperature +
   apply_top_k_top_p_triton + gumbel_sample), M = 1 and 4.
usage: python3 -m vopt_lmhead.test_sampler [--ndraw 20000]"""
import argparse, json, math, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vopt_lmhead import ext  # noqa: E402


def ref_probs(logits_row, T, k, p):
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
    l = (logits_row.float() / T).unsqueeze(0)
    kk = torch.tensor([k], device=l.device)
    pp = torch.tensor([p], device=l.device, dtype=torch.float32)
    masked = apply_top_k_top_p_pytorch(l.clone(), kk, pp)
    return torch.softmax(masked.double(), dim=-1)[0]


def timeit(fn, n=200):
    fn(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndraw", type=int, default=20000)
    a = ap.parse_args()
    ext.load_ext()
    ops = torch.ops.vopt_lmhead
    from safetensors import safe_open
    with safe_open("/model/outside.safetensors", "pt", device="cuda") as h:
        W = h.get_tensor("lm_head.weight")
    torch.manual_seed(0)
    res = {}
    dev = "cuda"
    for M in (1, 4):
        # hidden states with a realistic spread of logits (a peaked top, long tail)
        x = (torch.randn(M, 5120, device=dev) * 3.0).to(torch.bfloat16)
        y, amax = ops.bf16_gemv_argmax(x, W, -1, 0)
        idx_map = torch.arange(M, device=dev, dtype=torch.int32)
        temp = torch.full((M,), 1.0, device=dev)
        topk = torch.full((M,), 20, device=dev, dtype=torch.int32)
        topp = torch.full((M,), 0.95, device=dev)
        seeds = torch.randint(-2**62, 2**62, (M,), device=dev, dtype=torch.int64)
        if M == 4:  # mixed batch: row 1 greedy, row 2 T=0.7 k=50 p=0.8, row 3 k=5 p=1
            temp[1] = 0.0; temp[2] = 0.7; topk[2] = 50; topp[2] = 0.8; topk[3] = 5; topp[3] = 1.0
        # distribution test (row by row, positions vary -> independent noise)
        counts = torch.zeros(M, y.shape[1], device=dev, dtype=torch.float64)
        nd = a.ndraw if M == 1 else a.ndraw // 2
        for i in range(nd):
            pos = torch.full((M,), 1000 + i, device=dev, dtype=torch.int64)
            out = ops.sample_topk_topp(y, amax, idx_map, temp, topk, topp, seeds, pos)
            counts[torch.arange(M, device=dev), out] += 1
        for r in range(M):
            T, k, p = temp[r].item(), topk[r].item(), topp[r].item()
            if T == 0.0:
                ok = bool(counts[r, amax[r]].item() == nd)
                res[f"M{M}/row{r}/greedy_ok"] = ok
                continue
            pr = ref_probs(y[r], T, k, p)
            emp = counts[r] / nd
            support = pr > 0
            outside = int(counts[r][~support].sum().item())
            tv = 0.5 * (emp - pr).abs().sum().item()
            nz = support.nonzero().flatten()
            exp_c = pr[nz] * nd
            chi2 = (((counts[r][nz] - exp_c) ** 2) / exp_c.clamp_min(1e-12)).sum().item()
            # tie between the last kept and the first dropped value: which tied token top-p drops
            # is implementation-defined (torch sort order vs index order vs Triton pivots)
            sv_desc = (y[r].float() / T).sort(descending=True).values
            nkept = int(support.sum())
            boundary_tie = bool(nkept < y.shape[1] and sv_desc[nkept - 1].item() == sv_desc[nkept].item())
            res[f"M{M}/row{r}"] = {"T": T, "k": k, "p": round(p, 3), "kept": nkept,
                                   "boundary_tie": boundary_tie,
                                   "draws_outside_support": outside, "tv_distance": round(tv, 4),
                                   "chi2": round(chi2, 1), "dof": int(len(nz) - 1),
                                   "top_prob_ref": round(pr.max().item(), 4)}
        # timing: fused vs stock (eager, like vLLM's sampler)
        pos = torch.arange(M, device=dev, dtype=torch.int64)
        us_f = timeit(lambda: ops.sample_topk_topp(y, amax, idx_map, temp, topk, topp, seeds, pos))
        # GPU-only time: CUDA graph of 20 back-to-back calls (the op is graph-safe)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(20):
                ops.sample_topk_topp(y, amax, idx_map, temp, topk, topp, seeds, pos)
        g.replay(); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        us_fg = e0.elapsed_time(e1) * 1000 / 20
        # same with the per-request parameters in host-mapped (pinned) memory, as vLLM's UVA tensors
        tp_h, tk_h, pp_h, sd_h = (t.cpu().pin_memory() for t in (temp, topk, topp, seeds))
        us_uva = timeit(lambda: ops.sample_topk_topp(y, amax, idx_map, tp_h, tk_h, pp_h, sd_h, pos))
        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
        tk = topk[idx_map.long()]
        tp = topp[idx_map.long()]

        def stock():
            lf = torch.empty_like(y, dtype=torch.float32).copy_(y)
            lf = apply_top_k_top_p_triton(lf, tk, tp)
            return gumbel_sample(lf, idx_map, temp, seeds, pos, apply_temperature=True,
                                 is_drafting=False)
        us_s = timeit(stock)
        us_g = timeit(lambda: gumbel_sample(y, idx_map, temp * 0, seeds, pos, apply_temperature=False,
                                            is_drafting=False))
        res[f"M{M}/time_us"] = {"fused_sampler": round(us_f, 1), "fused_sampler_graph": round(us_fg, 1),
                                "fused_sampler_params_in_pinned_host": round(us_uva, 1), "stock_topk_topp_gumbel": round(us_s, 1),
                                "stock_greedy_gumbel_argmax": round(us_g, 1)}
        print(json.dumps({kk: v for kk, v in res.items() if kk.startswith(f"M{M}")}), flush=True)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
