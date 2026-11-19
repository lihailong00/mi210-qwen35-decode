#!/usr/bin/env python3
"""Per-kernel breakdown of steady-state single-stream decode from a vLLM torch-profiler
(kineto) trace taken with FULL_DECODE_ONLY graphs.

Method: every kernel replayed from a HIP graph carries the correlation id of its
hipGraphLaunch; each graph launch = one decode forward. Eager kernels (sampler, input prep)
are assigned to the step whose graph launch precedes them. We skip the first SKIP and the
last steps, and report per decode step (= per output token for 1 request, no MTP):
  period (graph-first-kernel to next graph-first-kernel), GPU busy sum, idle,
  per-category / per-kernel-name median time, FP8 GEMMs split by role with byte floors.
usage: analyze_trace.py trace.json.gz [--skip 4] [--json out.json] [--positions]
"""
import argparse, collections, gzip, json, re, statistics

PEAK = 1446e9  # measured read peak B/s
LAYER_TYPES = (["lin"] * 3 + ["full"]) * 16
H, I = 5120, 17408
ROLES = {  # role -> (N, K) per layer type
    "lin": [("gdn_in_qkvz", 16384, H), ("gdn_out", H, 6144), ("mlp_gate_up", 2 * I, H), ("mlp_down", H, I)],
    "full": [("attn_qkv", 14336, H), ("attn_o", H, 6144), ("mlp_gate_up", 2 * I, H), ("mlp_down", H, I)],
}
CATS = [
    ("fp8_gemm", r"block_scaled_mm|w8a8|_mi210_exact|fp8_gemv|scaled_mm"),
    ("act_quant", r"per_token_group_quant|quant_fp8|_per_token_group|dynamic.*quant|scaled_fp8_quant"),
    ("lm_head/bf16_gemm", r"wvSplitK|LLMM1|Cijk_|gemm|gemv|wvSpltK|skinny"),
    ("gdn_recurrent", r"fused_recurrent|sigmoid_gating|delta_rule|gated_delta"),
    ("gdn_conv", r"causal_conv1d"),
    ("gdn_misc", r"gdn|l2norm|fused_gdn_gating|rearrange"),
    ("attention", r"unified_attention|attention|reduce_segments|paged"),
    ("norm", r"gemma_rms|rms_norm|layernorm|RMSNorm|rmsnorm|norm"),
    ("rope", r"rope|rotary"),
    ("act/elementwise", r"silu|act_and_mul|elementwise|sigmoid|mul|add|copy|fill|cat|index|where"),
    ("sampler", r"sample|argmax|topk|top_k|softmax|gumbel|logsumexp|reduce|max"),
    ("memcpy", r"copyBuffer|memcpy|Memcpy|fillBuffer"),
]


def cat_of(name):
    for c, rx in CATS:
        if re.search(rx, name, re.I):
            return c
    return "other"


def short(name, n=90):
    name = re.sub(r"\(.*", "", name) if not name.startswith("void") else name
    return name[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--skip", type=int, default=4)
    ap.add_argument("--json")
    ap.add_argument("--positions", action="store_true")
    a = ap.parse_args()
    op = gzip.open if a.trace.endswith(".gz") else open
    ev = json.load(op(a.trace))["traceEvents"]
    rt = {}
    for e in ev:
        if e.get("cat") in ("cuda_runtime", "hip_runtime", "cuda_driver") and "args" in e:
            c = e["args"].get("correlation")
            if c is not None:
                rt[c] = e
    ks = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    mc = sorted((e for e in ev if e.get("cat") == "gpu_memcpy" or e.get("cat") == "gpu_memset"), key=lambda e: e["ts"])
    graphs = collections.OrderedDict()
    eager = []
    for k in ks:
        c = k["args"].get("correlation")
        r = rt.get(c)
        if r is not None and r["name"] in ("hipGraphLaunch", "cudaGraphLaunch"):
            graphs.setdefault(c, []).append(k)
        else:
            eager.append(k)
    eager += mc
    eager.sort(key=lambda e: e["ts"])
    gl = list(graphs.values())
    sizes = collections.Counter(len(g) for g in gl)
    print(f"kernels={len(ks)} graph_launches={len(gl)} graph sizes={dict(sizes)} eager={len(eager)}")
    main_size = sizes.most_common(1)[0][0]
    steps = [g for g in gl if len(g) == main_size]
    starts = [g[0]["ts"] for g in steps]
    # assign eager kernels to steps
    import bisect
    per_step_eager = [[] for _ in steps]
    for e in eager:
        i = bisect.bisect_right(starts, e["ts"]) - 1
        if 0 <= i < len(steps):
            per_step_eager[i].append(e)
    use = list(range(a.skip, len(steps) - 1))
    periods, busy, gspan, gbusy, ebusy = [], [], [], [], []
    for i in use:
        g = steps[i]
        p = starts[i + 1] - starts[i]
        gb = sum(k["dur"] for k in g)
        eb = sum(k["dur"] for k in per_step_eager[i] if k["ts"] < starts[i + 1])
        periods.append(p); gbusy.append(gb); ebusy.append(eb); busy.append(gb + eb)
        gspan.append(g[-1]["ts"] + g[-1]["dur"] - g[0]["ts"])
    med = statistics.median
    print(f"steady steps used={len(use)} kernels/graph={main_size}")
    print(f"period(=TPOT on GPU clock) median {med(periods)/1e3:.3f} ms; graph span {med(gspan)/1e3:.3f} ms; "
          f"graph kernel sum {med(gbusy)/1e3:.3f} ms; eager kernel sum {med(ebusy)/1e3:.3f} ms; "
          f"idle (period - busy) {med([p-b for p,b in zip(periods,busy)])/1e3:.3f} ms")
    # per-position medians in graph
    pos = []
    for j in range(main_size):
        names = steps[use[0]][j]["name"]
        d = med([steps[i][j]["dur"] for i in use])
        pos.append((names, d))
    # graph-internal gaps
    gaps = med([sum(max(0, g[j + 1]["ts"] - (g[j]["ts"] + g[j]["dur"])) for j in range(len(g) - 1)) for g in (steps[i] for i in use)])
    print(f"graph-internal gaps between consecutive kernels (sum) {gaps/1e3:.3f} ms over {main_size-1} boundaries "
          f"= {gaps/(main_size-1):.2f} us each")
    # eager per-step: names
    en = collections.defaultdict(list)
    for i in use:
        cnt = collections.defaultdict(float)
        for e in per_step_eager[i]:
            if e["ts"] < starts[i + 1]:
                cnt[short(e["name"])] += e["dur"]
        for n, v in cnt.items():
            en[n].append(v)
    # categories
    bycat = collections.defaultdict(lambda: [0.0, 0])
    byname = collections.defaultdict(lambda: [0.0, 0])
    fp8 = []
    for n, d in pos:
        c = cat_of(n)
        bycat[c][0] += d; bycat[c][1] += 1
        byname[short(n)][0] += d; byname[short(n)][1] += 1
        if c == "fp8_gemm":
            fp8.append(d)
    for n, v in en.items():
        d = sum(v) / len(use)
        c = "eager:" + cat_of(n)
        bycat[c][0] += d; bycat[c][1] += 1
        byname["[eager] " + n][0] += d; byname["[eager] " + n][1] += 1
    print("\n== per category (us per decode step, count)")
    for c, (d, n) in sorted(bycat.items(), key=lambda x: -x[1][0]):
        print(f"  {c:22s} {d:10.1f} us  n={n}")
    print("\n== per kernel name (us per step, launches per step, us per launch)")
    for nme, (d, n) in sorted(byname.items(), key=lambda x: -x[1][0])[:45]:
        print(f"  {d:9.1f} {n:4d} {d/n:8.2f}  {nme}")
    roles = []
    for t in LAYER_TYPES:
        roles += ROLES[t]
    out = {"period_ms": med(periods) / 1e3, "graph_sum_ms": med(gbusy) / 1e3, "eager_sum_ms": med(ebusy) / 1e3,
           "kernels_per_graph": main_size, "cat": {c: v[0] for c, v in bycat.items()}}
    if len(fp8) == len(roles):
        agg = collections.defaultdict(lambda: [0.0, 0, 0])
        for (r, N, K), d in zip(roles, fp8):
            agg[r][0] += d; agg[r][1] += 1; agg[r][2] += N * K
        print("\n== FP8 GEMMs by role (per step): time, count, bytes, byte floor @1446GB/s, floor+1.6us/launch, achieved GB/s")
        tot = [0, 0, 0]
        for r, (d, n, b) in agg.items():
            fl = b / PEAK * 1e6
            print(f"  {r:14s} {d:9.1f} us n={n:3d} {b/1e9:6.3f} GB floor {fl:8.1f} us  (+launch {fl+1.6*n:8.1f})  {b/(d*1e-6)/1e9:6.0f} GB/s")
            tot[0] += d; tot[1] += b; tot[2] += n
        print(f"  {'TOTAL':14s} {tot[0]:9.1f} us n={tot[2]} {tot[1]/1e9:6.3f} GB floor {tot[1]/PEAK*1e6:8.1f} us")
        out["fp8_roles"] = {r: {"us": v[0], "n": v[1], "bytes": v[2]} for r, v in agg.items()}
    else:
        print(f"\n(fp8 gemm count {len(fp8)} != expected {len(roles)}; role split skipped)")
    if a.positions:
        print("\n== first 60 graph positions (median us)")
        for j, (n, d) in enumerate(pos[:60]):
            print(f"  {j:4d} {d:8.2f} {cat_of(n):16s} {short(n, 110)}")
    if a.json:
        out["positions"] = [(short(n, 200), d) for n, d in pos]
        out["eager"] = {n: sum(v) / len(use) for n, v in en.items()}
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
