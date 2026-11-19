#!/usr/bin/env python3
"""Per-round breakdown of single-stream MTP spec decode from a vLLM torch-profiler trace.
A round starts at each launch of the target verify graph (largest graph). Inside a round the
kernels are grouped into segments: one segment per graph launch (G<size>) and one per run of
eager kernels (E). Prints the median segment timeline over steady rounds, then per-kernel
tables for the verify graph and for each draft graph, with FP8 GEMV byte floors.
usage: analyze_mtp.py trace.json.gz [--skip 3] [--json out.json]"""
import argparse, bisect, collections, gzip, json, re, statistics, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from analyze_trace import cat_of, short, PEAK, LAYER_TYPES, ROLES

med = statistics.median


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--json")
    a = ap.parse_args()
    ev = json.load(gzip.open(a.trace) if a.trace.endswith(".gz") else open(a.trace))["traceEvents"]
    rt = {}
    for e in ev:
        if e.get("cat") in ("cuda_runtime", "hip_runtime", "cuda_driver") and "args" in e:
            c = e["args"].get("correlation")
            if c is not None:
                rt[c] = e
    ks = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")), key=lambda e: e["ts"])
    gid = []
    for k in ks:
        c = k["args"].get("correlation")
        r = rt.get(c)
        gid.append(c if (r is not None and r["name"] in ("hipGraphLaunch", "cudaGraphLaunch")) else None)
    gsize = collections.Counter(g for g in gid if g is not None)
    sizes = collections.Counter(gsize.values())
    print("graph sizes (kernels per launch: launches):", dict(sorted(sizes.items())))
    big = max(sizes)
    # segments
    segs = []  # (label, [kernels])
    for k, g in zip(ks, gid):
        lab = ("G%d" % gsize[g], g) if g is not None else ("E", None)
        if segs and segs[-1][0] == lab:
            segs[-1][1].append(k)
        else:
            segs.append((lab, [k]))
    starts = [i for i, (lab, _) in enumerate(segs) if lab[0] == "G%d" % big]
    rounds = [segs[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]
    sig = collections.Counter(tuple(l[0] for l, _ in r) for r in rounds)
    print(f"rounds={len(rounds)}; structures:")
    for s, n in sig.most_common(5):
        print(f"  {n:4d} x {chr(32).join(s)}")
    main_sig = sig.most_common(1)[0][0]
    use = [r for r in rounds[a.skip:] if tuple(l[0] for l, _ in r) == main_sig]
    print(f"steady rounds used: {len(use)}")
    per = [r[-1][1][-1]["ts"] + r[-1][1][-1]["dur"] - r[0][1][0]["ts"] for r in use]
    nxt = []
    for r in use:
        i = rounds.index(r)
        if i + 1 < len(rounds):
            nxt.append(rounds[i + 1][0][1][0]["ts"] - r[0][1][0]["ts"])
    print(f"round period median {med(nxt)/1e3:.3f} ms (first-kernel to next round first-kernel)")
    print("\n== segment timeline (median over rounds): label, kernels, busy us, span us, gap-after us")
    tl = []
    for j, lab in enumerate(main_sig):
        busy = med([sum(k["dur"] for k in r[j][1]) for r in use])
        span = med([r[j][1][-1]["ts"] + r[j][1][-1]["dur"] - r[j][1][0]["ts"] for r in use])
        gaps = []
        for r in use:
            end = r[j][1][-1]["ts"] + r[j][1][-1]["dur"]
            if j + 1 < len(r):
                gaps.append(r[j + 1][1][0]["ts"] - end)
            else:
                i = rounds.index(r)
                if i + 1 < len(rounds):
                    gaps.append(rounds[i + 1][0][1][0]["ts"] - end)
        n = len(use[0][j][1])
        names = collections.Counter(short(k["name"], 60) for k in use[0][j][1])
        top = ", ".join(f"{nm}x{c}" for nm, c in names.most_common(4)) if lab == "E" else ""
        print(f"  {j:2d} {lab:6s} n={n:4d} busy {busy:9.1f} span {span:9.1f} gap {med(gaps) if gaps else 0:8.1f}  {top}")
        tl.append({"label": lab, "n": n, "busy_us": busy, "span_us": span, "gap_after_us": med(gaps) if gaps else 0,
                   "eager_names": dict(names) if lab == "E" else None})
    out = {"round_period_ms": med(nxt) / 1e3, "timeline": tl, "graphs": {}}
    # per graph kind tables
    for j, lab in enumerate(main_sig):
        if lab == "E":
            continue
        n = len(use[0][j][1])
        pos = [(use[0][j][1][p]["name"], med([r[j][1][p]["dur"] for r in use])) for p in range(n)]
        bycat = collections.defaultdict(lambda: [0.0, 0])
        byname = collections.defaultdict(lambda: [0.0, 0])
        fp8 = []
        for nm, d in pos:
            c = cat_of(nm)
            bycat[c][0] += d; bycat[c][1] += 1
            byname[short(nm)][0] += d; byname[short(nm)][1] += 1
            if c == "fp8_gemm":
                fp8.append(d)
        print(f"\n== segment {j} ({lab}): per category")
        for c, (d, k) in sorted(bycat.items(), key=lambda x: -x[1][0]):
            print(f"  {c:22s} {d:10.1f} us  n={k}")
        print("  -- per kernel name (us, launches, us/launch)")
        for nm, (d, k) in sorted(byname.items(), key=lambda x: -x[1][0])[:25]:
            print(f"  {d:9.1f} {k:4d} {d/k:8.2f}  {nm}")
        g = {"cat": {c: v[0] for c, v in bycat.items()}, "names": {k2: v for k2, v in byname.items()},
             "positions": [(short(nm, 160), d) for nm, d in pos]}
        roles = []
        for t in LAYER_TYPES:
            roles += ROLES[t]
        if len(fp8) == len(roles):
            agg = collections.defaultdict(lambda: [0.0, 0, 0])
            for (r, N, K), d in zip(roles, fp8):
                agg[r][0] += d; agg[r][1] += 1; agg[r][2] += N * K
            print("  -- FP8 GEMV by role: us, n, GB, floor us, GB/s")
            tot = [0, 0]
            for r, (d, k, b) in agg.items():
                print(f"  {r:14s} {d:9.1f} n={k:3d} {b/1e9:6.3f} floor {b/PEAK*1e6:8.1f}  {b/(d*1e-6)/1e9:6.0f} GB/s")
                tot[0] += d; tot[1] += b
            print(f"  {chr(84)+chr(79)+chr(84):14s} {tot[0]:9.1f} {tot[1]/1e9:6.3f} GB floor {tot[1]/PEAK*1e6:8.1f}")
            g["fp8_roles"] = {r: {"us": v[0], "n": v[1], "bytes": v[2]} for r, v in agg.items()}
        else:
            print(f"  (fp8 kernels: {len(fp8)}, {sum(fp8):.1f} us)")
        out["graphs"]["%d_%s" % (j, lab)] = g
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
