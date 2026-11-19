#!/usr/bin/env python3
"""TTFT of a ~512-token prompt (stdlib, host side): max_tokens=1, non-streaming, 1 warmup + N reps.
usage: ttft.py --base URL [--words 400] [--reps 5] [--out f.json]"""
import argparse, json, statistics, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--model", default="qwen38")
ap.add_argument("--words", type=int, default=420)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--out", default="")
a = ap.parse_args()
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
words = ("alpha beta gamma delta lighthouse keeper river mountain story map ocean " * 100).split()
res = []
for i in range(a.reps + 1):
    prompt = f"[{i}] " + " ".join(words[: a.words])  # distinct prefix per rep: no prefix-cache hit
    body = {"model": a.model, "prompt": prompt, "max_tokens": 1, "temperature": 0}
    req = urllib.request.Request(a.base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with op.open(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = (time.perf_counter() - t0) * 1000
    if i:
        res.append({"ms": dt, "prompt_tokens": d["usage"]["prompt_tokens"]})
out = {"ttft_ms_median": statistics.median(r["ms"] for r in res), "runs": res}
print(json.dumps(out))
if a.out:
    json.dump(out, open(a.out, "w"), indent=1)
