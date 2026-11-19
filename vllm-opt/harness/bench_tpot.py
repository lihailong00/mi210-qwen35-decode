#!/usr/bin/env python3
"""Single-stream decode TPOT bench (stdlib only, runs on the host).

Streams one /v1/completions request at a time (greedy, ignore_eos) and timestamps every
SSE chunk. TPOT = (t_last_chunk - t_first_chunk) / (tokens - 1), tokens =
usage.completion_tokens (falls back to the chunk count). Under MTP one chunk carries a whole
verify round, so a per-chunk TPOT would be 3-4x too high. First chunk = first token (TTFT).
Reports per-rep TPOT and the median, plus tok/s = 1000/TPOT.
usage: bench_tpot.py --base http://127.0.0.1:18100 [--n-out 512] [--reps 3] [--out f.json]
"""
import argparse, json, statistics, time, urllib.request

PROMPT = "Write a long, detailed story about a lighthouse keeper who discovers a hidden map."


def one(base, model, n_out, prompt, sampling=False, min_tokens=True):
    body = {"model": model, "prompt": prompt, "max_tokens": n_out, "min_tokens": n_out,
            "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    if not min_tokens:
        del body["min_tokens"]  # ignore_eos alone still yields n_out tokens
    if not sampling:
        body["temperature"] = 0  # greedy; --sampling keeps the model generation_config defaults
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); stamps = []; usage = None
    with op.open(req, timeout=600) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            p = raw[6:].strip()
            if p == b"[DONE]":
                break
            d = json.loads(p)
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices"):
                stamps.append(time.perf_counter())
    n = len(stamps)
    ct = (usage or {}).get("completion_tokens") or n
    tpot = (stamps[-1] - stamps[0]) / (ct - 1) * 1000 if n > 1 and ct > 1 else float("nan")
    # tpot_ss_ms kept as the reported field (token-based, from the first chunk)
    return {"ttft_ms": (stamps[0] - t0) * 1000, "chunks": n, "tokens": ct, "tpot_ms": tpot,
            "tpot_ss_ms": tpot, "usage": usage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--n-out", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--no-min-tokens", action="store_true", help="omit min_tokens (no EOS-ban logit bias)")
    ap.add_argument("--sampling", action="store_true", help="default sampling params (no temperature)")
    a = ap.parse_args()
    for _ in range(a.warmup):
        one(a.base, a.model, 32, PROMPT, a.sampling, not a.no_min_tokens)
    runs = [one(a.base, a.model, a.n_out, PROMPT, a.sampling, not a.no_min_tokens) for _ in range(a.reps)]
    med = statistics.median(r["tpot_ss_ms"] for r in runs)
    res = {"sampling": a.sampling, "min_tokens": not a.no_min_tokens, "n_out": a.n_out, "runs": runs, "median_tpot_ss_ms": med, "tok_s": 1000 / med}
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
