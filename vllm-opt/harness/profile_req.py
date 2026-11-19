#!/usr/bin/env python3
"""Profile one single-stream decode request with the vLLM torch profiler
(server must be started with PROFILE=1). usage: profile_req.py --base URL [--n-out 64]"""
import argparse, json, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--model", default="qwen38")
ap.add_argument("--n-out", type=int, default=64)
ap.add_argument("--sampling", action="store_true")
ap.add_argument("--no-min-tokens", action="store_true")
a = ap.parse_args()
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(path, body=None):
    req = urllib.request.Request(a.base + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with op.open(req, timeout=900) as r:
        return r.read()


body = {"model": a.model, "prompt": "Write a long story about a lighthouse keeper.",
        "max_tokens": a.n_out, "min_tokens": a.n_out, "temperature": 0, "ignore_eos": True}
if a.sampling:
    del body["temperature"]
if a.no_min_tokens:
    del body["min_tokens"]
post("/v1/completions", dict(body, max_tokens=16, min_tokens=16))  # warm
post("/start_profile")
t = time.perf_counter(); post("/v1/completions", body); dt = time.perf_counter() - t
post("/stop_profile")
print(json.dumps({"profiled_request_s": dt, "n_out": a.n_out}))
