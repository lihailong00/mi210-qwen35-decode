#!/usr/bin/env python3
"""Deployment benchmark (stdlib only, runs on the host against a running vLLM server).

Modes (each writes one JSON into --out-dir):
  tpot  : single-stream decode. Prompt = exactly --prompt-tokens tokens (text tokenized by the
          server's /tokenize, sent as a token-id list; a distinct first token per rep so prefix
          caching never helps). Streams --n-out tokens (greedy, ignore_eos, no min_tokens unless
          --min-tokens), TPOT = (t_last_chunk - t_first_chunk) / (usage.completion_tokens - 1)
          (field tpot_ss_ms; token-based, so it is right under MTP where one chunk carries a
          whole round; tpot_chunk_ss_ms is the old per-chunk number). --reps repeats, median.
  ttft  : prefill time of a --prompt-tokens prompt: max_tokens=1, non-streaming, wall time of the
          request, distinct prompt per rep, 1 warmup + --reps.
  ppl   : prompt_logprobs of a fixed text (PPL_TEXT below, ~700 tokens; or the first
          --prompt-tokens tokens of --text-file): per-token logprobs,
          mean NLL and perplexity; compare two files with --compare a.json b.json.
"""
import argparse, json, math, statistics, sys, time, urllib.request

OP = urllib.request.build_opener(urllib.request.ProxyHandler({}))

FILL = ("The harbour town woke slowly. Fishing boats rocked against the pier while gulls argued "
        "over scraps, and the old lighthouse keeper climbed the spiral stairs to polish the lens "
        "one more time. He kept a notebook of every ship that passed, the wind direction, the "
        "colour of the sky at dawn, and small observations about the people who walked the "
        "breakwater in the evening. ")

PPL_TEXT = """A memory hierarchy exists because no single storage technology is simultaneously large, fast and cheap. Registers sit closest to the arithmetic units and can be read every cycle, but there are only a few hundred of them per thread. Caches hold recently used data in small static memories; a hit costs a few nanoseconds, while a miss must travel to dynamic memory and costs a hundred times more. Programs run quickly when the data they touch next is already near the processor, which is why access patterns matter as much as instruction counts.

Graphics processors push this trade-off further. Thousands of threads are kept resident so that, while one group waits for memory, another can issue arithmetic. The hardware does not try to make any single access fast; it tries to keep enough accesses in flight that the memory system is never idle. A kernel that reads a large matrix once, multiplies it by a short vector and writes a small result is limited almost entirely by how many bytes per second the memory can deliver. For such kernels the useful question is not how many operations are performed but what fraction of the peak bandwidth is achieved.

Several details decide that fraction. Loads should be wide and aligned so that each request moves a full cache line. Neighbouring threads should read neighbouring addresses, so that the hardware can merge their requests. Enough independent loads must be issued before the first one is consumed, otherwise the processor stalls while the queue drains. At the end of a kernel some processors finish early and wait for the last ones, and this tail can cost several microseconds; when a kernel only lasts a few tens of microseconds, that waste is significant. Finally, launching a kernel is not free: the command processor has to fetch the launch packet, set up the work groups and, on a cold instruction cache, fetch the program itself.

Language models expose these effects directly. Generating a single token requires reading every weight of the network once, so the time per token is roughly the model size divided by the memory bandwidth, plus a fixed cost for each of the many small operations between the large matrix products. Normalisation layers, activation functions, rotary position embeddings and the bookkeeping of the attention cache each launch their own kernels unless they are fused. On a device with a bandwidth of about one and a half terabytes per second, a model with twenty-five gigabytes of weights cannot produce more than about sixty tokens per second, and every additional microsecond spent per layer moves the real speed further from that bound.

Engineers therefore approach the problem from two sides. They write matrix-vector kernels that stream the weights at close to peak bandwidth, decoding compressed formats on the fly so that fewer bytes have to be moved. And they merge the small operations into a handful of kernels per layer, record the whole sequence once as a graph and replay it, so that the processor spends its time moving weights instead of starting and stopping work. Measuring each step against its physical floor shows where the remaining time goes and which change is worth making next."""


def post(base, path, body, timeout=900):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with OP.open(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokens_of(base, model, n, salt):
    txt = FILL * (n // 60 + 2)
    ids = post(base, "/tokenize", {"model": model, "prompt": txt, "add_special_tokens": False})["tokens"]
    num = post(base, "/tokenize", {"model": model, "prompt": f"Story {salt}.", "add_special_tokens": False})["tokens"]
    ids = num + ids
    assert len(ids) >= n, (len(ids), n)
    return ids[:n]


def stream(base, model, prompt, n_out, sampling, min_tokens):
    body = {"model": model, "prompt": prompt, "max_tokens": n_out, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    if min_tokens:
        body["min_tokens"] = n_out
    if not sampling:
        body["temperature"] = 0
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); st = []; usage = None
    with OP.open(req, timeout=900) as r:
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
                st.append(time.perf_counter())
    n = len(st); k = min(16, n // 4)
    # With MTP one SSE chunk carries a whole verify round (up to K+1 tokens), so chunk gaps
    # are NOT per-token times. Per-token TPOT = decode span / (completion_tokens - 1); the
    # first chunk holds exactly the prefill token.
    ct = (usage or {}).get("completion_tokens") or n
    tpot_tok = (st[-1] - st[0]) / max(ct - 1, 1) * 1000
    return {"ttft_ms": (st[0] - t0) * 1000, "chunks": n, "tokens": ct,
            "tpot_chunk_ss_ms": (st[-1] - st[k]) / (n - 1 - k) * 1000,
            "tpot_ss_ms": tpot_tok, "usage": usage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tpot", "ttft", "ppl", "compare"])
    ap.add_argument("--base", default="")
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--n-out", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--sampling", action="store_true")
    ap.add_argument("--min-tokens", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--text-file", default="", help="ppl: score this file instead of PPL_TEXT")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    res = {}
    if a.mode == "tpot":
        stream(a.base, a.model, tokens_of(a.base, a.model, a.prompt_tokens, 99999)[:a.prompt_tokens],
               32, a.sampling, a.min_tokens)  # warmup
        runs = []
        for i in range(a.reps):
            ids = tokens_of(a.base, a.model, a.prompt_tokens, int(time.time() * 1000) % 100000 + i)
            runs.append(stream(a.base, a.model, ids, a.n_out, a.sampling, a.min_tokens))
            print(f"rep {i}: TPOT {runs[-1]['tpot_ss_ms']:.2f} ms, TTFT {runs[-1]['ttft_ms']:.0f} ms, "
                  f"{runs[-1]['tokens']} tokens in {runs[-1]['chunks']} chunks", flush=True)
        med = statistics.median(r["tpot_ss_ms"] for r in runs)
        res = {"prompt_tokens": a.prompt_tokens, "n_out": a.n_out, "sampling": a.sampling,
               "tpot_ms_median": med, "tok_s": 1000 / med, "runs": runs}
        print(f"TPOT median {med:.2f} ms = {1000/med:.2f} tok/s (prompt {a.prompt_tokens})")
    elif a.mode == "ttft":
        runs = []
        for i in range(a.reps + 1):
            ids = tokens_of(a.base, a.model, a.prompt_tokens, int(time.time() * 1000) % 100000 + i)
            t0 = time.perf_counter()
            d = post(a.base, "/v1/completions", {"model": a.model, "prompt": ids, "max_tokens": 1,
                                                 "temperature": 0})
            dt = (time.perf_counter() - t0) * 1000
            if i:
                runs.append({"ms": dt, "prompt_tokens": d["usage"]["prompt_tokens"]})
                print(f"rep {i}: {dt:.0f} ms ({d['usage']['prompt_tokens']} tokens)", flush=True)
        med = statistics.median(r["ms"] for r in runs)
        res = {"prompt_tokens": a.prompt_tokens, "ttft_ms_median": med, "runs": runs}
        print(f"TTFT median {med:.0f} ms (prompt {a.prompt_tokens})")
    elif a.mode == "ppl":
        prompt = PPL_TEXT
        if a.text_file:  # long check: first --prompt-tokens tokens of a fixed text file
            txt = open(a.text_file, encoding="utf-8").read()
            prompt = post(a.base, "/tokenize", {"model": a.model, "prompt": txt,
                                               "add_special_tokens": False})["tokens"][:a.prompt_tokens]
        d = post(a.base, "/v1/completions", {"model": a.model, "prompt": prompt, "max_tokens": 1,
                                             "temperature": 0, "prompt_logprobs": 0})
        pl = d["choices"][0]["prompt_logprobs"]
        toks, lps = [], []
        for e in pl[1:]:
            tid, v = next(iter(e.items()))  # prompt_logprobs=0: only the actual prompt token
            toks.append(tid); lps.append(v["logprob"])
        nll = -sum(lps) / len(lps)
        res = {"n": len(lps), "mean_nll": nll, "ppl": math.exp(nll), "tokens": toks, "logprobs": lps}
        print(f"prompt tokens scored {len(lps)}, mean NLL {nll:.4f}, PPL {math.exp(nll):.3f}")
    else:
        A, B = (json.load(open(f)) for f in a.files)
        assert A["tokens"] == B["tokens"]
        diffs = [abs(x - y) for x, y in zip(A["logprobs"], B["logprobs"])]
        res = {"n": len(diffs), "ppl_a": A["ppl"], "ppl_b": B["ppl"],
               "mean_abs_dlogprob": sum(diffs) / len(diffs), "max_abs_dlogprob": max(diffs)}
        print(json.dumps(res))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
