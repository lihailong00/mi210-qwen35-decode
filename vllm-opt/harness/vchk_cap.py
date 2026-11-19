#!/usr/bin/env python3
"""Correctness capture for the skeptic review: 16 prompts (incl. ~1k/2k/4k-token ones) x T greedy
tokens with chosen logprobs + top-5, no min_tokens (so the fused greedy argmax path is eligible),
at a given concurrency. --mode sample: T=0.7/top_k 20/top_p 0.95 sanity run (text only)."""
import argparse, json, urllib.request, concurrent.futures, time

BASEP = [
    "The history of the Roman Empire can be summarized as follows:",
    "def quicksort(arr):\n    \"\"\"Sort a list using quicksort.\"\"\"\n",
    "Explain step by step how a refrigerator works.",
    "Translate into French: The weather is lovely today and we are going to the beach.",
    "List ten interesting facts about octopuses.\n1.",
    "Q: If a train travels 120 km in 1.5 hours, what is its average speed?\nA:",
    "Write a short poem about autumn leaves falling in a quiet park.",
    "In linear algebra, the eigenvalues of a matrix are",
    "Once upon a time, in a village at the edge of a vast desert,",
    "The main differences between TCP and UDP are",
    "1, 1, 2, 3, 5, 8, 13, 21,",
    "<|im_start|>user\nWhat is the capital of Australia? Answer briefly.<|im_end|>\n<|im_start|>assistant\n",
]


def long_prompts():
    t = open(__import__("os").environ.get("MODEL_DIR", "/data/model/Qwen3.8-27B-FP8") + "/README.md", encoding="utf-8", errors="ignore").read()
    out = []
    for nch, off in ((3500, 0), (8000, 2000), (15000, 0), (16000, 4000)):
        s = t[off:off + nch]
        out.append(s + "\n\nSummarize the text above in three sentences:\n")
    return out


def one(base, model, prompt, n, mode):
    body = {"model": model, "prompt": prompt, "max_tokens": n, "ignore_eos": True,
            "logprobs": 5, "return_tokens_as_token_ids": True}
    if mode == "greedy":
        body["temperature"] = 0
    else:
        body.update(temperature=0.7, top_k=20, top_p=0.95, seed=None)
        body.pop("seed")
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with op.open(req, timeout=900) as r:
        d = json.loads(r.read())
    lp = d["choices"][0]["logprobs"]
    return {"prompt": prompt[:200], "plen": d["usage"]["prompt_tokens"], "tokens": lp["tokens"],
            "logprobs": lp["token_logprobs"], "top": lp["top_logprobs"],
            "text": d["choices"][0]["text"], "sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--mode", default="greedy")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ps = BASEP + long_prompts()
    with concurrent.futures.ThreadPoolExecutor(a.conc) as ex:
        res = list(ex.map(lambda p: one(a.base, a.model, p, a.tokens, a.mode), ps))
    json.dump(res, open(a.out, "w"))
    print("captured", len(res), "plens", [r["plen"] for r in res])


if __name__ == "__main__":
    main()
