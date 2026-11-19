#!/usr/bin/env python3
"""Greedy capture for correctness compare: N prompts x T tokens, chosen-token logprobs + top-5.
usage: capture.py --base URL --out ref.json [--tokens 128] [--conc 1]
--conc > 1 sends the prompts concurrently (exercises batched decode M>1)."""
import argparse, json, urllib.request, concurrent.futures

PROMPTS = [
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
]


def one(base, model, prompt, n, min_tokens=True):
    body = {"model": model, "prompt": prompt, "max_tokens": n, "min_tokens": n, "temperature": 0,
            "ignore_eos": True, "logprobs": 5, "return_tokens_as_token_ids": True}
    if not min_tokens:
        del body["min_tokens"]
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with op.open(req, timeout=900) as r:
        d = json.loads(r.read())
    lp = d["choices"][0]["logprobs"]
    return {"prompt": prompt, "tokens": lp["tokens"], "logprobs": lp["token_logprobs"],
            "top": lp["top_logprobs"], "text": d["choices"][0]["text"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="qwen38")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-min-tokens", action="store_true")
    a = ap.parse_args()
    ps = PROMPTS[: a.n]
    with concurrent.futures.ThreadPoolExecutor(a.conc) as ex:
        res = list(ex.map(lambda p: one(a.base, a.model, p, a.tokens, not a.no_min_tokens), ps))
    json.dump(res, open(a.out, "w"))
    print("captured", len(res), "x", a.tokens)


if __name__ == "__main__":
    main()
