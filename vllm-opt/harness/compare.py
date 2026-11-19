#!/usr/bin/env python3
"""Compare two capture.py outputs (reference vs candidate).
Reports per prompt: first divergence index, positional token match rate, and over the common
prefix (identical context) the mean/max |logprob diff| of the chosen tokens and the mean
|diff| of the top-5 logprobs present in both.
usage: compare.py ref.json cand.json"""
import json, sys

ref = json.load(open(sys.argv[1])); cand = json.load(open(sys.argv[2]))
tot_match = tot = 0; diffs = []; topd = []; firsts = []
for r, c in zip(ref, cand):
    n = min(len(r["tokens"]), len(c["tokens"]))
    match = sum(1 for i in range(n) if r["tokens"][i] == c["tokens"][i])
    first = next((i for i in range(n) if r["tokens"][i] != c["tokens"][i]), n)
    firsts.append(first)
    tot_match += match; tot += n
    for i in range(first):
        if r["logprobs"][i] is not None and c["logprobs"][i] is not None:
            diffs.append(abs(r["logprobs"][i] - c["logprobs"][i]))
        rt, ct = r["top"][i] or {}, c["top"][i] or {}
        for k in set(rt) & set(ct):
            topd.append(abs(rt[k] - ct[k]))
    print(f"prompt {r['prompt'][:40]!r}: first_divergence={first}/{n} positional_match={match/n:.3f}")
res = {"positional_token_match": tot_match / tot, "prefix_tokens_identical": sum(firsts),
       "tokens": tot, "full_match_prompts": sum(1 for f, r in zip(firsts, ref) if f == len(r["tokens"])),
       "mean_abs_logprob_diff_chosen": sum(diffs) / max(1, len(diffs)),
       "max_abs_logprob_diff_chosen": max(diffs) if diffs else 0.0,
       "mean_abs_logprob_diff_top5": sum(topd) / max(1, len(topd))}
print(json.dumps(res, indent=1))
