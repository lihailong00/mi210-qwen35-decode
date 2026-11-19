"""Compare mtp_bench.json runs: text identity of each prompt vs a reference run, speed, acceptance.
usage: cmp_runs.py REF_RUN_DIR RUN_DIR [RUN_DIR ...]   (dirs under runs/ or absolute)"""
import json, os, sys
import os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
def load(d):
    p = d if os.path.isabs(d) else os.path.join(R, d)
    return json.load(open(os.path.join(p, "mtp_bench.json")))
ref = load(sys.argv[1])
for d in sys.argv[1:]:
    r = load(d)
    row, sp = [], []
    for p in ref:
        a = ref[p][0]["text"]; c = r[p][0]["text"]
        n = next((i for i, (x, y) in enumerate(zip(a, c)) if x != y), min(len(a), len(c)))
        same = a == c
        tps = sum(x["decode_tok_s"] for x in r[p]) / len(r[p])
        acc = r[p][0]["accepted"] / r[p][0]["draft"] if r[p][0]["draft"] else float("nan")
        sp.append(tps)
        row.append("%s %.1f acc %.2f %s" % (p, tps, acc, "same" if same else "diff@%d/%d" % (n, len(a))))
    print("%-16s avg %.1f | %s" % (os.path.basename(d.rstrip("/")), sum(sp) / len(sp), " | ".join(row)))
