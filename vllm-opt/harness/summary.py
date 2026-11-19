"""Summarize the final A/B runs (runs/queue1.sh): mtp_bench tok/s per prompt, acceptance, text identity
vs K=0 of the same deployment (rep a), TPOT/TTFT, concurrency. usage: summary.py [runs_dir]"""
import json, os, re, sys
R = sys.argv[1] if len(sys.argv) > 1 else __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "runs")
P = ["code", "story", "json", "edit", "repeat"]
def L(d, f):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None
def num(t, name):
    m = re.search(re.escape(name) + r":\s+([0-9.]+)", t)
    return m.group(1) if m else "?"
for dep in "nv":
    ref = L(f"{dep}_k0a", "mtp_bench.json")
    print("##", {"n": "deploy-mtp", "v": "deploy v1"}[dep])
    for k in range(5):
        for rep in "ab":
            r = L(f"{dep}_k{k}{rep}", "mtp_bench.json")
            if not r: continue
            sp = [sum(x["decode_tok_s"] for x in r[p]) / len(r[p]) for p in P]
            acc = [r[p][0]["accepted"] / r[p][0]["draft"] if r[p][0]["draft"] else 0 for p in P]
            same = []
            for p in P:
                if not ref: same.append("?"); continue
                texts = [x["text"] for x in r[p]]
                a = ref[p][0]["text"]
                if all(t == a for t in texts): same.append("=")
                else:
                    c = next(t for t in texts if t != a)
                    same.append("diff@%d" % next((i for i, (x, y) in enumerate(zip(a, c)) if x != y), min(len(a), len(c))))
            print(f"K={k}{rep} avg {sum(sp)/5:6.1f} | " + " ".join(f"{p} {s:6.1f}" for p, s in zip(P, sp))
                  + " | acc " + " ".join(f"{a:.2f}" for a in acc) + " | vsK0a " + " ".join(same))
    for rep in "ab":
        for f in ("tpot128.json", "tpot4096.json", "ttft4096.json"):
            j = L(f"{dep}_k0{rep}", f)
            if j: print(f"K0{rep} {f}: " + json.dumps({k: v for k, v in j.items() if not isinstance(v, (list, dict))})[:300])
    for k in (0, 3):
        for C in (4, 16, 64):
            p = os.path.join(R, f"{dep}_conc{k}", f"c{C}.log")
            if os.path.exists(p):
                t = open(p).read()
                print(f"conc MTP={k} c{C}: out {num(t, 'Output token throughput (tok/s)')} tok/s, "
                      f"mean TPOT {num(t, 'Mean TPOT (ms)')} ms, mean TTFT {num(t, 'Mean TTFT (ms)')} ms")
