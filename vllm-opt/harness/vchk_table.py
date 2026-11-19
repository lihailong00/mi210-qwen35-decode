import json, sys
def m(ra, ca):
    ref = json.load(open(ra)); cand = json.load(open(ca)); tm = tot = 0; diffs = []; full = 0
    for r, c in zip(ref, cand):
        n = min(len(r["tokens"]), len(c["tokens"]))
        tm += sum(r["tokens"][i] == c["tokens"][i] for i in range(n)); tot += n
        f = next((i for i in range(n) if r["tokens"][i] != c["tokens"][i]), n); full += f == n
        diffs += [abs(r["logprobs"][i] - c["logprobs"][i]) for i in range(f) if r["logprobs"][i] is not None and c["logprobs"][i] is not None]
    return tm / tot, full, sum(diffs) / max(1, len(diffs))
O = "../../vchk-c/runs/orig"; P = "../../vchk-c/runs/opt"
rows = [("orig c1 vs orig c8 (engine noise)", O+"/g1.json", O+"/g8.json")]
for g in ("g1", "g8", "g16"):
    for nm, d in (("v1", P), ("deploy-mtp K=0", "b_vchk0"), ("deploy-mtp K=3", "c_vchk3")):
        rows.append((f"{g}: {nm} vs orig", f"{O}/{g}.json", f"{d}/{g}.json"))
rows.append(("g1: deploy-mtp K=3 vs K=0", "b_vchk0/g1.json", "c_vchk3/g1.json"))
for nm, a, b in rows:
    t, f, d = m(a, b); print(f"| {nm} | {t:.3f} | {f}/16 | {d:.4f} |")
