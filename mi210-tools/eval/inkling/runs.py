#!/usr/bin/env python3
# usage: runs.py <run_kernel_trace.csv> -- classify decode kernels into weight-streaming GEMVs and small ops,
# and report how many small kernels sit between consecutive GEMVs (per token, steady-state tokens only)
import csv, sys, re, collections

rows = []
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
rows.sort()

GEMV = re.compile(r"ink_gemv_bf16|ink_mmid_nvfp4|ink_moe_bf16|ink_router3|mul_mat_vec|ink_ctl_down|ink_gemv")
def short(n):
    n = n.split("(")[0]
    return n[:60]

# token boundaries: split right after each LM-head GEMV (unique grid size, largest grid of the token)
grid_of = {}
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        grid_of[int(r["Start_Timestamp"])] = int(r.get("Grid_Size") or r.get("Grid_Size_X") or 0)
maxgrid = max(grid_of.values())
toks, cur = [], []
for s_, e, n in rows:
    cur.append((s_, e, n))
    if grid_of.get(s_, 0) == maxgrid:
        toks.append(cur)
        cur = []
sizes = collections.Counter(len(t) for t in toks)
common = sizes.most_common(1)[0][0]
steady = [t for t in toks if len(t) == common]
print("tokens found", len(toks), "launch counts", dict(sizes.most_common(4)), "-> using", len(steady), "tokens with", common, "launches")

t = steady[len(steady) // 2]
g = [bool(GEMV.search(n)) for _, _, n in t]
print("GEMV-type launches per token", sum(g), " small-op launches per token", len(t) - sum(g))
runs, k = [], 0
for x in g:
    if x:
        runs.append(k)
        k = 0
    else:
        k += 1
runs.append(k)
print("small ops between consecutive GEMVs: histogram", dict(sorted(collections.Counter(runs).items())))
names = collections.Counter(short(n) for (_, _, n), x in zip(t, g) if not x)
print("small-op kernels per token:")
for n, c in names.most_common(30):
    print(f"  {c:4d}  {n}")
gn = collections.Counter(short(n) for (_, _, n), x in zip(t, g) if x)
print("GEMV-type kernels per token:")
for n, c in gn.most_common(15):
    print(f"  {c:4d}  {n}")
