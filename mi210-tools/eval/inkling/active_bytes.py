#!/usr/bin/env python3
"""usage: active_bytes.py <model.gguf> -- per-token decode bytes for Inkling-2-Mini (8/128 experts, core layers x3). Needs gguf-py on sys.path."""
import sys, re, collections
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
T = [(t.name, int(t.n_bytes)) for t in r.tensors]
per = collections.Counter()
for n,b in T:
    m = re.match(r"blk\.(\d+)\.(.*)", n)
    if not m: continue
    L = int(m.group(1)); rest = m.group(2)
    if "_exps" in rest: b = b * 8 / 128
    per[L] += b
reps = {L: (3 if 6 <= L < 18 else 1) for L in per}
tot = sum(per[L]*reps[L] for L in per)
out = [b for n,b in T if n=="output.weight"][0]
print("per-layer active MB", {L: round(per[L]/1e6,1) for L in sorted(per)})
print("layer passes", sum(reps.values()), "active GB/token layers", tot/1e9, "+ lm_head", out/1e9, "=", (tot+out)/1e9)
print("bound @1.4TB/s tok/s:", 1.4e12/(tot+out), " @1.2TB/s:", 1.2e12/(tot+out))
