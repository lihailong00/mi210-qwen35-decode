#!/usr/bin/env python3
# ts.py <prefix> [ntok] [full] : analyze the in-graph GPU timestamps of attn-misc-gpu-timestamps.patch
# (llama-bench ... with GGML_CUDA_TS=1 GGML_CUDA_TS_DUMP=60 GGML_CUDA_TS_OUT=<prefix>, tg64). Prints per-node real cost =
# timestamp delta - cost of one timestamp kernel (START1 - START0), aggregated by op and tensor name.
import sys, statistics, re
from collections import defaultdict
TICK = 25.0  # s_memrealtime ticks per us (MI210: 25 MHz)
pre = sys.argv[1]
ntok = int(sys.argv[2]) if len(sys.argv) > 2 else 30
labels = [l.rstrip('\n') for l in open(pre + '.labels')]
ts = [int(l) for l in open(pre + '.ts')]
L = len(labels)
# locate token starts: walk back from the end in steps of L
n = len(ts)
k = n // L
toks = []
end = n
for t in range(min(ntok, k)):
    seg = ts[end - L:end]
    toks.append(seg)
    end -= L
toks.reverse()
# sanity: START1-START0 small
c_ts = statistics.median([(s[1] - s[0]) / TICK for s in toks])
print(f"labels/token {L}, timestamps {n}, tokens used {len(toks)}, ts-kernel cost {c_ts:.2f} us")
span = [(s[-1] - s[0]) / TICK for s in toks]
print(f"graph span (first..last ts) median {statistics.median(span):.1f} us")
gaps = [(toks[i + 1][0] - toks[i][-1]) / TICK for i in range(len(toks) - 1)]
if gaps:
    print(f"between graphs (last ts -> next first ts) median {statistics.median(gaps):.1f} us")
# per position median delta
med = []
for j in range(1, L):
    med.append(statistics.median([(s[j] - s[j - 1]) / TICK for s in toks]))
def cat(label):
    parts = label.split(' ')
    kind = parts[0]
    op = parts[1] if len(parts) > 1 else ''
    name = parts[2] if len(parts) > 2 else ''
    base = re.sub(r'-\d+$', '', name)
    extra = ' '.join(parts[3:5]) if kind == 'U' else ''
    return f"{kind} {op} {base} {extra}".strip()
agg = defaultdict(lambda: [0, 0.0])
for j in range(1, L):
    c = cat(labels[j])
    agg[c][0] += 1
    agg[c][1] += med[j - 1] - c_ts
tot = sum(v[1] for v in agg.values())
print(f"sum of (delta - ts cost) over nodes: {tot:.1f} us; launches {L - 2}")
for c, (cnt, us) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:45]:
    print(f"{us:9.1f} us {cnt:5d} n {us / cnt:7.2f} us/launch  {c}")
if len(sys.argv) > 3:
    for j in range(1, L):
        print(f"{j:5d} {med[j - 1] - c_ts:8.2f}  {labels[j]}")
