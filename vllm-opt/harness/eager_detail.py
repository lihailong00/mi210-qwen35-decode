import sys, json, gzip, collections, statistics
sys.argv += []
exec(open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "analyze_mtp.py")).read().split("def main")[0])
ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
rt = {e["args"]["correlation"]: e for e in ev if e.get("cat") in ("cuda_runtime","hip_runtime") and "args" in e and "correlation" in e["args"]}
ks = sorted((e for e in ev if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset")), key=lambda e: e["ts"])
segs=[]
for k in ks:
    c=k["args"].get("correlation"); r=rt.get(c)
    g = c if (r is not None and r["name"] in ("hipGraphLaunch",)) else None
    lab = g if g is not None else "E"
    if segs and segs[-1][0]==lab and lab=="E": segs[-1][1].append(k)
    elif segs and segs[-1][0]==lab: segs[-1][1].append(k)
    else: segs.append((lab,[k]))
# rounds start at big graph
big=[i for i,(l,s) in enumerate(segs) if l!="E" and len(s)>1000]
for which in (1,5):
    tab=collections.defaultdict(list)
    for bi in range(3,len(big)-1):
        seg=segs[big[bi]+which][1]
        for j,k in enumerate(seg):
            tab[(j,short(k["name"],70))].append(k["dur"])
    print("== eager segment",which)
    for (j,n),v in sorted(tab.items()):
        if len(v)>=len(big)//2: print(f"  {j:3d} {statistics.median(v):8.1f} {n}")
# host side: runtime API calls per round
