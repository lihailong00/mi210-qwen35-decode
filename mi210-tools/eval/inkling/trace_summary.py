import csv,sys,re,collections,statistics
rows=list(csv.DictReader(open(sys.argv[1])))
ev=sorted((int(r["Start_Timestamp"]),int(r["End_Timestamp"]),r["Kernel_Name"],int(r.get("Grid_Size_X",0) or 0),int(r.get("Workgroup_Size_X",0) or 0)) for r in rows)
print("kernels", len(ev))
def norm(n):
    n=re.sub(r"\(anonymous namespace\)::","",n)
    m=re.match(r"(?:void )?([A-Za-z_0-9:]+)(<[^()]*>)?",n)
    return (m.group(1)+(m.group(2) or "")[:90]) if m else n[:100]
d=sorted(ev,key=lambda e:-(e[1]-e[0]))[:5]
for e in d: print("long", (e[1]-e[0])/1000, norm(e[2]), e[3], e[4])
TH=float(sys.argv[2]) if len(sys.argv)>2 else 400
heads=[i for i,e in enumerate(ev) if (e[1]-e[0])/1000>TH]
print("heads", len(heads))
segs=list(zip(heads[-21:-1],heads[-20:]))
fam=collections.defaultdict(list); nl=[]; walls=[]; busy=[]
for a,b in segs:
    s=ev[a+1:b+1]; walls.append((ev[b][1]-ev[a][1])/1000); nl.append(len(s)); busy.append(sum(e[1]-e[0] for e in s)/1000)
    c=collections.Counter(); t=collections.Counter()
    for e in s: k=norm(e[2]); c[k]+=1; t[k]+=(e[1]-e[0])/1000
    for k in c: fam[k].append((c[k],t[k]))
print("tokens",len(segs),"wall(prof) %.0f us, busy %.0f us, launches %.1f"%(statistics.mean(walls),statistics.mean(busy),statistics.mean(nl)))
tot=sorted(((statistics.mean(x[1] for x in v),statistics.mean(x[0] for x in v),k) for k,v in fam.items()),reverse=True)
for us,n,k in tot[:int(sys.argv[3]) if len(sys.argv)>3 else 45]: print("%8.1f us %6.1f n %7.1f us/call  %s"%(us,n,us/n,k))
