import csv,sys,re,collections
rows=list(csv.DictReader(open(sys.argv[1])))
ev=sorted((int(r["Start_Timestamp"]),int(r["End_Timestamp"]),r["Kernel_Name"],int(r.get("Grid_Size_X",0) or 0),int(r.get("Grid_Size_Y",0) or 0),int(r.get("Grid_Size_Z",0) or 0),int(r.get("Workgroup_Size_X",0) or 0)) for r in rows)
heads=[i for i,e in enumerate(ev) if (e[1]-e[0])/1000>400]
a,b=heads[-21],heads[-1]; T=20
s=ev[a+1:b+1]
pat=sys.argv[2]
c=collections.Counter(); t=collections.Counter()
for e in s:
    if re.search(pat,e[2]):
        k=(re.sub(r"\(.*","",e[2])[:70],e[3],e[4],e[5],e[6]); c[k]+=1; t[k]+=(e[1]-e[0])/1000
for k,v in sorted(t.items(),key=lambda x:-x[1]): print("%8.1f us/tok %6.1f calls/tok %7.2f us/call grid=%s,%s,%s wg=%s %s"%(v/T,c[k]/T,v/c[k],k[1],k[2],k[3],k[4],k[0]))
