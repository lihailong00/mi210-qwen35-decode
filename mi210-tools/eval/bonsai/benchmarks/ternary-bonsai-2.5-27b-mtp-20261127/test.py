#!/usr/bin/env python3
import json, os, pathlib, signal, socket, subprocess, sys, time
from urllib.request import Request, build_opener, ProxyHandler
root=pathlib.Path(os.environ.get('BONSAI_BENCH_DIR', '.'))
root.mkdir(parents=True,exist_ok=True)
tag,k,n=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
limit=int(sys.argv[4]) if len(sys.argv)>4 else 5
port=18088
model=os.environ['MODEL']
bindir=os.environ['BONSAI_BACKEND_DIR']
args=[bindir+'/llama-server','-m',model,'-dev','ROCm0','-ngl','99','-fa','on','-t','8','-b','4096','-ub','4096','-ctk','f16','-ctv','f16','-c','8192','-np','1','--host','127.0.0.1','--port',str(port),'--no-webui']
if k:args+=['--spec-type','draft-mtp','--spec-draft-n-max',str(k)]
# Do not disturb another process already using the test endpoint.
with socket.socket() as sock:
    if sock.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('test port is occupied')
op=build_opener(ProxyHandler({}))
def call(route,payload=None):
    body=None if payload is None else json.dumps(payload).encode()
    with op.open(Request('http://127.0.0.1:'+str(port)+route,data=body,headers={'Content-Type':'application/json'}),timeout=90) as r:return json.load(r)
out={'tag':tag,'k':k,'n_predict':n,'args':args,'requests':[]}
(root/(tag+'.command.json')).write_text(json.dumps(args,indent=2))
def stop(signum,frame):raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)
with (root/(tag+'.server.log')).open('w') as log:
    srv=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            if srv.poll() is not None:raise RuntimeError('server exited: '+str(srv.returncode))
            try:
                if call('/health').get('status')=='ok':break
            except Exception:pass
            time.sleep(0.5)
        else:raise RuntimeError('server startup timeout')
        def comp(prompt,count):
            t0=time.monotonic()
            r=call('/completion',{'prompt':prompt,'n_predict':count,'temperature':0,'seed':1,'ignore_eos':True,'cache_prompt':False,'return_tokens':True,'stream':False})
            r['_wall_seconds']=time.monotonic()-t0
            return r
        out['warmup']=comp('Hello',32)
        for p in json.loads((pathlib.Path(__file__).with_name('prompts.json')).read_text())[:limit]:
            r=comp(p['prompt'],n);r['_prompt_id']=p['id'];out['requests'].append(r)
            print(json.dumps({'id':p['id'],'k':k,'timings':r.get('timings',{}),'wall':r['_wall_seconds']},ensure_ascii=False),flush=True)
    finally:
        srv.terminate()
        try:srv.wait(timeout=15)
        except subprocess.TimeoutExpired:srv.kill();srv.wait(timeout=10)
        (root/(tag+'.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2))
