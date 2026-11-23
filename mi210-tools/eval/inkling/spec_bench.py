#!/usr/bin/env python3
"""Chat-mode decode speed test for Inkling-2-Mini (runs inside the mi210-opt container).
usage: spec_bench.py <bindir> <tag> [extra llama-server args...]
Starts llama-server with the Q8_K_XL model (+extra args, e.g. the DSpark drafter), runs 6 chat prompts
greedy (temp 0) and with the recommended sampling (temp 1, seed 1), 256 tokens each.
Writes /lab/results/inkling-<tag>.json and prints a summary."""
import json, os, subprocess, sys, time
from urllib.request import Request, build_opener, ProxyHandler
bindir, tag, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
port = 8092
M = "/data/model/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q8_K_XL.gguf"
env = dict(os.environ, LD_LIBRARY_PATH=bindir + ":/opt/rocm/lib")
log = open(f"/lab/results/inkling-{tag}.log", "w")
cmd = [f"{bindir}/llama-server", "-m", M, "-ngl", "999", "-fa", "on", "-c", "16384", "-np", "1",
       "--jinja", "--host", "127.0.0.1", "--port", str(port), "--no-webui"] + extra
log.write(" ".join(cmd) + "\n"); log.flush()
srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
op = build_opener(ProxyHandler({}))
def call(route, payload=None, timeout=900):
    body = None if payload is None else json.dumps(payload).encode()
    return json.load(op.open(Request(f"http://127.0.0.1:{port}{route}", data=body, headers={"Content-Type": "application/json"}), timeout=timeout))
PROMPTS = [
 "Write a Python function that parses an ISO-8601 duration string like P3DT4H5M into total seconds, with unit tests.",
 "Explain how a B+ tree handles insertion when a leaf node overflows. Be concrete and step by step.",
 "A train leaves at 9:40 and travels 315 km at 84 km/h, then stops 12 minutes, then travels 140 km at 70 km/h. When does it arrive? Show your work.",
 "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.",
 "用中文解释一下为什么 GPU 上的矩阵乘法要做分块（tiling），并举一个具体的例子。",
 "Write a bash script that finds the 10 largest files under a directory, skipping .git folders, and prints sizes in human-readable form.",
]
out = {"tag": tag, "cmd": cmd, "runs": []}
try:
    for _ in range(600):
        try:
            if call("/health", timeout=5).get("status") == "ok": break
        except Exception: pass
        if srv.poll() is not None: raise RuntimeError("server exited")
        time.sleep(1)
    else: raise RuntimeError("server did not become healthy")
    def chat(p, temp, n=256):
        t0 = time.time()
        r = call("/v1/chat/completions", {"messages": [{"role": "user", "content": p}], "max_tokens": n,
              "temperature": temp, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "seed": 1, "stream": False})
        t = r.get("timings", {}); m = r["choices"][0]["message"]
        return {"temp": temp, "wall": time.time() - t0, "tps": t.get("predicted_per_second"), "n": t.get("predicted_n"),
                "pps": t.get("prompt_per_second"), "prompt_n": t.get("prompt_n"),
                "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
                "content": (m.get("reasoning_content") or "") + "\n<<>>\n" + (m.get("content") or "")}
    chat("Hello", 0.0, 16)
    for temp in (0.0, 1.0):
        for p in PROMPTS: out["runs"].append(chat(p, temp))
finally:
    srv.terminate()
    try: srv.wait(30)
    except Exception: srv.kill()
json.dump(out, open(f"/lab/results/inkling-{tag}.json", "w"), ensure_ascii=False, indent=1)
for temp in (0.0, 1.0):
    rs = [r for r in out["runs"] if r["temp"] == temp]
    if not rs: continue
    n = sum(r["n"] or 0 for r in rs); tt = sum((r["n"] or 0) / (r["tps"] or 1) for r in rs)
    dn = sum(r["draft_n"] or 0 for r in rs); da = sum(r["draft_acc"] or 0 for r in rs)
    print(f"{tag} temp={temp} per-prompt tps", [round(r["tps"] or 0, 1) for r in rs],
          f"aggregate {n/tt:.1f} tok/s", f"accept {da}/{dn}" + (f" = {da/dn:.2f}" if dn else ""))
