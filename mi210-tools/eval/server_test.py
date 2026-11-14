#!/usr/bin/env python3
"""Server-mode functional + speed test (runs inside the mi210-opt container).
usage: server_test.py <bindir> <tag> <np> [port]
Starts llama-server, then: 3 sequential greedy requests, 4 concurrent requests, and a 2-turn prompt-cache check.
Writes /lab/results/server-<tag>-np<np>.json."""
import json, os, subprocess, sys, time, threading
from urllib.request import Request, build_opener, ProxyHandler

bindir, tag, np_ = sys.argv[1], sys.argv[2], int(sys.argv[3])
port = int(sys.argv[4]) if len(sys.argv) > 4 else 8090
M = "/data/model/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
env = dict(os.environ, CODEX_IQ3_PREPACK="1", CODEX_IQ3_XXS_PREPACK="1", LD_LIBRARY_PATH=bindir + ":/opt/rocm/lib")
log = open(f"/lab/results/server-{tag}-np{np_}.log", "w")
srv = subprocess.Popen([f"{bindir}/llama-server", "-m", M, "-dev", "ROCm0", "-ngl", "999", "-fa", "on", "-b", "4096", "-ub", "4096",
                        "-ctk", "f16", "-ctv", "f16", "-c", "65536", "-np", str(np_), "--host", "127.0.0.1", "--port", str(port), "--no-webui"],
                       stdout=log, stderr=subprocess.STDOUT, env=env)
op = build_opener(ProxyHandler({}))

def call(route, payload=None, timeout=600):
    body = None if payload is None else json.dumps(payload).encode()
    r = op.open(Request(f"http://127.0.0.1:{port}{route}", data=body, headers={"Content-Type": "application/json"}), timeout=timeout)
    return json.load(r)

out = {"tag": tag, "np": np_}
try:
    for _ in range(240):
        try:
            if call("/health", timeout=5).get("status") == "ok": break
        except Exception: pass
        time.sleep(1)
    else:
        raise RuntimeError("server did not become healthy")
    prompts = [open(f"/lab/eval/prompts/p{i}.txt", encoding="utf-8").read() for i in (1, 2, 3)]
    def comp(p, n=128, cache=False):
        r = call("/completion", {"prompt": p, "n_predict": n, "temperature": 0.0, "seed": 1, "ignore_eos": True,
                                 "cache_prompt": cache, "return_tokens": True, "stream": False})
        t = r.get("timings", {})
        return {"content": r.get("content", ""), "tokens": r.get("tokens", []), "tps": t.get("predicted_per_second"),
                "pps": t.get("prompt_per_second"), "cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n")}
    comp("Hello", 16)  # warm-up (also triggers one-time weight repacks)
    out["sequential"] = [comp(p) for p in prompts]
    # 4 concurrent requests (multi-sequence batches -> fused single-sequence paths must fall back correctly)
    res = [None]*4
    def worker(i): res[i] = comp(prompts[i % 3] + (" " * i), 96)
    th = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]
    out["concurrent"] = {"wall_s": time.time() - t0, "results": res}
    # prompt cache: turn 1, then turn 2 extending turn 1's prompt+output (cache_prompt true) vs the same prompt fresh
    t1 = comp(prompts[0], 64, cache=True)
    p2 = prompts[0] + t1["content"] + " Moreover,"
    t2c = comp(p2, 64, cache=True)
    t2f = comp(p2, 64, cache=False)
    out["prompt_cache"] = {"turn2_cached": t2c, "turn2_fresh": t2f, "same_text": t2c["content"] == t2f["content"],
                           "cache_n": t2c["cache_n"]}
finally:
    srv.terminate()
    try: srv.wait(30)
    except Exception: srv.kill()
json.dump(out, open(f"/lab/results/server-{tag}-np{np_}.json", "w"), ensure_ascii=False, indent=1)
s = out.get("sequential", [])
print(tag, "np", np_, "seq tps", [round(x["tps"] or 0, 2) for x in s],
      "concurrent wall %.1fs" % out.get("concurrent", {}).get("wall_s", -1),
      "conc tps", [round((x or {}).get("tps") or 0, 2) for x in out.get("concurrent", {}).get("results", [])],
      "cache same_text", out.get("prompt_cache", {}).get("same_text"), "cache_n", out.get("prompt_cache", {}).get("cache_n"))
