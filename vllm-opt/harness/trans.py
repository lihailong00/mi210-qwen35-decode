"""Exactness across changing batch size (dynamic K / batch transitions), greedy, one server.
Reference: each prompt alone (sequential). Then overlapped: P0 starts, P1+P2 join after DELAY s, so the
batch goes 1 -> 3 -> 2 -> 1 while P0 runs. All texts must equal their sequential reference
(every M<=8 kernel is per-row exact, so batch size must not change greedy text).
usage: trans.py BASE OUT.json"""
import json, sys, time, threading, requests
base, out = sys.argv[1], sys.argv[2]
P = ["Write a complete Python module implementing an LRU cache class with get, put, delete and resize methods, with docstrings and type hints.",
     "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
     "Generate a JSON array of 8 fictional employees. Each object has id, name, department, title, salary, start_date, skills (array of 3 strings) and manager_id.",
     "Explain step by step how a refrigerator works."]
N = [384, 160, 240, 300]
def ask(i, res):
    body = {"model": "qwen38", "messages": [{"role": "user", "content": P[i]}], "max_tokens": N[i],
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    r = requests.post(base + "/v1/chat/completions", json=body).json()
    res[i] = {"text": r["choices"][0]["message"]["content"], "sec": time.time() - t}
ref = {}
for i in range(len(P)): ask(i, ref)
ok = True; rounds = []
for rnd, (delay, joiners) in enumerate([(1.5, [1, 2]), (3.0, [1, 2, 3]), (0.5, [3])]):
    got = {}
    th = [threading.Thread(target=ask, args=(0, got))]; th[0].start(); time.sleep(delay)
    for j in joiners:
        t = threading.Thread(target=ask, args=(j, got)); t.start(); th.append(t); time.sleep(0.3)
    for t in th: t.join()
    row = {}
    for i, g in got.items():
        a, b = ref[i]["text"], g["text"]
        row[i] = "same" if a == b else "diff@%d" % next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        ok &= a == b
    print("round", rnd, "delay", delay, "joiners", joiners, row, flush=True); rounds.append(row)
print("ALL_SAME" if ok else "MISMATCH", flush=True)
json.dump({"ref": ref, "rounds": rounds, "ok": ok}, open(out, "w"), indent=1)
