import json, sys, time, re, requests
base, out = sys.argv[1], sys.argv[2]
CODE = '''def process(items, limit):
    result = []
    for i in range(len(items)):
        item = items[i]
        if item is None:
            continue
        value = item.get("value", 0)
        if value > limit:
            result.append({"id": item["id"], "value": value, "flag": True})
        else:
            result.append({"id": item["id"], "value": value, "flag": False})
    total = 0
    for r in result:
        total = total + r["value"]
    return result, total
'''
PROMPTS = {
  "code": "Write a complete Python module implementing an LRU cache class with get, put, delete and resize methods, with docstrings and type hints.",
  "story": "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
  "json": "Generate a JSON array of 8 fictional employees. Each object has id, name, department, title, salary, start_date, skills (array of 3 strings) and manager_id.",
  "edit": "Rewrite the following Python function so that it uses list comprehensions and sum(), keeps the same behaviour, and adds type hints. Output only the code.\n\n" + CODE,
  "repeat": "Repeat the following text exactly, then stop.\n\n" + CODE,
}
def metrics():
    t = requests.get(base + "/metrics").text
    g = lambda n: sum(float(m) for m in re.findall(r"^" + n + r"(?:\{[^}]*\})? ([0-9.e+]+)$", t, re.M))
    return g("vllm:spec_decode_num_draft_tokens_total"), g("vllm:spec_decode_num_accepted_tokens_total")
res = {}
for name, p in PROMPTS.items():
    runs = []
    for rep in range(2):
        d0, a0 = metrics()
        body = {"model": "qwen38", "messages": [{"role": "user", "content": p}], "max_tokens": 384,
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.time(); tf = None; tl = None; text = []; ntok = None
        with requests.post(base + "/v1/chat/completions", json=body, stream=True) as r:
            for line in r.iter_lines():
                if not line.startswith(b"data: ") or line == b"data: [DONE]": continue
                j = json.loads(line[6:])
                if j.get("usage"): ntok = j["usage"]["completion_tokens"]
                for c in j.get("choices", []):
                    s = c.get("delta", {}).get("content")
                    if s:
                        now = time.time(); tf = tf or now; tl = now; text.append(s)
        d1, a1 = metrics()
        tps = (ntok - 1) / (tl - tf) if ntok and tl and tl > tf else None
        runs.append({"tokens": ntok, "ttft_s": tf - t0, "decode_tok_s": tps, "draft": d1 - d0, "accepted": a1 - a0,
                     "text": "".join(text)})
        print(f"{name} rep{rep}: {ntok} tok, {tps:.1f} tok/s, accepted {a1-a0:.0f}/{d1-d0:.0f}", flush=True)
    res[name] = runs
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
