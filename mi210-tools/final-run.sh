#!/usr/bin/env bash
# usage (inside container): /lab/final-run.sh <bindir> <tag>   -> decode (d=0,8192) + prefill (512,2048,8192), jsonl in /lab/results/final/
B=$1; T=$2; O=/lab/results/final; mkdir -p $O
M=/data/model/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
export CODEX_IQ3_PREPACK=1 CODEX_IQ3_XXS_PREPACK=1
C="-m $M -dev ROCm0 -ngl 999 -fa on -b 4096 -ub 4096 -ctk f16 -ctv f16"
$B/llama-bench $C -p 0 -n 128 -d 0,8192 -r 2 -o jsonl > $O/$T-decode.jsonl 2> $O/$T-decode.err
$B/llama-bench $C -p 512,2048,8192 -n 0 -r 2 -o jsonl > $O/$T-prefill.jsonl 2> $O/$T-prefill.err
python3 - <<PY
import json
for f in ["$O/$T-decode.jsonl","$O/$T-prefill.jsonl"]:
    for l in open(f):
        d=json.loads(l); print("$T", "p=%d n=%d d=%d"%(d["n_prompt"],d["n_gen"],d["n_depth"]), "%.2f t/s"%d["avg_ts"], [round(x,2) for x in d["samples_ts"]])
PY
