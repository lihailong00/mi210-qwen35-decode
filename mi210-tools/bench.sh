#!/usr/bin/env bash
# usage (inside container): /lab/bench.sh <bindir> <tag> [extra llama-bench args]   -> decode tg128 at depth 0 and 8192
B=$1; T=$2; shift 2
M=/data/model/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
export CODEX_IQ3_PREPACK=${CODEX_IQ3_PREPACK:-1} CODEX_IQ3_XXS_PREPACK=${CODEX_IQ3_XXS_PREPACK:-1}
mkdir -p /lab/results/bench
$B/llama-bench -m $M -dev ROCm0 -ngl 999 -fa on -b 4096 -ub 4096 -ctk f16 -ctv f16 -p 0 -n 128 -r 2 "$@" -o jsonl > /lab/results/bench/$T.jsonl 2> /lab/results/bench/$T.err
python3 -c "
import json,sys
for l in open(\"/lab/results/bench/$T.jsonl\"):
    d=json.loads(l); print(\"$T\", \"p=%d n=%d d=%d\"%(d[\"n_prompt\"],d[\"n_gen\"],d[\"n_depth\"]), \"%.2f t/s\"%d[\"avg_ts\"], d[\"samples_ts\"])
"
