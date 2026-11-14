#!/usr/bin/env bash
# usage (inside container): /lab/eval/ppl-only.sh <bindir> <tag>
B=$1; T=$2; O=/lab/results/eval-$T; mkdir -p $O
M=/data/model/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
export CODEX_IQ3_PREPACK=${CODEX_IQ3_PREPACK:-1} CODEX_IQ3_XXS_PREPACK=${CODEX_IQ3_XXS_PREPACK:-1}
$B/llama-perplexity -m $M -dev ROCm0 -ngl 999 -fa on -b 512 -ub 1 -c 512 --chunks 8 -f /lab/eval/ppl.txt > $O/ppl.log 2>&1
echo "$T $(grep -E "Final estimate" $O/ppl.log | sed "s/.*Final/Final/")"
