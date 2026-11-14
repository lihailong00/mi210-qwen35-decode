#!/usr/bin/env bash
# usage (inside container): /lab/eval/eval.sh <bindir> <tag>
# PPL with ubatch=1 exercises the single-token decode path for every token; greedy completions check end-to-end output.
set -uo pipefail
B=$1; T=$2; O=/lab/results/eval-$T; mkdir -p $O
M=/data/model/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
export CODEX_IQ3_PREPACK=${CODEX_IQ3_PREPACK:-1} CODEX_IQ3_XXS_PREPACK=${CODEX_IQ3_XXS_PREPACK:-1}
$B/llama-perplexity -m $M -dev ROCm0 -ngl 999 -fa on -b 512 -ub 1 -c 512 --chunks 8 -f /lab/eval/ppl.txt > $O/ppl.log 2>&1
grep -E "Final estimate" $O/ppl.log || tail -5 $O/ppl.log
for p in p1 p2 p3; do
  $B/llama-completion -m $M -dev ROCm0 -ngl 999 -fa on -f /lab/eval/prompts/$p.txt -n 128 --temp 0 -s 1 --ignore-eos -no-cnv --no-display-prompt --simple-io > $O/$p.out 2> $O/$p.err < /dev/null
done
md5sum $O/p*.out
