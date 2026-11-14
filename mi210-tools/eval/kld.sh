#!/usr/bin/env bash
# usage (inside container): /lab/eval/kld.sh <bindir> <tag> [base]   (base: save reference logits from this run)
B=$1; T=$2; MODE=${3:-cmp}; O=/lab/results/kld-$T; mkdir -p $O
M=/data/model/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
REF=/lab/eval/ref-logits.bin
export CODEX_IQ3_PREPACK=${CODEX_IQ3_PREPACK:-1} CODEX_IQ3_XXS_PREPACK=${CODEX_IQ3_XXS_PREPACK:-1}
if [ "$MODE" = base ]; then
  $B/llama-perplexity -m $M -dev ROCm0 -ngl 999 -fa on -b 512 -ub 1 -c 512 --chunks 8 -f /lab/eval/ppl.txt --kl-divergence-base $REF > $O/kld.log 2>&1
else
  $B/llama-perplexity -m $M -dev ROCm0 -ngl 999 -fa on -b 512 -ub 1 -c 512 --chunks 8 --kl-divergence-base $REF --kl-divergence > $O/kld.log 2>&1
fi
grep -E "Final estimate|Mean PPL|Mean    KLD|Maximum KLD|99.9%  KLD|Same top p|RMS Δp|Mean    Δp" $O/kld.log | sed "s/^ *//" | head -12
