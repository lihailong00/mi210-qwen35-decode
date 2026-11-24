#!/bin/bash
# inside container: kldpp.sh <bindir> <tag> [ub=512]
#   PREFILL-path logits (llama-perplexity -c 2048 --chunks 2 -b 2048 -ub <ub>, crosses the 1024 sliding window)
#   -> md5 of the logits, PPL, and KL divergence vs the ORIGINAL build prefill logits (/lab/results/inkkldpp-base.bin)
B=$1; TAG=$2; UB=${3:-512}
M=/data/model/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q8_K_XL.gguf
export LD_LIBRARY_PATH=$B:/opt/rocm/lib
O=/lab/results/inkkldpp-$TAG.bin
ARGS="-m $M -f /lab/eval/ppl.txt -c 2048 --chunks 2 -b 2048 -ub $UB -ngl 999 -fa on"
$B/llama-perplexity $ARGS --kl-divergence-base $O > /lab/results/inkkldpp-$TAG.log 2>&1
echo "$TAG md5 $(md5sum < $O | cut -c1-12) $(grep -o "Final estimate: PPL = [0-9.]*" /lab/results/inkkldpp-$TAG.log) $(grep -oE "prompt eval time.*" /lab/results/inkkldpp-$TAG.log | head -1)"
if [ -f /lab/results/inkkldpp-base.bin ] && [ "$O" != /lab/results/inkkldpp-base.bin ] && ! cmp -s $O /lab/results/inkkldpp-base.bin; then
  $B/llama-perplexity $ARGS --kl-divergence-base /lab/results/inkkldpp-base.bin --kl-divergence > /lab/results/inkkldpp-$TAG-cmp.log 2>&1
  grep -E "Mean    KLD|Maximum KLD|99.0%   KLD|Same top p" /lab/results/inkkldpp-$TAG-cmp.log | sed "s/^/  /"
fi
[ "$TAG" = base ] || rm -f $O
