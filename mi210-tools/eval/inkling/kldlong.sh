#!/bin/bash
# inside container: kldlong.sh <bindir> <tag> <ctx> [ub=2048] [extra llama-perplexity args...]
#   env KLD_BSUF=<s>: use / write the reference /lab/results/kldlong-<ctx>-base<s>.bin instead
#   Long-context prefill KLD (p8-long): llama-perplexity on /lab/eval/ppl.txt repeated to 2 x <ctx> tokens (perplexity needs 2 x ctx),
#   -c <ctx> --chunks 1 -b 2048 -ub <ub> (the logits of the second half of the chunk are scored).
#   tag "base": writes the reference /lab/results/kldlong-<ctx>-base.bin -- run it with the ORIGINAL build
#   (/lab/unsloth-llama/base-bin) at ub 2048. Any other tag: md5 of the run's logits, PPL, and KLD vs that reference
#   (mean / median / 99% / max, same top p). Natural variation: the original build at ub 1024 / 512 vs the reference.
B=$1; TAG=$2; CTX=$3; UB=${4:-2048}; shift $(( $# < 4 ? $# : 4 )); EXTRA="$*"
M=/data/model/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q8_K_XL.gguf
export LD_LIBRARY_PATH=$B:/opt/rocm/lib
C=$(( 2*CTX / 27000 + 1 ))
F=/tmp/kldlong-$C.txt
if [ ! -s $F ]; then rm -f $F; for i in $(seq 1 $C); do cat /lab/eval/ppl.txt >> $F; done; fi
BASE=/lab/results/kldlong-$CTX-base$KLD_BSUF.bin
ARGS="-m $M -f $F -c $CTX --chunks 1 -b 2048 -ub $UB -ngl 999 -fa on $EXTRA"
if [ "$TAG" = base ]; then
  $B/llama-perplexity $ARGS --kl-divergence-base $BASE > /lab/results/kldlong-$CTX-base$KLD_BSUF.log 2>&1
  echo "base ctx $CTX ub $UB rc=$? md5 $(md5sum < $BASE | cut -c1-12) $(grep -o "Final estimate: PPL = [0-9.]*" /lab/results/kldlong-$CTX-base$KLD_BSUF.log) $(grep -oE "prompt eval time.*" /lab/results/kldlong-$CTX-base$KLD_BSUF.log | head -1)"
  grep -iE "error|failed|out of memory" /lab/results/kldlong-$CTX-base$KLD_BSUF.log | head -3
else
  O=/lab/results/kldlong-$CTX-$TAG.bin
  $B/llama-perplexity $ARGS --kl-divergence-base $O > /lab/results/kldlong-$CTX-$TAG-run.log 2>&1
  echo "$TAG ctx $CTX ub $UB rc=$? md5 $(md5sum < $O | cut -c1-12) $(grep -o "Final estimate: PPL = [0-9.]*" /lab/results/kldlong-$CTX-$TAG-run.log) $(grep -oE "prompt eval time.*" /lab/results/kldlong-$CTX-$TAG-run.log | head -1)"
  rm -f $O
  $B/llama-perplexity $ARGS --kl-divergence-base $BASE --kl-divergence > /lab/results/kldlong-$CTX-$TAG.log 2>&1
  echo "  $(grep -E "Mean    KLD|Median  KLD|99.0%   KLD|Maximum KLD|Same top p" /lab/results/kldlong-$CTX-$TAG.log | tr -s " " | tr "\n" ";")"
  grep -iE "error|failed|out of memory" /lab/results/kldlong-$CTX-$TAG.log | head -3
fi
