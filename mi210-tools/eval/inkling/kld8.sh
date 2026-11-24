#!/bin/bash
# inside container: kld8.sh <bindir> base|<tag> [ub=512]
#   8-chunk prefill KLD (-c 2048 --chunks 8): "base" writes the reference with GGML_CUDA_INK_PFATTN=0 at ub 512
#   (bit-identical to the original build's prefill), otherwise compares the run (current env) against it.
B=$1; TAG=$2; UB=${3:-512}
M=/data/model/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q8_K_XL.gguf
export LD_LIBRARY_PATH=$B:/opt/rocm/lib
BASE=/lab/results/pfkld8-base.bin
ARGS="-m $M -f /lab/eval/ppl.txt -c 2048 --chunks 8 -b 2048 -ub $UB -ngl 999 -fa on"
if [ "$TAG" = base ]; then
  GGML_CUDA_INK_PFATTN=0 $B/llama-perplexity $ARGS --kl-divergence-base $BASE > /lab/results/pfkld8-base.log 2>&1
  echo "base md5 $(md5sum < $BASE | cut -c1-12) $(grep -o "Final estimate: PPL = [0-9.]*" /lab/results/pfkld8-base.log)"
else
  $B/llama-perplexity $ARGS --kl-divergence-base $BASE --kl-divergence > /lab/results/pfkld8-$TAG.log 2>&1
  echo "$TAG ub $UB: $(grep -E "Mean    KLD|Maximum KLD|99.0%   KLD|Median  KLD|Same top p" /lab/results/pfkld8-$TAG.log | tr -s " " | tr "\n" ";")"
fi
