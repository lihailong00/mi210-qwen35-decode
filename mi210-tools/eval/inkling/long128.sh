#!/bin/bash
# inside container: long128.sh <bindir> <n_prompt> <ub>  -> llama-bench prefill of a long prompt with host profilers
B=$1; NP=$2; UB=${3:-2048}
export LD_LIBRARY_PATH=$B:/opt/rocm/lib
GGML_CUDA_INK_HOSTPROF=1 LLAMA_INK_HOSTPROF=1 $B/llama-bench -m /data/model/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q8_K_XL.gguf -ngl 999 -fa 1 -n 0 -p $NP -ub $UB -b $UB -r 1 --no-warmup > /lab/results/long-$NP-$UB.log 2>&1
echo "rc=$?"; grep -E "pp[0-9]+ " /lab/results/long-$NP-$UB.log | cut -c60-
grep -iE "hostprof|cycle|capture|replay|direct|decode|build|alloc|inputs|compute|wait" /lab/results/long-$NP-$UB.log | tail -40
