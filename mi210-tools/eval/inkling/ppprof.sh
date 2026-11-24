#!/bin/bash
# inside container: ppprof.sh <bindir> <tag> <n_prompt> <ubatch>  -> rocprofv3 kernel summary of llama-bench prefill (warmup + 1 run)
B=$1; TAG=$2; NP=$3; UB=$4; M=/data/model/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q8_K_XL.gguf
rm -rf /lab/results/ppprof-$TAG
LD_LIBRARY_PATH=$B:/opt/rocm/lib timeout 600 rocprofv3 --kernel-trace --output-format csv -d /lab/results/ppprof-$TAG -o run -- $B/llama-bench -m $M -ngl 999 -fa 1 -n 0 -p $NP -ub $UB -b 8192 -r 1 > /lab/results/ppprof-$TAG.log 2>&1
grep -E "pp[0-9]" /lab/results/ppprof-$TAG.log | cut -c60-
python3 /lab/inktools/ppsum.py /lab/results/ppprof-$TAG/run_kernel_trace.csv 45
