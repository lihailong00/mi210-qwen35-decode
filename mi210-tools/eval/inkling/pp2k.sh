#!/bin/bash
# inside container: pp2k.sh <bindir> [reps=3] [ENV=val ...] -> llama-bench pp2048 at ubatch 2048 (tok/s)
B=$1; R=${2:-3}; shift; shift
env "$@" LD_LIBRARY_PATH=$B:/opt/rocm/lib $B/llama-bench -m /data/model/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q8_K_XL.gguf -ngl 999 -fa 1 -n 0 -p 2048 -ub 2048 -b 2048 -r $R 2>&1 | grep pp2048 | sed "s|^.*pp2048 *|$B $* pp2048 |"
