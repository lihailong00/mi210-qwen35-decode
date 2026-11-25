#!/bin/bash
# inside container: swafull.sh <bindir> <copies> <ctx> [extra args]  -> prompt eval speed of a long prompt (ppl.txt repeated)
B=$1; C=$2; CTX=$3; shift 3
M=/data/model/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q8_K_XL.gguf
export LD_LIBRARY_PATH=$B:/opt/rocm/lib
F=/tmp/longp-$C.txt; rm -f $F; for i in $(seq 1 $C); do cat /lab/eval/ppl.txt >> $F; done
$B/llama-completion -m $M -ngl 999 -fa on -c $CTX -b 2048 -ub 2048 -n 8 --temp 0 --seed 1 -no-cnv -f $F "$@" > /tmp/swaf.txt 2> /tmp/swaf.err
echo "copies=$C ctx=$CTX $* rc=$? $(grep -oE "prompt eval time.*" /tmp/swaf.err | head -1)"
grep -iE "error|failed|out of memory" /tmp/swaf.err | head -3
echo "  tail: $(tail -c 120 /tmp/swaf.txt | tr "\n" " ")"
