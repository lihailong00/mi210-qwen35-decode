# Benchmark job for start.sh bench (runs on the host). STEPS (space list, default all but prof/conc):
#   t128   single-stream TPOT, 128-token prompt, NOUT (512) tokens x REPS (3), greedy
#   t4k    the same with a 4096-token prompt
#   s128   the same as t128 with default sampling (generation_config T/top_k/top_p)
#   ttft4k prefill time of a 4096-token prompt (max_tokens 1, 5 reps)
#   ppl    prompt_logprobs on a fixed ~700-token text; ppl6k: first 6000 tokens of the model README.md
#   cap    8 prompts x 128 greedy tokens with top-5 logprobs (capture.json), cap4: same at concurrency 4
#   prof   torch-profiler request (start.sh PROFILE=1)
#   conc   vllm bench serve random 512/128 at CONC ("4 16 64")
#   mtp    mtp_bench.py: 5 chat prompts x 2 reps, greedy, 384 tokens, thinking off (tok/s, acceptance, text)
#   vchk   correctness captures vs original vLLM (vchk_cap.py: 16 prompts, c1/c8/c16 greedy, c8 sampled)
STEPS=${STEPS:-"t128 t4k ttft4k ppl cap cap4"}
B=$H/bench_deploy.py
for s in $STEPS; do
  echo "== [$(date +%T)] $s"
  case $s in
    t128) python3 $B tpot --base $BASE --prompt-tokens 128 --n-out ${NOUT:-512} --reps ${REPS:-3} --out $OUT/tpot128.json ;;
    t4k) python3 $B tpot --base $BASE --prompt-tokens 4096 --n-out ${NOUT:-512} --reps ${REPS:-3} --out $OUT/tpot4096.json ;;
    s128) python3 $B tpot --base $BASE --prompt-tokens 128 --n-out ${NOUT:-512} --reps ${REPS:-3} --sampling --out $OUT/tpot128_sampling.json ;;
    ttft4k) python3 $B ttft --base $BASE --prompt-tokens 4096 --reps 5 --out $OUT/ttft4096.json ;;
    ppl) python3 $B ppl --base $BASE --out $OUT/ppl.json ;;
    ppl6k) python3 $B ppl --base $BASE --text-file ${MODEL_DIR:-/data/model/Qwen3.8-27B-FP8}/README.md --prompt-tokens 6000 --out $OUT/ppl6k.json ;;
    cap) python3 $H/capture.py --base $BASE --out $OUT/capture.json --tokens 128 --n 8 | tail -2 ;;
    cap4) python3 $H/capture.py --base $BASE --out $OUT/capture_c4.json --tokens 128 --n 8 --conc 4 | tail -2 ;;
    trans) python3 $H/trans.py $BASE $OUT/trans.json ;;
    mtp) python3 $H/mtp_bench.py $BASE $OUT/mtp_bench.json ;;
    vchk1) python3 $H/vchk_cap.py --base $BASE --out $OUT/g1s.json --tokens ${VTOK:-8} --conc 1 | tail -1 ;;
    vchk)
      C=$H/vchk_cap.py
      python3 $C --base $BASE --out $OUT/g1.json --tokens 128 --conc 1 | tail -1
      python3 $C --base $BASE --out $OUT/g8.json --tokens 128 --conc 8 | tail -1
      python3 $C --base $BASE --out $OUT/g16.json --tokens 64 --conc 16 | tail -1
      python3 $C --base $BASE --out $OUT/s8.json --tokens 96 --conc 8 --mode sample | tail -1 ;;
    prof) python3 $H/profile_req.py --base $BASE --n-out ${PROF_N:-48} --no-min-tokens; sleep 25 ;;
    conc)
      for C in ${CONC:-4 16 64}; do
        N=$(( C < 8 ? 8 : C*2 ))
        docker exec $NAME vllm bench serve --backend vllm --base-url $BASE --model $MODEL --tokenizer /model \
          --dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos \
          --num-prompts $N --max-concurrency $C --percentile-metrics ttft,tpot,itl > $OUT/c$C.log 2>&1
        echo "-- concurrency $C"; grep -E "Output token throughput|Mean TPOT|Mean TTFT" $OUT/c$C.log
      done ;;
  esac
done
true
