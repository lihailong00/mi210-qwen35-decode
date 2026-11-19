# job for vllm_run.sh: TPOT bench + reference capture (+ profile when server has PROFILE=1)
set -x
python3 $H/bench_tpot.py --base $BASE --n-out ${NOUT:-512} --reps ${REPS:-3} --out $OUT/tpot.json | tail -4
[ "${CAPTURE:-1}" = "1" ] && python3 $H/capture.py --base $BASE --out $OUT/capture.json --tokens 128 --n 8
[ "${CAPTURE4:-0}" = "1" ] && python3 $H/capture.py --base $BASE --out $OUT/capture_c4.json --tokens 128 --n 8 --conc 4
if [ "${PROFILE:-0}" = "1" ]; then python3 $H/profile_req.py --base $BASE --n-out ${PROF_N:-48}; sleep 20; fi
for C in ${CONC:-}; do
  N=$(( C < 8 ? 8 : C*2 ))
  docker exec $NAME vllm bench serve --backend vllm --base-url $BASE --model $MODEL --tokenizer /model \
    --dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos \
    --num-prompts $N --max-concurrency $C --percentile-metrics ttft,tpot,itl > $OUT/c$C.log 2>&1
  echo "== concurrency $C"; grep -E "Output token throughput|Mean TPOT" $OUT/c$C.log
done
true
