#!/bin/bash
# Locked vLLM run: take the shared GPU lock, start a vLLM server for the official
# Qwen3.8-27B-FP8 checkpoint, run a host-side job script against it, save logs, remove
# the container, release the lock.
#
# usage: vllm_run.sh <tag> <job.sh> [extra vllm serve args...]
#   <job.sh> runs on the HOST with env: BASE (http://127.0.0.1:PORT) MODEL OUT NAME PORT H (harness dir)
# env knobs:
#   ROLE      role name (default scout) -> container vllm-opt-$ROLE-$tag, out dir $WD/runs/<tag>
#   PORT      host port (default 18100; each role uses its own 181xx range)
#   CACHE     cache dir mounted at /cache (default $WD/cache; created). Never point at /data/model.
#   OVERLAYS  file with lines "<host file> <container path>" bind-mounted read-only (overlay vLLM modules)
#   VOPT      1 -> mount /data/tools/mi210-opt/vllm-opt read-only at /vopt and set
#             PYTHONPATH=/vopt/$ROLE/plugin (usercustomize.py hook, see plugin/)
#   PLUGIN_DIR override plugin dir inside the container (default /vopt/$ROLE/plugin)
#   EXTRA_ENV space separated K=V passed to the container (e.g. "VOPT_PATCHES=fp8gemv")
#   PROFILE   1 -> enable vLLM torch profiler, traces to /cache/prof/<tag> (host: $CACHE/prof/<tag>)
#   MTP       speculative tokens (default 0 = off)
#   MAXLOCK   max seconds to hold the GPU lock (default 840 = 14 min); the job is killed at the deadline
#   IMG       image (default 2e7da1ad1c66)
#   BASE_OVERLAY 0 -> do not apply the reference fp8_utils.py overlay (default 1 = same as peak/vllm_bench.sh)
set -u
ROLE=${ROLE:-fp8gemv}
W=/data/tools/mi210-opt
WD=$W/vllm-opt/$ROLE
TAG=$1; JOB=$2; shift 2
PORT=${PORT:-18110}
CACHE=${CACHE:-$WD/cache}
IMG=${IMG:-2e7da1ad1c66}
MAXLOCK=${MAXLOCK:-840}
MTP=${MTP:-0}
OUT=$WD/runs/$TAG; mkdir -p $OUT $CACHE
NAME=vllm-opt-$ROLE-$TAG
H=$(cd "$(dirname "$0")" && pwd)
BASE_OVERLAY=${BASE_OVERLAY:-1}
COMP_DEFAULT='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'
COMP=${COMP:-$COMP_DEFAULT}   # vLLM --compilation-config JSON
OV=/data/model/vllm-qwen38-fp8/optimization/fp8_utils.py
VLLM_PY=/usr/local/lib/python3.12/dist-packages/vllm
TGT=$VLLM_PY/model_executor/layers/quantization/utils/fp8_utils.py

MOUNTS=(--mount type=bind,src=/data/model/Qwen3.8-27B-FP8,dst=/model,readonly
        --mount type=bind,src=$CACHE,dst=/cache)
[ "$BASE_OVERLAY" = "1" ] && MOUNTS+=(--mount type=bind,src=$OV,dst=$TGT,readonly)
if [ -n "${OVERLAYS:-}" ]; then
  while read -r SRC DST; do
    [ -z "$SRC" ] && continue; case "$SRC" in \#*) continue;; esac
    MOUNTS+=(--mount type=bind,src=$SRC,dst=$DST,readonly)
  done < "$OVERLAYS"
fi
ENVS=(-e HIP_VISIBLE_DEVICES=0 -e HSA_NO_SCRATCH_RECLAIM=1 -e VLLM_ROCM_USE_AITER=0 -e HF_HUB_OFFLINE=1
      -e TRANSFORMERS_OFFLINE=1 -e TRITON_CACHE_DIR=/cache/triton -e TORCHINDUCTOR_CACHE_DIR=/cache/torch
      -e VLLM_CACHE_ROOT=/cache/vllm -e TORCH_EXTENSIONS_DIR=/cache/torch_ext)
if [ "${VOPT:-0}" = "1" ]; then
  MOUNTS+=(--mount type=bind,src=$W/vllm-opt,dst=/vopt,readonly)
  ENVS+=(-e PYTHONPATH=${PLUGIN_DIR:-/vopt/$ROLE/plugin})
fi
for kv in ${EXTRA_ENV:-}; do ENVS+=(-e "$kv"); done
SPEC=()
[ "$MTP" != "0" ] && SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}")
PROF=()
if [ "${PROFILE:-0}" = "1" ]; then
  mkdir -p $CACHE/prof/$TAG
  PROF=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/cache/prof/$TAG\",\"ignore_frontend\":true,\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":true,\"torch_profiler_dump_cuda_time_total\":false,\"torch_profiler_record_shapes\":false}")
fi

if ss -ltn | grep -q ":$PORT "; then echo "port $PORT busy"; exit 2; fi
echo "[$(date +%T)] waiting for GPU lock"
exec 9>$W/gpu.lock; flock 9
T0=$(date +%s)
echo "[$(date +%T)] lock acquired; starting $NAME on port $PORT"
docker rm -f $NAME >/dev/null 2>&1
docker run --detach --name $NAME --network host --ipc host --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit stack=67108864 \
  "${MOUNTS[@]}" "${ENVS[@]}" \
  --entrypoint vllm $IMG serve /model --served-model-name qwen38 --host 127.0.0.1 --port $PORT \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs ${MAX_SEQS:-128} --gpu-memory-utilization 0.92 \
  --max-num-batched-tokens 8192 --attention-backend TRITON_ATTN --language-model-only \
  --compilation-config "$COMP" \
  "${SPEC[@]}" "${PROF[@]}" "$@" > $OUT/cid.txt
UP=0
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && { UP=1; break; }
  docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null | grep -q true || break
  sleep 5
done
docker logs $NAME > $OUT/server-start.log 2>&1
if [ $UP != 1 ]; then
  echo "SERVER FAILED"; tail -60 $OUT/server-start.log
  docker rm -f $NAME >/dev/null; exit 1
fi
echo "[$(date +%T)] server up after $(( $(date +%s) - T0 )) s; vram $(( $(cat /sys/class/drm/card1/device/mem_info_vram_used)/1048576 )) MiB"
LEFT=$(( MAXLOCK - ($(date +%s) - T0) - 20 ))
[ $LEFT -lt 30 ] && LEFT=30
BASE=http://127.0.0.1:$PORT MODEL=qwen38 OUT=$OUT NAME=$NAME PORT=$PORT H=$H CACHE=$CACHE TAG=$TAG \
  timeout $LEFT bash $JOB 2>&1 | tee $OUT/job.log
docker logs $NAME > $OUT/server.log 2>&1
docker rm -f $NAME >/dev/null
echo "[$(date +%T)] done; lock held $(( $(date +%s) - T0 )) s"
