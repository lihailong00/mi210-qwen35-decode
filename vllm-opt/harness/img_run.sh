#!/bin/bash
# Run a command inside the vLLM image with the GPU, under the shared GPU lock.
# usage: img_run.sh <tag> "<command>"      (command runs with cwd /vopt/$ROLE/plugin)
# mounts: /data/tools/mi210-opt/vllm-opt -> /vopt (ro), $CACHE -> /cache (rw), model -> /model (ro)
# env: ROLE (scout) CACHE ($WD/cache) MAXLOCK (840 s) IMG (2e7da1ad1c66) EXTRA_ENV "K=V ..."
set -u
ROLE=${ROLE:-fp8gemv}
W=/data/tools/mi210-opt; WD=$W/vllm-opt/$ROLE
TAG=$1; CMD=$2
CACHE=${CACHE:-$WD/cache}; IMG=${IMG:-2e7da1ad1c66}; MAXLOCK=${MAXLOCK:-840}
NAME=vllm-opt-$ROLE-$TAG
mkdir -p $CACHE
ENVS=(-e HIP_VISIBLE_DEVICES=0 -e TORCH_EXTENSIONS_DIR=/cache/torch_ext -e TRITON_CACHE_DIR=/cache/triton
      -e PYTHONPATH=/vopt/$ROLE/plugin -e VLLM_ROCM_USE_AITER=0)
for kv in ${EXTRA_ENV:-}; do ENVS+=(-e "$kv"); done
exec 9>$W/gpu.lock; flock 9
T0=$(date +%s)
docker rm -f $NAME >/dev/null 2>&1
timeout $MAXLOCK docker run --rm --name $NAME --network host --ipc host --device /dev/kfd --device /dev/dri \
  --group-add video --security-opt seccomp=unconfined \
  --mount type=bind,src=$W/vllm-opt,dst=/vopt,readonly --mount type=bind,src=$CACHE,dst=/cache \
  --mount type=bind,src=/data/model/Qwen3.8-27B-FP8,dst=/model,readonly \
  "${ENVS[@]}" -w /vopt/$ROLE/plugin --entrypoint bash $IMG -c "$CMD"
RC=$?
docker rm -f $NAME >/dev/null 2>&1
echo "[img_run $TAG] rc=$RC lock held $(( $(date +%s) - T0 )) s"
exit $RC
