#!/bin/bash
# Optimized vLLM deployment of the official Qwen3.8-27B-FP8 on one MI210 (gfx90a), v2: MTP speculative
# decode with per-token exact verify kernels (MTP output byte-identical to MTP=0 under greedy).
# Nothing inside the image is changed: the plugin dir is bind-mounted read-only and activated via
# PYTHONPATH (usercustomize.py post-import hooks); HIP extensions are built into $CACHE/torch_ext.
#
# usage:
#   start.sh [serve] [extra vllm serve args...]     normal server, foreground (Ctrl-C / docker rm -f stops it)
#   start.sh bench <tag> <job.sh> [extra args...]   benchmark run: take the GPU lock, start detached,
#                                                   wait until ready, run <job.sh> on the host, save
#                                                   logs to runs/<tag>/, remove the container, release
# env:
#   PORT (18340)  NAME (vllm-opt-deploy-mtp-serve | vllm-opt-deploy-mtp-<tag>)  CACHE (deploy-mtp/cache)
#   MTP=n         MTP speculative tokens per step (default 3; 0 = off)
#   MTP_SCHED     dynamic K by batch size (vLLM num_speculative_tokens_per_batch_size, JSON [[from,to,K],..]);
#                 NOT usable with this vLLM build: engine start fails in the speculator graph capture
#   BLOCK_SIZE    KV/mamba block size (default 816 = what MTP=4 needs; one value for every MTP keeps
#                 long-prompt prefill chunking identical across K, so MTP output = MTP=0 output; auto = vLLM picks)
#   MTP_BLOCK_DROP 0 (default) = spec config disable_eagle_block_drop: prefill chunks split at the same block
#                 boundaries as MTP=0 (vLLM otherwise backs off one block with MTP, so long prompts are
#                 prefilled in different chunks and the first tokens can differ from MTP=0); 1 = vLLM default
#   V1=1          run deploy v1 instead (deploy/plugin, its 3 packages, cache_v1) for A/B
#   PLUGIN        plugin dir to mount (default deploy-mtp/plugin)
#   GPU_LOCK      serve mode: 1 = hold $GPU_LOCK_FILE while the server runs (default 1; 0 = skip, only when nobody else uses the GPU)
#   GPU_LOCK_FILE flock file shared by every GPU user on the host (default /data/tools/mi210-opt/gpu.lock)
#   MODEL_DIR     official checkpoint dir (default /data/model/Qwen3.8-27B-FP8), mounted read-only at /model
#   FP8_OVERLAY   optional fp8_utils.py bind-mounted over vLLM's copy (the host's pre-existing tuned Triton
#                 path for M>8; default /data/model/vllm-qwen38-fp8/optimization/fp8_utils.py; skipped if absent)
#   V1_PLUGIN     deploy v1 plugin dir used by V1=1 (default /data/tools/mi210-opt/vllm-opt/deploy/plugin)
#   ORIG=1        original setup (stock vLLM + reference fp8_utils overlay, no vopt patches) for A/B
#   VOPT=0        same as ORIG=1 (disable every custom piece)
#   per-piece opt-outs (default 1 = on):
#     VOPT_FP8GEMV  weight-only FP8 GEMV for M<=8 (+ tiled Triton fallback for larger M)
#     VOPT_LMHEAD   lm_head bf16 GEMV (M<=8)
#     VOPT_SAMPLER  fused greedy argmax + fused top-k/top-p sampler (needs VOPT_LMHEAD)
#     VOPT_BF16SMALL split-K GEMV for small bf16 linears (in_proj_ba)
#     VOPT_NORMS    fused GemmaRMSNorm(+residual)      VOPT_ACT  SiLU*mul
#     VOPT_GDN      fused GDN decode kernel            VOPT_ATTN fused qk-norm/RoPE/gate + sigmoid gate
#     VOPT_SPEC     vopt_spec package: int4 draft lm_head, greedy verify fast path, GDN metadata fast path
#   finer switches passed into the container when set on the host (see README):
#     VOPT_FP8_EXACT VOPT_GDN_SPEC VOPT_ATTN_SPEC VOPT_DRAFT_HEAD VOPT_DRAFT_VOCAB VOPT_DRAFT_DELTA
#     VOPT_REJECT_FAST VOPT_GDN_MD VOPT_LMHEAD_MAXM VOPT_FC_GEMV (and anything via EXTRA_ENV)
#   PROFILE=1 (bench mode: torch profiler traces into $CACHE/prof/<tag>)
#   MAXLOCK (bench mode: max seconds holding the lock, default 840)
#   EXTRA_ENV "K=V K2=V2" extra container env (e.g. VOPT_FP8_MAXM=8)
set -u
D=$(cd "$(dirname "$0")" && pwd)
LOCKF=${GPU_LOCK_FILE:-/data/tools/mi210-opt/gpu.lock}
MODEL_DIR=${MODEL_DIR:-/data/model/Qwen3.8-27B-FP8}
MODE=serve
case "${1:-}" in serve) shift;; bench) MODE=bench; TAG=$2; JOB=$3; shift 3;; esac
PORT=${PORT:-18340}
MTP=${MTP:-3}
PKGS=vopt_fp8gemv,vopt_lmhead,vopt_rest
if [ "${V1:-0}" = 1 ]; then
  PLUGIN=${PLUGIN:-${V1_PLUGIN:-/data/tools/mi210-opt/vllm-opt/deploy/plugin}}; CACHE=${CACHE:-$D/cache_v1}
else
  PLUGIN=${PLUGIN:-$D/plugin}; CACHE=${CACHE:-$D/cache}
  [ "${VOPT_SPEC:-1}" = 1 ] && PKGS=$PKGS,vopt_spec
fi
IMG=${IMG:-2e7da1ad1c66}
MAXLOCK=${MAXLOCK:-840}
if [ $MODE = bench ]; then NAME=${NAME:-vllm-opt-deploy-mtp-$TAG}; OUT=$D/runs/$TAG; mkdir -p $OUT
else NAME=${NAME:-vllm-opt-deploy-mtp-serve}; fi
mkdir -p $CACHE

ON() { [ "${ORIG:-0}" != 1 ] && [ "${VOPT:-1}" = 1 ] && [ "$(eval echo \${$1:-1})" = 1 ]; }
P=()
ON VOPT_FP8GEMV && P+=(fp8gemv)
if ON VOPT_LMHEAD; then P+=(lmhead); ON VOPT_SAMPLER && P+=(greedy topk); fi
ON VOPT_BF16SMALL && P+=(bf16gemv)
ON VOPT_NORMS && P+=(norms)
ON VOPT_ACT && P+=(act)
ON VOPT_GDN && P+=(gdn)
ON VOPT_ATTN && P+=(attn)
PATCHES=$(IFS=,; echo "${P[*]:-}")

OV=${FP8_OVERLAY:-/data/model/vllm-qwen38-fp8/optimization/fp8_utils.py}
TGT=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py
MOUNTS=(--mount type=bind,src=$MODEL_DIR,dst=/model,readonly
        --mount type=bind,src=$CACHE,dst=/cache)
if [ -f "$OV" ]; then MOUNTS+=(--mount type=bind,src=$OV,dst=$TGT,readonly)
else echo "[deploy] note: FP8_OVERLAY $OV not found, using the image's fp8_utils.py" >&2; fi
ENVS=(-e HIP_VISIBLE_DEVICES=0 -e HSA_NO_SCRATCH_RECLAIM=1 -e VLLM_ROCM_USE_AITER=0 -e HF_HUB_OFFLINE=1
      -e TRANSFORMERS_OFFLINE=1 -e TRITON_CACHE_DIR=/cache/triton -e TORCHINDUCTOR_CACHE_DIR=/cache/torch
      -e VLLM_CACHE_ROOT=/cache/vllm -e TORCH_EXTENSIONS_DIR=/cache/torch_ext -e PYTORCH_ROCM_ARCH=gfx90a)
if [ -n "$PATCHES" ]; then
  MOUNTS+=(--mount type=bind,src=$PLUGIN,dst=/vopt_plugin,readonly)
  ENVS+=(-e PYTHONPATH=/vopt_plugin -e VOPT_PKGS=${VOPT_PKGS:-$PKGS} -e VOPT_PATCHES=$PATCHES)
  for v in VOPT_FP8_EXACT VOPT_GDN_SPEC VOPT_ATTN_SPEC VOPT_DRAFT_HEAD VOPT_DRAFT_VOCAB VOPT_DRAFT_DELTA            VOPT_REJECT_FAST VOPT_GDN_MD VOPT_LMHEAD_MAXM VOPT_FC_GEMV; do
    [ -n "${!v:-}" ] && ENVS+=(-e "$v=${!v}")
  done
fi
for kv in ${EXTRA_ENV:-}; do ENVS+=(-e "$kv"); done
SPEC=()
BS=(); [ "${V1:-0}" != 1 ] && [ "${BLOCK_SIZE:-816}" != auto ] && BS=(--block-size ${BLOCK_SIZE:-816})
EBD=""; [ "${V1:-0}" != 1 ] && [ "${MTP_BLOCK_DROP:-0}" = 0 ] && EBD=",\"disable_eagle_block_drop\":true"
SCHED=""; [ -n "${MTP_SCHED:-}" ] && SCHED=",\"num_speculative_tokens_per_batch_size\":$MTP_SCHED"
[ "$MTP" != 0 ] && SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP$SCHED$EBD}")
PROF=()
if [ $MODE = bench ] && [ "${PROFILE:-0}" = 1 ]; then
  mkdir -p $CACHE/prof/$TAG
  PROF=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/cache/prof/$TAG\",\"ignore_frontend\":true,\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":true,\"torch_profiler_dump_cuda_time_total\":false,\"torch_profiler_record_shapes\":false}")
fi
DOCKER=(docker run --name $NAME --network host --ipc host --device /dev/kfd --device /dev/dri --group-add video
  --security-opt seccomp=unconfined --ulimit memlock=-1 --ulimit stack=67108864 "${MOUNTS[@]}" "${ENVS[@]}")
SERVE=(--entrypoint vllm $IMG serve /model --served-model-name qwen38 --host 127.0.0.1 --port $PORT
  --dtype bfloat16 --max-model-len ${MAX_LEN:-8192} --max-num-seqs ${MAX_SEQS:-128} --gpu-memory-utilization 0.92
  --max-num-batched-tokens 8192 --attention-backend TRITON_ATTN --language-model-only "${BS[@]}"
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' "${SPEC[@]}" "${PROF[@]}")

if ss -ltn | grep -q ":$PORT "; then echo "port $PORT busy"; exit 2; fi
echo "[deploy-mtp] MTP=$MTP plugin=$PLUGIN pkgs=${VOPT_PKGS:-$PKGS} patches: ${PATCHES:-<none: original vLLM>}"

if [ $MODE = serve ]; then
  if [ "${GPU_LOCK:-1}" = 1 ]; then exec 9>$LOCKF; echo "[deploy] waiting for GPU lock"; flock 9; fi
  docker rm -f $NAME >/dev/null 2>&1
  exec "${DOCKER[@]}" --rm "${SERVE[@]}" "$@"
fi

# ---- bench mode
echo "[$(date +%T)] waiting for GPU lock"
exec 9>$LOCKF; flock 9
T0=$(date +%s)
echo "[$(date +%T)] lock acquired; starting $NAME on port $PORT"
echo "MTP=$MTP V1=${V1:-0} PLUGIN=$PLUGIN PKGS=${VOPT_PKGS:-$PKGS} PATCHES=$PATCHES EXTRA_ENV=${EXTRA_ENV:-}" > $OUT/patches.txt
docker rm -f $NAME >/dev/null 2>&1
"${DOCKER[@]}" --detach "${SERVE[@]}" "$@" > $OUT/cid.txt
UP=0
for i in $(seq 1 150); do
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && { UP=1; break; }
  docker inspect -f '{{.State.Running}}' $NAME 2>/dev/null | grep -q true || break
  sleep 4
done
docker logs $NAME > $OUT/server-start.log 2>&1
if [ $UP != 1 ]; then
  echo "SERVER FAILED"; tail -60 $OUT/server-start.log; docker rm -f $NAME >/dev/null; exit 1
fi
echo "[$(date +%T)] server up after $(( $(date +%s) - T0 )) s"
LEFT=$(( MAXLOCK - ($(date +%s) - T0) - 20 )); [ $LEFT -lt 30 ] && LEFT=30
BASE=http://127.0.0.1:$PORT MODEL=qwen38 OUT=$OUT NAME=$NAME PORT=$PORT H=$D/harness CACHE=$CACHE TAG=$TAG \
  timeout $LEFT bash $JOB 2>&1 | tee $OUT/job.log
docker logs $NAME > $OUT/server.log 2>&1
docker rm -f $NAME >/dev/null
echo "[$(date +%T)] done; lock held $(( $(date +%s) - T0 )) s"
