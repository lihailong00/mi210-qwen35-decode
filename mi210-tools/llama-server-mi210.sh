#!/usr/bin/env bash
# Optimized llama-server (b12110 + MI210 single-stream decode optimizations) for the MI210.
# Usage: ./llama-server-mi210.sh -m /data/model/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf -ngl 999 -fa on -c 65536 -np 1 --port 8080
# Notes: the binaries were built under /lab/wt-final/build (container path); the CPU backend is looked up there, so the
# workspace is mounted at /lab exactly as during the build. Image: the ROCm 7.2.1 image the build was made with.
set -euo pipefail
ws=/data/tools/mi210-opt
bin=/lab/wt-final/build/bin
render_device="$(find /sys/bus/pci/devices/0000:03:00.0/drm -maxdepth 1 -name "renderD*" -print -quit)"
if [[ ! -x "$ws/wt-final/build/bin/llama-server" || -z "$render_device" || ! -c "/dev/dri/${render_device##*/}" || ! -c /dev/kfd ]]; then
  echo "optimized build or MI210 device nodes missing" >&2; exit 1
fi
exec docker run --rm --pull=never --network host --name "llama-mi210-$$" \
  --device /dev/kfd --device "/dev/dri/${render_device##*/}" \
  --mount "type=bind,src=$ws,dst=/lab,readonly" \
  --mount type=bind,src=/data/model,dst=/data/model,readonly \
  --workdir /lab \
  --env LD_LIBRARY_PATH=$bin:/opt/rocm/lib \
  --env CODEX_IQ3_PREPACK=1 --env CODEX_IQ3_XXS_PREPACK=1 \
  --entrypoint $bin/llama-server \
  local/llama-fastmtp-rocm-profile:20261003 --host 127.0.0.1 "$@"
