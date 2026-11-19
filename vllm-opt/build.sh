#!/bin/bash
# Build all four HIP extensions (vopt_fp8gemv, vopt_lmhead, vopt_rest, vopt_spec) inside the vLLM image with
# torch.utils.cpp_extension. No GPU and no GPU lock needed. Output: $CACHE/torch_ext/* (default
# deploy-mtp/cache; FP8_R1=1: plugin_r1 -> cache_r1). If skipped, the first server start builds them (~5 min cold) into the same cache.
set -eu
D=$(cd "$(dirname "$0")" && pwd)
PLUGIN=$D/plugin; CACHE_DEF=$D/cache
[ "${FP8_R1:-0}" = 1 ] && { PLUGIN=$D/plugin_r1; CACHE_DEF=$D/cache_r1; }   # previous FP8 GEMV (see start.sh)
CACHE=${CACHE:-$CACHE_DEF}; IMG=${IMG:-2e7da1ad1c66}
mkdir -p $CACHE/torch_ext
BN=vllm-opt-deploy-mtp-build
docker rm -f $BN >/dev/null 2>&1 || true
docker run --rm --name $BN --mount type=bind,src=$PLUGIN,dst=/vopt_plugin,readonly \
  --mount type=bind,src=$CACHE,dst=/cache -e TORCH_EXTENSIONS_DIR=/cache/torch_ext -e MAX_JOBS=16 \
  -e PYTORCH_ROCM_ARCH=gfx90a -e PYTHONPATH=/vopt_plugin --entrypoint bash $IMG -c '
set -e
cd /tmp
python3 -c "from vopt_fp8gemv import fp8_gemv as F; F.load_ext()" & P1=$!
python3 -c "from vopt_lmhead import ext; ext.load_ext()" & P2=$!
python3 -c "from vopt_rest import ext; ext.load_ext()" & P3=$!
python3 -c "from vopt_spec import ext; ext.load_ext()" & P4=$!
wait $P1; wait $P2; wait $P3; wait $P4
'
echo "built into $CACHE/torch_ext"
