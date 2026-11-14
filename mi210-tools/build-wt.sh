#!/usr/bin/env bash
# usage (inside container): /lab/build.sh <build-dir> [targets...]
set -euo pipefail
export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
SRC=${1}; B=${2}; shift 2 || true
T=${@:-ggml-hip llama-bench test-backend-ops llama-perplexity llama-completion}
if [ ! -f "$B/CMakeCache.txt" ]; then
cmake -S $SRC -B "$B" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm \
  -DAMDGPU_TARGETS=gfx90a -DGPU_TARGETS=gfx90a -DGGML_HIP=ON \
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=OFF \
  -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_RCCL=OFF \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=OFF \
  "-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16" \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
fi
cmake --build "$B" --target $T --parallel ${PAR:-6}
ls -la "$B/bin" | head -40
