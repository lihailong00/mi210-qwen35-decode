# MI210 LLM 推理优化

在 AMD Instinct MI210（gfx90a / CDNA2，64 GB HBM2e，ROCm 7.2.1）上优化大模型推理，目标是在**不重新量化、不降精度**的前提下逼近这张卡的硬件上限。主体是 llama.cpp 的 HIP 后端（本仓库的源码树），另有一个给 vLLM 打补丁的插件目录 [vllm-opt/](vllm-opt/README.md)。

## 结果

| 模型 | 后端 / 权重 | 单流 decode（基线 → 现在） | 投机解码（MTP / DSpark） | prefill pp8192 | 上限（decode / prefill） | 文档 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B | llama.cpp，IQ3_S GGUF | 45.6 → **80.3** token/s | K=3 约 **155** token/s，输出与不开逐字节相同 | 1220 → **2125** token/s | 124 / 约 2980 | [qwen3.8-27b](docs/models/qwen3.8-27b/README.md) |
| Qwen3.8-27B | vLLM，官方 FP8 | 9.9 → **42** token/s | K=3 **106–122** token/s，输出与 MTP=0 逐字节相同 | — | 约 53 / — | [qwen3.8-27b](docs/models/qwen3.8-27b/README.md)、[vllm-opt](vllm-opt/README.md) |
| Inkling-2-Mini | llama.cpp（Unsloth 合并版），UD-Q8_K_XL GGUF | 33.3 → **83.6** token/s（小算子融合 + GEMV + 注意力 / 路由，KLD 0.0010） | DSpark n_max 2：贪心 38.3 token/s（验证未融合，比不开慢），输出与不开不同 | pp2048 513 token/s | 约 205–240 / — | [inkling-2-mini](docs/models/inkling-2-mini/README.md) |

正确性的标准是：非位精确的改动与基线 logits 的 KL 散度处在纯计算重排的噪声底线上；融合、多列 kernel 和投机解码要求逐位 / 逐字节相同。方法见 [docs/mi210/method.md](docs/mi210/method.md)。

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/mi210/hardware.md](docs/mi210/hardware.md) | MI210 的实测峰值、性能模型、kernel 启动与 graph、冷指令缓存、内存系统、调度、ISA、数学库、测量工具的失真 |
| [docs/mi210/method.md](docs/mi210/method.md) | 优化流程、测量与 A/B 协议、正确性标尺（KLD 噪声底线、位精确检查）、工程约定、共享 GPU 规则、**新模型上手清单** |
| [docs/kernels/README.md](docs/kernels/README.md) | 已有的每项优化：触发条件、是否与模型无关、文件、开关 |
| [docs/models/](docs/models/README.md) | 每个模型的概览、上限、结果和完整工作记录；新增模型的步骤 |

## 适用范围

- 所有改动都针对 HIP / CDNA2（gfx90a），其他 GPU 走 llama.cpp 原来的代码路径。CUDA/MUSA 构建的修复只做过 clang 语法检查。
- 按量化类型写的 decode GEMV（IQ3_S/IQ3_XXS、IQ4_XS/Q4_K、IQ2/IQ1/Q2_K、Q6_K）、q8 激活缓存与生产者、反量化、hipBLASLt 默认开启、多数 prefill 小算子融合，对任何模型都会生效。
- 一部分优化是按 Qwen3.8-27B 的形状或计算图写的（头维 256 的 FA decode、GQA 6 的分发、5120 列的定形 norm、GEMM 固定表、融合 DeltaNet 解码等），其他模型会自动回退到原路径。逐项列表见 [docs/kernels/README.md](docs/kernels/README.md)。
- decode 优化覆盖单序列、1–16 个 token（单 token 与投机验证）；多序列批次走原路径。

## 编译

需要 ROCm 7.2.x（hipcc/clang）、CMake ≥ 3.28、Ninja。示例（与实验时的配置相同）：

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm \
  -DAMDGPU_TARGETS=gfx90a -DGPU_TARGETS=gfx90a -DGGML_HIP=ON \
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=OFF \
  -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_RCCL=OFF \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=OFF "-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16" \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_UI=OFF
cmake --build build --target ggml-hip llama-bench llama-perplexity llama-completion llama-server --parallel 16
```

HIP 源文件在 configure 时按通配符收集：新增 `.cu` 文件后必须重新运行 cmake，否则 libggml-hip.so 加载失败（报 `invalid device: ROCm0`，或者悄悄只用 CPU）。

## 运行

用法与上游 llama.cpp 相同。每项优化都有环境变量开关（默认开，设 0 关闭），列表见 [docs/kernels/README.md](docs/kernels/README.md)。

```bash
./build/bin/llama-bench  -m <model.gguf> -ngl 999 -fa on -b 4096 -ub 4096 -p 512,2048,8192 -n 128 -r 2
./build/bin/llama-server -m <model.gguf> -ngl 999 -fa on -c 65536 -np 1 --host 127.0.0.1 --port 8080
```

带 IQ3_S / IQ3_XXS 权重的模型请设置 `CODEX_IQ3_PREPACK=1 CODEX_IQ3_XXS_PREPACK=1`（基线补丁的开关；所有测量都在这个设置下进行）。各模型的具体命令（包括 MTP）在各自的文档里。

## 分支与标签

- `main`：当前代码、文档和实验脚本。
- `baseline`：对照基线，即 llama.cpp b12110（提交 cd959bae）+ ROCm 7.2 兼容补丁 + IQ3_S/IQ3_XXS 4-bit 展开补丁（"保留版"）。`git diff baseline main` 就是全部优化。
- 已有标签 `int-v8-base`、`int-v9`、`int-v11`、`mi210-final` 是 Qwen3.8-27B decode 优化的中间集成节点（74.0 / 76.0 / 78.0 / 80.3 token/s），保持不变。
- 以后的标签用 `<模型id>/<名字>` 的形式，例如 `qwen3.8-27b/mtp-r3`。

本仓库没有 llama.cpp 的上游提交历史；基线是一次性导入的源码快照。

## 实验工具

`mi210-tools/` 收录实验时用的脚本（路径按原实验环境写死，使用前需按自己的目录修改）：

- `gpu-run.sh`：用 flock 串行化 GPU 作业，记录温度与显存，85°C 中止。
- `build-wt.sh`、`bench.sh`、`final-run.sh`、`merge-int.sh`：编译、测速、A/B、分支合并。
- `eval/`：KL 散度（`kld.sh`）、困惑度与贪心输出（`eval.sh`）、服务器测试（`server_test.py`）、提示词。困惑度文本为 llama.cpp 仓库中 `README.md docs/build.md docs/function-calling.md docs/multimodal.md` 依次拼接。
- `profiling/`：rocprofv3 trace 汇总脚本、graph 内 GPU 时间戳补丁（剖析专用，不要合入正式版本）及其分析脚本。
- `gemv-bench/`：用真实权重测 GEMV kernel 的独立微基准。
- `peak-bench/`：HBM 带宽、MFMA 峰值、逐张量读取下限和 prefill GEMM 选核的微基准。
- `llama-server-mi210.sh`：在容器里启动服务器的示例脚本。

## 许可

代码基于 llama.cpp（MIT 许可），沿用其 LICENSE。
