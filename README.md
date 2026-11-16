# mi210-qwen35-decode

在单张 AMD Instinct MI210（gfx90a，ROCm 7.2.1）上，把 llama.cpp b12110 跑 Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf（架构 qwen35）的单流解码（不开 MTP）从 **45.6 提升到 80.3 token/s（+76%）**。同一个 GGUF，没有重新量化或降精度；与基线 logits 的 KL 散度处在纯计算重排的噪声底线上。

| 指标 | 基线（保留版） | 本仓库 | 变化 |
| --- | --- | --- | --- |
| decode tg128，上下文 0 | 45.6 token/s | 80.3 token/s | +76% |
| decode tg128，上下文 8192 | 44.2 | 74.6 | +69% |
| decode tg128，上下文 32768 | 40.7 | 64.4 | +58% |
| prefill pp512 / pp2048 / pp8192 | 893 / 1099 / 1220 | 1255 / 2024 / 2075 | +41% / +84% / +70% |
| llama-server 单请求 | 45.6 | 79.1 | +73% |
| 每 token kernel 启动数 | 1933 | 601 | −69% |
| 显存峰值 | 24.1 GB | 28–32 GB | 重排权重副本 |

prefill 的提升来自两处：让 rocBLAS 在 CDNA2 上默认使用 hipBLASLt 后端（rocBLAS 自带选核对 ffn_down 形状只跑到 47 TFLOPS），更快的权重反量化启动方式、逐形状的 GEMM 选核，为 wave64 重写、后来又拆成并行预处理加波分工串行两段的 fp32 MFMA DeltaNet kernel，flash attention 的装载预取和 GQA 6 分发修正，以及四项小算子融合（卷积输入、SwiGLU 直接写 f16、f16 副本共享、残差加法并入 GEMM）；本机实测的 decode / prefill 上限约 124 / 2980 token/s（prefill 受 300 W 功耗上限限制），见交接文档第十三节。

原定目标 90 token/s 未达到。原因分析、全部优化细节、失败的尝试和后续方向见 [docs/HANDOVER.md](docs/HANDOVER.md)。

## 分支与标签

- `main`：最终版本的代码（= 标签 `mi210-final`）加本说明、交接文档和实验脚本。
- `baseline`：对照基线，即 llama.cpp b12110（提交 cd959bae）+ ROCm 7.2 兼容补丁 + IQ3_S/IQ3_XXS 4-bit 展开补丁（"保留版"）。`git diff baseline main` 就是全部优化。
- 标签 `int-v8-base`（74.0 token/s）、`int-v9`（76.0）、`int-v11`（78.0）、`mi210-final`（80.3）为中间集成节点。

本仓库没有 llama.cpp 的上游提交历史；基线是一次性导入的源码快照。

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

HIP 源文件在 configure 时按通配符收集：新增 `.cu` 文件后必须重新运行 cmake，否则 libggml-hip.so 加载失败（报 `invalid device: ROCm0`）。

## 运行

所有测量都在下面的环境变量下进行（基线补丁的开关；新的 IQ3 路径优先，旧路径作为回退）：

```bash
export CODEX_IQ3_PREPACK=1 CODEX_IQ3_XXS_PREPACK=1
./build/bin/llama-bench -m Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf -ngl 999 -fa on -b 4096 -ub 4096 -ctk f16 -ctv f16 -p 0 -n 128 -r 2
./build/bin/llama-server -m Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf -ngl 999 -fa on -c 65536 -np 1 --host 127.0.0.1 --port 8080
```

每项优化都有关闭开关（例如 `LLAMA_GDN_FUSED_DECODE=0`、`GGML_MMVQ_IQ3=0`），完整列表见交接文档附录。

## 适用范围

- 只优化 HIP / CDNA2（gfx90a）上的单 token、单序列解码；其他 GPU 和多序列批次走原代码路径。
- 融合 DeltaNet 算子只用于 qwen35 架构、单序列、未开启投机回滚的情况，其余自动回退。
- CUDA/MUSA 构建的修复只做过 clang 语法检查，未在 NVIDIA 平台编译测试。

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
