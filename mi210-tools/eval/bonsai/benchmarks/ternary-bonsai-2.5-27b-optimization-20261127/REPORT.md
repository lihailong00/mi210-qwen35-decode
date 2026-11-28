# Ternary-Bonsai-2.5-27B：MI210 第一轮优化

2026-11-27；单卡 MI210 64 GiB，ROCm 7.2.1，llama.cpp HIP。本轮不启用 MTP，不重新量化，不修改下载的 GGUF。

## 模型

- 文件：`$MODEL`，7,912,178,944 字节。
- SHA-256：`b0f83de8481c3200c529aef5fb495dbfedf2a4a3f017f7a3ea0a69c67be1d306`。
- qwen35：64 主干层（48 DeltaNet、16 full attention）和 1 MTP 层；hidden=5120，FFN=17408；24 Q 头、4 KV 头，头维 256。
- 主干主要为 Q2_0，64 值/18 字节，保留所有编码值 -1/0/1/2。Hadamard block=1024，与量化 block 不同。DeltaNet alpha/beta 权重为 F32，不能直接套 BF16 专用 prep。

## 实测

单位 token/s，每行两侧使用相同 batch/ubatch。原始基线为保留版 b12110 (`cd959bae9`)；优化版包含项目现有优化和新 Q2_0 内核。

| 项目 | 原始基线 | 优化后 | 变化 |
| --- | ---: | ---: | ---: |
| decode tg128，空上下文 | 45.0 | 79.4 | +76.6% |
| decode tg128，8K 上下文 | 43.1 | 74.2 | +72.3% |
| decode tg128，32K 上下文 | 40.0 | 63.6 | +59.1% |
| prefill pp8192，ubatch=512 | 865.4 | 1302.5 | +50.5% |
| prefill pp8192，ubatch=4096 | 1188.8 | 1949.9 | +64.0% |
| prefill pp512，ubatch=4096（补充测量） | 881.0 | 1328.2 | +50.8% |
| prefill pp2048，ubatch=4096（补充测量） | 1073.1 | 1879.9 | +75.2% |

- 短上下文 decode：原始版 45.0 → 复用项目现有优化、新内核关闭 55.2 → 开启新内核 79.4。
- prefill 收益来自已有 GDN、FA、GEMM 等优化，以及较大的 ubatch；新单 token GEMV 不负责大批量 prefill。最初 ubatch=512 基线与优化版 ubatch=4096 的差距同时包含代码和配置收益。
- llama-server，np=1、context=8192，三个固定提示、贪心各生成 128 token：现有优化版 55.47 → 本次 78.70。三条文本和 token 序列一致；缓存续写与不使用缓存的续写一致。
- 短上下文 benchmark 显存峰值：新内核关闭 7.96 GiB，开启 14.97 GiB；长上下文和服务还需额外工作空间。
- MTP 未启用、未测量。本轮尚未达到此前估算的硬件上限。

## 实现

复用 Qwen 的 DeltaNet、注意力、激活缓存和 prefill 优化；借鉴 IQ3/IQ4 的无损重排缓存，为 Q2_0 增加同尺寸 SoA 副本，原权重保留供多列计算和 prefill 使用。

`ggml/src/ggml-cuda/mmvq-q2-0.cu` 用 dp4a 做整数点积，两个输出行共用激活读取，支持 SwiGLU 和残差融合。保持原 Q2_0 GEMV 的 128 线程累加及两 wave 归约顺序。

触发条件：CDNA2 wave64；连续权重矩阵；4096≤K≤32768、K%256=0；128≤N≤1048576、N%8=0；单列连续 F32 输出。其他硬件、形状、融合方式、权重视图、缓存分配不足或首次 graph capture 无缓存时回退原路径。缓存由既有 buffer 管理并随 buffer 释放。

关闭开关：`GGML_MMVQ_Q2_0=0`。初测 1/2/4/8 行配置约为 76.4/82.1/76.5/67.4 token/s；最终只保留 2 行实现，不保留调参环境变量。

rocprof 追踪 32 次生成及 warm-up：Q2_0 GEMV 累计时间 468.3→282.8 ms，11121 次调用均被专用路径替换。Profiler 有额外开销，速度结论取自不带 profiler 的运行。

## 正确性与回归

- 22 个权重算子案例，优化开/关各运行一次，均通过 CPU 参考对比。覆盖真实 K、SwiGLU/残差、原地操作、短 K、奇数行和多列回退。
- 连续 512 token 的全部原始 float32 logits，开关新内核逐字节相同；各 508,559,360 字节，SHA-256 `c078223d85c0b6ab81ed46add482bd2a64171fe2b2a535151615194f7159c4b8`。
- decode KLD（8×512 token、ubatch=1）：新内核和原 main 相对保留版均为 0.000115±0.000006；保留版关闭融合的噪声测量为 0.000104±0.000005。现有图融合包含浮点重排，新内核没有新增差异。
- prefill（2×4096 token、ubatch=4096）：优化版和基线噪声检查平均 KLD 均在工具的六位小数输出中显示 0.000000；Same-top 分别为 99.951% 和 99.976%。这不代表 prefill 原始浮点输出逐位相同。
- Qwen IQ3_S 回归：修改前 80.55、修改后 80.87 token/s；128 token 原始 logits 逐字节一致，SHA-256 `a16e39a8386a261a3ee094c0ae26d0009c9f5aadf84e33233d5803dd53ac16ae`。
- Inkling 的 Unsloth checkout/binary 与 vLLM 后端均未修改；未重跑这些后端。
- backend、CLI、bench、server、算子测试目标构建及补丁反向检查均通过。Web UI 资源下载超时，构建沿用未嵌入 UI 的状态；HTTP API 测试已通过。
- 编译和执行验证限于 MI210/HIP，未在 CUDA 或其他 AMD 架构实测。

## 测量方法

全层 GPU offload、FA on、8 CPU threads、F16 KV、不使用 MTP。各测试串行使用 GPU；开始温度低于 60°C、85°C 停止，单次超时 840 秒。

短上下文按 base→on→off、off→on→base、base→on→off 三轮，每进程计时 3 次。表格取进程均值的中位数；全部原始样本保留，包括一次基线约 29.8 token/s 的异常慢样本，不静默删除。

8K/32K decode 每版本各一进程、3 次计时，重复程度低于短上下文。8K prefill 每配置各 4 个独立进程，A/B、B/A、A/B、B/A 交替；每进程只测一个 prompt 长度。首轮计时 3 次，其余 1 次，聚合时每进程等权。pp512/pp2048 每版本各一进程、3 次计时，仅作补充参考。

## 复现

在 `mi210-opt` 容器内执行：

```bash
cd ${LLAMA_CPP}
cmake --build build --target ggml-hip llama-cli llama-bench llama-server test-backend-ops -j 4
M="$MODEL"

# decode：-d 可改为 8192 或 32768
./build/bin/llama-bench -m "$M" -ngl 99 -fa on -t 8 -b 2048 -ub 512 -ctk f16 -ctv f16 -p 0 -n 128 -d 0 -r 3

# prefill：每个进程只测一个 prompt 长度
./build/bin/llama-bench -m "$M" -ngl 99 -fa on -t 8 -b 4096 -ub 4096 -ctk f16 -ctv f16 -p 8192 -n 0 -r 1

# 关闭新内核做 A/B
GGML_MMVQ_Q2_0=0 ./build/bin/llama-bench -m "$M" -ngl 99 -fa on -t 8 -b 2048 -ub 512 -p 0 -n 128 -r 3

# 交互运行
./build/bin/llama-cli -m "$M" -ngl 99 -fa on -t 8 -b 4096 -ub 4096 -c 8192

# 算子测试
./build/bin/test-backend-ops test -b ROCm0 -o 'GEMM_FUSION|MIX_PAIR' -p '(type_a|type_1)=q2_0' -j 1
```

原始 float32 等价检查工具：`mi210-tools/eval/bonsai/logits.cpp`。单独编译到 build/bin 外时，设 `BONSAI_BACKEND_DIR=${LLAMA_CPP}/build/bin`；比较原 main 时，`LD_LIBRARY_PATH` 和 `BONSAI_BACKEND_DIR` 都指向保存的 main-before 目录。
