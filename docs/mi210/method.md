# 优化方法、验收标准与新模型上手清单

本文是从 Qwen3.8-27B 的几轮优化里总结出来、与模型无关的做法。硬件事实见 [hardware.md](hardware.md)，各项已有优化的适用范围见 [../kernels/README.md](../kernels/README.md)。

## 1. 总体流程

1. **先算上限，再定目标。** 每 token 读多少字节（decode）、算多少 FLOP（prefill），按实测峰值换成 token/s；再按"逐节点下限"（每个节点 1.6 µs + 字节 / 1.46 TB/s）算出现实天花板。目标给一个区间，不要承诺一个点。
2. **测时间分布，定优先级。** 不只看 GEMV：Q27 的基线有约 1460 个小 kernel 和 513 次激活量化，占了四分之一时间。
3. **先校准正确性标尺**（第 3 节），再开始改。
4. **按模块并行开发**，每条线一个分支和独立编译目录，GPU 用锁共享；统一入口（launch 返回 false 即回退原路径）。
5. **频繁集成**，每次集成跑速度 A/B 和 KLD，重要节点打标签。
6. **对抗式审查**：性能代码请人专门去推翻它，并在压力条件下测（第 4 节）。
7. **看清收益递减就换打法或停手。** 用逐节点数据估算每个方向的剩余收益，合计到不了目标就停（Q27 的 decode 就是这样停在 80 token/s，转向 MTP）。

## 2. 测量

- **速度 A/B 必须交替进行**：同一二进制用环境变量开关 A/B，或新旧二进制交替（A→B→B→A），每臂至少 3 轮。共享 GPU 上单次噪声约 ±0.5–1%，比后期很多优化的收益还大；llama-server 端到端的噪声更大（同一构建 72–78 token/s 都见过）。
- **prefill 要用单一尺寸的进程**：每个进程只跑一种 n，组间轮流，每组 ≥ 4 个进程；GEMM 选核在不同进程间可能呈双峰。
- **模型内的真实节点耗时**用 graph 内 GPU 时间戳，不用 rocprofv3 的小 kernel 数字（[hardware.md](hardware.md) 第 10 节）。
- **微基准只用来筛选**，结论一律在模型内复核（冷指令缓存、脏 L2、调度偏向都只在模型里出现）。
- **投机解码的速度**：一个 SSE chunk 带一整轮 token，按 chunk 间隔算 TPOT 会把 MTP 低估 3–4 倍；用 `usage.completion_tokens` 除以 decode 时间。
- **报告要写清楚**：提示集、贪心与否、算术平均还是按 token 加权、每个配置几轮。

## 3. 正确性标尺

**KLD 对照噪声底线**（非位精确的改动，例如 GEMV 求和顺序变化）：

1. 用基线二进制 `llama-perplexity --kl-divergence-base` 保存参考 logits。decode 路径用 `-c 512 --chunks 8 -ub 1`，使每个 token 都走单 token 路径；prefill 路径用大 ubatch。
2. 测噪声底线：基线本身只改变计算顺序（例如 `GGML_CUDA_DISABLE_FUSION=1`）与参考的 KLD。Q27 为平均 KLD 0.00114、top-1 98.82%。
3. 候选版本用 `--kl-divergence` 对比。验收线定在噪声底线附近：Q27 用的是平均 KLD ≤ 0.0015、top-1 ≥ 98.3%、困惑度偏差 ≤ 0.5%。

**位精确的改动**（融合、代码缩减、多列 kernel）要与上一版逐位相同：

- 固定 GEMM 选核（`GGML_CUDA_GEMM_TUNE=0`）后，logits 是确定的，可以直接比 `--kl-divergence-base` 输出文件的 md5。
- 多列 / 多 token kernel 要求每列与单 token 路径逐位相同；各模块都有 `*_VERIFY` 调试开关做逐调用比较。
- 投机解码的输出必须与不开投机逐字节相同（贪心）。
- q8_1 激活必须与 ggml 的 `quantize_q8_1` 位精确：d = amax/127，q = roundf(x/d)，ds = (d, d·sum)。

**测试**：每个新 kernel 在 `tests/test-backend-ops.cpp` 里加用例（包括对抗性输入：NaN、±inf、边界形状、非连续布局），融合要用计数器（`GGML_CUDA_FUSION_STATS=1`）确认用例真的走到了融合路径。

## 4. 对抗式审查与压力测试

- 每个问题由独立的反驳者尝试推翻，确认后才修；修复后由另一人重跑所有探针。
- 在压力条件下测：后台 GPU 负载、限制 CU 数、`--no-warmup`、交替加载两个模型、长时间 soak。Q27 的多 workgroup 原地 rms_norm 在空闲 GPU 上从不出错，GPU 被占用时 50/50 次算错。
- 前提条件要全部前置检查，不满足就回退原图，不能在构造时断言中止（其他量化配方、其他模型会触发）。

## 5. 工程约定

- **每项优化都有环境变量开关**，默认开；关掉后与之前逐位相同。开关列表维护在模型文档附录和 [../kernels/README.md](../kernels/README.md)。
- 新增 `.cu` 文件后必须重新运行 cmake（HIP 源文件在 configure 时通配收集），否则 libggml-hip.so 加载失败、悄悄退回 CPU（检查显存峰值）。
- 只提交 LF 行尾；合并用 `-X ignore-space-at-eol`。
- 公共入口（`ggml-cuda.cu` 的 `ggml_cuda_try_fuse` 各融合入口、`q8-producers.cu`）是集成冲突的高发点，两边都要保留、手工核对顺序。
- 并行的 agent 各用自己的子目录，否则同名文件互相覆盖。

## 6. 共享 GPU

- 所有 GPU 作业经过 `mi210-tools/gpu-run.sh`：flock 串行化（锁文件 `/data/tools/mi210-opt/gpu.lock`），等降到 60°C 再开始，85°C 中止。docker 起的 vLLM 服务也要在 `flock` 下运行。
- 每次占锁不超过约 15 分钟；`gpu-run.sh` 超时只杀锁的持有者、不杀容器里的作业，要自己清理残留进程。
- 服务器上还有别的业务：不要按镜像名批量 `docker stop`，只按确切的容器名操作。

## 7. 新模型上手清单

新模型的文档放在 `docs/models/<模型id>/`，结构见 [../models/README.md](../models/README.md)。

1. **清点结构**：层类型和数量、各维度（隐藏维、FFN、头数、头维、GQA 比）、每个权重的量化类型和字节数、词表大小、是否有 MTP / 草稿头。
2. **算上限**：decode 每 token 字节数（权重 + 状态 + 每个上下文位置的 KV），prefill 每 token FLOP；换成物理上限和逐节点下限。
3. **测基线**：上游原版（`baseline` 分支）和本仓库 `main` 各测一遍 decode / prefill。`main` 上的优化不一定会触发，用 `GGML_CUDA_FUSION_STATS=1` 和 kernel trace（按 kernel 名统计）确认哪些路径生效、哪些回退。
4. **对照适用范围**：按 [../kernels/README.md](../kernels/README.md) 的"适用条件"一栏逐项核对。标为"形状特化"或"模型特定"的项通常不会触发，是第一批要做通用化的地方。
5. **建正确性基线**：保存参考 logits、测噪声底线、存几条贪心输出的 md5。
6. **按时间分布开始优化**，流程同第 1 节。
7. **回归**：任何改动公共代码的提交，都要在已登记的所有模型上复测速度和 KLD / md5，结果写进各自的模型文档。
