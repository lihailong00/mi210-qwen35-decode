# Inkling-2-Mini

第二个模型。基线 decode 33.3 token/s；第一阶段（小算子融合，logits 与基线逐位相同）后 **44.8** token/s。

- **权重**：Unsloth 的 `Inkling-2-Mini-UD-Q8_K_XL.gguf`（ModelScope `unsloth/Inkling-2-Mini-GGUF` 根目录，Unsloth 循环层布局，22,867,451,616 字节，SHA-256 以 dfb04bf3 开头）。专家权重 NVFP4 共 11.47 GB，其余（注意力、最后 4 层的专家、词嵌入、LM head）为 BF16 共 11.35 GB。
- **投机解码 drafter**：`dspark/dspark-Inkling-2-Mini-BF16.gguf`（架构 dflash，DSpark = DFlash 块 drafter + Markov 头，3 层，块大小 8，647,792,800 字节，SHA-256 以 b8985695 开头）。
- 服务器上的位置：`/data/model/Inkling-2-Mini-GGUF/`（带 `SHA256SUMS` 和 `download.sh`）。

## 代码来源

本仓库的源码树（b12110）**没有 `inkling2` 架构**，跑不了这个模型。基线用的是 Unsloth 的 llama.cpp 合并版：

- release `b12588-mix-c0c3de8` 的源码包 `llama.cpp-source-b12588-mix-c0c3de8.tar.gz`（SHA-256 92ee8439…7607，源提交 883286076efe）。它是上游 b12588 加上 Inkling 1/2、DSpark（ggml-org PR 32218）和若干修复的合并结果。
- 注意：git 标签 `b12588-mix-c0c3de8` 本身指向的是 Unsloth fork 的 master，里面**没有** inkling2 代码；必须用 release 里的源码包。
- 服务器上解压在 `/data/tools/mi210-opt/unsloth-llama`（容器内 `/lab/unsloth-llama`），已建 git 仓库，导入提交带标签 `upstream-b12588-mix`。编译参数与仓库根目录 README 相同（gfx90a、HIP graph、MMQ MFMA、FA）。基线测量时未改任何代码；优化在 `inkling-opt` 分支上，见下文"decode 融合"。

上游主线的支持需要 `Mainline_Support/` 下的另一种布局和 llama.cpp PR 32704，且不支持 DSpark，这里没有用。

## 结构要点

- 架构 `inkling2`：循环 MoE，总参数 26.1B，每 token 激活约 3.4B。
- 24 层权重；第 6–17 层（12 层）每个 token 跑 3 遍，所以 `block_count` = 48 次层计算。循环部分带 forget/gain 门（秩 64）。
- 隐藏维 2304；128 个专家，每 token 选 8 个，专家 FFN 1152（gate/up 合并存放）；没有共享专家。
- 注意力：24 个 q 头、8 个 kv 头（GQA 3），头维 128，带输出门；48 次层计算里每 6 次有 5 次是滑动窗口 1024，1 次是全注意力；没有 RoPE，用相对位置偏置（relative_length 16，extent 1024）、qk clip、attention sink、k/v 短卷积（核 4）。
- 词表 201024，最终 logit softcap 30。

和 Qwen3.8-27B 的已有优化对照：头维 256 的 FA decode、GQA 6 分发、5120 列的定形 norm、DeltaNet 融合都不会触发；按量化类型写的 GEMV 也不覆盖 NVFP4 和 BF16。基本是从零开始。

## 上限

decode 每 token 读取（上下文 0，按每层选 8/128 个专家、循环层算 3 遍）：

| 项目 | 大小 |
| --- | --- |
| 48 次层计算的激活权重 | 4.91 GB（普通层约 91 MB，循环层约 96 MB，最后 4 层专家是 BF16，约 183 MB） |
| LM head（BF16） | 0.93 GB |
| 合计 | 5.83 GB |
| 物理上限（1.2–1.4 TB/s） | **约 205–240 token/s** |

计算脚本：`mi210-tools/eval/inkling/active_bytes.py`。

## 基线结果

单卡 MI210，Unsloth 合并版原样编译，`-ngl 999 -fa on`。

**llama-bench**（`-b 2048 -ub 2048 -r 2`）

| 测试 | token/s |
| --- | --- |
| decode tg128，上下文 0 | 33.3 |
| decode tg128，上下文 8192 | 31.3 |
| prefill pp512 / pp2048 | 552 / 513 |
| prefill pp512 / pp2048，上下文 8192 | 307 / 363 |

decode 只到上限的约 15%。

**llama-server，DSpark 投机解码**：6 个对话提示（代码、算法讲解、算术、短篇故事、中文技术解释、bash 脚本），每个生成 256 token，`-c 16384 -np 1 --jinja`。贪心为 temp 0；采样用官方推荐参数 temp 1.0 / top-p 1.0 / top-k 0 / min-p 0，seed 1。脚本：`mi210-tools/eval/inkling/spec_bench.py`。

| 配置 | 贪心 token/s | 接受率 | 采样 token/s | 接受率 |
| --- | --- | --- | --- | --- |
| 不开投机 | 33.1 | — | 32.3 | — |
| DSpark，n_max 1 | 33.8 | 0.73 | 31.3 | 0.65 |
| DSpark，n_max 2 | **36.8**（+11%） | 0.56 | **33.3**（+3%） | 0.49 |
| DSpark，n_max 3（模型卡推荐值） | 30.4（−8%） | 0.46 | 27.9（−14%） | 0.41 |

- 在 MI210 上 DSpark 几乎没有收益：n_max 3 时每步平均接受约 2.3 个 token，但一步验证 + 起草的耗时约为单 token decode 的 2.6 倍。原因很可能是 MoE：4 个 token 在每层最多会选到 32 个不同的专家，验证时读的专家权重接近单 token 的 4 倍，而不像稠密模型那样几乎不变。
- **开 DSpark 后贪心输出与不开不同**：6 个提示里 0–1 个相同，在第 100–600 个字符处开始分叉。Unsloth 的文档也说明不保证相同。这和 Qwen3.8-27B 的 MTP（逐字节相同）标准不一样，优化时要先弄清分叉来自批量验证的数值差（不同 kernel、不同归约顺序）还是实现问题。

## decode 剖析（基线）

测量：rocprofv3 kernel trace，`llama-bench -p 0 -n 48`，取最后 20 个 token（以 LM head 为分界）。rocprofv3 会把小 kernel 拉长到约 4.4 µs、并让 graph 回放变成 CPU 受限，所以下面小 kernel 的时间偏大，大 kernel 的时间基本可信。真实 decode 是 33.1 token/s（30.2 ms/token）。

- **每 token 约 7240 次 kernel 启动**，平均每次层计算约 150 次（Qwen3.8-27B 优化后整个 token 只有 601 次）。graph 节点 12658 个。
- **HIP graph 是开着的**，每个 token 复用同一个 graph；关掉后掉到 27.4 token/s。注意：`GGML_CUDA_DISABLE_GRAPHS` 只看是否设置，设成 0 也会关掉 graph（`ggml-cuda/graph.cuh:66`），做 A/B 时不能用 `=0` 当"开"。
- 每次层计算的小算子（次数/层计算）：二元运算 add/mul/sub/div 31，scale 15，一元运算 12，concat 12，各种 copy 约 11，rms_norm 9，以及 sum_rows、fill、clamp、pad、get_rows、set_rows、memcpy 等各 1–4 次。来源（`src/models/inkling2.cpp`）：
  - `clip()`：Q、K、相对位置向量 R 各一次，每次 9 个 kernel（mul、sqr、sum_rows、sqrt、scale_bias、fill、div、clamp、mul），共 27 个。
  - `control()`：循环第 2、3 遍时 q/k/v/mlp 各一次，每次 concat + rms_norm + 3 个秩 64 的小 GEMV + 2 个 tanh + scale_bias + 2 个 mul + add。
  - `mix()`（按深度的注意力残差）：每层 2 次，循环第 2、3 遍再加 2 次；每次对已有的每个 block 各做一次 concat，再 rms_norm、repeat、两个小 mul_mat、softmax、转置 copy。
  - `convolve()`：k、v、注意力输出、FFN 输出各一次短卷积，每次 transpose/concat/状态写回/ssm_conv/add/scale；全注意力层的 k、v 还有 4 个动态 tap，每个 cont + mul + add。
  - 注意力的 dense 路径（decode 时 n_tokens ≤ 32 都走这里）：每次都对相对位置偏置权重做一次 pad，再 mul_mat、sub、permute copy、fill、get_rows、permute copy，再进 FA。
- 大 kernel 的带宽（读取量 / 实测时间）：

| kernel | 每 token | 带宽 | 说明 |
| --- | --- | --- | --- |
| NVFP4 专家 GEMV（`mul_mat_vec_q`，88 次） | 3.65 ms | 约 0.43 TB/s | 每次 8 个专家；上限约 1.3 TB/s |
| BF16 q / gate（3072×2304，96 次） | 1.57 ms | 0.87 TB/s | |
| BF16 o（2304×3072，48 次） | 0.84 ms | 0.81 TB/s | |
| BF16 k / v（1024×2304，96 次） | 0.80 ms | 0.57 TB/s | |
| BF16 rel（384×2304）、sink（24×2304） | 0.66 ms | 0.24 TB/s 以下 | 纯延迟 |
| BF16 专家（第 20–23 层，8 次） | 0.50 ms | 0.68 TB/s | |
| 循环控制的秩 64 GEMV（64×4608 等，约 290 次） | 约 1.9 ms | — | 纯延迟 |
| LM head（BF16） | 0.75 ms | 1.24 TB/s | 正常 |
| flash attention（vec + combine，上下文接近 0） | 1.1 ms | — | 18.6 µs/次，按 1536/2048 的 KV 长度带 mask 扫描 |
| `topk_moe` | 0.66 ms | — | 13.6 µs/次 |
| mix 的 softmax | 1.0 ms | — | |

- 粗略时间分配（30.2 ms）：大 GEMV 约 11 ms（按带宽算本该约 4.5 ms），约 6000 个小 kernel 约 12–15 ms，注意力、路由和其余约 4 ms。

**优化空间估计**（尚未实现，数字是估算）：

| 方向 | 预计节省 |
| --- | --- |
| 小算子融合：clip、control、mix、convolve、注意力的偏置准备各写成一个 kernel，或并进相邻 GEMV 的前后处理；目标每次层计算 25–35 个 kernel | 8–12 ms |
| NVFP4 专家 GEMV 提到约 1.1 TB/s（参照 Qwen 线上 IQ/Q4_K GEMV 的做法） | 约 2 ms |
| BF16 GEMV 提到约 1.2 TB/s；同一输入的 q、gate、rel、sink 合成一次 GEMV | 约 2 ms |
| 小 GEMV、topk、FA 的延迟 | 约 2 ms |

合计约 14–18 ms，即 decode 约 60–80 token/s（现 33）。带宽上限 205–240 token/s 离得还远，主要因为 48 次层计算本身就带来大量必须的 GEMV 启动。多 token 验证的效率提高后，DSpark 才可能有明显收益。

## decode 融合（第一阶段）

代码在 Unsloth 合并版的 `inkling-opt` 分支（服务器 `/data/tools/mi210-opt/unsloth-llama`，提交 8e9a5f5、5c7417f），补丁导出在 [patches/](patches/)（对 `upstream-b12588-mix` 用 `git am` 应用）。新增文件 `ggml/src/ggml-cuda/inkling-fuse.cu/.cuh`，`ggml-cuda.cu` 只加了几处挂钩。

做法：在 HIP 后端里识别 `src/models/inkling2.cpp` 生成的节点序列，用一个 kernel 代替。模型代码和 CPU 路径不变。每个融合 kernel 都照原来各个 kernel 的算术顺序写（归约树、block 大小、FMA 合并与否），所以 **decode 的 logits 与基线逐位相同**。

| 融合 | 原来的节点 | 现在 | 每次层计算省下的 kernel |
| --- | --- | --- | --- |
| clip（Q、K、R 的 L2 裁剪） | [mul] sqr sum_rows sqrt scale fill div clamp mul | 1 | 约 23 |
| SITU 激活 | scale tanh scale sigmoid mul scale tanh scale mul | 1 | 8 |
| MoE down 尾部 | repeat get_rows + 加权求和 | 1 | 约 2 |
| 循环控制门 control | concat rms_norm ；forget/gain GEMV tanh scale mul tanh mul add | 2（down GEMV 照旧） | 8 × 4（只在循环第 2、3 遍） |
| mix（按深度的注意力） | concat 链 rms_norm 转置拷贝 repeat 打分 GEMV softmax | 1（最后的加权 GEMV 照旧） | 约 6 × 每层 2–4 次 |
| 短卷积 convolve | concat 状态写回 ssm_conv add scale | 1（FFN 那次是 2） | 约 15 |

- 每 token kernel 数：7243 → 3497。
- 没融合的：全注意力层 k/v 的动态卷积（带 4 个动态 tap）、3 个 query 的 mix（`mul_mat_vec_f<.., 3, ..>` 还没能逐位复现）、注意力的相对位置偏置准备、多处 view 的 cont 拷贝、残差 add。
- 开关（设 0 关闭）：`GGML_CUDA_INK_FUSE`（全部）、`GGML_CUDA_INK_CLIP`、`_SITU`、`_MOE_DOWN`、`_CTRL_NORM`、`_CTRL_GATE`、`_MIX`、`_SCONV`。`GGML_CUDA_FUSION_STATS=1` 打印命中次数，`GGML_CUDA_DUMP_NODES=<文件>` 导出 graph 节点，`GGML_CUDA_INK_VERIFY=1`（配合 `GGML_CUDA_DISABLE_GRAPHS=1`）让融合 kernel 把中间结果写到暂存区、原节点照常运行，再逐位比较，用来定位不一致。

**结果**（单卡 MI210，Q8_K_XL）

| 测试 | 基线 | 融合后 |
| --- | --- | --- |
| llama-bench tg128 | 33.3 token/s | **44.8** token/s（+34%） |
| llama-server 贪心（6 个提示 × 256 token） | 33.1 | **44.4** |
| llama-server 采样（temp 1） | 32.3 | **42.8** |
| llama-server + DSpark n_max 2，贪心 | 36.8 | 38.3（验证 graph 是多 token 的，没有融合，现在比不开还慢） |

正确性：`kld.sh`（`llama-perplexity -ub 1`，每步都是单 token decode graph，3 个 512 token 的块）保存的全部 logits 的 md5 与基线相同（`01ec923fac19`）；llama-server 贪心输出 6/6 与基线逐字相同，开 DSpark 时也是 6/6、接受数相同。

**做的过程中确认的事实**（gfx90a / ROCm 7.2.1）：

- HIP 的 `__fmul_rn` / `__fadd_rn` 挡不住编译器把乘加合并成 FMA；要和分开的 graph 节点逐位一致，得在相应代码块里写 `#pragma clang fp contract(off)`。
- 设备代码里 `WARP_SIZE` 是 32，物理 wave 是 64：`block_reduce` 用前者，`mul_mat_vec_f` 用后者，复现时两种都要照搬。
- 把后面节点的结果提前算（或把输入推后读）时，ggml-alloc 会复用中间缓冲区，必须逐个检查中间节点的读写范围；view 节点不读写内存，0 行的空张量的 `ggml_nbytes` 不可信，两者都要排除。
- `GGML_CUDA_DISABLE_GRAPHS` 只看是否设置，设成 0 也会关 graph。

## 下一步方向

1. 剩下的大头是两类 GEMV：BF16 投影（q/gate/o/k/v/rel/sink，约 0.6–0.9 TB/s）和 NVFP4 专家（约 0.43 TB/s）。把它们提到 1.1–1.3 TB/s 预计每 token 省 4–5 ms；GEMV 改写会改变归约顺序，正确性按 KLD 噪声底线验收（见 [method.md](../../mi210/method.md)）。
2. 继续融合：注意力的相对位置偏置准备（pad 是常量权重，可提到加载时）、mix 输出的 view/cont 拷贝、残差 add，3 个 query 的 mix，动态卷积。
3. 多 token 验证（DSpark）的 graph 也要融合，并做专家去重，DSpark 才可能有收益。
4. 是否把 inkling2 架构移植进本仓库的源码树：目前优化都在 Unsloth 合并版上，以补丁形式保存在 [patches/](patches/)。
