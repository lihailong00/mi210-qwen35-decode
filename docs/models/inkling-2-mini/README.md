# Inkling-2-Mini

第二个模型，目前只有**基线测量**，还没有做优化。

- **权重**：Unsloth 的 `Inkling-2-Mini-UD-Q8_K_XL.gguf`（ModelScope `unsloth/Inkling-2-Mini-GGUF` 根目录，Unsloth 循环层布局，22,867,451,616 字节，SHA-256 以 dfb04bf3 开头）。专家权重 NVFP4 共 11.47 GB，其余（注意力、最后 4 层的专家、词嵌入、LM head）为 BF16 共 11.35 GB。
- **投机解码 drafter**：`dspark/dspark-Inkling-2-Mini-BF16.gguf`（架构 dflash，DSpark = DFlash 块 drafter + Markov 头，3 层，块大小 8，647,792,800 字节，SHA-256 以 b8985695 开头）。
- 服务器上的位置：`/data/model/Inkling-2-Mini-GGUF/`（带 `SHA256SUMS` 和 `download.sh`）。

## 代码来源

本仓库的源码树（b12110）**没有 `inkling2` 架构**，跑不了这个模型。基线用的是 Unsloth 的 llama.cpp 合并版：

- release `b12588-mix-c0c3de8` 的源码包 `llama.cpp-source-b12588-mix-c0c3de8.tar.gz`（SHA-256 92ee8439…7607，源提交 883286076efe）。它是上游 b12588 加上 Inkling 1/2、DSpark（ggml-org PR 32218）和若干修复的合并结果。
- 注意：git 标签 `b12588-mix-c0c3de8` 本身指向的是 Unsloth fork 的 master，里面**没有** inkling2 代码；必须用 release 里的源码包。
- 服务器上解压在 `/data/tools/mi210-opt/unsloth-llama`（容器内 `/lab/unsloth-llama`），已建 git 仓库，导入提交带标签 `upstream-b12588-mix`。编译参数与仓库根目录 README 相同（gfx90a、HIP graph、MMQ MFMA、FA），未改任何代码。

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

## 下一步方向

1. 剖析单 token decode（33 token/s ≈ 30 ms/token，而读权重只需约 4–5 ms），先找出与带宽无关的大头：48 次层计算的 kernel 启动数、MoE 路由与 `mul_mat_id`、相对位置偏置注意力、短卷积、循环门等小算子。
2. NVFP4 / BF16 的 decode GEMV 与 MoE 专家 GEMV。
3. 多 token 验证时的专家去重与分组，决定 DSpark 能否变成正收益。
4. 先把 inkling2 架构移植进本仓库的源码树，或者把本仓库的通用优化移植到 Unsloth 合并版上；二选一需要先评估改动量。
