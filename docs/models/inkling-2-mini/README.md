# Inkling-2-Mini

第二个模型。基线 decode 33.3 token/s；第一阶段（小算子融合，logits 与基线逐位相同）后 44.8 token/s；第二阶段（GEMV）后 50.7 token/s；第三阶段（剩余小算子、注意力偏置、decode 注意力、MoE 路由）后 83.6 token/s；第四阶段（NVFP4 GEMV 重写，逐位相同）后 88.6 token/s；第五阶段（再减一轮 launch、主机侧开销，逐位相同）后 **109.8** token/s，平均 KLD 0.00099。prefill（pp2048）从 519 提到 **6280** token/s。

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

## decode GEMV（第二阶段）

代码在同一分支（提交 b888be8，补丁 0003），新文件 `ggml/src/ggml-cuda/inkling-gemv.cu`。只处理单 token。

- **NVFP4 MoE 专家**（`MUL_MAT_ID`）：原来 CDNA 上 `mul_mat_vec_q` 每行一个 block（2 个 wave），一行只有 36 个 64 值块，线程大多空闲，还要调度 18432 个小 block。现在 8 条 lane 一行、256 线程一个 block；q8_1 激活放进 LDS；int8 点积与原来相同；UE4M3 尺度原来在 gfx90a 上走 `ldexpf` + 除法的慢路径，换成按位构造（结果完全相同）；每专家 scale 的 RESHAPE/REPEAT/GET_ROWS/MUL 一并融合。最好情况约 1.3 TB/s。
- **BF16 投影**（`MUL_MAT`）：16 字节读取（8 个 bf16），按行数选 32 / 64 / 256 条 lane 一行（256 即一行一个 block）。共用同一激活向量的几个 GEMV（如 q、rel、sink、gate，非循环层还有 k、v）合成一次 launch，后面的提前计算，同样做缓冲区复用检查。q/gate/o 约 1.2 TB/s，LM head 约 1.4 TB/s。
- **mix** 改为单个 1024 线程 block：所有状态一次读入，rms 的两级归约树与原来相同（仍逐位相同），归一化结果放 LDS，转置输出按连续段写；状态数 S ≤ 5 用模板实例化，避免寄存器数组溢出到 scratch。
- 开关：`GGML_CUDA_INK_NVFP4=0`、`GGML_CUDA_INK_BF16=0`、`GGML_CUDA_INK_BF16_MERGE=0`。

| 测试 | 第一阶段 | 第二阶段 |
| --- | --- | --- |
| llama-bench tg128 | 44.8 token/s | **50.7** token/s |
| llama-server 贪心 | 44.4 | **50.2** |
| llama-server 采样 | 42.8 | **48.1** |

正确性：两个 GEMV 只改变 fp32 部分和的相加顺序（权重、int8 激活量化、乘积都相同），所以这里的 KLD 就是本模型的求和顺序噪声底线：与基线 logits 的平均 KLD 0.00079，top-1 相同 99.87%（Qwen3.8-27B 的同类底线是 0.00114）。关掉这两个开关时 logits 仍与基线逐位相同。贪心输出在第 130–600 个字符后开始与基线不同。

经验：

- gfx90a 没有 FP8 硬件，`ggml_cuda_ue4m3_to_fp32` 走带分支、`ldexpf` 和除法的回退路径，每 16 个值一次，在 GEMV 里是明显的 ALU 开销。
- 单 block 内核的寄存器数组只要有运行时下标就会溢出到 scratch（rocprofv3 的 `Scratch_Size` 列可以看到），改成模板参数后消失。
- 从一个 CU 发出跨步 4 字节的写（每次写都落在不同 cache line）很慢；让每个线程写一段连续区间。
- rocprofv3 下单个 kernel 的耗时会被放大、方差很大，取舍以 llama-bench 交替 A/B 为准（`ksweep.sh` 只用来看相对差别）。

## decode 第三阶段（小算子、注意力、路由）

代码在分支 `inkling-opt`（合并提交 39de315，补丁 0004 是相对 b888be8 的整体 diff）。四项分别在独立分支上实现并各自经过一轮独立审查，然后合并。全部只处理单 token、单序列，其余情况回退到原路径。

| 项 | 新文件 | 内容 | 单独增益（从 50.6 起） | 正确性 |
| --- | --- | --- | --- | --- |
| 剩余小算子 | `inkling-ops.cu` | mix 连同最后的加权和与 CONT 拷贝一个 kernel；convolve 连同残差 ADD、K/V cache 写入、动态 tap 一个 kernel；精简的 mix / RMS_NORM+MUL / control norm / clip kernel；q8_1 激活由上游 kernel 直接写出 | → 65.2 | 逐位相同 |
| 注意力偏置 | `inkling-attn.cu` | PAD + 表 GEMM + SUB + GET_ROWS + mask 相加 + FILL 合成一个 kernel，只算实际用到的 n_kv 个位置 | → 55.5 | 逐位相同 |
| decode 注意力 | `inkling-fattn.cu` | 单 query 专用 kernel（每个 head 一个 block，按 mask 跳过尾部），n_kv > 256 时分块再合并 | → 53.1 | KLD |
| MoE 路由 | `inkling-moe.cu` | SIGMOID 到 SCALE 的路由尾部一个 kernel；最后 4 层 BF16 专家的 GEMV | → 52.8 | 路由逐位相同，BF16 专家按 KLD |

合并后的结果（各项节省的时间相加，约 7.9 ms/token）：

| 测试 | 第二阶段 | 第三阶段 |
| --- | --- | --- |
| llama-bench tg128 | 50.7 token/s | **83.6** token/s |
| llama-server 贪心 | 50.2 | **81.4** |
| llama-server 采样 | 48.1 | **76.1** |
| kernel 数 / token | 约 3450 | 约 1785 |

正确性（`kld.sh`，3 个 chunk，对比原版 logits）：

| 配置 | 结果 |
| --- | --- |
| 默认 | 平均 KLD 0.000992，最大 0.1235，top-1 相同 99.608% |
| `GGML_CUDA_INK_FATTN=0`（tg128 78.6） | 平均 KLD 0.000786，top-1 相同 99.869%（与第二阶段相同） |
| 再加 `GGML_CUDA_INK_BF16_EXPERTS=0` | 同上，与第二阶段 logits 逐位相同 |
| `GGML_CUDA_INK_FATTN=0 GGML_CUDA_INK_NVFP4=0 GGML_CUDA_INK_BF16=0` | md5 01ec923fac19，与原版逐位相同 |

1425 token 的 prompt（超过 1024 的滑动窗口）下，逐位相同的那部分融合开和关的贪心输出一致。

主要开关：`GGML_CUDA_INK_OPS=0`（`inkling-ops.cu` 全部，另有 18 个子开关）、`GGML_CUDA_INK_ATTN_BIAS=0`、`GGML_CUDA_INK_FATTN=0`、`GGML_CUDA_INK_ROUTER=0`、`GGML_CUDA_INK_BF16_EXPERTS=0`。

注意：

- decode 注意力 kernel 的 V 求和用 fp32（原 kernel 用半精度累加），与原 kernel 的相对 rms 差约 2e-4。top-1 相同率 99.608% 离 99.5% 的线只有一个 token（3 个 chunk 中一个 token 是 0.13%），需要更严格时用 `GGML_CUDA_INK_FATTN=0`。
- 注意力偏置里的 16 项点积要与 rocBLAS（ROCm 7.2.1，Tensile）的累加顺序一致才逐位相同：从 0 开始按 k=0..15 顺序 fmaf。换 ROCm 版本后用 `GGML_CUDA_INK_VERIFY=1 GGML_CUDA_DISABLE_GRAPHS=1` 重新核对。
- `mul_mat_vec_f` 在 gfx90a 上 ncols_dst 为 1、2 时乘积先舍入再相加，为 3 时用 FMA；3 个 query 的 mix 按这个规律才能逐位相同。
- 这些 kernel 假定 64 lane 的 wave（DPP、ds_bpermute、ballot），只适用于 gfx90a 这类 CDNA 设备；`inkling-fattn.cu` 有设备检查，`inkling-ops.cu` 没有。
- `inkling-ops.cu` 的 dispatch 必须排在其他 Inkling 融合之前；K/V cache 的写入被提前到卷积 kernel 里。
- kernel 的耗时主要跟每个 wave 执行的指令数走，barrier 和额外的依赖读取几乎不花时间；精简指令路径比合并 launch 更有效（mix kernel 精简后 +5.7 token/s）。
- 没做的：conv 状态的 GET_ROWS（48 次/token）、gate 的 sigmoid×mul（48 次/token）、NVFP4 GEMV 仍只有 0.64–0.80 TB/s。

## NVFP4 GEMV 带宽（第四阶段）

代码在分支 `inkling-opt`（提交 b9fee35，补丁 0005），只改 `inkling-gemv.cu`。三个方向各自独立实现并审查（减少指令数 +3.5、调整 lane 布局 +1.4、重新设计 +5.3 token/s），三者都与原 kernel 逐位相同，合并的是最快的重新设计版 `ink_mmid_nvfp4_v4`。

做法：

- 每个线程只处理一个 36 字节的权重块，64 条 lane 全部在用；三次读取在任何准备工作之前发出。原 kernel 是 8 条 lane 一行、每条 lane 串行 3–5 轮"读取、等待、计算"。
- 4 个 E2M1 码的解码用两次 `v_perm_b32` 异或得到（利用选择子 8..15 的特殊含义），每 64 个值的指令数从 249 降到约 170；UE4M3 尺度查 LDS 里的 256 项表。
- 4 个子块乘积先写进 LDS，第二阶段再按原 kernel 的顺序（8 条链，然后 4/2/1 的树）相加。这样 lane 的分配与 fp32 求和顺序解耦，所以结果逐位相同。
- 只为 K=2304（36 块）和 K=1152（18 块）实例化；其他形状回退到旧 kernel。`GGML_CUDA_INK_NVFP4_V=1` 切回旧 kernel。

| 测试 | 第三阶段 | 第四阶段 |
| --- | --- | --- |
| llama-bench tg128 | 83.4 token/s | **88.6** token/s |
| llama-server 贪心 | 81.4 | **86.9** |
| llama-server 采样 | 76.1 | **80.7** |

| GEMV（单独循环测，含约 2 µs 的空 launch） | 旧 | 新 | 只读不算 |
| --- | --- | --- | --- |
| gate/up，23.9 MB | 30.5 µs（0.78 TB/s） | 22.3 µs（1.07 TB/s） | 20.6–20.9 µs |
| down，11.9 MB | 18.6 µs（0.64 TB/s） | 13.4 µs（0.89 TB/s） | 12.0 µs |

正确性：`kld.sh` 的 logits md5 与第三阶段相同（默认配置、`GGML_CUDA_INK_FATTN=0`、关闭 graph 三种情况），KLD 统计不变；1425 token prompt 的贪心输出相同；prompt 处理速度不变（pp512 约 658 token/s）。

经验：

- 这张卡上 kernel 的时间大致等于"读取模式的访存时间 + 一个 wave 串行执行的指令数 + 约 1 µs"。wave 越短、读取越早发出越好；down 原来慢是因为 18 块分给 8 条 lane 只用上 75% 的 lane 轮次。
- 同样字节量只读不算也要 20.6 / 12.0 µs（扣掉空 launch 约 1.27 TB/s），新 kernel 离这个上限只剩约 1.5 µs/次，NVFP4 GEMV 基本到头。
- 每条 lane 流式读连续一段（而不是跨行分布）明显更慢；从全局内存读激活或尺度表比放 LDS 慢；512/1024 线程的 block 更慢。
- 改变求和顺序的 18-lane 版本更简单，但 top-1 相同率掉到 99.608%（关掉注意力 kernel 时），没有采用。
- `v_perm_b32` 选择子的特殊含义和 `fp contract(off)` 只在 gfx90a + ROCm 7.2.1 上验证过；换工具链后用 `kld.sh` 的 md5 复核。
- `inkling-gemv.cu` 里还留着测速用的代码（`GGML_CUDA_INK_NVBENCH`）和多个试验用的实例化，编译时间变长，可以以后清理。

## 第五阶段：再减一轮 launch 与主机侧开销

代码在分支 `inkling-opt`（合并提交 4f9a18f，补丁 0006 是相对 b9fee35 的整体 diff）。四项按文件归属分给四个分支并行实现，各自经过独立审查后合并。本轮只接受逐位相同的改动：三种配置的 logits md5 都不变。

| 分支 | 内容 | launch / token | 单独增益 |
| --- | --- | --- | --- |
| `r5-ctrl` | K/V/Q 三条 control 链的同一阶段作为同一次 launch 的不同 block；Q clip + K/V conv 一次 launch；conv 状态直接读、不再 GET_ROWS；同一 layer pass 的两到三个 mix 一次 launch；CONT 拷贝并进 norm | 1785 → 1331 | +9.9 |
| `r5-attn` | sink 和 gate 投影提前到第一个 BF16 launch 里算；偏置 / mask 和 sigmoid×mul 并进注意力 kernel | 1785 → 1587 | +3.6 到 +4.9 |
| `r5-moe` | 路由 GEMV + top-k 尾部一次 launch；专家加权求和并进后面的 conv kernel | 1785 → 1689 | +2.6 |
| `r5-host` | 输入上传改为暂存后随 graph 排队；一个 ggml graph 只用一个 HIP graph；graph 变化时立即重新捕获；采样的 softmax / argmax 用辅助线程 | 不变 | +1.8（采样 +6） |

合并后的结果：

| 测试 | 第四阶段 | 第五阶段 |
| --- | --- | --- |
| llama-bench tg128 | 88.6 token/s | **109.8** token/s |
| tg64，深度 2048 / 8192 | — | 101.8 / 98.6 |
| llama-server 贪心 | 86.9 | **106.4** |
| llama-server 采样（temp 1.0） | 80.7 | **103.6** |
| kernel 数 / token | 1785 | 1037 |
| pp512 | 658 | 658 |

正确性（合并版）：`kld.sh` 默认 md5 b2b38cf906c9、`GGML_CUDA_INK_FATTN=0` 为 05c856b2ade6、再关 NVFP4 / BF16 为 01ec923fac19，与第四阶段完全相同；`GGML_CUDA_DISABLE_GRAPHS=1` 下默认 md5 也相同；1425 token prompt 的贪心输出与合并前相同。采样在固定种子下文本不变。

### 减 launch 的做法

- **多个互不依赖的小算子合成一次 launch**：用 `blockIdx` 选择算子，每个 block 跑原 kernel 的函数体，算术不变，也不会让任何一个 wave 变长。这是本轮的主要手段，每去掉一次 launch 约省 2.2–2.8 µs。
- **提前计算放进持久 scratch**，而不是提前写 graph 的缓冲区，这样不必担心 ggml-alloc 的缓冲区复用；到原节点位置时再让消费者读 scratch。
- **原版小 kernel 比一次空 launch 贵得多**：空 launch 约 2 µs，而 `moe_weighted_reduction_f32` 4.1 µs、路由 GEMV 5.1 µs，原因是运行时的循环次数让每次迭代都成为一轮"读取、等待、计算"。同样的算术改成固定循环次数、读取提前后是 2.1 / 3.1 µs。
- BF16 GEMV 的合并结果取决于 ggml-alloc 的缓冲区布局，lane 数又决定求和顺序，所以参考 md5 只对 `kld.sh` 的布局有效。

主要开关：`GGML_CUDA_INK_CTLGRP`、`_CONT_FOLD`、`_QKVR`、`_RS`、`_MIX_PAIR`、`_MIX_TRI`（control / conv / mix）；`GGML_CUDA_INK_BF16_EARLY`、`_FABIAS`、`_FAGATE`（注意力）；`GGML_CUDA_INK_ROUTER=1`、`_MOE_RED=0`、`_RED_CONV=0`（MoE）；`GGML_CUDA_INK_STAGED_INPUTS=0`、`GGML_CUDA_GRAPH_CHUNK_SIZE=512`、`GGML_CUDA_GRAPH_FAST_RECAPTURE=0`、`LLAMA_SAMPLER_THREADS=1`（主机侧）。

### 主机侧开销的测量

用 `GGML_CUDA_INK_HOSTPROF=1` / `LLAMA_INK_HOSTPROF=1` 可以打印每 token 的分解。第四阶段代码上的结果（µs / token，改前 → 改后）：

| 项 | llama-bench | 说明 |
| --- | --- | --- |
| 整个 token | 11243 → 约 10990 | |
| 等 GPU | 10993 → 约 10910 | 主线程 99% 的时间在 `hipStreamSynchronize` 里自旋 |
| 主机合计 | 250 → 76 | 原先估计的 1–2 ms 是错的 |
| 输入上传（12 次，共 12 KB） | 133 → 7.7 | 原来每次都是阻塞拷贝 |
| `hipGraphLaunch` | 51.5（25 段）→ 12.9（1 段） | 每段在回放时还要占 5–7 µs 的 GPU 时间 |
| logits 读回 | 约 40 | 本来就是异步的 |

| 项 | llama-server | 说明 |
| --- | --- | --- |
| temp 1.0 的 dist 采样 | 991 → 299 | 关掉所有截断采样器时要对 20 万个候选做 softmax |
| 贪心链 | 100 → 62 | |
| set_logits + 其他服务端工作 | 约 145 | 没有动 |

graph 每当 n_kv 的 256 对齐增长时要重新捕获一次，约 14 ms（平均 54 µs / token）；一段式 graph 让捕获的那个 token 从 14 ms 变成 21 ms。在 rocprofv3 下自动保持 512 节点分段（强制一段会慢 3 倍）。

### 注意

- 本轮的融合在 `GGML_CUDA_INK_VERIFY` 模式下会自动关闭，没有逐元素比对，只靠 logits md5 和长文本对比验证。
- 持久 scratch、计数器和宿主侧的映射表是进程全局的，适用于按顺序求值的多个 context（目标模型 + drafter），不适用于两个线程同时求值。
- 立即重新捕获要求所有融合在第一次见到新形状时就能在捕获中合法运行；出问题时用 `GGML_CUDA_GRAPH_FAST_RECAPTURE=0`。
- 采样线程池在候选数 ≥ 32768 时才启用，最多 7 个辅助线程。
- 没测：DSpark、多序列同时 decode、超过约 1750 token 以外的更长生成（只测了 tg64 深度 8192 的速度）。

## 持久化 kernel 可行性测试（结论：不值得做）

想法：整个 decode 用一个常驻 kernel 跑，层与层之间用 kernel 内同步代替 launch，并在串行小算子执行时提前读下一个 GEMV 的权重。先用微基准测这件事值不值得做。代码：[mi210-tools/eval/inkling/pk.cpp](../../../mi210-tools/eval/inkling/pk.cpp)（`hipcc -O3 --offload-arch=gfx90a`），[runs.py](../../../mi210-tools/eval/inkling/runs.py) 用 rocprofv3 trace 统计每 token 的 kernel 序列。

同步原语的开销（MI210，ROCm 7.2.1）：

| 项 | 开销 |
| --- | --- |
| HIP graph 里一次空 kernel（1 / 385 个 block） | 1.33 / 1.60 µs |
| HIP graph 里一次单 block 的依赖小算子（读全局、归约、写全局） | 3.61 µs |
| 两个 block 之间的标志传递（单向） | 0.37 µs |
| 自写的全局 barrier（104 / 385 个 block） | 1.08 / 1.31 µs |
| cooperative groups `grid.sync()`（104 / 385 个 block） | 0.41 / 1.13 µs |

一段 = S 个依赖的小算子（单 block，类似 RMS norm）+ 一个 3072×2304 的 BF16 GEMV（14.2 MB，每段换一份权重），µs / 段：

| 做法 | S=1 | S=2 | S=4 | S=8 |
| --- | --- | --- | --- | --- |
| 只有 GEMV（下限） | 13.9 | 13.9 | 13.9 | 13.9 |
| launch，小算子各自读写全局（现在的形态） | 18.2 | 23.4 | 34.1 | 52.9 |
| launch，小算子合成一个 kernel，仍读写全局 | 18.3 | 22.5 | 31.1 | 46.0 |
| launch，小算子合成一个 kernel，中间结果留在寄存器 | 16.6 | 17.4 | 18.1 | 22.0 |
| 常驻 kernel，读写全局，预取权重 | 17.8 | 21.1 | 30.0 | 45.8 |
| 常驻 kernel，中间结果留在寄存器，预取权重 | 14.8 | 15.3 | 16.6 | 18.3 |

所有变体的输出与 launch 版逐位相同。

结论：

- 一次 launch 在 graph 里只要 1.3–1.6 µs，而全局 barrier 也要 1.1–1.3 µs，用 barrier 代替 launch 几乎省不了什么。
- 小算子贵在全局内存往返（权重流把 8 MB 的 L2 冲掉，每次读写约 1.5–2 µs 延迟），不在 launch：同样 8 个小算子，读写全局约 4 µs / 个，中间结果留在寄存器只要约 0.5 µs / 个。这一点在普通 launch 模式下靠融合也能拿到。
- 预取权重只有在小算子很短时才藏得住：小算子读写全局时，和预取争带宽，几乎不重叠。
- 和最好的 launch 做法（中间结果留在片上）相比，常驻 kernel 每段只多省 1.5–3.8 µs。

实际模型每 token 1037 次 launch = 401 次 GEMV 类 + 636 次小算子，两个 GEMV 之间多是 0–3 个小算子（0 个 96 处、1 个 112 处、2 个 84 处、3 个 100 处）。按上面的数推算，常驻 kernel 每 token 能省约 0.4–1.1 ms，即约 +5% 到 +14%（114–125 token/s）。代价是把整个单 token forward 重写成一个 kernel：寄存器用量取最大的那一段，会压低 GEMV 的占用率；mix 等 1024 线程的单 block kernel 要改写；常驻 kernel 占满所有 CU，和这张卡上的其他进程冲突。所以不做。

更值得做的是测试里看到的另一点：把 GEMV 前后的小算子并进 GEMV 的读入或写出阶段（每个 GEMV block 本来就要读入整个输入向量，可以顺便算 norm；逐通道的 conv 可以在写出时做），省掉小算子的全局往返。候选：`ink_conv_plain`（96 次 / token）、`ink_norm_mul`（83）、control 相关（96）、clip（54）。

## prefill 优化（pp2048：519 → 6280 token/s）

代码在分支 `inkling-opt`（ab37db3），补丁 0007 是相对 4f9a18f 的整体 diff（不含各分支的 NOTES 和测试脚本）。目标是 2K prompt（llama-bench pp2048，ubatch 2048）达到 6000 token/s。分三轮完成，每一项都在独立分支上实现，并经过独立审查。

| 阶段 | 内容 | pp2048（ubatch 2048） |
| --- | --- | --- |
| 起点 | Unsloth 的带状注意力参考实现、逐 token 的 mix、MMQ 专家矩阵乘 | 519 |
| 第一轮 | 新 prefill 注意力（`inkling-pfattn.cu`，分块、只走窗口内的 key、矩阵单元、偏置现场计算）+ 批量 mix（`inkling-pfmix.cu`，逐位相同） | 2390 |
| 第二轮 | MoE 分组 GEMM（`inkling-pfmoe.cu`：NVFP4 精确反量化成 bf16，bf16 矩阵单元，SITU 并进 gate_up 尾部）；小算子批量融合（`inkling-pfops.cu`，逐位相同）；稠密投影（`inkling-pfgemm.cu`）；注意力改 f16 矩阵单元（Q/P 拆成高低两部分保精度） | 5570 |
| 第三轮（全部逐位相同） | 主机侧：相同 mask 视图只上传一次、mask 填充多线程、分段捕获、捕获时保留激活转换缓存；MoE 尾部（加权求和、bf16 交接、BF16 专家 SITU）；合并 forget/gain；gate 的 sigmoid×mul 并进注意力；conv/ctrl 读取提前 | **6280** |

最终结果（ab37db3）：

| prompt | ubatch 512 | ubatch 1024 | ubatch 2048 |
| --- | --- | --- | --- |
| pp512 | 4205 | 4290 | 4229 |
| pp2048 | 3376 | 5384 | **6276** |
| pp8192 | 2386 | 2582 | 2259 |

12k token 的 prompt（llama-completion）约 2100 token/s。decode 不变（tg128 约 110，三种 md5 不变）。

正确性：prefill 的数值不再与原版逐位相同（注意力、MoE、稠密投影改了求和顺序或中间精度），标准是"不超过原版自己只换 ubatch 大小时的差异"：原版 ubatch 1024/2048 对 512 的 8 段 KLD 为 0.0024–0.0029、top-1 相同率约 98.4%。最终版本 8 段 KLD：ubatch 2048 平均 0.00282、中位数 0.00051、top-1 相同 98.45%；ubatch 512 平均 0.00230、98.51%。第三轮全部逐位相同（prefill md5 与第二轮合并版一致）。检查脚本：`kldpp.sh`（2 段）、`kld8.sh`（8 段），对照原版 prefill logits。

主要开关（均可单独关闭以还原前一版本）：`GGML_CUDA_INK_PFATTN`、`_PFATTN_F16`、`_PFATTN_GATE`、`_PFMIX`、`_PFMOE`（及 `_PFMOE_RED`、`_XBF`、`_DSPLIT`、`_SITU0`）、`_PFOPS`（及子开关）、`_PFGEMM`（及 `_PAIR`、`_CAPCACHE`；`_GATEO` 默认关）、`GGML_SCHED_INK_DEDUP`、`LLAMA_INK_MASK_THREADS`、`GGML_CUDA_INK_PFGRAPH_CHUNK`、`_PFGRAPH_LEAN`。

经验：

- 原来的瓶颈都在"参考实现"上：带状注意力逐 key 循环（2048 token 时占 62%），mix 被拆成约 120 万个小 block，MMQ 在 gfx90a 上每个 CU 只能放 1 个 block、每条 MFMA 配约 14 条标量指令（只到 int8 峰值的 14%）。
- NVFP4（E2M1 × UE4M3）可以精确表示成 bf16，反量化后走 bf16 矩阵单元比 int8 路线快（4077 对 3767），而且激活用 bf16 比 q8_1 更精确，KLD 反而下降。
- ggml 会把 MUL_MAT_ID 和每专家的 NVFP4 scale 乘法融合后直接调 MMQ，大多数调用根本到不了 `ggml_cuda_mul_mat_id`；要在那个融合点接入。
- hipcc 会把 lambda 捕获的寄存器数组放到 LDS 或 scratch，占用率骤降；同一迭代里用到的全局读取会被下沉到使用处，要跨迭代使用加 `sched_barrier`。都要看 `.s` 文件确认。
- AMD 上 `__fmul_rn` / `__fadd_rn` 仍会被收缩成 FMA，要自己写带 `fp contract(off)` 的辅助函数。
- 主机侧 30 ms 的空隙不是缺少 graph 捕获（2048 token 的 ubatch 本来就在捕获和回放），而是 48 份相同的 8 MB mask 视图每层同步上传一次、mask 在 CPU 上单线程填充。
- f16 矩阵单元在 gfx90a 上把次正规数当 0，Q 拆高低两部分时要给低位部分加缩放。
- 只用 2 段文本测 KLD 时，平均值会被个别 token 的翻转左右；判断要用 8 段，并以原版自己换 ubatch 的差异为参照。

没做的：pp8192 在 ubatch 2048 时反而比 1024 慢（2259 对 2582），长 prompt 没有专门优化；MoE GEMM 约 80 TFLOPS（峰值 181）；DSpark 的多 token 验证路径。

## 下一步方向

现在约 9.1 ms/token，其中读权重按 1.27 TB/s 算最少 4.6 ms，主机侧不到 0.1 ms。剩下约 1037 个 kernel/token（每个 layer pass 约 21 个）。各分支记录的未做项：

1. `conv_plain` 并进读它的 mix（96 次/token，估计 +1.3 token/s）；非循环层 mix 后面的 norm_mul；control norm 并进 mix 或 down GEMV。
2. rel 投影并进 Q 的 BF16 launch（需要 clip kernel 读 scratch），随后 R clip 可以并进 Q clip 的 launch；循环层第 2、3 遍的 k/v 投影。
3. SITU + q8 量化仍是独立的一次 launch（44 次/token），结构上并不进 GEMV。
4. DSpark 的多 token 验证 graph 没有融合，也没有专家去重。
5. 是否把 inkling2 架构移植进本仓库的源码树：目前优化都在 Unsloth 合并版上，以补丁形式保存在 [patches/](patches/)。
