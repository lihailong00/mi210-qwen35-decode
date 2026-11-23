# MI210 上 Qwen3.8-27B IQ3\_S 单流解码优化：工作交接文档

2026-11-14

## 一、背景、目标与最终结果

单流 decode（不开 MTP）从 45.6 提升到 80.3 token/s，提升 76%；目标 90 token/s 未达到，还差每 token 约 1.35 ms。

任务是在单张 AMD Instinct MI210 上，用 llama.cpp b12110 跑 Qwen3.8-27B-GSQ-RCO-IQ3\_S-mtp.gguf（架构 qwen35），提高单流解码速度。约束：同一个 GGUF，不重新量化、不降精度；允许无损的权重重排、新 kernel、融合和启动开销优化。时间窗为 12 小时（2026-11-14，08:30 起）。

起点是"保留版"：b12110 加 ROCm 7.2 兼容补丁，再加一个把 IQ3\_S/IQ3\_XXS 权重展开成 4-bit 副本的补丁（每 256 个权重 134 字节）。保留版 decode 为 45.6 token/s。

| 指标 | 保留版基线 | 最终版本 | 变化 |
| --- | --- | --- | --- |
| decode tg128，上下文 0 | 45.63 token/s | 80.33 token/s | +76.0% |
| decode tg128，上下文 8192 | 44.23 token/s | 74.57 token/s | +68.6% |
| decode tg128，上下文 32768 | 40.65 token/s | 64.43 token/s | +58.5% |
| prefill pp8192 | 1220 token/s | 2125 token/s | +74%（第十三节的各项 prefill 修正；只做 decode 优化时为 1245） |
| llama-server 单请求（-np 1） | 45.6 token/s | 79.1 token/s | +73% |
| 每 token kernel 启动数 | 1933 | 601 | −69% |
| 显存峰值（llama-bench） | 24.1 GB | 28.1–31.9 GB | +4–8 GB |

质量：与基线 logits 的平均 KL 散度为 0.00125，处在纯计算重排的噪声底线（0.00114）上。显存增加来自同尺寸的重排权重副本，原权重保留给 prefill 使用。

## 二、环境与工作区

所有工作都在一台单 MI210 服务器的 `/data/tools/mi210-opt` 下完成，编译和运行在一个长期运行的容器里。

**硬件与系统**

- 服务器：普通用户（无 sudo），Ubuntu 24.04，Ryzen 7 9700X（16 线程），60 GB 内存。
- GPU：一张 MI210（gfx90a，CDNA2，104 CU，wave64，64 GB HBM2e），PCI 03:00.0，对应 card1 / renderD128。另有一块 AMD 核显（renderD129），容器里只映射 MI210。
- 宿主机没有安装 ROCm，编译和运行都在 Docker 容器里，ROCm 7.2.1。
- 模型：`/data/model/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`，12,120,016,960 字节，SHA-256 以 58fd8267 开头。

**工作容器 `mi210-opt`**

- 镜像 `local/llama-fastmtp-rocm-profile:20261003`（ROCm 7.2.1 + rocprofv3 + cmake/ninja/python3）。
- 以 uid 1000 运行，附加 video(44)/render(992) 组，`--network host`，`seccomp=unconfined`。
- 挂载：`/data/tools/mi210-opt` → `/lab`（读写），`/data/model`（只读）。容器内所有路径都用 `/lab/...`。
- 主进程是 `sleep infinity`，用 `docker exec mi210-opt bash -lc '...'` 执行命令；停掉后用 `docker start mi210-opt` 恢复。

**目录结构（宿主机路径）**

| 路径 | 用途 |
| --- | --- |
| `src/` | git 仓库主工作树，master = 保留版源码；所有分支和标签都在这个仓库里 |
| `wt-<分支>/` | 各分支的 git worktree，各自带 `build/` 编译目录 |
| `wt-final/` | 最终版本（分支 final，标签 mi210-final），二进制在 `wt-final/build/bin` |
| `build/` | 保留版的完整编译（对照用） |
| `baseline-retained/` | 官方 b12110 工具 + 保留版 libggml-hip.so（A/B 对照的 A） |
| `gemv/` | GEMV 微基准框架、真实权重切片（`gemv/w/*.bin`）和各模块的独立开发仓库 |
| `eval/` | 评估脚本、困惑度文本、提示词、基线 logits（`ref-logits.bin`） |
| `results/` | 所有测量结果、trace 和日志 |
| `deliver/` | 交付物：完整补丁和启动脚本 |
| `logs/` | 编译日志、每次 GPU 作业的温度与显存采样 |

**关键脚本**

- `gpu-run.sh <标签> "<容器内命令>"`（宿主机执行）：用 flock 串行化 GPU 作业；开始前等 GPU 降到 60°C 以下；每秒记录温度和显存；任一传感器达到 85°C 就杀掉作业。会自动透传 `GGML_*`、`LLAMA_*`、`CODEX_*`、`HIP_*` 环境变量。**所有 GPU 作业都必须经过它**。
- `build-wt.sh <源码目录> <编译目录> [目标...]`（容器内）：首次配置 cmake，然后编译；并行度由环境变量 PAR 控制。
- `bench.sh`、`final-run.sh`、`eval/kld.sh`、`eval/eval.sh`、`eval/server_test.py`：测速与评估，用法见第十二节。
- `merge-int.sh`：把若干分支合并到 integration（忽略行尾差异并统一为 LF，自动应用 `int-patches/` 里的补丁）。


## 三、基线测量与性能模型

保留版每个 token 约 21.9 ms：权重 GEMV 约 16.5 ms，其余 1460 个小 kernel 约 4.5 ms。两个一起压缩，是之后所有工作的出发点。

**模型结构（影响优化的部分）**

- 64 层，其中 48 层是 Gated DeltaNet（线性注意力），16 层是全注意力（第 il%4==3 层）。另有 1 层 MTP，关闭 MTP 时不读。
- 隐藏维 5120，FFN 17408。注意力 24 个 q 头、4 个 kv 头，头维 256，attn\_q 宽 12288（含输出门）。IMROPE，旋转 64 维，分段 11/11/10，base 1e7。
- DeltaNet：16 个 k 组、48 个 v 头，状态维 128，d\_inner 6144，卷积核 4（conv 通道 10240）。每层递归状态 3 MB。词表 248320。
- 混合精度：每 token 读 IQ3\_S 3.64 GB、IQ4\_XS 3.02 GB、IQ3\_XXS 2.08 GB、Q4\_K 1.58 GB（其中 LM head 0.715 GB），另有少量 IQ2\_S/Q2\_K/IQ2\_XS/IQ2\_XXS/IQ1\_M/BF16。同一层的 gate 与 up 常常类型不同（64 层里只有 24 层相同）。
- 每 token 至少读取 11.35 GB 权重（GGUF 原始尺寸，不含 MTP 和只读一行的 token\_embd）。保留版的 4-bit 展开副本使实际读取变成 12.9 GB。

**硬件性能模型（实测）**

| 量 | 实测值 |
| --- | --- |
| 纯读带宽上限 | 13–50 MB 张量 1.33–1.42 TB/s；1 GiB 连续读 1.455 TB/s |
| 单个 GEMV 的读取底线 | t ≈ 1.6 µs + 字节数 / 1.46 TB/s |
| HIP graph 回放中一个空 kernel | 约 1.35 µs；实际小 kernel 约 2–3 µs |
| 单 block 的 rms\_norm（5120 元素） | 4.4–5.7 µs（延迟链） |
| HIP graph | llama.cpp 每 token 8 个 graph 块回放；关闭会慢 1.6 ms |
| 时钟 | 核心 1700 MHz、显存 1600 MHz，220–245 W，无降频 |

由此得到硬约束：纯读 11.35 GB 至少 7.8 ms；90 token/s 对应 11.1 ms，等于整个 token 平均要跑到 1.02 TB/s，约为带宽上限的 75%，包括所有小 kernel 和间隙。

**基线每 token 时间分解**（rocprofv3，校准后的真实估计）

| 类别 | 启动数 | 耗时 |
| --- | --- | --- |
| 权重 GEMV | 473 | 16.5 ms（平均 0.80 TB/s，带宽上限的 55%） |
| 激活量化（quantize\_q8\_1 + IQ3 求和） | 513 | 1.39 ms |
| 各类 norm | 305 | 1.08 ms |
| 全注意力链 | 约 80 | 0.89 ms |
| DeltaNet 小算子链 | 约 720 | 0.69 ms + 小 kernel 开销 |
| 逐元素、拷贝等 | 约 370 | 1.04 ms |
| 空闲（token 之间） | — | 0.20 ms |

基线的重灾区：

- IQ2\_S/IQ2\_XS/IQ2\_XXS/IQ1\_M 只占 0.73 GB，却要 2.3 ms（约 0.3 TB/s）。码表查表走全局内存，受延迟限制。
- 48×5120 的 BF16 ssm\_alpha/ssm\_beta 矩阵每 token 96 次、每次 11 µs，共 1.07 ms，只用了 48 个 workgroup。
- 每个 GEMV 前都有单独的激活量化 kernel，但只有 257 个不同的输入向量。
- 通用 mul\_mat\_vec\_q 在 gfx90a 上每个 128 线程块只算一行，K=5120 时最后一轮只有 25% 线程在工作。

## 四、工作方法

工作按"先测量、再并行开发、频繁集成、用 KL 散度验收"的循环进行，共四轮开发加一轮审查修复。

**分工与分支**（每条工作线一个 git 分支和独立 worktree/编译目录，GPU 通过锁共享）

| 轮次 | 分支 | 内容 |
| --- | --- | --- |
| 第 1 轮 | gemv | GEMV 框架：q8 激活缓存、q8 生产者、低比特模块接入、BF16 细矩阵、各类尾部融合 |
| 第 1 轮 | gemv-lowbit（独立仓库 `gemv/lowbit`） | IQ2\_S/IQ2\_XS/IQ2\_XXS/Q2\_K/IQ1\_M 新 kernel |
| 第 1 轮 | deltanet | 融合的 Gated DeltaNet 单 token 解码算子 |
| 第 1 轮 | attn | 注意力层前处理融合、异步上传、FA 门融合、norm |
| 第 2 轮 | gemv-iq3、gemv-iq4k | IQ3 紧凑布局 kernel；IQ4\_XS/Q4\_K kernel |
| 第 3 轮 | iq3-r3、iq4k-r3 | 同输入 GEMV 合并为一次启动、移动读取窗口布局、小形状 |
| 第 4 轮 | r4-mixgroup、r4-ffndown、r4-attn | 混合类型配对、ffn\_down 前序指令顺序、短上下文 FA 解码、小代码 norm |
| 审查后 | fixes | 修复代码审查确认的 12 个缺陷 |
| 收尾 | diet-gdn、diet-gemv | 缩减关键路径指令数（冷指令缓存） |

**集成流程**

1. 各分支验证通过后提交；integration 分支从 master 重新合并各分支（`merge-int.sh`）。
2. 冲突手工解决后编译，跑 bench 和 KLD；重要节点打标签（int-v8-base、int-v9、int-v11）。
3. 后期的分支都从最新标签分出，只在其上叠加新提交，减少重复冲突。
4. 最终版 = int-v11 + fixes + diet-gemv + diet-gdn，分支 final，标签 mi210-final。

**各集成版本的速度**

| 版本 | 时间 | decode token/s | 新增内容 |
| --- | --- | --- | --- |
| 基线 | 08:45 | 45.6 | 保留版 |
| v1 | 10:35 | 60.1 | 融合 DeltaNet、注意力前处理、GEMV 框架 |
| v2 | 11:20 | 61.5 | q8 生产者、异步上传 |
| v3 | 11:50 | 62.5 | 低比特模块 v3、多块 norm |
| v4 | 12:30 | 66.2 | IQ3 紧凑布局 + LDS 码表 |
| v5 | 13:05 | 72.3 | IQ4_XS/Q4_K 模块 |
| v7 | 14:25 | 73.4 | 门控 norm 融合 |
| v8 | 14:55 | 74.0 | DeltaNet 输出直接写 q8 |
| v9 | 15:20 | 76.0 | 同输入合并、移动读取窗口 |
| v10 | 16:30 | 77.4 | 混合配对、短上下文 FA |
| v11 | 16:55 | 78.0 | 小代码 norm、前序重排 |
| 最终 | 17:45 | 80.3 | 缺陷修复 + 代码瘦身 |

最大的两次跳跃来自第一次集成（融合 DeltaNet 等，+14.5）和 IQ4\_XS/Q4\_K 模块（+6.1）；后期每轮只增加 1–3 token/s。

**验收标准**

- 速度：同一二进制用环境变量开关做交替 A/B（至少 3 轮），或新旧二进制交替跑；单次测量噪声约 ±0.5–1%。
- 质量：平均 KLD ≤ 0.0015，top-1 一致率 ≥ 98.3%，困惑度比值偏差在 0.5% 以内。噪声底线的来历见第六节。
- 不要求位精确：GEMV 累加顺序变化在这个带递归状态的模型里会被放大。但融合类改动（算子串联融合、代码缩减）都要求与上一版位精确，并用 KLD 表和贪心输出 md5 确认。
- 量化激活（q8\_1）必须与 ggml 的 quantize\_q8\_1 位精确一致：d = amax/127，q = roundf(x/d)，ds = (d, d·sum)。

## 五、已完成的优化

收益最大的三项是：融合 DeltaNet 算子（约 2.8 ms）、按类型重写的 GEMV（约 4 ms）、消除量化和小 kernel 启动（约 2 ms）。下面按组件列出原理、文件、收益和关闭开关；所有开关默认都是开。

### 1. 激活量化：q8 缓存与生产者

- 原理：生成 GEMV 输入的 kernel（RMS\_NORM+MUL、SWIGLU、SILU/SIGMOID+MUL、DeltaNet 输出、FA 合并）顺便写出 q8\_1 块和每 32 元素整数和，与 quantize\_q8\_1 位精确。按流维护一个缓存，输入相同的连续 GEMV 只量化一次。
- 缓存条目按值保存（基张量、数据指针、ne10），每次 graph 计算开始时清空，上下文销毁时释放。
- 文件：`mmvq.cu`（缓存）、`q8-producers.cu`（生产者 kernel）、`ggml-cuda.cu`（调用点）。
- 收益：量化启动从每 token 513 次降到约 24 次。
- 开关：`GGML_MMVQ_Q8_CACHE=0`，`GGML_Q8_PRODUCERS=0`（分项 `_NORM`/`_GNORM`/`_CONT`）。

### 2. 重排权重缓存（各 GEMV 模块共用）

- `ggml_cuda_get_repacked()`：按权重 buffer 缓存重排后的副本，键里包含布局类型；只在非 graph 捕获的首次计算时创建，随 buffer 释放。
- 副本与原权重同尺寸（或多 1.8%），原权重保留给 prefill 的 MMQ 路径。显存因此增加 4–8 GB。

### 3. IQ3\_S / IQ3\_XXS 模块（`iq3gemv.cuh`、`mmvq-iq3.cu`）

- 布局：无损重排为 IQ3\_S 112 字节/256 权重（GGUF 为 110）、IQ3\_XXS 98 字节/256，取代保留版 134 字节的 4-bit 展开副本，每 token 少读 1.56 GB。
- 码表放进 LDS，并预先把符号展开（IQ3\_S 32 KB、IQ3\_XXS 16 KB）：每 4 个权重只需一次 ds\_read\_b32 加一次 dot4，没有符号运算。
- 行按 4 行一组交错，每条读指令读连续 1 KB；再用"步主序"布局，把同一步所有单元的数据放在一起，GPU 只读一个移动窗口。
- 读取与解码分开，在途数据不参与 ALU 运算。常驻式：104 个 workgroup × 16 wave，每行 16 lane，每 wave 4 行。要求 K%256==0、N%4==0。
- 尾部：残差加法；融合 SwiGLU gate+up（S+S、XXS+XXS、S/XXS 混合，混合只用于普通 SwiGLU）。
- 多任务 kernel `gemv_mj`：同为 IQ3 的 qkv+z、k+v 一次启动。
- 前序：先发码表读取再读激活；码表展开从每字 7 条指令减到 2 条。
- 收益：17408×5120 IQ3\_S 34.5 µs（1129 GB/s，原 39.6 µs），IQ3\_XXS 32.6 µs（原 46.1 µs）；融合 gate+up 约 1.2 TB/s；模型内 IQ3 总耗时 7.0 → 约 5.5 ms。
- 开关：`GGML_MMVQ_IQ3=0`（回到 4-bit 展开路径），`_MULTI=0`，`_PF=5`（旧前序），`_SM=0`（旧布局），`_SMALLN=1`（1024 行走旧 kernel，默认关）。

### 4. IQ4\_XS / Q4\_K 模块（`iq4k-gemv.cuh`、`mmvq-iq4k.cu`）

- 布局：同尺寸 SoA，码流（每 256 权重 128 字节）加头信息流（IQ4\_XS 8 字节、Q4\_K 16 字节）。行按 16 字节交错；K=17408 用"轮次分块"布局保持移动窗口。
- kernel：每 CU 一个常驻 workgroup，激活以 16 字节对齐记录暂存在 LDS；跨行乒乓寄存器流水；IQ4\_NL 查表用 v\_perm。按形状选配置，支持残差尾部和同类型 SwiGLU。
- ISA 教训：循环里任何条件计算、条件读、写全局内存或中途 break，都会让编译器插入 vmcnt(0)，把流水串行化。
- 小形状（N≤2048）：k+v 一次启动，沿用 stock 结构保持位精确。
- 收益：约 −0.8 ms；LM head（Q4\_K 715 MB）775 → 512 µs（1.41 TB/s）。
- 开关：`GGML_MMVQ_IQ4K=0`，`GGML_MMVQ_IQ4K_GROUP=0`（`_SMALL=0`；`_BIG=1` 可开启大分组，实测更慢），`GGML_MMVQ_IQ4K_X`/`_Q`="L:NW" 覆盖配置。

### 5. 低比特模块（`lowbit-gemv.cuh`、`mmvq-lowbit.cu`）

- 覆盖 IQ2\_S、IQ2\_XS、IQ2\_XXS、Q2\_K、IQ1\_M。同尺寸 SoA：流 A 每 32 权重 8 字节（码表索引 + 转置后的符号位），流 B 为高位/scale，流 C 为每 256 的缩放。
- 码表放在 LDS 地址 0；符号用 v\_perm 选择字构造，每 4 权重 2 条指令；IQ2\_XXS/XS 的符号奇偶位在寄存器里重建。
- 收益：独立测试 IQ2\_S 690–896 GB/s、IQ2\_XS 742–796、IQ2\_XXS 701–818、Q2\_K 926–1018、IQ1\_M 680（为 stock 的 2.1–2.7 倍）；模型内 2.8 → 1.2 ms。
- 开关：`GGML_MMVQ_LOWBIT=0`。

### 6. 融合 Gated DeltaNet 解码（`gdn-decode.cu`、`ggml.h`/`ggml.c`、`ggml-cpu/ops.cpp`、`src/models/qwen35.cpp`）

- 新增两个 ggml op：`GGML_OP_GDN_DECODE_PREP`、`GGML_OP_GDN_DECODE_OUT`，追加在 GGML\_OP\_GLU 之后（GGML\_OP\_COUNT=104）。RPC 协议补丁号 +1，meta 后端和 RPC 都报告不支持。CPU 有参考实现。
- prep：q/k 卷积 + silu（卷积状态原地移位）、q/k L2 归一化、alpha/beta 的 BF16 投影（每头 2 个长 5120 的点积）、门值计算。
- out：v 卷积；通过 s\_copy 索引在缓存行里原地读写递归状态并做 delta 规则更新；门控输出 norm（rms\_norm·w·silu(z)）；可选 q8\_1 尾部供 ssm\_out 使用。每头 8 个 block，最后到达的 block 做 norm。
- 替换了每层 15 个小 kernel 和 2 个 BF16 GEMV，去掉了每层 3 MB 的状态拷贝。与未融合链位精确（归约树、lane 映射、fma 模式全部照抄），`tests/test-gdn-decode.cpp` 对 4 步解码逐位校验。
- 启用条件：单 token、单序列、n\_rs\_seq==0（无投机回滚快照）、f32 状态、所有相关张量在同一个 CUDA/ROCm 设备、构造前提全部满足；否则回退原图。
- 精简版：针对 qwen35 形状（H\_k=16、H\_v=48、S=128、卷积 4）的特化 kernel，prep 758→638 条指令，out 755→491 条。
- 收益：单独 45.7 → 51.5 token/s（约 −2.8 ms）；精简再 +1.2%。
- 开关：`LLAMA_GDN_FUSED_DECODE=0`，`GGML_GDN_DECODE_SPLIT=0`，`GGML_GDN_DECODE_Q8=0`，`GGML_GDN_DIET=0`（精简版）。

### 7. 注意力层（`rope.cu`、`fattn-common.cuh`、`fattn.cu`、`fattn-dec.cu`、`qwen35.cpp`）

- attn\_prep：q/k 的 rms\_norm+mul、IMROPE、gate 拷贝、K/V 写入缓存合为一个 kernel，每个注意力层 5 次启动变 1 次。需要 qwen35.cpp 让 q/k/v 投影相邻。θ 幂表由捕获外的初始化 kernel 填充。
- FA 门融合：sigmoid(gate)·attn 及其 q8\_1 在 FA 合并 kernel 里完成。
- 短上下文 FA 解码（KV≤1024）：单次启动的 split-KV，8 个分块 × kv 头，6 个 GQA 头共享 K/V 读取，kernel 内屏障替代合并 kernel，同时做门和 q8。模型内每层 28.4 → 约 16–17.5 µs。要求整个 grid 同时驻留（8×kv 头数 ≤ CU 数）。
- 融合的 norm+rope 支持 MROPE/IMROPE（作为回退路径）。
- 开关：`GGML_CUDA_DISABLE_ATTN_PREP_FUSION=1`，`GGML_CUDA_DISABLE_FA_GATE_FUSION=1`，`GGML_CUDA_FA_DEC=0`，`GGML_CUDA_FA_DEC_MAX_KV`（默认 1024），`GGML_CUDA_FUSED_ROPE_NO_MROPE=1`。

### 8. rms\_norm 与 q8 生产（`q8-producers.cu`、`norm.cu`）

- 一行分给 ne0/256 个 workgroup，每个重做完全相同的整行归约（保证位精确），只写自己的切片和 q8。
- 代码缩减：直接用 roundf，归约改用 DPP/ds\_swizzle，398 → 308 条指令；模型内每次 5.1 → 4.3 µs。
- 输出与输入原地重叠时，改用单 workgroup 的原地版本（见第六节缺陷 F1）。
- 开关：`GGML_Q8_NORM_SMALL=0`，`GGML_Q8_PRODUCERS_NORM=0`。

### 9. 混合模块配对（`mixgemv.cuh`）

- 一个 IQ3 GEMV 和一个 IQ4\_XS/Q4\_K GEMV 共用输入时一次启动：16 对 qkv+z、16 对未融合 gate/up。一个前序暂存码表和两种激活布局，wave 跑完 IQ3 单元直接进入 IQ4 单元。
- 收益：上下文 0 为 +0.14%（噪声内），8K 为 +0.7%；位精确。
- 开关：`GGML_MMVQ_MIX=0`，`GGML_MMVQ_MIX_GLU=1`（SwiGLU 尾部，实测慢 1%，默认关）。

### 10. 主机侧与上传（`ggml-backend.cpp`、`ggml-cuda.cu`）

- split 输入异步上传（每个 split 一次同步），小的 H2D 上传合并为一次拷贝加一个分发 kernel。token 之间 GPU 空闲 201 → 约 112 µs。
- 开关：`GGML_SCHED_SYNC_UPLOADS`，`GGML_CUDA_NO_UPLOAD_BATCH`，`GGML_CUDA_UPLOAD_BATCH_ALTERNATE`。

### 11. GEMV 前序代码缩减（diet-gemv）

- 去掉运行时除法、无用的寄存器清零、多余的地址计算。前序长度：iq3g 132 → 81 条指令，iq4k 189 → 132，配对 kernel 148 → 98。
- 收益：+1.1%（上下文 0），+0.9%（8K）；位精确。

## 六、正确性验证

最终版本的输出分布与基线的差异，和"只改变计算顺序"造成的差异一样大；代码审查确认的 12 个缺陷已全部修复并独立复核。

**KL 散度测试方法**

- 文本：`eval/ppl.txt`（若干英文技术文档拼接，约 80 KB）。用 llama-perplexity，`-c 512 --chunks 8 -ub 1`：每个 token 都走单 token 解码路径，正好覆盖被优化的代码。
- 参考 logits 由保留版用 `--kl-divergence-base` 保存在 `eval/ref-logits.bin`；候选版本用 `--kl-divergence` 对比。
- 噪声底线：保留版本身开 `GGML_CUDA_DISABLE_FUSION=1`（只是换了一种计算顺序），得到平均 KLD 0.00114、top-1 一致 98.82%、困惑度比 1.0013。
- 这个模型有递归状态，微小的浮点差异会随 token 累积放大，所以困惑度会漂移 0.1–0.4%，贪心输出在接近平局的 token 上会分叉。判断以 KLD 为准。

| 版本 | 平均 KLD | top-1 一致 | 困惑度比 |
| --- | --- | --- | --- |
| 噪声底线（保留版关闭融合） | 0.00114 | 98.82% | 1.0013 |
| 第一次集成（60.1 token/s） | 0.00113 | 98.82% | 1.0035 |
| v5（72.3 token/s） | 0.00122 | 98.48% | 1.0021 |
| 最终版本（80.3 token/s） | 0.00125 | 98.78% | 1.0016 |

最终版本的困惑度为 3.8018（基线 3.7981）。各版本间只有重排类改动会改变 KLD；其余改动的 KLD 表逐块完全相同。

**哪些是位精确的**

- 与基线位精确：融合 DeltaNet 算子、q8 生产者、norm、各类尾部融合、异步上传。
- 只改变 fp32 累加顺序：各 GEMV 模块、短上下文 FA 解码。
- 更正：attn\_prep 的注释写着位精确，实际约百万分之 1–3 的 K 缓存元素与未融合路径差 1 个 f16 ulp（融合指令内的舍入），已包含在上表 KLD 里。

**贪心输出**

- 3 个提示词（英文散文、Python 代码、中文问答），各生成 128 token，`--temp 0 --ignore-eos`，比较 md5。
- 最终版本 p1 与基线完全一致；p2、p3 在接近平局的 token 上分叉，内容等价。最终版 md5：p1 `857b01f7…`、p2 `ba86ecd8…`、p3 `9e035710…`。

**服务器模式**

- llama-server，`-c 65536`，分别用 `-np 1` 和 `-np 4`：3 个顺序请求、4 个并发请求、两轮对话的 prompt 缓存复用。全部正常，无崩溃。
- 与基线输出比较：np=1 有 3/7 个输出完全一致，np=4 有 4/7，其余在 26–93 token 后的平局处分叉。

**代码审查与修复**

三位审查者分别审 GEMV 模块、图级融合与 q8 缓存、DeltaNet 算子，共报 13 个问题。每个问题再由独立反驳者尝试推翻，12 个被确认。修复在分支 fixes，由另一位审查者重新跑所有探针复核：修复前全部复现，修复后全部通过，默认路径仍位精确。

| 编号 | 缺陷 | 触发条件 | 修复 | 验证 |
| --- | --- | --- | --- | --- |
| F1 | 多 workgroup 的 rms\_norm+q8 在输出与输入原地重叠时有数据竞争 | **默认路径**：LM head 前的最终 norm 是原地的；GPU 被其他进程占用时触发 | 重叠时改用单 workgroup 原地 kernel，仍产出 q8 | 修复前加干扰 50/50 次出错；修复后 0/1000 |
| F2 | attn\_prep 在多流并发区域里与 K/V 写入竞争 | `GGML_CUDA_GRAPH_OPT=1` | 并发区域内不做该融合（FA 门融合同理） | 3 次运行输出一致 |
| F3 | attn\_prep 的 θ 幂表首次发布不安全 | `--no-warmup`、交替加载两个模型 | 捕获外由初始化 kernel 填表；捕获中缺表则自己算 | 首次启动错行 17–29 → 0 |
| F4 | q8 缓存解引用已失效的张量指针，且从不释放 | 关闭 q8 生产者或关闭融合 | 按值存键，先判有效，按流释放 | SIGSEGV → 正常 |
| F5 | 非 HIP（CUDA/MUSA）无法编译 | CUDA/MUSA 构建 | 补桩函数、HIP 专用代码加守卫 | clang 语法检查通过（本机无 CUDA 工具链） |
| F6 | 融合 DeltaNet 路径会被 RPC 设备采用 | `GGML_RPC=ON` 并卸载到远端 | 只在 CUDA/ROCm 后端启用；RPC 报告不支持 | 远端不再崩溃 |
| F7 | ssm\_alpha/ssm\_beta 类型不同时硬中止 | 其他量化配方 | 构造前提全部前置检查 | 中止 → 正常回退 |
| F8 | 混合 IQ3 gate/up 融合被用于 GEGLU 等非 SwiGLU | 其他模型 + 混合 IQ3 | 只允许非交换的普通 SwiGLU | NaN → 正确 |
| F9 | MoE 共享专家匹配器接受混合 IQ3 门 | MoE + 混合 IQ3 | 要求三者同类型 | 中止 → 正确 |

未修项：RPC 服务端拒绝未知 op（需要协议变更）；test-backend-ops 里没有为这些修复加测试用例。

## 七、最终性能与剩余时间分解

最终版本每个 token 的 GPU 节点耗时为 12.27 ms（601 次启动），加上 token 之间 0.11 ms 空闲，对应 80.3 token/s。其中 GEMV 占 10.42 ms，是剩余差距的主体。

**最终 A/B 协议**：A = 保留版（官方 b12110 工具 + 保留版 libggml-hip），B = 最终版本，按 A→B→B→A 顺序。每轮跑 llama-bench 的 decode（上下文 0 和 8192，`-r 2`）和 prefill（512/2048/8192），每轮前等 GPU 降温，峰值温度 73°C。decode 四轮分别为 45.77、80.07、80.59、45.49 token/s，汇总见第一节的表。

**每 token 节点耗时**（在每个 graph 节点后插入 GPU 时间戳 kernel 测得，扣除时间戳 kernel 自身 2.1 µs，30 个 token 平均）

| 节点 | 启动/token | 耗时 | 说明 |
| --- | --- | --- | --- |
| FFN gate（含融合 gate+up、混合配对） | 64 | 3.57 ms | 融合的每次约 62.8 µs |
| FFN down + 残差 | 64 | 2.42 ms | 每次约 37.8 µs |
| 线性注意力 qkv（含合并的 z） | 48 | 1.44 ms |  |
| ssm\_out + 残差 | 48 | 0.88 ms |  |
| 未融合的 FFN up | 17 | 0.55 ms |  |
| LM head（Q4\_K，0.715 GB） | 1 | 0.51 ms | 1.41 TB/s，已接近上限 |
| 注意力 q/o/k/v 与单独的 z | 66 | 1.05 ms |  |
| **GEMV 合计** | — | **10.42 ms** | 平均 1.09 TB/s，约为读取底线模型的 79% |
| rms\_norm（含 q8） | 129 | 0.62 ms | 每次 4.8 µs |
| DeltaNet prep + out | 96 | 0.72 ms |  |
| FA + 门 | 16 | 0.29 ms |  |
| attn\_prep、SwiGLU 等 | 49 | 0.20 ms |  |
| **非 GEMV 合计** | — | **1.85 ms** |  |
| token 之间空闲 | — | 0.11 ms | 主机侧采样、上传、graph 启动 |

要到 90 token/s（11.1 ms）还需省 1.35 ms。若 GEMV 达到读取底线模型的 90%（约 9.2 ms），再把非 GEMV 压到 1.5 ms 以下，就能达到。为什么这两步难，见第八、九节。

## 八、试过但无效或放弃的方案

下面这些方向都已实测，收益为零、为负或不值得复杂度；继续研究时可以直接跳过，或换一种思路。

| 方案 | 实测结果 | 原因或备注 |
| --- | --- | --- |
| 多流并发（`GGML_CUDA_GRAPH_OPT=1`） | 69.8 vs 72.3 token/s，更慢 | 会关掉部分融合，跨流同步代价大 |
| 关闭 HIP graph | 慢 1.5–1.7 ms/token | graph 回放显著降低启动开销，必须保留 |
| `HIP_FORCE_DEV_KERNARG=1` | 无变化 | graph 回放已把 kernel 参数放在显存 |
| GEMV 内部自行量化 fp32 输入（常驻 kernel） | IQ3/IQ4\_XS/IQ2 均更慢，只有 LM head 受益 | 每个 workgroup 重复量化，前序变长；改为生产者侧写 q8 |
| 融合 gate+up 的 kernel 顺便写自己输出的 q8 | 原子版 −0.2 token/s；自旋等待版 +0.1% | kernel 末尾的栅栏/原子链约 3.8 µs，超过省下的一次量化启动（约 3.1 µs） |
| 混合配对 kernel 加 SwiGLU+q8 尾部 | −1%（74.84 vs 75.63） | 同上，完成序列约 5 µs，单独的 swiglu\_q8 只要约 3 µs |
| 把 rms\_norm 放进 GEMV 前序 | −1.3%，且不再位精确 | 每个 workgroup 重做归约，前序变长 |
| 大 GEMV 分组（q+k+v、qkv+z、混合 gate+up 的通用任务表 kernel） | 连续启动时每组省 4.8–9.5 µs，graph 回放中反而多 110 µs/token | graph 回放里大 GEMV 几乎没有固定开销可省；合并两个大 GEMV 只省约 0.4 µs |
| 软件流水、stream-K、批量读、激活放 VGPR、每 CU 2–4 个 workgroup、R=2、每项 4 个子块 | IQ3/IQ4K/低比特模块均持平或更慢 | 稳态循环已接近读取速率（约 1.26–1.5 TB/s），瓶颈在头尾 |
| K 分割（split-K）处理 1024 行小矩阵 | 无收益 | 合并开销抵消并行度收益 |
| 调度类：动态/配对 workgroup、提高 wave 优先级、旋转循环、均匀循环、展平行映射 | 全部无收益 | 尾部不均来自硬件偏爱老 wave，软件很难绕开 |
| 预取下一个 GEMV 的前几步到 L2、提前读残差 | 无变化 | L2 冷启动不是瓶颈 |
| 在前序里预取第一项权重（各种 PF 变体） | −0.65% 或持平 | 条件预取会引入 vmcnt(0) 串行化 |
| ssm\_out 前 DeltaNet 状态的 L2 脱脏写回（非时间性存储、buffer\_wbl2） | 无效 | ssm\_out 因此每次多 2.2–3.0 µs（约 120 µs/token），gfx90a 上无解 |
| DeltaNet 单次启动变体（冗余 prep 或带就绪标志的票号排序） | 无收益 |  |
| DeltaNet prep 放到侧流与 z 投影重叠 | 49.1 vs 51.9 token/s，更慢 | 跨流同步约 13 µs |
| DeltaNet 精简版的"预热指令缓存"变体 | 噪声内 | 保留为 `GGML_GDN_DIET=2/3`，默认不用 |
| MFMA 单次启动的解码注意力（两个版本） | 热测 11.2 vs 12.5 µs 更快，模型内 29.7/33.0 vs 26.0 µs 更慢 | 代码更长，冷指令缓存代价更大 |
| GQA 专用解码 FA kernel | 模型内比 tile kernel 慢 | 已删除 |
| 新 FA 解码用于 KV>1024 | 1024 时持平，更长更慢 | 默认上限 1024，更长走 tile 路径 |
| 更小线程块（256/512）的 norm | 模型内更慢 |  |
| 零拷贝上传分发、推迟上传 flush | token 周期变化 ≤ 10 µs |  |
| 我最初的朴素 SoA 常驻原型（IQ3） | 比 LDS 码表版慢 | LDS bank 冲突、在途读取太少 |
| 11 月 5 日旧实验：q8 求和融合、SiLU+q8 融合、CDNA2 线程布局 | +0.6%、+0.4%、+1.9%（后者非位精确） | 当时未保留；本次该方向已被生产者侧 q8 取代 |

## 九、关键经验与硬件行为发现

最重要的一条经验：在模型里测，而不是在热的微基准里测。很多"微基准更快"的 kernel 放进模型后反而更慢。

1. **冷指令缓存**：模型内每次 kernel 派发都从冷的指令缓存开始，关键路径上每条指令约 8–13 ns。同一算法的 FA 解码：4714 条指令 76.5 µs，1427 条 33 µs，847 条 17.4 µs。norm 从 398 条减到 308 条，5.1 → 4.35 µs。
2. **剖析工具失真**：rocprofv3 把任何小 kernel 都显示为 ≥ 4.3 µs，并把 graph 回放拖成 CPU 瓶颈（每节点约 47 µs）。小 kernel 的真实成本约 1.3–1.9 µs。GEMV 的 trace 时长可信（±3%），小 kernel 要用 graph 内 GPU 时间戳测。
3. **graph 回放中的启动成本**：两个 GEMV 之间插一个额外 kernel，无论形状约多 3.5 µs；连续排列的额外 kernel 每个只约 0.7 µs。合并两个大 GEMV 只省约 0.4 µs；分组只对小 GEMV 或前序很重的模块（IQ3 的 LDS 码表）划算。
4. **把小 kernel 吸进常驻 GEMV 不划算**：常驻 kernel 末尾的完成序列（栅栏、原子计数）约 3.8–5 µs，比它省下的那次启动还贵。
5. **码表类量化（IQ*）*\*：码表放全局内存时受查表延迟限制，只有带宽上限的 25–30%。码表放 LDS 快 2.3 倍。GGUF 里 66/74/82/98/110 字节的块只有 2 字节对齐，只读不算也只有 650–800 GB/s，必须重排成宽而合并的读取。
6. **DRAM 局部性**：每个 wave 各自顺序读一大片分散地址时只有 1.16 TB/s；所有 wave 共同推进一个移动窗口可达 1.42 TB/s。这就是"步主序"和"轮次分块"布局的来源。
7. **尾部不均**：硬件调度器偏爱老 wave，同一 CU 上两个 workgroup 的第二个晚 3–6 µs 结束。稳态循环本身已接近读取速率，损失主要在启动、前序和尾部。
8. **编译器与 ISA**：循环里的条件读、条件计算、全局写或 break，会让编译器插入 vmcnt(0) 等全部读完，流水归零。改 kernel 后要看反汇编。
9. **前一个 kernel 的脏 L2 行**：DeltaNet 写完 3 MB 状态后，紧接的 ssm\_out GEMV 要替它写回，每次多 2–3 µs。
10. **递归模型的数值特性**：纯计算重排就会让困惑度漂 0.1–0.4%、贪心输出在平局处分叉；不要用困惑度或贪心 md5 判断正确性，用 KLD 对照噪声底线。
11. **功耗与时钟**：decode 时 220–245 W（上限 300 W），核心稳定 1700 MHz，不受功耗限制。
12. **并发与 prefill 不受影响**：所有改动只针对 ne11==1（单 token），prefill 和多序列批次走原路径。

## 十、经验总结：这次优化可以借鉴的教训

这次最有价值的不是某个 kernel，而是方法：先算上限、在真实环境里测、先校准正确性标尺、用数据决定何时停手。第九节是硬件事实，本节是做法上的教训。

1. **先算物理上限，再定目标。** 每 token 至少读 11.35 GB，光读就要 7.8 ms；90 token/s 意味着整个 token 平均跑到带宽上限的 75%。这一步在开始一小时内就能算出，应据此给出可达区间（当时估计 75–90），而不是承诺一个点。
2. **先测时间分布，再定优先级。** 基线 profile 显示，除 GEMV 外还有约 1460 个小 kernel 和 513 次量化启动，占了约四分之一时间。只盯 GEMV 会漏掉一半的机会；融合 DeltaNet 一项就带来最大的单次跳跃。
3. **测量工具也会骗人，要尽早建立"模型内真实成本"的测法。** rocprofv3 把小 kernel 夸大到 3 倍以上；热的微基准忽略了冷指令缓存、脏 L2 和调度偏向。graph 内 GPU 时间戳工具到下午才做出来，如果第一个小时就有，很多误判可以避免。
4. **以模型内的端到端结果为准，微基准只用来筛选。** 多个方案在微基准里更快、放进模型反而更慢：MFMA 注意力、Q4\_K 调参、各种软件流水和预取。每个候选都要过模型内 A/B 这一关。
5. **正确性标尺要先校准，再定验收线。** 带递归状态的模型里，纯计算重排就会让困惑度漂 0.1–0.4%、贪心输出在平局处分叉。先测出噪声底线（KLD 0.00114），再以 KLD 为准；第一次集成时曾把 +0.29% 的困惑度误当成缓存 bug 排查了半小时。
6. **A/B 要交替、同二进制、带开关。** 共享 GPU 上单次测量噪声约 ±1%，比后期很多优化的收益还大。每项优化都带环境变量开关，既能做同二进制 A/B，又能出问题时一键回退。
7. **数据布局比指令微调更值钱。** 最大的 kernel 收益来自重排：码表放进 LDS（2.3 倍）、无损 SoA 带来宽而合并的读取、移动读取窗口。在循环内抠指令、加流水，收益往往为零。
8. **融合不是越多越好，先给"一个边界值多少钱"定价。** 在 graph 回放里，合并两个大 GEMV 只省约 0.4 µs，把小 kernel 吸进常驻 kernel 的完成同步却要 3.8–5 µs。先用空 kernel 探针测出边界成本，再决定合并哪些。
9. **看清收益递减，及时换打法。** 前两轮拿到了约三分之二的提升，后期每轮只有 1–3%（见第四节的图）。到这一步，继续单项调优不如转向结构性改动（如常驻 FFN kernel），或者接受结果。
10. **并行开发要先定接口和边界。** 按模块和文件分工、统一的模块入口（launch 返回 false 即回退）、GPU 锁、固定的集成标签，让多条线能同时推进。冲突仍集中在公共入口文件（ggml-cuda.cu 的融合入口和 q8 生产者），而且有一次是行尾格式（CRLF）引起的整文件冲突：要统一 LF，并频繁集成。
11. **为性能写的"聪明"代码最需要对抗式审查。** 多 workgroup 重复归约加原地读写，在空闲 GPU 上从未出错、所有测试都通过，却在 GPU 被占用时 50/50 次算错。性能改动要专门在压力条件下测（CU 屏蔽、并发负载、无预热），并请人专门去推翻它。
12. **共享机器上的操作要精确、可逆。** 一条按镜像名批量停容器的清理命令，误停了自己的工作容器 7 分钟。清理只按确切名字，GPU 作业要有温度保护和串行锁，不碰别的服务。

## 十一、剩余差距与后续研究方向

（prefill 优化完成后的复核结论见第十三节第 14 小节：在现有约束下 decode 现实可达约 84–86 token/s，已不再继续。）

到 90 token/s 还差每 token 1.35 ms；单项调优已接近收敛，最有希望的是结构性改动——把整个 FFN 合成一个常驻 kernel。

| 优先级 | 方向 | 预计收益 | 难度/风险 | 说明 |
| --- | --- | --- | --- | --- |
| 1 | 常驻 FFN kernel：gate/up → SwiGLU → down 一次启动，中间用轻量的全网格同步 | 0.3–0.6 ms（估计） | 高 | 每层省 1–2 个启动边界和一次前序/尾部。风险：网格同步本身可能要 3–5 µs（见第九节第 4 条），寄存器压力、整个 grid 必须同时驻留 |
| 2 | 继续缩减关键路径指令数 | 0.1–0.2 ms | 低 | 未做：低比特模块和 iq4k 小形状 kernel 的前序；attn\_prep（约 1200 条指令）；DeltaNet prep 里约 20 条地址计算和 log1pf（约 130 条，不能位精确地缩短） |
| 3 | 融合 gate+up 之后的 ffn\_down 仍需单独量化 | 约 0.1 ms | 中 | 31 个节点每个多 3.1–4.5 µs。融合 kernel 自己写 q8 的尝试因完成序列太贵而失败，需要更便宜的办法 |
| 4 | 长上下文 FA 解码（KV > 1024） | 8K 时约 0.3–0.5 ms，32K 更多 | 中 | 现在 8K 为 74.6、32K 为 64.4 token/s。新 kernel 在 KV>1024 时输给 tile 路径，可试预取和更多分块 |
| 5 | 缓解尾部不均 | 若可解约 0.2–0.5 ms | 高 | 已试多种调度方案均无效；可考虑按硬件偏好有意分配不等量工作 |
| 6 | 多序列并发解码（M=2–8） | 对服务器吞吐很大 | 中 | 本次优化只覆盖 M=1；4 个并发请求时总吞吐仍约 75 token/s，与基线相同。把各 GEMV 模块扩展到小 M 即可受益 |
| 7 | MTP / 投机解码 | 可超过 90 token/s | 中 | 不在本次范围。验证阶段是 M=4 的 GEMV，仍用 stock kernel；把新模块扩展到 M=4 后再开 MTP，收益会叠加 |
| 8 | 去掉重复的权重副本 | 显存 −4–8 GB | 中 | prefill 的 MMQ 仍读原布局；若 MMQ 能读重排布局，就只需一份权重 |

建议从第 2 项入手（低风险、快速见效），第 1 项作为主要研究方向先做原型：先在微基准里测一次网格同步在模型内的真实代价。

## 十二、操作手册

以下命令都在服务器宿主机 `/data/tools/mi210-opt` 下执行；用 GPU 的一律套 `gpu-run.sh`。

**新开分支与编译**

```bash
cd /data/tools/mi210-opt/src && git worktree add -b mybranch /data/tools/mi210-opt/wt-mybranch mi210-final
docker exec mi210-opt bash -lc "PAR=12 /lab/build-wt.sh /lab/wt-mybranch /lab/wt-mybranch/build ggml-hip llama-bench llama-perplexity llama-completion llama-server"
```

新增 `.cu` 文件后必须重新配置：`docker exec mi210-opt bash -lc "cmake /lab/wt-mybranch/build"`，再编译。首次完整编译约 6–10 分钟。

**测速与评估**

| 任务 | 命令（宿主机） | 输出 |
| --- | --- | --- |
| decode 测速 | `./gpu-run.sh t1 "/lab/bench.sh /lab/wt-mybranch/build/bin t1 -d 0,8192"` | `results/bench/t1.jsonl` |
| KLD 对比基线 | `./gpu-run.sh k1 "/lab/eval/kld.sh /lab/wt-mybranch/build/bin t1"` | `results/kld-t1/kld.log` |
| 困惑度 + 贪心 md5 | `./gpu-run.sh e1 "/lab/eval/eval.sh /lab/wt-mybranch/build/bin t1"` | `results/eval-t1/` |
| 服务器测试 | `./gpu-run.sh s1 "python3 /lab/eval/server_test.py /lab/wt-mybranch/build/bin t1 4 8090"` | `results/server-t1-np4.json` |
| 完整 decode + prefill | `./gpu-run.sh f1 "/lab/final-run.sh /lab/wt-mybranch/build/bin t1"` | `results/final/t1-*.jsonl` |

运行环境保持 `CODEX_IQ3_PREPACK=1 CODEX_IQ3_XXS_PREPACK=1`（脚本已默认设置）。用环境变量做 A/B 时，在 `gpu-run.sh` 前导出变量即可透传。

**剖析**

- kernel trace：在容器里用 `/opt/rocm/bin/rocprofv3 --kernel-trace --output-format csv --output-directory <目录> -- <llama-bench ... -n 32 -r 1>`，再用 `python3 /lab/results/quick_trace.py <csv>` 按 LM head 切分 token 并按 kernel 汇总。小 kernel 的时长会被夸大。
- 模型内真实节点耗时：把 `results/attn-misc-gpu-timestamps.patch` 打到一个单独的 worktree（已有 `wt-ts`），编译后用 `GGML_CUDA_TS=1 GGML_CUDA_TS_DUMP=60 GGML_CUDA_TS_OUT=<前缀>` 跑 llama-bench，再用 `python3 results/attn-misc-ts.py <前缀> 30 full` 分析。该补丁不要合进正式版本。
- GEMV 微基准：`gemv/gemv_bench.hip` 与各模块自己的基准（`gemv/iq3/iq3_bench.hip`、`gemv/iq4k/iq4k_bench.hip`、`gemv/lowbit/lowbit_bench.hip`），用真实权重 `gemv/w/*.bin`，编译参数：`hipcc -O3 --offload-arch=gfx90a -std=c++17 -DGGML_USE_HIP -I/lab/src/ggml/include -I/lab/src/ggml/src -I/lab/src/ggml/src/ggml-cuda`。记住微基准是热的，结论要在模型内复核。

**启动优化版 llama-server**

```bash
/data/tools/mi210-opt/deliver/llama-server-mi210.sh -m /data/model/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf -ngl 999 -fa on -c 65536 -np 1 --port 8080
```

**常见坑**

- 新 `.cu` 文件不重新配置 cmake：libggml-hip.so 有未定义符号，加载失败，报 `invalid device: ROCm0`。
- 二进制依赖编译时路径 `/lab/...` 查找 CPU 后端：必须把工作区挂在容器的 `/lab`，否则报 `no CPU backend found`。
- 一个分支曾用 CRLF 行尾提交，整个文件变成冲突：只提交 LF，合并时用 `-X ignore-space-at-eol`。
- 重复出现的冲突点：`ggml-cuda.cu` 里 `ggml_cuda_try_fuse` 的各融合入口（FA 门、BF16 尾部、混合配对、iq4k 分组、attn\_prep，顺序有意义）、`q8-producers.cu` 的 norm kernel、`mixgemv.cuh` 的 gemv\_pair 模板参数。两边都要保留，结构要手工核对。
- 服务器上还有其他业务容器：不要按镜像名批量 `docker stop`（曾因此误停 mi210-opt），测试容器请用固定名字。
- GPU 与其他作业共享时噪声约 ±1%：只信交替进行的 A/B。
- `llama-completion` 不加 `-c` 会按模型最大上下文 262144 分配 KV，显存峰值会到 40 GB，属正常。

## 十三、追加：本机性能上限与 prefill GEMM 修正

### 1. 硬件峰值（实测）

| 项目 | 实测 | 说明 |
|---|---|---|
| HBM 读带宽（单 kernel） | 1446 GB/s（48 MB）、1433（256 MB）、1399（1 GB）、1361（4 GB） | 标称 1638 GB/s 的约 88% |
| MFMA FP16→FP32 | 170.9 TFLOPS | 循环 kernel；1700 MHz 下标称 181 |
| MFMA INT8→INT32 | 171.4 TOPS | gfx90a 上与 FP16 同速，INT8 没有算力优势 |
| prefill 时的频率 | 1565–1700 MHz | 功耗上限 300 W |

### 2. decode 上限

每 token 读取 11.35 GB 权重（不含 token_embd 和 MTP 层），另有 0.30 GB DeltaNet 状态读写（48 层 × 3.15 MB f32，读一次写一次）。KV cache 每个上下文位置 64 KB。

| 上限 | ms/token | token/s |
|---|---|---|
| 物理上限：全部字节以 1446 GB/s 读取，上下文 0 | 8.06 | 124（只算权重为 127） |
| 纯读取，每个 ≥1 MB 权重一次 launch（401 个），HIP graph，无计算 | 8.87 | 112.8 |
| 同上，加状态读写 | 9.25 | 108 |
| 同上，每个张量一次 launch（含 norm，共 850 个） | 9.39 | 106.5 |
| 当前版本 | 12.45 | 80.3 |

上下文 8192 时物理上限约 119 token/s（当前 74.6），32768 时约 105（当前 64.4）。只要每个权重张量单独 launch，约 108 就是实际天花板；要超过它，必须让读取跨越张量边界连续进行（持久 kernel 或融合 kernel）。

### 3. prefill 上限

每 token 计算量：GEMM 48.65 GFLOP（按层形状精确计算），注意力约 1.61 GFLOP（平均因果上下文 4096），DeltaNet 约 0.2 GFLOP，合计约 50.5 GFLOP。

| 上限 | token/s |
|---|---|
| 物理上限：50.5 GFLOP 以 171 TFLOPS 计算（零数据峰值；真实数据受功耗上限限制，见第 6 小节） | 约 3390 |
| 物理上限（真实数据，约 150 TFLOPS） | 约 2980 |
| GEMM 全部用最优库 kernel（hipBLASLt 平均约 150 TFLOPS），其余开销为零 | 约 3000 |
| 当前版本（pp8192） | 2125 |

修正前，rocprofv3 测得每 token 0.804 ms：GEMM 0.577 ms（权重先反量化为 f16，再走 rocBLAS，平均只有 84 TFLOPS），非 GEMM 0.227 ms（逐 token 递推的 DeltaNet kernel 0.065，flash attention 0.047、只有 34 TFLOPS，反量化 0.031，其余小算子约 0.08）。

### 4. prefill GEMM 修正（`ggml-cuda.cu` 的 `ggml_cuda_init`）

- **问题**：rocBLAS 自带的 Tensile 选核启发式在 gfx90a 上对 ffn_down 形状（m=5120、n=4096、k=17408，f16 输入、f32 输出）选了 64×64 宏块 kernel，只有 47.5 TFLOPS，而 rocBLAS 自己的 227 个候选里最好的能到 148。这一个形状就占了修正前 prefill GPU 时间的约 30%。其他形状也慢 10–25%。
- **修正**：设 `ROCBLAS_USE_HIPBLASLT=1` 后，rocBLAS 把 GEMM 交给 hipBLASLt，其默认选核在所有 prefill 形状上达到 140–160 TFLOPS；每个 4096-token ubatch 的 GEMM 时间从 2197 ms 降到 1364 ms，比 rocBLAS 逐形状调优的最好结果（1414 ms）还快。
- **实现**：在 `ggml_cuda_init` 中，如果所有设备都是 CDNA2、用户没有设置这个环境变量，就设置 `ROCBLAS_USE_HIPBLASLT=1`。rocBLAS 在第一次创建 handle 时读取它，而 handle 是之后才按需创建的。要恢复原来的 rocBLAS kernel，设 `ROCBLAS_USE_HIPBLASLT=0`。
- **结果**（A→B→B→A，ub 4096）：

  | 测试 | 修正前 | 修正后 | 变化 |
  |---|---|---|---|
  | pp512 | 918 | 1000 | +9% |
  | pp2048 | 1135 | 1538 | +36% |
  | pp8192 | 1248 | 1578 | +26% |
  | pp2048，上下文 8192 | 988 | 1335 | +35% |
  | pp8192，上下文 8192 | 1083 | 1364 | +26% |
  | decode（上下文 0 / 8192） | 79.5 / 74.0 | 79.8 / 74.2 | 不变 |

- **正确性**：prefill 路径（`-c 512 -b 2048 -ub 2048`，8 块）与原 rocBLAS kernel 相比，平均 KLD 0.000002，top-1 100%。这与只改 ubatch 大小、仍用原 kernel 时的差异相同（0.000002，top-1 99.95%）。

### 5. prefill 速度随批次大小的变化，以及反量化修正

llama-bench 中 pp512 是一个 512-token 批次，pp2048 是一个 2048-token 批次，pp8192 是两个 4096-token 批次（ub 4096）。按 token 平均的 GPU 时间（hipBLASLt 修正后、反量化修正前，rocprofv3）：

| 每 token 开销 | pp512 | pp2048 | pp8192 | 规律 |
|---|---|---|---|---|
| 权重反量化为 f16 | 232 µs | 59 µs | 31 µs | 每批次固定约 120 ms |
| GEMM | 581 µs | 408 µs | 392 µs | 批次小时效率低（约 84 → 124 TFLOPS） |
| DeltaNet | 63 µs | 64 µs | 64 µs | 每 token 恒定 |
| flash attention | 11 µs | 14 µs | 47 µs | 随上下文增长 |
| 其他 | 92 µs | 83 µs | 84 µs | 每 token 恒定 |

- **MMQ（直接用量化权重做矩阵乘）不划算**：强制所有批次走 MMQ 后 prefill 降到约 845 token/s。MMQ 在 gfx90a 上各类型只有 40–67 TOPS，每种类型、包括 512-token 批次都比反量化 + hipBLASLt 慢（只有 IQ4_XS 在 512 时快约 2%）。因此保留 llama.cpp 原来的选择（CDNA2 上 ne11 ≤ 128 才用 MMQ）。
- **反量化 kernel 的问题**：原来每个 256 值的超块用一个 32 线程的 block（Q2_K 为 64），每线程分散写 2 字节，wave64 一半的 lane 空闲，只有约 0.5 TB/s。
- **修正（`convert.cu`）**：在 HIP 上，IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS/Q4_K/Q2_K 改用 256 线程 block，每组线程用原来的逐线程公式把一个超块展开到 LDS，再用 16 字节写回。输出逐位相同。`GGML_CUDA_DEQUANT_LDS=0` 恢复原来的启动方式。
- **结果**（A→B→B→A）：每批次反量化 119 → 66 ms（约 0.9 TB/s）。pp512 1006 → 1119（+11%），pp2048 1544 → 1597（+3.4%），pp8192 1583 → 1608（+1.6%），decode 不变。每个 chunk 的困惑度完全相同，test-backend-ops MUL_MAT 2490/2490 通过。

### 6. 功耗上限对 prefill 上限的修正

前面的 MFMA 峰值（171 TFLOPS）和 GEMM 微基准用的是全零或常数操作数，功耗低，频率能保持在 1700 MHz。换成随机数据后：

| 测试 | 零/常数数据 | 随机数据 |
|---|---|---|
| MFMA FP16 循环（纯寄存器） | 173 TFLOPS | 150.6 TFLOPS |
| hipBLASLt 默认 kernel，实际 prefill 形状合计 | 约 146 TFLOPS | 约 122 TFLOPS |
| hipBLASLt 每个形状的最优 kernel | 约 154 TFLOPS | 约 128 TFLOPS |

原因是显卡功耗上限 300 W（`power1_cap` 已等于 `power1_cap_max`，无法再调高），真实数据下功耗更高，频率被压到约 1600 MHz 以下。因此：

- prefill 的物理上限应按约 150 TFLOPS 计算：50.5 GFLOP ÷ 150.6 TFLOPS ≈ **2980 token/s**（不是 3390）。
- 用库里最好的 GEMM kernel（约 128 TFLOPS）、其余开销为零时约 **2600 token/s**。
- 模型里实测的 GEMM 约 124 TFLOPS，已经接近库 kernel 在真实数据下的水平，GEMM 部分剩余空间很小。

### 7. 卷积输入拼接、卷积与 GEMM 逐形状选核

- **拼接（`concat.cu`）**：DeltaNet 层在卷积前把卷积状态和转置后的激活拼成按通道排列的输入，通用的非连续拼接 kernel 以 C 个元素的步长读取，每层每批次 1.89 ms。改为 64×64 的 LDS 分块转置后为 0.74 ms。`GGML_CUDA_CONCAT_TILE=0` 关闭。
- **长序列卷积（`ssm-conv.cu`）**：同样改为 64×64 LDS 分块，0.87 → 0.67 ms。`GGML_CUDA_SSM_CONV_TILE=0` 关闭。两者的逐元素计算和顺序不变，输出逐位相同。
- **GEMM 逐形状选核（`ggml-cuda.cu` 的 `ggml_cuda_hip_gemm_tuned`）**：n ≥ 2048 的大 GEMM 第一次遇到某个（m、k、n 档位）时，把一小组已知较好的 hipBLASLt solution index（只取库报告为对该问题有效的）与默认选择各计时一次，缓存最快的；候选不比默认快 2% 以上就保留默认。每个形状每个进程一次，约 0.1–0.2 s。`GGML_CUDA_GEMM_TUNE=0` 关闭。不同 kernel 只有 fp32 求和顺序的差别；因为按计时选择，不同进程之间 prefill 的 logits 可能有这个量级的差别。
- **结果**（A→B→B→A，ub 4096，三项一起）：pp2048 1601 → 1715（+7%），pp8192 1610 → 1687（+5%），pp512 不变，decode 不变（80.0 / 74.5）。prefill 路径 KLD 0.000002、top-1 99.95%，与噪声底线相同。

修正后 pp8192 每 token 约 0.585 ms：GEMM 约 375 µs，DeltaNet 65 µs，flash attention 47 µs，SwiGLU 16 µs，f32→f16 转换 15 µs，反量化 16 µs，拼接 + 卷积 16 µs，其余约 35 µs。

### 8. DeltaNet prefill 的分块 MFMA kernel（`gated-delta-net-mfma.cu`）

原来的 prefill kernel（`gated_delta_net_cuda`）逐 token 递推，每层每个 4096-token 批次 5.5 ms，占 pp8192 的约 11%。代码里已有的分块 MMA 版本只支持 32-lane 的 warp（NVIDIA、RDNA3.5）。新 kernel 是为 CDNA2 的 wave64 和 fp32 MFMA 重新写的：

- **算法**：每 16 个 token 一块。块内的递推改写成 (I + A)·δ = β·(v − G·K·S₀)，其中 A 是严格下三角；输出 = G·Q·S₀ + P·δ；状态 S ← G_C·S₀ + Kᵀ·(δ·G_C/G_t)。K·S₀、Q·S₀、K·Kᵀ、Q·Kᵀ、P·δ 和状态更新都用 `mfma_f32_16x16x4f32`，三角方程用前代法求解（每个 lane 一列）。全程 fp32，没有 bf16 拆分。
- **布局**：一个 block 负责一个 head 的 64 个 value 列，4 个 wave 各管 16 列，把自己那 128×16 的状态一直放在寄存器里。乘积的深度下标拆成 i = 16·it + 4·k + m，这样 B 操作数正好就是状态 tile 的寄存器，状态完全不经过 LDS。
- **调优中学到的硬件行为**（都是实测）：
  - 这个硬件上一个 wave 大约每 10 个周期才执行一条普通指令（一条 fp32 MFMA 约 35 个周期），所以关键路径上的**指令条数**才是成本，而不是 MFMA 本身：把全部 MFMA 换成普通乘加只省 5%。
  - 对寄存器数组的循环必须完全展开。留下真正的循环时，编译器用 `s_set_gpr_idx` 动态索引寄存器，每个元素多出好几条指令（这一项单独占了 25%）。
  - 对越界 token 的处理不要用“先清零、再条件加载”的写法：对加载结果做 select 会迫使 wave 立刻等待这次全局加载。改成钳位下标的无条件加载，使用时再屏蔽。
  - 全局加载提前一个块发出；K/Q 的 LDS 副本按 lane 排列（一次 16 字节读取供 4 条 MFMA 使用）并按块的奇偶双缓冲（省一个 barrier）；指针/屏蔽的 64 位运算只留在最后一个不完整块的路径上。
  - `hip.h` 里的 `CDNA` 宏只在 device 编译阶段有定义，host 侧的启动函数不能放在 `#if defined(CDNA)` 里。
- **适用条件**：CDNA2、标量 gate（非 KDA）、head 维度 128、只输出一个状态槽（K ≤ 1）、token 数 ≥ 32、q/k 行 16 字节对齐。不满足时自动回到原来的逐 token kernel。`GGML_CUDA_GDN_MFMA=0` 关闭。
- **结果**：独立基准中每次调用 5.5 → 2.17 ms，与双精度 CPU 参考的相对误差 2.5e-7。模型里（A→B→B→A，ub 4096）：pp512 1136 → 1175（+3.4%），pp2048 1720 → 1842（+7.1%），pp8192 1693 → 1821（+7.5%），decode 不变（80.6 / 74.8）。prefill 路径 KLD 0.000002、top-1 99.95%，与噪声底线相同。`test-backend-ops -o GATED_DELTA_NET` 48/48 通过（新增了 128 维、长序列的用例）。
- **还可以做的**：kernel 每块约 1.4 万个周期，MFMA 只占其中约 4 千；前代法约 2.3 千个周期只在一个 wave 上跑。独立基准和 CPU 参考在 `mi210-tools/peak-bench/gdn_mfma_bench.hip`。

### 9. prefill flash attention（`fattn-mma-f16.cuh` 的 CDNA 快速路径 + GQA 6 的分发）

pp8192 时 `flash_attn_ext_f16<256,256,8,8>` 每次调用 12.1 ms（约 34 TFLOPS），每 token 47 µs。四个方向并行分析（配置、汇编、数据流、另写 kernel），每条建议再由独立的 agent 复测，结论是 MFMA 只占这个 kernel 的 15–25%，其余是三类浪费：

- **分发**：GQA 比为 6 时走 ncols2=8 的实例，每 8 个 head 列有 2 列是补零，25% 的 MFMA、softmax 和 K/V 装载都白做。现在 ≥ 64 个 token 的批次改走已有的 ncols2=2 实例（32 个 token × 2 个 head，K/V 每个 KV head 装载 3 次）。单独这一项 12.1 → 9.1 ms。`GGML_CUDA_FATTN_GQA6_MIN_BATCH` 设最小批次，0 关闭。
- **装载延迟**：原来每次迭代有 9 次全局装载，每次后面都紧跟一个 `s_waitcnt vmcnt(0)`，wave 被串行地卡住。消融实验表明代价在延迟而不在带宽（把所有装载指向同一批行只省 0.4–0.7 ms，去掉装载省 3 ms）。现在一个 tile 的装载一起发出，并且提前一个阶段：本次迭代的 K 在上一次迭代的 VKQ 阶段装载并留在寄存器里，V 和 mask 在迭代开头发出、KQ 的 MFMA 之后才写入 LDS。gfx90a 上 `s_barrier` 之前只需要 `lgkmcnt(0)`，全局装载可以跨 barrier 保持在途。
- **V 操作数的拼装**：原来每条 VKQ 的 MFMA 要 4 次 `ds_read_u16` + 2 次 `v_perm_b32`。现在 V 在写入 LDS 时就转置好（8 字节单元 (r4, d) = V[4·r4+0..3][d]），一次 8 字节读取就是一个 A 操作数。
- **softmax**：exp 换成单条 `v_exp_f32`（`exp2(x·log2e)`，相对误差约 1e-6，远小于随后转成 f16 的精度）；wave 内没有任何最大值变化时跳过对 16 个累加器 tile 的重新缩放。

调优中学到的硬件行为：

- **LDS bank 冲突是一阶效应。** 完全相同的指令，只改 V 在 LDS 里的布局，相差 2.3 ms：慢的布局让一次写入的 64 个 lane 只落在 32 个 dword bank 中的 4 个上。现在每 16 个 d 的 128 字节补到 144 字节，使各 lane 的地址在 bank 上均匀分布。规则：每条 `ds_read`/`ds_write` 的（dword 地址 mod 32）要在 lane 间均匀分布。
- 指令条数不能单独预测时间：朴素的转置布局每次迭代少约 250 条指令，却慢 15%。
- LDS 的 64 KB 是按 CU 算的，不是按 block；这个 kernel 用 255 个 VGPR（每 SIMD 2 个 wave），每 CU 正好一个 512 线程的 block。调配置表（线程数、nbatch_fa、K2/V2、Q 是否放寄存器）都不如现状；`nbatch_combine=128` 会让 8x8 实例挂死。
- 另写一个专用 kernel（3 个计算 wave + 1 个装载 wave、32x32x8 的 MFMA、惰性 softmax）的独立原型是 4.8 ms，只比改进后的现有 kernel 快 0.9 ms，但要重做 mask、KV_max、sinks、kv_rows 等全部通用性，没有采用。原型在服务器的 `peak/fa-alt/fa_proto2.hip`。

适用范围：CDNA2 上 DKQ=DV=256、512 线程、ncols=64 且 ncols2>1 的实例（8x8、16x4、32x2），其它实例、decode、稀疏和 V_is_K_view 路径不变。`-DGGML_CUDA_FATTN_MMA_CDNA_FAST=0` 只编译通用路径。

结果（A→B→B→A，ub 4096）：

| 测试 | 之前（token/s） | 之后（token/s） | 变化 |
|---|---|---|---|
| pp512 | 1170 | 1176 | +0.5% |
| pp2048 | 1835 | 1848 | +0.7% |
| pp8192 | 1808 | 1895 | +4.8% |
| pp16384 | 1668 | 1841 | +10.4% |
| decode（深度 0） | 79.9 | 79.4 | 不变（噪声内） |

- kernel 每次调用 12.1 → 5.7 ms（412 GFLOP，约 73 TFLOPS）；mask 扫描 kernel 从 0.09 升到 0.26 ms（32 行一个 tile，block 数变少），已计入上面的结果。
- 正确性：prefill 路径 KLD 0.000002、top-1 99.95%（噪声底线）；c=8192 的 perplexity 3.5638 → 3.5636；快速路径与通用路径逐张量比较 NMSE ≤ 2.3e-9；`test-backend-ops -o FLASH_ATTN_EXT` 4065/4065（新增 5 个 GQA 6 prefill 用例）。独立评审（同步、适用条件、数值三个角度，逐条复核）没有发现缺陷。
- 注意：kv_rows（索引表）在 `ggml_cuda_get_best_fattn_kernel` 里目前只对 NVIDIA 开放，所以快速路径里的 `if (indices)` 分支在 CDNA 上到不了；它们在放开该限制的私有构建里验证过一次（仅 head 256）。
- 还剩：kernel 的 MFMA 下限约 2.75 ms（150 TFLOPS）；剩下的是每次迭代 4 个 barrier、约 275 条普通指令和 LDS 流量。

### 10. prefill 小算子融合（四项，后端内模式匹配，不改图的语义）

flash attention 之后，pp8192 每 token 562 µs 里 GEMM 和注意力以外的小算子还有约 68 µs。四项融合各自先在私有构建里做原型，每项由两个独立的 agent 复核（一个找正确性问题，一个重测速度），再合并成一个补丁并整体复核两轮。

| 融合 | 做法 | 每 token 节省（pp8192） | 开关（设 0 关闭） |
|---|---|---|---|
| 卷积输入 | `CONCAT → SSM_CONV+SILU`：不再生成转置后的 concat 缓冲，`ssm_conv4_silu_stream_f32` 直接在按 token 排列的 qkv GEMM 输出上做 4 行模板计算；`conv_fuse_prep_f32` 只写后面还会被读到的部分（状态列、尾部几列、每个分块前的 3 行） | 约 11.8 µs | `GGML_CUDA_CONV_FUSE` |
| SwiGLU | `GLU(SWIGLU) → MUL_MAT`：`swiglu_f32_f16_kernel` 把 `silu(gate)·up` 直接写成 ffn_down GEMM 的 f16 输入，省掉 f32 写出和随后的转换 | 约 8.2 µs | `GGML_CUDA_SWIGLU_F16` |
| f16 副本共享 | 同一个 F32 激活被多个 GEMM 读取时（q/k/v、gate/up、qkv/z 等）只转换一次；融合的 rms_norm·weight kernel 顺带写出 f16 副本 | 约 7.1 µs（其中 rms_norm 写出约 2.7） | `GGML_CUDA_F16_SHARE`、`GGML_CUDA_RMS_NORM_F16` |
| 残差加法 | `MUL_MAT → ADD(residual, mm)`：GEMM 以 beta=1 直接累加到残差上；与 SwiGLU 组合成三节点融合 | 约 3.6 µs | `GGML_CUDA_GEMM_ADD_FUSION` |

四项相加 30.7 µs，合在一起实测 28.5 µs。gate/up GEMM 的输出没有改成 f16（那会损失精度）。

实现要点：

- **统一的路由判断。** `ggml_cuda_mul_mat_get_route()` 是 MUL_MAT 选 kernel 族的唯一位置，`ggml_cuda_mul_mat` 自己按它分发，三项融合都通过 `ggml_cuda_mul_mat_cublas_gemm_type()` 询问，不存在各自复制一份分发条件的问题。
- **残差加法需要改图里的操作数顺序。** 原图是 `add(cur, residual)`，ggml-alloc 把和放在 GEMM 输出上，折叠就得先拷贝残差。`qwen35.cpp` 在 n_tokens ≥ 256 时改成 `add(residual, cur)`，和就落在残差上，GEMM 直接累加。条件化是因为无条件交换会改变单 token 的缓冲布局，让一层全注意力丢掉 `fattn_dec_gate` 融合。两种形式在最后一层的节点顺序不同，所以大批次之后跟一个小的多 token 批次时调度器会重新 reserve 一次（已验证结果正确，pp4196 为 1918 对 1805）；`qwen35.cpp` 里的注释写明了这一点。
- **f16 副本缓存的失效规则。** 每个上下文两个专用缓冲；条目按张量指针、数据范围、形状、类型和 stream 匹配；每次图计算开始时全部清空，执行任何输出字节与源重叠的节点之前清除对应条目（融合组内每个非 view 成员都算，包括 beta=1 累加写入的 ADD 节点）；并发 stream 区域内不使用缓存。
- **卷积融合的适用条件。** F32、d_conv = 4、无 bias、n_t ≥ 256；concat 的全部引用必须是这个 SSM_CONV 加上若干只读尾部列的 view，并且引用数要等于图的 use count（否则另一个调度分片里的读者会读到没有生成的缓冲）；concat 与 SSM_CONV 之间不能有别的 CONCAT；输出要么与 GEMM 输出完全重合，要么完全不相交。分块长度随批次变化（n_t/16 取整到 32 的倍数，32–256）：固定 256 时 256–400 token 的批次反而比不融合慢约 1%。
- **bit 一致。** 新的卷积 kernel 关掉了 fma 合并（`#pragma clang fp contract(off)`），因为旧 kernel 编译出来是分开的乘和加；SwiGLU kernel 保留原来的 expf 和除法。
- f16 副本共享和 rms_norm 的 f16 输出目前只在 HIP 构建里启用（非 HIP 构建没有编译验证过）；其余融合本身就在 `GGML_USE_HIP` 内。`GGML_CUDA_FUSION_STATS=1` 在上下文销毁时打印各融合的计数，`GGML_CUDA_F16_SHARE_VERIFY=1`（配合 `GGML_CUDA_DISABLE_GRAPHS=1`）在每次复用时按位复核。

结果（独立从头构建，各臂交替运行，ub 4096；“之前”为同一构建关掉全部开关，与上一版一致）：

| 测试 | 之前（token/s） | 之后（token/s） | 变化 |
|---|---|---|---|
| pp256 | 919 | 944 | +2.8% |
| pp512 | 1180 | 1222 | +3.6% |
| pp1024 | 1537 | 1618 | +5.3% |
| pp2048 | 1850 | 1952 | +5.5% |
| pp8192 | 1889 | 2000 | +5.9% |
| pp16384 | 1834 | 1938 | +5.7% |
| decode（深度 0） | 79.8 | 79.9 | 不变（噪声内） |

- 正确性：固定 GEMM 选核（`GGML_CUDA_GEMM_TUNE=0`）时，`llama-perplexity --kl-divergence-base` 的输出在“全开 / 全关 / 上一版”之间 md5 完全相同，覆盖 c512、c4096、c2999、c300、c256、c400、跨阈值的 256+44 和 4096+100、以及 `-ngl 40` 部分卸载；对参考的 KLD 0.000002、top-1 99.951%；c=8192 的 perplexity 3.5637 不变；贪心生成逐字节相同。
- decode 的 kernel 名称和调用次数与上一版完全相同（60 个名称、82409 次调用）。
- 测试：`test-backend-ops` 新增 `SSM_CONV_CONCAT_SILU`（22 例）和 `GEMM_FUSION`（45 例），用计数器确认每个用例确实走到融合 kernel；其中 `concat_between=1` 是原型里一个静默错误（两个 CONCAT 之间计划被清掉）的回归用例，`res-reused` 在去掉组后失效逻辑的构建上会失败。另有 195 个对抗性合成图场景（对 CPU 后端、对关闭开关逐字节比较）全部通过。全量 `test-backend-ops` 只有与本改动无关、上一版同样存在的 `TOPK_MOE` 失败。
- 过程中发现但没有在这里处理的两个问题：
  - 逐形状的 GEMM 选核对 5120×6144 这个形状不稳定，约 40% 的进程会选到稳态慢 11% 的 kernel（突发计时里两者相同），pp8192 因此呈双峰，相差约 0.8%。
  - 2D 权重 × 3D（广播）src1 的批量 cuBLAS 路径在同一进程先跑过其它图之后会出现 GPU 访存错误，与本改动无关，未修。

### 11. DeltaNet 第二轮（两段式 WY 递推 + 波分工的串行 kernel，以及相邻小算子）

第三轮的分块 kernel（`gdn_mfma_kernel`）每次调用 2.4 ms（pp8192 每 token 28 µs）。四个方向并行做原型、各由两个独立 agent 复核，其中三个有效，合并后又整体复核两轮。

- **两段式（WY）递推**，新文件 `gated-delta-net-wy.cu`。
  - `gdn_wy_prep_kernel` 完全并行（每个序列 × key head × 16 token 分块一个 wave）：算出块内的 T = (I + A)^-1（fp64 前代）、P 和各个门控向量。
  - `gdn_wy_kernel` 是串行部分，每个分块只剩和状态相关的 delta、输出和状态更新三组 MFMA。
  - 一次调用从 2406 µs 降到 175 + 720 µs。
- **串行 kernel 按波分工。** 8 个 wave：4 个只做 MFMA（每个 SIMD 一个，各持 128×16 的状态切片），4 个负责把下一分块的 K、Q、K^T 和 prep 的结果搬进双缓冲的 LDS。gfx90a 上实测：同一个 wave 在相关的 MFMA 之间插入普通指令不会被隐藏（MFMA 一步从 32.8 周期涨到 8 条普通指令时的 48.8），而同一 SIMD 上另一个只做普通指令的 wave 可以完全重叠。这一项比 4 波版本快约 25%（876 → 660 µs，独立测试）。
- **门控的数值。** 块内的衰减指数 s_tj 按“恰好覆盖的那些 gate 的和”来算（对第 j 列从 u = j+1 累加到 t，加到块尾就是 G_C/G_j），不用两个大前缀相减；低于 -1e30 的 gate（包括 -inf）按 -1e30 处理，相当于精确的重置。指数本身不截断，所以正的 gate 和 NaN 与其它 kernel 行为一致。对 fp64 参考（77 种形状和门控模式、最长 64K token、模型实际数据）：输出 / 末状态 NMSE 约 5e-14 / 1e-14（原分块 kernel 8e-14 / 4e-14），模型数据 5e-13（原 6.6e-13）；1/16 的 gate 为 -30 到 -inf 时仍然有限且精度不变（原分块 kernel 最差 0.26，-inf 时 NaN）；只有“几乎平行的 key、beta = 1、无衰减”这种病态情形是原分块 kernel 误差的 1.2 倍。
- **相邻小算子。**
  - 输出的 gated RMS norm（rms_norm·w → silu(z)·normed → 转 f16）合成一个 `rms_norm_gated_f16_kernel`，直接写 ssm_out GEMM 的 f16 输入；为此 graph_optimize 在多 token 且融合能触发时让 x 存活到 z 的 GEMM 之后。
  - q/k 的 L2 norm 改成每 256 线程 block 处理 8 行（`rms_norm_f32_rows32`，≥ 8192 行才用，按位相同）。
  - 这两项与原路径按位相同。
- **试过不划算的方向**：更细的 value 切分和按序列分段的并行扫描（一个 CU 同时跑两个 block 只有 1.33 倍吞吐，多出来的工作抵不过）；完整的 FLA 三段式（每个分块的状态要写出读回，流量比整个 kernel 还大）。
- **开关**（设 0 关闭）：`GGML_CUDA_GDN_WY`（回到原分块 kernel）、`GGML_CUDA_GNORM_F16`、`GGML_CUDA_RMS_NORM_ROWS`；`GGML_CUDA_GDN_MFMA=0` 仍然关掉所有分块路径。新 kernel 用到的编译器内建函数和 `-mllvm -amdgpu-mfma-vgpr-form` 选项都先检测，旧编译器上自动退回原 kernel。

结果（同一构建交替运行 4 轮，ub 4096；“之前”为上一版主线构建）：

| 测试 | 之前（token/s） | 之后（token/s） | 变化 |
|---|---|---|---|
| pp512 | 1220 | 1255 | +2.9% |
| pp2048 | 1951 | 2024 | +3.7% |
| pp8192 | 1988 | 2075 | +4.4% |
| pp8192（`GGML_CUDA_GEMM_TUNE=0`） | 1938 | 2015 | +4.0% |
| decode（深度 0） | 80.0 | 80.2 | 不变（噪声内） |

- 每 token 的 GPU 时间：DeltaNet 递推 28.3 → 10.4 µs，相邻小算子约 -5 µs；归因（固定选核）为递推 +2.7%、gated norm +1.0%、L2 norm +0.1%。
- 正确性：关掉新路径时与上一版按位相同（md5，固定选核）；打开时对参考的 KLD 0.000001、top-1 100%，c=8192 perplexity 3.5637 不变；decode 以及 2–4 token 批次的 kernel 名称和调用次数与上一版相同。
- 测试：`test-backend-ops` 新增 GATED_DELTA_NET 用例（分块边界、多序列、rq3 > 1、置换布局）和 20 个 GATED_DELTA_NET_ADV 对抗门控用例，以及 rows32 和 gated norm 的用例；另有 126 个 fp64 对照的对抗性场景（包括正 gate、NaN、±inf、跨块的大 gate）全部通过。
- 原分块 kernel（关掉 `GGML_CUDA_GDN_WY` 时使用）对极大的负 gate 仍然会失真或出 NaN，这里没有改它。

### 12. GEMM 选核第二轮（固定 kernel 表 + FFN gate/up 合并）

GEMM 占 pp8192 每 token 约 410 µs（共 519 µs）。四个方向并行做原型、各由两个独立 agent 复核；有效的两项合并后又经过两轮整体复核（三个和两个独立 agent），第一轮发现的问题全部修掉后才合入。

- **自写 GEMM 不划算。** hipBLASLt 的 kernel 每个时钟已达到纯 MFMA 速率的 92–95%；在 300 W 功耗上限下，最好的主循环约 128 TFLOPS，库在模型的实际形状上已经是 125–131 TFLOPS。原型自写 kernel 慢 29–61%。hipBLASLt 自己的 API 也没有比 rocBLAS solution index 更快的 kernel（LT 的 index = rocBLAS index − 1）。
- **空间在选核。** 库的默认选择比最好的有效 solution 慢 3–41%，n = 512 的 K 很大、beta = 1 的形状最差；旧的逐形状选核只在 n ≥ 2048 时运行，而且 5120×6144 beta=1 在约 40% 的进程里选到稳态慢 11% 的 kernel（突发计时里两者相同）。
- **固定 kernel 表（`ggml-cuda.cu` 的 `ggml_cuda_hip_gemm_pins`）。** 对模型的 8 个 f16 GEMM 形状，在 n = 512/724/1024/1448/2048/3000/4096 上做了全候选的稳态扫描（随机数据、背靠背重复），按精确的（m、k、n、beta）记下赢家，共 56 项，另加 6 项合并 gate/up 的形状。
  - 表项直接使用，不调用 `rocblas_gemm_ex_get_solutions`：这个查询每个形状约 73 ms，进程里第一次约 590 ms，原型的冷启动变慢（512 token 的首个请求慢 0.8 s）全部来自它。库不接受的 index 在 rocBLAS 里约 6 µs 就返回失败、不写 C（40 个 index × 3 个形状实测），这时该形状退回库默认并缓存。
  - 只在这台机器的组合上使用：rocBLAS 版本串完全等于 `5.2.0.dabb6df2b9`、加载的 hipBLASLt 为 `libhipblaslt.so.1.2.70201`、设备名为 “AMD Instinct MI210” 且 104 个 CU。其它情况完全走旧的选核（包括它的有效性查询），也不合并。
  - 表外的 n ≥ 2048 仍按旧的 n 档位做稳态计时，但不调用有效性查询，候选为旧的 10 个加上同形状在相邻表 n 上的表项（每个形状约 60 次调用）；n > 4096 用轻量计时（约 31 次调用）。表外的 n < 2048 用库默认，与之前相同。缓存的键包含设备号。
- **FFN gate/up 合并（`ggml_cuda_try_fuse_gate_up`）。** 两份权重反量化到同一块 f16 缓冲，一个 m = 2×17408 的 GEMM 同时算出 gate 和 up，带步长的 SwiGLU kernel（`swiglu_f32_f16_strided`）直接写 ffn_down 的 f16 输入，ffn_down 保留残差加法的 beta = 1 融合。
  - 只在表里有合并形状的 n 上合并。是否合并按模型内实测决定（同一构建、单一 n 的进程、轮流运行，每组 8–20 个进程，合并组的每个进程都要快于不合并组的每个进程）：512 +0.55%、724 +1.0%、1024 +0.2%、2048 +0.76%、3000 +0.27%、4096 +0.32% 合并；1448 慢 0.5%，不合并。只测 GEMM 本身的成对计时在 1448 上给出了相反的结论（功耗上限下的频率行为），所以不用它做决定。
  - 如果库在运行时拒绝了合并形状的 index，该 n 不再合并（只有固定 kernel 时合并才划算）。
  - 代价：内存池多保留约 0.7 GB 显存（ub 4096 实测 +714 MiB）。
- **试过不划算的方向**：在第二个 stream 上提前反量化下一个权重。原型只在 512–2048 token 上有 +2.5%，而且会提前转换图里稍后才写入的张量（本模型不会触发，但有复现用例）；修好之后叠加在前两项上，pp512 反而慢 4.9%（功耗上限：GEMM 被反量化挤占的时间等于它隐藏的时间），所以没有合入。
- **开关**：`GGML_CUDA_GEMM_TABLE=0` 精确回到旧的选核；`GGML_CUDA_GATE_UP_MERGE=0` 关闭合并（=2 在所有尺寸上强制合并，测试用）；`GGML_CUDA_GEMM_TUNE=0` 关闭所有选核（包括合并）。

结果（单一尺寸的进程，与上一版主线构建交替运行，每组 6–14 个进程，ub 4096）：

| 测试 | 之前（token/s） | 之后（token/s） | 变化 |
|---|---|---|---|
| pp512 | 1262 | 1612 | +27.8% |
| pp1024 | 1712 | 1883 | +10.0% |
| pp1448 | 1701 | 1932 | +13.5% |
| pp2048 | 2023 | 2074 | +2.5% |
| pp2500 | 2024 | 2071 | +2.3% |
| pp4096 | 2083 | 2136 | +2.5% |
| pp8192 | 2074 | 2125 | +2.5% |
| pp16384 | 2008 | 2058 | +2.5% |
| pp8192（ub 512） | 1238 | 1583 | +27.9% |
| pp8192（ub 8192） | 2069 | 2084 | +0.7% |
| decode tg128 | 79.9 | 79.8 | 不变（噪声内） |

- pp8192 不再呈双峰（8 个进程相差 0.4%）；上一版的 pp2500、pp3000 也是双峰，表里的“之前”是平均值。
- 新进程的首个请求在所有测过的尺寸上都比上一版快：pp512 734 vs 823 ms，pp1448 1193 vs 1264 ms，pp2500 2627 vs 3039 ms，pp4096 2322 vs 4031 ms，ub 8192 的 pp6000 4384 vs 5310 ms。
- 正确性：所有表项和合并形状对 fp32 参考逐一检查（紧凑和带填充的布局、beta 0/1），结果确定、不碰填充区；故障注入（错误的 rocBLAS/hipBLASLt 版本、其它设备名或 CU 数、拒绝 index、调用失败）下结果都正确，守卫不通过时行为与上一版相同。固定选核（`GGML_CUDA_GEMM_TUNE=0`）时，新开关全关以及默认设置的 logits 与上一版 md5 相同；默认设置对参考的 KLD 0.000001。
- 测试：`test-backend-ops` 新增真实 FFN 形状的合并用例、强制合并用例，以及一个真正走到 beta = 1 GEMM 的残差加法用例。
- 遗留：约 1/94 个默认设置的进程在 pp2048 上慢 0.3%（仍快于不合并和上一版的所有进程），在 profiler 下没有复现，原因未知。表项的 index 在运行时失败一次，该形状在本进程里就固定用库默认，不再重新计时（与旧选核相同）；故障注入下每隔一次调用失败时 pp2048 比上一版慢 5%，实际安装因为版本守卫不会遇到。表和合并决定只适用于这张卡、300 W 上限和这个模型的形状。

### 13. 剩余 prefill 方向

- 非 GEMM 部分：DeltaNet 串行 kernel 离 MFMA 下限还有约 1.3 倍、prep kernel 受 LDS 限制只有每 SIMD 2 个 wave（合计约 +0.5%）；还剩 6144 宽的单消费者 f32→f16 转换（约 1.8 µs/token）和约 0.9 TB/s 的流式 kernel 离带宽上限的差距。反量化离带宽下限（约 45 ms/批次）还差约 20 ms，主要影响短 prompt。
- GEMM 部分已接近库 kernel 在功耗上限下的水平（第 12 小节），选核也已固定；剩下的是反量化（约 16 µs/token）和 GEMM 之间的重叠，在功耗上限下基本不划算。

微基准源码在 `mi210-tools/peak-bench/`：`peak_bench.hip`（带宽、MFMA、方阵 GEMM）、`bound_bench.hip`（逐张量读取 graph、实际形状 GEMM 全候选扫描）、`gemm_sweep.hip`（不同 token 数下的选核对比）、`mfma_rand.hip`（零数据与随机数据下的 MFMA 峰值，显示功耗上限的影响）。编译：`hipcc -O3 --offload-arch=gfx90a -std=c++17 <file> -lrocblas`。

### 14. decode 差距复核与结论（prefill 优化完成后）

结论：在“同一 GGUF、不损失精度、不用 MTP”的约束下，decode 现实可达约 84–86 token/s，冒险做每层一个 megakernel 乐观约 87；90 没有实测数据支持，因此 decode 不再继续优化。越过 90 的现实办法是 MTP / 投机解码（见第十一节第 7 项）。

**逐节点计时**（主线 5d01ec5，graph 内 GPU 时间戳，上下文 0，80.5 token/s，12.43 ms/token）：每 token 601 个节点、读写 11.82 GB（1.46 TB/s 下 8.09 ms）。每个节点都跑到“1.6 µs + 字节 / 1.46 TB/s”的下限时合计 9.06 ms（约 110 token/s）；实测节点合计 11.99 ms，另有约 0.1 ms 图间空闲和约 0.3 ms 未归属的派发开销。

| 部分 | 节点数 | 实测 µs/token | 下限 | 差距 |
|---|---|---|---|---|
| 权重 GEMV（含 LM head） | 308 | 10330 | 8290 | 2040 |
| 　其中低比特（iq2/iq1，0.84 TB/s） | 44 | 1169 | 740 | 429 |
| rms_norm + q8 / swiglu | 162 | 587 | 269 | 318 |
| flash attention 解码 + gate | 16 | 271 | 38 | 233 |
| DeltaNet 解码（prep + out） | 96 | 639 | 426 | 213 |
| attention 预处理 | 16 | 157 | 27 | 130 |

- 每个 GEMV 比下限模型多 5–7 µs（启动序言、码本查表延迟、指令缓存冷启动、尾部不均、派发），融合的 ffn_gate+up 约 10 µs，IQ3_XXS 的 ffn_down 约 12 µs；LM head 已在下限（1.40 TB/s）。
- 上下文 8192 时多出的 1.1 ms 中 0.79 ms 来自 flash attention：每层 66 µs，只有 0.52 TB/s（下限 25 µs）。
- 计时补丁和全部 601 个节点的数据在服务器 `peak/dec-prof/`。

**各方向的估计**（四个角度并行分析，每项有微基准或原型支撑，再由独立审查去重；单项收益 1–2%，与约 1.7% 的测量噪声同量级）：

| 方向 | 预计节省 µs/token | 说明 |
|---|---|---|
| rms_norm+q8 定形 kernel（5120 列、640 线程、无除法） | 100–170 | 去掉 norm 实测省 0.52 ms，换成空 kernel 只省 1.0 µs/节点 |
| attention 预处理、短上下文 FA 解码、DeltaNet prep 减指令 | 130–220（估计） | 冷指令缓存下每条关键路径指令约 8–13 ns |
| 低比特 scale 改用 float 计算 | 30–60 | 相对误差约 3.5e-5，需 KLD 把关 |
| IQ3_XXS / 低比特 / IQ4 的解码改进 | 0–400 | 探索性，无具体方案 |
| 常驻 FFN kernel | 0–300，期望约 150 | 纯读原型省 0.46–0.62 ms，但块形状与生产的 104×1024 不同；需整网格常驻，模型内同步成本未测 |
| 每层一个 megakernel（与上一项二选一） | 0–600 | 纯读原型省 0.92–1.40 ms，风险更大 |
| 长上下文 split-KV MFMA FA | 8K 100–250，32K 300–690 | 对比的 tile 版尚未在同一测试程序里计时 |

合计中值约 0.7 ms（约 84–85 token/s），乐观端约 1.15 ms，到不了 90 所需的 1.34 ms。已排除、不要重试：限制 GEMV 网格到约 2 块/CU、logits 拷贝与 LM head 重叠、去掉上传后的同步、兄弟 GEMV 合并、AnyOrder launch、双流图、跨 kernel 的 L2 预取，以及第八节的全部条目。

### 15. MTP 投机解码：多 token 验证的 GEMV kernel

llama.cpp 已支持本模型自带的 MTP 头（`llama-server --spec-type draft-mtp --spec-draft-n-max K`）。直接打开时 decode 反而变慢：K=3 只有 56.6 token/s（不开为 78.7）。草稿接受率其实很高（K=3 时每轮产出：写代码 3.51 个 token、JSON 3.78、写故事 2.18），慢在验证：主模型一次处理 K+1 个 token 时，第五节的所有 decode GEMV 都只在 ne11 == 1 时启用，2–8 个 token 退回 llama.cpp 原来的 mul_mat_vec_q，只有 0.2–0.4 TB/s，一步 4 个 token 要 49 ms（1 个 token 14 ms）。

- **N 列 kernel。** 每个 decode 模块各加一套模板 kernel，列数 N 为编译期参数（实例化 2–6），权重读一次供 N 个 token 使用：IQ3_S/IQ3_XXS（单个、残差、SwiGLU 含 S/XXS 混合、qkv+z 多任务）、IQ4_XS/Q4_K（含 LM head 和 1024 行的 K/V）、低比特（IQ2_S/XS/XXS、Q2_K、IQ1_M，普通和残差）、IQ3+IQ4 混合配对。单 token kernel 不变。
- **逐列与单 token 按位相同。** 每列用与 N=1 相同的 q8 激活和相同的逐行求和树，结果与把这一列单独送进单 token 路径逐位相同；`test-backend-ops` 和模型内（`GGML_MMVQ_NC_VERIFY`）共检查上千次调用，0 个不一致。
- **融合在 2–NC_MAX 个 token 时也生效**：MUL_MAT+ADD、gate/up SwiGLU、qkv+z 多任务、混合配对；qwen35 在这些尺寸下调整 qkv/z 的顺序，让配对能形成。
- **q8 激活缓存修正。** 缓存现在也服务 N 列调用；任何节点或融合组写入某个缓存条目的源字节时，该条目会被丢弃。这顺带修好了一个原有问题：N=1 时非矩阵乘的原地写入之后，缓存里可能是过期的激活。新增回归用例 res-inplace / res-reused。
- **开关**：`GGML_MMVQ_NC=0` 精确回到原来的 kernel 和输出；`GGML_MMVQ_NC_MAX`（本节时默认 4、最大 6，第 16 小节起默认和最大都是 16）是最大批次；`GGML_MMVQ_NC_IQ3` / `_IQ4K` / `_LOWBIT` / `_MIX=0` 分别关掉一类。
- 2–4 个 token 的短 prompt 分块也会走新 kernel，因此不开 MTP 时部分 prompt 的贪心输出会与之前不同（每列与单 token 结果一致，更接近逐 token 计算）；需要与之前逐字节相同时设 `GGML_MMVQ_NC=0`。

结果（llama-server，4 个 prompt 各 384 token，贪心，token/s）：

| 草稿数 K | 之前 | 之后 | 写代码 | 写故事 | JSON |
|---|---|---|---|---|---|
| 0（不开 MTP） | 79.1 | 79.5 | 79.5 | 79.5 | 79.4 |
| 1 | 51.2 | 88.8 | 92.0 | 81.4 | 94.5 |
| 2 | 54.9 | 99.1 | 109.7 | 80.5 | 115.5 |
| 3 | 56.4 | 99.0 | 115.2 | 73.6 | 124.7 |

- 验证一步（模型内逐节点计时）：N=2 36.8 → 21.4 ms，N=3 43.9 → 24.3，N=4 49.2 → 27.5。K=4、5 需要 `GGML_MMVQ_NC_MAX=6`（94.2、87.3 token/s），不如 K=2–3。
- 单 token decode、prefill 和 N=1 的 logits 不变。
- **还没做到的**：
  - 开 MTP 时写代码和写故事的输出仍与不开时不同（JSON、翻译相同）。第一个出现差异的是 DeltaNet 层里 BF16 的 ssm_alpha/ssm_beta 小矩阵乘，之后是 DeltaNet 递推、gated norm 和带 gate 的 flash attention：这些在 N=1 时走融合 kernel，N>1 时走通用图。
  - N=4 的 GEMV 是单 token 的 1.5–2.4 倍（目标 1.3 倍）：多列时这些 kernel 受普通指令吞吐限制；混合配对已在 NC=4 用 int8 MFMA，其余类可照此改写。

### 16. MTP 第二轮：int8 MFMA 的 N 列 GEMV、逐 token 精确的 DeltaNet 与注意力

第 15 小节之后，验证一步仍然贵：4 个 token 要 24 ms（模型内逐节点合计，单 token 12.5 ms）。其中 GEMV 是单 token 的 1.5–2.4 倍，DeltaNet 的非矩阵乘部分在 N>1 时走通用图，约 4.9 ms（N=1 的融合 kernel 只要 0.64 ms）。8 个 token 以上退回原来的 mul_mat_vec_q / MMQ，一步 57–67 ms，所以更大的 K 和 n-gram 草稿都不划算。另外开 MTP 时写代码、写故事的输出与不开时不同。

这一轮分六部分，每列（每个 token）的结果都与单 token 路径逐位相同：

- **四类 GEMV 改用 int8 MFMA**（`v_mfma_i32_4x4x4i8`），N 最大 16。整数部分用 MFMA 算出与 N=1 dp4a 链相同的 int32，浮点部分（缩放、逐行求和树、残差、SwiGLU、q8）照搬 N=1 的运算顺序：
  - IQ3_S/IQ3_XXS：单个、残差、SwiGLU、qkv+z 多任务、K 分段的 ffn_down，N=4–16；
  - IQ4_XS/Q4_K：含 N 行的 LM head 和小行数的 K/V，N=3–16；
  - 低比特 IQ2_S/XS/XXS、Q2_K、IQ1_M：N=3–16；
  - IQ3+IQ4 混合配对：N=4–16；
  - N 较小时保留第 15 小节的 dp4a 版本（那里更快）。
- **DeltaNet（O1）**：一个序列 1–16 个 token 都走两个融合 launch（PREP、OUT），包括 BF16 的 ssm_alpha/beta、卷积窗口、递推和 MTP 回滚用的快照槽。
- **注意力（O2）**：一次 launch 处理 2–16 个 query 的 fattn-dec，带 gate 和 q8 输出；KV 超过 1024 时每个 query 走单 token 的通用调用，保证逐位相同。norm、SwiGLU 的 q8 产出也支持 N 行。
- **N > 8 被拒绝时的退路**：某个张量不被 N 列 kernel 接受时回到原来的 MMQ（不再触发 mul_mat_vec_q 最多 8 列的断言）。
- **开 MTP 时的 prompt 预填充**：llama.cpp 在 n_rs_seq > 0 时用逐 token 递推代替分块 WY kernel，最后几位与不开 MTP 不同，这是第一个分歧点。现在递推只负责写快照，输出由与不开 MTP 相同的分块 kernel 重算（`GGML_CUDA_GDN_KEEP_RS_EXACT`）。
- `GGML_MMVQ_NC_MAX` 默认改为 16：N 超过它就回到不精确的 MMQ。

结果：

验证一步（模型内逐节点合计，ms）：

| N | 之前 | 现在 |
|---|---|---|
| 1 | 12.50 | 12.49 |
| 2 | 18.90 | 14.66 |
| 4 | 23.99 | 17.19 |
| 8 | 67.29 | 23.56 |
| 16 | 57.24 | 37.25 |

llama-server，5 个 prompt（写代码、写故事、JSON 各 384 token，改代码 502，复述 153），贪心，token/s：

| 配置 | 写代码 | 写故事 | JSON | 改代码 | 复述 | 总体 |
|---|---|---|---|---|---|---|
| 不开投机 | 79.3 | 79.3 | 79.2 | 76.6 | 79.4 | 78.5 |
| 之前 MTP K=3 | 115.2 | 73.6 | 125.9 | 131.1 | 116.2 | 107.9 |
| MTP K=1 | 114.2 | 99.0 | 117.1 | 115.7 | 115.0 | 111.6 |
| MTP K=2 | 135.2 | 99.1 | 143.1 | 143.6 | 137.8 | 129.0 |
| **MTP K=3** | **153.6** | **97.2** | **167.4** | **171.2** | **157.1** | **142.8** |
| MTP K=4 / 5 / 7 | | | | 171.8 / 188.4 / 206.4 | | 127.9 / 128.2 / 125.1 |
| ngram-simple（n 12, m 15） | 78.8 | 78.1 | 79.5 | 137.7 | 162.1 | 94.1 |
| MTP K=3 + ngram-simple | 154.0 | 97.4 | 150.6 | 179.7 | 200.9 | 144.1 |

- **推荐** `--spec-type draft-mtp --spec-draft-n-max 3`。K=3 每秒 44.8 轮（之前 33.6）。K ≥ 4 时每轮变贵的幅度超过多猜中的 token，只有改代码这类高接受率的内容更快。以复制、改写为主的负载可再加 `--spec-type ngram-simple`。
- **所有投机配置的输出都与不开投机逐字节相同**，5 个 prompt 全部一致。
- **没有退步**：单 token decode、pp512/2048/8192 与之前相同（在噪声内），N=1 logits 不变。
- **不开投机时，写代码和写故事的输出与第 15 小节版本不同**：prompt 末尾 2–16 个 token 的分块现在走逐 token 精确的路径。把所有新开关设为关（见附录）可逐位回到之前的版本。
- **还没做到的**：
  - N=16 一步 37 ms，目标约 21 ms。各类 GEMV 的 N=4 / N=16 相对 N=1：IQ3 1.47 / 3.06，IQ4 1.28 / 2.82，低比特 1.58 / 4.56，混合配对 1.37 / 3.09。
  - KV 超过 1024 时精确注意力每层每轮多约 0.2 ms（`GGML_CUDA_FA_DEC_NQ_LONG=0` 用精度换速度）。
  - （第 17 小节已继续处理，现在约差 5%）开 MTP 时 prompt 预填充仍比不开 MTP 慢约 9%（10k token 的 prompt：约 1800 vs 1970 token/s，server、4096 token 分块；硬件上限约 2980）。原先慢 17%（约 1640）：快照递推对整块每个 token 逐个计算，每层每块约 3.8 ms。现在分块 kernel 先算前 n−K+1 个 token 得到最老的快照，递推只跑最后 K−1 个 token（`GGML_CUDA_GDN_KEEP_RS_TAIL`），输出不变。剩下的差距主要是草稿层要把整个 prompt 过一遍。
  - N 列注意力假设一个 KV cache 只有一个序列（`-np 1`）。

### 17. MTP 第三轮：prefill 的 CPU 开销、验证与草稿 kernel、单 token 小算子

目标是把开 / 不开 MTP 的 prefill 和 decode 都压到接近硬件能做到的程度。输出约束不变：开 MTP 的输出与不开逐字节相同，单 token logits 和 prefill 结果与之前逐位相同。

**prefill**（llama-server，`-ub 4096`，token/s；括号内为开 MTP K=3）：

| prompt | 第 16 小节 | 本节 |
|---|---|---|
| 约 0.9k token | 1410（1200） | 1435（1339） |
| 约 3k token | 1910（1590） | 1930（1835） |
| 约 10k token | 1970（1636） | 2010（1897） |

- **token embedding 查表放到 GPU。** llama.cpp 默认把输入层（token_embd 的 GET_ROWS）放在 CPU，它是每个计算图前面的一个 CPU split：4096 token 的块要 30–60 ms，GPU 在等；开 MTP 时草稿上下文每块还要再做一次。全部层都在一块 GPU 上时，输入层也放到这块 GPU（IQ2_S 的 token_embd 多占 0.4 GB 显存），GPU 反量化的行与 CPU 逐位相同。`LLAMA_INPUT_ON_GPU=0` 关闭。
- **快照递推只跑尾部，分块 kernel 只跑一遍。** 开 MTP 时（n_rs_seq > 0），每个 prompt 块原来对全部 token 逐个递推，只为写回滚快照（每层每块约 3.8 ms）。现在把对整块的分块 WY 调用在 16 token 对齐处拆成两次（分块边界与一次调用相同，输出和最终状态逐位相同），递推只从拆分点跑到块尾写快照。`GGML_CUDA_GDN_KEEP_RS_TAIL`、`GGML_CUDA_GDN_KEEP_RS_SPLIT` 设 0 分别退回。
- server 与 llama-bench 的差距只有 2.5–5%。第一次遇到 2048–4095 和 4096 以上这两个长度区间时，GEMM 选核各多花约 1.25 s（每个进程一次）。
- 开 MTP 时剩下约 5% 的差距：草稿层本身的计算，以及草稿上下文每块约 20–30 ms 的 CPU 准备（把 hidden state 拷回主机再上传）。要去掉后者，需要在 server 里让草稿与下一块的目标计算重叠。
- 物理上限约 2980 token/s（300 W 功耗限制下的真实算力），GEMM 已接近库 kernel 在该限制下的水平（第 6、12 小节）。

**MTP decode**（K=3，每轮约 22.2 → 20.7 ms）：

- **验证 GEMV**：
  - IQ3 的 N 列 MFMA kernel 新增第二种形式（MFMA 操作数互换，每个 lane 直接得到自己行的 4 列和，去掉 DPP 广播；buffer resource 取址，主循环 208 → 160 条指令），网格用满全部 CU，小行数和 N=2 按形状选 dp4a。
  - 低比特 ffn_down 每组 4 个单元。
  - attn_k + attn_v 在 N=2–16 时合成一次 launch。
  - 每列仍与单 token 路径逐位相同。验证一步（逐节点合计）N=4 17.2 → 16.3 ms，其中 IQ3 类 6.67 → 5.88 ms。
- **草稿一步 1.06 → 0.92 ms**：
  - 草稿层的 Q6_K 用新的单 token GEMV（同尺寸的无损 SoA 副本，约 1.23 TB/s，原版约 0.92）。
  - 每轮对已接受 token 的追赶解码只算 KV（0.77 → 0.17 ms）。
  - 草稿的 top-k（k ≤ 32、248k 列）用新的选择 kernel。
  - 草稿数值可以变化，只影响接受率；实测接受数与之前相同。
- **单 token 小算子**（逐位相同）：
  - 定形的 rms_norm+q8；
  - attention 预处理的 cos/sin 表；
  - FA 解码展开；
  - DeltaNet prep 提前取数。
  - 逐节点 −0.26 ms；端到端 +0.4–0.9%，在噪声边缘。
- **草稿词表截短**（`LLAMA_MTP_DRAFT_VOCAB=n`，默认关）：
  - 草稿 LM head 只算前 n 个 token id，其余为 −inf；输出不变，只影响接受率。
  - 5 个英文 / 代码 prompt：n=65536 时 +3%。
  - 中文 prompt：n=65536 时 124 → 72 token/s（常用汉字的 id 大多在 65536 以上），比不开 MTP 还慢；n=131072 时不亏也不赚。
  - 只适合纯英文 / 代码负载。
- **GPU 采样**（`-bs`、`--spec-draft-backend-sampling`）实测没有收益。

结果（llama-server，贪心，token/s；5 个 prompt：写代码、写故事、JSON、改代码、复述）：

| 配置 | 写代码 | 写故事 | JSON | 改代码 | 复述 | 总体 | 中文 4 个 prompt |
|---|---|---|---|---|---|---|---|
| 不开投机 | 79.5 | 79.5 | 79.4 | 78.1 | 79.7 | 79.1 | 79.4 |
| MTP K=3，第 16 小节 | 154.1 | 97.8 | 168.4 | 172.3 | 158.0 | 143.7 | 123.8 |
| **MTP K=3，本节** | **166.4** | **106.1** | **180.9** | **185.1** | **170.4** | **154.3–155.0** | **133.3** |
| MTP K=2 / K=4，本节 | | | | | | 138.8 / 135.9 | 124.9 / 110.8 |

- 所有投机配置的输出与不开投机逐字节相同（9 个 prompt）。逐位检查覆盖 test-backend-ops（全部 VERIFY 开关，0 不一致）、test-mtp-nq、各 ubatch 和快照数下的 logits md5。
- 单 token decode 与 prefill 不退步：tg128 +0.4%，pp512 / pp2048 在 ±0.1% 内。
- 新开关全部关掉时，与第 16 小节版本的输出、草稿数和速度完全相同。
- **还没做到的**：
  - N=4 时各类 GEMV 相对 N=1：IQ3 1.30、IQ4 1.29、低比特 1.54、混合配对 1.37（目标约 1.25）。IQ3 剩下的是 K 分段的 barrier 偏差和每步指令数；混合配对可改为每个 lane 持有自己的行。
  - 草稿与下一块 prompt 计算的重叠（见上）。
  - 逐节点每轮省了约 2.1 ms，端到端只少了约 1.5 ms，差额原因未查。

## 十四、vLLM 部署：官方 FP8 checkpoint 的单流 decode 与 MTP

前面各节针对 llama.cpp 和 IQ3_S GGUF。这一节是另一条线：用 vLLM 跑官方的 Qwen3.8-27B-FP8 checkpoint（自带 MTP 层），不重新量化，只做无损的权重重排。代码在本仓库的 `vllm-opt/` 目录，说明见 [vllm-opt/README.md](../vllm-opt/README.md)。运行环境是 vLLM 镜像 `2e7da1ad1c66`（vllm 0.30.0+rocm723）和单卡 MI210。镜像本身不做任何修改：插件目录以只读方式挂载，通过 `PYTHONPATH` 和 `usercustomize.py` 的 post-import hook 给 vLLM 打补丁；HIP 扩展用 `torch.utils.cpp_extension` 编译。

### 1. 上限与原版的瓶颈

- **每 token 要读的字节数**：FP8 权重 24.33 GB，bf16 LM head 2.54 GB，DeltaNet 状态 0.3 GB，合计 27.2 GB。按 1446 GB/s 算，单流 decode 上限约 53 token/s；算上 kernel 启动开销，现实上限约 50。
- **原版只有 9.9 token/s**，原因有两个：
  - gfx90a 没有 FP8 硬件，vLLM 走 Triton 的 FP8 block-scaled 矩阵乘，只跑到 0.2–0.4 TB/s；
  - 每 token 约 2700 个原生 norm 等小 kernel，整张 graph 共 2989 个 kernel。

### 2. 第一版（deploy v1）：不开 MTP，42 token/s

- **weight-only FP8 GEMV**：权重无损重排成 [N/64][K/16][64][16] 的 tile，每个 lane 负责一行；e4m3 转 f16 只要 3 条 VALU 指令。带宽 1.21 TB/s。
- **LM head**：bf16 GEMV（M≤8），尾部融合 argmax；另有融合的 top-k/top-p 采样器。
- **融合小算子**：GemmaRMSNorm、SiLU、DeltaNet decode、注意力相关算子都做成融合 kernel。每 token 的 graph 从 2989 个 kernel 降到 628 个。
- **结果**：
  - 单流 9.9 → 42 token/s，达到字节上限 53 的 79%；
  - 剩下约 5 ms 中，FP8 GEMV 比字节下限多 3.2 ms，其他 kernel 多 1.7 ms；
  - 并发吞吐 c4 71、c16 90、c64 124 token/s；
  - 与原版的差异在引擎自身的噪声范围内。

### 3. 第二版（deploy-mtp）：MTP 投机解码

**在 v1 上直接开 MTP 的情况**：K=3 只有 81 token/s，而且输出与不开 MTP 不同。原因如下：

- 一轮 42.7 ms，其中验证（4 个 token）的 graph 占 33.2 ms：
  - FP8 GEMV 在 M=4 时 27.2 ms（M=1 时 20.1 ms）；
  - DeltaNet 回退到原版 Triton 投机路径，约 4.0 ms（我们的 M=1 融合 kernel 只要 0.48 ms）。
- 3 次草稿约 7.1 ms，其中每步 1.8 ms 的 2.54 GB LM head 占了 5.4 ms。
- 输出逐层对比，第一处分歧是 FP8 GEMV：M≥2 时用了不同的 split-K 配置，M≥3 时用了成对 FMA 累加。之后依次是 DeltaNet 原版路径和注意力：Triton unified attention 对 4 个 query 走 2D 路径，对 1 个 query 走 3D split-softmax，两者结果不同。
- deploy v1 在 MTP 下还有两个潜在 bug：
  - `--enforce-eager` 时，vLLM 传入的是带步长的列视图，GDN kernel 因此报错；
  - MTP 下卷积状态宽 3+K 列，v1 的 kernel 读第 0–2 列，而原版读最后 3 列。

**这一轮的改动（每个 token 的结果都与单 token 路径逐位相同）：**

- **FP8 GEMV 逐列精确**：M=1..8 全部沿用 M=1 的 split-K 配置和累加顺序，速度基本不变。M>8（prefill、大并发）仍走 Triton 回退。
- **验证路径的 DeltaNet**（`gdn_spec`）：多 token 的融合 kernel，与逐 token 的 gdn_decode 逐位相同，并写 MTP 拒绝草稿后回滚用的状态槽。1.4 ms，原版路径约 4 ms。
- **验证注意力**：把 4 个验证 query 当成 4 个共享 block table 的单 query 序列（seqused_k = L+1..L+4），与单 token 路径走同一个 kernel，逐位相同，而且不更慢。
- **草稿一侧**：
  - 草稿 LM head 只用词表前 151552 行加 33 个 added token，加载时生成 int4 副本（393 MiB）。kernel 先选出 top-4，再用原 bf16 LM head 对候选重算，所以草稿等于 bf16 子集上的 argmax。每步从 1.8 ms 降到 0.35 ms。
  - 贪心验证直接用验证 LM head 尾部融合的 argmax，1 个 kernel 代替 5 个 Triton kernel。
  - GDN metadata 的约 19 个小 op 合成 1 个 kernel。
  - 草稿近似只影响接受率，不影响最终输出。
- **两个引擎层面的原因**（不在我们的 kernel 里；原版 vLLM 开 MTP 时输出也会因此变化）：
  - vLLM 按 K 自动选 block size（784/800/816）。前缀缓存对齐模式下，DeltaNet 的 prefill 按 block 边界分块，block size 不同，数值就不同。改为所有 K 固定 `--block-size 816`。
  - 开 MTP 时调度器会把可缓存位置回退一个 block（eagle block drop），长 prompt 的 prefill 分块因此不同。改为在 speculative config 里设 `disable_eagle_block_drop: true`。

**一轮的时间**：42.7 ms → 30.4 ms。

| 部分 | v1 | deploy-mtp |
| --- | --- | --- |
| 验证 graph（M=4） | 33.18 ms | 25.67 ms（其中 FP8 22.70，gdn_spec 1.44，注意力 0.26） |
| 验证后 eager（LM head M=4 + 判定） | 2.04 | 1.99 |
| 草稿 prefill graph | 2.55 | 0.97 |
| 草稿 decode graph | 4.51 | 1.64 |
| 验证前 eager | 0.46 | 0.12 |

### 4. 结果

单流 decode，`mtp_bench.py`（5 个对话提示，贪心，最多 384 token，关 thinking），各提示 tok/s 的算术平均，每个配置都在新启动的服务上测，与 v1 交替：

| K | deploy-mtp r2（现行） | r1 | v1 |
| --- | --- | --- | --- |
| 0 | 42.3–42.5 | 42.2 | 42.2 |
| 1 | 75.9 | 73.5 | 62.9 |
| 2 | 101.6 | 96 | 76.5 |
| 3（默认） | 121.0–122.6 | 112.5–113.4 | 80.4–81.3 |
| 4 | 123.7 | 115–116 | 81.5 |

r1 是第一次合入的 deploy-mtp，r2 是第 5 小节换了 FP8 GEMV 之后的版本。下面各条的数字是 r1 的，r2 的对应数字见第 5 小节。

- K=3 各提示：code 111、story 80、json 122、edit 122、repeat 130。接受率分别为 0.79、0.48、0.92、0.90、1.00。
- 5 个提示里有两个短输出（128、132 token），按 token 加权为 104.8。独立复核另外加了 5 个新提示（sql、中文、数学、表格、5.1k token 长文摘要），K=3 为 106–108 token/s；12 个 0.8k–7k token 的长上下文请求平均约 105。所以**更普遍的数字是 105–108 token/s**，相对 v1 的约 1.40 倍在各组上都成立。
- **输出**：贪心时开 MTP 与同一部署 MTP=0 逐字节相同。K=1..4、长 prompt（最长 6.5k token）、1500 token 长输出、thinking 模式、带前缀缓存的多轮对话都成立，logprobs 差为 0。v1 开 MTP 时只有 2–3/5 个提示相同。
- **MTP=0 不退步**：TPOT 23.8 ms、4096 token prefill 5.44 s 都与 v1 持平；并发 c4 80（v1 72）、c16 91、c64 124 token/s。
- **并发 ≥4 时 MTP 更慢**：c4 61 对 80，c64 90 对 124；c2 时更快（62–75 对 53）。vLLM 按 batch 大小动态改 K 的选项在这个版本上启动即失败，所以 K 只能在启动时选：交互式单用户用默认的 `MTP=3`，多路吞吐用 `MTP=0`。
- **复核**：两个独立复核都通过。
  - 正确性：4600 多次 GEMV 逐位抽查（含 NaN/Inf 邻行），各位置拒绝时的状态回滚，各种请求参数，采样分布检验（无偏差），300 s 压力测试（0 错误）。
  - 速度：从零构建、重测，与上表的差距在 1% 以内。

### 5. r2：FP8 GEMV 的 split-K 工作区（K=3 113 → 122 token/s）

硬件利用率测试（rocprofv3 计数器，`peak/hwutil`）表明，验证 graph 里的 kernel 全部受访存限制，没有一个受算力限制。r1 的 FP8 GEMV 在 M=4 时平均只有 1.07 TB/s，问题在 split-K 的工作区：每个 row group 一块独立的 fp32 工作区，M=4 时一次调用约 2.8 MB 脏数据，在权重流的中途被写回 HBM，和权重读取抢带宽。

改动（代码在 `vllm-opt/plugin/vopt_fp8gemv/csrc`，服务器上的开发目录是 `vllm-opt/opt3-fp8`、`opt3-fp8b`）：

- **L2 常驻的工作区槽位**：row group rg 用槽位 rg % 64，同时存在的部分和只有约 64 组，留在 L2 里。复用槽位的块先等上一个使用者归约完；每个槽位有使用计数，graph replay 安全。row group 不超过 96 的形状（down 和 out）每组独占一个槽位，不用等。
- **指定归约块**：每个 wave 最多一个 128-k 块的形状，由 b=B−1 的块归约，其他块写完部分和就退出；求和顺序与 r1 的“最后到达者”相同。
- **单缓冲 MFMA kernel**：每个 wave 最多一个块时用无循环版本，45 VGPR，occupancy 8。
- **原子 ticket 定序**：会等待的调用里，每个块先用 atomicInc 领工作序号，不用 blockIdx。所有等待都只等序号更小的工作，而领到更小序号的块一定已经驻留在运行，所以不依赖 GPU 按顺序派发 workgroup（实测派发顺序与 blockIdx 不同的块约占 98%）。最后一次领号时计数器回绕到 0。ticket 比不用 ticket 的版本慢约 0.4%，换来不会因派发顺序卡死。

**精确性**：所有形状 M=1..8 的每一行都与 M=1 调用逐位相同，M=1 与 r1 逐位相同，所以 MTP=0 的文本不变，开 MTP 后仍与 MTP=0 逐字节相同。三轮独立复核：槽位数边界（NSLOT 1/2/3/7/64/128）、打乱领号顺序、限制 CU 数、后台 GPU 负载、1 万次 graph replay、600 s 混合请求 soak（883 个请求，178 个中途取消，0 错误），都没有错误和挂起。

**效果**（K=3，torch profiler，每轮）：验证中的 FP8 22.7 → 20.6 ms（gdn_in_qkvz 3.70 → 3.21，gate_up 9.76 → 9.12，down 5.63 → 4.96，attn_qkv 1.10 → 0.96），GDN kernel 没改也从 1.44 降到 1.23 ms（写回争用少了），一轮 30.4 → 28.1 ms。接受的 token 数与 r1 完全相同，提速全部来自每轮更短。

| 测量 | r2 | r1 |
| --- | --- | --- |
| K=3，5 个测试提示，算术平均 | 121.0–122.6（7 轮中 6 轮） | 112.3–113.4 |
| K=3，按 token 加权 | 113 | 104.8 |
| K=3，5 个新短提示 | 112.1 | 103.8 |
| K=3，含两个 5k token prompt 的新提示 | 106.0 | — |
| K=0 TPOT | 23.7 ms | 23.7–23.8 ms |
| MTP=3 并发 c4 / c16 | 58.6 / 84.7 | 55.6 / 79.8 |
| MTP=0 并发 c4 / c16 / c64 | 82 / 90 / 124 | 80 / 90 / 124 |

**用法**：服务器上的 deploy-mtp 保留了 r1 的插件和扩展缓存（`plugin_r1/`、`cache_r1/`），`FP8_R1=1 ./start.sh serve` 切回 r1。失败模式是挂起而不是错误结果；调试时可以用编译选项 `-DVOPT_SPIN_MAX=n` 限制等待次数。`VOPT_WS_NSLOT` 只能在编译期设置。

**没有合入的尝试**：重写的 GDN 验证 kernel `gdn_spec2`（每 token 的 LDS barrier 从 4 次减到 2 次，结果逐位相同）没有变快（113.1–113.3 对 113.3–113.7）。GDN 每层约 26 µs 主要花在写回 4 个 fp32 回滚状态槽，与相邻 FP8 GEMV 争 DRAM，与 kernel 结构无关。

**还剩多少**：r2 验证 FP8 平均 1.18 TB/s（峰值 82%），gate_up、qkvz、attn_qkv 已到 1.23–1.26 TB/s，接近流式 kernel 在 gfx90a 上实际能达到的 1.2–1.3 TB/s。16.8 ms 的字节下限并不现实，估计每轮还能省约 1.2 ms，主要在 gdn_out、attn_o、down 这几个小形状上。

### 6. 限制与教训

- **逐字节相同的前提**：贪心采样；每步每个请求不超过 8 行，且不与 prefill 混在同一步（否则走 M>8 的 Triton 回退，与 M=1 不是逐位相同，v1 和原版的 batch 噪声也一样）；复用的前缀缓存块须由没有其他请求的 prefill 写入。
- **开 MTP 时 `min_p` 和 `logit_bias` 请求会被 vLLM 拒绝**（HTTP 400）。另外，带 stop 字符串时 usage 的 completion_tokens 会多计 1，原版 vLLM 也是这样。
- **block size 816 的代价**：KV cache 小约 4%（MTP=0 时 28.0 万 token，自选时 29.2 万）；长 prompt 的 MTP=0 输出与 v1 不再逐位相同，但仍在引擎噪声内。`BLOCK_SIZE=auto` 可以恢复自选。
- **剩余空间**：验证中的 FP8 GEMV 为 20.6 ms（r1 22.7 ms），字节下限 16.8 ms，gdn_out 和 attn_o 的带宽偏低（800–970 GB/s）。验证 LM head 1.99 ms，下限 1.76 ms。这两处都受“逐行沿用 M=1 运算顺序”的约束。
- **测量教训**：开 MTP 时，一个流式 SSE chunk 带一整轮（最多 K+1 个）token。按 chunk 间隔算 TPOT 会把 MTP 低估 3–4 倍，早先版本曾据此误判“随机文本上 MTP 更慢”（实际快 2–3 倍）。每 token 时间要按 `usage.completion_tokens` 计算。`bench_deploy.py` 和 `bench_tpot.py` 已经改正。

## 附录：开关、文件、分支

代码在服务器仓库 `/data/tools/mi210-opt/src`，最终版本为标签 `mi210-final`；相对保留版改动 39 个文件、+12,493/−79 行、86 个提交，完整补丁为 `deliver/mi210-final-vs-retained.patch`。

**环境变量开关**（未注明的默认都是开，设为 0 关闭）

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `LLAMA_GDN_FUSED_DECODE` | 开 | 融合 DeltaNet 解码；0 回到原图 |
| `GGML_GDN_DIET` | 1 | 0 用通用 GDN kernel；2/3 为预热变体 |
| `GGML_GDN_DECODE_SPLIT` / `GGML_GDN_DECODE_Q8` | 开 | out kernel 每头拆分多 block / 产出 ssm\_out 的 q8 |
| `GGML_MMVQ_Q8_CACHE` | 开 | q8 激活缓存 |
| `GGML_Q8_PRODUCERS`（`_NORM`/`_GNORM`/`_CONT`） | 开 | 生产者侧写 q8 |
| `GGML_Q8_NORM_SMALL` | 开 | 小代码版 norm kernel |
| `GGML_MMVQ_IQ3` | 开 | IQ3 模块；0 回到 4-bit 展开路径 |
| `GGML_MMVQ_IQ3_MULTI` / `_SM` / `_PF` | 开 / 开 / 7 | 同输入合并 / 步主序布局 / 前序顺序（5 为旧版） |
| `GGML_MMVQ_IQ3_SMALLN` | 关 | 1024 行改走旧 kernel |
| `GGML_MMVQ_IQ4K` / `_GROUP` | 开 | IQ4\_XS/Q4\_K 模块 / 分组（`_SMALL`、`_BIG`；后者默认关） |
| `GGML_MMVQ_IQ4K_X` / `_Q` | — | 覆盖 IQ4\_XS/Q4\_K 的 "L:NW" 配置 |
| `GGML_MMVQ_LOWBIT` | 开 | 低比特模块 |
| `GGML_MMVQ_MIX` / `_ORD` / `_GLU` | 开 / 开 / 关 | 混合配对 / 其前序顺序 / SwiGLU 尾部 |
| `GGML_MMVQF`（`_Q4K`、`_IQ4XS`、`_COOP`） | 1 | 早期的常驻自量化 kernel；2 为全部类型，`_IQ4XS` 为实验选项 |
| `GGML_MMVF_THIN` / `GGML_MMVF_EPI` | 开 | BF16 细矩阵 kernel / 其尾部融合（开启融合 DeltaNet 时不会触发） |
| `GGML_CUDA_DISABLE_ATTN_PREP_FUSION` | 关 | 设 1 关闭 attn\_prep |
| `GGML_CUDA_DISABLE_FA_GATE_FUSION` | 关 | 设 1 关闭 FA 门融合 |
| `GGML_CUDA_FA_DEC` / `GGML_CUDA_FA_DEC_MAX_KV` | 开 / 1024 | 短上下文 FA 解码及其 KV 长度上限 |
| `GGML_CUDA_FUSED_ROPE_NO_MROPE` | 关 | 设 1 时 norm+rope 融合不覆盖 MROPE/IMROPE |
| `GGML_CUDA_CONV_FUSE` / `GGML_CUDA_SWIGLU_F16` / `GGML_CUDA_GEMM_ADD_FUSION` | 开 | prefill 小算子融合：卷积输入 / SwiGLU 直接写 f16 / 残差加法并入 GEMM |
| `GGML_CUDA_F16_SHARE` / `GGML_CUDA_RMS_NORM_F16` | 开 | GEMM 输入的 f16 副本共享 / rms\_norm 顺带写出 f16 副本 |
| `GGML_CUDA_FUSION_STATS`、`GGML_CUDA_F16_SHARE_VERIFY`、`GGML_CUDA_CONV_FUSE_CHUNK` | 关 | 融合计数 / 复用时按位复核 / 覆盖卷积分块长度 |
| `GGML_CUDA_GDN_MFMA` / `GGML_CUDA_GDN_WY` | 开 | prefill DeltaNet 分块路径（0 回到逐 token kernel）/ 两段式 WY 递推（0 回到单 kernel 分块版） |
| `GGML_CUDA_GNORM_F16` / `GGML_CUDA_RMS_NORM_ROWS` | 开 | DeltaNet 输出 gated norm 直接写 f16 / q·k L2 norm 的多行 kernel |
| `GGML_CUDA_GEMM_TUNE` / `GGML_CUDA_GEMM_TABLE` | 开 | prefill GEMM 选核（0 用库默认，logits 确定）/ 固定 kernel 表（0 回到逐形状计时） |
| `GGML_CUDA_GATE_UP_MERGE` | 开 | FFN gate/up 合成一个 GEMM（只在表里有的 n 上；2 为测试用的强制合并） |
| `GGML_MMVQ_NC` / `GGML_MMVQ_NC_MAX` | 开 / 16 | 2–NC_MAX 个 token 时用 N 列 decode GEMV（MTP 验证）/ 最大列数（2–16；超过它回到不精确的 MMQ） |
| `GGML_MMVQ_NC_IQ3` / `_IQ4K` / `_LOWBIT` / `_MIX` | 开 | 分别关掉一类 N 列 kernel |
| `GGML_MMVQ_NC_IQ3_MFMA` / `_IQ4K_MFMA` / `_LOWBIT_MFMA` / `_MIX_MFMA` | 开 | 各类 N 列 kernel 的 int8 MFMA 版（0 回到 dp4a 版，最多 6 列） |
| `LLAMA_GDN_FUSED_DECODE_N` | 16 | 1–N 个 token 时 DeltaNet 走融合 PREP/OUT（0 回到通用图） |
| `GGML_CUDA_FA_DEC_NQ` / `GGML_CUDA_FA_DEC_NQ_LONG` | 开 | N 个 query 的 fattn-dec（带 gate、q8）/ KV > 1024 时逐 query 精确（0 更快但不逐位相同） |
| `GGML_Q8P_NC` | 开 | norm、SwiGLU 等为 N 行直接产出 q8 |
| `GGML_CUDA_GDN_KEEP_RS_EXACT` | 开 | 开 MTP 时 prompt 预填充的输出与不开 MTP 逐位相同（0 回到逐 token 递推） |
| `GGML_CUDA_GDN_KEEP_RS_TAIL` | 开 | 开 MTP 时预填充的快照递推只跑每块最后 K−1 个 token（0 回到对整块递推） |
| `GGML_CUDA_GDN_KEEP_RS_SPLIT` | 开 | 开 MTP 时预填充的分块 kernel 拆成两次、只跑一遍（0 回到两遍） |
| `LLAMA_INPUT_ON_GPU` | 开 | 全部层在一块 GPU 上时，token_embd 查表也在 GPU 上（0 放回 CPU） |
| `LLAMA_MTP_DRAFT_VOCAB` | 0 | MTP 草稿 LM head 只算前 n 个 token id（只适合纯英文 / 代码负载） |
| `GGML_MMVQ_NC_IQ3_MF2` / `GGML_MMVQ_NC_IQ3_MF_GRID` | 开 | IQ3 N 列 MFMA 第二种形式 / 网格用满全部 CU |
| `GGML_MMVQ_NC_LB_MF_UM4` / `GGML_MMVQ_NC_IQ4K_GROUP` | 开 | 低比特 ffn_down 每组 4 单元 / N 列时 attn_k+attn_v 合并 launch |
| `GGML_MMVQ_Q6K` / `GGML_MMVQ_Q6K_NC` | 开 / 关 | MTP 草稿层的 Q6_K 单 token GEMV / 2–4 列 Q6_K kernel（只用于草稿，默认关） |
| `GGML_CUDA_TOP_K_FAST` / `LLAMA_MTP_KV_ONLY` | 开 | 大词表小 k 的 top-k 选择 kernel（改变并列时的顺序，贪心不受影响）/ 草稿追赶解码只算 KV |
| `GGML_Q8_NORM_SHAPED` / `GGML_CUDA_ATTN_PREP_CS` / `GGML_CUDA_FA_DEC_UNROLL` / `GGML_GDN_PREP_EARLY` | 开 | 单 token 小算子：定形 rms_norm+q8 / cos·sin 表 / FA 解码展开 / DeltaNet prep 提前取数 |
| `GGML_MMVQ_NC_IQ4K_SN_MAX` / `_IQ4K_DOWN_MAX` / `_LB_Q2K_MAX` | 16 | 个别形状超过该列数改走 MMQ（更快但不精确） |
| `GGML_MMVQ_NC_VERIFY`、`GGML_GDN_DECODE_N_VERIFY`、`GGML_CUDA_FA_DEC_NQ_VERIFY`、`GGML_Q8P_NC_VERIFY` | 关 | 逐列与单 token 路径逐位比较（调试用） |
| `GGML_SCHED_SYNC_UPLOADS` / `GGML_CUDA_NO_UPLOAD_BATCH` | 关 | 设 1 回到同步上传 / 不合并小上传 |
| `GGML_CUDA_ATTN_PREP_VERIFY`、`GGML_GDN_DECODE_Q8_SELFTEST`、`GGML_MMVQ_MIX_EMPTY*` 等 | 关 | 调试与标定用 |
| `CODEX_IQ3_PREPACK` / `CODEX_IQ3_XXS_PREPACK` | 需设 1 | 保留版补丁的开关；所有测量都在 =1 下进行 |

**新增文件**（均在 `ggml/src/ggml-cuda/`，另有 `tests/test-gdn-decode.cpp`）

| 文件 | 内容 |
| --- | --- |
| `iq3gemv.cuh`、`mmvq-iq3.cu` | IQ3 模块 |
| `iq4k-gemv.cuh`、`mmvq-iq4k.cu` | IQ4\_XS/Q4\_K 模块 |
| `lowbit-gemv.cuh`、`mmvq-lowbit.cu` | 低比特模块 |
| `mixgemv.cuh` | 混合 IQ3+IQ4 配对 kernel |
| `mmvq-fused.cu`、`mmvq-fused.cuh`、`mmvq-fused-host.cuh` | 早期常驻 GEMV 框架与各模块入口声明 |
| `q8-producers.cu` | q8 生产者 kernel（norm、SwiGLU、门、门控 norm） |
| `gdn-decode.cu`、`gdn-decode.cuh` | 融合 DeltaNet 解码 kernel |
| `fattn-dec.cu`、`fattn-dec.cuh` | 短上下文 FA 解码 |

**修改的主要文件**：`ggml-cuda.cu`（融合入口、重排缓存、上传合并）、`mmvq.cu`、`mmvf.cu`、`rope.cu`（attn\_prep）、`fattn-common.cuh`/`fattn.cu`、`norm.cu`、`iq3-repack.cuh`、`ggml.h`/`ggml.c`（新 op）、`ggml-cpu/ops.cpp`、`ggml-backend.cpp`（异步上传）、`ggml-rpc`、`ggml-backend-meta.cpp`、`src/models/qwen35.cpp`、`tests/test-backend-ops.cpp`。

**分支与标签**

- `master`：保留版源码（对照基线）。
- 标签：`int-v8-base`（74.0 token/s）、`int-v9`（76.0）、`int-v11`（78.0）、`mi210-final`（80.3，等于分支 final）。
- 工作分支：gemv、deltanet、attn、gemv-iq3、gemv-iq4k、iq3-r3、iq4k-r3、r4-mixgroup、r4-ffndown、r4-attn、fixes、diet-gdn、diet-gemv、integration，内容都已合入 final。低比特、IQ3、IQ4K 模块各自还有独立开发仓库（`gemv/lowbit`、`gemv/iq3`、`gemv/iq4k`），含基准和实验变体。
- `wt-ts`：打了时间戳补丁的剖析版本（游离 HEAD），不要当正式版本用。
