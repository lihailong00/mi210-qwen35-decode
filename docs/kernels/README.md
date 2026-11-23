# 已有优化的适用范围

本表按组件列出 `main` 上的每项 HIP 后端优化：什么条件下触发、是否与模型无关、代码在哪、怎么关。目的是让新模型上手时能快速判断哪些优化会自动生效、哪些需要通用化。原理、收益和实测细节见"详情"一栏所指的 [Qwen3.8-27B 交接文档](../models/qwen3.8-27b/HANDOVER.md)（下文"Q27 §X"指它的第 X 节）。

**通用性**的含义：

- **通用**：按算子或量化类型触发，任何模型满足条件都会走到。
- **形状特化**：针对某个形状写的快速版本，其他形状自动回退到通用版本或原路径（不会出错，但拿不到收益）。
- **模型特定**：依赖某个模型的计算图写法（`src/models/*.cpp`）或只为某个模型的形状调好的参数表，其他模型基本不会触发。

文件都在 `ggml/src/ggml-cuda/`，除非另行注明。开关默认都是开，设 0 关闭（另行注明的除外）。

## decode：单 token GEMV（ne11 == 1）

| 优化 | 适用条件 | 通用性 | 文件 | 开关 | 详情 |
| --- | --- | --- | --- | --- | --- |
| IQ3_S / IQ3_XXS 模块（无损重排 + LDS 码表 + 移动读取窗口；残差、SwiGLU、多任务尾部） | 权重为 IQ3_S/IQ3_XXS，K%256==0，N%4==0 | 通用 | `iq3gemv.cuh`、`mmvq-iq3.cu` | `GGML_MMVQ_IQ3`（及 `_MULTI`、`_SM`、`_PF`） | Q27 §五.3 |
| IQ4_XS / Q4_K 模块（同尺寸 SoA） | 权重为 IQ4_XS/Q4_K；配置只按类型和 K 选（K > 8192 用轮次分块布局），是在 Q27 的形状上调出来的 | 通用 | `iq4k-gemv.cuh`、`mmvq-iq4k.cu` | `GGML_MMVQ_IQ4K`、`_GROUP`、`_X`/`_Q` 覆盖配置 | Q27 §五.4 |
| 低比特模块 | IQ2_S/IQ2_XS/IQ2_XXS/Q2_K/IQ1_M | 通用 | `lowbit-gemv.cuh`、`mmvq-lowbit.cu` | `GGML_MMVQ_LOWBIT` | Q27 §五.5 |
| Q6_K 单 token GEMV | Q6_K，K%256==0、K ≤ 32768、N ≥ 64（为 MTP 草稿层写的） | 通用 | `q6k-gemv.cuh`、`mmvq-q6k.cu` | `GGML_MMVQ_Q6K` | Q27 §十三.17 |
| IQ3 + IQ4 混合配对（两个 GEMV 共用输入时一次启动） | 一个 IQ3 和一个 IQ4_XS/Q4_K GEMV 读同一个输入 | 通用（混合量化配方才有） | `mixgemv.cuh` | `GGML_MMVQ_MIX` | Q27 §五.9 |
| BF16 细矩阵 GEMV | 行数很少的 BF16 矩阵 | 通用 | `mmvf.cu` | `GGML_MMVF_THIN`、`GGML_MMVF_EPI` | Q27 §三、§五 |
| 重排权重缓存 | 以上各模块共用；显存增加约一份同尺寸副本 | 通用 | `ggml-cuda.cu`（`ggml_cuda_get_repacked`） | 随各模块 | Q27 §五.2 |

## decode：激活量化、norm 与小算子

| 优化 | 适用条件 | 通用性 | 文件 | 开关 | 详情 |
| --- | --- | --- | --- | --- | --- |
| q8 激活缓存（输入相同的连续 GEMV 只量化一次） | 单 token 与 N 列 GEMV | 通用 | `mmvq.cu` | `GGML_MMVQ_Q8_CACHE` | Q27 §五.1 |
| q8 生产者（norm、SwiGLU、门、门控 norm 顺带写 q8） | 生产者后面紧跟量化 GEMV | 通用 | `q8-producers.cu` | `GGML_Q8_PRODUCERS`（`_NORM`/`_GNORM`/`_CONT`）、`GGML_Q8P_NC` | Q27 §五.1、§五.8 |
| 多 workgroup rms_norm + q8 | 任意宽度；原地重叠时用单 workgroup 版本 | 通用 | `q8-producers.cu`、`norm.cu` | `GGML_Q8_NORM_SMALL` | Q27 §五.8 |
| 定形 rms_norm + q8 | **只在恰好 5120 列时**；其他宽度用上一项 | 形状特化 | `q8-producers.cu`（`Q8P_SHAPED_NCOLS`） | `GGML_Q8_NORM_SHAPED` | Q27 §十三.17 |
| 门控 norm 写 q8 | 行宽 128 的 gated norm（DeltaNet 输出） | 形状特化 | `q8-producers.cu` | `GGML_Q8_PRODUCERS_GNORM` | Q27 §五.1 |
| split 输入异步上传、小上传合并 | 所有模型 | 通用 | `ggml-backend.cpp`、`ggml-cuda.cu` | `GGML_SCHED_SYNC_UPLOADS=1`、`GGML_CUDA_NO_UPLOAD_BATCH=1` 关闭 | Q27 §五.10 |
| token embedding 查表放 GPU | 全部层在一块 GPU 上 | 通用 | `src/llama-model.cpp` | `LLAMA_INPUT_ON_GPU` | Q27 §十三.17 |
| 大词表小 k 的 top-k | k ≤ 32 | 通用 | `top-k.cu` | `GGML_CUDA_TOP_K_FAST` | Q27 §十三.17 |

## decode：注意力

| 优化 | 适用条件 | 通用性 | 文件 | 开关 | 详情 |
| --- | --- | --- | --- | --- | --- |
| 短上下文 FA 解码（单次启动 split-KV，带门和 q8） | **头维 256**，KV ≤ 1024，8×kv 头数 ≤ CU 数 | 形状特化 | `fattn-dec.cu`、`fattn-dec.cuh` | `GGML_CUDA_FA_DEC`、`GGML_CUDA_FA_DEC_MAX_KV` | Q27 §五.7 |
| N 个 query 的 FA 解码（MTP 验证） | 同上；假设一个 KV cache 只有一个序列（`-np 1`） | 形状特化 | `fattn-dec.cu` | `GGML_CUDA_FA_DEC_NQ`、`_NQ_LONG` | Q27 §十三.16 |
| attn_prep（q/k norm + IMROPE + gate 拷贝 + 写 KV 一次启动） | qwen3.5 / qwen3-next 的门控注意力图；需要 `qwen35.cpp` 让 q/k/v 投影相邻 | 模型特定 | `rope.cu`、`ggml-cuda.cu` | `GGML_CUDA_DISABLE_ATTN_PREP_FUSION=1` 关闭 | Q27 §五.7 |
| FA 门融合（sigmoid(gate)·attn 并入 FA 合并） | 门控注意力（qwen3.5 / qwen3-next） | 模型特定 | `fattn-common.cuh`、`fattn.cu` | `GGML_CUDA_DISABLE_FA_GATE_FUSION=1` 关闭 | Q27 §五.7 |
| norm + rope 融合支持 MROPE/IMROPE | 用 MROPE/IMROPE 的模型 | 通用 | `rope.cu` | `GGML_CUDA_FUSED_ROPE_NO_MROPE=1` 关闭 | Q27 §五.7 |

## decode：Gated DeltaNet（线性注意力）

| 优化 | 适用条件 | 通用性 | 文件 | 开关 | 详情 |
| --- | --- | --- | --- | --- | --- |
| 融合 DeltaNet 解码：新 op `GDN_DECODE_PREP/OUT`，每层 15 个小 kernel + 2 个 BF16 GEMV → 2 次启动 | 图在 `src/models/qwen35.cpp` 里构造；单序列、f32 状态、1–16 个 token | 模型特定（qwen3next、qwen35moe 结构相同，可移植） | `gdn-decode.cu`、`ggml.c`、`ggml-cpu/ops.cpp`、`src/models/qwen35.cpp` | `LLAMA_GDN_FUSED_DECODE`、`LLAMA_GDN_FUSED_DECODE_N`、`GGML_GDN_DECODE_SPLIT`、`GGML_GDN_DECODE_Q8` | Q27 §五.6、§十三.16 |
| 精简版 DeltaNet kernel | **H_k=16、H_v=48、S=128、卷积 4、alpha/beta 行宽 5120**，其他形状用通用 GDN kernel | 形状特化 | `gdn-decode.cu` | `GGML_GDN_DIET`、`GGML_GDN_PREP_EARLY` | Q27 §五.6 |

## MTP / 投机解码验证（2–16 个 token）

| 优化 | 适用条件 | 通用性 | 文件 | 开关 | 详情 |
| --- | --- | --- | --- | --- | --- |
| N 列 GEMV（每列与单 token 逐位相同；N≥3–4 用 int8 MFMA） | 上面各量化模块，N ≤ `GGML_MMVQ_NC_MAX`（16），超过回到不精确的 MMQ | 通用 | 各 `*gemv.cuh` | `GGML_MMVQ_NC`、`_NC_MAX`、`_NC_IQ3`/`_IQ4K`/`_LOWBIT`/`_MIX`、`*_MFMA` | Q27 §十三.15–17 |
| 开 MTP 时 prefill 的快照递推只跑尾部、输出精确 | GATED_DELTA_NET 且 n_rs_seq > 0 | 通用（DeltaNet 类模型） | `gated-delta-net*.cu` | `GGML_CUDA_GDN_KEEP_RS_EXACT`/`_TAIL`/`_SPLIT` | Q27 §十三.16–17 |
| 草稿追赶解码只算 KV、草稿词表截短 | qwen35 的 MTP 头 | 模型特定 | `src/models/qwen35.cpp` | `LLAMA_MTP_KV_ONLY`、`LLAMA_MTP_DRAFT_VOCAB`（默认 0） | Q27 §十三.17 |

## prefill

| 优化 | 适用条件 | 通用性 | 文件 | 开关 | 详情 |
| --- | --- | --- | --- | --- | --- |
| CDNA2 上默认 `ROCBLAS_USE_HIPBLASLT=1` | 所有设备都是 CDNA2 且用户没设 | 通用 | `ggml-cuda.cu`（`ggml_cuda_init`） | `ROCBLAS_USE_HIPBLASLT=0` 关闭 | Q27 §十三.4 |
| 反量化 256 线程 + LDS 暂存 | IQ2_XXS/XS/S、IQ3_XXS/S、IQ4_XS、Q4_K、Q2_K | 通用 | `convert.cu` | `GGML_CUDA_DEQUANT_LDS` | Q27 §十三.5 |
| 逐形状 GEMM 选核（运行时计时） | n ≥ 2048 的 f16 GEMM | 通用 | `ggml-cuda.cu`（`ggml_cuda_hip_gemm_tuned`） | `GGML_CUDA_GEMM_TUNE` | Q27 §十三.7、§十三.12 |
| 固定 GEMM kernel 表 | **只对 Q27 的 8 个 f16 形状，且 rocBLAS / hipBLASLt 版本串、设备名、CU 数完全匹配** | 模型特定 | `ggml-cuda.cu`（`ggml_cuda_hip_gemm_pins`） | `GGML_CUDA_GEMM_TABLE` | Q27 §十三.12 |
| FFN gate/up 合成一个 GEMM | 只在表里有合并形状的 n 上 | 模型特定 | `ggml-cuda.cu`（`ggml_cuda_try_fuse_gate_up`） | `GGML_CUDA_GATE_UP_MERGE` | Q27 §十三.12 |
| FA prefill CDNA 快速路径 | DKQ=DV=256，ncols=64 且 ncols2>1 的实例 | 形状特化 | `fattn-mma-f16.cuh` | 编译选项 `-DGGML_CUDA_FATTN_MMA_CDNA_FAST=0` | Q27 §十三.9 |
| GQA 比 6 的 FA 分发修正 | GQA = 6，批次 ≥ 64 | 形状特化 | `fattn.cu` | `GGML_CUDA_FATTN_GQA6_MIN_BATCH`（0 关闭） | Q27 §十三.9 |
| concat / 长序列卷积 LDS 分块 | 通道排列的 concat、SSM_CONV | 通用 | `concat.cu`、`ssm-conv.cu` | `GGML_CUDA_CONCAT_TILE`、`GGML_CUDA_SSM_CONV_TILE` | Q27 §十三.7 |
| 卷积输入融合 `CONCAT → SSM_CONV+SILU` | F32、d_conv = 4、无 bias、n_t ≥ 256 | 通用（SSM 类） | `ssm-conv.cu`、`ggml-cuda.cu` | `GGML_CUDA_CONV_FUSE` | Q27 §十三.10 |
| SwiGLU 直接写 f16 | `GLU(SWIGLU) → MUL_MAT` | 通用 | `ggml-cuda.cu` | `GGML_CUDA_SWIGLU_F16` | Q27 §十三.10 |
| f16 副本共享、rms_norm 顺带写 f16 | 同一 F32 激活被多个 GEMM 读取（仅 HIP 构建） | 通用 | `ggml-cuda.cu`、`norm.cu` | `GGML_CUDA_F16_SHARE`、`GGML_CUDA_RMS_NORM_F16` | Q27 §十三.10 |
| 残差加法并入 GEMM（beta=1） | 图里写成 `add(residual, cur)`；**`qwen35.cpp` 在 n_tokens ≥ 256 时改了操作数顺序，其他模型要做同样的改动才会触发** | 通用（需改模型图） | `ggml-cuda.cu`、`src/models/qwen35.cpp` | `GGML_CUDA_GEMM_ADD_FUSION` | Q27 §十三.10 |
| DeltaNet 分块 MFMA / 两段式 WY | CDNA2、标量 gate、**头维 128**、token ≥ 32 | 形状特化（DeltaNet 类） | `gated-delta-net-mfma.cu`、`gated-delta-net-wy.cu` | `GGML_CUDA_GDN_MFMA`、`GGML_CUDA_GDN_WY` | Q27 §十三.8、§十三.11 |
| gated norm 写 f16、多行 L2 norm | DeltaNet 输出 norm；≥ 8192 行 | 通用 | `norm.cu` | `GGML_CUDA_GNORM_F16`、`GGML_CUDA_RMS_NORM_ROWS` | Q27 §十三.11 |

## 需要统一的命名

`CODEX_IQ3_PREPACK` / `CODEX_IQ3_XXS_PREPACK` 是"保留版"基线补丁的开关，所有测量都在 =1 下进行。后续计划把它们和本表中的开关统一到一套前缀，旧名保留兼容。

## 新模型上手时最可能要做的通用化

按对其他模型的影响大小排列：

1. **GEMM 固定表**：从代码里搬出来，按 (m, k, n, beta) 查找、可以从文件加载，附一个生成工具。
2. **头维 / GQA 特化**：FA decode 只支持头维 256，FA prefill 的 GQA 修正只针对 6。头维 128 的模型（大多数 Llama / Qwen dense）拿不到这两项。
3. **残差加法的图写法**：其他模型的 `src/models/*.cpp` 要改成 `add(residual, cur)` 才能触发 beta=1 融合。
4. **定形 norm（5120 列）和精简 DeltaNet**：换成按宽度模板化的版本。
5. **DeltaNet 融合解码**：移植到 qwen3next / qwen35moe 的图构造。
6. **IQ4_XS / Q4_K 配置**：为新形状扫一遍（可用 `GGML_MMVQ_IQ4K_X` / `_Q` 覆盖配置做实验）。
