# 模型

每个优化过的模型在本目录下有一个子目录 `docs/models/<模型id>/`。模型 id 用小写的"系列 + 版本 + 规模"，例如 `qwen3.8-27b`；同一个模型的不同量化或后端放在同一个目录里。

## 已登记的模型

| 模型 id | 架构 | 后端 / 权重 | 状态 | 文档 |
| --- | --- | --- | --- | --- |
| `qwen3.8-27b` | qwen35（48 层 Gated DeltaNet + 16 层全注意力，带 MTP） | llama.cpp，IQ3_S GGUF；vLLM，官方 FP8 | decode / prefill / MTP 已完成 | [qwen3.8-27b/](qwen3.8-27b/README.md) |
| `inkling-2-mini` | inkling2（循环 MoE，128 专家选 8，24 层权重跑 48 次，带 DSpark drafter） | llama.cpp（Unsloth 合并版 b12588），UD-Q8_K_XL GGUF（NVFP4 + BF16） | decode 33.3 → 109.8 token/s，prefill pp2048 519 → 6280、128K 1515 → 4013 token/s | [inkling-2-mini/](inkling-2-mini/README.md) |
| `ternary-bonsai-2.5-27b` | qwen35 + Hadamard 1024，带 MTP | llama.cpp，Q2_0 GGUF | Q2_0 GEMV、四行交错 MFMA、融合输入准备与分组投影，单请求 K=3 为 152.6 token/s | [ternary-bonsai-2.5-27b/](ternary-bonsai-2.5-27b/README.md) |

## 每个模型目录里放什么

| 文件 | 内容 |
| --- | --- |
| `README.md` | 概览：模型文件（路径、大小、哈希）、结构要点、上限、结果表、复现命令、文档索引 |
| `HANDOVER.md` 或其他工作记录 | 完整的过程记录：每项优化的原理、收益、失败的尝试、开关 |

概览的结果表至少包含：decode tg128（上下文 0 / 8192 / 32768）、prefill（pp512 / pp2048 / pp8192）、llama-server 单请求、开投机解码时的数字（如果有），以及物理上限和正确性结论（KLD、与基线是否逐位相同）。数字都要写明测量条件（命令、提示集、几轮、交替顺序）。

## 新增一个模型

1. 按 [../mi210/method.md](../mi210/method.md) 第 7 节的清单清点结构、算上限、测基线。
2. 建 `docs/models/<模型id>/README.md`，填上面的概览内容；在上表里加一行。
3. 在 [../kernels/README.md](../kernels/README.md) 里对照适用条件，记下哪些已有优化生效、哪些没有生效以及原因。
4. 优化过程中改了公共代码的，要在所有已登记的模型上复测（速度和 KLD / md5），并更新各自的结果表。
5. 新模型的里程碑标签用 `<模型id>/<名字>` 的形式，例如 `qwen3.8-27b/mi210-final`。
6. 有了新的、与模型无关的硬件发现，补进 [../mi210/hardware.md](../mi210/hardware.md)；新的通用优化补进 [../kernels/README.md](../kernels/README.md)。
