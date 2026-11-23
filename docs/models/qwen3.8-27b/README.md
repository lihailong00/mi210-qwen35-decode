# Qwen3.8-27B

本仓库优化的第一个模型，两条线：

- **llama.cpp + IQ3_S GGUF**：`Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`（架构 qwen35，12,120,016,960 字节，SHA-256 以 58fd8267 开头）。同一个 GGUF，不重新量化、不降精度。
- **vLLM + 官方 FP8 checkpoint**：`Qwen3.8-27B-FP8`（自带 MTP 层），只做无损的权重重排。代码在 [vllm-opt/](../../../vllm-opt/README.md)。

完整的工作记录见 [HANDOVER.md](HANDOVER.md)（下文"§X"指它的第 X 节）。

## 结构要点

- 64 层：48 层 Gated DeltaNet（线性注意力），16 层全注意力（第 il%4==3 层）；另有 1 层 MTP。
- 隐藏维 5120，FFN 17408，词表 248320。
- 注意力：24 个 q 头、4 个 kv 头（GQA 6），头维 256，带输出门，IMROPE。
- DeltaNet：16 个 k 组、48 个 v 头，状态维 128，卷积核 4，每层递归状态约 3 MB（f32）。
- IQ3_S GGUF 是混合量化：每 token 读 IQ3_S 3.64 GB、IQ4_XS 3.02 GB、IQ3_XXS 2.08 GB、Q4_K 1.58 GB（含 0.715 GB 的 LM head），另有少量 IQ2_S/Q2_K/IQ2_XS/IQ2_XXS/IQ1_M/BF16；同一层的 gate 与 up 常常类型不同。

## 上限

| 项目 | llama.cpp IQ3_S | vLLM FP8 |
| --- | --- | --- |
| decode 每 token 读取 | 11.35 GB 权重 + 0.30 GB 状态 + 64 KB/上下文位置 | 24.33 GB FP8 + 2.54 GB bf16 LM head + 0.3 GB 状态 |
| decode 物理上限（上下文 0） | 124 token/s | 约 53 token/s |
| decode 逐节点下限 | 约 110 token/s | — |
| prefill 物理上限（300 W 功耗上限下的真实算力） | 约 2980 token/s | — |

详见 §十三.1–3、§十三.6、§十三.14、§十四.1。

## 结果

**llama.cpp，IQ3_S**（单卡 MI210，基线为"保留版"：b12110 + ROCm 7.2 兼容补丁 + IQ3 4-bit 展开补丁）

| 指标 | 基线 | 现在 | 变化 |
| --- | --- | --- | --- |
| decode tg128，上下文 0 | 45.6 token/s | 80.3 token/s | +76% |
| decode tg128，上下文 8192 | 44.2 | 74.6 | +69% |
| decode tg128，上下文 32768 | 40.7 | 64.4 | +58% |
| prefill pp512 / pp2048 / pp8192 | 893 / 1099 / 1220 | 1612 / 2074 / 2125 | +81% / +89% / +74% |
| llama-server 单请求 | 45.6 | 79.1 | +73% |
| llama-server，MTP K=3（5 个提示总体） | — | 154–155 | 输出与不开投机逐字节相同 |
| 每 token kernel 启动数 | 1933 | 601 | −69% |
| 显存峰值 | 24.1 GB | 28–32 GB | 重排权重副本 |

正确性：不开投机时与基线 logits 的平均 KLD 0.00125，处在纯计算重排的噪声底线（0.00114）上；开 MTP 的输出与不开逐字节相同。decode 在现有约束下现实可达约 84–86 token/s，已停止（§十三.14）。

**vLLM，FP8**（vLLM 镜像 `2e7da1ad1c66`，vllm 0.30.0+rocm723，镜像不改）

| 指标 | 原版 | 现在 |
| --- | --- | --- |
| 单流 decode，不开 MTP | 9.9 token/s | 42 token/s |
| 单流 decode，MTP K=3 | — | 106–122 token/s（输出与 MTP=0 逐字节相同） |
| 并发 64 路，不开 MTP | — | 约 124 token/s |

## 复现

编译见仓库根目录的 [README](../../../README.md)。运行时保留"保留版"补丁的开关：

```bash
export CODEX_IQ3_PREPACK=1 CODEX_IQ3_XXS_PREPACK=1
M=/data/model/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
# decode / prefill
./build/bin/llama-bench -m $M -ngl 999 -fa on -b 4096 -ub 4096 -ctk f16 -ctv f16 -p 0 -n 128 -r 2
# 服务（不开投机）
./build/bin/llama-server -m $M -ngl 999 -fa on -c 65536 -np 1 --host 127.0.0.1 --port 8080
# 服务（MTP，推荐 K=3）
./build/bin/llama-server -m $M -ngl 999 -fa on -c 65536 -np 1 --spec-type draft-mtp --spec-draft-n-max 3 --port 8080
```

vLLM 部署：

```bash
cd vllm-opt
./build.sh                     # 预编译 4 个 HIP 扩展（不用 GPU）
./start.sh serve               # 默认 MTP=3，端口 18340，模型名 qwen38
MTP=0 ./start.sh serve         # 多路并发服务用这个
```

评估脚本（KLD、困惑度、贪心 md5、服务器测试）在 `mi210-tools/eval/`，用法见 §十二。

## 文档索引

| 主题 | 位置 |
| --- | --- |
| 环境、工作区、基线测量 | §二、§三 |
| decode 各项优化（GEMV 模块、q8、DeltaNet、注意力） | §五 |
| 正确性验证与代码审查修复 | §六 |
| 时间分解、无效的尝试 | §七、§八 |
| 经验与硬件发现（通用部分已整理到 [docs/mi210/](../../mi210/hardware.md)） | §九、§十 |
| 操作手册 | §十二 |
| 上限、prefill 各轮优化 | §十三.1–13 |
| decode 差距复核与结论 | §十三.14 |
| MTP 三轮 | §十三.15–17 |
| vLLM FP8 部署与 MTP | §十四，以及 [vllm-opt/README.md](../../../vllm-opt/README.md) |
| 全部开关、新增文件、分支与标签 | 附录 |
