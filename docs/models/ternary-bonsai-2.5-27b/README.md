# Ternary-Bonsai 2.5 27B on MI210

单卡 MI210 64 GiB，ROCm 7.2.1，llama.cpp HIP。目标模型为 Ternary-Bonsai-2.5-27B-Q2_0-mtp.gguf，SHA-256 为 `b0f83de8481c3200c529aef5fb495dbfedf2a4a3f017f7a3ea0a69c67be1d306`。

## 源码与运行

主代码包含 Q2_0 单列 GEMV、四行交错 MFMA、多列验证、融合 Hadamard 和 Q8 输入准备、共享输入的分组投影，以及八列专用内核。DFlash 借用主模型的嵌入和输出权重时采用权重所属模型的 Hadamard 变换。常规单请求最快的已测配置为原生 MTP3。使用本项目的 ROCm 构建，令 `MODEL` 指向本地 GGUF。

```sh
./build/bin/llama-server -m "$MODEL" -ngl 99 -fa on -t 8 -b 4096 -ub 4096 \
  -ctk f16 -ctv f16 -c 8192 -np 1 --spec-type draft-mtp --spec-draft-n-max 3
```

## 实测

| 工作负载 | 结果 |
|---|---:|
| 单流无 MTP，短上下文 tg128 | 45.0 → 79.4 token/s |
| 八列验证，生成时每轮 257 次 Q2_0 投影 | 20.41 → 14.01 ms |
| 原生 MTP3，九题综合 | 145.5 token/s |
| 原生 MTP7，八列优化前后，同一轮对比 | 103.3 → 128.2 token/s |
| DFlash7，八列优化前后，同一轮对比 | 102.1 → 130.0 token/s |
| DFlash7 最后一次九题对照，原生 / 手工重排 / 训练打分头 | 129.4 / 136.5 / 135.2 token/s |

九题均为单请求贪心生成，每题 256 个 token，按总输出 token 数除以总生成时间，排除预填充。DFlash7 的 200 token/s 在三个真实在线配置中均未达到。使用已知答案选择候选的速度单列在实验报告中。

## 实验源码与报告

以下目录包含对应的测试程序、阶段补丁和公开测量数据。关键实现已经直接集成在推理源码。四列融合版的可检出提交为 `9cf9b81`，八列内核和 DFlash 权重变换修正位于 `51b7846`。历史补丁对应各报告里的阶段基线。无损 FFN 三值打包、FFN 权重预取及草稿选择实验保存在独立目录。候选打分器目录含冻结参数、重排和训练源码，以及为本次实验编写的训练题。

- [单卡 MI210 上的 200 token/s 可行性测试](../../../mi210-tools/eval/bonsai/benchmarks/bonsai-200-feasibility-20261127/REPORT.md)
- [MI210 上 DFlash7 的草稿接受率与 200 token/s](../../../mi210-tools/eval/bonsai/benchmarks/bonsai-dflash7-prefix-20261128/REPORT.md)
- [用 Bonsai 2.5 的生成轨迹训练 DFlash7 候选打分](../../../mi210-tools/eval/bonsai/benchmarks/bonsai-dflash7-training-20261128/REPORT.md)
- [Bonsai 27B 真实 FFN 权重的无损三值打包](../../../mi210-tools/eval/bonsai/benchmarks/bonsai-ffn-real-trit-20261127/REPORT.md)
- [MI210 上 Bonsai K=3 的下一步优化](../../../mi210-tools/eval/bonsai/benchmarks/bonsai-mtp-budget-20261127/REPORT.md)
- [MI210 上 Bonsai 27B 的 FFN gate/up 优化实验](../../../mi210-tools/eval/bonsai/benchmarks/bonsai-mtp-ffn-prefetch-20261127/REPORT.md)
- [Bonsai 27B 在 MI210 上的 MTP 输入融合与分组投影](../../../mi210-tools/eval/bonsai/benchmarks/bonsai-mtp-fused-input-20261127/REPORT.md)
- [MI210 上 Bonsai 27B 的 MTP 验证优化](../../../mi210-tools/eval/bonsai/benchmarks/bonsai-mtp-mfma-20261127/REPORT.md)
- [MI210 上 Bonsai 27B 的下一步优化选择](../../../mi210-tools/eval/bonsai/benchmarks/bonsai-mtp-options-20261127/REPORT.md)
- [MI210 上 Bonsai K=3 验证耗时与带宽](../../../mi210-tools/eval/bonsai/benchmarks/bonsai-mtp-roof-20261127/REPORT.md)
- [Bonsai 27B 的 MI210 四行交错权重优化](../../../mi210-tools/eval/bonsai/benchmarks/bonsai-mtp-tile4-20261127/REPORT.md)
- [Bonsai 在 MI210 上的 MTP 多 token 验证](../../../mi210-tools/eval/bonsai/benchmarks/bonsai-mtp-verify-20261127/REPORT.md)
- [MI210 上 Bonsai 27B 的下一步优化](../../../mi210-tools/eval/bonsai/benchmarks/bonsai-next-priority-20261127/REPORT.md)
- [单卡 MI210 保留 Bonsai 2.5 Q2_0 时的 200 token/s 路线](../../../mi210-tools/eval/bonsai/benchmarks/bonsai-q2-200-routes-20261127/REPORT.md)
- [MI210 上 Bonsai Q2_0 八列验证实测](../../../mi210-tools/eval/bonsai/benchmarks/bonsai-q2-eight-20261128/REPORT.md)
- [Bonsai Q2_0：MTP 实测](../../../mi210-tools/eval/bonsai/benchmarks/ternary-bonsai-2.5-27b-mtp-20261127/REPORT.md)
- [Ternary-Bonsai-2.5-27B：MI210 第一轮优化](../../../mi210-tools/eval/bonsai/benchmarks/ternary-bonsai-2.5-27b-optimization-20261127/REPORT.md)
