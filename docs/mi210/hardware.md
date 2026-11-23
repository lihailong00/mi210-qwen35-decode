# MI210（gfx90a）硬件事实：实测值与经验规则

本文汇总在单张 AMD Instinct MI210 上做推理优化时**实测**得到、与具体模型无关的硬件行为。数字来自 ROCm 7.2.1、300 W 功耗上限的这台机器；出处是各模型的工作记录（目前是 [Qwen3.8-27B 交接文档](../models/qwen3.8-27b/HANDOVER.md)，下文简称"Q27 交接"），换卡、换 ROCm 版本后要重测。

做新模型时先读本文和 [method.md](method.md)：这里的很多结论都是花了几个小时才排除的误区。

## 1. 基本参数

| 项目 | 值 |
| --- | --- |
| 架构 | CDNA2，gfx90a，wave64 |
| 计算单元 | 104 CU，每 CU 4 个 SIMD |
| 显存 | 64 GB HBM2e，标称 1638 GB/s |
| LDS | 每 CU 64 KB（按 CU 计，不是按 block） |
| 功耗上限 | 300 W（`power1_cap` 已等于 `power1_cap_max`，不能调高） |
| 时钟 | 核心最高 1700 MHz，显存 1600 MHz |

## 2. 实测峰值

| 项目 | 实测 | 说明 |
| --- | --- | --- |
| HBM 读带宽（单 kernel） | **1446 GB/s**（48 MB）、1433（256 MB）、1399（1 GB）、1361（4 GB） | 标称的约 88% |
| 13–50 MB 张量的纯读 | 1.33–1.42 TB/s | 单个 GEMV 能达到的上限 |
| 流式 kernel（读写都有：转换、卷积模板、FP8 GEMV） | 0.9–1.3 TB/s | 达不到纯读峰值，按这个估现实上限 |
| MFMA FP16→FP32 | 170.9 TFLOPS（零数据）/ **150.6**（随机数据） | 1700 MHz 标称 181 |
| MFMA INT8→INT32 | 171.4 TOPS | 与 FP16 同速：**INT8 没有算力优势**，int8 MFMA 的价值只在于和 dp4a 结果逐位一致、指令更少 |
| rocBLAS / hipBLASLt GEMM | 方阵 4096³ FP16 145 TFLOPS；真实形状随机数据约 122–131 TFLOPS | 功耗上限下 GEMM 主循环最好约 128 TFLOPS |

**一定要用随机数据测算力。** 全零或常数数据功耗低，频率能保持在 1700 MHz，会把算力高估 15–18%。真实数据下频率被功耗上限压到约 1565–1600 MHz。decode（220–245 W）不受功耗限制，prefill 受。

微基准源码：`mi210-tools/peak-bench/`（`peak_bench.hip` 带宽 / MFMA / 方阵 GEMM，`bound_bench.hip` 逐张量读取 graph 与全候选 GEMM 扫描，`mfma_rand.hip` 零数据与随机数据对比，`gemm_sweep.hip` 不同 token 数下的选核）。

## 3. 性能模型

- **单个 GEMV 的下限**：t ≈ 1.6 µs + 字节数 / 1.46 TB/s。把模型每个节点都代入这个式子求和，就是"逐节点下限"，比纯读取的物理上限更接近可达值。
- **decode 上限的层次**（以 Q27 为例，见 Q27 交接第十三节第 2 小节）：全部字节以峰值读取（物理上限）> 每个权重一次 launch 的纯读取 graph > 再加每个张量一次 launch。只要每个权重张量单独 launch，就到不了物理上限；要越过，必须让读取跨越张量边界连续进行（持久 kernel 或融合 kernel）。
- **prefill 上限**：每 token FLOP ÷ 约 150 TFLOPS（随机数据的 MFMA 峰值）。库 GEMM 在功耗上限下约 128 TFLOPS，自写 GEMM 不划算（见第 9 节）。

## 4. kernel 启动与 HIP graph

- llama.cpp 的 decode 用 HIP graph 回放（Q27 每 token 8 个 graph 块）；关掉 graph 每 token 慢约 1.6 ms。**必须保留。**
- graph 回放中一个空 kernel 约 1.35 µs，实际小 kernel 约 1.3–3 µs。
- 两个 GEMV 之间插一个额外 kernel，无论形状约多 3.5 µs；连续排列的额外 kernel 每个只多约 0.7 µs。
- **合并两个大 GEMV 只省约 0.4 µs。** 分组只对小 GEMV 或前序很重的 kernel（例如要把码表搬进 LDS 的 IQ3）划算。
- **把小 kernel 吸进常驻 kernel 通常不划算**：常驻 kernel 末尾的完成序列（栅栏、原子计数）约 3.8–5 µs，比省下的那次启动还贵。
- 多流（`GGML_CUDA_GRAPH_OPT=1`）更慢：跨流同步约 13 µs，还会关掉部分融合。
- `HIP_FORCE_DEV_KERNARG=1` 无效果（graph 回放已把参数放在显存）。

## 5. 冷指令缓存

模型内每次 kernel 派发都从冷的指令缓存开始，关键路径上每条指令约 **8–13 ns**。

- 同一算法的 FA 解码：4714 条指令 76.5 µs，1427 条 33 µs，847 条 17.4 µs。
- rms_norm 从 398 条减到 308 条：5.1 → 4.35 µs。
- 推论：热的微基准会系统性地误导；代码更长的"更快"kernel 放进模型常常更慢（例如 MFMA 单次启动的解码注意力：热测更快，模型内更慢）。缩短前序（去掉运行时除法、多余的地址计算和寄存器清零）是低风险的稳定收益。

## 6. 内存系统

- **DRAM 局部性**：每个 wave 各自顺序读一大片分散地址时只有约 1.16 TB/s；所有 wave 共同推进一个"移动窗口"可达 1.42 TB/s。权重重排（步主序、轮次分块）的依据就是这个。
- **GGUF 块的对齐**：66/74/82/98/110 字节的量化块只有 2 字节对齐，只读不算也只有 650–800 GB/s，必须无损重排成宽而合并的读取（SoA、16 字节对齐）。
- **码表类量化（IQ\*）**：码表在全局内存时受查表延迟限制，只有带宽上限的 25–30%；放进 LDS 快约 2.3 倍。
- **脏 L2 行**：前一个 kernel 写了一大块数据（例如 DeltaNet 写 3 MB 状态），紧接着的 GEMV 要替它写回，每次多 2–3 µs。非时间性存储、`buffer_wbl2` 都没能解决。
- **split-K 工作区**：每组一块独立 fp32 工作区时，脏数据在权重流中途被写回 HBM、和权重读取抢带宽。把同时存在的部分和限制在约 64 个槽位、常驻 L2 后，FP8 GEMV 平均 1.07 → 1.18 TB/s（Q27 交接第十四节第 5 小节）。
- 跨 kernel 的 L2 预取、提前读残差：无效果（L2 冷启动不是瓶颈）。

## 7. 调度

- **硬件偏爱老 wave**：同一 CU 上两个 workgroup 的第二个晚 3–6 µs 结束，每个常驻 GEMV 都有这段尾部不均。动态 / 配对 workgroup、wave 优先级、旋转循环等软件调度手段全部无效。
- **派发顺序与 blockIdx 不同**：graph 内实测约 98% 的 block 开始顺序与 blockIdx 不一致。**任何自旋等待都不能依赖按序派发**；要等待时用 atomicInc 领工作序号，只等序号更小的工作（领到更小序号的块一定已经驻留）。
- 需要整个 grid 同时驻留的 kernel（kernel 内屏障、网格同步），要保证 block 数 ≤ 可同时驻留的数量，否则会挂死。
- 一个 CU 同时跑两个 block 只有约 1.33 倍吞吐；序列并行的扫描、更细的切分往往抵不过多出的工作。

## 8. 计算单元与 ISA

- 一个 wave 大约每 10 个周期执行一条普通指令，一条 fp32 MFMA（16x16x4）约 35 个周期：**关键路径上的指令条数才是成本**。
- 同一个 wave 在相关的 MFMA 之间插入的普通指令**不会被隐藏**（一步从 32.8 周期涨到插 8 条指令时的 48.8）；同一 SIMD 上另一个只做普通指令的 wave 可以完全重叠。所以要把 MFMA 和 VALU 工作分到不同的 wave（波分工）。
- 循环里的条件读、条件计算、全局写或 break，会让编译器插入 `s_waitcnt vmcnt(0)`，流水归零。改 kernel 后要看反汇编。
- 对刚加载的值做 select 会迫使 wave 立刻等这次加载：改用钳位下标的无条件加载，使用时再屏蔽。
- 对寄存器数组的循环必须完全展开，否则编译器用 `s_set_gpr_idx` 动态索引，每个元素多出好几条指令。
- 全局加载要一起发出、提前一个阶段；`s_barrier` 之前只需要 `lgkmcnt(0)`，全局加载可以跨 barrier 保持在途。
- **LDS bank 冲突是一阶效应**：每条 `ds_read`/`ds_write` 的（dword 地址 mod 32）要在 64 个 lane 间均匀分布。同样的指令只改布局，FA prefill kernel 里相差 2.3 ms（约 10 ms 中）。
- `__builtin_shufflevector` 的下标必须是字面常量（用模板参数，不要用展开的循环变量）。
- `hip.h` 的 `CDNA` 宏只在 device 编译阶段有定义，host 侧启动代码不能放在 `#if defined(CDNA)` 里。
- 不要用 `fminf(x, 0)` 钳位指数：AMD 的 minNum 会吞掉 NaN，还会悄悄改变正值的行为；用精确的分段和代替。

## 9. 数学库

- rocBLAS 自带的 Tensile 选核在 gfx90a 上会选到很差的 kernel（Q27 的 ffn_down 只有 47.5 TFLOPS，最优 148）。CDNA2 上默认设置 `ROCBLAS_USE_HIPBLASLT=1`（已在 `ggml_cuda_init` 里做），hipBLASLt 默认选核全面更好。
- hipBLASLt 的 kernel 每个时钟已达到纯 MFMA 速率的 92–95%，**自写 GEMM 不划算**；空间在选核：库默认比最好的有效 solution 慢 3–41%（n=512、K 大、beta=1 最差）。
- `rocblas_gemm_ex_get_solutions` 每个形状约 73 ms（进程首次约 590 ms），不要放在请求路径上；库不接受的 solution index 约 6 µs 就返回失败、不写输出。hipBLASLt 的 index = rocBLAS index − 1，API 里没有更快的 kernel。
- 选核要看稳态（随机数据、背靠背重复），突发计时会选错；合并类决定要用模型内的单一尺寸进程 A/B，不能用只测 GEMM 的成对计时（功耗上限下的频率行为会反转结论）。
- MMQ（直接用量化权重做矩阵乘）在 gfx90a 上只有 40–67 TOPS，即使 512 token 的批次也比"反量化 + hipBLASLt"慢。

## 10. 测量工具

- **rocprofv3 会失真**：把任何小 kernel 显示为 ≥ 4.3 µs，并把 graph 回放拖成 CPU 瓶颈（每节点约 47 µs）。GEMV 这类长 kernel 的时长可信（±3%）。
- 小 kernel 的真实成本用 **graph 内 GPU 时间戳**测：`mi210-tools/profiling/attn-misc-gpu-timestamps.patch`（只用于剖析，不要合入正式版本）。
- 硬件计数器（受访存还是受算力限制）可用 rocprofv3 的计数器模式。

## 11. 数值

- 带递归状态的模型（DeltaNet / SSM 类）中，纯 fp32 求和重排就会让困惑度漂 0.1–0.4%、贪心输出在接近平局的 token 上分叉。**不要用困惑度或贪心 md5 判断非位精确改动的正确性**，用 KLD 对照噪声底线（方法见 [method.md](method.md)）。
- e4m3 → f16 转换 3 条 VALU 指令即可；`v_mfma_f32_4x4x1f32` 的每个元素等于一次正确舍入的 fma，可以用来做与标量 fmaf 逐位一致的多列计算。
