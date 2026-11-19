# Qwen3.8-27B-FP8 在 MI210 上的 vLLM 单流 decode 优化部署

对象：官方 checkpoint `/data/model/Qwen3.8-27B-FP8`，不重新量化，`--language-model-only`。
运行环境：vLLM 镜像 `2e7da1ad1c66`（vllm 0.30.0+rocm723），单卡 MI210（gfx90a）。

镜像本身不做任何修改。所有改动都以只读 bind mount 的插件目录加载，通过 `PYTHONPATH` 和 `usercustomize.py` 的 post-import hook 对 vLLM 模块做 monkeypatch。HIP 扩展由 `torch.utils.cpp_extension` 编译，缓存在 `cache/torch_ext`。

## 结果

下表为优化前后对比。除注明外，单流 TPOT 都是先跑 32 token 预热，再跑 3 次、每次 512 个输出 token，取稳态 TPOT（第 16 个 token 之后）的中位数。请求为 greedy、`ignore_eos`、不带 `min_tokens`。每个数据都由独立的服务实例测得，优化版和原版交替运行。

| 指标 | 原版 vLLM | 本部署 | 测量 |
|---|---|---|---|
| 单流 decode，128 token prompt | 100.91 / 101.02 ms → **9.9 tok/s** | 23.82 / 23.83 / 23.85 ms → **41.9–42.0 tok/s** | `bench_deploy.py tpot`，runs: origA、orig2 / opt1、opt2、opt3 |
| 单流 decode，4096 token prompt | 101.49 ms → 9.85 tok/s | 24.20 / 24.07 ms → 41.3–41.5 tok/s | runs: origB / opt1、opt2 |
| 单流 decode，默认采样（T=1, top_k 20, top_p 0.95） | — | 23.87 / 23.87 ms → 41.9 tok/s | runs: optP、opt3 `s128` |
| 4096 token prompt 的 prefill（TTFT，max_tokens=1，5 次中位数） | 5645 ms | 5452 ms | `bench_deploy.py ttft`，runs: origA / opt1 |
| 并发 4（vllm bench serve random 512/128，8 个请求） | 31.86 tok/s | 71.54 tok/s | runs: origC / optC |
| 并发 16（32 个请求） | 82.67 tok/s | 90.39 tok/s | 同上 |
| 并发 64（128 个请求） | 98.12 tok/s | 123.63 tok/s | 同上 |

物理上限参考：每 token 需读约 27.2 GB（FP8 层权重 24.33 GB、lm_head 2.54 GB、DeltaNet 状态约 0.3 GB）。按实测读带宽 1446 GB/s 计算，上限约 53 tok/s；本部署每 token 628 个 graph kernel 加 1 个 lm_head，按每次 launch 约 1.6 µs 的固定开销计入后，下限约 19.8 ms（约 50 tok/s）。本部署约 41.9–42.0 tok/s（23.8 ms），是 53 tok/s 字节上限的 79%、计入 launch 后下限的约 83%。剩下的 5 ms 中，FP8 GEMV 比字节下限多 3.2 ms（K=5120 的形状 +1.5、down +0.9、N=5120,K=6144 的形状 +0.85），其余 graph kernel 多 1.7 ms，eager 输入准备 0.2 ms。

### 正确性

1. **Greedy 生成（8 个 prompt × 128 token，带 top-5 logprobs，`compare.py`）。** 原版自身在并发 1 与并发 4 之间（batch 形状不同）就有 67.2% / 0.0122 的差异。本部署相对原版的差异与此相当，处于引擎自身的 batch 形状噪声水平。

   | 对比 | 逐位置 token 一致率 | 所选 token 的 mean \|Δlogprob\| |
   |---|---|---|
   | 原版 c1 vs 原版 c4（引擎自身噪声） | 67.2% | 0.0122 |
   | 本部署 c1 vs 原版 c1 | 66.8% | 0.0114 |
   | 本部署 c4 vs 原版 c4 | 67.2% | 0.0128 |

   一致率不到 100%，是因为 bf16 下近似并列的 token 一旦翻转，后续文本就整段分叉。|Δlogprob| 只统计分叉前上下文相同的部分。

2. **Perplexity 式检查（prompt_logprobs，per-token）。**

   | 文本 | 原版 mean NLL | 本部署 mean NLL |
   |---|---|---|
   | 601 token 固定文本（`PPL_TEXT`） | 1.7316 | 1.7345（差 +0.0029，标准误 0.0041，不显著） |
   | 模型 README.md 前 6000 token | 0.3609 | 0.3580（−0.0030，标准误 0.0020；关掉 attn 补丁时为 0.3620，+0.0011 ± 0.0016） |

   逐项消融（同一 601 token 文本）的结果如下：
   - 只开 fp8gemv：与原版逐位相同。prefill 走 tiled Triton fallback。
   - 只开 norms：+0.0013 ± 0.0040。
   - 只开 attn：+0.0082 ± 0.0044。
   - 作为参照，原版把 prefill 改成 256 token 分块，也会产生 +0.002 ± 0.0013 的变化。

3. **算子级检查（`tests/test_prefill_ops.py`，M=601 / 4096，对照 fp64 参考）。**
   - 融合 GemmaRMSNorm：与原生实现只有约 1e-5 比例的元素差 1 ulp，residual 输出逐位相同，fp64 误差两者相同。
   - 融合 qk-norm/RoPE：比原生路径更接近 fp64。q 的 rel_l2 为 0.00187，原生为 0.00197；原因是 norm 与 rope 之间不再经过一次 bf16 舍入。

   各 kernel 自身的单元测试（对 fp64）见各 builder 目录，结论都是 bf16 舍入级误差。

## 改了什么

插件目录在 `plugin/`，内含三个包。下表的"补丁名"即 `VOPT_PATCHES` 中的名字，"关闭开关"是对应的环境变量。

| 包 / 补丁名 | 作用 | 替换掉的东西 | 关闭开关 |
|---|---|---|---|
| `vopt_fp8gemv` / `fp8gemv` | FP8 block-scaled 权重 only GEMV，M≤8。fp8 按位精确解码到 f16，fp32 累加，bf16 输出。加载时把 fp8 字节无损重排成 `[N/64][K/16][64][16]` 的 1 KB tile（原张量释放）。按形状和 M 调过 split-K；最后到达的 block 按固定顺序归约，结果确定，计数器自复位，可安全放进 graph。M>8（prefill、大 batch）走"动态 fp8 激活量化 + 参考 exact-decode Triton kernel"的 tiled 版本 | Triton `_mi210_exact_fp8_block_scaled_mm` + `per_token_group_quant` | `VOPT_FP8GEMV=0` |
| `vopt_lmhead` / `lmhead` | lm_head（248320×5120 bf16）专用 GEMV，M≤6 | vLLM wvSplitK（M=1 时 2.1 ms；M=3/5 时 9–22 ms） | `VOPT_LMHEAD=0` |
| `vopt_lmhead` / `greedy`、`topk` | greedy：在 GEMV 尾部融合 argmax。采样：融合 top-k(≤64)/top-p/temperature 采样器。都只在请求没有其他 logits 处理时启用，否则回退原版 | gumbel / argmax / top-k-top-p 的约 30 个小 kernel | `VOPT_SAMPLER=0` |
| `vopt_lmhead` / `bf16gemv` | 小 bf16 线性层（`in_proj_ba`，96×5120）改用 split-K GEMV | wvSplitK | `VOPT_BF16SMALL=0` |
| `vopt_rest` / `norms` | GemmaRMSNorm（含残差加法）合成 1 个 HIP kernel | 每次约 11 个原生 PyTorch kernel，共 161 处 | `VOPT_NORMS=0` |
| `vopt_rest` / `act` | SiLU×mul 多 block kernel（M≤256） | vLLM `act_and_mul`（每 token 1 个 block） | `VOPT_ACT=0` |
| `vopt_rest` / `gdn` | GDN（DeltaNet）纯 decode 的一步合成 1 个 kernel：conv1d 更新、SiLU、q/k L2 norm、门控、delta rule、gated RMSNorm | zeros、causal_conv1d_update、packed recurrent、RMSNormGated 共 4 个 kernel | `VOPT_GDN=0` |
| `vopt_rest` / `attn` | 在 ROCm 上启用 vLLM 自带的 Triton 融合 qk-norm+RoPE+gate（原版只在 CUDA 上启用），sigmoid(gate)×out 合成 1 个 kernel | 每个 attention 层约 25 个小 kernel | `VOPT_ATTN=0` |

所有补丁遇到不支持的情况都回退原版代码，包括：prefill、混合 batch、spec decode、其他 dtype 或布局、M 超上限。

效果是每 token 的 decode graph 从 2989 个 kernel 降到 628 个。

服务参数与参考启动脚本相同：`TRITON_ATTN`、`FULL_DECODE_ONLY` 图、`mode 0`、`VLLM_ROCM_USE_AITER=0`、参考 `fp8_utils.py` overlay、`max-model-len 8192`、`max-num-seqs 128`、`gpu-memory-utilization 0.92`。

## 怎么用

```bash
cd /data/tools/mi210-opt/vllm-opt/deploy

# 1) 预编译三个 HIP 扩展（不用 GPU，不用锁；约 30 秒）。
#    不做的话，第一次启动服务时也会自动编译到同一个 cache。
./build.sh

# 2) 正常服务（前台运行，Ctrl-C 或 docker rm -f vllm-opt-deploy-serve 停止）。
#    默认端口 18140，模型名 qwen38，只监听 127.0.0.1。
PORT=18140 ./start.sh serve
#    和其他会话共用 GPU 时，让服务运行期间一直持有 GPU 锁：
GPU_LOCK=1 ./start.sh serve
#    额外参数会原样传给 vllm serve，例如：
./start.sh serve --no-enable-prefix-caching

# 3) 基准模式：取锁，后台启动，等服务就绪，在主机上跑 job，把日志存到 runs/<tag>/，删容器，放锁
STEPS="t128 t4k ttft4k ppl ppl6k cap cap4" ./start.sh bench mytag harness/job.sh
ORIG=1 STEPS="t128" ./start.sh bench mytag_orig harness/job.sh       # 原版 vLLM，用于 A/B
STEPS="conc" CONC="4 16 64" ./start.sh bench mytag_c harness/job.sh  # 并发吞吐
PROFILE=1 STEPS="prof" ./start.sh bench mytag_p harness/job.sh       # torch profiler
python3 harness/analyze_trace.py cache/prof/mytag_p/*.gz             # 逐 kernel 分解
python3 harness/compare.py runs/origA/capture.json runs/mytag/capture.json
```

### 开关（环境变量）

- `ORIG=1` 或 `VOPT=0`：关闭全部自定义部分，即原版 vLLM 加参考 overlay。
- 单项关闭（默认都是 1）：`VOPT_FP8GEMV`、`VOPT_LMHEAD`、`VOPT_SAMPLER`、`VOPT_BF16SMALL`、`VOPT_NORMS`、`VOPT_ACT`、`VOPT_GDN`、`VOPT_ATTN`。例如 `VOPT_GDN=0 ./start.sh serve`。
- 细调（通过 `EXTRA_ENV="K=V ..."` 传入容器）：
  - `VOPT_FP8_MAXM`（默认 8）
  - `VOPT_FP8_CFG="N,K,M:R,S,B;..."`（覆盖 GEMV 分块表）
  - `VOPT_LMHEAD_MAXM`（默认 6）
  - `VOPT_ACT_MAXM`（默认 256）
  - `VOPT_GDN_MAXM`（默认 512）
  - `VOPT_FUSED_QK`（默认 1）
- 其他：`PORT`、`NAME`、`CACHE`、`MAX_LEN`、`MAX_SEQS`、`MTP=n`（未验证，见下文）、`MAXLOCK`（基准模式最长持锁秒数，默认 840）。

启动日志中应能看到：
- `Selected VoptFp8GemvKernel for Fp8LinearMethod`
- `[vopt_fp8gemv] extension loaded`
- `[vopt_lmhead] lm_head patched`
- `[vopt_rest] GemmaRMSNorm patched` 等

## 剩余时间分解（每 token，torch profiler，runs/optP，42 个稳态 decode 步）

profiler 测得每步周期 24.00 ms，不开 profiler 实测 23.83 ms（profiler 每个 kernel 约多算 0.85 µs）。其中 graph 内 kernel 合计 21.94 ms，graph 外 eager kernel 2.03 ms，GPU 空闲 0.03 ms，即完全是 GPU-bound。

| 部分 | 实际 | 下限（字节 / 1446 GB/s） | 说明 |
|---|---|---|---|
| FP8 GEMV 合计（256 次） | 20.09 ms（1.21 TB/s） | 16.82 ms（加 launch 17.23） | 主要差距所在 |
| ├ mlp gate_up 34816×5120（64 次） | 8.87 ms，1287 GB/s | 7.89 ms | |
| ├ mlp down 5120×17408（64 次） | 4.85 ms，1176 GB/s | 3.94 ms | |
| ├ gdn in_qkvz 16384×5120（48 次） | 3.16 ms，1276 GB/s | 2.78 ms | |
| ├ gdn out 5120×6144（48 次） | 1.76 ms，856 GB/s | 1.04 ms | 前一个 kernel 写回 3 MB GDN 状态，占用带宽（推测） |
| ├ attn qkv 14336×5120（16 次） | 0.95 ms，1243 GB/s | 0.81 ms | |
| └ attn o 5120×6144（16 次） | 0.51 ms，990 GB/s | 0.35 ms | |
| lm_head（eager，1 次） | 1.81 ms | 1.76 ms | 已接近下限 |
| GDN decode（48 次） | 0.48 ms | 约 0.21 ms（状态读写） | |
| GemmaRMSNorm（129 次） | 0.48 ms | launch-bound | 可融合进下一个 GEMV 的前导 |
| SiLU×mul（64 次） | 0.24 ms | launch-bound | 可融合进 gate_up 的尾部 |
| in_proj_ba（48 次） | 0.19 ms | launch-bound | 可并进 qkvz 的 launch |
| attention（16 层 × 5 个 kernel） | 0.38 ms | | |
| graph 外输入准备与采样的小 kernel | 0.22 ms | | 含 48 µs 的 pageable H2D 拷贝 |

## 已知限制 / 后续

- **MTP / spec decode 未验证。** 开 MTP 时 GDN 融合路径会回退原版。FP8 GEMV 会接管 MTP 层的线性层，但这一点没测过。
- **M=8 的 GEMV 只有约 0.7 TB/s。** 因此并发 5–8 的收益比 1–4 小。M>8 走 Triton fallback；c16、c64 已不低于原版。
- **采样器的语义差异。** 融合采样器只在请求没有 penalties、bias、bad words、min_p、显式 seed、processed logprobs、`min_tokens` 时启用，否则回退原版。随机数流与原版不同：分布相同，但同一 seed 下结果不逐位相同，所以带 seed 的请求会回退原版。
- **prefix caching。** `--no-enable-prefix-caching` 可再省约 0.15 ms/token（这是 builder 的测量，本部署未复测），但会失去 prefix caching，所以没有默认开启。
- **进一步提速的方向**（都是估计）：
  - FP8 GEMV 从 1.21 TB/s 提到 1.3 TB/s 以上，约 −1.5 ms；
  - norm、SiLU、in_proj_ba 融合进 GEMV，约 −0.5 ms；
  - 二者合计可到约 45 tok/s。

## 文件

- `start.sh`：启动器，有服务模式和基准模式。
- `build.sh`：预编译三个扩展。
- `plugin/`：
  - `usercustomize.py`：加载器；
  - `vopt_fp8gemv/`、`vopt_lmhead/`、`vopt_rest/`：三个包，含 kernel 源码 `csrc/`、补丁和单元测试。
- `harness/`：
  - `job.sh`：基准步骤 `t128 t4k s128 ttft4k ppl ppl6k cap cap4 prof conc`；
  - `bench_deploy.py`、`capture.py`、`compare.py`、`analyze_trace.py`、`img_run.sh`。
- `tests/test_prefill_ops.py`：prefill 尺寸（M=601/4096）下 norm 与 qk-rope 的数值检查。
- `runs/`：所有测量结果，每个目录含 `job.log`、`server.log`、json 结果。
- 各部分的开发记录和单元测试在 `../fp8gemv`、`../lmhead`、`../rest`。

GPU 共享：基准模式和 `GPU_LOCK=1` 的服务模式都会持有 `/data/tools/mi210-opt/gpu.lock`。容器名为 `vllm-opt-deploy-*`，结束时自动删除。

## 补充说明（独立复核后）

- **启动时间**：服务就绪约 80–220 秒，取决于模型文件是否在页缓存中（另一个会话会把 27 GiB 权重挤出缓存），以及首次启动时的 Triton JIT（新缓存目录首次多约 45 秒）。
- **GPU 锁**：`start.sh serve` 默认持有 `/data/tools/mi210-opt/gpu.lock`（`GPU_LOCK=1`）。只有确定没有别人用 GPU 时才设 `GPU_LOCK=0`。
- **正确性检查要包含不带 `min_tokens` 的请求**：`min_tokens` 会触发 logit bias，使贪心快速路径回退到原版采样器。复核用不带 `min_tokens` 的 16 个 prompt 测过融合 argmax 路径：与原版的逐位置一致率 64.7%、|Δlogprob| 0.0130；原版自身（c1 对 c8）为 61.4%、0.0155，属于引擎噪声。脚本：`/data/tools/mi210-opt/vllm-opt/vchk-c/harness/vchk_cap.py`。
- **bit 一致的范围**：FP8 的 tiled 回退路径在 M ≥ 65（prefill）时与原版逐位相同；M = 9–64（9–64 条并发的 decode）时因分块累加顺序不同，有 ≤ 2 bf16 ulp 的差别。
- **top-p 截断处的并列**：恰好在截断值上并列的 logit，保留哪一个由实现决定（按下标）。原版 torch 与 Triton 实现之间在这里本来也不一致。
- **不要开 torch.compile**：自定义算子会写内部工作区和静态全局变量，但 schema 没有声明为可变。部署用 compilation mode 0 加 graph 是安全的；开启 inductor 编译可能被重排或删除。
- **TPOT 统计**：`bench_deploy.py` 按 SSE chunk 计时。带 logprobs 或多字节 token 时，几个 token 可能合并成一个 chunk，per-chunk TPOT 最多偏高约 4%；文中的 t128/t4k 数据已核对 chunk 数 = token 数。
- **偶发停顿**：复核时 15 个优化版请求中有 1 个出现过一次 185 ms 的 token 间隔（该请求稳态 TPOT 25.7 ms），未能复现，原因未知。
