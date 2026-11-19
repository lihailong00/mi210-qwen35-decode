# Qwen3.8-27B-FP8 在 MI210 上的 vLLM 部署 v2：MTP 投机解码

> 仓库说明：本目录是服务器上部署目录 `/data/tools/mi210-opt/vllm-opt/deploy-mtp` 的源码拷贝（插件、启动脚本、测量脚本、单元测试）。文中的 `runs/...` 运行记录、`cache/` 预编译缓存，以及 `../deploy`（v1）、`../mtp-*`、`../vchk-*` 等工作目录只在服务器上，没有放进仓库。`V1=1` 需要 v1 的插件目录（`V1_PLUGIN`）；仓库里没有 v1，可以用本插件的各个 `VOPT_*` 开关得到与 v1 相同的输出。主机路径可以用环境变量改：`MODEL_DIR`（模型目录）、`GPU_LOCK_FILE`（GPU 锁）、`FP8_OVERLAY`（可选，主机上原有的 fp8_utils.py 覆盖文件，只影响 M>8 的回退路径，不存在时自动跳过）。`harness/vllm_run.sh` 和 `harness/img_run.sh` 是开发时用的脚本，路径按服务器写死。

对象：官方 checkpoint `/data/model/Qwen3.8-27B-FP8`（自带 MTP 层），不重新量化，`--language-model-only`。
运行环境：vLLM 镜像 `2e7da1ad1c66`（vllm 0.30.0+rocm723），单卡 MI210（gfx90a）。

本目录 = deploy v1（`../deploy`，只读参考，未改动）的完整拷贝，加上 MTP 投机解码的全部改动。镜像本身不做任何修改：插件目录以只读 bind mount 挂载，经 `PYTHONPATH` 和 `usercustomize.py` 的 post-import hook 对 vLLM 打补丁；HIP 扩展用 `torch.utils.cpp_extension` 编译，缓存在 `cache/torch_ext`。v1 的原始说明在 `README_v1.md`，v1 的测量记录在 `runs_v1/`。

## 一句话结论

- 单流 decode（5 个对话提示各自 tok/s 的算术平均，贪心）：**MTP K=3 为 112.5–113.4 tok/s**，deploy v1 同样 K=3 为 80.4–81.3 tok/s（+40%），v1 不开 MTP 为 42 tok/s。按 token 加权（总 token / 总 decode 时间）为 104.8。独立复核另加 5 个新提示（sql、中文、数学、表格、5.1k token 长文摘要）和 12 个 0.8k–7k token 长上下文请求，K=3 为 106–108 tok/s（算术平均）/ 约 105（长上下文）。**更普遍的数字是 105–108 tok/s**，相对 v1 的 1.40 倍在各组上都成立。
- **开 MTP 的输出与同一部署 MTP=0 的输出逐字节相同**（贪心）。K=1、2、3、4 全部成立，包括 914–6392 token 的长 prompt，连 logprobs 都相同。deploy v1 开 MTP 时有 3/5 个提示不同。
- MTP=0 时速度（TPOT 23.84–23.90 ms）、prefill、并发 c4/c16/c64 都没有比 v1 退步。c4 反而更快（78.8 tok/s，v1 为 71.5）。
- **推荐 K=3**，这也是 `start.sh` 的默认值，适用于交互式、单用户、低并发（≤2 路）的场景。高并发吞吐场景用 `MTP=0`（见下文"何时不要开 MTP"）。

## 结果

所有数据都在新启动的服务上测得，与 deploy v1（`V1=1`）交替运行，每个配置至少 2 轮。运行目录在 `runs/`（下表"runs"一列），汇总脚本是 `harness/summary.py` 和 `harness/vchk_table.py`。

### 单流 decode，`mtp_bench.py`（5 个对话提示，贪心，最多 384 token，关 thinking），tok/s

每格为两轮的数值（a / b），括号内是 5 个提示的平均接受率（每个草稿 token 被接受的比例）。

| K | 本部署 平均 | v1 平均 | 本部署 code | story | json | edit | repeat | 与本部署 K=0 文本相同 | v1 与 v1 K=0 相同 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 42.4 / 42.2 | 42.2 / 42.2 | 42.0 | 42.3 | 42.3 | 42.7 | 42.7 | — | — |
| 1 | 73.4 / 73.5 | 62.9 / 62.9 | 73.6 | 67.8 | 74.2 | 75.2 | 76.3 | 5/5 | 3/5 |
| 2 | 96.3 / 95.9 | 76.1 / 76.8 | 96.0 | 78.5 | 101.8 | 102.3 | 103.2 | 5/5 | 2/5 |
| **3** | **113.0 / 111.8** | 80.4 / 80.7 | 110.7 | 79.9 | 122.4 | 122.3 | 129.9 | 5/5 | 2/5 |
| 4 | 115.2 / 116.2 | 81.7 / 81.2 | 111.6 | 72.9 | 129.1 | 122.0 | 140.3 | 5/5 | 2/5 |

上表是第一轮 A/B（runs `n_k*`、`v_k*`，当时 block size 由 vLLM 自选）。最终配置（block size 固定为 816，并关闭 eagle block drop，见下文"正确性"）复测：

| 配置 | 平均 | code | story | json | edit | repeat | runs |
|---|---|---|---|---|---|---|---|
| 本部署 K=3 | 112.8 / 113.4 / 113.2 / 112.5 | 110.4 | 79.7 | 122.1 | 122.1 | 129.6 | `b_k3a` `b_k3b` `c_vchk3` `f_k3` |
| 本部署 K=4 | 115.7 / 115.9 | 113.2 | 73.1 | 129.4 | 122.2 | 140.6 | `b_k4a` `b_k4b` |
| 本部署 K=0 | 42.1 / 42.1 / 42.2 | 42.0 | 42.0 | 42.0 | 42.3 | 42.3 | `b_k0a` `b_k0b` `f_k0` |
| v1 K=3（交替） | 80.7 / 81.3 | 78.8 | 55.8 | 88.7 | 87.3 | 92.7 | `v_k3c` `v_k3d` |
| v1 K=0（交替） | 41.8 | 42.1 | 42.1 | 42.1 | 40.5 | 42.4 | `v_k0c` |

接受率（本部署，K=3）：code 0.79、story 0.48、json 0.92、edit 0.90、repeat 1.00。除 json（v1 为 0.94）外与 v1 相同。json 的差别不是草稿头造成的（`VOPT_SPEC=0` 也是 0.92），而是精确 GEMV 让生成的文本略有不同；接受率随文本变化。同理，5.1k token 长文摘要提示在默认配置下文本与 K=0 相同、接受率 0.69（96.9 tok/s），改用 `BLOCK_SIZE=auto` 或 `MTP_BLOCK_DROP=1` 时生成另一段同样合理的文本，接受率 0.79（约 106 tok/s）。

**推荐 K=3 的理由**：K=4 平均只多 2.5%，但 story 这类难预测的文本从 79.9 掉到 73 tok/s；K 越大，并发时和难预测文本上的损失越大，KV cache 也越小（block size 自选时，K=0/3/4 分别约 29.2 万 / 16.5 万 / 14.5 万 token）。如果负载以代码、JSON、复述类文本为主，可以用 `MTP=4`。

### MTP=0 与 v1 的对比（不退步）

| 指标 | 本部署 | deploy v1 | runs |
|---|---|---|---|
| 单流 TPOT，128 token prompt（`bench_deploy.py tpot`，3×512 token） | 23.69 / 23.85 / 23.90 / 23.89 / 23.84 ms | 23.86 / 23.87 / 23.84 ms | `n_k0a` `n_k0b` `b_k0a` `b_k0b` `f_k0` / `v_k0a` `v_k0b` `v_k0c` |
| 单流 TPOT，4096 token prompt | 24.06 / 24.06 / 24.17 ms | 24.18 / 24.23 ms | 同上 |
| 4096 token prefill（TTFT，max_tokens=1，5 次中位数） | 5453 / 5452 / 5438 / 5440 ms | 5454 / 5459 ms | 同上 |
| 并发 4（`vllm bench serve` random 512/128，8 个请求） | 79.98 / 78.76 tok/s | 71.52 tok/s | `n_conc0` `b_conc0` / `v_conc0` |
| 并发 16（32 个请求） | 91.04 / 90.84 tok/s | 90.95 tok/s | 同上 |
| 并发 64（128 个请求） | 123.79 / 123.69 tok/s | 123.70 tok/s | 同上 |

c4 的提升来自 M=2..8 的新 FP8 GEMV。

开 MTP（K=3）时的 4096 token prefill 为 5535 ms（草稿模型也要 prefill，多约 95 ms），4096 token prompt 后的 decode 见下一节。

### 何时不要开 MTP

MTP 的收益取决于草稿被接受的比例，以及每轮验证 4 个 token 的额外成本（K=3 一轮约 30.4 ms，不开 MTP 每 token 23.8 ms）。**单流时，所有测过的提示上 K=3 都更快**（最差的 story 接受率 0.48，仍有 80 tok/s，MTP=0 为 42）。**并发 ≥4 时 K=3 反而更慢**：

| 场景 | MTP=0 | MTP=3 | runs |
|---|---|---|---|
| 单流，随机 token prompt，`ignore_eos` 生成 512 token（`bench_deploy.py tpot`） | 23.84 ms/token（41.9 tok/s） | 7.7–12.1 ms/token（82–130 tok/s）；4096 token prompt 时 104–111 tok/s | `f_k0` / 复核 `../mtp-vchk-s/dm/runs/m3perf` |
| 并发 2（`vllm bench serve` random，默认采样） | 53.8 / 52.9 tok/s | 61.8 / 74.7 tok/s | `f_conc0` `n_t0` / `f_conc3` `n_t3` |
| 并发 4 | 79.98 / 78.76 tok/s | 55.8 / 57.3 tok/s | `n_conc0` `b_conc0` / `n_conc3` `f_conc3` |
| 并发 16 | 91.0 tok/s | 83.5 / 84.9 tok/s | 同上 |
| 并发 64 | 123.8 tok/s | 89.1 / 90.4 tok/s | 同上 |

注：本文早先版本把上表第一行写成 MTP=3 为 30.50 ms/token、比 MTP=0 慢。那是测量错误：`bench_deploy.py` 当时按 SSE chunk 间隔计时，而开 MTP 时一个 chunk 带一整轮（最多 K+1 个）token。脚本已改为按 `usage.completion_tokens` 计算每 token 时间（`tpot_ss_ms`，chunk 间隔另存为 `tpot_chunk_ss_ms`）。贪心生成的随机 prompt 输出重复性高，接受率很高，所以这一行也不是“难预测文本”的例子。

所以：交互式单用户（对话、代码、结构化输出）用 `MTP=3`；需要吞吐的多路服务用 `MTP=0`。vLLM 有按 batch 大小动态改 K 的选项（`num_speculative_tokens_per_batch_size`，`start.sh` 里是 `MTP_SCHED`），本可以自动做这种切换，但在这个 vLLM 版本上开启后，引擎在草稿模型 graph capture 时断言失败（`input_batch.make_dummy: assert 0 < num_reqs <= num_tokens`，调用栈里没有本插件的帧，runs `e_a`）。schedule 里写 K=0 或 K=1 都会失败，所以目前不能用。

另外，只要开了 MTP（包括默认的 `MTP=3`），vLLM 自带的参数检查就会拒绝带 `min_p > 0` 或 `logit_bias` 的请求，返回 HTTP 400（MTP=0 和 v1 正常）。需要这两个参数的客户端请用 `MTP=0`。

## 正确性

### MTP 输出与 MTP=0 逐字节相同（贪心）

验证路径里的每个算子，都与 M=1 的 decode 路径逐 token 位级相同：

- FP8 GEMV：M=2..8 时每列与 M=1 相同。
- GDN：多 token 验证 kernel，等价于逐 token 跑 M=1 kernel，包括回滚。
- attention：验证 query 拆成单 query 序列，走与 decode 相同的 3D 路径。
- lm_head 与 argmax：逐行相同；并列时取最小下标。
- 贪心验证：与 K=0 的融合 argmax 是同一个 kernel，用同一个并列规则。

各 kernel 的单元测试见各包，汇总在下文"改了什么"。

集成时还发现两个 vLLM 引擎层面的差别，它们会让长 prompt 的 prefill 在 MTP 开/关之间分块不同，因此第一个 token 就可能不同。这两处与我们的 kernel 无关，原版 vLLM 自己 MTP=0 和 MTP=3 之间也有：原版 runs `p_o0` / `p_o3` 中，914 token 的 prompt 第 0 个 token 就不同。

1. **block size 随 K 变化**：开 MTP 后 conv state 变宽（3+K 列），vLLM 自动选的 attention/mamba block size 是 784（K=0）、800（K=1..3）、816（K=4）。prefix caching 的 align 模式下，GDN prefill 在 block 边界处切分，block 不同，切分就不同。本部署固定使用 `--block-size 816`（`BLOCK_SIZE`，默认 816），这个值满足 K≤4 的需要。
2. **eagle block drop**：开 MTP 时，vLLM 调度器会把可缓存位置回退一个 block（`use_eagle_block_drop`），prefill 分块因此与 MTP=0 不同。本部署在 speculative config 中设置 `disable_eagle_block_drop: true`（`MTP_BLOCK_DROP=0`，为默认值）。vLLM 说明这一设置只影响 prefix cache 命中时最后一个 block 的复用，可能影响接受率，但不关闭草稿器。实测 5 个提示的接受率不变。

做了这两项之后的检查：

- `mtp_bench`：5 个对话提示，K=1/2/3/4 每轮的文本都与 K=0 逐字节相同。
- `vchk_cap.py`：16 个 prompt，含 914 / 3006 / 5519 / 6392 token 的长 prompt，completions 接口带 top-5 logprobs，所以走原版 rejection sampler，不走快速路径。c1 时 K=3 对 K=0 为 **16/16 全同，chosen 和 top-5 logprob 差均为 0.0000**（`c_vchk3` 对 `b_vchk0`）。K=4 的前 8 个 token 也是 16/16 相同（`c_vchk4`）。
- 修复前（`n_vchk3` 对 `n_vchk0`）只有 12/16 相同，4 个长 prompt 不同。

**逐字节相同的范围**：贪心采样，并且每一步的 batch 只含 decode、每个请求不超过 8 行（单流 K≤7，或 2 路 K=3 且这一步两路都只在 decode；两个请求同时开始时首个 prefill 被合批，任何 K 下都与单独运行不同）。以下情况不保证相同：

- 并发时 batch 组成变化。例如新请求的 prefill 与其他请求的 decode 合在同一步，或验证 batch 超过 8 行（c≥3 且 K=3），这时会走 M>8 的 Triton FP8 回退。在本部署的 MTP=0 下，同一个请求单独跑和与别人重叠跑，文本也会不同（`harness/trans.py`，runs `n_t0`），v1 和原版同样如此。
- 随机采样：草稿接受与否会改变随机数的使用。
- 命中的前缀缓存块是在与其他请求重叠的 prefill 中写入的。这时即使请求单独发送，文本也可能与全新服务器上的单独运行不同（MTP=0 同样如此，是 batch 噪声经前缀缓存传下来的，与 MTP 无关）。需要严格可复现时，用 `./start.sh serve --no-enable-prefix-caching` 启动（serve 后面的参数会原样传给 vllm serve）。

### 与原版 vLLM 的对比（`vchk_cap.py`，16 个 prompt，不带 min_tokens，贪心 + top-5 logprobs）

参照是 `../vchk-c/runs/orig`（原版 vLLM，block size 自选），表格由 `harness/vchk_table.py` 生成。

| 对比 | 逐位置 token 一致率 | 128 token 全同的 prompt 数 | 分叉前 chosen 的 mean \|Δlogprob\| |
|---|---|---|---|
| 原版 c1 对原版 c8（引擎自身噪声） | 0.614 | 7/16 | 0.0155 |
| c1：v1 对原版 | 0.647 | 6/16 | 0.0130 |
| c1：本部署 K=0 对原版 | 0.576 | 5/16 | 0.0127 |
| c1：本部署 K=3 对原版 | 0.576 | 5/16 | 0.0127 |
| c8：v1 / 本部署 K=0 / 本部署 K=3 对原版 | 0.599 / 0.690 / 0.563 | 5 / 5 / 5 | 0.0138 / 0.0143 / 0.0150 |
| c16：v1 / 本部署 K=0 / 本部署 K=3 对原版 | 0.613 / 0.583 / 0.495 | 6 / 5 / 3 | 0.0157 / 0.0144 / 0.0143 |
| c1：本部署 K=3 对本部署 K=0 | **1.000** | **16/16** | **0.0000** |

|Δlogprob| 都在引擎自身噪声（0.0155）以内。一致率不到 100% 的原因是：bf16 下近似并列的 token 一旦翻转，后面整段就分叉。本部署 K=0 的一致率与 v1 不同，是因为 block size 816 与原版的 784 不同，长 prompt 的 prefill 分块不同。block size 自选时，本部署 K=0 与 v1 在 c1 下的 2048 个 token 全部相同（`n_vchk0` 对 `../vchk-c/runs/opt`）。随机采样（c8，T=0.7）的文本正常（`s8.json`）。

## 改了什么（相对 v1）

v1 的全部内容（三个包 `vopt_fp8gemv`、`vopt_lmhead`、`vopt_rest`，补丁名与开关）保持不变，说明见 `README_v1.md`。v2 的插件 `plugin/` 由三个 builder 的包组装而成。每项都有关闭开关，遇到不支持的情况一律回退到 v1 或原版路径。

| 包 | 改动 | 开关（默认） | 来源与详细说明 |
|---|---|---|---|
| `vopt_fp8gemv` | **M=2..8 每列与 M=1 位级相同的 FP8 GEMV**：每个形状所有 M 都用 M=1 的 split-K (S,B)，归约树相同。M≥3 改走 MFMA 多列路径：`v_mfma_f32_4x4x1f32` 每个元素等于一次正确舍入的 fma，已在 GPU 上逐位验证与 fmaf 相同；x 广播到 16 个 block，权重解码与 M=1 相同。M=2 保留 M=1 的 fma_mix 链。M=4 的时间是 M=1 的 1.12 倍（v1 为 1.22 倍），验证中的 FP8 从 27.2 ms 降到 22.5 ms。M=1 的 kernel 与 v1 逐位相同。 | `VOPT_FP8_EXACT`（1；0 = v1 的按 M 调优表，不精确） | `../mtp-gemv/README.md` |
| `vopt_rest` | ① **多 token GDN 验证 kernel `gdn_spec`**：每层 1 次 launch，替换原版 Triton spec 路径约 400 个 kernel。逐 token 与 v1 的 1-token `gdn_decode` 位级相同，遵守 vLLM 的 slot 契约：从 `[i, num_accepted-1]` 读状态，把第 t 个 token 之后的状态写进 `[i,t]`，conv 窗口偏移，跳过 NULL slot。每轮 4.0 ms 降到 1.4–1.6 ms。② **验证 attention**：每个验证 token 作为单 query 序列交给原版 `unified_attention`，从而走与 decode 相同的 3D split-softmax 路径，逐 token 位级相同，也更快（每层 42 µs 降到 13 µs）。metadata 由一个小 kernel 写进持久 buffer，可放进 graph。③ 修复 v1 在 MTP + `--enforce-eager` 下的崩溃（strided 索引）。 | `VOPT_GDN_SPEC`（1）、`VOPT_ATTN_SPEC`（1）、`VOPT_ATTN_SPEC_MAXQ`（8）、`VOPT_GDN_NT`（auto） | `../mtp-ops/README.md` |
| `vopt_spec`（新包） | ① **草稿 lm_head**：词表前 151552 行加 33 个 added token，加载时生成草稿专用的 int4 副本（393 MiB）。一个 kernel 完成 GEMV 与 top-4，再对窗口内的候选用原 bf16 lm_head 重算，所以草稿等于 bf16 子集上的 argmax。每步 1.8 ms 降到 0.35 ms。② **贪心验证快速路径**：直接用验证 lm_head GEMV 尾部融合的 argmax 做接受判定，与 K=0 是同一个 kernel，一个 kernel 代替 5 个 Triton kernel。③ **GDN metadata 快速路径**：每个 mamba 组约 19 个小 op 合成 1 个 Triton kernel，验证前的 eager 段从 0.46 ms 降到 0.12 ms。草稿近似只影响接受率，不影响输出。 | `VOPT_SPEC`（1，0 = 整包关闭）、`VOPT_DRAFT_HEAD`（int4\|fp8\|bf16\|full\|stock）、`VOPT_DRAFT_VOCAB`（151552）、`VOPT_DRAFT_DELTA`（int4 为 3.0）、`VOPT_REJECT_FAST`（1）、`VOPT_GDN_MD`（1） | `../mtp-draft/README.md` |
| `vopt_lmhead` | lm_head 与 in_proj_ba 的 GEMV 支持到 M=8，逐行与 M=1 相同；M=7..8 拆成 4+(M−4) 两次 launch，argmax 仍然输出。这样 c=2、K=3（M=8）时不会掉回 wvSplitK（9–22 ms）。MTP 的 fc（bf16 5120×10240）在 M≤2 时改用 split-K GEMV（88 µs 降到约 73 µs）。 | `VOPT_LMHEAD_MAXM`（8，v1 为 6）、`VOPT_FC_GEMV`（1） | `../mtp-draft/README.md` |
| `start.sh` | 默认 `MTP=3`；固定 `--block-size 816`；speculative config 加 `disable_eagle_block_drop`；新增 `V1=1`（跑 v1 做 A/B）、`PLUGIN`、细调开关透传；端口 18340；容器名 `vllm-opt-deploy-mtp-*` | `MTP`、`BLOCK_SIZE`、`MTP_BLOCK_DROP`、`V1` | 本目录 |

在集成版里各部分对 K=3 的贡献（runs 见各 builder 目录）：v1 81.0 → 只加精确 GEMV 92.4 → 加 GDN/attention、vopt_lmhead 98.0 → 加 vopt_spec 112.8 tok/s。

### K=3 一轮的时间分解（torch profiler，30 个稳态轮次的中位数，`runs/ref_prof_k3/analysis.txt`）

| 段 | v1 | 本部署 |
|---|---|---|
| 验证 graph（M=4，629 个 kernel） | 33.18 ms | 25.67 ms，其中 FP8 22.70、GDN 1.44、lm_head 外的 bf16 0.77、attention 0.26 |
| 验证后的 eager 段（lm_head M=4，贪心验证） | 2.04 ms | 1.99 ms |
| 草稿 prefill graph | 2.55 ms | 0.97 ms |
| 草稿第 2、3 步 graph | 4.51 ms | 1.64 ms |
| 验证前的 eager 段 | 0.46 ms | 0.12 ms |
| **一轮** | **42.71 ms** | **30.44 ms** |

按平均每轮约 3.4 个 token 计算，约为 113 tok/s。剩下的主要差距是验证中的 FP8：22.7 ms，M=4 的字节下限为 16.8 ms。其中 gdn_out、attn_o 的带宽只有 770–850 GB/s，down 为 1025 GB/s。

## 怎么用

```bash
cd /data/tools/mi210-opt/vllm-opt/deploy-mtp

# 1) 预编译四个 HIP 扩展（不用 GPU，不用锁）。cache/ 里已有编译好的版本。
./build.sh

# 2) 服务（前台；默认 MTP=3，端口 18340，模型名 qwen38，只监听 127.0.0.1；默认持有 GPU 锁）
./start.sh serve
MTP=0 ./start.sh serve            # 多路吞吐场景：关闭 MTP
MTP=4 ./start.sh serve            # 代码/JSON 为主的单流场景
PORT=18341 ./start.sh serve --no-enable-prefix-caching   # 额外参数原样传给 vllm serve

# 3) 基准模式（取锁 → 后台启动 → 就绪后在主机上跑 job → 日志存 runs/<tag>/ → 删容器 → 放锁）
STEPS="mtp" MTP=3 ./start.sh bench mytag harness/job.sh                  # mtp_bench：5 个提示的速度、接受率、文本
STEPS="mtp t128 t4k ttft4k" MTP=0 ./start.sh bench mytag0 harness/job.sh
V1=1 STEPS="mtp" MTP=3 ./start.sh bench mytag_v1 harness/job.sh          # deploy v1 对照
STEPS="conc" CONC="4 16 64" MTP=0 ./start.sh bench mytag_c harness/job.sh
STEPS="vchk" MTP=3 ./start.sh bench mytag_v harness/job.sh               # 正确性采集（对照原版）
ORIG=1 MTP=0 STEPS="vchk" ./start.sh bench mytag_o harness/job.sh       # 原版 vLLM
PROFILE=1 MTP=3 ./start.sh bench mytag_p harness/job_prof.sh            # torch profiler
python3 harness/analyze_mtp.py cache/prof/mytag_p/*.gz                  # 每轮分段 + 逐 kernel
python3 harness/summary.py                                              # 汇总 runs/ 下 n_*/v_* 的 A/B
```

`harness/job.sh` 的步骤：`t128 t4k s128 ttft4k ppl ppl6k cap cap4 prof conc`（v1 已有），以及新增的：

- `mtp`：mtp_bench；
- `vchk`：vchk_cap 的 c1/c8/c16 贪心与 c8 采样；
- `vchk1`：只做 c1、`VTOK` 个 token；
- `trans`：batch 变化时的文本一致性。

启动日志里应能看到：
- `[vopt_fp8gemv] extension loaded (MAXM=8, EXACT=1)`
- `[vopt_spec] extension loaded`
- `[vopt_rest] ...`
- `[vopt_lmhead] lm_head patched (MAXM=8, greedy=True)`

### 开关

- **部署级**：
  - `MTP`（默认 3）
  - `BLOCK_SIZE`（816；`auto` = vLLM 自选，MTP 开关之间长 prompt 的输出可能不同）
  - `MTP_BLOCK_DROP`（0；1 = vLLM 默认的 eagle block drop）
  - `V1=1`（跑 deploy v1，使用 `cache_v1/`）
  - `ORIG=1` / `VOPT=0`（原版 vLLM）
  - `PORT`、`NAME`、`CACHE`、`PLUGIN`、`MAX_LEN`、`MAX_SEQS`、`GPU_LOCK`（默认 1）、`MAXLOCK`（bench 最长持锁秒数）
- **v1 的单项关闭**（默认都是 1）：`VOPT_FP8GEMV`、`VOPT_LMHEAD`、`VOPT_SAMPLER`、`VOPT_BF16SMALL`、`VOPT_NORMS`、`VOPT_ACT`、`VOPT_GDN`、`VOPT_ATTN`。
- **v2 新增**：
  - `VOPT_SPEC=0`：去掉 vopt_spec 包。
  - 直接透传进容器的开关：`VOPT_FP8_EXACT`、`VOPT_GDN_SPEC`、`VOPT_ATTN_SPEC`、`VOPT_DRAFT_HEAD`、`VOPT_DRAFT_VOCAB`、`VOPT_DRAFT_DELTA`、`VOPT_REJECT_FAST`、`VOPT_GDN_MD`、`VOPT_LMHEAD_MAXM`、`VOPT_FC_GEMV`。
  - 其他开关用 `EXTRA_ENV="K=V ..."` 传入，例如 `VOPT_FP8_CFG`、`VOPT_ATTN_SPEC_MAXQ`、`VOPT_GDN_NT`、`VOPT_FC_MAXM`、`VOPT_GDN_MD_CHECK=1`（与原版 metadata 逐字段比较）。
- 例：`MTP=3 VOPT_DRAFT_HEAD=fp8 VOPT_GDN_SPEC=0 ./start.sh serve`。
- 注意：`VOPT_FP8_EXACT=0`、`VOPT_GDN_SPEC=0`、`VOPT_ATTN_SPEC=0` 中任一项会让 MTP 输出不再与 MTP=0 相同。草稿相关开关（`VOPT_DRAFT_*`）只影响速度。

### 单元测试（都取 GPU 锁）

```bash
H=/data/tools/mi210-opt/vllm-opt
# vopt_rest：gdn_spec 对逐 token gdn_decode（R 1..8、T 1..8、多轮回滚）、验证 attention、逐 token 算子、graph replay
ROLE=mtp-ops bash $H/mtp-ops/harness/img_run.sh t "python3 -m vopt_rest.test_spec"
# vopt_spec / vopt_lmhead：草稿头 vs fp64、贪心验证 vs vLLM rejection_sample（含并列）、M=1..8 逐行精确、fc
ROLE=mtp-draft bash $H/mtp-draft/harness/img_run.sh u "python3 vopt_spec/test_spec_ops.py; python3 vopt_lmhead/test_rows_exact.py; python3 vopt_lmhead/test_fc.py"
# vopt_fp8gemv：M=1..8 × 5 个形状逐列 = M=1、fp64、graph replay（本目录 tests/ 有拷贝）
cd $H/mtp-gemv && EXTRA_ENV=PYTORCH_ROCM_ARCH=gfx90a bash harness/img_run.sh t "python3 /vopt/mtp-gemv/tests/fp8_mtp.py --real"
```

这些命令在各 builder 目录下运行，那里的插件与本目录 `plugin/` 逐文件相同。测试日志见：
- `../mtp-gemv/runs/t3.log`
- `../mtp-ops/runs/unit/unit_final.json`
- `../mtp-draft/runs/unit/unit_all.log`

结论都是 0 处位差。

## 已知限制 / 后续

- **MTP 在并发 ≥4 时更慢**（见"何时不要开 MTP"；单流时即使接受率低的文本也比 MTP=0 快）。vLLM 的动态 K（`MTP_SCHED`）在这个版本里启动即失败；要自动切换，需要修 vLLM v2 speculator 的 capture，或在插件里加一个按 batch 大小跳过草稿的补丁。
- **逐字节相同的前提**：贪心采样，每步每个请求 ≤8 行，且不与 prefill 混在同一步。否则走 M>8 的 FP8 Triton 回退，它与 M=1 不是位级相同。这一点与 v1、原版的 batch 噪声同级。另外，复用的前缀缓存块也须由无其他请求的 prefill 写入。
- **block size 816**：
  - MTP=0 的长 prompt 输出与 v1（784）不再逐位相同，但 |Δlogprob| 仍在引擎噪声内。
  - mamba page 补齐到 816 token 的 attention page，KV cache 因此略小：MTP=0 时 28.0 万 token（自选时 29.2 万，−4%），MTP=3 时 16.2 万（自选时 16.5 万）。速度不受影响。
  - `BLOCK_SIZE=auto` 可恢复 vLLM 自选的行为。
- **草稿副本占显存**：int4 草稿头占 393 MiB，在草稿模型加载时生成，所以只有 `MTP>0` 才有，KV cache 相应变小。
- **草稿词表前缀**：默认 151552，中文覆盖率 ≥99.8%。中文为主的负载可以设 `VOPT_DRAFT_VOCAB=248320`（int4 全词表，每步约 535 µs）。
- **剩余优化空间**：验证中的 FP8 GEMV 为 22.7 ms，字节下限 16.8 ms，gdn_out 与 attn_o 的带宽偏低。验证 lm_head 为 1.99 ms，下限 1.76 ms。要保持逐行精确，必须沿用 M=1 的运算顺序。
- **带 stop 字符串时 `completion_tokens` 多计 1**：开 MTP 时，文本相同，usage 报 66 个 token，MTP=0 报 65。这是原版 vLLM 对停止位置之后的草稿 token 的计数方式（v1 也一样），只影响按 token 计费或统计的场景。
- **开 MTP 时不支持 `min_p` 和 `logit_bias`**：vLLM 的投机解码参数检查会直接返回 HTTP 400，需要时用 `MTP=0`。
- 不要开 torch.compile（同 v1）。

## 文件

- `start.sh`：启动器；`build.sh`：预编译 4 个扩展。
- `plugin/`：`usercustomize.py`，以及 `vopt_fp8gemv`（来自 mtp-gemv）、`vopt_rest`（来自 mtp-ops）、`vopt_lmhead` 和 `vopt_spec`（来自 mtp-draft）。
- `harness/`：
  - v1 的工具：`job.sh`、`bench_deploy.py`、`capture.py`、`compare.py`、`analyze_trace.py` 等；
  - 新增：`mtp_bench.py`、`mtp_job.sh`、`vchk_cap.py`、`trans.py`、`summary.py`、`vchk_table.py`、`cmp_runs.py`、`analyze_mtp.py`、`eager_detail.py`、`job_prof.sh`。
- `tests/`：`test_prefill_ops.py`（v1），以及 `fp8_mtp.py`、`bf16_mtp.py`（mtp-gemv 的 GEMV 精确性测试拷贝）。
- `cache/`：本部署的扩展与 Triton 缓存；`cache_v1/`：`V1=1` 对照用。
- `runs/`：
  - `queue1..5.sh`/`.log`：测量队列。
  - `n_*`、`v_*`：第一轮 A/B。
  - `b_*`、`f_*`：block size 816 之后的复测。
  - `c_vchk*`：最终配置的正确性检查。
  - `p_o*`：原版 vLLM 的 MTP=0/3 对照。
  - `e_a`：动态 K 启动失败的日志。
  - `ref_prof_k3/`：profile 分解。
- `README_v1.md`、`runs_v1/`：v1 的说明与测量记录。
