# Qwen3.8-27B-FP8 在 MI210 上的 vLLM 部署 v2：MTP 投机解码

> 仓库说明：本目录是服务器上部署目录 `/data/tools/mi210-opt/vllm-opt/deploy-mtp` 的源码拷贝（插件、启动脚本、测量脚本、单元测试）。r1 的插件和缓存（`plugin_r1/`、`cache_r1/`）只在服务器上，仓库里的旧 FP8 kernel 可以从 git 历史（提交 1e31ec7）取回。文中的 `runs/...` 运行记录、`cache/` 预编译缓存，以及 `../deploy`（v1）、`../mtp-*`、`../vchk-*` 等工作目录只在服务器上，没有放进仓库。`V1=1` 需要 v1 的插件目录（`V1_PLUGIN`）；仓库里没有 v1，可以用本插件的各个 `VOPT_*` 开关得到与 v1 相同的输出。主机路径可以用环境变量改：`MODEL_DIR`（模型目录）、`GPU_LOCK_FILE`（GPU 锁）、`FP8_OVERLAY`（可选，主机上原有的 fp8_utils.py 覆盖文件，只影响 M>8 的回退路径，不存在时自动跳过）。`harness/vllm_run.sh` 和 `harness/img_run.sh` 是开发时用的脚本，路径按服务器写死。

对象：官方 checkpoint `/data/model/Qwen3.8-27B-FP8`（自带 MTP 层），不重新量化，`--language-model-only`。
运行环境：vLLM 镜像 `2e7da1ad1c66`（vllm 0.30.0+rocm723），单卡 MI210（gfx90a）。

本目录 = deploy v1（`../deploy`，只读参考，未改动）的完整拷贝，加上 MTP 投机解码的全部改动。镜像本身不做任何修改：插件目录以只读 bind mount 挂载，经 `PYTHONPATH` 和 `usercustomize.py` 的 post-import hook 对 vLLM 打补丁；HIP 扩展用 `torch.utils.cpp_extension` 编译，缓存在 `cache/torch_ext`。v1 的原始说明在 `README_v1.md`，v1 的测量记录在 `runs_v1/`。

**r2（2026-11-20）**：FP8 GEMV 换成 `../opt3-fp8b` 的版本（L2 常驻的 split-K 工作区槽位，见下文"r2：FP8 GEMV 改动"），其余三个包不变。M=1 的结果与 r1 逐位相同，所以 MTP=0 的文本不变，MTP 输出仍与 MTP=0 逐字节相同。r1 的插件和扩展缓存保留在 `plugin_r1/`、`cache_r1/`，`FP8_R1=1` 切回 r1 做对照。r1 版的说明是 `README.md.pre-r2`。

## 一句话结论

- 单流 decode（5 个对话提示各自 tok/s 的算术平均，贪心）：**MTP K=3 约为 122 tok/s**（r2，7 轮中 6 轮为 121.0–122.6）。r1（合入 opt3-fp8b 之前）为 112.5–113.4，deploy v1 同样 K=3 为 80.4–81.3，v1 不开 MTP 为 42 tok/s。按 token 加权（总 token / 总 decode 时间）为 113（r1 104.8）。复核用的 5 个新短提示（regex、信件、rust、菜谱、k8s yaml）上，r2 K=3 为 112.1 tok/s（3 轮均值，同批交替的 r1 为 103.8）；另一组含两个约 5k token 长 prompt 的新提示为 106.0（`../opt3-vchk2-r/runs/b3`）。**更普遍的数字是 106–113 tok/s**，相对 r1 的约 +8% 在各组上都成立。
- **开 MTP 的输出与同一部署 MTP=0 的输出逐字节相同**（贪心）。K=1、2、3、4 全部成立，包括 914–6392 token 的长 prompt，连 logprobs 都相同。deploy v1 开 MTP 时有 3/5 个提示不同。
- MTP=0 时速度（TPOT r2 23.66–23.68 ms，r1 23.74–23.90）、prefill、并发 c4/c16/c64 都没有比 v1 退步。c4 反而更快（r2 81–83 tok/s，r1 78.8–80.0，v1 为 71.5）。
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

### r2：合入 opt3-fp8b 后（同样 5 个提示，tok/s）

| 配置 | 平均 | code | story | json | edit | repeat | 与 r1 K=0 文本相同 | runs |
|---|---|---|---|---|---|---|---|---|
| **r2 K=3** | **121.9 / 122.6 / 117.6 / 121.7 / 122.2 / 121.0 / 122.3** | 119.6 | 86.4 | 132.1 | 131.9 | 140.1 | 5/5 | `r2_k3a`；`../opt3-fp8b/runs/e_b3a..c`；`../opt3-vchk2-s/runs/b3a..c` |
| r2 K=0 | 42.3 / 42.4 / 42.5 / 42.5 | 42.3 | 42.4 | 42.3 | 42.8 | 42.8 | 5/5 | `r2_k0a`；`../opt3-fp8b/runs/e_b0a`；`../opt3-vchk2-s/runs/b0a` `b0b` |
| r2 K=1 | 75.9 | 75.9 | 69.6 | 77.4 | 77.6 | 78.8 | 5/5 | `../opt3-vchk2-r/runs/b1` |
| r2 K=2 | 101.6 | 98.0 | 83.2 | 108.1 | 109.0 | 109.8 | 5/5 | `../opt3-vchk2-r/runs/b2` |
| r2 K=4 | 123.7 | 121.1 | 78.2 | 138.2 | 130.9 | 150.1 | 5/5 | `../opt3-vchk2-r/runs/b4` |
| r1 K=3（同批交替） | 113.0 / 110.9 / 113.1 | 110.6 | 79.7 | 122.3 | 122.6 | 129.6 | 5/5 | `../opt3-vchk2-s/runs/r3a..c` |

"平均"列是各轮的 5 提示算术平均；各提示的列是这些轮的中位数。K=1/2/4 和部分 K=3 来自两个独立复核目录，它们的插件与本目录 `plugin/` 逐文件相同。接受的 token 数与 r1 完全相同（K=3 为 1948/2658），提速全部来自每轮更短。单轮偶尔有某个提示掉到 110 左右（例如 `e_b3b` 的 edit 109.5），r1 也有同样的现象（`r3b` 的 code 101.6），属于服务器噪声。

r2 的其他测量（`../opt3-vchk2-s/runs/summ.txt`，每个配置 3 轮，与 r1 交替）：

| 指标 | r2 | r1 |
|---|---|---|
| K=3，5 个新短提示（regex、信件、rust、菜谱、k8s yaml，384 token） | 112.1 tok/s | 103.8 |
| K=3 并发 4 / 16（`vllm bench serve` random 512/128） | 58.6 / 84.7 tok/s | 55.6 / 79.8 |
| K=0 TPOT（128 token prompt） | 23.66–23.68 ms | 23.74–23.77 |
| K=0 并发 4 / 16 / 64 | 82.0 / 89.8 / 123.6 tok/s | 79.8 / 90.3 / 123.8 |
| K=0 4096 token prefill | 5439–5441 ms | 5439–5441 |

另一组新提示（`../opt3-vchk2-r/runs/b1..b4` 的 `fresh.json`：rust、中文、nginx 日志 5.1k token、单摆、文档问答 5.0k token，各 512 token）：K=1/2/3/4 为 71.3 / 92.7 / 106.0 / 102.1 tok/s。

接受率（本部署，K=3）：code 0.79、story 0.48、json 0.92、edit 0.90、repeat 1.00。除 json（v1 为 0.94）外与 v1 相同。json 的差别不是草稿头造成的（`VOPT_SPEC=0` 也是 0.92），而是精确 GEMV 让生成的文本略有不同；接受率随文本变化。同理，5.1k token 长文摘要提示在默认配置下文本与 K=0 相同、接受率 0.69（96.9 tok/s），改用 `BLOCK_SIZE=auto` 或 `MTP_BLOCK_DROP=1` 时生成另一段同样合理的文本，接受率 0.79（约 106 tok/s）。

**推荐 K=3 的理由**：K=4 平均只多 2.5%（r2 只多约 1.5%：123.7 对 122），但 story 这类难预测的文本从 79.9 掉到 73 tok/s；K 越大，并发时和难预测文本上的损失越大，KV cache 也越小（block size 自选时，K=0/3/4 分别约 29.2 万 / 16.5 万 / 14.5 万 token）。如果负载以代码、JSON、复述类文本为主，可以用 `MTP=4`。

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

MTP 的收益取决于草稿被接受的比例，以及每轮验证 4 个 token 的额外成本（K=3 一轮 r2 约 28.1 ms、r1 约 30.4 ms，不开 MTP 每 token 23.7 ms）。**单流时，所有测过的提示上 K=3 都更快**（最差的 story 接受率 0.48，仍有 80 tok/s，MTP=0 为 42）。**并发 ≥4 时 K=3 反而更慢**（下表为 r1 的数据；r2 的 c4/c16 见上文"r2"一节，结论不变）：

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
| `vopt_fp8gemv` | **M=2..8 每列与 M=1 位级相同的 FP8 GEMV**：每个形状所有 M 都用 M=1 的 split-K (S,B)，归约树相同。M≥3 改走 MFMA 多列路径：`v_mfma_f32_4x4x1f32` 每个元素等于一次正确舍入的 fma，已在 GPU 上逐位验证与 fmaf 相同；x 广播到 16 个 block，权重解码与 M=1 相同。M=2 保留 M=1 的 fma_mix 链。M=4 的时间是 M=1 的 1.12 倍（v1 为 1.22 倍），验证中的 FP8 从 27.2 ms 降到 22.5 ms。M=1 的 kernel 与 v1 逐位相同。**r2**：split-K 工作区改为 L2 常驻的槽位，加指定归约块、单缓冲 MFMA、原子 ticket 定序，验证中的 FP8 再从 22.7 ms 降到 20.6 ms，结果逐位不变（见下文"r2：FP8 GEMV 改动"）。 | `VOPT_FP8_EXACT`（1；0 = v1 的按 M 调优表，不精确）；`FP8_R1=1`（r1 版本） | `../mtp-gemv/README.md`；r2：`../opt3-fp8b`（生成脚本 `../opt3-fp8/runs/final_apply.py`） |
| `vopt_rest` | ① **多 token GDN 验证 kernel `gdn_spec`**：每层 1 次 launch，替换原版 Triton spec 路径约 400 个 kernel。逐 token 与 v1 的 1-token `gdn_decode` 位级相同，遵守 vLLM 的 slot 契约：从 `[i, num_accepted-1]` 读状态，把第 t 个 token 之后的状态写进 `[i,t]`，conv 窗口偏移，跳过 NULL slot。每轮 4.0 ms 降到 1.4–1.6 ms。② **验证 attention**：每个验证 token 作为单 query 序列交给原版 `unified_attention`，从而走与 decode 相同的 3D split-softmax 路径，逐 token 位级相同，也更快（每层 42 µs 降到 13 µs）。metadata 由一个小 kernel 写进持久 buffer，可放进 graph。③ 修复 v1 在 MTP + `--enforce-eager` 下的崩溃（strided 索引）。 | `VOPT_GDN_SPEC`（1）、`VOPT_ATTN_SPEC`（1）、`VOPT_ATTN_SPEC_MAXQ`（8）、`VOPT_GDN_NT`（auto） | `../mtp-ops/README.md` |
| `vopt_spec`（新包） | ① **草稿 lm_head**：词表前 151552 行加 33 个 added token，加载时生成草稿专用的 int4 副本（393 MiB）。一个 kernel 完成 GEMV 与 top-4，再对窗口内的候选用原 bf16 lm_head 重算，所以草稿等于 bf16 子集上的 argmax。每步 1.8 ms 降到 0.35 ms。② **贪心验证快速路径**：直接用验证 lm_head GEMV 尾部融合的 argmax 做接受判定，与 K=0 是同一个 kernel，一个 kernel 代替 5 个 Triton kernel。③ **GDN metadata 快速路径**：每个 mamba 组约 19 个小 op 合成 1 个 Triton kernel，验证前的 eager 段从 0.46 ms 降到 0.12 ms。草稿近似只影响接受率，不影响输出。 | `VOPT_SPEC`（1，0 = 整包关闭）、`VOPT_DRAFT_HEAD`（int4\|fp8\|bf16\|full\|stock）、`VOPT_DRAFT_VOCAB`（151552）、`VOPT_DRAFT_DELTA`（int4 为 3.0）、`VOPT_REJECT_FAST`（1）、`VOPT_GDN_MD`（1） | `../mtp-draft/README.md` |
| `vopt_lmhead` | lm_head 与 in_proj_ba 的 GEMV 支持到 M=8，逐行与 M=1 相同；M=7..8 拆成 4+(M−4) 两次 launch，argmax 仍然输出。这样 c=2、K=3（M=8）时不会掉回 wvSplitK（9–22 ms）。MTP 的 fc（bf16 5120×10240）在 M≤2 时改用 split-K GEMV（88 µs 降到约 73 µs）。 | `VOPT_LMHEAD_MAXM`（8，v1 为 6）、`VOPT_FC_GEMV`（1） | `../mtp-draft/README.md` |
| `start.sh` | 默认 `MTP=3`；固定 `--block-size 816`；speculative config 加 `disable_eagle_block_drop`；新增 `V1=1`（跑 v1 做 A/B）、`FP8_R1=1`（r2 新增：r1 的 FP8 GEMV）、`PLUGIN`、细调开关透传；端口 18340；容器名 `vllm-opt-deploy-mtp-*` | `MTP`、`BLOCK_SIZE`、`MTP_BLOCK_DROP`、`V1`、`FP8_R1` | 本目录 |

在集成版里各部分对 K=3 的贡献（runs 见各 builder 目录）：v1 81.0 → 只加精确 GEMV 92.4 → 加 GDN/attention、vopt_lmhead 98.0 → 加 vopt_spec 112.8 tok/s → r2 的 FP8 GEMV 约 122 tok/s。

### r2：FP8 GEMV 改动（`../opt3-fp8b`）

**r1 为什么慢**：r1 的 split-K 给每个 row group 一块独立的 fp32 工作区，M=4 时一次调用约有 2.8 MB 脏数据。这些 cache line 在权重流的中途被写回 HBM，和权重读取抢带宽。权重本身只读一次、不进 L2 复用，所以这部分写回完全是额外开销。

**r2 的四处改动**（每一处都不改变任何一次浮点运算的顺序）：

1. **L2 常驻的工作区槽位**：row group rg 使用槽位 rg % 64（`VOPT_WS_NSLOT`，默认 64），同时存在的部分和只有约 64 组，工作区留在 L2，不再写回。复用槽位的块先等上一个使用者（rg−64）的归约块读完；每个槽位有使用计数，最后一次使用时清零，graph replay 安全。row group 不超过 96 的形状（down，N=5120，80 组）每组独占一个槽位，完全没有等待。
2. **指定归约块**：当每个 wave 最多只处理一个 128-k 块时（gate_up、qkvz、attn_qkv、gdn_out/attn_o），由 row group 内工作序号最大的块 b=B−1 归约。其他块写部分和、release fence、无返回原子加计数后直接退出；归约块等齐 B−1 份后按 ((0+v0)+…+v_{B−2})+v_{B−1} 求和，与 r1 的"最后到达者"顺序相同。down 的每个 wave 要流 2–3 个块，仍用最后到达者。
3. **单缓冲 MFMA**：每个 wave 最多一个块时，改用单缓冲、无循环的 MFMA kernel（45 VGPR，occupancy 8）。M=4 的 gate_up 从 139 降到 132 µs，M=8 从 167 降到 149 µs。
4. **原子 ticket 定序**：可能等待的调用中，每个块开始时用一次 atomicInc 取工作序号，不用 blockIdx.x。所有等待都只等序号更小的工作，而取到更小序号的块必然已经驻留并在运行，所以前进性不依赖硬件按 blockIdx 顺序派发 workgroup。最后一个块取号时计数器回绕到 0，不需要另外清零。不会等待的调用（B=1，或每组独占槽位且没有指定归约块）走静态版本，不含 ticket 代码。

**精确性**：所有形状、M=1..8 的每一行都与 M=1 调用逐位相同，M=1 与 r1 逐位相同。两个独立复核（`../opt3-vchk`、`../opt3-vchk2-r`/`-s`）覆盖了槽位数边界值（NSLOT=1/2/3/7/64/128）、打乱 ticket 顺序（`VOPT_DBG_SHUFFLE`）、后台负载、限制 CU 数的压力测试，以及 600 s 混合请求 soak（883 个请求，0 错误）。端到端：K=0 文本与 r1 相同，K=1..4 与 K=0 相同。

**效果**（K=3，torch profiler，每轮验证 graph 中的 FP8，单位 ms）：

| 角色 | r1 | r2 |
|---|---|---|
| gdn_in_qkvz | 3.70 | 3.21 |
| gdn_out | 1.94 | 1.88 |
| mlp_gate_up | 9.76 | 9.12 |
| mlp_down | 5.63 | 4.96 |
| attn_qkv | 1.10 | 0.96 |
| attn_o | 0.60 | 0.52 |
| **合计** | **22.7** | **20.6** |

GDN kernel 没改，但也从 1.44 降到 1.23 ms，因为写回争用少了。一轮从 30.4 ms 降到 28.1 ms。

**注意**：

- 失败模式是挂起（spin-wait 等不到），不是错误结果。调试时可以加编译选项 `-DVOPT_SPIN_MAX=n`：每个等待最多轮询 n 次，超时就置错误标志并放弃。
- `VOPT_BMAJOR=1`（旧的 b-major 块顺序）会破坏"只等更小序号"的论证，现在直接编译报错。
- `VOPT_WS_NSLOT` 只能在编译期设置，不是运行时环境变量。它经 `VOPT_FP8_CFLAGS` 传给编译器，`build.sh` 不透传这个变量。要试别的槽位数，需要单独的缓存目录，首次启动会编译约 5 分钟：`EXTRA_ENV="VOPT_FP8_CFLAGS=-DVOPT_WS_NSLOT=32" CACHE=$PWD/cache_ns32 ./start.sh serve`。
- `FP8_R1=1` 改用 `plugin_r1/` 与 `cache_r1/`（r1 的扩展已编译好），其他参数不变。

### K=3 一轮的时间分解（torch profiler，30 个稳态轮次的中位数；r1：`runs/ref_prof_k3/analysis.txt`；r2：`../opt3-fp8b/runs/e_pb2_analysis.txt`，复测 `e_pb_analysis.txt` 为 28.16 ms）

| 段 | v1 | r1 | r2 |
|---|---|---|---|
| 验证 graph（M=4，629 个 kernel） | 33.18 ms | 25.67 ms，其中 FP8 22.70、GDN 1.44、lm_head 外的 bf16 0.77、attention 0.26 | 23.41 ms，其中 FP8 20.64、GDN 1.23、lm_head 外的 bf16 0.74、attention 0.30 |
| 验证后的 eager 段（lm_head M=4，贪心验证） | 2.04 ms | 1.99 ms | 1.99 ms |
| 草稿 prefill graph | 2.55 ms | 0.97 ms | 0.94 ms |
| 草稿第 2、3 步 graph | 4.51 ms | 1.64 ms | 1.62 ms |
| 验证前的 eager 段 | 0.46 ms | 0.12 ms | 0.13 ms |
| **一轮** | **42.71 ms** | **30.44 ms** | **28.12 ms** |

按平均每轮约 3.4 个 token 计算，r1 约为 113 tok/s，r2 约为 122 tok/s。剩下的主要部分仍是验证中的 FP8：20.6 ms，M=4 的字节下限为 16.8 ms（24.3 GB / 1.446 TB/s）。各角色的带宽：qkvz 1261、gate_up 1251、attn_qkv 1230、down 1148、attn_o 970、gdn_out 803 GB/s。

**硬件利用率与现实余量**（`/data/tools/mi210-opt/peak/hwutil`，rocprofv3 计数器，r1 的 kernel 单独运行）：

- FP8 GEMV M=4 的 DRAM 流量为 0.57–0.97 TB/s（峰值 1.446），VALU 发射率 26–44%，MFMA 忙 12–21%。
- lm_head bf16 M=4 为 1.06 TB/s，VALU 53%；int4 草稿头为 0.95 TB/s。

没有一个 kernel 受算力限制，瓶颈都是访存。图内的实际带宽比单独运行时高，因为单独运行是冷启动、每次只有一个 launch。r2 验证 FP8 平均 1.18 TB/s，为峰值的 82%。gate_up、qkvz、attn_qkv 已到 1.23–1.26 TB/s，接近 gfx90a 上流式 kernel 实际能达到的 1.2–1.3 TB/s。所以 16.8 ms 的字节下限并不现实，大形状已经没有多少空间。剩下的余量在 down（1.15 TB/s），以及 gdn_out、attn_o 这两个小形状（launch 尾部占比大）。按 1.25 TB/s 估算，FP8 现实的余量约 1.2 ms/轮（约 4%），而不是到字节下限的 3.8 ms。

### 未合入：gdn_spec2

同期还试过重写的 GDN 验证 kernel `gdn_spec2`（`../opt3-rest`，`VOPT_GDN_SPEC2=1`），结果与 `gdn_spec` 逐位相同。它分三段：先算所有 token 的卷积、归一化和门控，再不加 barrier 地做全部 token 的 delta-rule 递推，最后一次做完 gated RMSNorm。每个 token 的 LDS barrier 从 4 次减到 2 次，还可以把 in_proj_ba GEMV 融进 kernel。但端到端 K=3 为 113.1–113.3 tok/s，对照为 113.3–113.7（`../opt3-rest/runs/e_new3*`、`e_base3*`），没有收益。原因是每轮的 fp32 状态写回约 0.75 GB，集中写出时与相邻的 FP8 GEMV 抢 HBM 带宽，kernel 自己省下的时间被邻居变慢抵消了。opt3-rest 保留作参考，没有合入。

## 怎么用

```bash
cd /data/tools/mi210-opt/vllm-opt/deploy-mtp

# 1) 预编译四个 HIP 扩展（不用 GPU，不用锁）。cache/ 里已有编译好的版本。
./build.sh

# 2) 服务（前台；默认 MTP=3，端口 18340，模型名 qwen38，只监听 127.0.0.1；默认持有 GPU 锁）
./start.sh serve
MTP=0 ./start.sh serve            # 多路吞吐场景：关闭 MTP
MTP=4 ./start.sh serve            # 代码/JSON 为主的单流场景
FP8_R1=1 ./start.sh serve         # r1 的 FP8 GEMV（plugin_r1/ + cache_r1/），A/B 用
PORT=18341 ./start.sh serve --no-enable-prefix-caching   # 额外参数原样传给 vllm serve

# 3) 基准模式（取锁 → 后台启动 → 就绪后在主机上跑 job → 日志存 runs/<tag>/ → 删容器 → 放锁）
STEPS="mtp" MTP=3 ./start.sh bench mytag harness/job.sh                  # mtp_bench：5 个提示的速度、接受率、文本
STEPS="mtp t128 t4k ttft4k" MTP=0 ./start.sh bench mytag0 harness/job.sh
V1=1 STEPS="mtp" MTP=3 ./start.sh bench mytag_v1 harness/job.sh          # deploy v1 对照
FP8_R1=1 STEPS="mtp" MTP=3 ./start.sh bench mytag_r1 harness/job.sh      # r1 FP8 GEMV 对照
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
  - `FP8_R1=1`（r1 的 FP8 GEMV：`plugin_r1/` + `cache_r1/`；`FP8_R1=1 ./build.sh` 编译它）
  - `ORIG=1` / `VOPT=0`（原版 vLLM）
  - `PORT`、`NAME`、`CACHE`、`PLUGIN`、`MAX_LEN`、`MAX_SEQS`、`GPU_LOCK`（默认 1）、`MAXLOCK`（bench 最长持锁秒数）
- **v1 的单项关闭**（默认都是 1）：`VOPT_FP8GEMV`、`VOPT_LMHEAD`、`VOPT_SAMPLER`、`VOPT_BF16SMALL`、`VOPT_NORMS`、`VOPT_ACT`、`VOPT_GDN`、`VOPT_ATTN`。
- **v2 新增**：
  - `VOPT_SPEC=0`：去掉 vopt_spec 包。
  - 直接透传进容器的开关：`VOPT_FP8_EXACT`、`VOPT_GDN_SPEC`、`VOPT_ATTN_SPEC`、`VOPT_DRAFT_HEAD`、`VOPT_DRAFT_VOCAB`、`VOPT_DRAFT_DELTA`、`VOPT_REJECT_FAST`、`VOPT_GDN_MD`、`VOPT_LMHEAD_MAXM`、`VOPT_FC_GEMV`。
  - 其他开关用 `EXTRA_ENV="K=V ..."` 传入，例如 `VOPT_FP8_CFG`、`VOPT_FP8_CFLAGS`（FP8 GEMV 的编译选项，如 `-DVOPT_WS_NSLOT=32`，需配合单独的 `CACHE`）、`VOPT_ATTN_SPEC_MAXQ`、`VOPT_GDN_NT`、`VOPT_FC_MAXM`、`VOPT_GDN_MD_CHECK=1`（与原版 metadata 逐字段比较）。
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

这些命令在各 builder 目录下运行，那里的插件与本目录 `plugin/` 逐文件相同；`vopt_fp8gemv` 除外，`../mtp-gemv` 里是 r1 版本，与 `plugin_r1/` 相同。r2 的 FP8 GEMV 测试在 `../opt3-fp8b/tests/`（`fp8_mtp.py`、`stress.py`、`bgload.py`、`gemv_fuzz.py`、`tk_check.py`），复核日志在 `../opt3-vchk/runs/unit/` 和 `../opt3-vchk2-r/runs/unit/`。测试日志见：
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
- **剩余优化空间**：验证中的 FP8 GEMV 为 20.6 ms（r1 22.7），字节下限 16.8 ms，但现实余量约 1.2 ms（见"硬件利用率与现实余量"），主要在 down、gdn_out、attn_o。验证 lm_head 为 1.99 ms，下限 1.76 ms。要保持逐行精确，必须沿用 M=1 的运算顺序。
- **带 stop 字符串时 `completion_tokens` 多计 1**：开 MTP 时，文本相同，usage 报 66 个 token，MTP=0 报 65。这是原版 vLLM 对停止位置之后的草稿 token 的计数方式（v1 也一样），只影响按 token 计费或统计的场景。
- **开 MTP 时不支持 `min_p` 和 `logit_bias`**：vLLM 的投机解码参数检查会直接返回 HTTP 400，需要时用 `MTP=0`。
- **r2 FP8 GEMV 的 spin-wait**：出问题时表现为挂起，而不是错误结果（见"r2：FP8 GEMV 改动"的注意事项）。两个复核都没有发现挂起；需要排除这个因素时用 `FP8_R1=1`。
- 不要开 torch.compile（同 v1）。

## 文件

- `start.sh`：启动器；`build.sh`：预编译 4 个扩展。r2 之前的版本是 `start.sh.pre-r2`、`build.sh.pre-r2`、`README.md.pre-r2`。
- `plugin_r1/`、`cache_r1/`：r1 的完整插件目录和扩展缓存（`FP8_R1=1`）。
- `plugin/`：`usercustomize.py`，以及 `vopt_fp8gemv`（r2 来自 opt3-fp8b，r1 来自 mtp-gemv）、`vopt_rest`（来自 mtp-ops）、`vopt_lmhead` 和 `vopt_spec`（来自 mtp-draft）。
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
  - `r2go.sh`/`.log`、`r2_k3a`、`r2_k0a`：r2 合入后的检查（K=3、K=0）。
- `README_v1.md`、`runs_v1/`：v1 的说明与测量记录。
