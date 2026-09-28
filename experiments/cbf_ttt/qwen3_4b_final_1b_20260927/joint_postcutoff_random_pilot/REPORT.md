# 发布后论文双门控写入效应试点（2026-09-28）

## 目标、模型和来源

按标签采集前写定的 [`DUAL_GATE_POSTCUTOFF_PILOT.md`](../../../../DUAL_GATE_POSTCUTOFF_PILOT.md)，使用真正晚于 [Qwen3 官方 2025-04-29 发布日](https://qwenlm.github.io/blog/qwen3/) 首次提交的公开论文，核查新候选 `ΔW` 是否对会话后续文本有正收益。模型为 hku-gpu2 上 Qwen3-4B In-Place TTT 1B-token 最终 checkpoint `/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt`。实验协议 `joint_postcutoff_v1` 是 opt-in，原始 baseline 和旧标签不变。

从 [arXiv 官方 Atom API](https://export.arxiv.org/api/query) 的 `cs.CL`、2026-09-01 至 09-26 首投查询中按首投日期取前 300 条候选，以种子 118 打乱后依次下载 PDF、`pdftotext` 提取文本，收集至少 4256 Qwen tokens 的 48 篇不同论文。共尝试 51 篇，3 篇因长度不足排除；入选论文首投于 2026-09-21 至 09-24，最短 4432、token 中位数 9503。官方 Atom XML SHA256 为 `9d647b67b617f065cdc89ab901851a798c54d6b79193f4440a4e30d157c5d4c4`。来源 ID、日期、PDF URL、PDF/文本哈希及 token 数见 [`documents.jsonl.meta.json`](documents.jsonl.meta.json)。针对本次 1B 续训语料的大小写不敏感**精确标题**扫描为 0/48 匹配，见 [`title_audit.json`](title_audit.json)。发布日期与标题扫描不能证明与所有早期网页或基础预训练语料全文零重合；论文也可能有此前公开的相似草稿。

每组四篇 A/B/C/D，各取 4096-token 已见前缀及 32+128-token 未见续写，构造 `both_relevant / old_only / new_only / neither_relevant` 四条件。12 组按源组划分 train/dev/test=8/2/2，共 48 场景，第二边界枚举 `α,g∈{0,0.5,1}`。目标 128-token 完整片段不在可见 context 中。场景文件 SHA256 `73903bedb138dc966f71b495236ee23c1c13b4a75125b49bd1fe9ece01ba6ae5`，构造元数据见 [`scenarios.jsonl.meta.json`](scenarios.jsonl.meta.json)。两张 5090 的 worker 于服务器时间 08:54:59 启动，08:55:43/44 结束；48/48 条成功，无 OOM。单场景 smoke 为 1.420 秒、13.854 GiB 峰值 reserved。

先前按 API 时间顺序选取前 48 篇的链路预检已经产生标签，但**在查看该批损失前**发现它不符合预设的固定种子抽样；该轮结果未用于以下阶段门。本报告只针对修正抽样后在独立目录采集的主试点，抽样修正提交为 `cea5429`，早于本批标签生成。

## 固定另一轴后的效应与阶段门

写入收益定义为固定旧记忆保留 `α=1` 时的 `G_write=J(1,0)−J(1,1)`；正值表示写入当前 `ΔW` 降低 NLL。保留收益定义为固定不写入 `g=0` 时的 `G_keep=J(0,0)−J(1,0)`；正值表示保留旧会话快记忆降低 NLL。因而不会用 `11` 对 `00` 的差值混合两种控制作用。

| 指标 | 结果 |
|---|---:|
| `new_only` 写入收益 >0.005 NLL 的独立源组 | **1/12**，预设至少 3/12 |
| `new_only` 平均写入收益 | **−0.01193 NLL**，预设 >0 |
| `old_only` 或 `neither_relevant` 写入损害 >0.005 的独立源组 | 12/12，预设至少 3/12 |
| `old_only` 保留收益 >0.005 的源组 | 2/12，预设至少 3/12 |
| `neither_relevant` 清除收益 >0.005 的源组 | 6/12，预设至少 3/12 |
| 四角点最优 `00/01/10/11` | 28/4/13/3 |
| 每条标签平均 / p95 耗时 | 1.075 / 1.423 秒 |
| 最大 CUDA allocated / reserved | 12.648 / 13.854 GiB |

`both_relevant` 平均写入效应为 −0.01465，`old_only` 为 −0.01903，`neither_relevant` 为 −0.03809 NLL。完整聚合见 [`summary.json`](summary.json)，预注册阶段门的机器可读判定见 [`gate.json`](gate.json)。在本地从保存的逐条标签独立复算，`gate.json` 与服务器逐字节相同，`summary.json` 的数据结构相等。

针对 bf16 小差值的数值风险，从**训练分组**选一个正写入和一个负写入例，在同一 checkpoint 上用 float32 重跑 `0/1` 四角点：正例的写入收益 bf16/float32 为 +0.00882/+0.00932，负例为 −0.02993/−0.03255，符号一致。float32 每条约 1.89 秒，峰值 reserved 27.320 GiB。这只能检查两个例子的计算精度，不能证明其余 46 条不受精度影响。

**写入阶段门、保留阶段门和联合阶段门均未通过。停止正式标签扩量及联合控制器训练。** 这里显示当前 1B 续训 checkpoint 对这些新论文的即时 `ΔW` 多数没有带来续写 NLL 收益，尤其对无关候选常有损害；不能据此断言所有推理任务都应关闭 TTT。下一步应先检查候选更新的任务相关性、尺度和目标设计，找到可复现的正写入机制，再在新的论文及任务上重新预注册试点。已经查看的 dev/test 只作探索，不再用于正式结论。

## 复现和数据位置

下载、审计、构造、两卡采集及聚合命令见 [`DUAL_GATE_POSTCUTOFF_PILOT.md`](../../../../DUAL_GATE_POSTCUTOFF_PILOT.md)。输出目录为 `/home/ctj/cbf_ttt_postcutoff_random_pilot_20260927`；原始 PDF 缓存位于首次链路预检目录，由主试点复用。官方元数据 API 对 hku-gpu2 返回 HTTP 406，因此本机保存官方 Atom XML 后传至服务器，`--api-feed` 核对查询并记录 XML 哈希。重新运行请使用新输出目录，以免覆盖已有标签。

原始 PDF、提取全文、预分词场景和逐样本标签只在 hku-gpu2 与本地忽略目录保存；GitHub 只提交代码、公开论文来源元数据、聚合数值和本报告。此前自动审批拒绝上传完整派生标签，本轮未绕过该限制。
