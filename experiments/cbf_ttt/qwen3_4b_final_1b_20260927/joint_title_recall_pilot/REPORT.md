# 发布后论文标题回忆与 KV 隔离试点（2026-09-29）

## 问题与预注册协议

此前发布后论文 continuation 试点在 `new_only` 中只有 1/12 组出现有益写入。由于查询紧邻上下文，本轮用新的论文源组询问刚读论文的完整标题，并分别测量 attention KV 完整与清空 KV 但保留 session 快记忆的读取条件。后者隔离测试 `M'=αM+gΔW` 能否承载新论文信息。实施前的假设、四条件场景与 `0.005` NLL、至少 3/12 组的阶段门已写入 [`DUAL_GATE_TITLE_RECALL_PILOT.md`](../../../../DUAL_GATE_TITLE_RECALL_PILOT.md)。没有根据本轮结果修改阈值。

模型为 hku-gpu2 上最终的 Qwen3-4B 1B-token continued-pretraining checkpoint：`/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt`，bf16，2 张 NVIDIA RTX 5090。候选边界为第二个完整 4096-token chunk；每两块采一条，`α,g∈{0,0.5,1}`。同一 pending 候选在全部动作与两读取条件中共用。每组四篇不同论文，四类场景分别为 `both_relevant/old_only/new_only/neither_relevant`，train/dev/test 为 8/2/2 个源组。标题 NLL 由相同答案的 teacher forcing 计算。

## 来源与数据审计

官方 arXiv `cs.CL` API feed 的 SHA256 为 `9d647b67b617f065cdc89ab901851a798c54d6b79193f4440a4e30d157c5d4c4`；从 feed 前 300 条中排除前两批入选的 **88 个唯一 source ID** 后，以种子 119 打乱、顺序筛选。53 篇尝试后得到 48 篇：3 篇少于 4256 Qwen token，2 篇 API 标题未在 PDF 提取文本的前 4096 token 中匹配。入选论文首投时间为 2026-09-21 06:48:27 UTC 至 2026-09-24 10:49:37 UTC，48 个 ID 互异，全文长度 4313 至 73827 Qwen token；另以种子 120 分配源组和 split。当前 1B 续训语料中，48 个**精确标题**匹配为 **0**。来源逐篇信息见 [`documents.jsonl.meta.json`](documents.jsonl.meta.json)，场景与分片见 [`scenarios.jsonl.meta.json`](scenarios.jsonl.meta.json)、[`shard_manifest.json`](shard_manifest.json)，审计见 [`title_audit.json`](title_audit.json)。精确标题零匹配不等于全文去重证明。

## 执行与结果

远程 `tests.test_cbf_ttt`、`tests.test_cbf_title_recall`、`tests.test_cbf_joint_pilot_summary` 共 12 项 PyTorch/纯数据测试通过。单场景 smoke 的两个 future 均有有限损失，耗时 1.670 秒，最大 reserved 显存 14.094 GiB。正式试点 48/48 条于两卡 tmux 完成，GPU 日志显示约 2026-09-29 00:33:46—00:34:37（UTC+8），没有 OOM；逐标签平均 **1.255 秒**、p95 **1.689 秒**、最大 **1.865 秒**，最大 CUDA allocated/reserved 为 **12.974/14.098 GiB**。

预注册阶段门只使用 `memory_only`、固定另一门为 1 或 0 的四角点：

| 指标 | 结果 | 阈值 |
|---|---:|---:|
| `new_only` 有益写入：`J10−J11>0.005` | **1/12**，平均 **−0.02139 NLL** | ≥3/12 且均值 >0 |
| `old_only` 或 `neither` 无关写入有害：`J11−J10>0.005` | **8/12** | ≥3/12 |
| `old_only` 旧记忆保留有益：`J00−J10>0.005` | **0/12** | ≥3/12 |
| `neither` 清除有益：`J10−J00>0.005` | **9/12** | ≥3/12 |

写入和保留阶段门都**未通过**，联合阶段门也未通过。`memory_only` 四角点最优计数为 `00/01/10/11 = 22/20/6/0`；KV 完整则为 `19/7/10/12`，说明读取条件明显改变动作排序，但并没有建立稳定的快记忆写入增益。`new_title` 查询的平均 NLL，KV 完整的 `10/11` 为 0.577/0.580；仅快记忆为 4.965/4.986。不能从这一指标推断自由生成准确率。完整聚合见 [`summary.json`](summary.json) 和 [`gate.json`](gate.json)。

本地从 6 个逐条标签文件复算 `summary.json` 和 `gate.json`，分别与远程 SHA256 `44747e32202d092da9e0555b24824123e3995bc984b03c1f634a2dca299a73cd`、`dfe214c5df0f7ea1db72b4e650ff2c39be58c7e17d30f9fd07672a4e614c290e` 完全一致。原始 PDF、全文、场景及逐条标签保存在 hku-gpu2 的 `/home/ctj/cbf_ttt_title_recall_pilot_20260929`；本地逐条标签由本目录 `.gitignore` 排除，不上传 GitHub。

## 复现命令

在 hku-gpu2 的 `/home/ctj/cbf_ttt_joint_exp_20260927`、已激活 `/home/ctj/miniconda3/envs/cbf_ttt_train_py311` 环境中，按 [`MODIFICATION_LOG.md`](../../../../MODIFICATION_LOG.md) 的“发布后论文标题回忆与 KV 隔离试点”段依次执行数据下载、标题审计、场景构造、两卡分片。先用单场景 `collect-joint --grid 0,0.5,1 --every 2` smoke，再在两张 GPU 分别运行：

```bash
MODEL=/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt
ROOT=/home/ctj/cbf_ttt_title_recall_pilot_20260929
bash scripts/run_cbf_joint_pilot.sh "$MODEL" "$ROOT" 0 0
bash scripts/run_cbf_joint_pilot.sh "$MODEL" "$ROOT" 1 1
python -m scripts.summarize_cbf_joint_pilot --labels "$ROOT"/*_joint_labels.jsonl --output "$ROOT/summary.json"
python -m scripts.evaluate_cbf_title_recall_gate --labels "$ROOT"/*_joint_labels.jsonl --output "$ROOT/gate.json"
```

两条 `run` 命令需要并行，脚本会检查并拒绝覆盖已有标签；复现时使用新输出目录。原始 baseline 模型和默认配置未改，全部新增行为仅通过 `joint_title_recall_v1` 场景启用。

## 决策与限制

**停止正式反事实标签扩量、双输出控制器训练及公开基准对照。** 当前 `ΔW` 在全新论文标题任务上未表现为可稳定检索的会话写入。下一步应先研究候选更新的目标/梯度与信息写入机制，而不是用这批近乎单侧动作标签训练控制器。

KV 清空是人为隔离条件，标题精确匹配可能对快权重过难；`neither_relevant` 的未见第三篇标题本来不可从上下文确定，其比较只用于量化干扰，不能单独解释成合理预测。PDF 提取、论文发布时间和精确标题扫描都不能排除早期草稿或其他相似文本。12 组样本也只支持本轮阶段门结论，不能外推到所有检索任务。
