# 双门控机制拆分与自然 continuation 小试点（2026-09-27）

## 目标与假设

按照执行前的 [`DUAL_GATE_MECHANISM_PILOT.md`](../../../../DUAL_GATE_MECHANISM_PILOT.md) 做两项诊断：对 `joint_v2` 候选边界后的 gap 写入进行冻结对照，以及用自然文本 continuation NLL 检查旧快记忆保留率 `α` 和当前候选写入率 `g` 的收益。两者均为 opt-in 协议；原始 In-Place TTT baseline 和历史标签语义没有改变。

模型为 hku-gpu2 上的 Qwen3-4B In-Place TTT 1B-token 最终 checkpoint `/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt`，两张 RTX 5090，`cbf_ttt_train_py311` 环境。采集代码 commit `8d0fb98`。自然文本取自 `/home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl` 中首批满足长度要求的 48 条不同记录；这些记录来自**该模型使用过的预训练语料**。每条前 4096 token 为已见 chunk，后续 32 token 为查询前缀，再后续 128 token 为评分目标。目标 128-token 完整序列不出现在可见 context 中。此处“未见”仅指本次 session，不能当作预训练之外的独立测试。构造参数、源组划分和使用行的 SHA256 见 [`scenarios.jsonl.meta.json`](scenarios.jsonl.meta.json)。

自然试点共 12 源组，每组 `both_relevant / old_only / new_only / neither_relevant` 四条件，按源组分成 train/dev/test 8/2/2。每条在第二个完整 chunk 后枚举 `α,g∈{0,0.5,1}`，只在这个稀疏边界取标签，共 48 条。两个 worker 分别于 22:50:31 和 22:51:03 启动，于 22:51:14 和 22:51:46 结束；48/48 标签成功，无 OOM。单场景 smoke 为 1.403 秒、13.854 GiB 峰值 reserved。

## 机制拆分

从 `joint_v2` 的一个源组取四类场景。每类固定候选动作，再对两个 gap chunk 分别执行 `normal_11`（继续写入）或 `freeze_10`（保留已有快记忆、跳过后续写入）。四类场景平均 NLL：

| 候选动作 / 后续策略 | gap0 | gap1 | gap2 |
|---|---:|---:|---:|
| `11 / normal_11` | 0.669 | 2.238 | 8.848 |
| `11 / freeze_10` | 0.669 | 1.105 | 1.776 |
| `00 / normal_11` | 0.639 | 1.001 | 1.376 |
| `00 / freeze_10` | 0.639 | 0.841 | 0.623 |

`11/normal_11` 的平均快记忆范数比率从 gap0 的 0.0253 升至 gap2 的 0.0549；冻结路径保持 0.0253。四条诊断平均各 13.469 秒，最大峰值 reserved 20.313 GiB。逐动作完整聚合见 [`gap_diagnostic_summary.json`](../joint_v2_pilot/gap_diagnostic_summary.json)。这提示本组长间隔退化在后续写入累积时放大；`11/freeze_10` 仍高于 `00/freeze_10`，当前候选写入本身也有代价。四类样本来自**同一源组**，只用于定位机制，不能作泛化统计；冻结后续写入也不是已验证的最佳部署策略。

## 自然 continuation 结果与阶段门

| 指标 | 结果 |
|---|---:|
| 四角点最优次数 `00 / 01 / 10 / 11` | 33 / 7 / 7 / 1 |
| 四角点平均 NLL `00 / 01 / 10 / 11` | 2.367 / 2.408 / 2.386 / 2.464 |
| `both_relevant` 中 `11` 对 `00` 平均 NLL 改善 | −0.0123 |
| `new_only` 中 `11` 对 `00` 平均 NLL 改善 | −0.0172 |
| 两种有用新信息条件下改善 >0.005 的**独立源组** | 2/12（各 1 组） |
| 每条标签平均 / p95 耗时 | 1.079 / 1.428 秒 |
| 最大 CUDA allocated / reserved | 12.648 / 13.854 GiB |
| 内部网格点优于最佳角点的比例 | 12/48；平均优势 0.00078 NLL |

`both_relevant` 中 `00/01/10/11` 最优为 6/3/3/0；`new_only` 中为 7/1/3/1。平均 `new_continuation` 查询的 `11` NLL 为 1.996，`00` 为 1.979。完整聚合见 [`summary.json`](summary.json)；[`gate.json`](gate.json) 按执行前 `>0.005` NLL 且至少 `3/12` 独立源组的规则复算，结果为 **未通过**。两个相关条件在同一源组使用相同新 chunk 和新查询，按源组去重，不能当作 24 个独立例子。

**停止正式标签扩量、联合控制器训练和公开基准评测。** 当前自然 continuation 任务未产生足够稳定的正写入样本；直接训练控制器主要会学习关闭快记忆，不能检验双门控研究动机。这是本模型、语料和试点任务的结果，不代表推理时更新总是无益。已查看探索性 dev/test，后续正式独立测试须使用新文档及新源组。下一轮方法工作应先检查候选 `ΔW` 的任务相关性和收益形成条件，再预注册新阶段门并用未参与预训练的文档复核。

## 复现与数据位置

构造和两卡采集命令见 [`DUAL_GATE_MECHANISM_PILOT.md`](../../../../DUAL_GATE_MECHANISM_PILOT.md)。已有文件不可覆盖，重跑需换输出目录。聚合复算：

```bash
ROOT=/home/ctj/cbf_ttt_natural_pilot_20260927
V2=/home/ctj/cbf_ttt_joint_v2_pilot_20260927
python -m scripts.summarize_cbf_joint_pilot --labels "$ROOT"/{train,dev,test}_shard{0,1}_joint_labels.jsonl --output "$ROOT/summary.json"
python -m scripts.evaluate_cbf_natural_gate --labels "$ROOT"/{train,dev,test}_shard{0,1}_joint_labels.jsonl --output "$ROOT/gate.json"
python -m scripts.summarize_cbf_gap_diagnostic --input "$V2/gap_diagnostic_four.jsonl" --output "$V2/gap_diagnostic_summary.json"
```

原始自然场景、逐样本标签和 gap 轨迹保存在 hku-gpu2 对应输出目录；逐样本标签与诊断轨迹也已下载至本地同名实验目录，由 `.gitignore` 排除。GitHub 只保存代码、构造元数据、聚合 JSON 和本报告。此前自动审批拒绝上传含派生预训练内容的完整标签，未绕过该限制。
