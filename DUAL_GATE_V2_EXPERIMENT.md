# 双门控修订试点：执行前协议

2026-09-27。上一轮 `joint_v1` 的 48 条短尾精确代码问答中，`00` 赢 37 条；该结果触发 [`DUAL_GATE_EXPERIMENT_PLAN.md`](DUAL_GATE_EXPERIMENT_PLAN.md) 的暂停条件。本轮是独立的 `joint_v2` 探索试点，不能与旧标签混合训练或把看过的 pilot test 组当正式 test。

执行状态：48 条 `joint_v2` 标签已完成，预设阶段门未通过；结果见 [`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/joint_v2_pilot/REPORT.md`](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/joint_v2_pilot/REPORT.md)。以下规则保留执行前写定的原文，不按结果修改阈值。

## 场景与假设

每个源组仍有四类：旧规则相关×新独立规则、旧规则相关×重复/矛盾噪声、旧规则已过时×可信纠正、旧规则已过时×无可信替代。旧规则在第一块，新候选在第二块，两个 chunk 均为 4096 token。规则示例给出票号 `11/23/35` 到 `SKU-(n+offset)` 的映射；查询票号 `47/53`，**未展示这些输入输出对**，构造器检查目标答案 token 序列没有出现在 context 或 gap 中。噪声和重复候选与有用候选位于同一票号主题、同一完整 chunk 长度，但其语义核心不完全等长；此限制必须随结果报告。

每场景在同一决策边界生成两个 future：`gap=0` 和 `gap=2` 个 4096-token 自然背景 chunk。gap 不包含目标规则或答案，后续 gap 按原始 `11` 策略更新，attention KV 保持真实推理状态、不人为清空。分别保存两种间隔和每类查询的 9 动作损失，聚合损失为所有查询的等权平均。旧相关×新有用同时考察旧规则和新规则的 held-out 输出；旧相关×噪声考察旧规则；纠正考察新规则；双方无效用独立算术题检测干扰。算术题不用于“目标答案未出现在全文”断言，只用于无相关规则的诊断。

## 预定规模和阶段门

沿用 12 个源组、每组 4 场景、`α,g∈{0,0.5,1}`、只采第二块边界、两张 5090 的试点。train/dev/test 为 8/2/2 组，但这三者在试点中都仅供协议诊断；未来正式 test 必须重新留出。每条仍记录秒数和 CUDA allocated/reserved 峰值。先单场景 smoke，再两卡并行。

以源组为独立单位，预设有意义的 NLL 差为 `0.005`：

1. 在 `gap=2` 的新有用/纠正场景中，写入角点相对拒写角点的优势至少出现在各 3/12 个源组；在重复/噪声场景中，拒写优势至少出现在 3/12 个源组。
2. 在 `gap=2` 的旧规则相关场景中，保留旧记忆相对清除旧记忆的优势至少出现在 3/12 个源组。比较时对另一个门取四角点中的最优值，不预设相互独立。
3. 至少两个不同角点各在 3 个以上源组获得有意义的优势；如果几乎所有场景仍由 `00` 获胜，就暂停训练控制器。若超过 20% 场景的内部网格点比最佳角点低 `0.005` 以上 NLL，转向连续系数模型。
4. 标签无非有限数、无 OOM；至少四类代表场景重复的 9 动作损失最大差低于 `0.001`。显存余量不足时先缩短 gap，而非隐式截断。

这些阈值是继续扩大之前的工程筛选条件，不是统计显著性或论文结论。若未通过，保存负面结果和场景限制，继续诊断 TTT 更新对 held-out 任务的作用；不训练一个学习退化动作的控制器。

## 运行命令

在 hku-gpu2 的 `/home/ctj/cbf_ttt_joint_exp_20260927`，使用现有 `cbf_ttt_train_py311` 环境：

```bash
ROOT=/home/ctj/cbf_ttt_joint_v2_pilot_20260927
MODEL=/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt
mkdir -p "$ROOT"
python -m tasks.build_cbf_scenarios --tokenizer /home/ctj/models/Qwen3-4B --output "$ROOT/scenarios.jsonl" --background-data /home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl --objective joint_v2 --future-mode short_tail --chunk-size 4096 --context-chunks 2 --future-chunks 1 --futures-per-scenario 2 --joint-gap-chunks 2 --train-groups 8 --dev-groups 2 --test-groups 2 --variants-per-group 4 --seed 84
python -m scripts.shard_cbf_scenarios --input "$ROOT/scenarios.jsonl" --output-dir "$ROOT/shards" --shards 2
bash scripts/run_cbf_joint_pilot.sh "$MODEL" "$ROOT" 0 0
bash scripts/run_cbf_joint_pilot.sh "$MODEL" "$ROOT" 1 1
python -m scripts.summarize_cbf_joint_pilot --labels "$ROOT"/{train,dev,test}_shard{0,1}_joint_labels.jsonl --output "$ROOT/summary.json"
```

两条 `run` 命令需在独立 tmux 会话同时运行。脚本拒绝覆盖已有标签。阶段门通过后再另行估算正式规模、实现联合控制器及独立评测；本轮不会自动执行这些步骤。
