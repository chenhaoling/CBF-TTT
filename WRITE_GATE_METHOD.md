# CBF-TTT 两种独立的测试时记忆控制

用户在 2026-09-27 澄清：噪声或重复样本产生的候选更新 `ΔW_t` 可能无用，因此应决定**当前更新是否写入**。这与附件 `CBF_TTT_method.md` 已写明的**是否遗忘历史记忆**是两个不同问题。附件保留为最初的遗忘版本；本文记录修正后的研究范围。

## 更新式与语义

对每个 TTT 层，固定预训练权重 `W0`，令 `M_t` 为会话快记忆，当前完整 chunk 产生沿用 In-Place TTT 规则的候选 `ΔW_t`：

```text
M_t = α_t M_(t-1) + g_t ΔW_t
W_eff,t = W0 + M_t
```

- `α_t∈[0,1]`：历史**保留率**。`0` 清除旧会话增量，`1` 完整保留旧记忆。
- `g_t∈[0,1]`：当前写入强度。`0` 跳过当前候选，`1` 完整写入。对噪声或重复样本，可降低 `g_t`，而不必清除旧记忆。
- 原始 In-Place TTT 同一边界协议对应 `(α_t,g_t)=(1,1)`。

| `α_t` 保留旧记忆 | `g_t` 写入当前候选 | 快记忆状态 | 可对应的推理情景 |
|---:|---:|---|---|
| 0 | 0 | `M_t=0` | 旧会话已失效，当前 chunk 又是噪声；重置快记忆 |
| 0 | 1 | `M_t=ΔW_t` | 旧会话失效或发生可信纠错，只保留当前候选 |
| 1 | 0 | `M_t=M_(t-1)` | 旧记忆仍有用，当前 chunk 重复、无关或不可信；跳过写入 |
| 1 | 1 | `M_t=M_(t-1)+ΔW_t` | 旧记忆仍有用，当前 chunk 有可兼容的新信息；继续累积 |

这里的“旧知识”特指会话内 TTT 快记忆 `M`；预训练基础权重 `W0` 与 attention KV 不由此式清除。`(0,1)` 是快记忆层面的整体替换，不能保证精确覆盖某一条语义事实。

当前代码提供两个**独立的一维实验模式**：`--update-rule forget` 固定 `g=1`，其历史遗留输出 `f_t` 表示遗忘率，联合式中的保留率为 `α=1-f_t`；`--update-rule write` 固定 `α=1` 并学习 `g`。运行时 `CBFSession.commit_both(alpha, write_gate)` 已按本文件的保留率约定实现双控制式，便于未来设计联合实验。尚未训练二维控制器或构造 3×3 联合反事实标签；不能用一维标签推断联合最优。

## 反事实监督

对同一决策边界保存前向/KV/快记忆与当前候选，遍历 `0,0.5,1`，只改变当前边界相应控制量，其余条件、后续输入和参考策略保持一致。旧 `forget` 数据的参考是遗忘率 `f=0`，等价于保留率 `α=1`；`write` 的参考是 `g=1`（总是写入）。以未来问答平均 NLL 选择经验最优值，旧遗忘标签平局按较小遗忘率选择，新写入标签平局按较小写入率选择，保存完整损失表、收益、逐条耗时和显存。

`write` 场景构造器在第二个 chunk 放入三类候选：新的权威事实、重复的已知事实、与任务无关的随机标记；第一块提供先验事实，后续查询检验旧事实或新事实。使用自然背景时，背景从训练语料抽取。此模板仅是受控诊断：KV 仍可直接携带候选文本，候选即使未写入 `ΔW`，未来问答也可能不变；应报告平坦标签比例，并在更长时距、KV 不直接可见或真实噪声条件下复核。

## 运行入口与兼容性

```bash
# 构造写入决策场景；按源组分 train/dev/test。
python -m tasks.build_cbf_scenarios --tokenizer /home/ctj/models/Qwen3-4B \
  --output /home/ctj/cbf_write_pilot/scenarios.jsonl \
  --background-data /home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl \
  --objective write --future-mode short_tail --chunk-size 4096 \
  --context-chunks 2 --future-chunks 1 --futures-per-scenario 1 \
  --train-groups 10 --dev-groups 3 --test-groups 3 --variants-per-group 3 --seed 42
python -m scripts.shard_cbf_scenarios --input /home/ctj/cbf_write_pilot/scenarios.jsonl \
  --output-dir /home/ctj/cbf_write_pilot/shards --shards 2
MODEL=/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt
bash scripts/run_cbf_formal_labels.sh "$MODEL" /home/ctj/cbf_write_pilot 0 0 write
bash scripts/run_cbf_formal_labels.sh "$MODEL" /home/ctj/cbf_write_pilot 1 1 write
python -m scripts.summarize_cbf_formal_labels --directory /home/ctj/cbf_write_pilot --shards 2 --boundary 2
```

`tasks.cbf_ttt train` 从标签的 `update_rule` 识别控制语义，拒绝遗忘和写入标签混合；checkpoint 保存该语义，加载后运行时拒绝不匹配模式。旧标签与旧 checkpoint 没有该字段时按 `forget` 解释。原有 `forget` CLI 默认值和历史结果保持可复现；以前 3600 条遗忘标签与控制器**不能**证明写入门控的效果，也不能改名后直接复用。正式写入实验需先做小规模标签耗时/峰值显存和标签分布检查，再定规模。
