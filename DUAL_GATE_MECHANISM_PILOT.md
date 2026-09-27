# 双门控后续写入诊断与自然 continuation 试点

2026-09-27。`joint_v2` 在两个 gap chunk 后出现 `11` NLL 大幅增加，且 48/48 聚合标签选择 `00`；详见 [`joint_v2` 报告](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/joint_v2_pilot/REPORT.md)。本轮先定位累计写入效应，再用更贴近 TTT 预训练目标的自然文本 NLL 检查当前 `ΔW` 的收益。两项都是小规模机制诊断，不训练控制器或运行公开基准。

## A. 逐 gap 机制拆分

从同一源组的四类 `joint_v2` 场景取候选边界，在完全相同的初始 KV、旧快记忆和当前 `ΔW` 上枚举四角点 `00/01/10/11`。每个动作分别沿两条后续路径运行两个 4096-token gap chunk：

- `normal_11`：每个 gap 按原始 TTT 累积写入。
- `freeze_10`：保留当前快记忆，但跳过每个 gap 的新候选写入。

在候选边界后、gap1 后、gap2 后用相同 held-out 查询分别评分，并记录快记忆相对基础权重的范数。两条路径在 gap0 必须相同；`normal_11` 在 gap2 的结果应与原 `joint_v2` 标签一致。这样可以观察长间隔损失增幅是当前写入本身、后续累计写入，还是两者的交互。它不假设冻结后续写入是最终推理策略。

## B. 自然文本 continuation

从既有 1B-token FineWeb-Edu/LongCrawl64 混合 JSONL 读取每组四条**不同**的 6143-token 记录。每条取前 4096 token 作一个已见 chunk，随后 32 token 作未见查询前缀，再后续 128 token 作逐 token NLL 目标。每组四个条件：

| 条件 | 旧 chunk | 候选 chunk | 未见目标 |
|---|---|---|---|
| `both_relevant` | A 前缀 | B 前缀 | A 和 B 的 continuation |
| `old_only` | A 前缀 | C 前缀 | A 的 continuation |
| `new_only` | A 前缀 | B 前缀 | B 的 continuation |
| `neither_relevant` | A 前缀 | C 前缀 | 独立 D 的 continuation |

所有 A/B/C/D 记录在源组之间互不重复；目标的 128-token 连续片段不出现在 context 中。当前小试点为 12 组、每组四条件、3×3 网格、只在第二边界采标签；两卡各处理源组分片。先单场景 smoke 并检查显存/耗时，再运行全部 48 场景。`joint_natural_v1` 协议与合成 `joint_v1/v2` 隔离。这里的自然记录来自模型已经使用的预训练语料，因而只能诊断训练分布内的即时适应行为，不能当未见语料泛化分数；试点 dev/test 也不能再作为正式 test。

若自然试点依旧几乎全选 `00`，或 `11` 在 `new_only` 与 `both_relevant` 中没有至少 3/12 个源组超过 `0.005` NLL 的优势，就暂停控制器标签。若出现多样可辨的门控收益，再用全新未见文档和源组复核，并重新估算正式规模。这里的阈值是工程筛选，不是显著性判定。

## 运行命令

在 hku-gpu2 的 `/home/ctj/cbf_ttt_joint_exp_20260927` 和 `cbf_ttt_train_py311` 环境：

```bash
MODEL=/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt
SOURCE=/home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl
V2=/home/ctj/cbf_ttt_joint_v2_pilot_20260927
ROOT=/home/ctj/cbf_ttt_natural_pilot_20260927
python -m scripts.diagnose_cbf_gap --model "$MODEL" --data "$V2/shards/train_shard0.jsonl" --split train --limit 4 --output "$V2/gap_diagnostic.jsonl"
python -m tasks.build_cbf_natural_scenarios --tokenizer /home/ctj/models/Qwen3-4B --data "$SOURCE" --output "$ROOT/scenarios.jsonl" --train-groups 8 --dev-groups 2 --test-groups 2 --context-tokens 4096 --query-tokens 32 --answer-tokens 128 --seed 118
python -m scripts.shard_cbf_scenarios --input "$ROOT/scenarios.jsonl" --output-dir "$ROOT/shards" --shards 2
bash scripts/run_cbf_joint_pilot.sh "$MODEL" "$ROOT" 0 0
bash scripts/run_cbf_joint_pilot.sh "$MODEL" "$ROOT" 1 1
python -m scripts.summarize_cbf_joint_pilot --labels "$ROOT"/{train,dev,test}_shard{0,1}_joint_labels.jsonl --output "$ROOT/summary.json"
```

两条 `run` 命令在不同 GPU 的 tmux 会话中同时执行。正式模型测试、训练和新数据发布只有在机制诊断表明标签有用后才继续；原始逐样本数据保留本机与远程，GitHub 仅记录代码和聚合结果。
