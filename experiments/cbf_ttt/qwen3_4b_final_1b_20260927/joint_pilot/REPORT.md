# 双门控反事实标签试点（2026-09-27）

## 目标与协议

检查旧会话快记忆保留率 `α` 与新候选更新写入率 `g` 是否能在实际 checkpoint 上形成可辨且多样的联合决策，并测量构造成本。更新为 `M'=αM+gΔW`；四角点按 `(α,g)` 命名。该试点只生成反事实监督标签，**未训练联合控制器**。

- 模型：Qwen3-4B In-Place TTT 1B-token 最终 checkpoint，远程路径 `/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt`；其 `config.json` SHA256 为 `9ffb7420c132c9ea7de55e221baac89594699f942bc14f2327fc5014e038e672`。执行代码为 Git commit `aded608`，分析脚本在此后补充源组统计。
- 硬件：hku-gpu2 两张 RTX 5090，每卡分别处理一个 source-group 分片；bf16 推理。代码目录 `/home/ctj/cbf_ttt_joint_exp_20260927`，输出目录 `/home/ctj/cbf_ttt_joint_pilot_20260927`。
- 数据：从既有 FineWeb-Edu + LongCrawl64 1B-token 混合文件读取 96 条不同背景记录；12 个合成源组，train/dev/test 为 8/2/2 组，每组四个因素场景，共 48 场景。仅背景取自预训练文本，事实与问答由模板生成；它不是公开基准测试。
- 每场景两个 4096-token context chunk；只在第二个边界标注。未来为短尾续接，所有分支看到相同 future 和查询，并对后续上下文采用固定 `11` 更新。网格 `α,g∈{0,0.5,1}`，每标签 9 个动作损失。`flat_tolerance=1e-4`；比较相同源组的四个变体时以源组为单位。

## 完成情况与结果

独立克隆仓库后，远程 `tests.test_cbf_ttt`、`tests.test_cbf_scenarios`、`tests.test_cbf_joint_pilot_summary` 共 20 项通过。第一次真实模型 smoke 揭示 `collect-joint` 对不存在的 `args.controller` 的访问错误，修复并推送后单场景成功。试点两个 tmux worker 于服务器时间 15:30:45 同时开始，15:31:37 均完成；48/48 条标签均为有限损失，未发生 OOM。

| 指标 | 结果 |
|---|---:|
| 标签 / 独立源组 | 48 / 12 |
| 四角点最佳次数 `00/01/10/11` | 37 / 2 / 7 / 2 |
| 四角点平坦比例，阈值 `1e-4` | 0/48 |
| 内部网格点超过最优角点，阈值 `1e-4` | 10/48（20.8%） |
| 内部点平均 / 最大收益（NLL） | 0.000889 / 0.013657 |
| 平均 `J00/J01/J10/J11` | 0.383572 / 0.419680 / 0.401905 / 0.448177 |
| 源组平均 `J11−J00` | 0.064605；12/12 组为正 |
| 平均绝对交互项 `|J11−J10−J01+J00|` | 0.023739 |
| 平均 / p95 每标签耗时 | 1.367 / 1.705 秒 |
| 峰值 CUDA allocated / reserved | 12.650 / 13.861 GiB |

四个场景类型的 `00` 最佳次数分别为：旧相关×新有效 9/12、旧相关×新噪声 8/12、旧冲突×新纠正 9/12、旧冲突×新噪声 11/12。四种类型中固定 `00` 的平均 NLL 均低于固定 `11`。从同一源组四种类型再次构造标签，4×9 个损失与首次运行逐值完全一致（最大绝对差 0）；这是确定性检查，不代表不同硬件或精度的数值误差为零。

完整 48 条标签与四情景重复标签保存在本机本目录以及远程输出目录；**原始标签不提交到 GitHub**。GitHub 仅保存本报告、构造元数据和聚合汇总。原始远程日志在 `/home/ctj/cbf_ttt_joint_pilot_20260927/gpu0.log` 与 `gpu1.log`。本试点的 dev/test 源组已用于探索性分析，后续正式评估必须重新留出未看过的源组。

## 阶段门决定

**不进入正式 4800 场景标签、控制器训练和公开基准评测。** 预设停止条件是某些角点几乎没有可辨优势，或网格内部点经常超过角点。本试点的 `01`、`11` 各只赢 2/48，`00` 在有用新信息场景也赢 9/12；另有 10/48 的内部优势。这组模板主要区分“关闭全部快记忆”与其他动作，不能用来学习所需的双门控决策。资源本身足够：显存峰值低于单卡容量，平均逐标签时间约 1.37 秒；现阶段瓶颈是任务识别性而非 GPU。

可能机理：两块事实都仍在 attention KV 中，直接问答能够从 KV 读取信息，因而 `ΔW` 写入对精确代码回忆未必有益；当前候选的快权重更新可能对短尾 QA 产生干扰。`old_conflict_new_noise` 的 future 还直接写出更新后的正确代码，进一步削弱了当前候选门控的辨别价值。这些是由结果与场景模板得出的解释假设，尚未通过消融证实。不能把 `00` 的胜率解释成双门控方法有效，更不能拿旧一维遗忘控制器结果充当联合实验。

## 下一轮试点应修订的内容

1. 用模型真正可能通过 TTT 学到的局部规律、风格或映射构造候选，再在**未直接出现答案**的 held-out continuation 上评分；噪声和重复候选应与有效候选匹配长度与主题。
2. 延长写入与评分间隔，分别报告 KV 可见与较长间隔；保持实际推理的 KV 管理，不人为清空 KV 冒充线上效果。
3. 为旧记忆价值和新更新价值分别准备独立的诊断查询；删除 future 直接泄露答案的模板。保留同状态 3×3 试点和两卡资源记录，先检验四角点是否都有非偶然优势。
4. 若修订后内部系数仍经常获益，控制器应预测连续 `(α,g)` 或四角点损失加插值，而不能强行只分类四动作。正式规模和 test 组数须在新试点后重新计算。

## 复现命令

在远程仓库 `/home/ctj/cbf_ttt_joint_exp_20260927`，使用 `/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python`，按 [`MODIFICATION_LOG.md`](../../../../MODIFICATION_LOG.md) 中的场景生成与分片命令准备数据，然后在两个 tmux 会话中分别运行：

```bash
MODEL=/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt
ROOT=/home/ctj/cbf_ttt_joint_pilot_20260927
bash scripts/run_cbf_joint_pilot.sh "$MODEL" "$ROOT" 0 0
bash scripts/run_cbf_joint_pilot.sh "$MODEL" "$ROOT" 1 1
python -m scripts.summarize_cbf_joint_pilot --labels "$ROOT"/{train,dev,test}_shard{0,1}_joint_labels.jsonl --output "$ROOT/summary.json"
```

运行脚本会拒绝覆盖已有标签；复现时请改用新的输出目录。baseline 原始入口和配置没有改动。
