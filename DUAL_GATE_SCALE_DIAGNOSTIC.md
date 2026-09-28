# 候选更新方向与尺度诊断

2026-09-28，在查看本轮尺度诊断结果前固定。发布后论文随机试点中，`new_only` 的写入收益在 `g=1` 时平均为负，写入阶段门未通过。本轮只诊断候选 `ΔW` 的方向与尺度，不训练控制器、不修改原有 baseline 公式或放宽既定阶段门。

使用随机主试点的 **8 个 train 源组**，每组 `new_only` 与 `old_only` 两类场景，共 16 条；dev/test 不参与本轮诊断。两块 4096-token context 的 KV、第一块形成的旧快记忆和第二块候选 `ΔW` 固定不变。对每条场景分别计算 `M'=M+sΔW`，`s∈{-2,-1,-0.5,0,0.25,0.5,1,2}`，使用相同的未见 32+128-token 续写评分。负数和大于 1 的 `s` **仅用于局部方向探针**，并非可部署门控，也不进入控制器标签。诊断还记录第二块 surprise、候选更新相对基础权重的范数比率、各尺度下快记忆范数、逐条耗时与峰值显存。

`s=0/1` 必须逐值重现现有 3×3 标签中的 `10/11`；否则停止解释新轨迹并排查实现。汇总分别报告两种条件各尺度的平均 NLL、逐源组最佳尺度及正尺度相对 `s=0` 的收益。若最优集中在 `0<s<1` 而 `s=1` 受损，提示尺度问题；若负尺度普遍优于非负尺度，提示本任务上的候选方向不合适，不能直接推断原始 TTT 更新公式符号错误；若各尺度近似同分，提示该续写指标缺少分辨率。这些解释均是机制假设，不替代后续独立任务复核。

运行入口：

```bash
ROOT=/home/ctj/cbf_ttt_postcutoff_random_pilot_20260927
MODEL=/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt
python -m scripts.diagnose_cbf_update_scale --model "$MODEL" --data "$ROOT/scenarios.jsonl" --split train --regimes new_only,old_only --scales=-2,-1,-0.5,0,0.25,0.5,1,2 --reference-labels "$ROOT/train_shard0_joint_labels.jsonl" "$ROOT/train_shard1_joint_labels.jsonl" --output "$ROOT/update_scale_train.jsonl"
python -m scripts.summarize_cbf_update_scale --input "$ROOT/update_scale_train.jsonl" --output "$ROOT/update_scale_summary.json"
```

原始逐场景轨迹只留在远程和本地忽略目录；GitHub 仅记录聚合、脚本与报告。

## 执行结果

16 条 bf16 轨迹均通过 `s=0/1` 参考标签校验；同一 16 条 float32 曲线也完成。两种精度下，`new_only` 的平均 NLL 在 `s=0` 附近最低，`s=1/2` 明显更差；float32 的负尺度改善超过 0.005 NLL 为 0/8 组。减小到 `s=0.25` 未带来平均正收益。完整曲线、资源及解释见 [`发布后论文报告`](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/joint_postcutoff_random_pilot/REPORT.md)。
