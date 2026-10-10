# 仅训练遗忘策略：严格复现与快记忆内容验证

## 结论

第一阶段总体未通过。六种策略的快记忆内容门槛全部失败，因此不能把旧实验中完整上下文下的差值解释为“自适应 session 遗忘已经有效”。

三个学习控制器在 memory-only 条件下均比固定完全清除略好，但它们的正确快记忆仍比空快记忆差，而且 Correct 与 Wrong 的差异没有通过来源组 bootstrap 和场景比例门槛。完整上下文条件下，三个控制器也都弱于固定完全清除。按照执行前登记的顺序，快记忆内容门槛优先；后续控制器差值只能作为诊断，不能覆盖该失败。

## 协议

- 模型：Qwen3-4B / 1B-token In-Place TTT 最终 checkpoint，冻结 backbone，固定 `g=1`。
- 数据：旧控制器 test 的 300 场景、100 个来源组；`correction/stable/topic_shift` 各 100。
- 策略：固定遗忘率 0/0.5/1，以及固定 500 个训练来源组的 controller seed 42/43/44。
- 查询：每场景一个 historical 和一个 current-or-new 查询。
- `full_context` 保留 KV 与快记忆；`correct/wrong/empty_memory` 都从空 KV 开始。wrong memory 来自同 regime、不同来源组的固定互换 donor。
- 内容门槛只用 historical 查询，因为 current-or-new 事实可能只位于不足一个 TTT chunk 的 future tail 中。

详细预注册判据见 [`FORGETTING_ONLY_VALIDATION_PLAN.md`](../../../../FORGETTING_ONLY_VALIDATION_PLAN.md)。

## 快记忆内容结果

下表为 historical 查询。收益定义为对照 NLL 减去 Correct-memory NLL，正数表示正确记忆更好。

| 策略 | Correct NLL | Correct vs Wrong 收益 | 95% CI | 正场景比例 | Correct vs Empty 收益 | 95% CI | 内容门槛 |
|---|---:|---:|---:|---:|---:|---:|---|
| fixed 0 | 4.652129 | +0.002739 | [-0.006580, 0.012173] | 50.33% | +0.004402 | [-0.019148, 0.027781] | 失败 |
| fixed 0.5 | 4.661958 | +0.001287 | [-0.005536, 0.008437] | 49.33% | -0.005427 | [-0.017806, 0.006746] | 失败 |
| fixed 1 | 4.678300 | +0.001980 | [-0.004897, 0.008775] | 51.33% | -0.021769 | [-0.027931, -0.015175] | 失败 |
| controller seed 42 | 4.668013 | +0.003596 | [-0.004086, 0.011266] | 52.33% | -0.011483 | [-0.020430, -0.002121] | 失败 |
| controller seed 43 | 4.667907 | +0.003802 | [-0.003964, 0.011615] | 52.00% | -0.011376 | [-0.020158, -0.002151] | 失败 |
| controller seed 44 | 4.666194 | +0.004311 | [-0.003282, 0.012265] | 51.67% | -0.009664 | [-0.019779, 0.000607] | 失败 |

所有 Correct-vs-Wrong 区间都跨 0，正场景比例接近随机的 50%。除 fixed 0 的极小正均值外，正确快记忆对空记忆没有平均收益；fixed 0 同样没有达到 0.005 阈值，且区间跨 0。

## 控制器与固定完全清除

旧 test 中最强固定策略是 fixed 1。三个控制器在完整上下文下均更差：

| 控制器 | Full-context NLL | 相对 fixed 1 收益 | 95% CI |
|---|---:|---:|---:|
| seed 42 | 0.351107 | -0.003888 | [-0.005028, -0.002852] |
| seed 43 | 0.351540 | -0.004321 | [-0.005659, -0.003124] |
| seed 44 | 0.353360 | -0.006142 | [-0.007819, -0.004553] |
| fixed 1 | **0.347218** | 0 | — |

在 Correct-memory historical 条件下，三个控制器相对 fixed 1 分别改善 0.010286、0.010393、0.012106，区间下界均大于 0。这个局部比较满足原 `adaptive_forgetting_gate` 的数值条件，但不能独立解释：对应的六种快记忆全部没有通过内容门槛，且控制器正确快记忆比空记忆更差。最终 `stage_gate.passed=false`。

## KV 与快记忆的区别

完整上下文 NLL 约为 0.35；清空 KV 后，即使保留正确快记忆，NLL 上升到约 4.65–4.70。这个差距说明旧 short-tail 任务主要依赖正常上下文/KV 路径。它不证明快记忆完全没有任何信息，但表明当前协议下无法从快记忆独立可靠地读取历史事实。

## 完整性与资源

- 11 项服务器测试通过。
- 六个策略各覆盖 300 场景、100 来源组；全部损失有限，快记忆范数非零，wrong donor 跨组且互为配对。
- `fixed_0/fixed_0.5/fixed_1/controller_seed42` 的 full-context 结果逐场景精确复现旧封存 rollout，最大误差均为 0。
- 模型参数版本在每次采集前后不变；独立审计通过。
- 每策略采集 294–303 秒，约 0.98–1.01 秒/场景；最大 allocated 14.115 GiB、reserved 16.336 GiB/GPU。
- 任务 17:30:47 入队，20:25:55 满足空闲条件，20:41:39 完成；含等待总墙钟 3 小时 10 分 52 秒，GPU 可用后的流水线约 15 分 44 秒。

原始 JSONL 保存在服务器 `/home/ctj/cbf_ttt_forgetting_only_validation_v1`。Git 只归档聚合、审计、各策略资源摘要和报告。

## 对论文主线的影响与下一步

目前可以保留的旧结论只有：在保留完整上下文/KV 的 short-tail 协议中，遗忘旧快权重比持续累积更好；最佳方法仍是固定完全清除。尚不能声称学习控制器利用 session 快记忆内容做出了有效选择。

按预注册停止条件，下一步不是扩大旧标签或做 history-only 特征消融，而是先构造动态赋值、memory-only 可读的数据：同一查询/实体跨 session 随机赋值，事实必须跨完整 TTT chunk 写入，训练/验证按世界分组，并在训练前固定 Correct > Wrong/Empty 门槛。只有 writer 在新世界上通过该门槛，才重新比较固定与学习遗忘策略。

## 复现

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_forgetting_only_validation_REPEAT \
bash scripts/run_cbf_forgetting_only_validation.sh
```

脚本拒绝覆盖已有 ROOT，并在结果完成后执行独立审计。
