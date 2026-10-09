# 多决策遗忘实验：运行登记

状态：**已完成**。2026-10-08 18:33 +08:00 远程核实：完成时间 16:15:42，双分片各 8 场景，合计 16/16。完整结果见 [REPORT.md](REPORT.md)。

- 启动时间：2026-10-08 15:36:48 +08:00。
- 采集代码：`2a57ac1`。
- 机器：hku-gpu2，两张 RTX 5090；tmux：`cbf-multidecision-20261008`。
- 远程目录：`/home/ctj/cbf_ttt_multidecision_20261008`。
- 19 项测试通过；来源审计通过，固定 16 场景、4 来源组。
- smoke 复现三个全局单点动作和全程清除，采集函数要求每个旧评分误差≤1e-5。
- 每条轨迹平均 smoke 耗时（含桥接和全程对照）3.880 秒，最大 allocated 22.182 GiB。
- 按全部 1224 条新 GPU 轨迹估计，两卡耗时加 25% 余量约 49.5 分钟。实际结束时间以完成标记为准。

## 研究问题

1. 同一日程网格上，穷举全局多点遗忘是否优于单点？
2. 局部采样动作序列是否优于全局，候选数量相同时是否仍有收益？
3. 是否超过每步清除、不写入等强对照，而不只是避免持续累积退化？

局部完整空间未穷举，当前是预先固定采样集合；结果不能被描述为完整局部 oracle。来源均已观察过，历史 pilot/confirm 只作分组展示。端到端联合训练（图片第三条）未执行。

## 输出和检查

候选设计见 [design.json](design.json)，资源门槛见 [estimate.json](estimate.json)，完整方案见 [MULTIDECISION_FORGETTING_PLAN.md](../../../../MULTIDECISION_FORGETTING_PLAN.md)。

```bash
ssh hku-gpu2
# 查看任务（先确认 session 仍存在）
tmux ls
tail -n 5 /home/ctj/cbf_ttt_multidecision_20261008/collect_0.log
tail -n 5 /home/ctj/cbf_ttt_multidecision_20261008/collect_1.log
# 以下两文件只应在完整采集/审计后使用
cat /home/ctj/cbf_ttt_multidecision_20261008/completed_at.txt
cat /home/ctj/cbf_ttt_multidecision_20261008/summary.json
```

脚本在任一采集失败时停止，不生成完成标记。最终需核对 16 场景齐全、来源/model 一致、单点/清除复现、全局嵌套最优值单调性、骨干不变，然后归档最终报告。本轮已通过上述检查；多点相对单点有改善，选择性遗忘超越强对照的门槛未通过。
