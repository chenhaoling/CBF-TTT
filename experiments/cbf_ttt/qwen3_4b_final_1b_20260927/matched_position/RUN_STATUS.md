# 同键值位置对照

状态：已完成，独立输入重建与汇总审计通过。

- hku-gpu2双RTX5090，Qwen3-4B最终1B checkpoint；采集提交 `6a06b5dd441aa16e86e0a3c3c658135ef80e34bc`。
- ROOT `/home/ctj/cbf_ttt_matched_position_20261009`；tmux `cbf-matched-position-20261009` 正常退出。
- 2026-10-09 16:13:46–16:28:54 +08:00，908秒。
- 112/112轨迹、224/224查询；96桥接评分NLL误差0、预测和输入哈希一致。
- 15项启动测试通过；父缓存/骨干权重未变，无OOM或失败。
- 不满足“一致的末段目标干扰增强”规则；两donor平均p6−p1目标0个百分点、p6−p3为+12.5个百分点。
- 补充颜色重合诊断，主设计未改。旧confirm未评分，M=0，无新训练/V/F。
- [完成报告](REPORT.md)、[主聚合结果](summary.json)、[执行审计与辅助诊断](execution_audit.json)。
