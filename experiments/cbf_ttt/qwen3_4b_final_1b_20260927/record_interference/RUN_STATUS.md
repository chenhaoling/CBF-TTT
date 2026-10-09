# 固定槽位记录干扰实验

状态：完成且独立审计通过。只读输入干预实验，未训练或写入M。

- 采集提交 `f163f176e9f392412b3d0fc14ba7b2cf3c38cdcf`，hku-gpu2双RTX5090、Qwen3-4B最终1B checkpoint。
- ROOT `/home/ctj/cbf_ttt_record_interference_20261009`；tmux `cbf-record-interference-20261009` 已正常退出。
- 2026-10-09 15:18:53至15:39:54 +08:00，1261秒。
- 160/160轨迹、320/320查询；64个桥接评分最大NLL误差0、预测一致。
- 11项启动测试通过；来源/每槽内容/槽外不变及summary重算审计通过，权重与父缓存未变。
- natural0目标/锚点100%/87.5%，通过本轮dev条件可读性门槛。旧confirm未评分，V/F未启动。
- [完成报告](REPORT.md)、[聚合结果](summary.json)、[执行审计](execution_audit.json)、[来源修复核验](source_guard_audit.json)。
