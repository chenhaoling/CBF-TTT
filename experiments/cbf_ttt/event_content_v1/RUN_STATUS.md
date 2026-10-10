# 少量世界内容训练状态

方案见[EVENT_CONTENT_TRAINING_PLAN](../../../EVENT_CONTENT_TRAINING_PLAN.md)。两臂CE/content_pair各400步，hku-gpu2双5090，ROOT=/home/ctj/cbf_ttt_event_content_v1，tmux=cbf-event-content-v1。2026-10-10 14:20:09 +08:00已启动，代码3901eca。

2026-10-10 14:28:25 +08:00训练/三点评估/汇总完成，wall496秒；随后独立审计通过。每臂400步，合计800步/816策略查询。test未评分，原始骨干保持不变。

本地13项测试8通过、5因无torch跳过（1.037秒）；服务器全部13项通过（1.895秒）。CE/content_pair训练集完整编号分别4/16、7/16，开发集均0/16；两臂train/dev门槛均未通过。成对方案train与dev的anchor-twin数字NLL优势均为负，没有证实可靠单事实绑定。未追加训练/启动控制器。详见[正式报告](REPORT.md)与summary/design/execution_audit。

终态completed_content_training，实验tmux已退出；检查时两卡利用率0%、显存139/18MiB。所有本轮训练和审计已完成，无本轮后台训练任务。
