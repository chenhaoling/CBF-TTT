# 事件语料writer 100步训练状态

采集代码3a0457e，hku-gpu2，tmux cbf-event-writer-100step，ROOT=/home/ctj/cbf_ttt_event_writer_100step_v1。2026-10-10 11:01:41 +08:00启动。

2026-10-10 11:03:58 +08:00全部100步及四点评估完成，wall137秒。服务器9项测试全部通过（1.649秒），独立完成审计通过。单步0.299468秒，训练峰值allocated10.186427GiB。272次dev策略查询、16次最终train probe齐全，test未评分。

终态 completed_writer_warmup，passed_memory_gate=false。100步正确/空/错配NLL为2.053219/4.277348/2.235196，严格EM均0/16；两个dev世界中的一个错配优于正确，anchor正确对twin没有优势。没有自动延长训练或启动控制器。见[完整报告](REPORT.md)。

方案：[EVENT_WRITER_TRAINING_PLAN.md](../../../EVENT_WRITER_TRAINING_PLAN.md)。

SSH随后恢复，服务器已快进同步52c5390报告提交，已补跑已有生成的只读统计：100步正确记忆首个编号正确0/16、裸code_为7/16；完整KV首行正确12/16但严格EM仍0。未启动新训练/模型评分。检查时两张5090利用率均0%。
