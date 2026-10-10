# 事件语料writer 100步训练状态

采集代码3a0457e，hku-gpu2，tmux cbf-event-writer-100step，ROOT=/home/ctj/cbf_ttt_event_writer_100step_v1。2026-10-10 11:01:41 +08:00启动。

服务器9项测试全部通过（1.649秒），原始Qwen3-4B骨干+新FP32 writer加载成功。100步optimizer更新已完成，单步约0.30秒，训练峰值allocated约10.2GiB。第二张GPU仍在评估固定0/25/50/100 checkpoint；完整结果/独立审计待完成，尚不能据训练loss判断记忆有效。

方案：[EVENT_WRITER_TRAINING_PLAN.md](../../../EVENT_WRITER_TRAINING_PLAN.md)。
