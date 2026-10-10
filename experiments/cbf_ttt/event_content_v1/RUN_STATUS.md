# 少量世界内容训练状态

方案见[EVENT_CONTENT_TRAINING_PLAN](../../../EVENT_CONTENT_TRAINING_PLAN.md)。两臂CE/content_pair各400步，hku-gpu2双5090，ROOT=/home/ctj/cbf_ttt_event_content_v1，计划tmux=cbf-event-content-v1。尚未启动。

本地13项测试8通过、5因无torch跳过（1.037秒）；服务器将补齐全部13项，失败不训练。独立脚本语法与shell语法检查通过。test保持未评分，无方法效果结论。
