# 固定12K checkpoint/native诊断运行状态

- 服务器 hku-gpu2，两张 RTX5090；tmux `cbf-checkpoint-native-20261010`。
- ROOT `/home/ctj/cbf_ttt_checkpoint_native_20261010`。
- 源码提交 `7077d5b`，启动 2026-10-10 09:59:10 +08:00。
- 11 项服务器测试通过（8.399秒）；本地无torch，张量测试以服务器结果为准。
- 预登记：原始plain + 10000/40000/81381各plain/native；附加最终zero_lr控制。64个12K场景；总464 rollout/1856 query，binding主指标。
- 当前正在中间DCP导出/逐张量核验，完成后自动进行两卡收集；本轮准确率结果尚未核验，不能宣称实验完成。
- 方案：[CHECKPOINT_NATIVE_PLAN.md](../../../../CHECKPOINT_NATIVE_PLAN.md)。完成后通过独立审计再归档报告。
