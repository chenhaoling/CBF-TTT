# 可读性校准运行记录

状态：完成并通过结果/来源审计；仅开发集可读性校准，不是记忆价值验证通过。

- hku-gpu2，双RTX5090，Qwen3-4B / 1B checkpoint。
- 采集提交 `c07d98cfeccb7bb639ca32c7797888bced8a9e0e`。
- ROOT：`/home/ctj/cbf_ttt_readability_20261009`。
- 2026-10-09 14:45:09至14:58:49 +08:00，13分40秒；tmux `cbf-readability-20261009` 已正常退出。
- 8项测试通过，128/128上下文、384/384查询；32个桥接评分最大误差0。
- 来源仅原dev0–7；confirm未评分。骨干和父缓存未变。
- 预定远距门槛选择cloze，但其近距双键目标准确率0%；不据此启动完整V/F。
- 完整结果、限定解释与下一步见 [REPORT.md](REPORT.md)，审计见 [execution_audit.json](execution_audit.json)。
