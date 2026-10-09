# 无颜色重合四场景读出校准

状态：实验完成，独立审计通过；读取门槛失败（`stopped_by_disjoint_Q`）。

- hku-gpu2双RTX5090，最终1B Qwen3-4B；采集提交 `45525be`。
- UTC+8：2026-10-09 18:48:08–18:59:10，662秒。
- 80轨迹/288查询齐全；32桥接评分误差0，权重/父缓存不变。
- 原QA失败14/16格，binding失败12/16格；无格式被选中。
- M=0，旧confirm未评分；未进入新来源、M写入、V/F或控制器训练。
- ROOT `/home/ctj/cbf_ttt_disjoint_readout_20261009`；tmux已正常结束。
- 完整结果、限制、复现方式和后续建议见 [REPORT.md](REPORT.md)。
