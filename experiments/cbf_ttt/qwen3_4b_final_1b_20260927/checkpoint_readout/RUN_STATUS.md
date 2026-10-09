# 原始与最终checkpoint读取对照

状态：全部采集完成，独立审计通过；两模型均未通过统一读取门槛。

- 采集提交 `87866f698654ca6b50b1f8f6414630e51a715d1c`。
- 2026-10-09 23:27:34–23:52:12 +08:00，1478秒（24分38秒）。
- hku-gpu2双RTX5090；176上下文/608查询，全部齐全。
- 原始模型32查询原生/仓库路径与实际主干权重一致；最终模型288查询复现上轮CBF M=0，全候选NLL误差均0。
- binding目标均值原始73.4375%、最终48.4375%；锚点原始100%、最终59.375%。6/8来源组平均下降，2/8持平。
- 原始binding失败3/16格，最终12/16格；原QA分别失败9/16与14/16。
- 完整KV、M=0，无训练/新来源/confirm/V/F，原始权重未修改。
- ROOT `/home/ctj/cbf_ttt_checkpoint_readout_20261009`；tmux `cbf-checkpoint-readout-20261009` 已退出，终态 `completed_checkpoint_diagnostic`。
- [报告与复现命令](REPORT.md)；[执行前计划](../../../../CHECKPOINT_READOUT_PLAN.md)。
