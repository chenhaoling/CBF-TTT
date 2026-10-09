# 固定记录的长度对照

状态：采集与独立审计完成；最终checkpoint在6K/12K均未通过，终态 `stopped_by_length_Q`。

- hku-gpu2双RTX5090；采集提交 `a792bfc382b35a3dbd4530dc3ecf2770ba3fc59d`。
- 2026-10-10 00:11:23–00:26:30 +08:00，907秒（15分07秒）；服务器17项测试通过。
- 新采集288上下文/1152查询，三长度主分析384条件/1536查询（24K全量复用已审计旧结果；桥接不重复计数）。
- 原始模型12K/binding通过16格；6K/binding失败1格，24K失败3格。
- 最终模型binding在6K/12K/24K分别失败6/12/12格；QA也均未通过，无最终模型候选。
- 两模型各64次24K复现查询的全部候选NLL误差0、预测一致；1728条输入记录完整，summary重算和checkpoint哈希复核通过。
- M=0、完整KV，原权重不变；没有新来源、confirm、M写入/V/F或训练。
- ROOT `/home/ctj/cbf_ttt_length_readout_20261010`；tmux `cbf-length-readout-20261010` 已结束。
- [完整报告与复现](REPORT.md)；[冻结计划](../../../../LENGTH_READOUT_PLAN.md)。

归档同步说明：正式结果提交7f155aa已推送GitHub；最后同步服务器代码目录时SSH连续中断/超时，报告提交的服务器快进状态未确认。远程原始结果和独立审计已完成，本地归档完整；SSH恢复后执行git pull --ff-only origin main即可同步报告，不需重跑。
