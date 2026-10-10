# 事实优化干扰实验状态

执行前固定方案：[FACT_INTERFERENCE_PLAN](../../../FACT_INTERFERENCE_PLAN.md)。三臂sequential/joint_context/mixed_context，每臂800问题暴露，优化步800/200/200；ROOT=/home/ctj/cbf_ttt_fact_interference_v1，tmux=cbf-fact-interference-v1。

本地17项测试10通过、7无torch跳过（1.025秒）；Python/shell语法检查通过。首次服务器预检的toy干预数值分辨率断言失败，未进入prepare/训练；增强CPU fixture后定向测试通过（1.339秒），正式4B配置不变。

2026-10-10 15:22:07–15:35:36 +08:00，代码64fc0c7，wall809秒；服务器17项测试全通过（2.069秒）。原预检日志保留 `/home/ctj/cbf_ttt_fact_interference_v1_preflight_failed_077e0a4`。GPU0 sequential，GPU1 joint_context后mixed_context，三臂已完成。

终态completed_fact_interference；1200优化步、每臂800问题暴露、1224策略评分、120可回滚干预全部完成并通过独立审计。训练正确/错配分别14/16对14/16、14/16对7/16、13/16对5/16；后两臂train门槛通过，所有dev门槛失败（均0/16）。中点干扰事件9/16、10/16、13/16，支持局部干扰存在，不支持同context分组具有独特泛化优势。见[完整报告](REPORT.md)。

原始骨干保持不变，test未评分，未启动controller或其他后续训练。实验进程结束；检查时两卡0%利用率、139/18MiB显存。
