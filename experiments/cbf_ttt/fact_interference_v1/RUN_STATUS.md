# 事实优化干扰实验状态

执行前固定方案：[FACT_INTERFERENCE_PLAN](../../../FACT_INTERFERENCE_PLAN.md)。三臂sequential/joint_context/mixed_context，每臂800问题暴露，优化步800/200/200；ROOT=/home/ctj/cbf_ttt_fact_interference_v1，tmux=cbf-fact-interference-v1。

本地17项测试10通过、7无torch跳过（1.025秒）；Python/shell语法检查通过。服务器全部测试通过后才启动。当前尚未训练，无干扰或效果结论。
