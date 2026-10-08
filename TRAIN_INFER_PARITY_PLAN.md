# 训练/推理一致性与长序列适配：分阶段检查

## 本轮预注册（2026-10-08）

上一轮已确定原生 TTT 在七次连续更新中也退化，FP32 状态累积无改善。本轮首先检查训练与推理前向，再决定是否开展资源试跑。原模型与 runtime 不改动。

### 阶段 P：冻结 checkpoint 前向核验

- 使用最终 Qwen3-4B / 1B checkpoint 和已有四个 pilot stable 场景；confirm/test 不使用。
- 每场景截取相同前 6144 和 12288 token。训练代码整段前向、原生推理整段前向、原生推理每 4096 token 流式前向，共三条路径。
- 每个输入分别保持 checkpoint 的 TTT lr 和将 TTT lr 设为 0（零更新，但保留正常 KV/attention），共 16 个成对比较、48 条前向路径。TTT lr 为会话权重更新系数，与外层优化器学习率不同。
- 显式加载两种模型类，逐张量核验 checkpoint 完全相同，eval + inference_mode，统一 BF16、SDPA、关闭 TF32。训练前向指训练模型的函数，在 eval 模式检查，未启动优化器。
- 输出每条路径最后 128 个 token 的 next-token NLL、最终 hidden 的相对 L2/最大绝对误差/逐值相同比例、每比较耗时和峰值 allocated/reserved。
- 不能假定 BF16 批量/流式逐值相等。预设诊断阈值：任何来源的训练/流式 NLL 绝对差在 6144 大于 0.1 或 12288 大于 0.5，优先定位差异，暂停长序列训练；这不是统计显著性或等价性检验。即使未触发也如实报告所有差异。
- 小模型 FP32 非零候选测试覆盖完整/不完整块，验证数学路径、未来块对 writer 的有效梯度，以及后续 target 不应影响先前输出；此前原生/CBF 桥接测试同时运行。

代码：`scripts/diagnose_ttt_train_infer.py`、`tests/test_ttt_train_infer.py`、`scripts/run_ttt_train_infer.sh`。复用现有 torch/transformers/liger 依赖，无新依赖。使用显式模型类，防止 AutoModel 注册顺序混淆。

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_train_infer_20261008 bash scripts/run_ttt_train_infer.sh
```

### 阶段 R：通过 P 后的资源检查

P 未发现需要先修复的差异时，在真实训练语料上检查 6144/12288 长度的少步 forward/backward/optimizer 链路。只训练现有 TTT conv/proj，冻结 backbone/head，检查梯度、参数变更、峰值显存与耗时，不把资源检查当作算法收益实验。参数和数据选择在启动前另行登记。

### 阶段 L：资源可行后的小预算短/长对照

在同一初始化、同一可训练参数集合、相同 token 预算和优化器设定下比较短/长序列训练，另行固定数据构建、独立保留来源和成功标准，再执行。已有 pilot 只用于诊断。若稳定快记忆不能优于不写入/每步清除，继续定位 writer/readout，不扩大控制器训练。

风险：BF16 运算顺序与 attention kernel 造成小误差；全量 hidden 范数不等价于下游任务收益；四个已看过的来源不提供泛化结论；冻结 backbone 的适配不同于此前全参数预训练。两张 5090 上此前 8192 全参数训练 OOM，因此不能跳过资源检查直接扩量。
