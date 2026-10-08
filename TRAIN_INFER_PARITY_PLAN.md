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

## P 完成与 R 启动登记

P 48 条前向完成，7 项测试通过；训练/原生整段 hidden 与 NLL 全部逐值相同，训练/流式最大 NLL 差 0.005233，未触发门槛。零更新参考也有类似 hidden 运算误差，不支持以此解释原先 22 左右的长程退化。

R 固定使用现有 `mixed_1b.jsonl` 最前面的 packed 训练记录，经原 Qwen3 tokenizer 编码并加 EOS，分别截取 6144/12288 token。它可能跨文档，仅检查资源，不用来声称自然长文学习收益。每长度从最终 checkpoint 独立初始化，在一张卡上各做 3 步（两卡并行），只更新 7 层现有 conv/proj，其他权重冻结；writer master 参数和 AdamW moments 为 FP32，计算 autocast BF16，SDPA、非重入梯度检查点、lr=1e-5、weight_decay=.01、clip=1、batch=1。复用已安装 Liger 训练 loss。

每步记录 loss、每个 writer 的梯度范数、总梯度、耗时和峰值 allocated/reserved；检查每个 writer 的梯度有限非零、最终确实改变，以及冻结参数版本不变。三步用于覆盖优化器状态分配和后续稳定开销，不据此选择学习率。失败保留日志，未通过前不开始 L。不会保存或替换训练 checkpoint，输入 token 和逐参数日志只留服务器。

```bash
CUDA_VISIBLE_DEVICES=0 /home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
  -m scripts.probe_ttt_writer_training \
  --model /home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt \
  --training-data /home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl \
  --tokenizer /home/ctj/models/Qwen3-4B --length 6144 \
  --output /home/ctj/cbf_ttt_writer_resource_20261008/probe_6144.json
# 另一 GPU 同样执行 --length 12288，并改成独立输出文件。
```

## R 完成与 L 暂不启动（2026-10-08）

两长度各 3 步训练完成，14 个 writer 张量、45,964,800 个参数全部有有限非零梯度并实际更新，冻结骨干未变。6144/12288 峰值 allocated 分别 11.169/13.830 GiB，第 2–3 步平均 2.847/9.093 秒。资源可行。

但长序列三步 loss 为 1.9705→1.9188→4.4116，裁剪前梯度最大 10362；每步均实际 clip=1，梯度主要集中在第 35 层 conv。按新观察到的优化风险，L 暂不启动，先追加融合 loss/标准 CE 梯度核验和逐层相对 optimizer 位移诊断；不从这三步宣称最优学习率或训练收益。没有保存适配 checkpoint。P/R 完整结果与局限见 [完成报告](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/train_infer_parity/REPORT.md)。
