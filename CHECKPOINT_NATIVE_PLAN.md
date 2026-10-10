# 固定 12K 的 checkpoint 轨迹与原生 TTT 诊断（2026-10-10）

## 问题与预登记

上一轮原始 Qwen3-4B 的 12K/binding 通过来源×机制×查询的 75% 门槛，最终 1B 模型的 M=0 未通过。本轮区分训练过程中普通读取能力的变化，以及训练模型是否依赖原生 TTT。沿用已观察 dev，不用于泛化/论文显著性结论，不新增训练、标签或控制器。

在查看本轮评分前固定如下矩阵：

- 数据：上一轮 `length_readout_v1` 数据 SHA256 `dfaab6b69f23468d99881fa2a620f3ae4a79de60146caf4521edc6ac1e04aa22`，只取 segment_tokens=2048 的 64 个场景。8 来源组、4 机制、2 twin；每场景 target/anchor × binding/qa 四个查询。主要指标固定 binding，QA 只作辅助，不能逐 checkpoint 选格式。
- 模型：原始 Qwen3-4B；第 10000、40000、81381 步训练模型（约 122.88M、491.52M、1000.01M nominal token）。中间两点是在服务器文件清单确认后、读取评分前选定的早期/中期粗定位点。
- 7 个主要臂：original_plain；step10000_plain/native；step40000_plain/native；step81381_plain/native。每臂全部 64 场景，448 rollout / 1792 query。
- 附加 final_zero_lr：最终模型原生路径设 ttt_lr=0，限来源组 0、2 的全部 16 场景，64 query。总新计算 464 rollout / 1856 query。
- 同一串 12288 token 合并相邻 2048 段，以三个 4096 块输入，完全保留顺序/内容/查询/答案，记录源 context hash 和实际分块 hash。原始/最终 plain 全 64 场景与此前 2048 分块结果成对比较；报告所有 choice NLL 差、预测一致率，不假定 BF16 下分块严格等价。
- native 使用仓库 `inference_model.hf_qwen3` 原生 `TTTDynamicCache` 分支，cbf_enabled 为 false。原生 lr=.3、chunk=4096、7 TTT 层，全部基模型参数冻结；每场景新 cache，每查询深拷贝 prefix。无 alpha/g 控制器。查询短于一块，从整块边界开始，不能触发第四次写入。
- 全部 BF16/SDPA、关闭 TF32、seed=211，完整 KV 保留。两张 5090 按固定来源组分片，每卡顺序载入各臂。

## 质量检查与判读

1. 中间 DCP 只读，导出到本实验目录，复用官方 merge_dcp_to_hf；逐模型张量与 DCP 重新加载值完全一致，记录 metadata/输出权重 hash 及源文件大小/mtime，复制最终模型相同结构配置。拒绝覆盖旧目录。
2. 严格检查模型载入无缺失/错形；plain 仅允许忽略 14 个 conv/proj；native 不允许忽略。各臂前后参数 hash/version 一致，查询不能修改 prefix KV 或 fast weights。记录三次原生更新后各层相对位移及有限性，不修改更新尺度。
3. final_zero_lr 对同 checkpoint/plain4096：报告逐 choice NLL 误差及 top-choice/full-vocab 一致率。预设诊断触发条件为最大 choice NLL 差 >0.05 或 top-choice 一致率 <95%；触发则不得单独归因 native/plain 差异于记忆写入，先核查路径。阈值不是统计等价证明，不能事后放宽。
4. 原始 plain4096 必须在固定 binding 的全部16个 domain×regime×query 单元达到75%，否则标记新分块读取 gate 失败，不继续标签构造。分块差异仍全部报告。
5. 报告各步 plain/native 的机制、来源、target/anchor 准确率/NLL/首token命中率、8来源组 paired 差；定位退化在哪两个已测 checkpoint 之间，不推断未测步骤。即使 native 改善，也不能仅凭均值宣称局部遗忘有效，后续仍需有益记忆与选择性遗忘对照。
6. 任一加载、数值有限性、矩阵完整性或隔离检查失败均保留失败工件并停止，不扩量。运行时间/峰值 allocated 与 reserved 每 rollout/每 query 记录。

## 实现与复现

新增 `tasks/cbf_checkpoint_native.py`（固定输入、native 会话、收集/汇总）、`scripts/export_cbf_diagnostic_checkpoint.py`（经逐张量验证的只读 DCP 导出）、`scripts/run_cbf_checkpoint_native.sh`（测试、导出、双卡收集）、`tests/test_cbf_checkpoint_native.py`（分块与非零写入/查询隔离）。仅新增诊断文件，原训练/runtime/config 不修改。

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_checkpoint_native_20261010 bash scripts/run_cbf_checkpoint_native.sh
```

环境变量 ROOT / SOURCE（上一轮 length 输出）/ PYTHON / ORIGINAL / CHECKPOINTS。原始语料、逐样本结果、模型仅留服务器；GitHub 归档计划、代码、聚合报告与审计。最终以完整 summary+audit 为完成依据，不以 tmux 消失为依据。
