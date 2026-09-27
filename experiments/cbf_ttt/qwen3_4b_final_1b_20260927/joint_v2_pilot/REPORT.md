# 双门控 `joint_v2` 修订试点结果（2026-09-27）

## 目标与执行

按 [`DUAL_GATE_V2_EXPERIMENT.md`](../../../../DUAL_GATE_V2_EXPERIMENT.md) 的执行前协议，检查“新 `ΔW` 是否写入”和“旧会话快记忆是否保留”能否在未展示过目标答案的任务上形成多样、可辨的决策。`joint_v2` 是 opt-in 实验协议；原始 In-Place TTT baseline、旧 `forget`/`write`、`joint_v1` 的默认行为和历史标签语义没有改变。

- 最终模型：Qwen3-4B In-Place TTT 1B-token checkpoint，`/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt`；`config.json` SHA256 `9ffb7420c132c9ea7de55e221baac89594699f942bc14f2327fc5014e038e672`。
- 采集代码 Git commit `b8dd8a9`；阶段门分析脚本在采集启动后作为 `9ec2705` 提交，但阈值和规则已写在启动前提交的 `DUAL_GATE_V2_EXPERIMENT.md`。远程代码目录 `/home/ctj/cbf_ttt_joint_exp_20260927`，输出目录 `/home/ctj/cbf_ttt_joint_v2_pilot_20260927`。
- 两张 RTX 5090 并行；12 个源组（train/dev/test 为 8/2/2）、每组四类场景、共 48 条标签。每条在第二个 4096-token chunk 后比较 `α,g∈{0,0.5,1}` 的 9 个动作，分别评分 `gap=0` 和 `gap=2×4096` 自然背景 chunk。后续 gap 使用固定 `11` 更新，attention KV 不清空。
- 场景由同一 1B FineWeb-Edu/LongCrawl64 混合语料的 192 条背景记录填充。经实际 Qwen tokenizer 检查，这些记录均为 6143 token，构造 chunk 时只截取、没有重复平铺。场景文件 SHA256 `c885e6b1ce467eb57438402e2153047c48fee0f79b18e542693d037c8beb8564`。
- 实际 tokenizer 解码审计：48 个场景、96 个规则迁移查询均未在 context/gap 中出现目标答案字符串；另有 24 个独立算术诊断查询，未纳入该精确字符串审计。生成事实与查询仍是合成任务，不代表自然长程问答。

## 结果

两卡 tmux worker 于服务器时间 21:53:02 启动，21:59:14/21:59:22 完成；48/48 条均成功，未 OOM。包含四完整 chunk baseline 等价测试在内，远程 PyTorch 相关测试共 23 项通过；本地不依赖 PyTorch 的 19 项检查通过。四类场景取同一源组重复采集，其所有逐查询、逐 future、逐动作损失与首次运行相同（最大绝对差 0）。

| 指标 | 结果 |
|---|---:|
| 每条标签平均 / p95 耗时 | 14.915 / 15.842 秒 |
| 峰值 CUDA allocated / reserved | 14.828 / 18.564 GiB |
| 聚合四角点最佳 `00/01/10/11` | 48 / 0 / 0 / 0 |
| `gap=0` 四角点最佳 `00/01/10/11` | 39 / 3 / 4 / 2 |
| `gap=2` 四角点最佳 `00/01/10/11` | 47 / 0 / 1 / 0 |
| `gap=0` 平均 `J00/J01/J10/J11` | 0.540 / 0.568 / 0.552 / 0.592 |
| `gap=2` 平均 `J00/J01/J10/J11` | 0.878 / 3.979 / 1.472 / 6.965 |
| 网格内部点优于最佳角点 | 0/48 |

按执行前定义的 `0.005` NLL 阈值，在 `gap=2` 下：有用新信息、可信纠正的写入收益均为 **0/12 源组**；旧规则相关情景的保留收益也是 **0/12**。噪声/重复场景的拒写收益分别为 12/12 与 11/12 源组。只有 `00` 在至少三个独立源组获得有意义的角点优势。机器可读的全部检查在 [`gate.json`](gate.json)，按间隔和查询种类的损失在 [`summary.json`](summary.json)，泄露审计在 [`leakage_audit.json`](leakage_audit.json)。

长间隔的大损失增幅不能简单归咎于 CBF 分支实现：同一真实四块场景的原生 Qwen TTT `11` 查询 NLL 为 **8.6255**，CBF `11` 为 **8.6567**，差 **0.0312**；四块 tiny-model 对照也通过。两路径在 bf16 下并非逐值严格相等，仍需更多状态与精度对照。短间隔已有 39/48 条偏向 `00`，长间隔进一步放大这种偏向。当前证据说明这组合成规则迁移任务和无关背景 gap 不适合提供所需的双门控监督；不能据此声称真实推理中 TTT 写入必然有害。

## 阶段门与后续

**阶段门未通过。停止正式标签扩量、联合控制器训练及六项公开基准评测。** 内部系数比例和显存检查通过，但有用候选写入、旧记忆保留、角点多样性均未达到执行前阈值。48 条探索性 dev/test 场景已被查看；后续正式评测须重建独立源组。

下一轮应先做机理诊断：分别测量每个 gap chunk 的原生 `11`、固定 `00` 与“当前门控后冻结后续写入”路径，定位损失增幅来自候选写入、累计后续更新还是任务/KV；然后把监督目标转为与原始 TTT 预训练更一致的未见自然 continuation loss，并继续保留真实 KV 和分离的旧/新任务指标。只有在有用写入与旧记忆保留出现稳健正收益后，才重新评估正式规模和控制器结构。

## 复现与数据位置

场景生成、分片和两卡命令见 [`DUAL_GATE_V2_EXPERIMENT.md`](../../../../DUAL_GATE_V2_EXPERIMENT.md)。已有输出目录不可覆盖；复现时使用新目录。分析命令：

```bash
ROOT=/home/ctj/cbf_ttt_joint_v2_pilot_20260927
python -m scripts.audit_cbf_joint_v2_scenarios --data "$ROOT/scenarios.jsonl" --tokenizer /home/ctj/models/Qwen3-4B --output "$ROOT/leakage_audit.json"
python -m scripts.summarize_cbf_joint_pilot --labels "$ROOT"/{train,dev,test}_shard{0,1}_joint_labels.jsonl --output "$ROOT/summary.json"
python -m scripts.evaluate_cbf_joint_v2_gate --labels "$ROOT"/{train,dev,test}_shard{0,1}_joint_labels.jsonl --output "$ROOT/gate.json"
```

原始逐样本 JSONL 只保存在本机本目录和远程输出目录，由本目录 `.gitignore` 排除；GitHub 仅保存聚合结果、场景元数据、代码与报告。自动审批此前拒绝上传完整派生标签，未尝试绕过。
