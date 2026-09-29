# 标题答案梯度与 In-Place TTT 候选方向诊断（2026-09-29）

## 目的与方法

上一轮新论文标题回忆在 KV 清空后没有稳定写入收益。本轮按执行前的 [`DUAL_GATE_QUERY_GRADIENT_DIAGNOSTIC.md`](../../../../DUAL_GATE_QUERY_GRADIENT_DIAGNOSTIC.md)，检查原始候选 `ΔW` 是否沿着降低标题损失的方向改变会话快记忆。原始 In-Place TTT 更新是当前 chunk 表示的外积；本轮用标题答案对快记忆求梯度，只作为**事后 oracle** 对照，不能作为在线控制器输入。

仍用最终 Qwen3-4B 1B-token checkpoint `/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt`，bf16，hku-gpu2 两张 RTX 5090。只取上一轮标题试点 **8 个 train 源组**的 `new_only/old_only` 各一条，共 16 条；dev/test 均未使用。固定第二块候选、`α=1`、KV 清空和相同查询，在 `g=0` 的快记忆状态计算 `∇_M J`。比较原始 `+ΔW` 与总 Frobenius 范数相同的 `−∇_M J` 方向，步长均为 `0.25/1`；骨干权重冻结。所有 16 条的原始 `s=0/1` 均与已有 `memory_only` 四角点标签一致，梯度前向与推理前向 NLL 在预设 `1e-4` 容差内一致。

## 结果

| train 条件 | 原始方向局部导数为正 | 原始写入收益 `s=.25 / 1` | oracle 收益 `s=.25 / 1` | oracle 任一步 >0.005 |
|---|---:|---:|---:|---:|
| `new_only`（8 组） | **7/8** | **−0.00907 / −0.01764** NLL | **+3.15839 / +1.15395** NLL | **8/8** |
| `old_only`（8 组） | **5/8** | **+0.00048 / −0.01011** NLL | **+2.76696 / −0.01698** NLL | **8/8** |

`new_only` 的无更新基准平均 NLL 为 5.136；候选 `ΔW` 的总范数平均 1.270，答案梯度范数平均 35.172。梯度与候选的平均余弦只有 **+0.000265**，绝对余弦均值 **0.000387**，即方向几乎正交；`7/8` 的导数符号与原始写入造成的平均损害同向，但如此小的余弦可能受 bf16/有限样本影响。`old_only` 在 oracle `s=1` 反而不优于基准，说明等范数大步长会过冲，不能把 oracle 结果理解为一套可直接部署的更新规则。

预写的机制解释条件在 train 子集上成立：原始候选未带来稳定收益，而答案梯度方向能在同一快记忆接口降低答案 NLL。这支持**当前候选对该标题任务缺少目标对齐**的解释。它不证明原 In-Place TTT 在其他任务或正常 KV 推理下无效，也不证明基于答案梯度的在线方案可行。oracle 直接使用未来答案，收益包含针对已知标题的过拟合。

两卡共完成 16/16 条；平均每条 **0.970 秒**、最大 **1.512 秒**，最大 CUDA allocated/reserved 为 **14.631/16.031 GiB**，无 OOM。远程 tiny Qwen 和纯数据 10 项测试通过；一条真实 smoke 的 CLI `Path` JSON 打印问题已修正，并用新输出文件核实成功退出码。本地从两份逐条轨迹复算的 [`summary.json`](summary.json) 与远程 SHA256 相同。逐条轨迹只保留在本地忽略目录和远程 `/home/ctj/cbf_ttt_query_gradient_20260929`，不上传 GitHub。

## 复现

完整命令见 [`MODIFICATION_LOG.md`](../../../../MODIFICATION_LOG.md) 对应章节。在 hku-gpu2 的代码目录，先运行一条 `--max-scenarios 1` smoke；随后让 GPU 0/1 分别读取上一轮的 `shards/train_shard0.jsonl` 和 `shards/train_shard1.jsonl`，输出 `train_shard0_diagnostic.jsonl`、`train_shard1_diagnostic.jsonl`，再汇总：

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_query_gradient_20260929
python -m scripts.summarize_cbf_query_gradient --input "$ROOT/train_shard0_diagnostic.jsonl" "$ROOT/train_shard1_diagnostic.jsonl" --output "$ROOT/summary.json"
```

新增脚本均为独立诊断入口，未修改 baseline 模型、预训练参数、`CBFSession.commit_both` 或原控制器语义。

## 决策与下一步

正式反事实标签、双门控控制器训练及公开基准仍暂停。下一步应使候选生成机制在**训练阶段**学习真实写入效用：用独立训练文档构造“读入片段—查询片段”的 episode，只用训练 episode 的未来查询损失监督候选生成器，推理时仍仅从当前 chunk 产生 `ΔW`；同时放入重复信息和噪音片段验证 `g` 是否能拒写。先在新的、未用于本轮诊断的源组上测试固定 `g=1` 的有益写入，再恢复四条件联合阶段门。不能在当前已看过的 train/dev/test 源组上调候选机制后声称独立验证。

风险包括：KV 清空造成的分布偏移、标题精确 NLL 与生成准确率不同、bf16 极小余弦的数值敏感性、仅 8 个 train 组、以及事后答案梯度必然含标签泄漏。早前该模型的 float32 推理尺度试验峰值 reserved 已约 30.371 GiB，单卡 32 GiB 环境下直接做 4B float32 autograd 有较高 OOM 风险；若要核查导数微小符号，应采用分层/低秩梯度或更大显存设备。
