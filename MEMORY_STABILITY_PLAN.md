# 多 chunk 快记忆稳定性检查（执行前固定）

## 问题和范围

上一轮 32 场景的局部 oracle 全部选择清除，持续累积 NLL 达 22 左右。本轮定位数值/更新路径与累积不稳定的关系，不训练或选择控制器，不重新预训练，不改变上一轮失败判据。

仅复用选择性遗忘数据中 **4 个 pilot 来源组的 stable 场景**，每场景顺序读取 7×4096 token。它们已被看过，是机制诊断数据；不使用 confirm 或旧 writer test，不声称独立泛化验证。冻结最终 Qwen3-4B / 1B checkpoint，统一 BF16 backbone/head、保留正常 KV；关闭 TF32。

## 九条固定路径

| 路径 | 更新/读取运算 | 累积状态 | 提交规则 |
|---|---|---|---|
| native | 仓库原生 TTT forward | BF16 完整 W | 原生累积 |
| cbf | 已有 CBF forward | BF16 增量 M | 全保留 |
| clear | 已有 CBF forward | BF16 增量 M | 每 chunk 清除旧 M |
| none | 已有 CBF forward | M=0 | 不提交新更新 |
| native_ops_m | 与原生一致的 contraction/读取 | BF16 增量 M | 全保留 |
| native_ops_w | 与原生一致的 contraction/读取 | BF16 完整 W | 全保留，原生桥接核验 |
| fp32_m | CBF 运算 | FP32 增量 M | 全保留，读取前合成并转 BF16 |
| native_ops_m_fp32 | 原生运算 | FP32 增量 M | 全保留，读取前合成并转 BF16 |
| native_ops_w_fp32 | 原生运算 | FP32 完整 W | 全保留，读取前转 BF16 |

FP32 指快记忆累积算术，不是全模型 FP32，不涉及重新恢复 checkpoint 精度。所有候选仍从 BF16 激活产生。原生完整 chunk 不使用 down-projection bias，本诊断拒绝有该 bias 的模型。

## 测量和核验

- 每 chunk 提交后，在独立克隆中评分后续自然文本：32 token query +128 token answer。前六次来自下一自然 chunk，第七次沿用已有最后一个 query/answer。评分克隆不进入后续流。
- 记录每层 ||M||/||W0||、候选 ||delta||/||W0||（原生路径不可直接观察时置 null）、实际有效权重变化、M 与 delta 对齐、每步 NLL、时间和峰值 allocated/reserved。
- 对 native 和 native_ops_w，在每步计算完整有效权重和 hidden 字节哈希；如不相等则明确指出桥接未能逐值复现，不将差异单独归因于累积形式。
- 在每场景首个 cbf chunk 的相同 h、卷积后 target、P 上，比较原生 contraction 与 CBF 两次 matmul 的候选，并用这组输入的 FP32 运算作参考。该参考不改变主轨迹。
- cbf/clear 的最后三点评分必须逐值复现上一轮该场景持续累积/全程清除结果，否则停止解释并修复诊断。
- 检查参数版本不变、基础 down-projection 字节哈希不变、query 不改变会话状态；测试无误后按来源组双卡分片，所有九路径完整保留，不按结果增加超参数点。

## 解释顺序

先核验重现和桥接，再比较路径，再看记忆范数与 NLL 随 chunk 的变化。原生也退化则不能仅归因于 CBF 表示；切换某数值路径改善只支持该操作的配置效应。候选运算误差、有效权重更新范数与任务收益不是同一指标。只有累计路径有稳定任务收益，才值得重新评估更细粒度遗忘。长序列训练覆盖不足仍是待验证解释，本轮不因推测而启动训练。

## 实现与运行

新增独立 `scripts/diagnose_cbf_memory_stability.py`、`scripts/run_cbf_memory_stability.sh` 和 `tests/test_cbf_memory_stability.py`。自定义运算仅在诊断进程内临时替换 CBF hook，退出恢复；原 runtime 和模型文件不修改。无新依赖。

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_memory_stability_20261008 \
bash scripts/run_cbf_memory_stability.sh
```

输出目录必须全新；原始逐步/逐层轨迹保留远程，只发布代码、聚合曲线和审计。必要时失败停止，不缩短上下文或更换数据来绕过失败。

## 执行状态（2026-10-08）

九条路径全部完成，共 36 条轨迹、20 项测试通过，流水线 133 秒。原生/桥接全部 28 个边界逐值一致，24 个历史评分值完全复现。原生累积、CBF 累积和 FP32 累积均严重退化，清除/不写入保持较低 NLL。没有启动控制器扩量或长序列再训练。

完整结果、数值解释边界和下一步训练/推理一致性及短长序列对照建议见 [完成报告](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/memory_stability/REPORT.md)。
