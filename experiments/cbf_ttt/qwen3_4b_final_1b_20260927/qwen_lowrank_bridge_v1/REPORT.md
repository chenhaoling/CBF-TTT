# Qwen低秩Key-Value Memory桥接：内容门通过

## 结论

在原生dense writer连续失败后，本实验把session memory改为Qwen几何中的rank-one key-value组件：实体字符串的归一化输入embedding作为key，观测值token的归一化LM-head方向作为value；读取`M k`后作为残差加入冻结Qwen的query hidden，再用原LM head输出。

该表示在来源互斥的Q和V均通过预登记内容门：Correct候选与全词表首token准确率均为 **100%**，Wrong为 **0%**，Zero为 **12.5%**。V上Correct相对Wrong/Zero/Random的答案NLL收益为11.7500/2.3364/2.3784，八来源组bootstrap 95% CI均严格大于0。两个数据域均100%，EOS率为0。

这证明冻结Qwen的实际embedding与LM-head读出可以承载可读、session-local、加性的低秩fast memory，也将此前失败进一步定位到原生dense writer的内容形成路径。本轮不是遗忘策略结果，也不声称自动从任意自然文本中识别key/value；它通过了恢复多组件累积和遗忘控制实验的前置门。

## 表示与因果对照

每个事实存储为隐式rank-one矩阵：

`M = v k^T`, `read(q) = M q`, `h' = h_Qwen + scale * read(q)`。

实际不物化2560×2560矩阵，只存低秩因子。Correct使用对应value，Wrong使用同一key的相反颜色，Zero不加memory，Random对value逐维Rademacher翻转并保持范数。query始终fresh KV，因此结果不能由原文KV复制。

## Dev scale选择

dev组0–7一次性比较`[0,1,2,4,8,16]`；按预登记的候选准确率、全词表准确率、最差对照NLL分离和最后的低scale顺序选择16。

| Scale | Correct准确率 | Wrong | Zero | Correct NLL |
|---:|---:|---:|---:|---:|
| 0 | 12.5% | 12.5% | 12.5% | 2.3382 |
| 1 | 37.5% | 0% | 12.5% | 1.6991 |
| 2 | 84.38% | 0% | 12.5% | 1.1527 |
| 4 | 100% | 0% | 12.5% | 0.4314 |
| 8 | 100% | 0% | 12.5% | 0.0334 |
| 16 | 100% | 0% | 12.5% | 0.00021 |

scale 4已经达到100%准确率；选择16来自冻结的NLL优先规则。后续多组件实验必须报告scale 4/8/16敏感性，避免将过大margin误当成遗忘优势。

## Q与一次性V

| Split / Memory | 答案NLL | 候选NLL | 候选准确率 | 全词表首token |
|---|---:|---:|---:|---:|
| Q Correct | 0.000211 | 0.000151 | 100% | 100% |
| Q Wrong | 11.7522 | 11.7521 | 0% | 0% |
| Q Zero | 2.3313 | 2.2216 | 12.5% | 12.5% |
| Q Random | 2.3426 | 2.2278 | 14.06% | 15.63% |
| V Correct | 0.000214 | 0.000152 | 100% | 100% |
| V Wrong | 11.7502 | 11.7502 | 0% | 0% |
| V Zero | 2.3366 | 2.2230 | 12.5% | 12.5% |
| V Random | 2.3786 | 2.2566 | 9.38% | 9.38% |

Q通过后才首次执行V。V的Correct-vs-Zero NLL收益95% CI为`[2.32999,2.34348]`，Correct-vs-Random为`[2.30839,2.44169]`。

## 完整性、资源与复现

- 组0–7/8–15/16–23分别作为dev/Q/V，来源互斥；每组同一query覆盖八种反事实值。
- 服务器6项聚焦测试通过；独立审计重算dev选择、Q/V汇总、门控、文件hash及V解封，`audit_passed=true`。
- 提交`8217ee3`；双RTX5090；2026-10-12 15:45:09–15:45:46 +08:00，总墙钟37秒。
- 原始逐条dev/Q/V、聚合和审计均归档；模型checkpoint未改变。

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_qwen_lowrank_bridge_v1_repro \
SOURCE=/home/ctj/cbf_ttt_fastweight_content_v2_20261012 \
bash scripts/run_qwen_lowrank_bridge.sh
```

## 下一关键实验

构造多块、多个key/value组件的Qwen低秩session memory，并冻结本轮表示。每个decision boundary对`retain`、固定alpha、global reset、按key局部删除、学习控制器和反事实oracle重放相同查询。主结果必须在新来源、多个memory seed及scale 4/8/16上显示学习遗忘优于最强固定策略，并在自然ZsRE/RULER-QA/LongMemEval-S子集验证趋势后，才能形成投稿级结论。

## 限制

- writer使用数据中显式提供的实体字符串和值token，尚未解决自然文本自动抽取。
- value为单token颜色，不能外推到多token开放词表答案。
- 单组件读取没有累积干扰，本轮不能证明遗忘策略有效。
- scale在dev上选择且16产生接近饱和的logit margin；后续必须做scale敏感性与memory容量曲线。
