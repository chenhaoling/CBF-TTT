# 最终checkpoint Writer内容绑定恢复：Q门失败

## 结论

本实验从完成1B-token预训练的Qwen3-4B最终checkpoint出发，只训练官方七层`ttt_conv/ttt_proj`，使用同一步反事实配对、八值均衡、Correct-vs-Wrong/Zero对比目标和fresh-KV读取。结果仍未形成可泛化的当前事实绑定。

独立Q的Correct八选一和全词表首token准确率均为 **10.94%**，低于Wrong/Zero/Random的12.50%/12.50%/14.06%。Wrong-minus-Correct答案NLL为 **-0.01074**，8来源组bootstrap 95% CI **[-0.02051,-0.00293]**，即错误memory显著优于正确memory。Correct比Zero和Random分别低0.18384和0.17636 NLL，但准确率没有改善。这说明训练学到了“存在一个训练后memory”的通用语言概率效应，没有学到memory内容对应哪个颜色。

预登记Q门失败，组16–23的V保持未评分。当前证据不支持继续扩大同一dense writer的训练步数或学习率搜索；下一迭代应改变memory表示，使用可读的低秩key-value fast memory，再把遗忘控制接入实际Qwen查询。

## 设计

- 数据来自已审计`fastweight_content_v2`场景；每组仅轮换唯一颜色token，其余自然背景、实体、query和长度固定。
- 调参训练组0,1,2,3,4,7；调参开发组5,6；最终训练组0–7；Q组8–15；V组16–23。
- 每组同一query覆盖八种颜色，成对更新使用相隔四位的值，阻止慢参数记住固定实体答案。
- 第一块4096 tokens只提供冻结attention prefix；第二块生成可微`Delta W`；查询清空KV，只保留memory。
- 冻结全部非writer参数；writer与AdamW状态FP32，BF16 forward；逐层`||Delta W||/||W||` cap为0.3。
- 损失为八选一CE、0.25倍全词表CE、同一步对称swap和0.5倍Correct-vs-Zero margin。

## 调参结果

两个学习率各训练192步，初始化writer逐tensor相同。

| 学习率 | 末32步准确率 | Dev Correct | Dev Wrong | Correct-vs-Wrong NLL | Dev门 |
|---:|---:|---:|---:|---:|---|
| 3e-7 | 14.06% | 6.25% | 18.75% | +0.01172 | 失败 |
| 1e-6 | 12.50% | 12.50% | 12.50% | +0.00391 | 失败 |

按冻结选择规则，最差Correct-vs-control NLL较高的`3e-7`被选中。随后从未修改的最终checkpoint重新初始化，在组0–7完成256步训练，不沿用调参权重。

## 独立Q

| Memory | 答案NLL | 候选NLL | 八选一 | 全词表首token | EOS |
|---|---:|---:|---:|---:|---:|
| Correct | 2.14745 | 2.09505 | 10.94% | 10.94% | 0% |
| Wrong | **2.13671** | **2.08431** | 12.50% | 12.50% | 0% |
| Zero | 2.33129 | 2.22161 | 12.50% | 12.50% | 0% |
| Random | 2.32381 | 2.21399 | 14.06% | 14.06% | 0% |

| Correct相对对照 | NLL gain | 95% CI | 八选一增益 |
|---|---:|---:|---:|
| Wrong | -0.01074 | [-0.02051,-0.00293] | -1.56点 |
| Zero | +0.18384 | [0.17561,0.19175] | -1.56点 |
| Random | +0.17636 | [0.15394,0.19742] | -3.13点 |

FineWeb/LongCrawl的Correct八选一为12.50%/9.38%。结果不是某一域单独导致。Correct优于Zero/Random的NLL信号与Correct劣于Wrong同时出现，明确区分了通用扰动与内容绑定。

## 完整性与资源

- hku-gpu2双RTX5090，2026-10-12 15:19:06–15:33:15 +08:00，总墙钟14分09秒。
- 最终训练单步均值1.574秒，峰值allocated/reserved 12.558/13.643 GiB。
- 最终256步loss从前32步4.140降到末32步3.874，但准确率从10.94%降到6.25%，未出现可用学习轨迹。
- 服务器9项聚焦测试通过；所有训练预算、学习率、相同初始化、冻结主干、FP32状态、文件hash、Q重算、Random逐层等范数和V封存均由独立审计验证，`audit_passed=true`。
- 采集提交`672f9a9`。writer checkpoint、逐步日志留在服务器`/home/ctj/cbf_ttt_final_writer_binding_v1`；Git归档设计、逐条dev/Q指标、汇总和审计。

## 复现

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_final_writer_binding_v1_repro \
SOURCE=/home/ctj/cbf_ttt_fastweight_content_v2_20261012 \
MODEL=/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt \
bash scripts/run_final_writer_binding.sh
```

## 论文含义与下一步

这次结果联合此前实验排除了三类简单解释：输出格式、固定答案shortcut、以及“最终预训练writer只差少量监督”。更可信的瓶颈是当前dense down-projection delta及冻结LM读取路径没有形成稳定的实体到值寻址。

下一实验采用低秩key-value fast memory：由上下文产生key/value槽，query通过相似度读取并将结果作为Qwen残差；先在相同三段隔离上验证Correct > Wrong/Zero/Random，再固定memory writer，恢复多块累积与只训练遗忘策略。该表示必须保留session-local、在线更新、无主干权重改写和可控`alpha`，从而服务论文的遗忘主线，而不是继续为dense writer增加训练预算。

## 限制

- 当前监督任务是单token颜色，不是自然QA benchmark；阴性结果限定此writer结构和训练协议。
- 调参dev只有两个来源组，因此只用于选择固定学习率；核心结论来自八个独立Q来源组。
- 只训练256步且只有一个初始化seed；但训练与dev/Q均未出现内容方向信号，扩大同一协议缺乏依据。
- V未评分，不能把Q阴性结果表述为测试集效果。
