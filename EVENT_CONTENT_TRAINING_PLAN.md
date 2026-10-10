# 阶段A第二轮：少量世界的编号记忆与成对内容训练

## 执行前固定

上一轮100步原生writer链路通过，但完整答案0/16，正确与错配的改善不稳定。本轮检验小数据内容记忆，仍不进入控制器/联合训练，不回改上一轮门槛。

- 两个并行臂 `ce` / `content_pair`，hku-gpu2各用一张5090，独立从原始Qwen3-4B和seed301零conv初始化；不续接上轮writer。
- 复用上轮4096块/问题/答案原样。按group_id排序取前两个train世界，16问题/4上下文/8个twin问题对；dev固定原来两个世界16题，test不读。其他6个train世界本轮不训练/不评分。对照错配在每个split所选两个世界间交换，同twin索引。twin负对照仅用于值真的改变的anchor问题。
- 相同seed302确定400步顺序，8对每轮洗牌，50轮，每个问题恰好出现50次。每步分别写入两份twin记忆，以fresh KV读取两份正确及两份交叉记忆。训练时答案只进入损失和teacher forcing，不进入write/query prefix。
- 冻结骨干；原生7层conv/proj FP32 master/AdamW状态，BF16计算，lr1e-7、clip1、weight_decay0、ttt_lr0.3、g1。共400 optimizer steps，保存/评估0、100、400；终点固定400，不按dev挑checkpoint。沿用delta/base范数>1及非有限停止线，无自动调参或延长预算。

## 两个训练目标

答案联合分词仍沿用已有边界。逐token解码划分共享 `code_` 前缀与数字，断言前缀合并为code_、数字合并为五位目标数字；EOS另列。

1. `ce`：两个twin各自答案加EOS的普通全词表CE取平均。
2. `content_pair`：各自 `0.75 L_digits + 0.125 L_prefix + 0.125 L_EOS` 取平均；仅anchor对增加 `0.5 * mean(softplus(0.2 + NLL_digits(correct_memory) - NLL_digits(twin_memory)))`。两份记忆都保留梯度。不变值的非anchor不使用假负标签。

两臂都计算正确/交叉前向和成对项（ce臂权重为0），不改变推理更新规则。相同步数、样本和前向预算；实际耗时/峰值均报告，不声称严格相同硬件FLOPs。两项改进组成一个训练方案，本轮不能分别归因数字权重或成对项。

## 固定评估与判据

每个checkpoint评估所选train16题+dev16题：正确/空/异世界错配/完整KV，每split4个anchor另测twin。合计每臂408策略查询、两臂816。每次读fresh KV；完整KV是普通阅读对照。16token自由生成、不屏蔽EOS。

记录完整EM（沿用上轮）、首行EM、首个完整数字编号正确率（新登记内容指标）、答案/数字/EOS NLL、首token与im_end概率。编号提取固定为首个 `\bcode_[0-9]+\b`，必须与完整五位目标一致；不从候选选答案。所有结果和资源逐条保存。

- 训练可记忆门槛：400步正确记忆编号正确率≥75%，分别高于空和错配≥25百分点；两train世界各≥50%；anchor正确数字NLL低于twin。EM另外报告，不能把新内容门槛叫旧EM门槛通过。
- dev初步内容门槛：正确编号率≥25%，高于空/错配≥12.5百分点；两个dev世界相对错配的数字NLL均改善≥0.05；anchor正确数字NLL优于twin。只有2个世界，不作显著性/泛化定论。
- 成对方案相对CE：同一终点train/dev逐组同时报告，需实际超过CE才称有增益；只一个seed，不做正式统计推断。
- 若只有train通过，下一步扩充独立世界/值变化而非控制器；若train也失败，保留负结果并定位优化/容量问题；不反复延长到通过。若train/dev均通过，也先安排独立新世界确认，再进入多块retain。

## 检查、记录与兼容性

复用 `cbf_ttt/event_writer.py` 可微写入、原始骨干hash及上轮score；新增独立任务、shell、目标/数据测试、完成审计，不改baseline/runtime/上轮训练。CPU小模型验证成对梯度及twin负标签隔离；服务器先跑测试，失败不启动。每步梯度/范数/时间/显存、checkpoint和数据hash、冻结骨干前后hash保存。主报告及修改记录纳入Git；token/原文/权重留远程。

主要风险：极小训练组过拟合、随机编号难学、4096背景稀释、成对项可仅降低错误记忆概率、重复模板捷径、400步预算可能不足；因此必须同时看真实生成、空/错配与twin，不能以训练loss或排名成功代替记忆。

运行：`ROOT=/home/ctj/cbf_ttt_event_content_v1 bash scripts/run_cbf_event_content.sh`。SOURCE默认上轮`/home/ctj/cbf_ttt_event_writer_100step_v1`，MODEL/PYTHON控制模型/环境；新ROOT拒绝覆盖。本轮2世界/400步/权重/门槛冻结，改变设计需新计划/目录。

## 实际完成（原冻结设计不变）

2026-10-10 14:20:09–14:28:25 +08:00，496秒。13项服务器测试和独立审计通过，800优化步/816策略查询齐全。CE/content_pair训练完整编号4/16、7/16，开发均0/16；两臂train/dev门槛全未通过。成对方案训练EM较高但数字NLL/twin绑定不一致，不支持方法有效结论。未追加预算或进入B/C；[完整报告](experiments/cbf_ttt/event_content_v1/REPORT.md)记录全部分组、资源、代码/命令及后续建议。
