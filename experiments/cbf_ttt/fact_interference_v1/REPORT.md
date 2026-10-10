# 逐问题更新与同上下文联合监督：事实优化干扰

## 结论

**观察到了局部优化干扰，但它不能单独解释当前泛化失败。** 在逐问题训练的中点，16个单题干预中9个满足“自身数字NLL下降≥0.01且另一个事实上升≥0.01”；例如更新一个unit后自身下降0.061854，而anchor上升0.144978。从相同checkpoint出发作四题联合一步，四context共16个事实没有超过恶化阈值，平均数字NLL也改善更多。该证据直接来自可回滚有限更新，不仅来自梯度负余弦。

**同上下文联合监督在本例中更省写入计算，并通过了小样本训练内容门槛；尚无独特泛化优势。** 三臂终点训练正确/错配记忆完整答案为sequential14/16对14/16、joint_context14/16对7/16、mixed_context13/16对5/16。逐问题的高训练准确率不要求正确记忆，因此未通过内容门槛；两种batch训练通过train门槛，但开发集均0/16。mixed的训练数字NLL更低（0.1021 vs joint0.2071），joint只多答对1题，不能据此断言同context分组比一般batch更有效。

所有六个检查点的同状态联合一步，四事实平均数字NLL变化均好于单题一步平均；但联合也在部分状态伤害个别事实，且joint训练自身的终点仍有7/16单题干扰事件。sequential终点事件为0/16同时仍有4/48非目标事实损失上升；这是未满足“自身改善≥0.01”的联合条件，不能写成完全没有干扰。不同臂/阶段的发生率不可直接当作一个总体概率。

建议下一步转向独立世界和动态赋值语料的泛化训练，保留多问题batch与错配/twin检验；目前不进入遗忘控制器。不能将writer训练期的梯度竞争直接当作session记忆该被遗忘的证据。

## 目标、假设和范围

比较逐问题、同上下文四问题联合、跨上下文四问题批量监督，并用可回滚的一步AdamW干预判断是否存在“学会一个事实使另一个事实变差”。训练方案优劣和干扰是否存在是两个独立问题。

实际执行 `64fc0c774b5c2f61e3065fee20a38484e2c8caad`；hku-gpu2双5090，2026-10-10T15:22:07+08:00 至 2026-10-10T15:35:36+08:00，wall 809秒（含准备、训练、评分、梯度/干预；最终独立审计另计）。GPU0 sequential，GPU1依次joint_context/mixed_context。

原始Qwen3-4B骨干冻结，新初始化seed301的原生7层conv/proj FP32，BF16计算，AdamW lr1e-7/clip1/无weight decay，ttt_lr0.3、g1。全部普通答案+EOS CE，不包含上一轮成对或数字加权loss。

复用前轮32行4096装箱：两个train世界16题/4上下文、两个dev世界16题/4上下文，test未读取评分。每context4事实。seed403固定50轮，每题每轮一次。sequential与joint共用顺序，mixed转置每轮4×4题表。正确记忆读取使用fresh KV；联合组共享可微写入，四问题各自fresh KV。

| 臂 | 优化器更新次数 | 问题暴露数 | 训练写入块数 |
|---|---:|---:|---:|
| sequential | 800 | 800 | 800 |
| joint_context | 200 | 800 | 200 |
| mixed_context | 200 | 800 | 800 |

样本暴露相同，参数更新次数/计算量并不相同。joint vs mixed控制batch大小和更新数，仍有分组与顺序差异；不得把joint vs sequential的差异直接归因同context机制。每题50次暴露是重复学习，不是800条独立样本。

## 800问题暴露终点

编号正确率为预登记的首个完整code_数字串匹配；EM要求strip后的全部生成文本匹配。16新token上限、不屏蔽EOS。

| 臂 | 划分 | 策略 | 编号正确率 | 完整EM | 答案NLL | 数字NLL |
|---|---|---|---:|---:|---:|---:|
| sequential | train | correct | 87.50% | 87.50% | 0.152312 | 0.213237 |
| sequential | train | empty | 0.00% | 0.00% | 4.808934 | 2.767957 |
| sequential | train | wrong | 87.50% | 87.50% | 0.276087 | 0.386521 |
| sequential | train | full_kv | 87.50% | 0.00% | 0.075021 | 0.000015 |
| sequential | dev | correct | 0.00% | 0.00% | 8.193857 | 11.471340 |
| sequential | dev | empty | 0.00% | 0.00% | 4.277348 | 2.662896 |
| sequential | dev | wrong | 0.00% | 0.00% | 8.664144 | 12.129613 |
| sequential | dev | full_kv | 75.00% | 0.00% | 0.105974 | 0.000111 |
| joint_context | train | correct | 87.50% | 87.50% | 0.148050 | 0.207134 |
| joint_context | train | empty | 0.00% | 0.00% | 4.808934 | 2.767957 |
| joint_context | train | wrong | 43.75% | 43.75% | 0.401151 | 0.561395 |
| joint_context | train | full_kv | 87.50% | 0.00% | 0.075021 | 0.000015 |
| joint_context | dev | correct | 0.00% | 0.00% | 5.916080 | 8.280446 |
| joint_context | dev | empty | 0.00% | 0.00% | 4.277348 | 2.662896 |
| joint_context | dev | wrong | 0.00% | 0.00% | 6.190691 | 8.665024 |
| joint_context | dev | full_kv | 75.00% | 0.00% | 0.105974 | 0.000111 |
| mixed_context | train | correct | 81.25% | 81.25% | 0.073066 | 0.102075 |
| mixed_context | train | empty | 0.00% | 0.00% | 4.808934 | 2.767957 |
| mixed_context | train | wrong | 31.25% | 31.25% | 0.403546 | 0.564815 |
| mixed_context | train | full_kv | 87.50% | 0.00% | 0.075021 | 0.000015 |
| mixed_context | dev | correct | 0.00% | 0.00% | 6.605793 | 9.246504 |
| mixed_context | dev | empty | 0.00% | 0.00% | 4.277348 | 2.662896 |
| mixed_context | dev | wrong | 0.00% | 0.00% | 6.706576 | 9.386912 |
| mixed_context | dev | full_kv | 75.00% | 0.00% | 0.105974 | 0.000111 |

## 干扰的一步干预证据

每臂在400/800暴露checkpoint测全部4个train上下文。每context从同一writer/完整AdamW状态分别做4次单题更新和1次四题联合更新；每个分支后重新写入该context并测4题，再逐张量恢复参数、动量和step。无开发数据参与干预，分支不影响主训练。

一个干扰事件要求自身数字NLL下降至少0.01，且同context至少一个其他事实上升至少0.01。受损非目标事实对数统计所有上升≥0.01的非对角元素，不要求自身改善。梯度冲突为未裁剪CE梯度cos<−0.05；两类证据不同，AdamW动量可能使原始梯度角度与实际有限更新不一致。

| 臂 | 问题暴露 | 干扰事件/16单题分支 | 受损其他事实/48 | 负梯度对/24 | 联合一步受损事实/16 |
|---|---:|---:|---:|---:|---:|
| sequential | 400 | 9/16 | 13/48 | 13/24 | 0/16 |
| sequential | 800 | 0/16 | 4/48 | 5/24 | 0/16 |
| joint_context | 400 | 10/16 | 13/48 | 6/24 | 2/16 |
| joint_context | 800 | 7/16 | 14/48 | 9/24 | 2/16 |
| mixed_context | 400 | 13/16 | 30/48 | 3/24 | 3/16 |
| mixed_context | 800 | 6/16 | 15/48 | 6/24 | 2/16 |

| 臂 | 问题暴露 | 单题一步的四事实平均数字NLL变化 | 联合一步的四事实平均变化 |
|---|---:|---:|---:|
| sequential | 400 | -0.011416 | -0.062759 |
| sequential | 800 | +0.000982 | -0.021729 |
| joint_context | 400 | -0.109778 | -0.134974 |
| joint_context | 800 | -0.028026 | -0.043528 |
| mixed_context | 400 | +0.003145 | -0.052210 |
| mixed_context | 800 | -0.021768 | -0.054373 |

变化为after−before，负数表示改善。单题列先对每个单题分支的四事实求均值，再平均分支；联合列每context的一次联合更新。两者从同一checkpoint状态出发。联合受损率和“其他事实受损率”的目标集合不同，不直接相减解释为干扰减少百分比。各臂训练出的checkpoint不同，也不能把其诊断率差别视为同状态因果对照。

### 具体干扰例子

每臂每个checkpoint按固定context/id顺序展示第一个满足阈值的事件；非择优最大变化。全部4×4数字NLL变化/梯度余弦及联合向量在summary.json的probes.contexts中。

- sequential/400: 更新 `053736ed3ca26ad0.0.unit_3ca26ad0_0`，自身数字NLL变化 -0.061854；`053736ed3ca26ad0.0.asset_053736ed` 变化 +0.144978。
- sequential/800: 没有达到预登记阈值的干扰事件。
- joint_context/400: 更新 `053736ed3ca26ad0.0.unit_3ca26ad0_2`，自身数字NLL变化 -0.182955；`053736ed3ca26ad0.0.unit_3ca26ad0_0` 变化 +0.077579。
- joint_context/800: 更新 `053736ed3ca26ad0.0.unit_3ca26ad0_0`，自身数字NLL变化 -0.027142；`053736ed3ca26ad0.0.asset_053736ed` 变化 +0.027667。
- mixed_context/400: 更新 `053736ed3ca26ad0.0.asset_053736ed`，自身数字NLL变化 -0.126004；`053736ed3ca26ad0.0.unit_3ca26ad0_0` 变化 +0.024271。
- mixed_context/800: 更新 `053736ed3ca26ad0.0.asset_053736ed`，自身数字NLL变化 -0.252855；`053736ed3ca26ad0.0.unit_3ca26ad0_1` 变化 +0.019252。

## 分组、学习轨迹及内容门槛

| 臂 | 暴露 | train编号率 | dev编号率 | train数字NLL | dev数字NLL |
|---|---:|---:|---:|---:|---:|
| sequential | 0 | 0.00% | 0.00% | 2.767957 | 2.662896 |
| sequential | 400 | 81.25% | 0.00% | 0.321056 | 7.050168 |
| sequential | 800 | 87.50% | 0.00% | 0.213237 | 11.471340 |
| joint_context | 0 | 0.00% | 0.00% | 2.767957 | 2.662896 |
| joint_context | 400 | 12.50% | 0.00% | 0.909607 | 5.539829 |
| joint_context | 800 | 87.50% | 0.00% | 0.207134 | 8.280446 |
| mixed_context | 0 | 0.00% | 0.00% | 2.767957 | 2.662896 |
| mixed_context | 400 | 25.00% | 0.00% | 0.539573 | 7.113565 |
| mixed_context | 800 | 81.25% | 0.00% | 0.102075 | 9.246504 |

| 臂 | 划分 | 世界 | 正确编号率 | 错配编号率 | 错配减正确数字NLL |
|---|---|---|---:|---:|---:|
| sequential | train | 053736ed3ca26ad0 | 87.50% | 87.50% | +0.154563 |
| sequential | train | 202bfd9735db6b7c | 87.50% | 87.50% | +0.192005 |
| sequential | dev | 11162e7e35a26dad | 0.00% | 0.00% | +0.311200 |
| sequential | dev | 76d75d7ae1859607 | 0.00% | 0.00% | +1.005345 |
| joint_context | train | 053736ed3ca26ad0 | 87.50% | 50.00% | +0.183972 |
| joint_context | train | 202bfd9735db6b7c | 87.50% | 37.50% | +0.524549 |
| joint_context | dev | 11162e7e35a26dad | 0.00% | 0.00% | +0.266919 |
| joint_context | dev | 76d75d7ae1859607 | 0.00% | 0.00% | +0.502237 |
| mixed_context | train | 053736ed3ca26ad0 | 87.50% | 25.00% | +0.342920 |
| mixed_context | train | 202bfd9735db6b7c | 75.00% | 37.50% | +0.582560 |
| mixed_context | dev | 11162e7e35a26dad | 0.00% | 0.00% | +0.257159 |
| mixed_context | dev | 76d75d7ae1859607 | 0.00% | 0.00% | +0.023658 |

延用上一轮内容门槛；不因本轮干扰事件存在而放宽。固定800暴露终点：

- sequential: train=未通过，dev=未通过；anchor的twin减正确数字NLL train=+0.028859、dev=-0.070557。
- joint_context: train=通过，dev=未通过；anchor的twin减正确数字NLL train=+0.021528、dev=+0.257972。
- mixed_context: train=通过，dev=未通过；anchor的twin减正确数字NLL train=+0.035016、dev=+0.155539。

## 资源、检查与审计

| 臂 | 平均优化step秒 | 优化步骤合计秒 | 训练峰值allocated GiB | 训练峰值reserved GiB |
|---|---:|---:|---:|---:|
| sequential | 0.302409 | 241.927 | 10.188380 | 13.210938 |
| joint_context | 0.441722 | 88.344 | 11.987185 | 13.230469 |
| mixed_context | 1.148944 | 229.789 | 16.266350 | 17.726562 |

训练step表不含评估/干预，实际GPU服务时间与上述wall还包含加载、同步、状态备份和hash。干预各分支及全context、评估每次写入/查询均记录耗时与allocated/reserved峰值。

新增只读 `scripts/summarize_cbf_fact_resources.py` 从全部已保存资源记录生成 `resources.json`，无新模型调用。它保存输入JSONL/summary及脚本hash，本地已与审计对齐。每臂干预共8个context-run（含40分支）：sequential总39.634秒、joint37.927秒、mixed39.108秒，峰值allocated分别12.851/12.849/12.848GiB。context时间已包含分支时间，不能重复相加。CUDA reserved可能保留此前评估/干预的allocator缓存，不等于训练活跃张量量。

仅测得优化步骤用时为sequential241.927秒、joint88.344秒、mixed229.789秒。joint复用写入因此减少了计算，不是严格FLOPs对齐下的速度优越性结论；不能外推正式大规模训练加速倍数。

服务器17项测试通过（2.069秒）；本地17项10通过、7因无torch跳过（1.025秒）。首次预检曾因CPU微型随机网络一步变化低于FP32 CE分辨率而失败，没有进入模型训练；只放大toy fixture后定向测试通过，正式实验配置保持不变。失败日志保留远程_preflight_failed_077e0a4目录。

独立审计通过：32行源数据不变、三臂每题50次预算、1200优化步与CE分解、实际FP32参数更新/optimizer计数、1224策略评分解码/前轮空/full-KV精确桥接、120干预的恢复记录/读出漂移/独立事件计数、summary重算、初始writer逐值相同、骨干/原始源权重hash。梯度余弦审计核验形状/范围/对称性；没有存储全部原始梯度向量，也没有重新做梯度forward，不能把它表述为独立重算全部梯度。

## 文件与配置

| 文件 | 实现 |
|---|---|
| tasks/cbf_fact_interference.py | schedules固定分组/暴露；batch_loss共享可微memory；run训练；probe_context、restore、equal_state可回滚干预；cosine_matrix梯度关系；describe_probes/summarize全矩阵与汇总 |
| tests/test_cbf_fact_interference.py | 预算与顺序、干扰定义、联合图与独立平均梯度、真实AdamW恢复 |
| scripts/run_cbf_fact_interference.sh | 双卡调度三臂、预检/失败标记/汇总/独立审计 |
| scripts/audit_cbf_fact_interference.py | 来源、参数、计数、生成、干预和汇总审计 |
| scripts/summarize_cbf_fact_resources.py | 已保存训练/评分/干预资源分开汇总、来源hash，无新模型调用 |
| FACT_INTERFERENCE_PLAN.md | 执行前目标/阈值/预算/归因限制 |
| 本目录与MODIFICATION_LOG/README/TRAINING_CORPUS_PLAN | 结果、配置哈希、审计、状态及TODO |

复用原生event_writer及旧评分，baseline/runtime和旧实验入口未改，无新增依赖。路径环境变量ROOT/SOURCE/MODEL/PYTHON；研究矩阵及阈值冻结。evaluation的step字段为问题暴露，真实optimizer步数在checkpoint/日志另存，不能混用。

## 运行方式

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_fact_interference_repeat bash scripts/run_cbf_fact_interference.sh
/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python -m scripts.audit_cbf_fact_interference --root /home/ctj/cbf_ttt_fact_interference_v1 --model /home/ctj/models/Qwen3-4B
/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python -m scripts.summarize_cbf_fact_resources --root /home/ctj/cbf_ttt_fact_interference_v1
```

复现必须新ROOT，拒绝覆盖。原输出/home/ctj/cbf_ttt_fact_interference_v1，每臂保存writer_0/400/800.pt、optimizer_400/800.pt、steps/evaluation/probes JSONL和training_manifest/complete。writer状态需配合原始Qwen3-4B，不是完整HF模型。原始文本/tokens/逐条输出/权重留服务器；Git归档代码、summary/design/audit和报告。

## 与研究构思的关系与限制

本轮属于阶段A writer学习诊断。训练事实间干扰不等同会话中历史快权重应遗忘，也不证明全局/last1/last2决策有效。附件多边界反事实标签、策略初始化和联合架构训练尚未开展。

只有2训练/2开发世界及1seed，事实对和同世界变体相关，不以分支数作为独立统计样本。开发世界和值/模板同时变化，且已多次观察。干预阈值和负余弦阈值是预设描述规则；没有p值或泛化显著性结论。不同臂更新次数和计算量不同，mixed控制仍不能完全消除顺序差异。

干预只测同context的4事实，未测跨context/跨twin的全部影响。生成在训练题上成功还可能含query/实体ID记忆，sequential错配仍14/16正说明需要负对照。下一步应使同一查询在不同会话中对应重新随机赋予的值，并扩大独立训练世界、分开同模板新世界与新模板验证，阻止依靠固定实体到答案映射。该后续尚未生成或训练，不以当前dev调整并宣称盲测。

已完成：三臂固定预算训练、三点评估、两点可逆干预与独立审计。未完成：更多独立世界/seed、新来源确认、自然背景、选择性遗忘/多边界标签/控制器/联合训练、六真实基准。没有自动延长训练到通过。
