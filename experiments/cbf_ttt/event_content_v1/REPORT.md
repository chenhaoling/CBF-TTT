# 阶段A第二轮：双臂少量世界内容训练结果

## 结论

**两个400步训练臂均完成并通过执行审计，但train/dev内容门槛均未通过。** CE在训练集完整答对4/16，content_pair答对7/16（+18.75百分点），两者开发集均0/16。成对方案的训练集平均数字NLL反而较高（0.7348 vs 0.5783），且anchor正确记忆不优于twin（twin减正确NLL为−0.02336）。因此只能确认小训练集出现部分完整答案拟合，不能确认所加成对目标改善了可靠的单事实绑定。

开发集数字NLL为CE7.4470/content_pair7.6653，均明显高于空记忆2.6629；更低的部分训练loss没有带来该开发条件下的泛化。开发世界、实体和值以及措辞模板同时不同，不能将失败单独归因某一种变化。没有进入遗忘控制器或多步联合训练，也未自动延长预算。

## 目标与实现假设

上一轮100步的生成内容记忆门槛未通过。本轮固定两个训练世界，比较普通CE与数字加权/单事实成对目标，检验能否先学会完整编号。只训练原生writer，未进入遗忘控制器或多步联合训练。

实际执行：hku-gpu2双RTX5090，每卡一臂；2026-10-10T14:20:09+08:00 至 2026-10-10T14:28:25+08:00，496秒（训练、三点评估、准备/汇总；后续审计另计）。采集提交 `3901eca60331d25b068ca8c81712d9793b60588a`。

两臂均从原始Qwen3-4B、seed301零conv初始化；7层conv/proj FP32，骨干冻结，BF16计算，AdamW lr1e-7/clip1/无weight decay。两臂不是上一轮100步续训。每臂400步、8个query对每轮洗牌50轮，每个问题50次暴露。

train排序前两个世界16问题/4上下文，dev原两个世界16问题/4上下文；复用完整4096写入块。两份twin记忆分别写入，答案读取使用fresh KV。只有anchor的目标值在twin间改变，其他问题不作为twin负标签。训练标签/数字mask不进入write/query prompt。

CE为完整答案+EOS均值；content_pair为0.75数字+0.125前缀+0.125EOS，并仅在anchor加0.5 softplus(0.2+正确数字NLL−twin数字NLL)。两臂样本、顺序、初始化、前向预算相同，实际资源另报。此处同时改变两项损失，不能分别归因。

## 固定400步结果

编号正确率采用本轮执行前登记的首个完整code_数字串与真值比较；完整EM仍保持旧定义。两者不能互相替换。NLL越低越好。

| 臂 | 划分 | 策略 | 编号正确 | 完整EM | 答案NLL | 数字NLL |
|---|---|---|---:|---:|---:|---:|
| ce | train | correct | 25.00% | 25.00% | 0.413278 | 0.578284 |
| ce | train | empty | 0.00% | 0.00% | 4.808934 | 2.767957 |
| ce | train | wrong | 12.50% | 12.50% | 0.693786 | 0.971029 |
| ce | train | full_kv | 87.50% | 0.00% | 0.075021 | 0.000015 |
| ce | dev | correct | 0.00% | 0.00% | 5.417072 | 7.447014 |
| ce | dev | empty | 0.00% | 0.00% | 4.277348 | 2.662896 |
| ce | dev | wrong | 0.00% | 0.00% | 5.557036 | 7.645725 |
| ce | dev | full_kv | 75.00% | 0.00% | 0.105974 | 0.000111 |
| content_pair | train | correct | 43.75% | 43.75% | 0.525104 | 0.734783 |
| content_pair | train | empty | 0.00% | 0.00% | 4.808934 | 2.767957 |
| content_pair | train | wrong | 18.75% | 18.75% | 0.929160 | 1.285781 |
| content_pair | train | full_kv | 87.50% | 0.00% | 0.075021 | 0.000015 |
| content_pair | dev | correct | 0.00% | 0.00% | 5.519831 | 7.665306 |
| content_pair | dev | empty | 0.00% | 0.00% | 4.277348 | 2.662896 |
| content_pair | dev | wrong | 0.00% | 0.00% | 5.712723 | 7.950080 |
| content_pair | dev | full_kv | 75.00% | 0.00% | 0.105974 | 0.000111 |

## 分组与twin

| 臂 | 划分 | 世界 | 正确编号率 | 错配编号率 | 错配减正确数字NLL |
|---|---|---|---:|---:|---:|
| ce | train | 053736ed3ca26ad0 | 25.00% | 0.00% | +0.600230 |
| ce | train | 202bfd9735db6b7c | 25.00% | 25.00% | +0.185260 |
| ce | dev | 11162e7e35a26dad | 0.00% | 0.00% | +0.182175 |
| ce | dev | 76d75d7ae1859607 | 0.00% | 0.00% | +0.215248 |
| content_pair | train | 053736ed3ca26ad0 | 37.50% | 37.50% | -0.051161 |
| content_pair | train | 202bfd9735db6b7c | 50.00% | 0.00% | +1.153158 |
| content_pair | dev | 11162e7e35a26dad | 0.00% | 0.00% | -0.039456 |
| content_pair | dev | 76d75d7ae1859607 | 0.00% | 0.00% | +0.609005 |

anchor的twin减正确数字NLL（正数表示正确记忆更好）：

- ce/train: +0.029201
- ce/dev: -0.009225
- content_pair/train: -0.023363
- content_pair/dev: -0.093651

## 学习轨迹与门槛

| 臂 | step | train编号率 | dev编号率 | train数字NLL | dev数字NLL |
|---|---:|---:|---:|---:|---:|
| ce | 0 | 0.00% | 0.00% | 2.767957 | 2.662896 |
| ce | 100 | 0.00% | 0.00% | 2.191298 | 4.788439 |
| ce | 400 | 25.00% | 0.00% | 0.578284 | 7.447014 |
| content_pair | 0 | 0.00% | 0.00% | 2.767957 | 2.662896 |
| content_pair | 100 | 0.00% | 0.00% | 1.557815 | 4.047636 |
| content_pair | 400 | 43.75% | 0.00% | 0.734783 | 7.665306 |

固定400步内容门槛：

- ce: train=未通过，dev=未通过。
- content_pair: train=未通过，dev=未通过。

门槛全文在EVENT_CONTENT_TRAINING_PLAN.md/design.json。训练需编号≥75%、比空/错配≥25百分点、各世界≥50%和anchor数字NLL优势；dev需编号≥25%、比空/错配≥12.5百分点、两个世界数字NLL增益≥.05和anchor优势。未按最佳中间点挑选结果，未追加预算。

## 资源和审计

| 臂 | 平均训练step秒 | 峰值allocated GiB | 峰值reserved GiB |
|---|---:|---:|---:|
| ce | 0.700420 | 13.578647 | 13.902344 |
| content_pair | 0.686629 | 13.578647 | 13.902344 |

服务器13测试全通过（1.895秒），本地8通过5无torch跳过（1.037秒）。独立审计通过：32行数据/数字mask/twin重建、800优化步与每题50次暴露、目标分解、实际FP32参数更新及optimizer状态、816策略查询的生成解码和指标、summary重算、两臂初始writer逐张量相同、冻结骨干/源文件hash。上一轮dev空记忆/full-KV原指标完全复现。

逐步训练/写入/查询均保存时间及allocated/reserved峰值。评估另有数字NLL前向，耗时不能与上轮只有原指标的score直接混比；训练每步2条query，也不能按步数与上轮1条query比较计算效率。

每臂训练写入800个4096-token块，共3,276,800写入token暴露，来自仅4个重复上下文；不能称为同量独立训练语料。两臂checkpoint初始文件SHA不同是因为manifest包含arm名，writer张量已逐值验证相同。

## 修改文件、模块与附件对应

| 文件 | 本轮内容 |
|---|---|
| tasks/train_cbf_event_content.py | select_rows/split_answer准备两世界和数字mask；training_order固定次数；regions/objective可微成对loss；run/evaluate/extra_score训练与新指标；summarize/gates固定终点核验 |
| scripts/run_cbf_event_content.sh | 测试、prepare、GPU0 CE/GPU1 content_pair并行、完成汇总与独立审计 |
| scripts/audit_cbf_event_content.py | 无新forward的来源、目标、参数、结果审计和旧dev桥接 |
| tests/test_cbf_event_content.py | 数据隔离/次数/数字分词、对比梯度方向/假负例屏蔽、真实双memory原生梯度 |
| EVENT_CONTENT_TRAINING_PLAN.md | 执行前实验矩阵、预算、loss、门槛与停止规则 |
| 本目录及MODIFICATION_LOG/README/TRAINING_CORPUS_PLAN | 实际状态、summary/design/audit、结果与TODO |

沿用cbf_ttt/event_writer.py及已有runtime，没有改baseline或旧实验行为，无新增依赖。路径配置ROOT/SOURCE/MODEL/PYTHON；研究超参数冻结在计划和代码。

附件的全局+last1/last2、随机1–3边界、策略初始化后联合训练仍属于B/C，本轮验证其前置writer可训练性和内容学习，不能用单块结果声称选择性遗忘有效。

## 命令与工件

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
# 复现必须使用新的ROOT，拒绝覆盖
ROOT=/home/ctj/cbf_ttt_event_content_repeat bash scripts/run_cbf_event_content.sh
# 只读复核现有工件；审计JSON会重新生成
/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python -m scripts.audit_cbf_event_content --root /home/ctj/cbf_ttt_event_content_v1 --model /home/ctj/models/Qwen3-4B
```

服务器ROOT下ce/与content_pair/各保存writer_0/100/400.pt、optimizer_400.pt、steps.jsonl、evaluation_*.jsonl、training_manifest.json和complete.json。writer为独立状态，须结合原始4B模型及相同配置加载，不是完整HuggingFace模型。原文/tokens/逐条输出/权重留远程，Git仅汇总与代码。

## 局限和后续

只有2个训练世界、2个开发世界和1个seed；同组twin/多个问题相关，不能算独立样本。开发集已多次用于诊断，不是盲测。400步未达到门槛也不能证明任意训练预算或结构均无效；纯合成背景/随机编号未代表自然基准。

本轮已完成：两臂训练、预登记三点评估及独立审计。未完成：自然数据接入、新世界确认、多块retain、反事实动作标签、遗忘控制器/联合训练、六个真实基准。本轮结果不改原100步的负结论。后续方向按上述train/dev门槛决定，尚无自动追加实验。

下一步建议仍留在阶段A：对同一上下文的多个事实共同监督，比较逐query更新与同context多query梯度汇总，并显式检查双生anchor能否随唯一值变化切换答案。该建议针对不同事实间可能存在的优化干扰；当前结果尚未证明干扰就是原因。先要求小训练集稳定达到内容门槛，再扩大独立世界与模板，避免直接扩规模或进入控制器。后续需独立固定预算/对照，不将本轮400步延长到通过。
