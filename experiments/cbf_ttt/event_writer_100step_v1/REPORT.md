# 事件语料原生 writer：首次100步训练

## 结论

训练已实际完成，**固定100步的有用记忆门槛未通过**。答案loss能够反向更新原生conv/proj，骨干没有变化；但完整生成答案仍全部错误，不能据loss下降宣称已学会内容记忆，更不能据本轮判断遗忘策略有效或无效。

这是新构造语料的阶段A单块writer热身，未训练控制器，未进行全局/局部遗忘、多边界反事实标签或联合训练。初始化为原始Qwen3-4B，与之前全参数1B训练得到的骨干不同。

## 执行与数据

- hku-gpu2，GPU0训练、GPU1异步评估；2×RTX5090。采集代码 `3a0457e1056e4b7bfa92099392bc2b71c26c3c54`，审计代码见 `execution_audit.json` 的script SHA。
- 2026-10-10 11:01:41–11:03:58 +08:00，**137秒**，包括服务器测试、分词装箱、模型加载、优化与评估；随后独立审计另计。不是完整研究训练的耗时估计。
- 原始模型 `/home/ctj/models/Qwen3-4B`；输出 `/home/ctj/cbf_ttt_event_writer_100step_v1`。
- 事件原型warmup：train64问题/8世界/16上下文，dev16问题/2世界/4上下文。同世界的4问题与2个twin相关，不能视为独立样本。test未读取分词、未模型评分。
- 每条4096-token写入块：唯一合成维修背景前缀+完整事实后缀。不是FineWeb/LongCrawl自然文本；本轮没有新下载自然数据。
- 原始骨干冻结；7层原生conv/proj共14张量、45,964,800参数为FP32，BF16计算。conv零初始化；AdamW lr=1e-7、weight_decay=0、clip=1、seed301、固定100步，每步1问题。
- 丢弃写入KV后，以fresh KV和快权重回答；监督全部答案token及一个EOS。报告NLL不含EOS；自由生成最多16新token，遇EOS停止，strip后完整字符串EM。无候选限制，无结束token屏蔽。

## 结果

全部数字来自归档 `summary.json`。NLL越低越好，EM为完整字符串精确匹配。

| step | 正确记忆NLL | 空记忆NLL | 错配记忆NLL | 完整KV阅读NLL | 正确记忆EM |
|---:|---:|---:|---:|---:|---:|
| 0 | 4.277348 | 4.277348 | 4.277348 | 0.105974 | 0/16 |
| 25 | 2.729940 | 4.277348 | 2.748358 | 0.105974 | 0/16 |
| 50 | 3.009749 | 4.277348 | 3.059188 | 0.105974 | 0/16 |
| 100 | 2.053219 | 4.277348 | 2.235196 | 0.105974 | 0/16 |

所有四个主策略、四个checkpoint的严格EM均为0。固定终点100步：相对空记忆NLL改善2.224130，相对错配改善0.181977，但两个世界并不一致：

| dev世界 | 正确NLL | 错配NLL | 错配减正确 |
|---|---:|---:|---:|
| 11162e7e35a26dad | 1.964677 | 2.380454 | +0.415777 |
| 76d75d7ae1859607 | 2.141760 | 2.089937 | −0.051823 |

预登记要求：相对空/错配均改善NLL≥0.05且EM≥12.5百分点，两个dev世界各有正NLL增益。EM与分组条件均未通过，`passed_memory_gate=false`。不选中间最佳点，不延长训练直到通过。

4个anchor问题的正确减twin NLL为+0.012688，即正确记忆平均略差；没有显示可靠的单事实内容区分能力。最终预先指定的两个train世界16问题probe：NLL2.702084、EM0/16，也没有显示训练样本完整回忆成功。在线前10/后10步答案NLL为4.582447/1.937432；这些来自不同采样问题，不是配对同题改善。

100步正确记忆首token命中16/16，但答案共享 `code_` 前缀，首token不能代表编号正确。首token `<|im_end|>`平均概率约1.03e-7，本轮不能直接归因为之前1B骨干的首token立即终止现象。

### 生成文本检查的解释边界

只读检查已保存的100步输出，发现正确记忆出现 `code_` 就终止和错误数字串；完整KV阅读出现正确编号后继续解释、或先解释而耗尽16-token预算。因此完整KV的低NLL与严格EM0并不等于完全读不懂事实。正确记忆的失败也不能全部归于答案后多余说明：已见不完整/错误编号。

新增 `scripts/summarize_cbf_event_generation.py` 可从已有输出汇总首行匹配、首个数字编号匹配、裸 `code_` 和耗尽生成预算的次数。该分析是**事后描述**，不改变严格EM/门槛，不做新forward，不访问test。具体执行和计数见本目录后续诊断工件；若未归档工件，则仅以上逐条检查成立。

## 检查与资源

服务器9项单元测试全通过（1.649秒），包括真实原生delta/fresh-cache路径一致、梯度连接和答案位置；本地9项中6通过、3因无torch跳过，服务器已补全。

独立审计已通过：80条pack事实/答案边界重算、100步固定采样序列及答案/EOS loss分解、14个保存writer张量实际变化、FP32 optimizer状态、272次dev策略查询、16次train probe数量、原始模型文件重新hash及summary完全重算。审计没有重新训练/重新评分。

- 训练平均单步0.299468秒，峰值allocated **10.186427 GiB**、reserved **10.792969 GiB**。
- 100步checkpoint单上下文写入约0.218–0.220秒；评估会同时保留4份memory，不能与单memory部署峰值混同。
- 每步训练、每次写入/查询的时间与allocated/reserved峰值均记录在远程JSONL/summary，不能用137秒直接外推多块联合训练或反事实构造成本。
- 骨干hash/version不变；初始零conv三种fresh-KV策略相同；empty/full-KV各checkpoint指标完全相同。

## 文件与实现对应

| 文件 | 作用 |
|---|---|
| `tasks/pack_cbf_event_warmup.py` | train/dev装箱，完整事实、答案分词边界、错配/twin映射、hash，不读取test |
| `cbf_ttt/event_writer.py` | 原始模型严格加载、FP32 writer、可微单块快权重、fresh-KV答案logit/loss |
| `tasks/train_cbf_event_writer.py` | 固定100步训练、原子checkpoint、四点评估、生成/分组指标、终点门槛 |
| `scripts/run_cbf_event_writer.sh` | 测试、装箱、双GPU进程、失败标记与汇总；拒绝覆盖 |
| `tests/test_cbf_event_writer.py` | 装箱隔离、原生更新一致、梯度、监督位置 |
| `scripts/audit_cbf_event_writer.py` | 保存结果/参数/数据/骨干的独立完成审计 |
| `scripts/summarize_cbf_event_generation.py` | 无新模型调用的事后生成格式/内容统计 |
| `EVENT_WRITER_TRAINING_PLAN.md` | 执行前参数、门槛与停止规则 |
| 本目录summary/audit/report/status | 汇总证据与执行状态；原文、token、权重不上传GitHub |

所有功能为独立入口，未修改官方baseline模型/runtime/配置，无新依赖。路径环境变量ROOT/SOURCE/PYTHON/MODEL控制运行位置；本实验100步/学习率/矩阵在代码和计划中固定，不提供无记录的隐式续跑。

附件对应：本轮建立了独立策略训练及未来联合训练所需的可微原生writer基础。附件的“全局+last1/last2遗忘”“随机1–3边界”“最终联合训练”仍待阶段B/C，不能把本轮单块writer热身作为这三个主张的效果证据。

## 运行方式

新目录复现（两张卡），不要覆盖当前结果：

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_event_writer_100step_repeat bash scripts/run_cbf_event_writer.sh
```

对现有结果只读审计与统计：

```bash
/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python -m scripts.audit_cbf_event_writer \
  --root /home/ctj/cbf_ttt_event_writer_100step_v1 \
  --output /tmp/event_writer_audit_repeat.json
/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python -m scripts.summarize_cbf_event_generation \
  --root /home/ctj/cbf_ttt_event_writer_100step_v1 \
  --output /tmp/event_writer_generation_repeat.json
```

最终模型为 `writer_100.pt`，是writer状态，不是完整HuggingFace模型；须结合原始Qwen3-4B、相同7层配置及加载接口使用。`optimizer_100.pt`供状态核验；当前入口没有自动resume。

## 后续TODO与风险

1. 保留本轮负结果。下一轮应固定预算验证少量训练世界的内容记忆，先要求训练集完整编号回忆，再测新世界；加入只改变值的成对训练信号，减少共享前缀/EOS对目标的主导。此项是建议，尚未训练。
2. 在新方案执行前分开登记“事实编号正确”和“输出格式遵循”，并记录数字token的NLL。保留旧严格EM，不回改本轮门槛。生成截断和EOS监督的影响需独立对照，不能只删EOS让成绩变好。
3. 仅2个dev世界，不能作统计显著性或泛化结论；100步不足也不能直接排除。8个训练世界、单块4096、纯合成背景、任意多token编号均限制解释。
4. 获得可重复的正确vs错配/twin内容信号后，才接多块retain、全局/last1/last2反事实动作标签、策略初始化与联合训练。自然数据、盲测及真实六基准仍未完成。
