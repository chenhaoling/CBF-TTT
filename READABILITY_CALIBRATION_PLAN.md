# 可读性校准 v1：执行前登记（2026-10-09）

## 问题与范围

上一轮 Q 使用完整 KV、不写入 M，四场景目标准确率31.25%/100%/56.25%/62.5%，未满足每类75%。本轮只诊断读取协议，不训练、不评估记忆价值、不修改旧门槛。只使用已观察的8个dev来源组及stable双生版本（16会话）；旧confirm128会话不评分。

沿用Qwen3-4B/1B checkpoint、BF16、TF32关闭、两张5090，M始终为空、正常因果KV。新的场景从相同dev原始文档重新取背景，复核原文SHA；不是新独立泛化集。

## 固定对照

完整交叉3种时距×2种键负载×2种回答格式：

- short：2×4096 token，目标事实在第2块末尾。
- long_far：6×4096 token，同一目标事实在第2块末尾；前两块与short逐token相同。
- long_near：6×4096 token，目标移至第6块末尾；对比long_far区分长上下文与较早位置。位置/后续干扰随移动一起改变，不能声称完全隔离单一生理机制。
- single：仅插入目标键事实；dual：同块额外插入旧锚点键事实，锚点在前、目标在后。其余块只读自然背景，无额外人工记录。总token数固定，目标末尾位置相同；dual用锚点记录替代等长自然背景，不是删除所有可能的自然干扰。
- qa：复用原问题 `Question: What is the latest access color recorded for KEY in this session?\nAnswer:`。
- cloze：`Record: The access color for KEY is`。标签仍为相同带前导空格的单token颜色，不能把补全文本成绩称为自然对话能力。

辅助控制：long_far/dual增加target_first，将目标和锚点的两行记录交换；其余内容、长度、键和值不变。在两种格式下均评分，检验之前锚点强于目标是否受记录槽位影响。不能把交换行的收益全归于某一单独因素。

原路径桥接：在16条原dev stable的原始六块输入上重跑原两个query，逐条复现已完成Q的NLL、分类与greedy，NLL误差要求≤1e-5。未通过先修复，不解释新条件。

主矩阵96条上下文轨迹，两个query格式共享同一只读前缀、独立克隆；辅助16条、桥接16条，共128条轨迹、384次查询。没有oracle动作搜索、写入或梯度更新。

## 构造与验证

原始背景严格按上一轮design的parquet/row_index读取、SHA一致。只重构dev0–7，拒绝其他组。各chunk足4096，不能循环重复自然filler。重新构造时只截取每个原背景chunk前缀填满记录之外的空间；明确与原Q记录负载不同，原桥接单列。

双生版本除目标颜色token外相同；所有条件内八类标签平衡，键值不进入问题。short严格等于long_far前两块；target_first仅交换两条事实行。记录位置、实际长度、输入/来源/模型/代码哈希。事实解码检查与单元测试先通过。

## 指标与解释

按条件/来源域/源组报告目标八选一准确率、全词表首token准确率、NLL；dual同时报告锚点。不同query共享前缀不会影响主缓存，完整父缓存字节哈希不变，权重版本与全模型哈希不变。逐轨迹/查询记录耗时与allocated/reserved。

比较long_far−short、long_near−long_far、dual−single、cloze−qa以及交换槽位的配对差；NLL差只作辅助，不跨不同格式把概率差视为唯一证据。按8来源组汇总，不把版本/格式/候选当独立样本，不做小样本显著性断言。

下一轮完整任务的可用格式只从**long_far/dual/默认顺序**选择：target和anchor整体与两个来源域都须≥75%。两格式都通过时按两问题/两域最差准确率最大选择；相同则保留qa。这是开发集校准选择，不是独立验证。single、short、target_first只作诊断，不能用其较容易的成绩代替原长程双键问题。

若没有格式通过，则完整协议仍不可用，先报告失败原因线索，不进入V/F。若通过，下一步在新来源上构建全部四场景、再验证Q并重开V/F；本轮不动旧confirm，也不声称其他三类场景已经通过。

## 代码与执行

新增 `tasks/cbf_readability.py`（build/collect/summarize）、`tests/test_cbf_readability.py`、`scripts/run_cbf_readability.sh`。复用已有模型加载、CBFSession、score_query、计时与状态哈希，不修改baseline或旧采集入口，无新增依赖。

ROOT必须新建；测试→build→双卡collect→完整汇总/桥接审计→完成标记。初始源码、时间、资源和错误日志保留；失败不覆盖。原始背景/token/逐条结果只保留远程，公开设计元数据和聚合。总规模较前轮Q增加一倍轨迹，但不需双生缓存CPU存放；墙钟在真实运行中记录，不能根据旧GPU内核时间承诺ETA。

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_readability_20261009 bash scripts/run_cbf_readability.sh
```

此文件为执行前设计，结果尚未产生。
