# 开发集可读性校准：完成报告（2026-10-09）

## 结论

本轮完成128条上下文轨迹、384次查询，全部来源为已观察的8个dev组。原Q的16条stable会话、32个评分精确复现（NLL最大误差0、分类及greedy预测一致）。结果可复现，但**尚未证明快速记忆内容价值或遗忘机制有效**：本轮M始终为空，仅使用完整KV。

重建自然背景、只保留目标与锚点两条人工记录后，QA在所有布局的目标八选一准确率均为100%。原始stable桥接仍为31.25%。这提供了任务中人工记录负载/布局影响读取的线索；重建同时恢复了自然背景，不能独立归因于干扰记录数量，更不能说原四场景问题已经解决。

填空cloze呈现强烈的位置依赖：长程远事实双键目标/锚点均100%，但短程和长程近事实双键目标均0%。预登记只以远事实默认顺序选择格式，确实选中cloze；这一选择记录原样保留。**不能将该窄条件选择推广为完整任务推荐**。QA的跨位置成绩更稳定，但这是新观察后的下一轮候选判断，须重新登记并用新来源检验。

## 设计与范围

- 模型：Qwen3-4B，1B预训练最终checkpoint；两张RTX5090，BF16、TF32关闭。
- 背景：FineWeb-Edu和LongCrawl64各4个来源组，16条stable双生版本；同源版本/格式/布局不视作独立样本。
- short：2×4096 token、事实在第2块；long_far：6×4096、事实在第2块；long_near：6×4096、事实在第6块。
- single只含目标记录；dual另含锚点记录；默认目标在后，辅助target_first交换两行。单/双键比较会替换少量自然背景，不能声称完全隔离所有背景差异。
- QA沿用原提问，cloze补全记录前缀。两者使用相同八类、带前导空格的单token颜色答案。事实为人工绑定，不是自然任务基准成绩。
- 新旧原始文本、token数组、逐条评分仅在远程保存；本目录公开设计哈希、聚合结果与报告。
- 旧confirm128会话未评分；其他三类场景不在本轮校准范围内。

## 八选一结果

每行16个双生会话，来自8个来源组；百分比为正确数/16。每个来源域为8个会话、4个来源组。此处不作小样本显著性或独立泛化断言。

| 条件 | QA目标 | cloze目标 | QA锚点 | cloze锚点 |
|---|---:|---:|---:|---:|
| bridge | 31.25% | — | 87.50% | — |
| short/single/target_last | 100.00% | 43.75% | — | — |
| short/dual/target_last | 100.00% | 0.00% | 100.00% | 56.25% |
| long_far/single/target_last | 100.00% | 100.00% | — | — |
| long_far/dual/target_last | 100.00% | 100.00% | 87.50% | 100.00% |
| long_near/single/target_last | 100.00% | 56.25% | — | — |
| long_near/dual/target_last | 100.00% | 0.00% | 100.00% | 56.25% |
| long_far/dual/target_first | 100.00% | 100.00% | 100.00% | 100.00% |

FineWeb/LongCrawl的桥接目标分别为37.5%/25%；重建后的QA目标各条件、两个来源域均100%。远事实默认双键QA锚点两域均87.5%，cloze目标与锚点两域均100%。近事实双键cloze目标两域均0%，因此失败并非仅来自单一背景域。

槽位交换后QA目标仍100%，锚点87.5%→100%；只涉及2个会话，且数据已观察，不能据此建立普遍槽位规律。完整NLL、全词表首token准确率、来源域/来源组分层及五组配对差值见 `summary.json`。

### 八选一与全词表首token不能混淆

| 默认双键条件 | QA目标首token | cloze目标首token | QA目标NLL | cloze目标NLL |
|---|---:|---:|---:|---:|
| bridge | 6.25% | — | 3.416041 | — |
| short/dual/target_last | 75.00% | 0.00% | 0.952955 | 1.921201 |
| long_far/dual/target_last | 12.50% | 100.00% | 2.089495 | 0.116136 |
| long_near/dual/target_last | 87.50% | 0.00% | 0.966021 | 1.926977 |

QA远事实八选一100%，但首token只有12.5%；候选分类并不等于自由生成成功。首token指标也不等于多token生成的语义准确率。cloze远事实首token100%仍不能抵消近事实0%。不同问法的NLL只辅助解释，不当作遗忘收益。

## 审计与资源

- 采集提交：`c07d98cfeccb7bb639ca32c7797888bced8a9e0e`。
- 2026-10-09 14:45:09至14:58:49 +08:00，完整流水线820秒（13分40秒）。两分片64/64条，各795.830/774.890秒。
- 8项单元测试通过（2.408秒）；最终独立审计从原始行重算summary，与归档完全一致。
- dev来源身份、原始数据/设计哈希、128个上下文哈希、分片归属及32个桥接评分检查通过。父缓存完整哈希和骨干权重均未变；无失败标记，无confirm结果文件。
- 峰值allocated **14.770 GiB**，reserved **18.822 GiB**；无OOM。
- 重建短程rollout均值约0.695秒、单查询约0.045秒；长程约4.115秒、单查询约0.103秒。逐条原始记录包含时间及两种峰值显存；总体墙钟还包括加载、构造、哈希与其他开销，不能简单累加GPU计时替代墙钟。
- 详细时间/来源/文件哈希见 `execution_audit.json`；审计脚本为 `scripts/audit_cbf_readability.py`，审计于采集完成后执行，不改变实验代码与结果。

## 与附件三条建议的关系

本轮是为记忆价值和遗忘验证修复可读性前提，未新增图片第一条的全局/局部遗忘效应证据，也未重做第二条的随机多点遗忘；此前结果仍以各自报告为准。第三条端到端联合训练尚未开展。不能将当前100%读取成绩替代遗忘收益或联合训练收益。

## 修改与复现

新增 `tasks/cbf_readability.py`：build/collect/summarize；新增 `tests/test_cbf_readability.py` 与 `scripts/run_cbf_readability.sh`；新增完成审计 `scripts/audit_cbf_readability.py`。配置通过入口环境变量ROOT、SOURCE、PYTHON、MODEL、TOKENIZER传入，collector通过CLI选择分片。模型、baseline runtime、原训练配置未修改，无新增依赖。

在hku-gpu2复现（ROOT必须是未存在的新目录，约14分钟仅作本次规模参考）：

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_readability_replay_20261009 \
  bash scripts/run_cbf_readability.sh
```

复核本次已有结果（只读数据，写出新的审计JSON）：

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
  -m scripts.audit_cbf_readability \
  --root /home/ctj/cbf_ttt_readability_20261009 \
  --output /home/ctj/cbf_ttt_readability_20261009/execution_audit_recheck.json
```

原始数据依赖上一轮SOURCE目录及设计中指定的parquet原文件；仅从GitHub克隆聚合归档不能重新推理，需按数据文档下载并保持来源revision/校验一致。

## 后续顺序与未完成项

1. 在已观察dev中固定QA格式，进行原始背景上的人工记录负载/位置对照；将“移除其他键记录”和“改变背景”分离，先定位原stable失败。额外检查cloze失败时的答案token与预测分布，避免将格式效应当记忆机制。
2. 登记覆盖四场景、目标与锚点、两来源域的统一Q门槛。当前cloze的远距选型不能直接推进含近事实的完整V/F。
3. 构造新来源前补齐来源排除器对新 `design.json` / `chunks` 数据格式的识别，并测试实际来源ID去重。当前构造器只扫描旧 `*.meta.json`，旧文本扫描识别 `ids/context_ids/prefix`，下一次直接复用可能遗漏本轮/上一轮来源。当前校准是有意复用dev，不受此问题影响。
4. 新来源四场景通过Q之后再进入V：self/zero/twin内容干预；V成立后检查全局与局部遗忘F，再考虑联合训练。旧confirm继续保留，不能在此轮选择后称作完全未参与设计的新任务验证集。

本轮已完成校准与归档。以上是后续工作，尚未启动新V/F、训练控制器或端到端联合训练；无自然基准评测结果。
