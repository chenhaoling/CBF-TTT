# 无颜色重合的四场景读出校准 v1（2026-10-09，执行前登记）

## 目的

前几轮dev输入干预显示无关记录会干扰读取、位置效应不单调，且部分正确颜色与干扰颜色重合。本轮保留四类任务和干扰数量，只移除人工无关键记录与目标/锚点真值的颜色重合，检验当前4B checkpoint能否可靠读取。完整KV、M=0，无训练。不是遗忘控制器效果实验。

## 构造与范围

沿用已观察dev0–7（FineWeb-Edu/LongCrawl64各4组），四场景×两个版本共64条。每组固定排除：目标的两个双生值、锚点值、纠正任务旧值，共4种不同颜色；余下4色为干扰palette，在每条无关键记录上按 `(2*(chunk-1)+slot+group)%4` 循环分配。palette对所有场景和双生相同，不按某个版本或评分选择。对合法target/anchor键的任何记录（包括correction旧值）保持原值。

重编码每块原suffix；长度和起始offset必须相同，只有改变的颜色token允许不同，差异数须恰等于变色记录数。背景、设备键、记录数、顺序、事实位置、总6×4096长度保持不变。原始文本也可能自然出现颜色词，但无关键的人工绑定不含四个受保护颜色。干扰只有4色且重复，这是实验设计限制；不能称所有自然文本的颜色都不重合，也不能声称此任务等价自然问答。

原QA及一个固定的binding格式同时评分，共享只读前缀、查询独立克隆。binding模板为：

```text
Read only the session records for {key}. Ignore records for other devices. If this device has multiple records, use its last record.
Question: What is its access color? Answer with exactly one color word.
Answer:
```

不提供正确颜色、候选答案顺序或未来标签。候选仍为原8个带空格的单token颜色；八选一和全词表首token分别报告。保持先前near失败的cloze不进入本轮；不事后加第三个模板。

原输入桥接使用dev组0和2的四场景双生：16条、32原QA评分覆盖两域四场景。全批80轨迹、288查询（clean64×4、bridge16×2），双卡各40条。确认集不评分，所有原来源哈希校验，文件拒绝覆盖。

## 门槛与后续分支

每个格式单独计算4场景×2域×2问题共16个八选一单元，每格8会话/4组，全部≥75%才合格；整体各场景也报告。格式选择按这16格最低准确率最大，平局保留原qa；不得按场景、来源或问题选择不同格式。首token指标不用于暗中替换这个分类门槛。

- 无格式合格：本版本Q失败，停止进入新来源和M写入，不删难例、不降低门槛、不改模板追加试跑。结论是当前任务/读出能力不足；下一项应考虑原始Qwen3-4B与1B checkpoint的能力对照，而非继续无界改模板。
- 有格式合格：冻结全局格式与构造版本，使用已修复来源排除器从原始parquet选取新来源（全1B/旧数据扫描保留），进行同样的四场景新来源Q；新来源不能仅换设备名或颜色。该分支才能进入后续记忆内容检查。
- 新来源Q也通过后：按原memory-value的self/zero/twin、none/retain/half/clear、full_kv/memory_only协议进行资源smoke和V，遵守原资源与收益门槛；确认集与F按原预登记条件开启，不能因校准成功直接宣布遗忘有效。

本轮先执行固定80条dev矩阵。桥接要求NLL误差≤1e-5、预测一致、模型/权重一致；所有行和查询齐全，原始logits重算、父缓存/模型不变后才生成完成标记。每个轨迹/查询记录耗时、allocated/reserved，失败留日志。

## 实现与复现

新增 `tasks/cbf_disjoint_readout.py`（纯颜色变换、四场景校验、build/collect/summarize与门槛），必要单元测试、双GPU脚本与完成审计。复用readability多格式collector，仅新增可选validator/bridge_cells/identity_fields，旧默认行为保持不变。baseline模型、训练配置和runtime不改，无新依赖。

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_disjoint_readout_20261009 bash scripts/run_cbf_disjoint_readout.sh
```

SOURCE为原memory_value目录；ROOT/PYTHON/MODEL/TOKENIZER可配。预估约11分钟仅参考既有吞吐，实测时间另记。原始文本/token/逐条评分留远程，公开设计/聚合/审计/报告。

## 完成记录（2026-10-09）

采集提交45525be，18:48:08–18:59:10 +08:00，共662秒，80上下文/288查询。独立审计重新编码构造、检查624条干扰/144条保护记录、重算summary通过；32桥接查询误差0。峰值allocated/reserved为14.779/18.893 GiB。

原QA四场景目标为0/100/25/18.75%，binding为25/100/25/43.75%；原QA失败14/16格，binding失败12/16格，最弱格均0%。无格式合格，按预登记停止于 `stopped_by_disjoint_Q`，未进入新来源或M写入/V/F，confirm未评分。不调整门槛或追加模板。

完整分域、锚点、NLL、首token、资源与审计见 [完成报告](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/disjoint_readout/REPORT.md)。本轮是M=0读取诊断，不能判定遗忘机制无效。下一项建议原始Qwen3-4B与最终1B checkpoint在相同冻结任务上的能力对照；尚未启动。
