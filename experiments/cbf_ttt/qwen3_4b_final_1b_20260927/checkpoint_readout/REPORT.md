# 原始Qwen3-4B与最终1B checkpoint读取对照：完成报告

## 结论

固定任务对照及独立审计完成。两模型均未通过全16格的读取门槛，未选定通过门槛的格式。原始模型本身存在旧事实读取弱点，同时最终checkpoint在明确绑定指令下出现额外的准确率下降；两种问题并存。这个结果不能直接证明下降由训练算法而非配置、优化或导出细节造成，也不能据此判定遗忘机制有效或无效。本轮完整KV、M=0，没有训练。

## 固定任务结果

以下均为clean64的八选一正确率，每格16会话/8来源组；双生和格式不作为独立来源重复。

| 场景 | 原始QA目标 | 最终QA目标 | 原始binding目标 | 最终binding目标 | 原始binding锚点 | 最终binding锚点 |
|---|---:|---:|---:|---:|---:|---:|
| stable | 6.25% | 0.00% | 50.00% | 25.00% | 100.00% | 62.50% |
| correction | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 68.75% |
| recent1_distractor | 31.25% | 25.00% | 68.75% | 25.00% | 100.00% | 56.25% |
| recent2_distractor | 62.50% | 18.75% | 75.00% | 43.75% | 100.00% | 50.00% |

### 门槛

| 模型 | 格式 | 失败格数/16 | 最弱格准确率 |
|---|---|---:|---:|
| original_repo | qa | 9/16 | 0.00% |
| original_repo | binding | 3/16 | 50.00% |
| final_repo | qa | 14/16 | 0.00% |
| final_repo | binding | 12/16 | 0.00% |

### binding的配对变化（final−original）

| 问题 | 原始均值 | 最终均值 | 差值（百分点） |
|---|---:|---:|---:|
| target | 73.44% | 48.44% | -25.00 |
| anchor | 100.00% | 59.38% | -40.62 |

| 来源组 | binding全部目标/锚点平均差（百分点） |
|---:|---:|
| 0 | -56.25 |
| 1 | -43.75 |
| 2 | -6.25 |
| 3 | +0.00 |
| 4 | -75.00 |
| 5 | -68.75 |
| 6 | -12.50 |
| 7 | +0.00 |

绑定指令全部目标/锚点的组内均值在6/8来源组下降，2/8持平，无组改善。这是复用开发来源上的描述性配对结果，未报告显著性或推广置信区间。

### 分域、NLL和首token

| 域 | 场景 | 问题 | 格式 | 原始准确率 | 最终准确率 | 原始NLL | 最终NLL | 原始首token率 | 最终首token率 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| fineweb | stable | target | qa | 12.50% | 0.00% | 8.2921 | 4.8718 | 0.00% | 0.00% |
| fineweb | stable | target | binding | 50.00% | 25.00% | 1.7892 | 2.0317 | 50.00% | 25.00% |
| fineweb | stable | anchor | qa | 75.00% | 37.50% | 2.6257 | 3.1632 | 25.00% | 0.00% |
| fineweb | stable | anchor | binding | 100.00% | 75.00% | 0.1349 | 1.3380 | 100.00% | 75.00% |
| fineweb | correction | target | qa | 100.00% | 100.00% | 3.1544 | 2.6441 | 0.00% | 0.00% |
| fineweb | correction | target | binding | 100.00% | 100.00% | 0.0423 | 0.5998 | 100.00% | 100.00% |
| fineweb | correction | anchor | qa | 75.00% | 25.00% | 2.4799 | 3.3703 | 25.00% | 0.00% |
| fineweb | correction | anchor | binding | 100.00% | 100.00% | 0.2087 | 1.4564 | 100.00% | 100.00% |
| fineweb | recent1_distractor | target | qa | 25.00% | 0.00% | 7.3306 | 5.2159 | 0.00% | 0.00% |
| fineweb | recent1_distractor | target | binding | 75.00% | 0.00% | 1.0572 | 1.9405 | 75.00% | 0.00% |
| fineweb | recent1_distractor | anchor | qa | 75.00% | 37.50% | 2.3959 | 3.1874 | 25.00% | 0.00% |
| fineweb | recent1_distractor | anchor | binding | 100.00% | 62.50% | 0.1038 | 1.3586 | 100.00% | 62.50% |
| fineweb | recent2_distractor | target | qa | 87.50% | 25.00% | 3.6865 | 3.7744 | 0.00% | 0.00% |
| fineweb | recent2_distractor | target | binding | 75.00% | 62.50% | 0.7955 | 1.3460 | 75.00% | 62.50% |
| fineweb | recent2_distractor | anchor | qa | 75.00% | 0.00% | 2.5346 | 3.2490 | 25.00% | 0.00% |
| fineweb | recent2_distractor | anchor | binding | 100.00% | 50.00% | 0.1243 | 1.3706 | 100.00% | 50.00% |
| longcrawl | stable | target | qa | 0.00% | 0.00% | 11.1529 | 4.5899 | 0.00% | 0.00% |
| longcrawl | stable | target | binding | 50.00% | 25.00% | 4.4406 | 2.3630 | 50.00% | 25.00% |
| longcrawl | stable | anchor | qa | 25.00% | 25.00% | 7.3789 | 2.8963 | 0.00% | 0.00% |
| longcrawl | stable | anchor | binding | 100.00% | 50.00% | 0.1176 | 1.6970 | 100.00% | 50.00% |
| longcrawl | correction | target | qa | 100.00% | 100.00% | 3.7013 | 1.3588 | 0.00% | 50.00% |
| longcrawl | correction | target | binding | 100.00% | 100.00% | 0.0202 | 0.7455 | 100.00% | 100.00% |
| longcrawl | correction | anchor | qa | 25.00% | 37.50% | 6.4343 | 2.7980 | 0.00% | 0.00% |
| longcrawl | correction | anchor | binding | 100.00% | 37.50% | 0.1318 | 1.9257 | 100.00% | 37.50% |
| longcrawl | recent1_distractor | target | qa | 37.50% | 50.00% | 7.8202 | 2.7999 | 0.00% | 37.50% |
| longcrawl | recent1_distractor | target | binding | 62.50% | 50.00% | 1.6212 | 1.9064 | 62.50% | 50.00% |
| longcrawl | recent1_distractor | anchor | qa | 25.00% | 25.00% | 7.0160 | 2.7456 | 0.00% | 0.00% |
| longcrawl | recent1_distractor | anchor | binding | 100.00% | 50.00% | 0.1043 | 1.6958 | 100.00% | 50.00% |
| longcrawl | recent2_distractor | target | qa | 37.50% | 12.50% | 6.1772 | 2.9617 | 0.00% | 0.00% |
| longcrawl | recent2_distractor | target | binding | 75.00% | 25.00% | 0.8919 | 1.8445 | 75.00% | 25.00% |
| longcrawl | recent2_distractor | anchor | qa | 25.00% | 25.00% | 7.2980 | 2.8647 | 0.00% | 0.00% |
| longcrawl | recent2_distractor | anchor | binding | 100.00% | 50.00% | 0.1413 | 1.7489 | 100.00% | 50.00% |

NLL为正确颜色token的全词表负对数似然；首token率仅比较全词表argmax与正确单token颜色，不是自由生成的多token语义问答。原QA下NLL可以改善而候选排序准确率下降，不能只看NLL就宣称记忆有用。

## 协议、加载与独立审计

- 原始与最终模型使用同一仓库Qwen3类，TTT关闭、标准完整DynamicCache、BF16、SDPA、TF32关闭。固定六块4096 tokens；每条查询深拷贝缓存，权重和父缓存保持不变。
- 相同数据与两种查询，共用token ids、八候选与75%门槛，不使用chat模板或thinking开关；结论限于这个原始文本读出协议。没有构造新来源或评分confirm。
- 两模型各80上下文/288查询；原始Transformers原生类额外固定16上下文/32查询。共176上下文/608查询。
- 原始加载无缺失/额外/形状不匹配参数；最终只允许明确排除七层各conv/proj共14个TTT参数，主干完整。普通主干均4,022,468,096参数；每种路径的类、配置、加载信息、源权重文件与实际参数哈希见summary.loading_audits。
- 原始模型原生/仓库主干哈希一致，32查询全部候选NLL最大误差0且预测一致；这是固定两域四场景子样本校验，不宣称所有输入形式等价。
- 最终模型全部288查询复现上轮CBF M=0结果：全部八候选NLL最大误差0，八选一及全词表首token预测一致。因此当前对照排除了这次普通加载路径引入差异的解释。
- 独立脚本重算完整summary、核验旧构造审计和文件哈希、重新计算两个checkpoint源文件哈希、核验完成状态，全部通过。没有修改原checkpoint。

## 时间与显存

开始：2026-10-09T23:27:34+08:00；结束：2026-10-09T23:52:12+08:00；墙钟 1478 秒。hku-gpu2双RTX5090。

| 模型主路径 | 最高allocated GiB | 最高reserved GiB | 分片0秒（含加载/哈希） | 分片1秒 |
|---|---:|---:|---:|---:|
| original_repo | 14.694 | 18.957 | 656.381 | 634.864 |
| final_repo | 14.694 | 18.957 | 656.001 | 639.464 |

逐轨迹/查询耗时和峰值显存存于远程JSONL；显存为PyTorch allocated/reserved，不是整机占用。墙钟包括测试、加载与哈希核验，不能直接当纯前向吞吐。

## 文件、参数与baseline兼容性

- `CHECKPOINT_READOUT_PLAN.md`：执行前冻结输入、双模型、两条路径核验及结果解释分支。
- `tasks/cbf_checkpoint_readout.py`：PlainSession，严格load_plain/check_loading、parameter_digest/cache_digest、collect、validate_scores/compare_scores、summarize及源组配对差。
- `scripts/run_cbf_checkpoint_readout.sh`：双GPU顺序运行原生校验/原始主路径/最终主路径，测试与失败/完成标记，拒绝覆盖ROOT。
- `scripts/audit_cbf_checkpoint_readout.py`：已有结果只读重算与源checkpoint哈希复核。
- `tests/test_cbf_checkpoint_readout.py`：参数缺失/未知、分数篡改、全候选误差、完整汇总和原生不同权重拒绝测试。
- 本目录：RUN_STATUS、summary、execution_audit、REPORT及排除原始数据的.gitignore；README/MODIFICATION_LOG同步入口和全过程。
- 参数：ROOT新输出目录、SOURCE已完成disjoint归档、PYTHON、ORIGINAL、FINAL；CLI collect的--arm/--shard/--model/--source/--output，summarize和audit的--root/--source/--output。
- 无新依赖；baseline模型/runtime/训练配置及旧collector均未修改。关闭TTT仅在新加载的临时配置中生效。原始语料/token/逐条评分/权重保留远程。

采集代码提交：`87866f698654ca6b50b1f8f6414630e51a715d1c`；服务器启动前12项测试通过（2.123秒）。本地系统Python无torch，未把该环境的torch依赖测试失败记为通过；后续补充的完整汇总测试在本地独立通过。完成审计脚本在采集期间准备，不更改已冻结采集。

## 复现

依赖服务器保有原始两个模型和上一轮完整数据/评分，公开聚合归档不能单独重建自然背景。使用全新ROOT。

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_checkpoint_readout_repro bash scripts/run_cbf_checkpoint_readout.sh

/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python -m scripts.audit_cbf_checkpoint_readout \
  --root /home/ctj/cbf_ttt_checkpoint_readout_20261009 \
  --source /home/ctj/cbf_ttt_disjoint_readout_20261009 \
  --output /home/ctj/cbf_ttt_checkpoint_readout_20261009/execution_audit_recheck.json
```

## 附件对应、风险与下一步

这一步验证附件局部遗忘、多分支和联合训练所需的基础读取能力。它没有测量M的内容或遗忘收益，不能用checkpoint差异替代遗忘效应。原始模型的失败说明当前24K含干扰任务有共同读取瓶颈；最终模型的额外下降说明仍需定位checkpoint相关变化。这两个结论可以同时成立。

已完成：固定任务双模型对照、两个数值桥接、严格加载/缓存/权重/结果审计。未执行：新来源Q、M写入/V/F、标签扩容、控制器或端到端训练。

下一步建议优先做固定预算的长度对照：两模型都保留同样的事实键值和干扰记录，仅缩短自然背景，检查较短输入能否通过同一逐域四场景门槛，同时保留本次24K结果作长输入对照。事先冻结长度与门槛，禁止再根据分数不断增加模板；缩短背景会同时改变事实相对距离，须明确这个限制，不能把长度效应当遗忘收益。通过后才冻结可读协议并用新来源做Q，再检查M是否携带有用内容。checkpoint变化的训练/导出归因另行核验，不直接扩大遗忘控制器训练。

限制：8个已观察开发来源组、人工颜色绑定、四色循环干扰、共享双生背景、无chat模板以及有限原生校验覆盖；不外推到自然问答或公开benchmark整体表现。
