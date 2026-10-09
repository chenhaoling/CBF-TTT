# 固定记录6K/12K/24K读取对照：完成报告

## 结论与决策

最终checkpoint在两个预登记短长度下都未通过完整16格门槛，停止此版本标签扩容；没有追加更短长度、换模板或删除难例。较短输入的局部提升不能替代完整门槛，也不能判定遗忘机制无效。

原始模型按相同门槛的候选为12288 tokens / binding；两模型的通过情况分别报告，不能替最终模型选择原始模型的结果。

两模型均保留同一批事实、干扰和查询。缩短背景也改变了内容、绝对位置、事实间距及前向分段长度，因此本轮只识别该固定裁剪干预下的可读区间，不能单独归因于某一种长距离注意力机理或训练根因。完整KV、M=0，无训练。训练后的主干可能与TTT快记忆共同适配，关闭M后的下降不能直接代表原生TTT整体性能下降。

## 总体结果

各格为四场景等权均值；单个场景/问题为16会话、8来源组。候选准确率为八选一，不能当作自由生成问答分数。

| 模型 | 长度 | QA目标 | QA锚点 | binding目标 | binding锚点 | QA失败格/16 | binding失败格/16 |
|---|---|---:|---:|---:|---:|---:|---:|
| original_repo | 6K (6144) | 67.19% | 76.56% | 87.50% | 98.44% | 5/16 | 1/16 |
| original_repo | 12K (12288) | 75.00% | 79.69% | 93.75% | 100.00% | 6/16 | 0/16 |
| original_repo | 24K (24576) | 50.00% | 50.00% | 73.44% | 100.00% | 9/16 | 3/16 |
| final_repo | 6K (6144) | 45.31% | 87.50% | 51.56% | 92.19% | 7/16 | 6/16 |
| final_repo | 12K (12288) | 65.62% | 76.56% | 59.38% | 54.69% | 8/16 | 12/16 |
| final_repo | 24K (24576) | 35.94% | 26.56% | 48.44% | 59.38% | 14/16 | 12/16 |

### 四场景明细（binding）

| 模型 | 长度 | 场景 | 目标准确率 | 锚点准确率 | 目标NLL | 锚点NLL | 目标首token率 | 锚点首token率 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| original_repo | 6K (6144) | stable | 81.25% | 93.75% | 0.6434 | 0.1127 | 81.25% | 93.75% |
| original_repo | 6K (6144) | correction | 100.00% | 100.00% | 0.0223 | 0.0485 | 100.00% | 100.00% |
| original_repo | 6K (6144) | recent1_distractor | 68.75% | 100.00% | 1.0559 | 0.0744 | 68.75% | 100.00% |
| original_repo | 6K (6144) | recent2_distractor | 100.00% | 100.00% | 0.0340 | 0.0941 | 100.00% | 100.00% |
| original_repo | 12K (12288) | stable | 100.00% | 100.00% | 0.0710 | 0.0410 | 100.00% | 100.00% |
| original_repo | 12K (12288) | correction | 100.00% | 100.00% | 0.0210 | 0.0655 | 100.00% | 100.00% |
| original_repo | 12K (12288) | recent1_distractor | 75.00% | 100.00% | 1.0739 | 0.0354 | 75.00% | 100.00% |
| original_repo | 12K (12288) | recent2_distractor | 100.00% | 100.00% | 0.0600 | 0.0405 | 100.00% | 100.00% |
| original_repo | 24K (24576) | stable | 50.00% | 100.00% | 3.1149 | 0.1262 | 50.00% | 100.00% |
| original_repo | 24K (24576) | correction | 100.00% | 100.00% | 0.0313 | 0.1702 | 100.00% | 100.00% |
| original_repo | 24K (24576) | recent1_distractor | 68.75% | 100.00% | 1.3392 | 0.1041 | 68.75% | 100.00% |
| original_repo | 24K (24576) | recent2_distractor | 75.00% | 100.00% | 0.8437 | 0.1328 | 75.00% | 100.00% |
| final_repo | 6K (6144) | stable | 37.50% | 100.00% | 1.8877 | 1.1645 | 37.50% | 100.00% |
| final_repo | 6K (6144) | correction | 93.75% | 100.00% | 0.8657 | 1.3329 | 93.75% | 100.00% |
| final_repo | 6K (6144) | recent1_distractor | 25.00% | 81.25% | 2.0667 | 1.1898 | 25.00% | 81.25% |
| final_repo | 6K (6144) | recent2_distractor | 50.00% | 87.50% | 1.4682 | 1.1727 | 50.00% | 87.50% |
| final_repo | 12K (12288) | stable | 43.75% | 62.50% | 1.7240 | 1.3767 | 43.75% | 62.50% |
| final_repo | 12K (12288) | correction | 100.00% | 43.75% | 0.7272 | 1.5520 | 100.00% | 43.75% |
| final_repo | 12K (12288) | recent1_distractor | 43.75% | 56.25% | 1.7229 | 1.4277 | 43.75% | 56.25% |
| final_repo | 12K (12288) | recent2_distractor | 50.00% | 56.25% | 1.4331 | 1.4195 | 50.00% | 56.25% |
| final_repo | 24K (24576) | stable | 25.00% | 62.50% | 2.1974 | 1.5175 | 25.00% | 62.50% |
| final_repo | 24K (24576) | correction | 100.00% | 68.75% | 0.6727 | 1.6911 | 100.00% | 68.75% |
| final_repo | 24K (24576) | recent1_distractor | 25.00% | 56.25% | 1.9235 | 1.5272 | 25.00% | 56.25% |
| final_repo | 24K (24576) | recent2_distractor | 43.75% | 50.00% | 1.5953 | 1.5598 | 43.75% | 50.00% |

### 完整分域门槛

每域单格8会话/4来源组；目标与锚点均须≥75%，每格式共16格。

| 模型 | 长度 | 域 | 场景 | QA目标 | QA锚点 | binding目标 | binding锚点 |
|---|---|---|---|---:|---:|---:|---:|
| original_repo | 6K (6144) | fineweb | stable | 12.50% | 75.00% | 87.50% | 100.00% |
| original_repo | 6K (6144) | fineweb | correction | 100.00% | 87.50% | 100.00% | 100.00% |
| original_repo | 6K (6144) | fineweb | recent1_distractor | 62.50% | 75.00% | 87.50% | 100.00% |
| original_repo | 6K (6144) | fineweb | recent2_distractor | 100.00% | 75.00% | 100.00% | 100.00% |
| original_repo | 6K (6144) | longcrawl | stable | 12.50% | 87.50% | 75.00% | 87.50% |
| original_repo | 6K (6144) | longcrawl | correction | 100.00% | 100.00% | 100.00% | 100.00% |
| original_repo | 6K (6144) | longcrawl | recent1_distractor | 50.00% | 75.00% | 50.00% | 100.00% |
| original_repo | 6K (6144) | longcrawl | recent2_distractor | 100.00% | 37.50% | 100.00% | 100.00% |
| original_repo | 12K (12288) | fineweb | stable | 50.00% | 75.00% | 100.00% | 100.00% |
| original_repo | 12K (12288) | fineweb | correction | 100.00% | 100.00% | 100.00% | 100.00% |
| original_repo | 12K (12288) | fineweb | recent1_distractor | 50.00% | 100.00% | 75.00% | 100.00% |
| original_repo | 12K (12288) | fineweb | recent2_distractor | 100.00% | 75.00% | 100.00% | 100.00% |
| original_repo | 12K (12288) | longcrawl | stable | 62.50% | 62.50% | 100.00% | 100.00% |
| original_repo | 12K (12288) | longcrawl | correction | 100.00% | 100.00% | 100.00% | 100.00% |
| original_repo | 12K (12288) | longcrawl | recent1_distractor | 37.50% | 50.00% | 75.00% | 100.00% |
| original_repo | 12K (12288) | longcrawl | recent2_distractor | 100.00% | 75.00% | 100.00% | 100.00% |
| original_repo | 24K (24576) | fineweb | stable | 12.50% | 75.00% | 50.00% | 100.00% |
| original_repo | 24K (24576) | fineweb | correction | 100.00% | 75.00% | 100.00% | 100.00% |
| original_repo | 24K (24576) | fineweb | recent1_distractor | 25.00% | 75.00% | 75.00% | 100.00% |
| original_repo | 24K (24576) | fineweb | recent2_distractor | 87.50% | 75.00% | 75.00% | 100.00% |
| original_repo | 24K (24576) | longcrawl | stable | 0.00% | 25.00% | 50.00% | 100.00% |
| original_repo | 24K (24576) | longcrawl | correction | 100.00% | 25.00% | 100.00% | 100.00% |
| original_repo | 24K (24576) | longcrawl | recent1_distractor | 37.50% | 25.00% | 62.50% | 100.00% |
| original_repo | 24K (24576) | longcrawl | recent2_distractor | 37.50% | 25.00% | 75.00% | 100.00% |
| final_repo | 6K (6144) | fineweb | stable | 12.50% | 75.00% | 50.00% | 100.00% |
| final_repo | 6K (6144) | fineweb | correction | 87.50% | 50.00% | 100.00% | 100.00% |
| final_repo | 6K (6144) | fineweb | recent1_distractor | 0.00% | 87.50% | 12.50% | 62.50% |
| final_repo | 6K (6144) | fineweb | recent2_distractor | 12.50% | 87.50% | 25.00% | 75.00% |
| final_repo | 6K (6144) | longcrawl | stable | 37.50% | 100.00% | 25.00% | 100.00% |
| final_repo | 6K (6144) | longcrawl | correction | 100.00% | 100.00% | 87.50% | 100.00% |
| final_repo | 6K (6144) | longcrawl | recent1_distractor | 50.00% | 100.00% | 37.50% | 100.00% |
| final_repo | 6K (6144) | longcrawl | recent2_distractor | 62.50% | 100.00% | 75.00% | 100.00% |
| final_repo | 12K (12288) | fineweb | stable | 37.50% | 50.00% | 37.50% | 50.00% |
| final_repo | 12K (12288) | fineweb | correction | 100.00% | 37.50% | 100.00% | 37.50% |
| final_repo | 12K (12288) | fineweb | recent1_distractor | 37.50% | 62.50% | 37.50% | 50.00% |
| final_repo | 12K (12288) | fineweb | recent2_distractor | 50.00% | 62.50% | 25.00% | 50.00% |
| final_repo | 12K (12288) | longcrawl | stable | 75.00% | 100.00% | 50.00% | 75.00% |
| final_repo | 12K (12288) | longcrawl | correction | 100.00% | 100.00% | 100.00% | 50.00% |
| final_repo | 12K (12288) | longcrawl | recent1_distractor | 50.00% | 100.00% | 50.00% | 62.50% |
| final_repo | 12K (12288) | longcrawl | recent2_distractor | 75.00% | 100.00% | 75.00% | 62.50% |
| final_repo | 24K (24576) | fineweb | stable | 0.00% | 37.50% | 25.00% | 75.00% |
| final_repo | 24K (24576) | fineweb | correction | 100.00% | 25.00% | 100.00% | 100.00% |
| final_repo | 24K (24576) | fineweb | recent1_distractor | 0.00% | 37.50% | 0.00% | 62.50% |
| final_repo | 24K (24576) | fineweb | recent2_distractor | 25.00% | 0.00% | 62.50% | 50.00% |
| final_repo | 24K (24576) | longcrawl | stable | 0.00% | 25.00% | 25.00% | 50.00% |
| final_repo | 24K (24576) | longcrawl | correction | 100.00% | 37.50% | 100.00% | 37.50% |
| final_repo | 24K (24576) | longcrawl | recent1_distractor | 50.00% | 25.00% | 50.00% | 50.00% |
| final_repo | 24K (24576) | longcrawl | recent2_distractor | 12.50% | 25.00% | 25.00% | 50.00% |

### 按来源组配对差（binding，百分点）

每个值是该组四场景×两问题×双生的平均准确率差。组内长度、格式和双生不是独立重复，未做显著性或推广置信区间主张。

| 组 | 原始6K−24K | 原始12K−24K | 最终6K−24K | 最终12K−24K | 6K最终−原始 | 12K最终−原始 | 24K最终−原始 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | +0.00 | +0.00 | +18.75 | -6.25 | -37.50 | -62.50 | -56.25 |
| 1 | +18.75 | +18.75 | +12.50 | -6.25 | -50.00 | -68.75 | -43.75 |
| 2 | +25.00 | +25.00 | +0.00 | -43.75 | -31.25 | -75.00 | -6.25 |
| 3 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| 4 | +0.00 | +0.00 | +50.00 | +62.50 | -25.00 | -12.50 | -75.00 |
| 5 | -12.50 | +18.75 | +68.75 | +50.00 | +12.50 | -37.50 | -68.75 |
| 6 | +12.50 | +12.50 | -12.50 | +12.50 | -37.50 | -12.50 | -12.50 |
| 7 | +6.25 | +6.25 | +6.25 | -43.75 | +0.00 | -50.00 | +0.00 |

全部QA/binding、分域/来源组NLL与首token指标及逐格失败列表见summary.json；首token是全词表argmax匹配正确颜色token，不是多token生成语义评测。

## 构造、复用与样本计数

- 原disjoint clean64：FineWeb-Edu/LongCrawl64各4个已观察dev来源组，四场景×双生；无新来源或confirm。
- 每个4096-token段保留最后1024/2048/4096 tokens，六段顺序不变。全部header和两条记录保留，设备键、颜色、记录顺序、两种query及8候选逐值不变；无重新分词/填充。
- 独立检查144个新输入中的1728条记录。短窗嵌套，只裁去每段前部自然背景；自然背景仍可能包含颜色词。
- 新采集每模型128个短输入加16个24K复现输入，共288轨迹/1152查询。
- 24K主表复用上轮每模型64个clean的完整结果，重新验证原始分片、summary和审计哈希，以及当前模型的配置/源文件/实际权重完全一致。
- 主分析两模型×三长度×64=384条件/1536查询。32个重跑24K轨迹是复现检查，不额外当作新独立样本计入主表。
- 16格均≥75%才能通过；选最终模型最长通过的短长度，同长度选择最弱格最大格式，平局QA。候选仅供后续新来源Q，不自动启动写入或训练。

## 执行审计与成本

采集提交 `a792bfc382b35a3dbd4530dc3ecf2770ba3fc59d`；hku-gpu2双RTX5090。开始 2026-10-10T00:11:23+08:00，结束 2026-10-10T00:26:30+08:00，采集/汇总流程墙钟 **907秒（15分07秒）**，之后单独执行完成审计。服务器17项测试通过（6.184秒），本地相关12项测试通过（7.512秒）。

| 模型 | 长度 | 本轮轨迹数 | rollout均值秒 | query均值秒 | 峰值allocated GiB | 峰值reserved GiB |
|---|---|---:|---:|---:|---:|---:|
| original_repo | 6K (6144) | 64 | 0.534 | 0.038 | 9.346 | 10.482 |
| original_repo | 12K (12288) | 64 | 1.365 | 0.061 | 11.128 | 13.516 |
| original_repo | 24K (24576) | 16 | 4.140 | 0.106 | 14.694 | 18.908 |
| final_repo | 6K (6144) | 64 | 0.534 | 0.037 | 9.345 | 10.498 |
| final_repo | 12K (12288) | 64 | 1.367 | 0.061 | 11.128 | 13.516 |
| final_repo | 24K (24576) | 16 | 4.154 | 0.106 | 14.694 | 18.957 |

每条查询/轨迹的耗时和显存保存在远程逐条JSONL，公开归档提供聚合。峰值是PyTorch统计，不是整机占用；总墙钟含测试、构造、加载、权重/缓存哈希开销，不是纯前向吞吐。

严格加载：两模型普通主干4,022,468,096参数；最终只允许排除预定14个TTT conv/proj参数，主干无缺失/形状不匹配。共享仓库普通Qwen3路径、BF16/SDPA、TF32关闭、TTT关闭。每查询独立KV克隆，父缓存及权重不变。

两模型各64次24K查询的所有八候选NLL误差均为0，候选及全词表首token预测一致。独立审计重算summary、直接检查实际尾窗/header/记录/查询、重哈希checkpoint源文件并核验完成状态，全部通过。旧24K完整结果使用前也重新汇总并逐值核对。

## 文件、配置和复现

| 文件 | 本轮改动 |
|---|---|
| tasks/cbf_length_readout.py | 嵌套窗口make_rows/validate、build/load_data、collect、reference_summary、summarize/paired/select_length |
| tasks/cbf_checkpoint_readout.py | collector仅增加可选data_path/validator/identity_fields/expected_data_sha，旧默认保持 |
| scripts/run_cbf_length_readout.sh | 双卡双模型构造/采集/汇总；失败与完成标记；拒绝覆盖ROOT |
| scripts/audit_cbf_length_readout.py | 实际尾窗/完整记录独立检查、summary重算、checkpoint源文件哈希复核 |
| tests/test_cbf_length_readout.py | 嵌套与真值不变、裁断/篡改拒绝、最长合格选择、完整汇总/长度身份拒绝 |
| LENGTH_READOUT_PLAN.md | 执行前固定条件、门槛和后续分支；完成记录 |
| 本目录及README/MODIFICATION_LOG | 设计/汇总/审计/状态/完整报告与修改记录 |

参数ROOT（新输出目录）、SOURCE（原disjoint归档）、REFERENCE（上一轮checkpoint对照）、PYTHON/ORIGINAL/FINAL通过环境变量传入。三长度、格式、来源组、门槛是冻结协议常量。新CLI build使用--source/--reference/--output；collect使用--root/--arm/--model/--shard/--output；summarize和audit使用--root/--output。

baseline模型/runtime/训练配置不改，旧collector默认兼容，无新依赖。原始文本、token、逐条评分及权重保留远程，不上传GitHub。公开聚合文件不能单独重建原始背景，复现依赖服务器已有checkpoint和完整前轮归档。

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_length_readout_repro bash scripts/run_cbf_length_readout.sh

/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python -m scripts.audit_cbf_length_readout \
  --root /home/ctj/cbf_ttt_length_readout_20261010 \
  --output /home/ctj/cbf_ttt_length_readout_20261010/execution_audit_recheck.json
```

## 附件对应、限制与后续TODO

本轮只验证局部遗忘、多分支及联合训练所需的基础读取前提，没有写入M、测量记忆内容/遗忘收益或训练控制器。8组已观察开发背景、人工颜色绑定、四色循环干扰、无chat模板、多个读出条件均限制外推；不声称公开benchmark或自然问答能力改善。

下一步：保留原始模型已通过的固定条件作正对照，暂停此版本标签扩容，优先比较现有中间训练checkpoint在同一条件下的读取能力，定位M=0差异出现的阶段，并核验训练/导出及主干参数变化；应配合原生TTT路径对照，区分主干变化与对快记忆的依赖。后续须另行冻结这些对照及通过条件，再用新来源确认；不要继续搜索提示词或将原始模型的通过代替最终模型的通过。该checkpoint定位实验尚未执行。

已完成本轮构造、双模型短长度采集、旧结果复现、配对汇总、资源与独立审计。未执行新来源Q、M写入/V/F、标签扩容、控制器或端到端联合训练；没有修改原checkpoint。
