# 单次原生 Fast Weight 内容因果实验 v2：完成报告

## 结论

本轮先修复开放词表读出，再在独立来源上完成 Correct/Wrong/Zero/Random fast-weight 因果干预。基础读取达到100%，因此结果不再受“模型读不出上下文事实”的解释混淆。最终结论为：**当前 Qwen3-4B 最终 checkpoint 的单次原生 TTT 更新未表现出可检出的事实内容绑定，也没有内容效用。**

- Correct 相对 Wrong-twin 的答案 NLL 改善仅 **0.00216**，8来源组bootstrap 95% CI **[-0.00064, 0.00493]**，低于0.02门槛且CI跨0。
- Correct 相对 Wrong 的 own-vs-twin margin 提升 **0.01953**，95% CI **[-0.00781, 0.04297]**，低于0.05门槛且CI跨0。
- 等范数 Random 的 NLL 比 Correct 低 **0.01793**，95% CI **[0.00998, 0.02644]**；Zero 比 Correct 低 **0.02025**，95% CI **[0.01180, 0.02892]**。
- 四种策略的八选一和全词表首token均为100%，因此候选准确率已经饱和，NLL和事实对照才是本轮的有效辨别指标。

预登记内容绑定门和内容效用门均失败，终态为 `no_detectable_content_binding`。这不否定受控结构化/低秩实验中的遗忘控制结论；它说明当前原生 writer 尚未提供适合训练遗忘策略的可读事实快记忆。

## dev读出校准与独立Q

三个格式只在已观察dev组0–7比较：

| 格式 | 答案NLL | 八选一 | 首token | 最差域首token |
|---|---:|---:|---:|---:|
| lowercase | 0.2110 | 100% | 100% | 100% |
| options | **0.1382** | 100% | 100% | 100% |
| completion | 2.4882 | 100% | 31.25% | 0% |

按预登记规则选择 `options`。随后在来源互斥的Q组8–15上固定该格式：总体、FineWeb、LongCrawl的八选一和全词表首token均为 **100%**，答案NLL为0.1331，8/8来源组双twin均正确。Q通过后才首次评分V组16–23。

该对照表明v1停止确有格式成分，也说明v2没有通过放宽首token门槛进入V。

## V：保持同一KV，只替换Fast Weight

V共8来源组、16个twin上下文。每条上下文第一块禁写，事实所在第二块执行一次lr=0.3更新。查询始终使用self KV，只替换7层`W_fast`。

| 策略 | 正确答案NLL | twin答案NLL | own-vs-twin margin | 八选一 | 首token | EOS率 |
|---|---:|---:|---:|---:|---:|---:|
| Correct | 0.12824 | 6.68684 | 6.55859 | 100% | 100% | 0% |
| Wrong-twin | 0.13041 | 6.66947 | 6.53906 | 100% | 100% | 0% |
| Zero | **0.10799** | 6.69783 | **6.58984** | 100% | 100% | 0% |
| Random平均 | 0.11032 | 6.66891 | 6.55859 | 100% | 100% | 0% |

### 配对因果差

| 比较（正值代表Correct更优） | 均值 | 95% CI | 结论 |
|---|---:|---:|---|
| Wrong NLL − Correct NLL | +0.00216 | [-0.00064, 0.00493] | 无稳定内容差异 |
| Random NLL − Correct NLL | -0.01793 | [-0.02644, -0.00998] | Random显著更低 |
| Zero NLL − Correct NLL | -0.02025 | [-0.02892, -0.01180] | Zero显著更低 |
| Correct margin − Wrong margin | +0.01953 | [-0.00781, 0.04297] | 无稳定特异性 |

同一KV已经清楚包含事实，所以所有策略都能正确作答。若`Delta W`绑定了对应值，Correct应相对Wrong提高正确答案概率并扩大own-vs-twin margin；两项都未达到门槛。Zero/Random更低的NLL进一步表明当前更新主要增加小幅通用扰动，而非提供有用的事实增量。

## 设计、隔离与审计

- dev/Q/V分别使用组0–7、8–15、16–23，三个集合的原始FineWeb-Edu/LongCrawl64来源完全互斥。
- 每个twin仅改变一个颜色token；query、KV位置、候选和骨干均固定。Wrong只移植twin的fast weight。
- 三个Random对每层correct delta做固定seed Rademacher符号随机化，逐层范数比误差不超过1e-6。
- 每次干预从相同prefix深拷贝，父cache SHA-256不变；参数hash/version在分片前后不变。
- 独立审计从原始parquet重建48条场景，重算dev选择、Q gate、V summary、bootstrap、范数和来源/结果hash，`audit_passed=true`。
- 服务器11项测试通过；采集提交`a1984b2`。原始token、逐条JSONL和模型留服务器。

## 资源

- dev：16上下文×3格式=48查询；Q：16上下文×1格式=16查询；V：16上下文×6干预=96查询。
- 两张RTX5090；2026-10-12 14:39:14–14:41:58 +08:00，总墙钟 **164秒（2分44秒）**，含测试、构造、三阶段模型加载、采集、汇总和审计。
- V平均rollout 0.753秒、单干预查询0.0532秒；最高allocated/reserved为 **10.867/11.746 GiB**。

## 文件与复现

新增v2计划、任务、runner、审计和测试；v1公共评分函数仅增加可选`fmt`参数，默认`qa`行为不变。baseline训练/推理配置、checkpoint和原生更新公式均未修改。

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_fastweight_content_v2_repro \
SOURCE=/home/ctj/cbf_ttt_memory_value_20261009 \
bash scripts/run_cbf_fastweight_content_v2.sh

/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
  -m scripts.audit_cbf_fastweight_content_v2 \
  --root /home/ctj/cbf_ttt_fastweight_content_v2_repro \
  --tokenizer /home/ctj/models/Qwen3-4B \
  --output /home/ctj/cbf_ttt_fastweight_content_v2_repro/execution_audit_recheck.json
```

## 下一步

暂停在当前原生`Delta W`上训练遗忘控制器。下一关键实验应改造writer目标，而不是继续调遗忘策略：冻结Qwen骨干，只训练现有7层`ttt_proj/ttt_conv`，使用同一单块twin任务的“正确答案NLL + Correct-vs-Wrong对比margin”目标；以当前未训练/原训练writer为对照，并继续要求新来源上的Correct优于Wrong、Random和Zero。只有新writer通过本轮内容门，才恢复多块累积、last-1/last-2和遗忘控制器实验。

## 限制

- 人工颜色绑定和16条V上下文不是自然任务benchmark。
- Full-KV本身已经100%正确，存在准确率天花板；本轮依靠NLL与twin特异性检测增量。
- 只检查一次4096-token块尾更新、lr=0.3和7层联合移植；不能排除某一层或其他尺度含微弱内容。
- 阴性结论限定当前1B-token checkpoint及writer训练目标，不推广到所有In-Place TTT实现。
