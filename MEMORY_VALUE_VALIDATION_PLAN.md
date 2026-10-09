# 快速记忆价值验证：实验计划 v1

日期：2026-10-08（2026-10-09 开始执行）。状态：本轮已按 Q 失败门槛结束，V/F 与确认集未执行；文末记录实际 CLI、运行结果及报告。下文保留原始计划，待办以最新执行记录为准。

## 1. 研究问题与已有证据

验证对象为最终 Qwen3-4B / 1B-token checkpoint 的会话快速记忆。依次回答：

1. **内容价值 V**：正确会话记忆是否帮助回答当前会话事实，并超过无记忆和错配记忆？
2. **遗忘价值 F**：保留历史在有效事实上有收益，在失效事实上有代价，是否形成需要调节保留率的取舍？
3. **局部价值 S**：有用和无用贡献共存时，局部遗忘是否优于同状态全局遗忘？

V/F/S 分开报告，成功一个不等于其他成立。通过后才进入附件第③项的联合训练，不将 oracle 收益当作控制器效果。

### 已核对的历史结果

- [多决策实验](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/multidecision_forgetting/REPORT.md)：单/双/三点全局 oracle NLL 为 13.703427/5.997364/3.378568；采样局部无额外收益；每步清空/不写入为 2.084365/2.025309。多次干预缓解退化，尚未证明选择性记忆价值。
- [标题 writer 实验](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/task_writer_pilot/REPORT.md)：正确、错配和平均更新都改善 NLL；仅超过不写入不足以证明记住当前内容。
- [配对事实实验](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/paired_fact_writer_pilot/REPORT.md)：完整 KV 可读性 100%，两种 writer 清空 KV 后 dev 八选一准确率均为 12.5%，test 未评分。

因此本轮不是首次尝试人工事实或清空 KV。新增的是**新来源、多块时距、四种信息关系、统一的记忆干预，以及独立确认流程**。不重复旧 rank-8 writer 训练或已完成的数值诊断。

## 2. 固定模型与机制

- 服务器 hku-gpu2，两张 RTX 5090；BF16、eval，复用现有环境。
- 模型：`/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt`。
- Tokenizer：`/home/ctj/models/Qwen3-4B`。
- Python：`/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python`。
- 冻结骨干及 TTT conv/proj。本轮无 optimizer、无控制器训练，不重新预训练。
- 每会话从空 KV、M=0 开始，6×4096 token；保留原 TTT 学习率和层配置，不调写入尺度。所有固定策略从第一个 chunk 执行。
- 用 **a 表示历史保留率**，`M_new=a*M_old+ΔW`，g=1。最初附件的 alpha 是遗忘率，映射为 `alpha_forget=1-a`；实现时不能混用。
- `clear` 清旧但写新，不是 M 永远为零；`none` 才是不写入。
- 正常 KV 推进不改。清空 KV 仅在评分克隆上进行；每个策略的后续 ΔW 必须依自身状态重新计算。

## 3. 数据构建与隔离

### 3.1 小规模预算

**24 个新来源组：dev 8、confirm 16。** 每组四场景、两个事实版本，合计 **192 会话**（dev 64、confirm 128）。独立分析单位仍只有 24 个来源组。

每个 split 中 FineWeb-Edu/LongCrawl64 来源组各半，等长背景，等 token 预算。分组 seed=208，事实 seed=209。两 GPU 按 group_id 分片，同组全部版本/策略在同卡。

- LongCrawl64 每组用一篇足够长的文档提供六段不同背景。
- FineWeb-Edu 每组可用六篇不同文档各提供一段；明确标为多文档背景，不冒称单篇连续长文。
- 所有原文只能归属一个组和 split。同组不同场景复用背景是配对设计，不算新增独立来源。
- 为事实槽位预留固定长度，截取自然背景补足 4096 token；禁止循环重复 filler。
- 先盘点服务器原始 shard 和来源 manifest；缺失时按 [数据下载指南](DATA_DOWNLOAD_GUIDE.md) 获取原始数据。`mixed_1b.jsonl` 只用于重叠审计，不能直接当未见确认集。
- 排除既有实验全部已用来源 ID、规范化全文哈希及可取得的片段；扫描与已用 1B 续训语料的长锚点匹配。登记候选顺序、排除原因、最终片段哈希和数据版本。
- 锚点扫描只证明未发现指定匹配，不是语义去重或 Qwen 基座预训练去重。随机插入事实独立于背景，降低靠已有知识答题的可能。
- 若任一数据集来源不足，记录缺口并修订协议，不能根据结果改为单一数据集或重复背景。

只依长度、来源和结构选择数据，不依模型分数筛选。confirm 只做结构审计，首次模型评分前封存。

### 3.2 双生事实及标签

沿用八个可审计为单 token 的颜色值，配合独立实体键，例如设备 access color；使用随机会话标识。每次查询只问一个键，不提供正确值；八选一候选集合固定且排序不携带标签。

每组 A/B 两版本背景、问题、位置和长度相同，仅目标事实值不同。纠错场景共享同一个过期值，新值各不相同且均不等于旧值。标签按 split×场景均衡，旧值也需平衡；实体词形、数据集和模板不能决定答案。

每会话有两个独立查询：旧锚点键和目标键。锚点值在双生版本间相同，目标值不同；锚点标签在组间均衡，且不重复目标键。两个查询独立克隆、等权报告，互不写入记忆。目标双生识别用于内容价值门槛；锚点用于检查保留损伤。

### 3.3 四种场景

| 场景 | chunk 2 | chunk 4 | chunk 5 | chunk 6 | 正确答案依据 |
|---|---|---|---|---|---|
| stable | 锚点＋目标事实 | 无关内容 | 无关内容 | 无关内容 | chunk 2 的有效事实 |
| correction | 锚点＋过期目标事实 | 无关内容 | 无关内容 | 明确更正同一目标键 | 保留锚点，目标使用新值 |
| recent1_distractor | 锚点 | 有效目标事实 | 其他实体干扰事实 | 无关内容 | 锚点＋chunk 4 目标 |
| recent2_distractor | 锚点 | 其他实体干扰事实 | 有效目标事实 | 无关内容 | 锚点＋chunk 5 目标 |

chunk 1/3 为不同自然背景。各位置尽量匹配槽位长度/记录数，缺省槽位用其他实体记录补齐；记录实际 token 位置。干扰使用不同键，不能预设干扰一定有害。纠错明确说明新记录替代旧记录，避免真值歧义。

可选局部诊断在 chunk 6 提交前分支：H 为更早贡献、R2 为 chunk 4、R1 为 chunk 5，当前 ΔW 为 chunk 6。删除直接更新贡献不能撤销其对后续 ΔW 或 KV 的间接影响，不等同于精确语义删除。

## 4. 策略与因果对照

### 4.1 从会话起点运行的四种固定策略

| 策略 | 操作 | 目的 |
|---|---|---|
| none | `_forward(collect=False)`，M 为空 | 不写入基线 |
| retain | a=1、g=1 | 持续累积 |
| half | a=0.5、g=1 | 固定衰减 |
| clear | a=0、g=1 | 每步清旧写新 |

报告全部策略，但主确认策略只能由 dev 一次选定，不能逐 confirm 场景取最好值。

### 4.2 两种评分上下文

- **full_kv**：保留原生完整 KV，衡量实际设置。
- **memory_only**：现有 `clone_memory_only()` 保留 M、新建空 KV，再输入相同 query；位置编号按新会话重启。所有该条件对照使用同一位置协议。

分开报告，不把两种上下文的直接分数差当作 M 的贡献。memory_only 改变上下文和位置分布，失败不能单独否定原生快记忆；仅该条件成功也不能宣称部署收益。

### 4.3 相同读取状态上的三种 M 干预

每条 retain/half/clear 路径在两种 KV 条件分别比较：

1. **self**：当前版本自己的 M。
2. **zero**：相同读取分支的 M 置零，其他状态不动。
3. **twin**：替换为同来源、同场景、同策略的另一个事实版本 M；评分当前版本的正确答案。

full_kv 干预必须保留当前版本 KV，不得搬入对方 KV。twin 是使用双生轨迹的机制诊断，不是可部署策略。

独立 none 路径必须保留：full_kv 下 score 时 zero 和全程 none 不等价，之前的 M 可能已改变 KV。memory_only 下 zero 与 none 的相同查询评分应一致（NLL 误差≤1e-5）。

本轮没有训练集，不导入其他实验的平均 M 作为同配置对照。相同背景、不同事实的 twin 已直接检验收益是否依赖当前内容。

## 5. 指标、选型与预设门槛

### 5.1 指标

- 单 token 正确答案全词表 NLL、八选一准确率、全词表 greedy top-1 准确率。
- `B_none=NLL_none−NLL_self`：相对不写入的端到端收益。
- `B_zero=NLL_zero−NLL_self`：同读取状态移除 M 的收益。
- `B_twin=NLL_twin−NLL_self`：正确内容相对错配内容的收益。
- 双生两版本目标是否均正确，防止只学到常见答案。
- correction 单列新值正确率、旧值误选率和锚点保留率；局部场景分别报告锚点和近期目标。
- 按来源组、数据集、场景、KV 分层，组内版本和查询不算独立样本。

本轮是小规模独立复核，报告逐组配对差、均值、中位数、胜组数和最差组。不把工程阈值当作统计显著性，不做小样本显著性包装；正式泛化结论需要另行扩大样本量。

### 5.2 Q：可读性和状态检查

在 dev 四场景上验证 full_kv＋none，每种场景目标/锚点八选一准确率均须≥75%。未通过则本版本任务/读取设置不足，停止解释；不能删去难例。修订模板必须登记新版本。

检查来源/split/标签平衡、query 无目标值泄漏、chunk 长度、候选层完整、骨干未变、查询前后父状态哈希不变、memory_only zero/none 等价。两个 dev 组重复运行，NLL 重放差≤0.005；超限修复测量，不修改成功阈值绕过。

### 5.3 V：内容价值

分别在两种 KV 条件 dev stable 的目标问题上，从 retain/half/clear 选 p*，最大化 `min(mean(B_none),mean(B_zero),mean(B_twin))`；并列按 retain、half、clear 顺序。禁止逐来源或答案单独选策略。

两 KV 条件都要求：

- 三项平均收益各>0.05 nats/answer token；
- 至少 6/8 dev 组同时有三项收益>0。

准确率规则按条件区分：

- **memory_only**：self 八选一≥25%，比 none/zero/twin 各高≥12.5 个百分点，至少25%双生组的两目标均正确。
- **full_kv**：self 八选一和 greedy top-1 均不低于 none/zero/twin。完整 KV 可能已100%正确，不强求再提高12.5个百分点；通过时若准确率相同，结论只能是有内容依赖的概率改善，不能宣称准确率提升。

门槛比历史微小0.005差异更严格，是本轮工程标准，不重判旧结果，也非功效保证。

每个 KV 条件独立过 dev 才开启其 confirm；失败条件的 confirm 不评分。锁定 p*、代码、manifest 和完整规则后首次评估16组。门槛同上，正收益组数改为≥12/16，且每数据集≥6/8；不得在 confirm 重选策略。仅 clear 通过不能直接支持长期历史保留，必须结合 F。

### 5.4 F：保留与遗忘是否存在取舍

在 dev 对同一 KV 条件预选：retain/half 中 stable 目标 NLL 较低者 p_keep，half/clear 中 correction 目标 NLL 较低者 p_forget，并列取前者。两者相同时记为无取舍信号，不强行指定不同策略。

锁定后在获准条件的 confirm 检查：

- stable：p_keep 比 p_forget 目标 NLL 好>0.05，准确率不低；
- correction：p_forget 比 p_keep 目标 NLL 好>0.05，准确率高≥12.5个百分点，旧值误选率下降；
- 两方向分别≥12/16组 NLL 改善>0；
- correction 锚点损失必须单列。若目标更准、锚点更差，只能支持全局取舍，不能声称已解决选择性遗忘。

F 是随 V 预注册的确认分析，不进行额外搜索。V 通过/F 未通过时，优先保留有益固定策略，不据此声称自适应遗忘必要。

## 6. 条件式 S：局部动作空间

仅 V/F 均通过的 KV 条件进入 S；另外计预算，不默认包含第一批。

使用 dev 的两个局部场景、p_keep 前缀，在 chunk 6 候选已生成但未提交时分支，固定 g=1；穷举 `(aH,a2,a1)` 的0/0.5/1网格，共27动作，含3个全局动作。提交后独立评分；这是单边界短期读取诊断，不称完整多步效果。

另报同预算对照：预定局部 `(1,1,0)、(1,0,1)、(0,1,1)` 三动作对全局三个动作。27对3和3对3分别报告；全部最优值均是使用答案的事后选择。

dev 局部并集对全局平均 NLL 好>0.05、目标准确率高≥12.5个百分点、锚点准确率不降，且同预算比较有正收益，才在固定 confirm 局部场景执行同一协议。confirm 已用于 V/F，S 是预注册次级诊断，不称新盲测。

若成功，仅说明存在局部动作空间；还需不看未来答案的实际策略在另一批未见来源上验证。小样本/准确率天花板导致不能过门时，也保留连续指标，不能等同于理论否定。

## 7. 计算预算与日志

- V/F最多192会话×4策略＝**768条六块完整轨迹**；dev256，confirm最多512。没有条件通过dev时，confirm不运行。
- 写入路径每条2种KV×3种M干预，none每条2种KV，共每会话20评分分支，每分支两查询，最多**7680次单token查询前向**。
- 八选一、全词表NLL和greedy从同一次最后位置logits获得，不做八次前向。
- 背景处理量上限768×6×4096＝**18,874,368 token**，另加query；Q/smoke/repeat额外计量。
- S最多24×2局部场景×2版本×27＝**2592动作分支**，资源另计。
- 评分克隆顺序释放，双生M可暂存CPU，不长期并存多个完整KV。

旧多决策的3.594秒多为共享前缀后的分支，不能乘本轮完整轨迹数得出可靠ETA。

先用dev两个来源组（两域各一组）覆盖四场景、双生和四策略做smoke，64条完整轨迹；代码/配置完全相同且审计通过时可复用为dev结果。按每域整组墙钟、模型加载、评分与分片负载，取较慢卡预估剩余时间，加25%余量。数据下载/来源审计时间单列。

资源门槛：峰值allocated<30 GiB，V/F剩余预计墙钟≤120分钟才自动扩至允许的dev/confirm。未达门槛保留已完成结果并修订预算，不通过缩短上下文、删对照或改判据冒充完成。120分钟是预算上限，不是已测得ETA。

逐轨迹/评分干预记录CUDA同步耗时、峰值allocated/reserved、组/场景/版本/策略/KV/M来源、预测、八类分数、全词表NLL和代码/数据/模型/tokenizer哈希。分阶段记录墙钟及失败原因。缺行、重复、非有限值、来源不符或状态审计失败均禁止生成成功完成标记。

## 8. 实现清单与baseline兼容

| 待新增文件 | 内容 | 对应需求 |
|---|---|---|
| `tasks/build_cbf_memory_value.py` | 来源隔离、四场景双生构造、真值/标签审计、manifest | V/F/S数据，不预置最优动作 |
| `tasks/cbf_memory_value.py` | 四策略、KV模式、self/zero/twin、评分/选择/汇总 | 内容因果对照和确认门槛 |
| `tests/test_cbf_memory_value.py` | 真值与分组、候选时序、缓存/查询隔离、缺失结果拒绝、门槛 | 关键正确性 |
| `scripts/run_cbf_memory_value.sh` | 测试→build→smoke/Q→dev→锁定→条件式confirm→audit | 双卡、资源和完整性 |
| `experiments/.../memory_value_validation/` | 设计、选择、summary、audit、报告 | 可追溯归档 |

复用 `CBFSession._forward/commit_both/clone/clone_memory_only`、`CohortSession`、既有模型加载与显存计时。新增独立入口，不改baseline模型定义或训练配置，无必要不改runtime；预计无新增依赖。

实现注意：

- `clone_memory_only()`返回CBFSession，不带recent1/recent2，只用于不再写入的评分，不能用于继续局部更新。
- `score_answer()`内部会再克隆；新collector可在独立分支读logits，但必须验证父状态完全不变，避免多重克隆抬高显存。
- 旧builder/audit含固定16文档假设，不能直接用于本轮24组协议。
- Q检查、失败路径和确认集封存也需测试；算法实际未运行前不能填入成功标记。

拟定CLI合约（**待实现，当前不可运行**）：

```text
python -m tasks.build_cbf_memory_value --source-manifest ... --exclusion-manifest ...
    --training-data ... --tokenizer ... --dev-groups 8 --confirm-groups 16 --output ...
python -m tasks.cbf_memory_value collect --model ... --data ... --design ...
    --split dev --shard 0 --shards 2 --output ...
python -m tasks.cbf_memory_value select --design ... --inputs ... --output selection.json
python -m tasks.cbf_memory_value collect --model ... --data ... --design ...
    --split confirm --selection selection.json --shard 0 --shards 2 --output ...
python -m tasks.cbf_memory_value summarize --design ... --selection ... --inputs ... --output ...
ROOT=/home/ctj/cbf_ttt_memory_value_<run_id> bash scripts/run_cbf_memory_value.sh
```

未来入口拒绝覆盖已有ROOT，记录采集commit并用tmux运行。断线只重连检查同一任务，不因观察超时重复启动。续跑须校验代码/数据/模型/设计哈希，只跳过已审计完整行。

原文、token、逐条结果和张量留远程忽略目录；GitHub只归档代码、设计、哈希、聚合和报告，重要变更同步MODIFICATION_LOG。

## 9. 结果如何决定下一步

| 结果 | 可以支持的结论 | 后续 |
|---|---|---|
| Q失败 | 任务可读性或测量不足 | 修复并另登版本，不判记忆无效 |
| NLL好、准确率/twin失败 | 可能为通用输出偏移 | V未通过，不扩大标签 |
| full_kv有益、memory_only失败 | 依赖KV协同 | 原生设置继续，不声称独立事实存储 |
| memory_only有益、full_kv无增益 | 隔离读取存在内容记忆 | 确定有限缓存部署需求，另测应用价值 |
| V过、F不过 | 记忆有用，调节保留的需要未验证 | 优先固定策略 |
| V/F过、S不过 | 全局遗忘值得学习 | 全局控制器优先 |
| V/F/S过 | 内容价值、取舍和局部空间均有信号 | 附件第③项小规模联合训练对照 |

若V两个条件都失败，本轮结案为当前checkpoint在该任务上未验证内容记忆价值。不能否定所有TTT/遗忘机制，也不无限重复同类检查。后续可另行登记直接训练原conv/proj的有界适配，与独立控制器和联合训练比较；这是改变写入能力的新实验，不是本轮已做工作。

联合训练另需新train/dev/test、预算及梯度实现。现有inference_mode/detach推理接口不能直接拿来端到端求导。比较writer-only、controller-only、global-joint、local-joint；原lr=1e-5试跑有loss跳升，须检查实际参数位移，不能直接扩大。本轮不启动联合训练。

## 10. 交付状态与未来完成审计

本次已完成：核对旧报告与接口；固定问题、场景、来源隔离、对照、门槛、预算、实现列表和结论分支。README与修改记录同步。

未来实验须提供以下证据，不能用计划代替：

- [ ] 新来源manifest、排除/重叠审计、双生真值与split检查。
- [ ] 新入口、必要测试、真实smoke/Q、耗时显存及预算。
- [ ] dev完整结果、冻结选择；失败条件的confirm仍未评分。
- [ ] 获准条件confirm完整结果，没有重复、缺失或重选。
- [ ] V/F分别结论、全部强对照和来源/数据集分层。
- [ ] S若触发则单独记录，未触发则明确未执行。
- [ ] 骨干/父缓存/查询隔离和源码/模型/数据哈希审计。
- [ ] 最终REPORT、summary、audit、实际运行命令及MODIFICATION_LOG。

TODO：新数据、代码、测试与执行全部待下一阶段；自然事实问答、自由生成长答案、公开基准、端到端训练尚不属于已取得结果。

## 2026-10-09 实现登记（结果尚待采集）

数据构建、V/F collector/汇总、测试和双卡启动入口已实现，实际CLI见各模块 `--help`。构建器参数为 `--fineweb/--longcrawl/--training-data/--tokenizer/--old-root/--output`；collector统一 `--stage q|smoke|repeat|dev|confirm`，`summarize`处理对应阶段，不再单设select命令。新增独立CPU审计依赖 `pip install -r requirements-memory-value.txt`，用于高效扫描1B文本。FineWeb原始分片固定revision，候选按行组倒序读取。

先运行完整开发集Q可读性，再进入两个来源组smoke及重复核验；Q未过则按原门槛停止，不做writer对照或confirm。为避免复杂复用逻辑，smoke/repeat额外计量，正式dev完整重跑；资源预测按每卡剩余12组而非11组保守估计。所有其他场景、对照和门槛保持不变。S若触发须单独实现/计量，当前未执行。

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
# 使用全新ROOT，脚本拒绝覆盖已有目录
ROOT=/home/ctj/cbf_ttt_memory_value_20261009 bash scripts/run_cbf_memory_value.sh
```

## 2026-10-09 执行结果

本轮11:06:17按Q门槛正常终止：64/64个dev会话齐全；四场景目标准确率31.25%/100%/56.25%/62.5%，三类不足75%。确认集128会话未评分，V/F/S和联合训练未开展。这不是记忆价值失败结论；完整数据审计、资源和解释见[报告](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/memory_value_validation/REPORT.md)。未修改门槛或删除难例。
