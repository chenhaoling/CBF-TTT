# In-Place TTT自然文本门控v2可行性审查与执行附录

状态：**在任何v2门控数据、训练或测试结果产生前冻结**  
冻结日期：2026-10-19（Asia/Shanghai）  
用户附件SHA256：`0f96fd05a4c7c93ca99bc33c1a9e2ea83e8b8367c6293f5b843e7d7474ddfc8b`  
附件版本：`InPlace_TTT_Experiment_Plan_zh.md` v2.0，共492行、27,970字节

## 1. 审查结论

附件提出的自然文本NLL/PPL主线、A0/A1/A2分离、文档级留出、重复/干扰配对诊断、后续效用分析、
三seed和父文档级bootstrap在研究上可行。当前仓库能够复用Qwen3-4B-Base、原生`ttt_conv`/`ttt_proj`、
FSDP2训练、TTT推理cache、checkpoint转换、资源记录和独立审计。

方案不能原样立即启动，必须先完成下列修正：

1. 用户与导师已经确定“先复现In-Place TTT baseline，再探索是否更新”。附件把RULER降为可选且立即训练
   A0/A1/A2，与该顺序冲突。因此正式baseline的RULER保持必做；门控实验不得提前产生开发集或测试集结果。
2. 当前公共语料builder把多个原始文档packing成32K/128K的`content_split`，训练入口没有`doc_id`或
   `reset_mask`。它适合匹配baseline，但不满足附件要求的文档级状态重置、split继承和独立文档统计单位。
3. 官方训练模型在单条序列内使用chunk更新的`cumsum`，梯度可以从后续chunk回到早期更新；官方推理模型才
   通过`TTTDynamicCache.ttt_states`保存跨调用`past_w`。因此A2可以在训练模型的chunk增量上实现，但不能把
   推理cache语义直接当作训练时跨样本memory。
4. 当前source inventory只有训练来源，尚无冻结的gate train/dev/test文档清单、跨来源近重复簇或独立NLL
   目标。必须先按父文档划分，再构造窗口和重复/干扰变体。
5. sigmoid接近0不会减少更新矩阵计算。首轮只主张预测质量和写入量控制；计算加速必须有真实skip分支实测。

结论标签为：**设计可行，baseline立即执行，门控实现与评分在新增P0审计及baseline通过后执行。**

## 2. 已核验的官方与代码事实

- FineWeb-Edu官方数据卡提供网页文本、文档`id`和教育质量`score`。该score只描述语料筛选，不作为门控
  oracle或“应该写入”的标签。
- LongCrawl64官方说明每个发布文档为65,536个GPT-2 TikToken token，并执行截断、文档打乱和随机循环位移。
  当前builder只把合法GPT-2 IDs解码为文本，再使用固定Qwen3 tokenizer重新分词；不得直接送入原token IDs，
  也不得把循环顺序解释为事实时间线。
- 官方In-Place-TTT仓库要求用户提供`content_split` plaintext语料，推荐`ttt_chunk=4096`、LM-aligned
  chunk更新、DCP转换与RULER评测。仓库没有发布论文私有训练数据处理脚本。
- 本地`hf_models/hf_qwen3/modeling_qwen3.py`在训练forward内构造每chunk的`d_down_proj`并`cumsum`；第t个
  chunk只使用此前更新。`inference_model/hf_qwen3/modeling_qwen3.py`通过`TTTDynamicCache`延续`past_w`。

## 3. 冻结后的执行顺序

### B0：先完成初步baseline复现

1. 使用opt-in `preliminary_100m` profile构建并独立审计FineWeb-Edu/LongCrawl64等token语料：
   Stage 1为56,623,104 tokens（32K、27个GB64 step），Stage 2为41,943,040 tokens（128K、5 step）。
2. 在`hku-gpu2`执行corrected Qwen3-4B-Base Level-A：no-TTT与native-TTT各一个真实optimizer step，保存、
   转换、重载并完成RULER smoke；沿用已经冻结的显存、主存和swap门槛。
3. 完成N0（no-TTT）与A0（官方native TTT）的98,566,144-token配对训练、独立checkpoint审计和官方
   13-task RULER 4K--128K矩阵；256K保持单独资源门槛。
4. 只把该结果称为“初步公开数据匹配复现”，不声称论文私有数据或论文规模复现。

当前约35B-token full builder可以继续提供未来scale-up语料，但不是B0启动条件。缩小语料通过后，在GPU运行前
保存而不删除full partial，并停止其CPU进程，避免资源门槛互相干扰。

### P0：门控前的数据、边界和梯度审计

P0可以在B0运行期间开发和测试，但不得打开门控dev/test评分。

1. 从原始父文档开始做确定性train/dev/sealed-test划分；所有窗口、改写、重复和干扰变体继承父文档split。
2. 训练数据使用带`doc_id`和`reset_mask`的boundary-aware packing；attention与TTT累计更新都必须在文档边界
   重置。若该实现未通过测试，则退回“一条训练记录只含一个父文档”，不得用普通packing冒充文档隔离。
3. 记录精确重复与跨来源重复；近重复聚类算法、阈值和hash在看到测试结果前冻结。预训练模型不可控的历史
   污染与本项目split泄漏分开报告。
4. 单元与tiny-model测试必须证明：关闭新增逻辑逐位复现A0；`gate=0`不改变fast update；`gate=1`复现A0；
   chunk t的目标不受chunk t自身更新；后续chunk loss对门控参数有finite nonzero gradient；换文档后KV、fast
   state和控制器状态均重置；B评分期间禁止写入；sealed-test不能被训练/选参入口加载。
5. P0清单记录模型/tokenizer revision、来源revision、Qwen重分词、原始文档长度、packing、两源比例、实际
   tokens、边界数量、split与重复簇、吞吐和峰值显存。任何一项缺失都阻止P1。

### P1：最小自然文本门控比较

只有B0与P0均通过后启动。第一轮固定32K和单seed 42进行工程排错；每个训练arm精确消费56,623,104个
有效token和27个GB64 step：

| Arm | 定义 | 作用 |
|---|---|---|
| N0 | no-TTT，同骨干、数据和训练预算 | 排除额外训练收益 |
| A0 | 原生TTT更新，gate恒为1 | 直接baseline |
| A1 | 冻结的常数gate/更新尺度网格 | 排除单纯缩小步长 |
| A2 | 当前chunk或冻结局部窗口输入的两层MLP标量gate | 检验输入相关写入 |

A0/A1/A2必须共用原生更新位置、`ttt_conv`、`ttt_proj`、chunk=4096和相同训练数据。A2在推理决策时只使用
当前已到达chunk/冻结局部窗口；future loss只提供训练梯度，不进入feature或artifact。A1与A2对更新尺度使用
同等验证预算。每个方法最多6次等长短程搜索，验证集选择后冻结一次性test配置。

主评测是FineWeb-Edu与LongCrawl64分别报告的文档隔离NLL/PPL、token加权两源结果、文档宏平均、位置分桶、
相同目标不同历史长度曲线、gate/更新/fast-weight范数、tokens/s、wall time和峰值显存。每源先用200个dev父
文档检查管线；正式test目标为每源1,000个独立父文档，样本不足时报告实际数量且不重复窗口充数。

单seed只能排错。论文主结果对冻结配置运行42、123、2026三个seed，并同时报告训练seed波动和父文档分组
bootstrap区间。

### P2/P3：状态、残差和机制诊断

只有A2相对调好尺度的A1有稳定开发集收益时才进入：

- A3：显式状态输入；D0：无门控残差更新；按主张需要补D1/D2。
- 各200个父文档的原文重复与无关/相似主题干扰pilot，固定B和总token预算；正式扩到各500个父文档。
- 从相同更新前状态分叉update/skip，在独立train/dev材料上计算
  `U=NLL_skip(B)-NLL_update(B)`；测试效用只作冻结后分析，不能选阈值。
- 必做常数gate、norm规则、gate打乱、移除状态、暂停写入、移除fast state对照。

事实修订、LongMemEval或显式遗忘保护属于后续P5/P6。自然文本NLL或重复实验不能直接支持“识别过时事实”或
“选择性遗忘”的主张。

## 4. 停止条件与claim boundary

- baseline数据/hash/资源/转换/RULER任一门槛失败：停止B0，不打开P1评分。
- P0任一因果、reset、gradient或sealed-test检查失败：停止门控训练。
- A2不优于调好尺度的A1：先报告输入门控未显示独立收益，不升级A3。
- A3不优于A2：报告显式memory state未显示独立收益，不声称状态感知机制有效。
- 只优于A0但不优于N0：只能说明减少了有害更新，不能声称native writer产生净正收益。
- gate未实际跳过算子：不得声称推理加速。

所有新增路径必须opt-in；baseline配置、训练入口、writer与checkpoint均不可覆盖。所有GPU运行仍只允许在
`hku-gpu2`。

## 5. 当前立即动作

1. 构建并独立审计`preliminary_100m`语料，记录wall time、RSS和磁盘。
2. 缩小语料通过后保存full partial并释放builder资源。
3. 执行corrected Base Level-A与B0配对训练/RULER。
4. 同期只实现P0的文档split、boundary/reset与因果梯度测试；baseline未通过前不训练A1/A2、不打开门控test。

## 6. 官方参考

- FineWeb-Edu数据卡：<https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>
- LongCrawl64说明：<https://manifestai.com/articles/longcrawl64/index.html>
- In-Place-TTT官方仓库：<https://github.com/ByteDance-Seed/In-Place-TTT>
- RULER官方仓库：<https://github.com/NVIDIA/RULER>
