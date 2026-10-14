# CBF-TTT 实现记录

## 2026-10-14：Post-V20 跨来源语义控制器数据清单（前置准备）

在 V19 正式复现写出 148/320 行、且尚未读取任何 arm 汇总、因果规则或 V19/V20 结果时，固定并实现跨来源语义控制器的数据清单与泄漏审计。新增 `tasks/build_semantic_controller_source_inventory.py`、`scripts/audit_semantic_controller_source_inventory.py`、`scripts/run_semantic_controller_source_inventory.sh` 和 `tests/test_semantic_controller_source_inventory.py`。构建器只使用已观察的 CounterFact、MQuAKE-T 固定 train+tune，以及已物化的 ZsRE calibration；明确不打开 ZsRE dev/test、MQuAKE-CF、LoCoMo、LongMemEval-S 或模型 outcome。关系先确定五折；同一规范化主体若跨折，只保留其记录最多的一折，并逐行保存排除原因。关系表面在编码前对精确主体和值做掩码。

本地及 hku-gpu2 四项测试通过。真实清单源哈希分别为 CounterFact `d0170561...fc6f`、MQuAKE-T `58500a1d...9f2`、ZsRE calibration `d19bb92c...ec6f`；23,413 条候选语义记录中保留 22,496 条，排除 917 条跨折主体冲突和全部 374 条 MQuAKE dev；ZsRE calibration 因主体重叠排除 7 条，保留 260 条。五折同时满足 relation-disjoint、subject-disjoint 和非空，独立重算审计全部通过，manifest SHA256 为 `df3002f4913663ef829b865b3d1502305b712f5a96c401568c413a9fa2862836`。小型记录位于 `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/semantic_controller_source_inventory_v1/`，完整数据只保留服务器 `/home/ctj/cbf_ttt_semantic_controller_inventory_20261014`。

清单同时揭示不可消除的来源失衡：MQuAKE train+tune 的 1,494 条记录中 P6=995、P35=481，其余五个关系合计18；五折 MQuAKE 行数为790/10/7/445/1。不能在保持关系隔离时伪造来源均衡，因此在结果揭示前要求使用下述频次温度权重，并报告 micro、subject-macro、relation-macro 与分来源指标，不复制关系或样本。该清单不是效果证据，只有 V19/V20 前置门槛通过后才允许训练语义控制器；所有改动均为 opt-in 数据工具，不改变 baseline、原生 `ttt_proj/ttt_conv` 或默认推理。

随后新增 `tasks/prepare_semantic_controller_balancing.py`、`scripts/audit_semantic_controller_balancing.py` 与 `tests/test_semantic_controller_balancing.py`，把五折 outer/inner/train 角色和频次权重固定为可复算工件。首次完全等关系/共享来源权重仅使用计数，但真实清单上将约13,500条训练记录的有效样本量降至239--1,534，最大单条权重560；因此在任何模型 outcome 可见前明确拒绝。最终 v2 使用 relation-source group 频次 `n^-0.5`、均值归一化、每行每epoch一次；五折有效样本量10,739--12,458（80.35%--92.20%），最大权重25.29--25.81，全部资源门槛和独立逐行重算审计通过，manifest SHA256 为 `9eed136cb71ba92df18252dc80232bf691a5e0337ae88208babef33f2c4d795c`。反序输入测试还发现普通浮点求和的约1e-15顺序差，现按row ID排序并用`math.fsum`消除，保证逐字节复现。完整权重保留服务器 `/home/ctj/cbf_ttt_semantic_controller_balancing_sqrt_20261014`，compact manifest/audit 已归档。

## 目标与方法理解

本次实现依据工作区的 `../CBF_TTT_method.md`，在原版 In-Place TTT 推理模型之上加入**可选**的反事实收益驱动连续遗忘。基础 down-projection 权重、注意力权重和更新投影保持冻结；单次会话的快记忆为 `M`。对一个完整 chunk，先用 `W0 + M` 产生输出与原 TTT 候选 `ΔW`，再用共享的 `alpha` 提交 `M <- (1-alpha)M + ΔW`。`alpha=0` 为相同 chunk 协议下的标准累积更新。

以下是对设计稿中未指定实施细节的明确假设：

- 仅在推理期的 Qwen3/LLaMA3 TTT 层启用 CBF；原有 VeOmni 预训练不加入控制器参数。选中层直接采用 checkpoint 的 `ttt_layers`，所有层共享一个系数。
- 实验输入是**已分词 token ID** 的 JSONL，避免训练和评估因模板或 tokenizer 默认行为而改变边界。`future` 的 `continuation_ids` 拼接在当前样本剩余的 `context_ids` 后。
- 在没有外部语料的情况下，新增的构建器产生可复现的**受控合成事实账本**，覆盖稳定事实、事实更正、主题切换，以及低/高干扰。它用于验证方法流程与受控对照；真实任务效果仍需另备有来源的事实记录或基准数据。
- 当前资源试点使用官方 Qwen3-4B 主干，并显式将新加 TTT depthwise conv 初始化为零，以保留候选计算形状而使随机更新不改变输出。它只用于计时与显存容量决策，不能证明方法收益。
- 离线候选网格默认 `0,0.25,0.5,0.75,1`；数值并列容差默认 `1e-6`，并列取最小 `alpha`。后续参考策略固定为 `alpha=0`；初始状态采集策略也是 `alpha=0`，可改用训练后的控制器重采样。
- 一个 future 可有多个查询；候选损失对所有 future 的所有查询平均。每个查询从该候选分支的同一末状态复制 cache 独立评分。
- 上下文最后不足一个 chunk 的尾部只更新注意力 KV，不写入快记忆。查询与答案评分也只追加 KV，不写入快记忆。
- 推理及反事实评估使用 `model.eval()`、冻结参数和确定性 teacher forcing；因此分支复制完整 KV 与快记忆状态即可，不消耗随机采样状态。

## 仓库扫描与核心文件

| 职责 | baseline 文件 | 本次处理 |
|---|---|---|
| 预训练模型、TTT loss 与 chunk 更新 | `hf_models/hf_qwen3/modeling_qwen3.py`、`hf_models/hf_llama/modeling_llama.py` | 保持原样 |
| 推理模型、TTT cache 与生成 | `inference_model/hf_qwen3/modeling_qwen3.py`、`inference_model/hf_llama3/modeling_llama.py` | 加入 cache 开关控制的 CBF 分支 |
| TTT 模型配置 | 两份 `inference_model/*/configuration_*.py`、`configs/pretrain/*.yaml` | 保持原样，读取现有 `ttt_layers/ttt_chunk/ttt_target/ttt_lr/ttt_proj` |
| 数据、dataloader、optimizer、scheduler、训练 loss | `tasks/train_torch.py`，由 VeOmni 提供相关构造器 | 保持原样；控制器另行监督训练 |
| 原交互推理、验证与测试 | `tasks/infer.py`、`eval.sh`、`eval_config/` | 保持原样；CBF 采用独立实验入口 |
| 原日志与保存 | `tasks/train_torch.py` 的 logger、WandB 与 DCP/HF 保存 | 保持原样；CBF 保存 JSONL 损失表与 `.pt` 控制器 |

原仓库没有本方法所需的未来情景/标准答案数据、已适配的 HF TTT 权重或可直接复用的 CBF 标签生成器。本次新增场景构建器与标签生成器，可先产生受控合成数据；真实来源记录可经 `--sources` 输入。checkpoint 应先按照 baseline 的 `scripts/merge_dcp_to_hf.py` 转成 HF 格式。

## 修改文件清单与具体内容

| 文件 | 修改内容 | 对应方法章节 | 对 baseline 的影响 |
|---|---|---|---|
| `inference_model/hf_qwen3/modeling_qwen3.py` | 在 TTT decoder 的 MLP 处增加 `cbf_enabled` cache 分支 | 3.2、3.3、3.7 | 默认 cache 无此开关，原路径不变 |
| `inference_model/hf_llama3/modeling_llama.py` | 与 Qwen3 相同的可选分支 | 3.2、3.3、3.7 | 默认原路径不变 |
| `cbf_ttt/runtime.py` | `cbf_forward_mlp`、`CBFSession`、chunk 特征、快记忆提交、独立查询评分与贪心生成 | 3.3、3.4、3.7 | 新增独立模块 |
| `cbf_ttt/controller.py` | `ForgettingController`，语义投影、标量标准化、MLP 回归及端点裁剪 | 3.4、3.6 | 新增独立模块 |
| `cbf_ttt/experiment.py` | 场景校验、候选分支标签生成、训练/开发组隔离、控制器训练与完整 rollout 评估；逐条记录同步计时和 CUDA 峰值显存，并汇总 profile；非有限损失给出明确错误 | 3.5、3.6、3.8；小规模成本试点 | 新增独立模块 |
| `cbf_ttt/__init__.py` | 包入口 | 全部 | 新增 |
| `tasks/build_cbf_scenarios.py` | 按源组划分数据，构造稳定/更正/主题切换、历史/新事实问答与不同噪声的预分词场景；保存构建元数据 | 3.5.2、3.5.3、3.6.3 | 新增独立数据入口 |
| `tasks/cbf_ttt.py` | `collect/train/eval/generate` 命令行 | 3.5–3.7 | 新增独立实验入口 |
| `scripts/make_tiny_cbf_fixture.py` | 建立随机小型 Qwen3 TTT 模型及 tokenizer 文件，用于无正式权重时检验代码链路；不用于效果结论 | 验证 3.3–3.7 | 新增验证辅助脚本 |
| `scripts/download_qwen3_4b.py` | 从官方仓库按固定 revision 下载 Qwen3-4B base 的模型与 tokenizer 文件 | 用户指定的 4B 实验准备 | 新增下载辅助脚本；不影响 baseline |
| `scripts/make_resource_cbf_fixture.py` | 基于 HF Qwen3/LLaMA 分片软链接建立同参数规模的资源夹具；将 TTT conv 零权重写入独立 safetensors 分片和索引，避免缺失权重的非有限初始化 | 小规模资源试点 | 新增验证辅助脚本；不影响 baseline |
| `tests/test_cbf_ttt.py` | 验证更新形状、先输出后写入、投影与 bias、控制器区间外梯度、词面特征、两种小型模型会话，以及标签 profile 字段和摘要 | 3.3、3.4、3.6、3.7 | 新增测试 |
| `tests/test_cbf_scenarios.py` | 验证固定 token 边界、查询完整性、组间隔离与随机种子复现 | 3.5.2、3.5.3 | 新增测试 |
| `experiments/cbf_ttt/pilot_20260923/label_profile.jsonl` | 从远程完整标签提取 9 条逐标签耗时、起始/峰值/增量显存、chunk 长度和边界，不含庞大的特征向量 | 小规模资源试点实测记录 | 新增实验数据；不影响 baseline |
| `experiments/cbf_ttt/qwen3_4b_pilot_20260923/label_profile.jsonl` | 官方 Qwen3-4B 权重的 10 条逐标签计时/显存记录，覆盖 512/4096 chunk 和两张 5090 | 用户指定的 4B 资源试点 | 新增实验数据；不影响 baseline |
| `MODIFICATION_LOG.md` | 本记录 | 用户交付要求 | 新增文档 |

## 模块逻辑与需求对应

1. **3.3 候选更新：**`cbf_forward_mlp` 沿用原 MLP 的 gate/up/down、depthwise `ttt_conv`、可选 `ttt_proj.weight` 存储方向和 `ttt_lr`。当前输出先基于旧 `M` 计算，候选只写入 cache 的 `cbf_candidates`；整个模型当前 chunk 前向完成后 `CBFSession.commit` 才一次性更新各层。若有 down-projection bias，输出包含该固定 bias。
2. **3.4 状态特征：**`CBFSession.observe` 提取最后隐藏状态均值、当前 chunk 内排除首 token 的平均 NLL、token 熵/唯一比例/bigram 重复比例，以及每层旧记忆与候选相对范数和方向余弦。词表评分按 128 token 块执行，减少峰值 logits 内存。控制器只接收这些当前已知量。
3. **3.5 反事实：**`collect_labels` 在选中 chunk 边界复制已包含该 chunk KV 与候选的 cache；每个候选独立提交，读取相同剩余上下文与 future，按固定 `alpha=0` 参考策略更新，查询从末状态再次独立复制。保存完整候选损失、相对不遗忘收益及最优标签。
   - 每条标签另外记录 `label_time_s`、`start_allocated_gib`、`peak_allocated_gib`、`peak_reserved_gib`、`extra_peak_allocated_gib`。计时从选中边界的 `observe` 前开始，到全部候选及 future 查询评分结束；CUDA 在起止点同步并重置峰值统计。未选中边界的处理、写 JSONL、后续参考轨迹提交、模型加载不计入该条时间。CPU 的显存字段为 `null`。CLI 写入 `labels.jsonl.summary.json`，包括均值、中位数、p95、最大耗时和最高绝对峰值。峰值显存是整张进程 GPU 的绝对值，含模型常驻参数；增量峰值相对该标签开始时的分配量。
4. **3.6 控制器：**标量均值/标准差仅由训练集计算，语义投影与回归器训练时使用未裁剪分数 MSE；部署预测裁剪到闭区间 `[0,1]`。仅控制器参数参加 AdamW 优化，开发集 MSE 选择保存轮次，并将各轮训练/开发 MSE 写入 `.metrics.json`。使用 `collect --state-policy controller` 与再次 `train` 可完成一次或多次状态再采样。
5. **3.7 在线：**每个完整 chunk 一次 backbone 前向。会话在 context 尾部停止写入，查询/生成只读快记忆；新 `CBFSession` 建立新的 KV/快记忆。`eval` 对每个未来情景与策略分别开始独立会话，输出逐查询损失与每块系数，另存 `.summary.json`，包含与不遗忘基线比较的收益及有害遗忘比例。
6. **3.5 数据构建：**`tasks/build_cbf_scenarios.py` 使用 checkpoint tokenizer 在完整 chunk 内保留事实头部，再填充低/高干扰文本到精确 token 边界；上下文含已知事实与可见的状态提示，后续情景引入稳定确认、事实更正或新主题。每个 future 同时包含历史事实与当前/新事实查询。按 `group_id` 一次性划分训练、开发和测试，保存 seed、模板版本、边界及组清单到 `.meta.json`；后续答案不写入当前 chunk 的控制特征。

## 输入格式

每行一个 JSON 对象，`id` 唯一；相同源文档/会话/模板家族使用相同 `group_id`，它只能出现在一个 `split`。`context_ids` 是查询前的已有上下文；每个 `future` 表示同一评估协议下的一个额外后续上下文和一个或多个答案评分任务。实际使用时需提供非空 `query_ids` 和 `answer_ids`，下例仅展示结构：

```json
{"id":"s1","group_id":"document-A","split":"train","context_ids":[101,102,103,104],"futures":[{"continuation_ids":[105,106],"queries":[{"query_ids":[201,202],"answer_ids":[301,302]}]}]}
```

构建器生成的输入已经按源组划分 `train/dev/test`；外部数据也须按文档/会话/模板家族分组。future 同时包含历史检索与新信息问题。答案 token 的首个 token 由 query 的最后一个位置预测。若 prompt/answer 需要空格或特殊分隔符，须在 `query_ids`/`answer_ids` 中明确编码。`context_ids` 长度不足 `ttt_chunk` 时没有控制决策；它仍可参加完整 rollout 评估。

可选的 `--sources` JSONL 每行须包含 `group_id`、`entity`、`original_value`、`updated_value`、`new_entity`、`new_value` 六个非空字符串。每行代表一个独立源组；同一源组的多个模板变体保留在同一 split。`original_value` 与 `updated_value` 必须不同。构建器默认直接生成合成源组，无需此文件。噪声和查询均使用统一模板，研究者应另行检验真实语料、词表和模板偏差。

## 运行命令

以下命令从 `In-Place-TTT/` 执行。先安装 baseline README 中的 PyTorch、Transformers 4.57.3、einops、opt_einsum 等依赖并准备 HF 格式 TTT checkpoint 与 JSONL 场景。示例中的模型、数据和输出路径需自行替换。

```bash
# 从受控事实源构建分组场景；chunk-size 必须匹配 checkpoint 的 ttt_chunk
python -m tasks.build_cbf_scenarios --tokenizer /path/to/ttt_hf --output /path/to/scenarios.jsonl --chunk-size 4096 --train-groups 100 --dev-groups 20 --test-groups 20 --variants-per-group 1 --context-chunks 4 --future-chunks 1 --futures-per-scenario 1 --seed 42

# 如已有结构化事实源，可加入 --sources /path/to/source_facts.jsonl

# 初始 baseline 状态的反事实训练/开发标签
python -m tasks.cbf_ttt collect --model /path/to/ttt_hf --data /path/to/scenarios.jsonl --split train --output /path/to/train_labels.jsonl --grid 0,0.5,1 --every 2
python -m tasks.cbf_ttt collect --model /path/to/ttt_hf --data /path/to/scenarios.jsonl --split dev --output /path/to/dev_labels.jsonl --grid 0,0.5,1 --every 2

# 单独训练控制器；层 ID 必须与 checkpoint 的 ttt_layers 一致
python -m tasks.cbf_ttt train --train-samples /path/to/train_labels.jsonl --dev-samples /path/to/dev_labels.jsonl --layers 0 6 12 18 24 30 35 --output /path/to/controller.pt

# 可选：用冻结控制器重采样训练轨迹，再合并初始与重采样标签训练
python -m tasks.cbf_ttt collect --model /path/to/ttt_hf --data /path/to/scenarios.jsonl --split train --state-policy controller --controller /path/to/controller.pt --output /path/to/resampled_train.jsonl --grid 0,0.5,1 --every 2
python -m tasks.cbf_ttt train --train-samples /path/to/train_labels.jsonl /path/to/resampled_train.jsonl --dev-samples /path/to/dev_labels.jsonl --layers 0 6 12 18 24 30 35 --output /path/to/controller_v2.pt

# 独立完整会话测试；baseline 与控制器使用相同 chunk/尾部协议
python -m tasks.cbf_ttt eval --model /path/to/ttt_hf --data /path/to/scenarios.jsonl --split test --controller /path/to/controller_v2.pt --policies baseline controller --output /path/to/test_rollouts.jsonl

# 可选：测试状态的候选损失表仅用于诊断，不可加入控制器训练
python -m tasks.cbf_ttt collect --model /path/to/ttt_hf --data /path/to/scenarios.jsonl --split test --output /path/to/test_oracle_diagnostic.jsonl --grid 0,0.5,1 --every 2

# 用训练好的控制器适应上下文并贪心生成（也可加 --tokenizer）
python -m tasks.cbf_ttt generate --model /path/to/ttt_hf --controller /path/to/controller_v2.pt --context-ids '[101,102,103,104]' --query-ids '[201,202]' --max-new-tokens 64

# 张量单元测试
python -m unittest discover -s tests -p 'test_cbf_ttt.py'

# 场景构建测试（仅使用 Python 标准库）
python -m unittest discover -s tests -p 'test_cbf_scenarios.py'

# 无正式 TTT 权重时，构造随机小型模型，只用于 CPU/GPU 功能链路检查
python -m scripts.make_tiny_cbf_fixture --tokenizer-source /path/to/tokenizer --output /path/to/tiny_ttt_hf --chunk-size 128
python -m tasks.build_cbf_scenarios --tokenizer /path/to/tiny_ttt_hf --output /path/to/tiny_scenarios.jsonl --chunk-size 128 --train-groups 1 --dev-groups 1 --test-groups 1 --variants-per-group 1 --context-chunks 2 --future-chunks 1 --futures-per-scenario 1
```

构建器的 `--tokenizer/--sources/--chunk-size/--context-chunks/--future-chunks/--futures-per-scenario/--variants-per-group/--seed` 控制来源、长度与复现；三个 `--*-groups` 决定按源组划分比例。`--grid` 控制离线候选，必须包含 `0,1`；`--every` 控制采样边界间隔；`--tie-tolerance` 控制数值并列。`--state-policy` 可选 `baseline/controller`。`train` 的 `--width/--semantic-size/--epochs/--batch-size/--lr/--seed` 控制网络与优化。`--device/--dtype` 控制加载位置与精度。`eval --policies` 可包含 `baseline`、`controller` 或具体常数系数（如 `0.5`）。`collect` 自动保存逐条耗时/峰值和 `.summary.json`；不需要额外开关。这些开关仅属于新入口，原始 baseline YAML、`train.sh`、`eval.sh` 不受影响。

## 完成情况、验证与风险

- **已完成：**Qwen3/LLaMA3 可选 CBF 前向、单系数记忆提交、特征提取、有限网格反事实监督、逐标签耗时及显存 profile、控制器训练、状态再采样、完整会话 NLL 评估、贪心生成、受控场景构建、数据组校验与单元测试源码。
- **本地验证：**已运行 Python 语法编译、命令行 `--help` 解析、构建器标准库测试及 `git diff --check`。本地系统 Python 为 3.9，低于原仓库要求的 3.11，且没有安装 `torch`/`transformers`，因此不能在本机运行张量测试或真实 checkpoint。
- **内存风险：**每个选中层保存一份 down-projection `M` 和一份候选，反事实分支还复制 KV；长上下文和多候选会显著增加显存。可先调大 `--every`、缩短情景、分批收集。Qwen3-4B 的 4096-token 试点达到 19.80 GiB 保留峰值；已训练 checkpoint 与更多 future/query 仍需重新测量。
- **精度/维度风险：**实际 checkpoint 的 `ttt_layers`、隐藏宽度、`ttt_target` 与控制器必须一致；入口在会话及控制器加载时检查层、宽度、标量数。混合精度的多次快权重累积与 float32 可能有细微差异。
- **协议边界：**受控合成场景可验证流水线和机制，不能代替真实长上下文任务；真实语料需提供结构化事实源或扩展模板。通用 OpenCompass 默认入口不启用 CBF，方法比较请使用新 `eval` 命令。
- **后续 TODO：**在真实 Qwen3/LLaMA3 TTT 权重上跑端到端实验，测量在线延迟和峰值显存，扩展真实来源事实及跨模板评估，报告长上下文任务效果/相对候选最优差距/有害遗忘比例，并校准候选网格与超参数。

## hku-gpu2 远程验证（2026-09-23）

- **隔离位置：**`/home/ctj/cbf_ttt_verify_20260923`；tmux 会话为 `cbf_ttt_verify_20260923`，可用 `ssh hku-gpu2` 后执行 `tmux attach -t cbf_ttt_verify_20260923` 检查。上传的是不含 `.git` 和大图片的代码归档；没有修改服务器现有项目。
- **环境：**从远程 `serem_llm` 克隆独立 conda 环境 `cbf_ttt_verify_20260923`，安装 `transformers==4.57.3`、`einops==0.8.2`、`opt_einsum==3.4.0`；PyTorch 为 `2.9.1+cu130`，两张 RTX 5090 均用于后述烟测。克隆环境为 **Python 3.10.21**，低于 baseline `pyproject.toml` 声明的 Python 3.11，因此这里是源码功能验证，不能声称完成严格同版本复现。
- **张量测试：**远程 `python -m unittest discover -s tests -p 'test_cbf_*.py' -v` 共 **9 项通过**，覆盖 Qwen3/LLaMA 小模型会话、候选分支、数据构建与 `alpha=0` 同 chunk 边界下对 Qwen 原推理路径的输出等价性。记录在 `unit_tests_final.log`。
- **数据构建：**使用服务器已有 Qwen2.5 tokenizer 和随机小型 Qwen3 TTT fixture，构建了 3 个源组、每组 1 个场景，train/dev/test 各 1 个；每个 context 为两个 128-token chunk，future 为一个 128-token chunk。输出 `tiny_scenarios.jsonl` 及 `.meta.json`。fixture 由 `scripts/make_tiny_cbf_fixture.py` 随机初始化，**不是已适配训练的 TTT 权重**。
- **GPU0 float32 链路：**`collect` 生成 train 1 条、dev 1 条标签；`train` 保存控制器；`collect --state-policy controller` 再生成 1 条重采样标签；合并重训、`eval`、`generate` 均以退出码 0 结束。测试集单情景的 baseline 平均 NLL 为 `11.927899`，控制器为 `11.927894`。固定系数 `0/0.5/1` 的 rollout 均成功，且结果有限。各步日志分别在 `collect_train.log`、`collect_dev.log`、`controller_train.log`、`resample.log`、`controller_retrain.log`、`rollout_eval_final.log`、`rollout_eval_fixed.log` 和 `generate_full_chunk.log`。
- **GPU1 bfloat16 链路：**测试集 `eval` 与包含 `0/0.5/1` 的离线 `collect` 均以退出码 0 结束；记录在 `rollout_eval_bf16.log` 和 `collect_bf16.log`。baseline 与控制器的单情景 NLL 分别约为 `11.928050` 和 `11.928045`。
- **解释边界：**只有每 split 一个随机模型场景，开发 MSE 达 `116512.40625`，说明样本量不足且模型没有学到任务。上述极小 NLL 差异、`harmful_forgetting_fraction=0` 都只是功能烟测输出，不是 CBF-TTT 的有效性证据。此后已按用户要求下载官方 Qwen3-4B base，但仍无已适配的 In-Place TTT checkpoint 或真实来源事实数据。真实效果实验需补齐这些资产，并在 Python 3.11/目标依赖环境中重跑。

## Qwen3-4B 三点网格与稀疏边界资源试点（2026-09-23）

根据用户的模型选择，已从[官方 Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) 下载完整 base 权重到远程 `/home/ctj/models/Qwen3-4B`，下载日志 `download_qwen4b.log` 显示退出码 0；模型 revision 为 `1cfa9a7208912126459214e8b04321603b3df60c`。原仓库的 Qwen3 推理类可加载其架构。资源夹具选 7 个有效 TTT 层 `[0,6,12,18,24,30,35]`；官方 4B 为 36 层，baseline 8B 示例中的层号 `36` 对它越界。测试使用 `bfloat16`、一个 future chunk、每 future 两个查询、网格 `{0,0.5,1}`、`--every 2`。512 与 4096 场景均有 4 个 context chunk，选中第 2 和第 4 个边界。

| chunk token 数 | 场景与 GPU | 标签数 | 单条平均 / 最大耗时（秒） | 最高分配 / 保留峰值（GiB） |
|---:|---|---:|---:|---:|
| 512 | train 2 场景，GPU1 | 4 | 0.541 / 0.739 | 10.211 / 10.426 |
| 4096 | train 1 场景，GPU1 | 2 | 6.137 / 8.154 | 17.141 / 19.803 |
| 4096 | dev 1 场景，GPU0 | 2 | 6.378 / 8.486 | 17.142 / 19.803 |
| 4096 | test 1 场景，GPU1 | 2 | 6.151 / 8.187 | 17.142 / 19.803 |

4096 的第 2 边界比第 4 边界慢：train 为 `8.154/4.120` 秒，dev 为 `8.486/4.270` 秒，test 为 `8.187/4.115` 秒，主要因为前者的三个候选还要处理剩余两个 context chunk。两张卡的 4096 保留峰值均约 `19.80 GiB`。完整标签、汇总与运行日志保留在远程 `qwen4b_labels_512.jsonl`、`qwen4b_labels_4096*.jsonl` 及对应 `.summary.json`、`qwen4b_collect_*.log`；10 条精简逐标签记录同步到仓库 `experiments/cbf_ttt/qwen3_4b_pilot_20260923/label_profile.jsonl`。四个 `collect` 均退出码 0。另用 `/usr/bin/time` 实测 test 场景从启动独立进程、模型加载至输出完成共 `17.83` 秒，其中两条标签计时合计 `12.30` 秒；这是一个场景的完整墙钟时间。三个现有 4096 场景的 JSONL 分词构建单独实测 `2.46` 秒（含进程启动），不包含 GPU 标签生成。

计时从选中边界的 `observe` 前开始，到全部候选及 future 查询评分结束，CUDA 两端同步。计时不含模型加载、构建场景、未选中 chunk、文件写入和控制器训练。显存为进程绝对峰值，含常驻权重。零权重 TTT conv 保证下载的 Qwen3-4B 不因未训练的附加层而出现非有限更新；新增投影仍未经训练，三个候选的损失相同。**这只能验证计算与资源，不能评价最优 alpha、控制器或遗忘收益。**

先前保留的 `experiments/cbf_ttt/pilot_20260923/label_profile.jsonl` 是已废弃的 7B 资源试点历史记录；其耗时、显存和原 1024-token 规模建议均不用于现在的实验决策。现行资源夹具同时支持 Qwen3/LLaMA；初版缺失 TTT conv 初始化可能产生 `NaN`，现已通过独立 safetensors 零权重分片修正。标签器遇到非有限损失会抛出包含场景与边界的明确错误。

### 第一阶段正式规模决定

以**经过 In-Place TTT 训练、架构与本试点相同、`ttt_chunk=4096` 的 Qwen3-4B checkpoint**为条件，第一阶段安排 train/dev/test `100/20/20` 个源组、每组 1 个场景、4 个 context chunk、1 个 future chunk、每 future 2 个查询、每隔 2 个边界取标签。train/dev 分别得到 200/40 条；可选的 test oracle 诊断再得到 40 条，总计 280 条。6 条 4096 标签的平均计算时间约 `6.22` 秒，据此估算全部 280 条选中边界约 **29 分钟**；按照已测最大单条 `8.49` 秒估算约 **40 分钟**。这些估计不含未选中边界、模型加载、训练、评估和 I/O；在两张卡并行分 split 可缩短墙钟时间，但须保留各自的显存余量。

目前服务器只有官方 Qwen3-4B **base** 权重，没有已经训练的 4B In-Place TTT checkpoint，因此正式效果采集尚未启动。正式 checkpoint 若层数、chunk、精度或 future/query 数变化，先复测 1–2 个场景的显存，再确认本规模。合成账本支持受控机制测试；实际结论还需真实来源事实与任务集。

Qwen3-4B 资源试点复现命令（从远程项目目录运行）：

```bash
python -m scripts.download_qwen3_4b --output /home/ctj/models/Qwen3-4B
python -m scripts.make_resource_cbf_fixture --base /home/ctj/models/Qwen3-4B --output qwen3_4b_ttt_resource_4096 --chunk-size 4096 --ttt-layers 0 6 12 18 24 30 35 --ttt-lr 0.3
python -m tasks.build_cbf_scenarios --tokenizer qwen3_4b_ttt_resource_4096 --output qwen4b_pilot_4096.jsonl --chunk-size 4096 --train-groups 1 --dev-groups 1 --test-groups 1 --variants-per-group 1 --context-chunks 4 --future-chunks 1 --futures-per-scenario 1 --seed 42
CUDA_VISIBLE_DEVICES=1 python -m tasks.cbf_ttt collect --model qwen3_4b_ttt_resource_4096 --dtype bfloat16 --data qwen4b_pilot_4096.jsonl --split train --output qwen4b_labels_4096.jsonl --grid 0,0.5,1 --every 2
CUDA_VISIBLE_DEVICES=0 python -m tasks.cbf_ttt collect --model qwen3_4b_ttt_resource_4096 --dtype bfloat16 --data qwen4b_pilot_4096.jsonl --split dev --output qwen4b_labels_4096_dev.jsonl --grid 0,0.5,1 --every 2
```

拿到已训练的同配置权重后，用它和对应 tokenizer 替换上面资源夹具路径，并把源组数量调整为 `100/20/20`；训练控制器和完整测试命令见前文“运行命令”。

## Qwen3-4B 继续预训练试跑准备（2026-09-24）

用户要求先在两张 RTX 5090 上运行 Qwen3-4B 的 In-Place TTT 继续预训练，并从网络取得训练语料。原仓库的 `qwen3_longct.yaml` 面向 8B（其 `ttt_layers` 含 4B 越界的第 36 层），`data.train_path` 仍是占位符。基于此新增以下**独立**文件，未修改 baseline 预训练入口：

| 文件 | 内容 | 对原 baseline 的影响 |
|---|---|---|
| `configs/pretrain/qwen3_4b_smoke.yaml` | Qwen3-4B 的两 GPU 短程试跑配置；实际参数和结果见下方“等量语料试跑”更新 | 新配置，仅明确选择时使用 |
| `scripts/build_fineweb_pretrain_pilot.py` | 通过 Hugging Face streaming 读取 FineWeb-Edu `sample-10BT` 的有限文档，用 Qwen tokenizer 加 EOS 打包为 `content_split` JSONL；回编码检查每行足够长以进入第 2 个 TTT chunk，且不会超出 8,192 token；写入来源与长度元数据 | 新数据脚本，仅明确运行时使用 |

试跑数据选择 [FineWeb-Edu 官方数据卡](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 的英文教育文本随机子集，数据卡许可证为 ODC-BY。这里计划**按需流式下载**生成 16 条 8,192-token 附近的样本，不下载整个 10B-token 子集。采用文本重分词而非预分词输入，因为 README 指定的 [VeOmni commit 的数据变换](https://raw.githubusercontent.com/ByteDance-Seed/VeOmni/9b91e164bea9e17f17ed490aab5e076c2335ca25/veomni/data/data_transform.py)提供 plaintext 流程。每条样本先验证重分词长度，避免被 VeOmni 再次编码后掉到一个 chunk 内。不同网页由 EOS 隔开；这种跨文档打包只适合训练链路与吞吐试跑，尚不能代表原论文的长文档训练分布或 CBF 方法收益。

最初拟定的命令（历史记录；实际命令见下方更新）：

```bash
python -m scripts.build_fineweb_pretrain_pilot --tokenizer /home/ctj/models/Qwen3-4B --output /home/ctj/data/fineweb_edu_pilot.jsonl --max-records 16 --seq-len 8192 --chunk-size 4096
CUDA_VISIBLE_DEVICES=0,1 bash train.sh tasks/train_torch.py configs/pretrain/qwen3_4b_smoke.yaml --data.train_path /home/ctj/data/fineweb_edu_pilot.jsonl
```

**历史状态（已被下方更新取代）：**当时未开始下载或预训练，远程权限请求被拒绝。之后远程操作获得授权，数据、环境和训练的实际进展见下方。服务器首次只读检查确认有两张 RTX 5090、每张 32,607 MiB，磁盘可用约 2.3 TiB。

该阶段本地检查：新数据脚本已通过 Python AST 语法解析；用简单可逆 tokenizer 的 4 行打包测试均得到长度 16、进入第 2 个 chunk；`git diff --check` 通过。远程实测记录见下方。

### 原论文训练语料的公开性核查（2026-09-24）

用户要求从网络直接下载论文使用的原始训练数据。核对[原论文附录 9.1](https://arxiv.org/html/2604.06169#A9.SS1)后确认，作者明确称大规模预训练、继续预训练及消融数据是**自行收集**；Qwen3-4B 的继续预训练数据包含类似普通预训练文本的短文档，以及书籍、仓库级代码、合成检索和长上下文问答组成的长文档，分为 32k 与 128k 两阶段。论文给出了第一阶段约 20B token、第二阶段约 15B token 的规模，但没有公布原始样本清单、下载链接或可重建的混合比例。仓库 [README 数据准备章节](README.md#data-preparation)也要求用户自行提供 VeOmni 兼容数据，`configs/pretrain/qwen3_longct.yaml` 的训练路径仍是占位符。因此**不存在已确认可下载的“论文原始 20B 数据集”地址**；没有在服务器下载或把 FineWeb-Edu 冒称原始数据。FineWeb-Edu 脚本保持为公开替代语料的短程链路试跑工具。

## FineWeb-Edu + LongCrawl64 等量语料试跑（2026-09-24，取代上方旧状态）

### 目标、假设与文件

用户将训练来源指定为 FineWeb-Edu 和 LongCrawl64，先按 **1:1 Qwen token** 试跑；评测指定 ZsRE、MMLU、RULER、RULER/QA、LongMemEval-S、LoCoMo。它们是公开替代来源和基准，并非论文未发布的原始 20B/15B token 语料。当前只做 16 条、8 次更新的资源与端到端链路试跑；尚无证据支持正式训练规模或方法效果。Qwen3-4B 是 base 权重，配置中的 7 个 TTT 层均为 4B 的有效层号。

| 文件 | 具体修改与需求对应 | baseline 兼容性 |
|---|---|---|
| `configs/pretrain/qwen3_4b_smoke.yaml` | 两张 5090、4B、`ttt_chunk=4096`、序列 6144、8 步、FSDP2、BF16 参数/梯度、`anyprecision_adamw`；对应等量语料预训练短跑。`attn_implementation=sdpa` 避免缺失 `flash_attn`。 | 独立配置；原 `qwen3_longct.yaml` 和训练入口未改。 |
| `configs/pretrain/qwen3_4b_lr03_probe.yaml` | 仅将内部 TTT 更新率从 3 改为 0.3，并关闭 checkpoint 写盘，用同批数据比较早期数值稳定性。 | 独立诊断配置。 |
| `scripts/setup_qwen3_4b_train_env.sh` | 独立 Python 3.11 conda 环境，PyTorch 2.9.1 cu130、Transformers 4.57.3、固定 VeOmni commit 与训练依赖。 | 不改 baseline 环境。 |
| `scripts/build_fineweb_pretrain_pilot.py` | 流式读取官方 FineWeb-Edu `sample-10BT`，Qwen 分词后以 EOS 拼接成定长文本。 | 新增数据入口。 |
| `scripts/build_longcrawl_pretrain_pilot.py` | 读取发布方 LongCrawl64 parquet 的 `text`；若只有 GPT-2 TikToken IDs，先解码再用 Qwen 分词；不得直接把 GPT-2 IDs 交给 Qwen。 | 新增数据入口。 |
| `scripts/merge_pretrain_pilot.py` | 校验两边非空、等行数，再逐行交替拼成 1:1 训练 JSONL，保留来源名。 | 新增数据入口。 |
| `scripts/resize_pretrain_pilot.py` | 复用已下载的均衡样本，按 Qwen tokenizer 重编码并裁成 6144 token，检查仍跨越首个 4096-token chunk。 | 新增数据入口。 |
| `scripts/download_cbf_eval_data.py` | 下载 ZsRE、MMLU、LongMemEval-S、LoCoMo 原始评测数据并计算 SHA256；RULER 与 QA 标明由 OpenCompass 生成。 | 新增评测数据入口。 |
| `experiments/cbf_ttt/qwen3_4b_pretrain_smoke_20260924/` | 保存真实 8 步权重上的反事实标签和逐条时间/显存摘要，不保存多 GiB 远程 checkpoint。 | 独立实验记录。 |

远程工作区 `/home/ctj/cbf_ttt_verify_20260923`，模型 `/home/ctj/models/Qwen3-4B`，隔离训练环境 `/home/ctj/miniconda3/envs/cbf_ttt_train_py311`。用户指定的两个来源已实际下载：FineWeb-Edu 从 `HuggingFaceFW/fineweb-edu` 流式取样，LongCrawl64 从发布方 `manifestai/longcrawl64` 取得 `train/0-of-256.parquet`（约 10.9 GB，实测列为 `text`）。各生成 8 条 8192-token 样本，再重编码裁成各 8 条 6144-token 样本，合并为 16 条、总计 98,304 个 Qwen token。每条均跨越 4096-token 更新边界。FineWeb 脚本在写出有效文件后旧 Python 3.10 环境解释器退出时出现 `PyGILState_Release` 错误；后续检查 8 行完整、元数据完整，且合并与两卡训练均成功，仍需在正式批量流水线中修复该退出异常。

远程评测文件位于 `/home/ctj/data/cbf_ttt_eval/MANIFEST.json`：ZsRE 使用 KnowEdit 的 `ZsRE-test-all.json`（1301 条的知识编辑版本）；MMLU `cais/mmlu` dev 285、test 14042；LongMemEval-S cleaned 500 问；LoCoMo `locomo10.json` 10 个会话。RULER 与 RULER/QA 通过仓库已有 `eval_config/ruler_*.py` 引入 OpenCompass 生成器，没有固定下载文件。ZsRE 版本、LoCoMo 非文本内容处理和各基准的官方评分协议须在正式评测前固定；当前完成的是**数据获取与格式核对**，并未完成这六项正式指标。LongMemEval-S 的约 115k-token 长度超出当前短程 6144-token 训练条件，不能用截断后的分数冒充完整 LongMemEval-S。

### 实测训练及资源限制

最初 8192-token FSDP2 + FP32 参数/AdamW 在首次 `optimizer.step` 溢出。改用 `anyprecision_adamw` 后第一次更新成功、第二次反传 GPU OOM；减到 6144 仍在第二次反传 OOM（进程显存约 30.5 GiB/卡）。FSDP1 CPU 卸载虽可卸载参数、梯度和优化器状态，但服务器内存只有 62 GiB、swap 4 GiB，系统在加载/准备时以 SIGKILL 终止一个 rank，因而未采用。**FSDP2 + BF16 参数/梯度 + BF16 状态优化器**完成 8 次更新，DCP 和 HF 权重导出成功，未更改 baseline 实现。训练环节记录 GPU 最大分配约 22.12 GB（VeOmni logger）；运行时 `nvidia-smi` 曾见约 27–28 GiB/卡。总训练正文约 19 秒，加两次 DCP 写出及 HF 转换约 80 秒；DCP 约 32 GB，HF 权重约 8.4 GB。由于 `save_steps=8` 且 VeOmni 的默认 `save_epochs=1`，第 8 步保存了两次同一路径，浪费 I/O；正式配置应设置 `save_epochs: 0` 或调整保存间隔。

`ttt_lr=3` 的逐步训练 loss 为 `2.7969,20.5905,12.1207,12.0791,10.4522,12.8502,8.3211,8.3221`；同批数据、`ttt_lr=0.3`、外层 `lr=5e-6` 探针为 `2.7969,13.9545,8.7814,5.0937,3.2201,3.5250,3.2368,3.9742`；将外层学习率进一步降到 `1e-6` 后为 `2.7969,3.9602,3.3979,2.7819,2.5448,2.5241,2.2895,2.5154`。据此把短跑配置改为 `ttt_lr=0.3`、`lr=1e-6`、`save_epochs=0`，再独立运行一次带保存的 8 步，末步 loss `2.5157`，DCP 和 HF 导出成功。该比较只有同一批 16 条数据、没有独立验证集，**只能表明短程数值稳定性，不应作为收敛或效果证据**。BF16 参数/梯度的精度、优化器状态和高梯度范数仍需正式评估。

选定的 8 步 HF checkpoint：`/home/ctj/cbf_ttt_pretrain_qwen3_4b_stable_smoke/checkpoints/global_step_8/hf_ckpt`。它能被 `tasks.cbf_ttt collect` 加载，在一个 4-chunk 合成场景中以 `0/0.5/1` 三点网格、每隔两个边界生成 **2 条标签**；逐条为 `8.33/4.19 s`，均值 `6.26 s/条`，峰值分配 `17.14 GiB`、峰值保留 `20.03 GiB`。逐条原始数据与摘要在本节表格所列实验目录；原高学习率 checkpoint 的两条标签也保留以供对照。该场景仍是合成账本，不能代表公开基准或模型质量。

### 复现命令与当前限制

以下在远程仓库目录执行；预训练命令仅是资源短跑，不会训练至收敛。`PATH` 与 `PYTHONPATH` 应指向上述独立训练环境及仓库根目录。

```bash
python -m scripts.build_fineweb_pretrain_pilot --tokenizer /home/ctj/models/Qwen3-4B --output /home/ctj/data/cbf_ttt_pilot/fineweb_edu_8x8192.jsonl --max-records 8 --seq-len 8192 --chunk-size 4096
python -m scripts.build_longcrawl_pretrain_pilot --input /home/ctj/data/cbf_ttt_pilot/longcrawl64/train/0-of-256.parquet --tokenizer /home/ctj/models/Qwen3-4B --output /home/ctj/data/cbf_ttt_pilot/longcrawl64_8x8192.jsonl --max-records 8 --seq-len 8192 --chunk-size 4096
python -m scripts.merge_pretrain_pilot --fineweb /home/ctj/data/cbf_ttt_pilot/fineweb_edu_8x8192.jsonl --longcrawl /home/ctj/data/cbf_ttt_pilot/longcrawl64_8x8192.jsonl --output /home/ctj/data/cbf_ttt_pilot/mixed_16x8192.jsonl
python -m scripts.resize_pretrain_pilot --input /home/ctj/data/cbf_ttt_pilot/mixed_16x8192.jsonl --tokenizer /home/ctj/models/Qwen3-4B --output /home/ctj/data/cbf_ttt_pilot/mixed_16x6144.jsonl --seq-len 6144 --chunk-size 4096
CUDA_VISIBLE_DEVICES=0,1 bash train.sh tasks/train_torch.py configs/pretrain/qwen3_4b_smoke.yaml --data.train_path /home/ctj/data/cbf_ttt_pilot/mixed_16x6144.jsonl
python -m scripts.download_cbf_eval_data --output /home/ctj/data/cbf_ttt_eval
CUDA_VISIBLE_DEVICES=0 python -m tasks.cbf_ttt collect --model /home/ctj/cbf_ttt_pretrain_qwen3_4b_stable_smoke/checkpoints/global_step_8/hf_ckpt --dtype bfloat16 --data qwen4b_pilot_4096.jsonl --split train --output qwen4b_stable_8step_labels_4096.jsonl --grid 0,0.5,1 --every 2
```

后续 TODO：扩大 FineWeb/LongCrawl 等量 token 的独立训练集，确定可稳定下降的学习率与精度方案；训练经过验证的 TTT checkpoint 后才生成正式 CBF 标签和控制器；为 ZsRE/MMLU/LongMemEval-S/LoCoMo 固定评测适配与判分，配置 RULER QA 单独结果；完整长上下文评测需匹配训练/位置编码长度。原始 baseline 配置、训练入口及默认推理路径仍保持可用。

### 公开基准适配与最小链路验证

本次新增 `tasks/eval_public_benchmarks.py`，按实际下载数据结构解析 MMLU、ZsRE（KnowEdit 版本）、LongMemEval-S、LoCoMo；支持 `--inspect-only`、少量样本 `--max-examples`、baseline/controller 策略和逐题 JSONL。MMLU 用同一学科 dev 最多 5-shot 的 A/B/C/D 答案条件 NLL 排序，输出选择正确率；其余任务对贪心生成做 Unicode 字词归一化 exact match。ZsRE 将新目标事实放入上下文，测改写提问的召回；目前不测官方的 locality/portability 汇总。LoCoMo 使用文字对话及图片 caption，不下载原图片。这些生成模板/判分只是**链路诊断代理指标**，输出 `official_benchmark_score: false`，不得引用为官方 benchmark 分数。超出 `max_position_embeddings` 的完整上下文会写 `skipped: context_over_limit`，不暗中截断；LongMemEval-S 首题实测 Qwen token `105522`，大于当前 Qwen3-4B 的 `40960` 原生上限；LoCoMo 首个会话为 `16646` token。

新增 `eval_config/ruler_qa_4k.py`，仅运行 OpenCompass RULER 的 SQuAD/HotpotQA 两个 QA 子任务；`CBF_RULER_NUM_SAMPLES` 默认 100，可在短跑时改为 2。新增 `eval_config/ruler_4k_smoke.py`，保持完整 13 个 RULER 子任务但每项 1 条、单 worker。`eval_config/models.py` 仅增加环境变量 `CBF_EVAL_MODEL`/`CBF_EVAL_NAME` 可选覆盖，未设置时原来的占位模型配置不变。`tasks/run_opencompass.py` 先注册仓库 TTT 推理模型，再进入 OpenCompass CLI，避免命令行中的手工 Python 引号拼接。`scripts/setup_cbf_eval_env.sh` 可复建隔离的 Python 3.11 环境（OpenCompass 0.5.4、RULER NIAH 所需 wonderwords 3.0.1）；训练环境没有被新依赖修改。初次 QA 配置把 `os` 模块对象写进了 mmengine 自动输出，导致语法错误；现用表达式读取环境变量，重新执行成功。完整 RULER 初次运行发现缺少 `wonderwords`，现已安装并重跑。

实测 `--inspect-only --max-examples 2` 对四种原始文件均成功解析；选定 checkpoint 上 MMLU `0/2`、ZsRE `0/2`、LoCoMo `0/1`，样本太少且模型只训 8 步，不代表正式效果。LongMemEval-S 第一题被准确标为上下文超限、`scored=0, skipped=1`。OpenCompass RULER/QA 4k 每项 2 条成功跑通：SQuAD `2/2`、HotpotQA `1/2`；这些数字同样只是运行验证。完整 RULER 4k 的 **13 个子任务各 1 条**也已完成生成、推理、判分，逐项数值在 `public_eval/ruler_4k_1_each_scores.json`。原 OpenCompass 分组摘要只产生空白的 `4k` 组行，但每个子任务 JSON 均有 `score`；烟测配置现已移除该分组摘要，使后续运行逐项显示。相关逐题结果/摘要已保存到 `experiments/cbf_ttt/qwen3_4b_pretrain_smoke_20260924/public_eval/`。短上下文 MMLU/ZsRE 不含完整 TTT chunk，因而不会触发 CBF 决策，其分数只用于检查一般能力保持。

```bash
# 在 cbf_ttt_verify_20260923 环境中运行；仅为诊断代理指标
python -m tasks.eval_public_benchmarks --dataset mmlu --data-root /home/ctj/data/cbf_ttt_eval --model /home/ctj/cbf_ttt_pretrain_qwen3_4b_stable_smoke/checkpoints/global_step_8/hf_ckpt --output mmlu_probe.jsonl --max-examples 2
python -m tasks.eval_public_benchmarks --dataset zsre --data-root /home/ctj/data/cbf_ttt_eval --model /home/ctj/cbf_ttt_pretrain_qwen3_4b_stable_smoke/checkpoints/global_step_8/hf_ckpt --output zsre_probe.jsonl --max-examples 2
python -m tasks.eval_public_benchmarks --dataset locomo --data-root /home/ctj/data/cbf_ttt_eval --model /home/ctj/cbf_ttt_pretrain_qwen3_4b_stable_smoke/checkpoints/global_step_8/hf_ckpt --output locomo_probe.jsonl --max-examples 1
python -m tasks.eval_public_benchmarks --dataset longmemeval_s --data-root /home/ctj/data/cbf_ttt_eval --model /home/ctj/cbf_ttt_pretrain_qwen3_4b_stable_smoke/checkpoints/global_step_8/hf_ckpt --output longmemeval_probe.jsonl --max-examples 1

# 在独立 cbf_ttt_eval_opencompass 环境中运行；完整 RULER 可用原 eval_config/ruler_4k.py
bash scripts/setup_cbf_eval_env.sh
CBF_EVAL_MODEL=/home/ctj/cbf_ttt_pretrain_qwen3_4b_stable_smoke/checkpoints/global_step_8/hf_ckpt CBF_RULER_NUM_SAMPLES=2 CUDA_VISIBLE_DEVICES=0 python -m tasks.run_opencompass eval_config/ruler_qa_4k.py --debug
CBF_EVAL_MODEL=/home/ctj/cbf_ttt_pretrain_qwen3_4b_stable_smoke/checkpoints/global_step_8/hf_ckpt CUDA_VISIBLE_DEVICES=0 python -m tasks.run_opencompass eval_config/ruler_4k_smoke.py --debug
```

正式评测 TODO：为 MMLU 固定与公开 leaderboard 相同的 prompt/答案提取及版本；ZsRE 完成编辑成功率、改写、locality 和 portability；LongMemEval-S 使用经过验证的 128k 模型与官方 judge；LoCoMo 使用完整文字/图像协议及官方判分；RULER 在训练完成的 checkpoint 上以足够样本、预设随机种子重复运行。当前这些工作没有完成，不对外发布性能结论。

## 1B Qwen-token 训练配方（2026-09-24；只准备，尚未启动 1B 训练）

用户询问如何把等量 FineWeb-Edu/LongCrawl64 扩展到“1B 数据集”。这里把 1B 明确解释为**总计至少 1,000,000,000 个 Qwen3-4B token**，而非 1 GB 文件或每个来源各 1B。维持前次可运行的 `max_seq_len=6144`、`ttt_chunk=4096`、双 5090、BF16 FSDP2 和保守学习率。每种来源取 `81,381` 条完整记录，训练共 `162,762 × 6144 = 1,000,009,728` 个 token，等量误差仅为相对目标多 `9,728` token。该配置仍为短上下文预训练，不等于论文的 32k/128k 阶段，也未验证 1B 后收敛效果。

为让此规模可执行，本次做了以下增量修改，仍不改 baseline 训练入口：

| 文件 | 新增/修改内容 |
|---|---|
| `configs/pretrain/qwen3_4b_1b.yaml` | 独立 1B 配置：`train_size=1000009728`、`max_steps=81381`、`num_train_epochs=1`、每 10,000 步保存 DCP、末 epoch 再保存完整最终权重并导出 HF。 |
| `scripts/build_longcrawl_pretrain_pilot.py` | `--input` 现在接受一个或多个有序 parquet 分片；按顺序继续读取，并在元数据写出分片清单和实际 Qwen token 总数。原单分片调用保持兼容。 |
| `scripts/build_fineweb_pretrain_pilot.py` | 元数据新增实际 Qwen token 总数，便于核对 1B。 |
| `scripts/merge_pretrain_pilot.py` | 由一次性把两份 JSONL 全部读进 RAM 改为逐行交替流式写入 `.incomplete` 临时文件；行数不等时报错，成功后才替换目标文件。 |

实测发布方 LongCrawl64 `train/0-of-256.parquet` 有 `96,689` 行；取前 100 行用 Qwen tokenizer 得到合计约 `5.81M` token，粗略外推该分片约 `5.62B` Qwen token，足以覆盖本次所需 `0.5B`；这是非随机样本的容量估计，构建脚本会在实际不够时明确报错，可再下载其他分片并作为额外 `--input` 参数。远程已核对发布方存在 `train/1-of-256.parquet` 等命名。新版两分片读取加精确 128-token 打包在远程合成 parquet 上通过；流式合并在本地 3+3 条样本上通过，输入行数不等时会报错且不产生正式输出。远程 YAML 校验确认 `81381×2×6144=1000009728`，最终保存开关已设置。1B 全量构建及训练尚未执行，不能称作已验证吞吐或结果。

在 `hku-gpu2` 的 `/home/ctj/cbf_ttt_verify_20260923` 目录、激活 `cbf_ttt_train_py311` 环境后，顺序运行：

```bash
source /home/ctj/miniconda3/etc/profile.d/conda.sh
conda activate cbf_ttt_train_py311
export PYTHONPATH=$PWD
mkdir -p /home/ctj/data/cbf_ttt_1b
python -m scripts.build_fineweb_pretrain_pilot --tokenizer /home/ctj/models/Qwen3-4B --output /home/ctj/data/cbf_ttt_1b/fineweb_edu.jsonl --max-records 81381 --max-documents 10000000 --seq-len 6144 --chunk-size 4096
python -m scripts.build_longcrawl_pretrain_pilot --input /home/ctj/data/cbf_ttt_pilot/longcrawl64/train/0-of-256.parquet --tokenizer /home/ctj/models/Qwen3-4B --output /home/ctj/data/cbf_ttt_1b/longcrawl64.jsonl --max-records 81381 --max-documents 100000 --seq-len 6144 --chunk-size 4096
python -m scripts.merge_pretrain_pilot --fineweb /home/ctj/data/cbf_ttt_1b/fineweb_edu.jsonl --longcrawl /home/ctj/data/cbf_ttt_1b/longcrawl64.jsonl --output /home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl
wc -l /home/ctj/data/cbf_ttt_1b/*.jsonl
CUDA_VISIBLE_DEVICES=0,1 bash train.sh tasks/train_torch.py configs/pretrain/qwen3_4b_1b.yaml
```

训练前应检查两份 `.jsonl.meta.json` 的 `records=81381`、每条至少 `6143` 且最多 `6144` 个 token、各来源实测总数至少 `500000000`，合并文件应为 `162762` 行，两个来源总数至少 `1000000000` 且相对差异小于 `0.1%`。首次真实构建发现少量文本经 tokenizer decode/re-encode 后缩短 1 个 token：FineWeb-Edu 为 `500004858`、LongCrawl64 为 `500004822`，合计 `1000009680` 个 token，仍超过 1B。首个 LongCrawl 分片已在远程服务器，无需重复下载；若实际不够，可从发布方增加分片，按顺序传给多值 `--input`。

**关于 `max_steps` 的官方实现核对（2026-09-24）：**官方 [`configs/pretrain/qwen3_longct.yaml`](configs/pretrain/qwen3_longct.yaml) 同时设置 `train_size=20000000000` 和 `max_steps=5000`，并采用 `datasets_type=iterable`、`rmpad=false`、`global_batch_size=64`、`max_seq_len=65536`。按每步名义 token 数计算，`5000×64×65536=20,971,520,000` token，约为所称 20B。远程固定版本 VeOmni 的 `IterativeDataset` 没有 `__len__`；`tasks/train_torch.py` 因而将 `dataset_length=None` 交给 `TrainingArguments.compute_train_steps()`。该函数在 `rmpad=false` 且长度未知时使用 `max_steps`，缺少它会抛出 `Please provide dataset_length or max_steps!`。因此当前 1B 配方保留显式 `max_steps=81381`，每步全局 batch 2、序列 6144，合计 `81381×2×6144=1,000,009,728` token；它与官方 20B 配置遵循同一计步方式。`max_steps` 并非 VeOmni 所有数据模式的普遍必填项：mapping 数据有长度时可以按长度计步，启用 `rmpad` 时可按 `train_size` 估算，但这些都不是本次已经跑通的配置。

**资源和风险：**8 步试跑正文约 2.3 秒/步，线性外推 `81,381` 步约 52 小时纯计算，另加数据构建、加载、检查点 I/O 和不确定的长时吞吐；应留出约 2–3 天或更长。现有 DCP 单份约 32 GB，8 个中途点加最终点约 288 GB；服务器检查时尚有约 2.2 TB 可用。FineWeb 流式下载与 Qwen 重分词需要稳定网络和 CPU 时间，当前脚本中断后不支持从精确文档/token 状态续写，应在 tmux 内执行并预留重试时间。长跑前宜在新数据上再做 100–1000 步损失和资源观察，必要时调节学习率；本节并未启动正式训练。

## 训练与评测数据下载指南、GitHub 分发（2026-09-24）

新增 [`DATA_DOWNLOAD_GUIDE.md`](DATA_DOWNLOAD_GUIDE.md)，分别说明 Qwen3-4B 模型、FineWeb-Edu 流式获取、LongCrawl64 发布方 parquet 分片下载与 Qwen 重新分词、1:1 token 合并、ZsRE/MMLU/LongMemEval-S/LoCoMo 下载、RULER/QA 在线生成，以及 SHA256 校验和网络受限 NPU 机器上的代码/数据分离传输。`README.md` 的 Data Preparation 增加该指南链接；这是文档变更，未改 baseline 训练和推理逻辑。指南明确两卡 CUDA 配置及安装脚本不能直接当作 8 卡 NPU 方案，NPU 训练需匹配 CANN/torch_npu 环境、调整全局 batch/步数并完成兼容性短跑。用户提供的 GitHub 目标为 `chenhaoling/CBF-TTT`；服务器防火墙使本机不能直接登录 8 卡 NPU，实际克隆需在 NPU 服务器侧执行指南中的命令。模型权重、数据和 DCP 均不提交 Git。

## hku-gpu2 两卡 5090 的 1B 继续预训练启动（2026-09-24）

用户明确选择 `hku-gpu2`，本次新增 [`scripts/run_qwen3_4b_1b_hku_gpu2.sh`](scripts/run_qwen3_4b_1b_hku_gpu2.sh) 并同步到远程 `/home/ctj/cbf_ttt_verify_20260923/scripts/`。脚本激活既有 `cbf_ttt_train_py311` 环境，先构建 FineWeb-Edu 和 LongCrawl64 各 81,381 条 6144-token JSONL，检查各自元数据、行数及总 token 数，再交替合并为 162,762 行并检查，成功后才运行 `configs/pretrain/qwen3_4b_1b.yaml` 的两卡训练。任何构建或校验失败都会因 `set -euo pipefail` 停止，避免在不完整数据上开训。该脚本只为此服务器的已验证 CUDA 环境准备，不是 NPU 启动脚本。

远程 tmux session 为 `cbf_ttt_1b_20260924`，总日志为 `/home/ctj/cbf_ttt_1b_20260924.log`，输出目录由配置指定为 `/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b`。启动时两张 5090 空闲、模型目录和 11 GB LongCrawl64 首分片存在、磁盘约 2.2 TB 可用。2026-09-24 20:46 左右检查确认 session 与 `build_fineweb_pretrain_pilot` 进程存活，FineWeb JSONL 已增长到约 46 MB。23:49 检查时两份数据均构建完成，但初版校验要求每行恰好 6144 token；少量 decode/re-encode 缩短 1 个 token，导致校验退出，**训练未启动**。现修正脚本为校验实际 token 下界、合计至少 1B、来源差异低于 0.1%、记录与文件行数；保留这次停止记录，不把它视为训练成功。可用 `tmux attach -t cbf_ttt_1b_20260924` 或 `tail -f /home/ctj/cbf_ttt_1b_20260924.log` 观察；离开 tmux 用 `Ctrl-b d`。脚本通过 `bash -n`。

23:55 在相同 tmux 名称重新启动修正后的脚本；它复用已完成的两个来源文件，重新验证后合并出 `mixed_1b.jsonl`（约 4.16 GB），并进入两卡训练。23:59 左右检查到 rank0/rank1 都报告 `train_steps: 81381`，进度约第 88 步、约 `2.19 s/step`，两卡利用率均 100%、显存约 27.6–27.7 GiB，日志中的 loss 为有限值。若持续该速度，剩余纯训练约 49.5 小时，另加检查点保存时间；这只是早期速度外推，尚无收敛或最终结果。`data.train_size=1000009728` 是按固定长度的名义值，实测源元数据相加为 `1000009680`，差 48 token，实际 token 数仍超过 1B。首个检查点计划在第 10,000 步保存，因此目前尚无可恢复的正式中途检查点。

**完成状态（2026-09-27 01:29）：**远程日志确认 `81381/81381` 步、epoch 1/1 完成，训练进度条约 `49:31:42`、平均约 `2.19 s/step`；没有新的 traceback 或 `NaN` loss。VeOmni 在 epoch 结束时报告峰值显存 `22.12 GB`（框架分配量；训练期间 `nvidia-smi` 曾见每卡约 27.6–27.7 GiB）。DCP 成功保存到 `/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381`，HF 权重成功导出到其 `hf_ckpt/`，含两个 safetensors 分片约 4.97 GB 和 3.94 GB、索引、配置和 tokenizer；整个最终检查点目录约 40 GB。中途检查点已保存到每 10,000 步至 80,000 步。`scripts/run_qwen3_4b_1b_hku_gpu2.sh` 于 01:29:03 输出 `Pretraining finished`，tmux session 退出，两卡恢复空闲。这里完成的是 **1B token 的 In-Place TTT 继续预训练**；尚未在最终模型上生成正式反事实标签、训练控制器或完成六项正式评测，不能据此宣称 CBF 方法收益。

## 最终 1B 模型上的正式反事实标签构造（2026-09-27）

### 目标、研究假设与方案调整

使用上述第 `81381` 步 **Qwen3-4B In-Place TTT 最终 HF checkpoint**，对同一会话状态的 `alpha={0,0.5,1}` 分支做反事实答案损失比较，记录逐标签耗时和 CUDA 峰值显存。基础权重固定；只有快记忆中的历史增量按方法 3.3 的 `(1-alpha)M+ΔW` 衰减；`alpha=0` 仍是后续参考策略。场景中的账本事实和答案是按随机种子 42 生成的合成事实，**不等同于** ZsRE、MMLU 等公开评测集。自然文本背景取自该模型所用 `/home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl` 的不同 `content_split` 记录；这是训练语料的背景复用，不能据此声称完全独立的自然文本测试集。train/dev/test 按 `group_id` 一次性分组，三种场景变体始终位于同一 split。

第一轮沿用 4 个 4096-token context chunk、1 个完整 future chunk、每隔 2 个边界取标签，但已训练权重与旧的零卷积资源夹具不同。最终模型在高度重复的合成填充下 24 条标签全部选 `alpha=1`，平均标签耗时 `6.219 s`，峰值分配/保留显存 `17.145/20.033 GiB`；改用不同自然文本记录后，chunk 的平均唯一 token 比例由约 `1.1%` 提高到约 `31.7%`，24 条仍全选 1。将原始事实只放首块、保留完整未来 chunk 的另 24 条也全选 1。三个完整 future chunk 试点的损失显示重复累积 TTT 更新后明显漂移；现有 1B 预训练样本为 6144 token，训练中最多跨过一个 4096-token 更新边界，不能假定 4 个连续更新边界的推理质量已经得到训练覆盖。

为了区分当前遗忘动作本身与后续连续更新的影响，又做了短时距试点。直接删去 future continuation 的 12 条标签分布为 `alpha=0/0.5/1` 各 `2/1/9`，但更正与新主题的答案尚未在上下文出现，因此**只作为诊断，不用于正式训练**。正式采用 `short_tail`：future 的新事实先以不足一个 chunk 的尾部进入 KV，保证问答依据存在，但不额外写入 TTT 快记忆；context 为 2 个 4096-token chunk，仅在第 2 个边界评分。首块给出原始事实，后块不重复该事实，背景由不同自然文本记录填充。这个协议下 8 个训练源组、每组 3 变体的 24 条有效试点标签为 `alpha=0/0.5/1` 各 `5/2/17`；三个候选的平均损失分别为 `0.3301/0.3227/0.3106`，单条标签均值 `0.889 s`，最高分配/保留显存 `12.651/13.945 GiB`。三种状态都有样本，但 1 仍占多数，控制器训练时需检查类别/回归目标分布及独立开发集效果。

### 修改计划与文件对应

| 文件 | 本次具体修改 | 附件对应需求 | baseline 影响 |
|---|---|---|---|
| `tasks/build_cbf_scenarios.py` | 增加 `--background-data`、`--fact-placement first_only`、`--future-mode short_tail`；读取不同 `content_split` 背景记录，控制事实出现位置，并在短尾 future 中给出更正/新主题事实；元数据记录背景数和模式 | 3.5.2 后续情景、3.5.3 场景多样性与组划分 | 旧参数默认仍为合成填充、重复事实、完整 future chunk |
| `scripts/shard_cbf_scenarios.py` | 两次流式扫描，按每个 split 内排序后的源组均衡分片；检验唯一场景 ID、组与 split 隔离，写分片 manifest | 3.5.1 同状态分支、3.6.3 源组隔离；两卡并行 | 新独立脚本 |
| `scripts/run_cbf_formal_labels.sh` | 单 GPU/单分片顺序收集 train/dev/test，固定三点网格、BF16 与稀疏边界；已有输出拒绝覆盖 | 3.5 离线标签；逐标签成本记录 | 新独立脚本 |
| `scripts/summarize_cbf_formal_labels.py` | 完成后逐场景核对六份标签的 ID、组、split、边界、网格、有限损失、唯一性与完成摘要，并汇总 alpha/耗时/显存 | 3.5 标签质量与实验记录 | 新独立脚本 |
| `tests/test_cbf_scenarios.py` | 增加不同背景、首块事实、短尾 future、均衡分片与缺失标签检测 | 场景和划分正确性 | 仅测试 |
| `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/formal_label_manifest.json` | 版本化保存正式 3600 条标签的分组、系数、时间和显存汇总，不含原始背景或高维特征 | 实验追踪与复现 | 新独立结果文件 |
| `MODIFICATION_LOG.md` | 记录最终模型试点、正式协议、命令、状态和风险 | 用户要求的修改记录 | 文档 |

### 正式规模、数据与运行命令

正式场景为 train/dev/test `1000/100/100` 个互不重叠源组，每组 `stable/correction/topic_shift` 各一个变体，共 `3600` 场景、`3600` 条标签；每个场景的 2 个 context chunk 使用各不相同的自然文本背景，总计需要 `7200` 条背景记录。未来只有一个短尾情景，含历史事实与当前/新事实两种查询。每张 5090 接收 train/dev/test `1500/150/150` 场景；两卡样本数相同。远程位置为 `/home/ctj/cbf_ttt_1b_labels/formal_short_tail_v1`，最终权重为 `/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt`。远程场景元数据与分片清单已核对上述全部计数；正式运行使用 `cbf_formal_labels_gpu0`、`cbf_formal_labels_gpu1` 两个 tmux session，各自写 `gpu0.log`、`gpu1.log`。

最终模型 `config.json` 记录 `model_type=qwen3`、36 层、隐藏维 2560、TTT 层 `[0,6,12,18,24,30,35]`、`ttt_chunk=4096`、`ttt_lr=0.3`、`ttt_proj=true`。这七层恰为标签器提取记忆特征并共同应用同一 `alpha` 的层。

在 hku-gpu2 的 `/home/ctj/cbf_ttt_verify_20260923`、`cbf_ttt_train_py311` 环境下复现：

```bash
export PYTHONPATH=$PWD
MODEL=/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt
OUT=/home/ctj/cbf_ttt_1b_labels/formal_short_tail_v1
python -m tasks.build_cbf_scenarios --tokenizer /home/ctj/models/Qwen3-4B --output "$OUT/scenarios.jsonl" --background-data /home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl --fact-placement first_only --future-mode short_tail --chunk-size 4096 --train-groups 1000 --dev-groups 100 --test-groups 100 --variants-per-group 3 --context-chunks 2 --future-chunks 1 --futures-per-scenario 1 --seed 42
python -m scripts.shard_cbf_scenarios --input "$OUT/scenarios.jsonl" --output-dir "$OUT/shards" --shards 2
bash scripts/run_cbf_formal_labels.sh "$MODEL" "$OUT" 0 0  # 在独立 tmux 中运行
bash scripts/run_cbf_formal_labels.sh "$MODEL" "$OUT" 1 1  # 在另一 tmux 中运行
python -m scripts.summarize_cbf_formal_labels --directory "$OUT" --shards 2 --boundary 2
```

生成器、分片器与汇总器是新入口，baseline 模型、预训练配置、训练脚本及默认推理路径没有改变。标签文件不包含原始背景文本，但包含从模型得到的特征、候选损失、选中系数、逐条耗时和峰值显存；正式大文件保存在远程，不提交到 Git。以有效试点 `0.889 s/label` 粗估，每卡 `1800` 条约 27 分钟标签计算，另计模型加载、场景读取和 I/O；长时距旧试点的约 6.2 秒/条不应用于这个短尾协议。

### 正式完成结果与核对

两卡于服务器当地时间 `10:23:44` 开始，GPU1 在 `10:57:52` 完成、GPU0 在 `10:58:24` 完成，最大墙钟约 **34 分 40 秒**。自动完成性校验读入六份标签与六份场景分片，验证每个场景恰好一条、分组不跨 split/分片、边界为 2、网格精确为 `[0,0.5,1]`、候选损失与收益有限、每份收集摘要存在；最终 `formal_label_manifest.json` 成功写出。正式场景文件约 158 MB，六份标签合计约 187 MB，均保留在上述远程目录；小型汇总已复制到仓库 `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/formal_label_manifest.json`。

| split | 独立源组 | 标签数 | `alpha=0` | `alpha=0.5` | `alpha=1` |
|---|---:|---:|---:|---:|---:|
| train | 1000 | 3000 | 674 | 259 | 2067 |
| dev | 100 | 300 | 47 | 25 | 228 |
| test | 100 | 300 | 58 | 20 | 222 |
| 合计 | 1200 | **3600** | **779** | **304** | **2517** |

按场景类型汇总：correction 的 `0/0.5/1` 为 `225/85/890`，stable 为 `213/100/887`，topic_shift 为 `341/119/740`；每类各有 1200 条。全体逐标签平均耗时 `0.888574 s`、最大 `1.016993 s`，最高分配/保留峰值 `12.651886/13.945313 GiB`。对每个状态先在三点网格中取最低损失，再与同一状态的 `alpha=0` 比较，平均 oracle 差值为 `0.025511` NLL；这使用了未来答案挑选系数，**不是**可部署控制器或独立评测收益。`alpha=1` 约占 69.9%，应在后续控制器训练时监控是否退化为常数预测。

### 已完成、待完成与风险

- 已完成：最终 HF 模型上的四种场景/时距诊断、正式场景生成与均衡分片、两卡全部 3600 条标签及六份逐文件 `.summary.json`、一次性完成性校验和版本化汇总；本地与远程 `python3 -m unittest tests.test_cbf_scenarios -q` 均 8 项通过，`bash -n scripts/run_cbf_formal_labels.sh` 与 `git diff --check` 通过。
- 当时待完成：用 train/dev 标签训练并选择控制器，在 test 和目标公开基准上独立评测；控制器及合成场景 test 已于下节完成，六项公开基准仍未完成。
- 风险：短尾协议强调当前边界的近期问答，不能替代 4 个 context chunk 加完整 future chunk 的长时距效果；后者在试点全部偏向 `alpha=1`，说明当前 checkpoint 可能缺少多次连续更新的训练覆盖。正式 context 为 8192 token，也比预训练单样本 6144 token 长，仍需检测长度外推。自然背景来自同一 1B 预训练语料，事实虽新但背景不独立；未来应以未见语料和目标评测的实际任务情景复核。标签分布可能偏向 1；正式完成后应报告分布、最优与 `alpha=0` 的损失差和开发集控制器性能，不应仅以样本量称其有效。

## 控制器训练与固定系数对照（2026-09-27）

### 设计与文件

以正式 train 的 1000 个独立源组为母集，固定随机种子 42 打乱**源组**，取嵌套的前 `250/500/1000` 组，得到 `750/1500/3000` 条标签；同一个源组的三种场景不会跨子集或 train/dev。三种规模各训练种子 `42/43/44`，共九个控制器。每次沿用原 `tasks.cbf_ttt train` 入口与默认架构：语义投影 64、MLP 宽度 128、20 epoch、batch 64、AdamW 学习率 `1e-3`。固定的 100 个 dev 源组（300 条标签）用于选 epoch、比较规模和种子；test 的 100 个源组在模型确定之前不参与选型。训练在 hku-gpu2 的 `cbf_ttt_train_py311` 环境进行，控制器仅使用保存的特征与标签，在 CPU 上优化；评测另用最终 Qwen3-4B 和两张 5090 完整 rollout。

| 文件 | 本次改动 | 附件对应需求 | baseline 影响 |
|---|---|---|---|
| `scripts/prepare_cbf_controller_subsets.py` | 对正式标签按源组嵌套抽样、核对 train/dev 组隔离、保存每规模标签与系数分布 | 3.6.3 训练/开发分组隔离 | 新独立脚本 |
| `scripts/run_cbf_controller_learning_curve.sh` | 对三种组规模和三种随机种子运行九次控制器训练，保存权重与逐 epoch MSE | 3.6 控制器监督训练 | 新独立脚本 |
| `scripts/summarize_cbf_controller_learning_curve.py` | 比较九个模型的 dev 原始/裁剪 MSE、输出系数均值与离散程度，以及固定系数回归基线 | 3.6 训练和选择 | 新独立脚本 |
| `scripts/analyze_cbf_controller_rollouts.py` | 对相同场景的固定 `alpha=0/0.5/1` 和控制器做配对 NLL 比较、各类型汇总及源组 bootstrap 95% 区间 | 3.7–3.8 在线对照评测 | 新独立脚本 |
| `tests/test_cbf_controller_subsets.py`、`tests/test_cbf_rollout_analysis.py` | 检验组安全嵌套抽样、配对收益与源组计数 | 数据和统计闭环 | 仅测试 |
| `MODIFICATION_LOG.md` 与 `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/` | 记录命令、参数、结果和小型汇总；大模型、全量标签、rollout JSONL 继续留在远程 | 用户要求的实验保存与说明 | 文档及独立结果 |

远程训练目录 `/home/ctj/cbf_ttt_1b_labels/controller_study_20260927`，其中 `subset_manifest.json` 记录三组源组数与标签数。训练子集 `alpha=0/0.5/1` 分布依次为 `179/60/511`、`341/128/1031`、`674/259/2067`。固定 dev 的 300 条标签中，常数 `alpha=1` 的回归 MSE 为 `0.1775`，固定训练集均值 `0.73217` 的 MSE 为 `0.14299`。每规模按 dev 原始输出 MSE 选出的模型分别为 `g250_s43`：`0.13627`、`g500_s42`：`0.12699`、`g1000_s42`：`0.12631`；1000 组最佳模型的裁剪输出在 dev 上均值 `0.77481`、标准差 `0.16509`，说明没有严格塌缩为常数，但是否有任务收益仍以完整 rollout 为准。

dev rollout 使用同一份 300 场景，比较原协议 `alpha=0`、固定 `0.5`、固定 `1`、仅由训练标签均值确定的固定 `alpha=0.73217`，以及每规模选出的控制器。每个场景独立开始会话；差值使用相同场景配对，再按 100 个源组做 2000 次重采样区间。仅在 dev 上确定最终控制器后，再对 300 场景的 test 执行一次相同策略比较。固定 `alpha=1` 是必要的强对照，因为正式标签约 70% 选择 1；训练均值常数检验语义/记忆特征是否真的有额外价值。这里的 dev MSE 不是方法成功标准；需同时报告独立 test 的问答 NLL、相对固定 1 的配对收益与区间、退化场景比例和输出系数分布。

**dev 完成结果（300 场景、100 源组）：**下表的“相对固定 1 收益”为同场景 `NLL(alpha=1)-NLL(策略)`；负数说明比固定 1 更差。区间按源组重采样，属于用于模型选择的 dev 描述，不当作最终独立检验。

| 策略 | 平均答案 NLL | 相对固定 1 收益 | 源组 bootstrap 95% 区间 |
|---|---:|---:|---:|
| 原始累积更新 `alpha=0` | 0.353898 | -0.027880 | [-0.033686, -0.022896] |
| 固定 `alpha=0.5` | 0.341351 | -0.015334 | [-0.018366, -0.012831] |
| 训练标签均值常数 `alpha=0.73217` | 0.334309 | -0.008292 | [-0.010154, -0.006715] |
| 固定 `alpha=1` | **0.326017** | 0 | [0, 0] |
| 250 组控制器 `g250_s43` | 0.331784 | -0.005767 | [-0.006921, -0.004707] |
| 500 组控制器 `g500_s42` | **0.330130** | **-0.004113** | [-0.005110, -0.003108] |
| 1000 组控制器 `g1000_s42` | 0.330489 | -0.004472 | [-0.005550, -0.003401] |

虽然 1000 组控制器的 dev 标签 MSE 最低，实际 dev rollout 中 500 组模型在控制器之间最好；三个控制器均明显弱于固定 `alpha=1`。因此**按预先声明的 dev rollout 目标固定 `g500_s42` 作为本次 test 控制器**，不再根据 test 调整模型。这个结果显示仅凭 `alpha_star` 的均方误差改善不能证明反事实收益目标改善；当前短尾场景的系数偏向 1，后续可能需要损失敏感训练目标、场景再平衡或更长序列适配训练，但这些都不混入本次已冻结的 test 对照。

**独立 test 完成结果（300 场景、100 源组）：**两卡各处理 150 场景，每场景五个策略，共 1500 条 rollout。完成脚本核对每卡 750 行和两份摘要后合并；分析器要求每个策略覆盖同一批场景/future、每条损失和系数有限，并按源组生成配对区间。正值收益表示策略的答案 NLL 低于比较对象。

| 策略 | 平均答案 NLL | 相对 `alpha=0` 收益 | 相对固定 `alpha=1` 收益 |
|---|---:|---:|---:|
| 原始累积更新 `alpha=0` | 0.373329 | 0 | -0.026111 |
| 固定 `alpha=0.5` | 0.363151 | +0.010178 | -0.015933 |
| 训练均值常数 `alpha=0.73217` | 0.356382 | +0.016946 | -0.009164 |
| 固定 `alpha=1` | **0.347218** | **+0.026111** | 0 |
| 已选控制器 `g500_s42` | 0.351107 | +0.022222 | **-0.003888** |

控制器相对 `alpha=0` 的源组 bootstrap 95% 区间为 `[+0.017455,+0.027687]`，相对训练均值常数的收益为 `+0.005276`、区间 `[+0.003783,+0.006921]`，说明学到的状态变化优于简单训练均值；但相对固定 `alpha=1` 的收益为 **`-0.003888`**、区间 **`[-0.004974,-0.002742]`**，在 `60.67%` 的测试场景上更差。控制器第二块输出均值 `0.7681`、标准差 `0.1923`。correction/stable/topic_shift 的控制器 NLL 分别为 `0.288706/0.372434/0.392179`，固定 1 为 `0.285376/0.368163/0.388116`：三类均未超过固定 1。源组 bootstrap 区间描述本次固定样本的不确定性，不能修正合成数据与预训练背景重用的外推限制。

**结论：**这轮控制器相对原始累积 TTT 有收益，但在当前受控短尾协议中不优于“每次完全遗忘历史快记忆”的固定策略，因而不能宣称 CBF 的自适应控制优于最强常数基线。500 与 1000 源组之间的 dev NLL 也未单调改善，单纯增加同分布标签未必解决问题。下一轮应先预注册新的训练目标（例如依据候选损失差而非只拟合最优网格点）、增加真实未见场景，并针对 4-chunk/完整 future 的更新漂移检查或训练，再重新构造标签；这一轮 test 不用于调节现有模型。

最终控制器在远程 `/home/ctj/cbf_ttt_1b_labels/controller_study_20260927/controller_selected.pt`，大小约 776 KB，SHA-256 为 `5e26d0a53efed5c11e861c8aedbc4f3186e7b7a9a01a8f0928435e9578662d5a`。九次 CPU 拟合从 `11:37:09` 到 `11:37:41`，约 32 秒；两卡完整 dev/test rollout 另计。权重没有提交到公开 Git 仓库；小型 `subset_manifest.json`、`learning_curve_summary.json`、`dev_comparison_with_trainmean.json`、`test_comparison.json` 和 `controller_selection.json` 保存在 `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/controller_study/`，远程同名目录保留全量 rollout 与训练日志。

复现命令（远程仓库目录、conda 环境与上一节相同；`ROOT`、`STUDY` 为绝对路径）：

```bash
ROOT=/home/ctj/cbf_ttt_1b_labels/formal_short_tail_v1
STUDY=/home/ctj/cbf_ttt_1b_labels/controller_study_20260927
MODEL=/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt
python -m scripts.prepare_cbf_controller_subsets --train-labels "$ROOT/train_shard0_labels.jsonl" "$ROOT/train_shard1_labels.jsonl" --dev-labels "$ROOT/dev_shard0_labels.jsonl" "$ROOT/dev_shard1_labels.jsonl" --output-dir "$STUDY" --group-counts 250 500 1000 --seed 42
bash scripts/run_cbf_controller_learning_curve.sh "$ROOT" "$STUDY"
python -m scripts.summarize_cbf_controller_learning_curve --study-dir "$STUDY" --dev-labels "$ROOT/dev_shard0_labels.jsonl" "$ROOT/dev_shard1_labels.jsonl"
CUDA_VISIBLE_DEVICES=0 python -m tasks.cbf_ttt eval --model "$MODEL" --dtype bfloat16 --data "$STUDY/dev_scenarios.jsonl" --split dev --controller "$STUDY/controller_g1000_s42.pt" --policies baseline 0.5 1 controller --output "$STUDY/dev_g1000_rollouts.jsonl"
# dev 选出 g500_s42 后冻结模型；以下两条在两张卡上并行
CUDA_VISIBLE_DEVICES=0 python -m tasks.cbf_ttt eval --model "$MODEL" --dtype bfloat16 --data "$ROOT/shards/test_shard0.jsonl" --split test --controller "$STUDY/controller_selected.pt" --policies baseline 0.5 1 0.7321666666666666 controller --output "$STUDY/test_shard0_rollouts.jsonl"
CUDA_VISIBLE_DEVICES=1 python -m tasks.cbf_ttt eval --model "$MODEL" --dtype bfloat16 --data "$ROOT/shards/test_shard1.jsonl" --split test --controller "$STUDY/controller_selected.pt" --policies baseline 0.5 1 0.7321666666666666 controller --output "$STUDY/test_shard1_rollouts.jsonl"
cat "$STUDY/test_shard0_rollouts.jsonl" "$STUDY/test_shard1_rollouts.jsonl" > "$STUDY/test_all_rollouts.jsonl"
cat "$ROOT/shards/test_shard0.jsonl" "$ROOT/shards/test_shard1.jsonl" > "$STUDY/test_scenarios.jsonl"
python -m scripts.analyze_cbf_controller_rollouts --rollout "controller_g500=$STUDY/test_all_rollouts.jsonl" --scenarios "$STUDY/test_scenarios.jsonl" --output "$STUDY/test_comparison.json"
```

dev 的其余两种控制器各以 `--policies controller` 单独运行，训练均值常数也单独运行，再由 `scripts/analyze_cbf_controller_rollouts.py` 合并比较。脚本不修改任何 baseline 模型、预训练或原始评测路径。最终本地 10 项相关测试通过，远程 9 项子集测试及 1 项配对分析测试通过；训练、dev/test 完整 rollout、结果文件与 GPU 空闲状态均已核对。

### 与 Titans 的关系和创新性风险

[Titans 原论文（NeurIPS 2025）](https://proceedings.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf) 第 2.1 节式 (3) 已提出 `M_t=(1-α_t)M_{t-1}+S_t` 的自适应遗忘；附录 F 还明确将 TTT 层缺少遗忘机制列为 Titans 相对 TTT 的差异。因此，本项目不能把“在 TTT 快权重更新中加入可学习遗忘门”或该递推式本身作为独立创新点。Titans 的 `α_t` 是按通道的门，其 `S_t` 含梯度惊讶度及动量；此前遗忘实验是在既有 In-Place TTT 更新 `ΔW` 正常写入时，用共享系数衰减旧增量，并以未来问答损失的反事实候选比较构造离线监督标签。这里可研究的区别是**怎样用反事实任务收益监督决策**，不是门控记忆这个想法本身。下节记录用户进一步澄清的独立写入门控思路。

目前只有合成短尾场景的完整对照，且固定 `α=1` 在 dev/test 均优于控制器；所以这一训练方法尚未证明能带来超过简单遗忘策略的实际收益，也不能据此主张强方法创新。后续若要论证独立贡献，应先预注册损失敏感的控制目标，在自然且未见的长时距任务上比较固定系数、同结构无反事实监督的门控控制器、Titans 式数据依赖遗忘基线及本方法，并报告性能与额外标注计算成本。上述判断是基于论文公式与本次实验的研究定位，不代表已经复现或实测 Titans 模型。

## 双控制量语义修正（2026-09-27）

用户进一步澄清，原始动机还包括判定当前样本生成的 `ΔW_t` 是否值得写入：噪声或重复信息可能产生无用甚至有害的候选更新。它与“遗忘已有历史增量”是两个独立问题。旧附件描述的是遗忘版本，前述 3600 条标签和控制器也只验证了该版本；不能将旧 test 数字解释为写入门控的实验结果。按用户最新举例，联合式中的 `α_t` 定义为**旧记忆保留率**，新的方法定义见 [`WRITE_GATE_METHOD.md`](WRITE_GATE_METHOD.md)：`M_t=α_t M_(t-1)+g_tΔW_t`。旧 `forget` 标签和 CLI 的同名数值实际上是遗忘率 `f_t=1-α_t`，为保持历史 checkpoint 可复现，旧字段语义不变；`write` 模式固定 `α=1` 学习 `g`。`commit_both` 实现联合更新的数学操作，但**二维联合标签和双输出控制器未实现**。用户要求先讨论四种情形，本轮未启动任何新实验。

| 文件 | 本次修改 | 对应修正 | baseline 影响 |
|---|---|---|---|
| `cbf_ttt/runtime.py` | 将快记忆提交统一为独立的旧记忆衰减和当前候选写入两项；`write` 的 0/1 分别为跳过/完整写入 | 区分历史遗忘与当前样本采纳 | 旧 `forget` 默认路径不变 |
| `cbf_ttt/experiment.py`、`tasks/cbf_ttt.py` | 标签、训练 checkpoint、评测均保存或校验 `update_rule`；写入标签使用 `write_gate_star`，收益相对 `g=1` | 防止旧标签错用；形成写入反事实闭环 | 缺少字段的旧标签和 checkpoint 解释为 `forget` |
| `tasks/build_cbf_scenarios.py` | 新增 `--objective write`，第二块构造新事实/重复事实/无关噪声候选 | 让候选 `ΔW` 质量差异可观测 | 默认 `forget` 模板保持 |
| `tasks/eval_public_benchmarks.py`、`scripts/run_cbf_formal_labels.sh`、`scripts/summarize_cbf_formal_labels.py`、`scripts/prepare_cbf_controller_subsets.py`、`scripts/summarize_cbf_controller_learning_curve.py`、`scripts/analyze_cbf_controller_rollouts.py` | 传播模式、按正确参考系数汇总、拒绝混用 | 写入模式从采集到评测的实验入口 | 旧默认、旧结果文件兼容 |
| `tests/test_cbf_ttt.py`、`tests/test_cbf_scenarios.py`、`WRITE_GATE_METHOD.md` | 新增独立控制量、写入场景与收益参考测试；记录修正后的方法、假设和运行方式 | 正确公式及实验边界 | 仅测试和文档 |

写入实验应先按用户要求用少量源组、`0/0.5/1` 和稀疏边界做试点，检查每条标签耗时、峰值显存、平坦损失比例与门控标签分布后再决定正式规模。目前**尚未产生新的写入标签、训练写入控制器或完成其公开基准评测**。现有结果只能作为遗忘机制的历史对照。写入模板的主要风险是候选信息还保留在 attention KV 中，即使跳过快权重写入，近期问答仍可能直接读取它；需要更长时距或自然噪声任务来检验真实写入收益。双机制联合优化需要 3×3 等二维候选网格及相应双输出模型，不能由两套一维最优标签直接拼接得到。

### 四动作联合控制架构草案

新增 [`DUAL_GATE_DESIGN.md`](DUAL_GATE_DESIGN.md)，只记录方法设计，没有修改推理代码或启动实验。文档把四个角点的后续任务损失拆成旧记忆保留效应、当前写入效应和二者交互项，提出预测相对反事实损失的四动作控制器；同时给出由 `||αM+gΔW||²` 精确计算的快记忆容量约束。容量约束只能保证范数界，不能保证语义正确或不遗忘。四分支标签、联合控制器、容量过滤器均为待实现内容；开始标签构造前应先检查四种动作的可区分性与 attention KV 混杂。

### 双门控实验计划（未执行）

新增 [`DUAL_GATE_EXPERIMENT_PLAN.md`](DUAL_GATE_EXPERIMENT_PLAN.md)，按协议实现、两卡少量场景与 3×3 网格试点、阶段门、正式四角点标签、联合控制器、独立测试和六项公开基准安排实验。计划明确逐标签耗时/峰值显存、平坦标签与交互项、源组隔离、强固定角点对照、容量规则消融、长上下文限制和停止条件。用户此前要求先不开始实验；本次只写计划，没有运行任何新标签采集、控制器训练或模型评测。

### 双门控试点实现与执行状态（2026-09-27）

用户随后授权按上述计划进行实验。现已实现阶段 A 与阶段 B 所需入口，尚未把任何远程标签或性能数字当作完成结果。旧一维 `forget`/`write` 默认配置和已有标签文件保持原语义；联合标签使用独立的 `joint_v1` 协议，不能传给旧单输出控制器训练入口。

| 文件 | 修改内容 | 对应需求 |
|---|---|---|
| `cbf_ttt/experiment.py` | 新增 `collect_joint_labels`：同一已观测边界克隆 3×3 `(α,g)` 分支，统一未来 rollout，保存四角点、全网格损失、交互项、收益、特征、归一化范数二次型 `A/B/C`、逐标签耗时和显存峰值 | 两独立控制量的可辨性与资源试点 |
| `tasks/cbf_ttt.py` | 新增 `collect-joint` 命令与 `--grid`、`--every`、`--tie-tolerance` 参数 | 可配置稀疏边界采样 |
| `tasks/build_cbf_scenarios.py` | 新增 `--objective joint` 和四类成对场景：旧记忆相关/冲突 × 新候选有效/噪声，统一在第二块采集；记录候选类型与版本 2 | 四动作物理含义的反事实数据构造 |
| `scripts/run_cbf_joint_pilot.sh` | 两卡按 source-group 分片，各卡分别生成 train/dev/test 标签并拒绝覆盖已有文件 | 48 场景试点 |
| `scripts/summarize_cbf_joint_pilot.py` | 汇总平坦比例、四角点分布、交互量、内部网格点收益与时间/显存 | 决定是否扩大正式规模 |
| `tests/test_cbf_ttt.py`、`tests/test_cbf_scenarios.py`、`tests/test_cbf_joint_pilot_summary.py` | 添加联合更新、场景构造、9 点采集和汇总检查 | 接口与数据正确性 |
| `DUAL_GATE_EXPERIMENT_PLAN.md`、`MODIFICATION_LOG.md` | 更新执行状态和运行说明 | 可复现实验记录 |

试点命令（在含 PyTorch、模型和 1B 混合语料的远程环境中运行）：

```bash
ROOT=/home/ctj/cbf_ttt_joint_pilot_20260927
MODEL=/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt
mkdir -p "$ROOT"
python -m tasks.build_cbf_scenarios --tokenizer /home/ctj/models/Qwen3-4B --output "$ROOT/scenarios.jsonl" --background-data /home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl --objective joint --future-mode short_tail --chunk-size 4096 --context-chunks 2 --future-chunks 1 --futures-per-scenario 1 --train-groups 8 --dev-groups 2 --test-groups 2 --variants-per-group 4 --seed 42
python -m scripts.shard_cbf_scenarios --input "$ROOT/scenarios.jsonl" --output-dir "$ROOT/shards" --shards 2
CUDA_VISIBLE_DEVICES=0 bash scripts/run_cbf_joint_pilot.sh "$MODEL" "$ROOT" 0 0
CUDA_VISIBLE_DEVICES=1 bash scripts/run_cbf_joint_pilot.sh "$MODEL" "$ROOT" 1 1
python -m scripts.summarize_cbf_joint_pilot --labels "$ROOT"/*_joint_labels.jsonl --output "$ROOT/summary.json"
```

上面两条 `run` 命令需要并发运行；脚本自身设置 `CUDA_VISIBLE_DEVICES`，外层变量并非必须。先用单场景运行 `collect-joint` 做模型 smoke，再启动全试点。尚未完成阶段 C 正式标签、联合控制器训练、独立测试及公开基准评测；必须等试点证明四动作可辨且资源可承受。风险包括近期答案经 attention KV 泄露导致四角点近乎同分、bf16 数值波动、9 分支导致显存或耗时增加，以及每个合成源组的多个变体高度相关。部署到远程时需确认 checkpoint、tokenizer、语料路径及 conda 环境。当前 SSH 经 Cloudflare 代理时有间歇性握手超时；因此远程实验状态只能以实际输出文件和进程日志确认。

执行中发现并修复 `collect-joint` 的 CLI 入口错误：该子命令不定义 `--controller`，通用加载段原先直接访问 `args.controller` 导致模型加载后抛出 `AttributeError`；现使用 `getattr(..., None)`。该修复只影响未提供控制器参数的子命令，旧 `collect`/`eval`/`generate` 参数仍按原值使用。hku-gpu2 已从 GitHub commit `496f7ca` 克隆到 `/home/ctj/cbf_ttt_joint_exp_20260927`，远程 20 项测试通过；已生成 48 场景及两卡分片，首次单场景 smoke 因上述入口错误中断，未产出标签。修复同步到远程并通过 smoke 后才能启动全试点。

### 双门控试点结果与阶段门（2026-09-27）

上述 CLI 修复已推送为 `aded608`，hku-gpu2 拉取后单场景 smoke 通过。两张 5090 在 tmux 中完成 12 个源组 × 4 类场景 = **48 条**联合 3×3 标签，没有 OOM；平均每条 **1.367 秒**，最大 allocated/reserved 分别 **12.650/13.861 GiB**。同一源组四种情景重跑后 4×9 个损失逐值一致。`scripts/summarize_cbf_joint_pilot.py` 新增四角点平均损失、源组配对 `J11−J00`、内部最大收益和能量项统计，`tests/test_cbf_joint_pilot_summary.py` 同步扩展 fixture。

完整结果见 [`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/joint_pilot/REPORT.md`](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/joint_pilot/REPORT.md) 与同目录 `summary.json`、原始标签。四角点最优分布 `00/01/10/11=37/2/7/2`，内部点有可辨优势 10/48，12/12 个源组的固定 `00` 平均损失低于 `11`；因此预设阶段门**未通过**，暂停正式标签、联合控制器和公开基准。它是场景协议不适合训练目标的证据，并非模型在真实任务中应一律关闭快记忆的证明。下一步优先修订 held-out 后续任务、长间隔与 KV 混杂控制，再做同规模试点。

原始标签 JSONL 含预训练背景派生的完整逐样本 payload，自动审批拒绝将其上传到 GitHub。它们保留在本机实验目录和 hku-gpu2 输出目录，由该目录 `.gitignore` 排除；GitHub 仅提交代码、构造元数据、聚合 `summary.json` 与本报告。若需对外发布完整标签，需单独确认数据授权与分享范围。

### 修订试点 `joint_v2`（2026-09-27，进行中）

按用户要求继续此前报告提出的顺序。新增独立协议，保持原 `joint_v1` 和 baseline 不变。`tasks/build_cbf_scenarios.py` 新增 `--objective joint_v2`、`--joint-gap-chunks` 与未见票号的规则迁移场景；构造器检查目标规则答案未出现在 context/gap，候选保持同主题和同完整 chunk 长度。`cbf_ttt/experiment.py` 对同一决策分别记录短间隔与自然背景 gap 之后的每查询损失、每 future 损失、原 3×3 聚合损失以及时间和显存。`tasks/cbf_ttt.py` 区分 `joint_v1`/`joint_v2` 协议；`scripts/summarize_cbf_joint_pilot.py` 增加按 gap 和查询类型的四角点统计。`tests/test_cbf_scenarios.py`、`tests/test_cbf_ttt.py`、`tests/test_cbf_joint_pilot_summary.py` 覆盖新数据和输出形状。具体假设、阶段门及命令见 [`DUAL_GATE_V2_EXPERIMENT.md`](DUAL_GATE_V2_EXPERIMENT.md)。若仍由 `00` 垄断，不扩量也不训练控制器。

远程 PyTorch 测试最初发现旧 tiny fixture 的 query 没有 `kind` 字段；扩展标签记录时直接读取它导致 `KeyError`。`collect_joint_labels` 现为旧数据使用 `unspecified` 占位，新 `joint_v2` 仍保存明确的查询类型。此修复保证历史 `joint_v1` 标签入口兼容。

新增 `scripts/evaluate_cbf_joint_v2_gate.py` 和 `tests/test_cbf_joint_v2_gate.py`，将执行前写定的 0.005 NLL 阈值、按源组比较、四角点获胜多样性、内部系数比例及显存余量转换为可复算的 `gate.json`；脚本拒绝缺失或跨场景不匹配的源组。它只使用逐条标签计算聚合指标，不把原始标签上传到 GitHub。

新增 `scripts/audit_cbf_joint_v2_scenarios.py`：使用实际 Qwen tokenizer 解码 context 与 gap，逐项验证规则迁移查询的目标答案字符串没有出现在可见输入中，并只输出聚合检查数；中性算术题单独计数。这补充了构造器的 token 子序列检查。

v2 长间隔标签出现写入动作 NLL 急升后，将 `tests/test_cbf_ttt.py` 中原生 Qwen TTT 与 CBF `11` 的一致性检查从 2 个扩展到 4 个完整 chunk；同时核查本轮使用的 192 条自然背景记录经 Qwen tokenizer 编码均为 6143 token，因此 `_fit_chunk` 只截取而未重复平铺背景。该排查用于区分协议/模型行为和快记忆实现错误，不能单凭测试推断实际 4B checkpoint 的全部数值性质。

#### `joint_v2` 完成结果与停止决定

完整记录见 [`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/joint_v2_pilot/REPORT.md`](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/joint_v2_pilot/REPORT.md)。两卡 48/48 条标签于约 6 分 20 秒墙钟完成；每条平均 14.915 秒，峰值 CUDA allocated/reserved 为 14.828/18.564 GiB。实际 tokenizer 审计的 96 个规则迁移目标均不在可见输入中；四类场景重跑逐查询损失最大差 0。本地按执行前阈值重算的 `gate.json` 与远程逐字节一致。聚合四角点 `00/01/10/11` 最佳次数为 `48/0/0/0`；长间隔 `47/0/1/0`，有用新信息写入收益为 0/12 组，旧规则保留收益为 0/12 组，阶段门**未通过**。真实最终模型的一例原生 `11` NLL 8.6255 与 CBF `11` NLL 8.6567 接近但非完全相同，显示大损失不只是 CBF 分支特有；还需更多精度/状态对照。正式标签、联合控制器与公开基准按计划暂停，不将退化的 `00` 分布用于训练。

本轮 GitHub 仅上传 `REPORT.md`、`summary.json`、`gate.json`、`leakage_audit.json` 和场景元数据。原始标签在本机及 hku-gpu2 保存并由 `.gitignore` 排除，以遵守此前针对完整派生数据上传的自动审批拒绝。

### 后续写入机制与自然 continuation 诊断（已完成）

按用户继续实验的要求，新增 [`DUAL_GATE_MECHANISM_PILOT.md`](DUAL_GATE_MECHANISM_PILOT.md) 固定诊断协议。`scripts/diagnose_cbf_gap.py` 从同一联合候选状态分支，对四角点分别比较 gap chunk 后续 `normal_11` 与 `freeze_10` 路径，记录 gap0/1/2 的逐查询 NLL、快记忆范数和资源耗时，用于定位 `joint_v2` 长间隔损失增幅。`tasks/build_cbf_natural_scenarios.py` 从 1B 混合语料的不同记录切出 4096-token 已见前缀和 32+128-token 未见 continuation，按 `both_relevant/old_only/new_only/neither_relevant` 四条件分组；它复用原有分片及 3×3 collector。`cbf_ttt/experiment.py`、`tasks/cbf_ttt.py`、`scripts/summarize_cbf_joint_pilot.py` 识别独立 `joint_natural_v1` 协议并按查询类型汇总。`tests/test_cbf_ttt.py`、`tests/test_cbf_natural_scenarios.py` 增加路径与数据校验。自然记录来自已训练的语料，只能做机制试点；原始 baseline 默认路径不变。

新增 `scripts/summarize_cbf_gap_diagnostic.py`，计算各候选动作及后续策略在 gap0/1/2 的平均 NLL、快记忆范数比率和资源开销；新增 `scripts/evaluate_cbf_natural_gate.py`，按执行前 `0.005` NLL 和至少 3 个独立源组的门槛判断是否扩大标签。`tests/test_cbf_natural_gate.py` 检查源组去重、四条件完整性及轨迹完整性。两套脚本只输出聚合结果，不公开逐样本 payload。

hku-gpu2 的最终 Qwen3-4B checkpoint 上，`joint_v2` 同一源组的四类诊断表明 `11` 在正常后续写入下 gap0/1/2 平均 NLL 为 **0.669/2.238/8.848**，冻结后续写入则为 **0.669/1.105/1.776**。自然试点用 1B 混合语料的 48 条不同记录构造 12 源组 × 4 条场景，两张 5090 并行完成 48 条 3×3 标签，平均每条 **1.079 秒**，最大显存 reserved **13.854 GiB**。四角点最优 `00/01/10/11=33/7/7/1`，但有用新信息条件下 `11` 对 `00` 超过 0.005 NLL 的独立源组只有 **2/12**，未达到预设 3/12；阶段门未通过。完整过程、命令、算法含义及限制见 [`自然试点报告`](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/joint_natural_pilot/REPORT.md)，机器可读结果为同目录 `summary.json`、`gate.json` 与 `joint_v2_pilot/gap_diagnostic_summary.json`。

因此本轮**未**扩量正式反事实标签、训练双输出控制器或运行公开基准。自然文档来自当前 checkpoint 用过的预训练语料，目标只在本次 session 中未见，不能推断对全新文档的泛化。四类 gap 诊断来自同一源组，不能推断群体效应。原始逐样本标签和轨迹在本机忽略目录与远程输出目录保留，GitHub 只记录代码、场景元数据及聚合报告。后续 TODO：构造真正未参与预训练的任务相关文档，核查候选 `ΔW` 与任务增益的关系，并在新源组上预注册且复核阶段门；已有探索性 dev/test 不再用于正式结论。原始 baseline 路径未修改。

### 发布后文档与独立写入效应试点（2026-09-27 至 28，已完成）

为执行上述 TODO，先固定 [`DUAL_GATE_POSTCUTOFF_PILOT.md`](DUAL_GATE_POSTCUTOFF_PILOT.md)：从 Qwen3 发布之后首投的 arXiv `cs.CL` 论文 PDF 构造 12 个四文档源组，继续用 4096+32+128 token、第二边界和 3×3 网格。发布日期是来源代理；仅能保证公开的已发布基础权重不可能学到未来提交版本，不能保证没有此前公开的相似片段。正式控制器仍受预设阶段门约束。

| 文件 | 改动 | 对应需求及 baseline 影响 |
|---|---|---|
| `scripts/download_postcutoff_arxiv.py` | 按首投时间和类别查询官方 API，缓存 PDF，提取文本并筛选 Qwen token 长度；记录 ID、日期、URL 和哈希，不把全文写入仓库 | 构造可核验来源的新文档；opt-in 脚本 |
| `scripts/audit_postcutoff_corpus.py` | 一次性扫描 1B 续训语料中的入选论文标题，报告精确匹配数 | 辅助排查数据重合；只读 |
| `tasks/build_cbf_natural_scenarios.py` | 新增独立 `joint_postcutoff_v1` 协议及可配源组前缀，复用原四条件构造 | 避免与已见自然场景标签混用；默认旧协议不变 |
| `cbf_ttt/experiment.py`、`tasks/cbf_ttt.py`、`scripts/summarize_cbf_joint_pilot.py` | 传播新协议至采集、CLI 和汇总 | 复用既有联合分支；baseline 不变 |
| `scripts/evaluate_cbf_postcutoff_gate.py` | 固定 `α=1` 比较 `g=0/1` 的写入效应，固定 `g=0` 比较 `α=0/1` 的保留效应，按独立源组执行预设阶段门 | 防止 `11` 对 `00` 同时改变两轴的混杂 |
| `tests/test_cbf_postcutoff.py`、`tests/test_cbf_natural_scenarios.py`、`tests/test_cbf_ttt.py` | 检查 PDF 清理、标题审计、源组判据、新协议场景与采集闭环 | 只新增验证 |

原始 PDF、提取文本、场景及逐样本标签由实验目录 `.gitignore` 排除；GitHub 仅保存代码、来源元数据和聚合结果。

hku-gpu2 对 arXiv 官方元数据 API 返回 HTTP 406，而本机可从同一 URL 获取、远程可下载官方 PDF。`scripts/download_postcutoff_arxiv.py` 因此增加 `--api-feed`：接收本机保存并传至远程的原始 Atom XML，核对查询类别和日期，在来源元数据中记录其 SHA256；数据筛选顺序和预注册阶段门均不变。

来源预检成功获取 48 篇论文并完成一次两卡 48 条标签链路，但复核发现下载器当时按 API 时间排序取最先合格的 48 篇，**不符合预设固定种子抽样**。发现时尚未查看该批损失结果。`scripts/download_postcutoff_arxiv.py` 现先用种子 118 打乱 API 前 300 条，再按长度取 48 篇，并记录抽样方法和种子；另加可复用 PDF 缓存参数。原轮仅作为资源/代码预检，阶段门将在新目录的随机样本上独立执行，阈值不变。

#### 随机主试点结果与阶段门

两卡完成随机主试点 48/48 条标签，无 OOM。来源漏斗为 51 篇尝试、48 篇合格、3 篇长度不足，首投日期 2026-09-21 至 09-24；1B 续训语料中的精确标题匹配 0/48。每条标签平均 **1.075 秒**、p95 **1.423 秒**，最大 CUDA reserved **13.854 GiB**。固定 `α=1` 比较 `g=0/1`，`new_only` 中明显有益写入仅 **1/12** 源组，平均写入收益 **−0.01193 NLL**；无关写入损害 **12/12** 源组。固定 `g=0` 比较旧记忆，保留旧记忆有益 **2/12**、清除有益 **6/12**；写入、保留和联合阶段门均未通过。两个训练源组的 float32 四角点复核保留 bf16 写入效应符号，峰值 reserved 27.320 GiB，但不代表全部样本。完整模型、来源、哈希、命令和限制见 [`随机主试点报告`](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/joint_postcutoff_random_pilot/REPORT.md)，机器可读结果见同目录 `documents.jsonl.meta.json`、`scenarios.jsonl.meta.json`、`title_audit.json`、`summary.json` 和 `gate.json`。

正式反事实标签、双输出控制器训练及公开基准继续暂停。未来工作先排查当前 `ΔW` 的目标相关性和更新尺度，再用新论文/任务预注册有正写入样本的机制试点；已查看的探索性 dev/test 不作为正式测试。公开论文首投晚于 Qwen3 发布日是强来源代理，但不能证明此前无相似草稿；精确标题扫描不等于全文去重。PDF 转文本与 12 源组样本量也限制结论外推。所有新增逻辑均由独立脚本/协议启用，baseline 配置不受影响。

### 候选 `ΔW` 方向与尺度诊断（2026-09-28，已完成）

用户要求继续上述机制排查。新增 [`DUAL_GATE_SCALE_DIAGNOSTIC.md`](DUAL_GATE_SCALE_DIAGNOSTIC.md)，在看结果前固定 8 个训练源组的 `new_only/old_only` 条件和尺度网格 `−2,−1,−0.5,0,0.25,0.5,1,2`。`scripts/diagnose_cbf_update_scale.py` 对同一已观测状态仅在独立克隆中应用超出门控范围的临时尺度，不修改 `CBFSession.commit_both` 或 baseline；`s=0/1` 与已采集的 `10/11` 标签逐值校验。`scripts/summarize_cbf_update_scale.py` 按源组汇总正负尺度收益、最优尺度、候选范数、surprise、时间和峰值显存。`tests/test_cbf_ttt.py`、`tests/test_cbf_update_scale.py` 覆盖参考标签一致性和汇总完整性。负尺度只用来诊断候选方向，不是拟部署的更新规则。

远程 tiny-model 测试验证 `s=0/1` 与旧标签一致，但其某个随机初始化模型的候选范数恰为零，故原测试断言“范数必须大于零”过强；已改为非负检查。实际 4B 模型的尺度结果仍需另行验证。

最终 4B 模型的 16 条 bf16 轨迹全部通过 `s=0/1` 参考标签校验；为核查细微差异，同一批又完成 float32 全尺度复跑。`new_only` 在 `s=0/0.25/1/2` 的平均 NLL，bf16 为 **1.99441/1.99468/2.00362/2.04651**，float32 为 **1.99348/1.99391/2.00135/2.04441**。float32 下负尺度超过 0.005 NLL 的有益源组为 0/8，正尺度为 2/8；降低 `g` 虽缓解全写入损害，仍未建立平均正收益。bf16/float32 每条平均 **1.213/3.832 秒**，峰值 reserved **14.041/30.371 GiB**。两份聚合分别存放于随机主试点目录的 `update_scale_summary.json` 和 `update_scale_fp32_summary.json`，本地复算与远程逐字节一致，报告已补充原始假设、方法映射和限制。既定写入/保留阶段门未改变，正式标签与控制器训练继续暂停；后续需设计能真正检验新文档信息复用的任务，再用新源组预注册试点。
### 发布后论文标题回忆与 KV 隔离试点（2026-09-29，已完成）

本轮目标是在新源组上检验当前候选 `ΔW` 是否能承载新读论文的标题。此前紧接文档的 continuation 目标可能直接由 attention KV 提供，因而新增并列的 KV 完整、KV 清空但保留 session 快记忆两种读取条件。算法仍为 `M'=αM+gΔW`，固定第二个 4096-token chunk 的决策；后一条件是机制诊断，不是部署默认路径。四类场景及看标签前固定的阶段门见 [`DUAL_GATE_TITLE_RECALL_PILOT.md`](DUAL_GATE_TITLE_RECALL_PILOT.md)。如果门槛未过，不构造正式标签或训练控制器。

| 文件 | 具体修改 | 对应研究需求及 baseline 影响 |
|---|---|---|
| `cbf_ttt/runtime.py` | 新增 `clone_memory_only()`，复制快记忆张量并创建空 KV 的独立分支 | 分离快权重与 attention KV；仅显式调用生效 |
| `cbf_ttt/experiment.py` | 新增 `joint_title_recall_v1` 协议，检查两个 future 相同查询且按 KV 完整/清空顺序，记录逐 future NLL、`reset_kv`、时间和显存 | 对同一 `(α,g)` 比较两种读取条件；旧协议默认不变 |
| `tasks/cbf_ttt.py`、`scripts/summarize_cbf_joint_pilot.py` | 接受新协议，并将两类 future 分别汇总为 `kv_intact`/`memory_only` | 闭环 CLI 和聚合；旧协议字段及含义不变 |
| `scripts/download_postcutoff_arxiv.py` | 新增可重复的 `--exclude-metadata`、`--require-title-in-prefix`；记录排除集合哈希和标题检查 | 构造不重复且标题确实可见的新论文；默认下载行为不变 |
| `tasks/build_cbf_title_recall_scenarios.py` | 新建四论文源组和四条件标题查询，验证 ID 唯一、API/PDF 标题在首块匹配、组内标题唯一、两读取条件同题 | 数据构造；独立 opt-in 协议 |
| `scripts/evaluate_cbf_title_recall_gate.py` | 新建仅用 `memory_only` 四角点和完整源组判断的阶段门，同时报告 KV 完整条件 | 防止 KV 混杂与按场景伪重复 |
| `tests/test_cbf_ttt.py`、`tests/test_cbf_title_recall.py`、`tests/test_cbf_joint_pilot_summary.py` | 检查空 KV 与快记忆独立复制、tiny-model 采集闭环、构造、阶段门及两读取条件的分别汇总 | 验证新增路径；不改 baseline |
| `DUAL_GATE_TITLE_RECALL_PILOT.md`、本文件 | 预注册假设、实验顺序、阈值、风险、命令和结果 | 实验可复核 |

远程命令（目录 `/home/ctj/cbf_ttt_joint_exp_20260927`，环境 `/home/ctj/miniconda3/envs/cbf_ttt_train_py311`，模型 `/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt`）：

```bash
export PATH=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin:$PATH
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_title_recall_pilot_20260929
mkdir -p "$ROOT"
python -m scripts.download_postcutoff_arxiv --tokenizer /home/ctj/models/Qwen3-4B --output "$ROOT/documents.jsonl" --api-feed /home/ctj/cbf_ttt_postcutoff_pilot_20260927/api_feed.xml --pdf-cache /home/ctj/cbf_ttt_postcutoff_pilot_20260927/pdf_cache --max-results 300 --target 48 --selection-seed 119 --require-title-in-prefix 4096 --exclude-metadata /home/ctj/cbf_ttt_postcutoff_pilot_20260927/documents.jsonl.meta.json --exclude-metadata /home/ctj/cbf_ttt_postcutoff_random_pilot_20260927/documents.jsonl.meta.json
python -m scripts.audit_postcutoff_corpus --corpus /home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl --metadata "$ROOT/documents.jsonl.meta.json" --output "$ROOT/title_audit.json"
python -m tasks.build_cbf_title_recall_scenarios --tokenizer /home/ctj/models/Qwen3-4B --data "$ROOT/documents.jsonl" --output "$ROOT/scenarios.jsonl" --seed 120
python -m scripts.shard_cbf_scenarios --input "$ROOT/scenarios.jsonl" --output-dir "$ROOT/shards" --shards 2
bash scripts/run_cbf_joint_pilot.sh /home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt "$ROOT" 0 0
bash scripts/run_cbf_joint_pilot.sh /home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt "$ROOT" 1 1
python -m scripts.summarize_cbf_joint_pilot --labels "$ROOT"/*_joint_labels.jsonl --output "$ROOT/summary.json"
python -m scripts.evaluate_cbf_title_recall_gate --labels "$ROOT"/*_joint_labels.jsonl --output "$ROOT/gate.json"
```

两条 `run` 命令需分别在两张 GPU 的 tmux 窗口并发执行；正式启动前先对单场景 smoke。以上构建会检查标题在首块出现，若不匹配则继续筛选下一篇论文。主要风险是 PDF 提取标题差异、标题 NLL 不代表自由生成、KV 清空后任务过难、48 篇论文仍可能与旧公开草稿相似。原始 PDF、全文、场景和逐条标签仅留在远程或本地忽略目录，不上传 GitHub。

#### 完成结果与停止决定

远程 PyTorch/纯数据 12 项测试通过；48 篇新论文来源与前两批入选的 88 个唯一 ID 不重叠，53 篇尝试后得到 48 篇，1B 语料精确标题匹配 0/48。单场景 smoke 1.670 秒、峰值 reserved 14.094 GiB。两卡约 51 秒完成 48 条 3×3 网格标签，平均每条 1.255 秒、p95 1.689 秒，最大 reserved 14.098 GiB，无 OOM。本地从逐条标签复算的聚合文件与远程 SHA256 一致。完整来源、命令、汇总与限制见 [`标题回忆试点报告`](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/joint_title_recall_pilot/REPORT.md)，机器可读结果为同目录的 `summary.json`、`gate.json`、`title_audit.json` 和来源元数据；逐条标签由 `.gitignore` 排除。

预注册的 `memory_only` 阶段门未通过：`new_only` 有益写入 **1/12** 组，平均收益 **−0.02139 NLL**；无关写入有害 **8/12**；旧记忆保留有益 **0/12**，清除有益 **9/12**。KV 完整与仅快记忆的 `new_title` NLL 约 0.58 和 4.94，但清空 KV 后并未出现稳定正写入收益。按阶段门暂停正式标签、双输出控制器及公开基准对照。下一步需改进候选 `ΔW` 的信息写入目标或机制，再用新源组验证；不能将本轮 `neither` 未见标题视为可回答的问题。所有新增入口为 opt-in，原始 baseline 不受影响。
### 标题答案梯度与原始候选方向诊断（2026-09-29，已完成）

标题回忆阶段门失败后，继续定位原因。原 In-Place TTT 候选是当前 chunk 表示外积，不直接对未来标题任务求梯度。本轮固定现有最终 Qwen3-4B、8 个 train 源组的 `new_only/old_only` 共 16 条场景；在 `g=0` 且仅快记忆的读取状态计算标题 NLL 对 `M` 的梯度，记录它与原始 `ΔW` 的内积/余弦，并比较等范数原始方向与答案梯度 oracle 方向的 `s=0.25/1` 局部扰动。答案梯度只用于事后容量探针，不可作为在线控制器输入；本轮不训练控制器、不改变既有阶段门。具体分析假设和判据见 [`DUAL_GATE_QUERY_GRADIENT_DIAGNOSTIC.md`](DUAL_GATE_QUERY_GRADIENT_DIAGNOSTIC.md)。

| 文件 | 修改 | baseline 影响 |
|---|---|---|
| `scripts/diagnose_cbf_query_gradient.py` | 新增独立反事实梯度探针；只把克隆快记忆设为可求导叶子，核对 `s=0/1` 与已有标题标签，记录梯度方向、两扰动损失、耗时和峰值显存 | opt-in；原模型和运行时更新公式不变 |
| `scripts/summarize_cbf_query_gradient.py` | 按完整 train 源组汇总方向、收益和资源，并执行预先写定的机制解释条件 | 仅离线聚合 |
| `tests/test_cbf_ttt.py`、`tests/test_cbf_query_gradient.py` | tiny Qwen 梯度/参考标签闭环与纯数据汇总完整性检查 | 仅测试 |
| `DUAL_GATE_QUERY_GRADIENT_DIAGNOSTIC.md`、本文件 | 固定样本、判据、风险、执行命令和结果 | 仅文档 |

远程命令（先用 `--max-scenarios 1` 单场景 smoke；去掉该选项运行 16 条，输出换新路径）：

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
export PATH=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin:$PATH
ROOT=/home/ctj/cbf_ttt_title_recall_pilot_20260929
OUT=/home/ctj/cbf_ttt_query_gradient_20260929
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=0 python -m scripts.diagnose_cbf_query_gradient --model /home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt --data "$ROOT/scenarios.jsonl" --split train --reference-labels "$ROOT/train_shard0_joint_labels.jsonl" "$ROOT/train_shard1_joint_labels.jsonl" --output "$OUT/diagnostic.jsonl"
python -m scripts.summarize_cbf_query_gradient --input "$OUT/diagnostic.jsonl" --output "$OUT/summary.json"
```

执行前先跑 tiny Qwen 测试和一条 GPU smoke；若 autograd 与 inference NLL 或已有标签不一致，停止解释轨迹。答案梯度泄露未来标签，不能用 oracle 收益代表实际可部署的适应方法。原始诊断逐条记录留在本地/远程忽略目录，只公开聚合结果。

远程 tiny Qwen 首次测试触发测试 fixture 的零初始化 TTT 卷积，使候选范数为零；仅对该随机 tiny fixture 设置非零卷积权重后 10 项测试通过。真实最终模型的一条 `old_only` smoke 已计算并写入结果，`s=0/1` 与旧标签一致、梯度 NLL 与推理 NLL 差小于 `2e-7`，峰值 reserved 14.330 GiB；随后 CLI 打印阶段因 `Path` 未转字符串报错，未影响该条数值。已修正输出序列化，并用新输出文件确认命令成功退出。

修正后的单场景命令成功退出。两卡完成 8 个 train 源组的 16 条诊断，所有 `s=0/1` 均核对既有标签，平均每条 0.970 秒、最大 reserved 16.031 GiB。`new_only` 原始方向局部导数为正 7/8，原始写入 `s=.25/1` 平均收益为 −0.00907/−0.01764 NLL；等范数答案梯度 oracle 则为 +3.15839/+1.15395 NLL，8/8 组至少一个步长改善 >0.005。梯度与候选平均余弦仅 +0.000265，数值极小，故仅支持本任务的“候选与答案目标不对齐”机制解释，不能推断强稳定方向相关，也不能把使用答案的 oracle 当成可部署方法。`old_only` 的 oracle 大步长有过冲。详细结果、命令和下一步训练目标建议见 [`梯度机制诊断报告`](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/query_gradient_diagnostic/REPORT.md) 与同目录 `summary.json`；逐条记录由 `.gitignore` 排除。原控制器阶段门未改变，正式标签和控制器训练仍暂停。
### 内容驱动低秩写入器关键试点（2026-09-29，已完成）

用户要求完成下一步关键实验：验证仅从当前内容生成的更新能否在未见源组产生稳定写入收益。执行前固定 [`TASK_AWARE_WRITER_PILOT.md`](TASK_AWARE_WRITER_PILOT.md)：从原论文 API feed 排除前三批已用论文，构造新的 64/16/16 篇 train/dev/test。每篇一个 4096-token chunk，使用空旧记忆生成原始候选；以 rank-8 的 `D'=D+B(AD)` 对每层候选做可训练变换并限制 `||D'||≤||D||`。骨干、原卷积和投影冻结，初始 B=0 等价于 baseline。标题答案只作为训练 NLL 监督，写入器输入只有当前 chunk 的候选 D。

| 文件 | 修改内容 | 对 baseline 的影响 |
|---|---|---|
| `cbf_ttt/writer.py` | 新增 `LowRankWriter` 范数受限低秩候选变换、可对快记忆求导的 `memory_nll` | 全新 opt-in 模块；不改原运行时 |
| `tasks/build_cbf_writer_episodes.py` | 构造单论文 episode、源组划分、标题可见性校验及哈希清单 | 新任务协议 `task_writer_v1` |
| `tasks/cbf_writer.py` | `extract/train/test` 三入口，当前内容候选缓存、5 epoch 训练/dev 选型、dev 通过后才允许 test；保存逐步时间/显存、候选哈希、checkpoint 哈希 | 不改 baseline 参数或配置 |
| `tests/test_cbf_writer.py`、`tests/test_cbf_writer_episodes.py` | 验证初始等价、范数上限、候选 detach、骨干冻结、训练闭环、dev 失败禁止 test、源组与查询标签隔离 | 仅测试 |
| `scripts/audit_cbf_writer_results.py` | 从元数据与逐条输出独立核对来源排除/划分、训练预算、dev 选择与 test 聚合和阶段门 | 只读审计及聚合输出 |
| `TASK_AWARE_WRITER_PILOT.md`、本记录 | 固定预算、数据来源、对照、阶段门和风险 | 仅文档 |

默认配置为 rank=8、seed=123、AdamW lr=0.001/weight_decay=0.01、batch=1、clip=1、5 epochs。dev 在 `g=.5/1` 选模型和门值；需对无写入和最佳原始更新平均改善都 >0.005 且 ≥8/16 篇改善 >0.005，才冻结并测试。独立 test 增加错配论文和训练集平均更新对照，并记录 `g=0/.5/1`；阶段门只使用 dev 冻结的门值。单论文/标题/KV 清空只检验写入，不足以证明双门控有效。

运行命令（hku-gpu2，conda `cbf_ttt_train_py311`，代码目录 `/home/ctj/cbf_ttt_joint_exp_20260927`）：

```bash
ROOT=/home/ctj/cbf_ttt_writer_pilot_20260929
MODEL=/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt
python -m scripts.download_postcutoff_arxiv --tokenizer /home/ctj/models/Qwen3-4B --output "$ROOT/documents.jsonl" --api-feed /home/ctj/cbf_ttt_postcutoff_pilot_20260927/api_feed.xml --pdf-cache /home/ctj/cbf_ttt_postcutoff_pilot_20260927/pdf_cache --max-results 300 --target 96 --selection-seed 121 --require-title-in-prefix 4096 --exclude-metadata /home/ctj/cbf_ttt_postcutoff_pilot_20260927/documents.jsonl.meta.json --exclude-metadata /home/ctj/cbf_ttt_postcutoff_random_pilot_20260927/documents.jsonl.meta.json --exclude-metadata /home/ctj/cbf_ttt_title_recall_pilot_20260929/documents.jsonl.meta.json
python -m scripts.audit_postcutoff_corpus --metadata "$ROOT/documents.jsonl.meta.json" --corpus /home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl --output "$ROOT/title_audit.json"
python -m tasks.build_cbf_writer_episodes --data "$ROOT/documents.jsonl" --output "$ROOT/episodes.jsonl" --tokenizer /home/ctj/models/Qwen3-4B
CUDA_VISIBLE_DEVICES=0 python -m tasks.cbf_writer extract --model "$MODEL" --data "$ROOT/episodes.jsonl" --features "$ROOT/features" --shards 2 --shard 0
CUDA_VISIBLE_DEVICES=1 python -m tasks.cbf_writer extract --model "$MODEL" --data "$ROOT/episodes.jsonl" --features "$ROOT/features" --shards 2 --shard 1
CUDA_VISIBLE_DEVICES=0 python -m tasks.cbf_writer train --model "$MODEL" --data "$ROOT/episodes.jsonl" --features "$ROOT/features" --output "$ROOT/smoke" --smoke
CUDA_VISIBLE_DEVICES=0 python -m tasks.cbf_writer train --model "$MODEL" --data "$ROOT/episodes.jsonl" --features "$ROOT/features" --output "$ROOT/train_seed123"
# test 入口会核查 dev gate 与冻结 checkpoint/data 哈希。
CUDA_VISIBLE_DEVICES=0 python -m tasks.cbf_writer test --model "$MODEL" --data "$ROOT/episodes.jsonl" --features "$ROOT/features" --checkpoint-dir "$ROOT/train_seed123" --output "$ROOT/test_seed123"
```

两条 extract 在两个 tmux 窗口并行运行；smoke 成功后运行完整训练，只有 dev gate 通过才运行 test。PDF、文本、特征张量、checkpoint 和逐样本输出只保留远程或本地忽略目录；GitHub 仅提交代码、公开来源元数据和聚合结果。

新数据已完成：排除前三批 136 个唯一论文 ID，106 篇尝试中 7 篇长度不足、3 篇首块标题不匹配，得到 96 篇；精确标题扫描 1B 续训语料为 0/96。两卡各提取 48 篇，候选缓存总计约 32 GiB。远程 tiny 模型与数据测试通过，补充成功 dev gate 的测试分支以验证错配/平均更新对照的输出闭环。真实训练 smoke：286,720 参数、单步 0.442 秒、最大 reserved 11.551 GiB；初始 writer 与原始 `g=.5/1` 的 dev NLL 逐值一致。随后完成 64 篇 ×5 epoch 的固定预算训练。

#### 完成结果与阶段门

独立复核命令（使用远程保留的逐条结果，不重新评分或训练）：

```bash
python -m scripts.audit_cbf_writer_results --root /home/ctj/cbf_ttt_writer_pilot_20260929 --prior-metadata /home/ctj/cbf_ttt_postcutoff_pilot_20260927/documents.jsonl.meta.json /home/ctj/cbf_ttt_postcutoff_random_pilot_20260927/documents.jsonl.meta.json /home/ctj/cbf_ttt_title_recall_pilot_20260929/documents.jsonl.meta.json --output /home/ctj/cbf_ttt_writer_pilot_20260929/audit.json
```

320 步训练完成。dev 选择 epoch 1、g=1，NLL 4.37044，相对无写入改善 0.58407、15/16 篇改善 >0.005，通过预设 dev 门槛后冻结 checkpoint 再首次测试。16 篇 test 的 NLL 为：无写入 **5.15830**、原始更新 **5.15812**、当前论文 writer **4.44029**、错配论文 writer **4.44186**、训练集平均更新 **4.42475**。虽然 writer 比无写入改善 0.71801、16/16 篇有收益，其对错配只优 0.00157，且劣于平均更新 0.01554，因此**内容相关写入阶段门未通过**。本轮收益主要支持通用标题任务适配，不能证明当前论文信息被有效写入，更不能据此训练双门控。阶段门和模型参数未按 test 改动。

完整训练单步平均 0.237 秒、最大 reserved 11.631 GiB；test 含全部对照平均每篇 0.503 秒、最大 reserved 11.938 GiB，无 OOM。`audit_cbf_writer_results.py` 在本机从逐条轨迹独立复核：与旧源组无交叉、训练/dev/test 64/16/16、96 份候选缓存、每 epoch 完整 64 篇、320 步、dev 最优选择和 test 均值/阶段门均一致。详情见 [`关键试点报告`](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/task_writer_pilot/REPORT.md)，同目录保存公开来源元数据、曲线、选择、资源与聚合结果。原始全文、特征、checkpoint 和逐样本数据由 `.gitignore` 排除；远程 checkpoint 位于 `/home/ctj/cbf_ttt_writer_pilot_20260929/train_seed123/best.pt`。后续应在新源组上使用正确/错配记忆的训练约束及事实区分任务，再验证内容相关收益；正式反事实扩量与控制器训练继续暂停。

### 配对事实写入器关键试点（2026-09-30，训练与阶段门审计已完成）

用户要求继续下一步。执行前固定 [`PAIRED_FACT_WRITER_PILOT.md`](PAIRED_FACT_WRITER_PILOT.md)：用新论文的人工事实双生版本排除通用任务偏移，比较普通 NLL 与配对约束。48 个新论文源按 32/8/8 分组，各生成两条仅一个事实 token 不同的上下文，共 64/16/16 条，8 类标签严格均衡。论文正文作长文本干扰，颜色事实人为插入，结果不能等同自然事实问答。

| 文件 | 修改与需求对应 |
|---|---|
| `tasks/build_cbf_paired_facts.py` | 双生上下文、来源划分、均衡标签、单 token 差异校验及来源清单；固定 query 不含答案 |
| `cbf_ttt/writer.py` | 追加 `memory_choice_nll`，单次无答案 forward 返回单 token 候选的全词表 NLL；保持可微，原入口不变 |
| `tasks/cbf_paired_writer.py` | 复用候选缓存/低秩写入器；普通/配对训练、双生/平均对照、dev 选型、test 门槛、资源及哈希 |
| `scripts/run_cbf_paired_writer_pilot.sh` | 两卡提取、smoke、两种目标并行训练，配对支 dev 通过后才评分 test |
| `tests/test_cbf_paired_facts.py`、`tests/test_cbf_paired_writer.py` | 分组/均衡/单 token 差异、配对梯度符号、伪收益被拒绝、NLL 等价、骨干冻结和闭环 |
| `scripts/download_postcutoff_arxiv.py` | PDF 发生 IncompleteRead 中断，新增 HTTP 传输失败三次有限重试，耗尽后记录拒绝原因；保持筛选顺序 |
| `PAIRED_FACT_WRITER_PILOT.md`、本记录 | 固定预算、判据、失败停止规则及局限 |

新增 CLI 参数 `--objective paired/nll`、`--margin .1`；默认 rank=8、lr=.001、seed=134、epochs=5，每支 320 步。`--paired-dir/--nll-dir` 指向冻结选择。原始 baseline 配置和标题实验入口保持兼容，无新增依赖。数据生成 seed=132、来源筛选 seed=131，排除旧四批源 ID。`L_pair=L_correct+relu(.1+L_correct−L_twin)`，两条路径均回传写入器，候选 detach；两个版本均作为正样本。

dev 对 g=.5/1 按相对所有对照的最差平均 NLL 收益选 epoch/g。要求最差收益 >.005、至少半数源组对无写入及双生更新改善 >.005、八选一准确率 ≥.25 且高于每个对照 ≥.125。配对支失败则 test 保留未评分，不按已见结果调参。

运行（hku-gpu2，仓库 `/home/ctj/cbf_ttt_joint_exp_20260927`）：

```bash
ROOT=/home/ctj/cbf_ttt_paired_fact_20260930
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python
mkdir -p "$ROOT"
"$PYTHON" -m scripts.download_postcutoff_arxiv --tokenizer /home/ctj/models/Qwen3-4B --output "$ROOT/documents.jsonl" --api-feed /home/ctj/cbf_ttt_postcutoff_pilot_20260927/api_feed.xml --pdf-cache /home/ctj/cbf_ttt_postcutoff_pilot_20260927/pdf_cache --max-results 300 --target 48 --selection-seed 131 --exclude-metadata /home/ctj/cbf_ttt_postcutoff_pilot_20260927/documents.jsonl.meta.json --exclude-metadata /home/ctj/cbf_ttt_postcutoff_random_pilot_20260927/documents.jsonl.meta.json --exclude-metadata /home/ctj/cbf_ttt_title_recall_pilot_20260929/documents.jsonl.meta.json --exclude-metadata /home/ctj/cbf_ttt_writer_pilot_20260929/documents.jsonl.meta.json
"$PYTHON" -m unittest tests.test_cbf_paired_facts tests.test_cbf_paired_writer tests.test_cbf_writer
export ROOT PYTHON
bash scripts/run_cbf_paired_writer_pilot.sh > "$ROOT/pipeline.log" 2>&1
```

正式执行放在 tmux；输出目录必须是新目录，复现时更换 ROOT。每步耗时包括候选加载及梯度更新，进程另含模型加载/dev/保存，不将步内累计当总耗时。双路径反传增加显存；其他风险有候选差异过小、范数约束、人工事实/固定位置/有限答案类型。论文全文、特征、逐条结果及 checkpoint 保留远程/忽略目录，仅代码、协议、公开元数据及聚合结果上传 GitHub。四条件门控与正式扩量继续等待内容写入证据。

开始评分前补充 `sanity` 入口：dev 完整 KV、关闭快记忆写入的事实可读性准确率须 ≥.75，再启动提取与写入训练。该检查区分任务不可读与候选写入失败，不使用 test 或改变原有内容相关门槛。

远程 5 项单元测试通过（配对数据、梯度与骨干冻结、成功/失败门槛分支及旧 writer）；新增 `tests/test_cbf_postcutoff.py` 的部分 PDF 传输重试测试。新增 `scripts/audit_cbf_paired_writer.py`，独立复核来源排除、配对与均衡、两支初始化/训练顺序、每 epoch 完整 train 覆盖、dev 选择和实际 test 是否遵守门槛；输出聚合审计 JSON。

补充只读诊断 `scripts/summarize_cbf_paired_signal.py --root "$ROOT"`，在训练完成后度量 dev 双生候选的逐层相对 Frobenius 差异和非零差异比例，排查缓存是否完全相同；不运行模型、不使用 test 标签、不据此调参。输出 `candidate_signal.json`。事实可读性检查已达 16/16 正确；两支初始化 dev 数值一致。

#### 完成结果

两支各完成 5 epoch /320 步，远程独立审计通过。新 48 个论文源与旧 232 个源无交叉，96 个候选来源完整，两支训练顺序/初始化一致。完整 KV 下 dev 可读性准确率 100%；清空 KV 后，无写入 NLL 7.16931、准确率 12.5%，原始候选 NLL 7.13607、准确率 18.75%。普通 NLL 支选 epoch 1/g=1：NLL 2.25829、准确率 12.5%；配对支选 epoch 2/g=1：NLL 2.22286、准确率 12.5%。配对支对双生错配仅改善 0.00781 NLL，组内改善达标 2/8；两支均未达到准确率门槛，因此 **test 未评分**，没有进一步调整预算或阈值。

本轮仅支持当前受限低秩写入器仍未学会可靠的事实区分，不能据 NLL 下降声称内容记忆成功，也不能据此否定所有双门控架构。完整说明见 [`配对事实试点报告`](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/paired_fact_writer_pilot/REPORT.md)。远程审计输出为 `/home/ctj/cbf_ttt_paired_fact_20260930/audit.json`，原始训练轨迹和 checkpoint 保存在同目录。SSH 网络多次断连影响本地归档，但不改变已完成的训练和已返回的审计结果。

2026-10-03 已完成聚合结果本地同步：候选均值耗时 0.914 秒/条、最大 reserved 9.096 GiB；配对/普通 NLL 训练均值 0.509/0.263 秒每步、最大 reserved 14.115/11.430 GiB。后台流程于 2026-09-30 23:48:31（北京时间）完成。只读 dev 信号诊断确认双生候选不是相同缓存，相对 Frobenius 差异中位数 0.2378%（0.1682%–0.4649%）；该诊断没有涉及 test。公开元数据、选择、曲线、资源、审计及信号摘要均归档在 `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/paired_fact_writer_pilot/`，原始 token/全文/逐条结果/模型仍不上传。后续 TODO 是候选生成与写入表达能力的独立实验，而非继续扩量当前控制器。

### 写入—读取链路检查（2026-10-03，核心诊断完成，附加核验受阻）

用户要求下一步检查。代码核查发现当前读出在模型 dtype 中合并 W₀+M，并以该 dtype 计算词表 logits；BF16 是否限制微小事实差异需要实测，暂不认定是 bug。固定 [`WRITER_PATH_DIAGNOSTIC.md`](WRITER_PATH_DIAGNOSTIC.md)，只用已评分的 8 个 dev 源组，冻结已有模型、writer、候选及门值，不评分 test、不训练。

| 新增文件 | 功能及检查目标 |
|---|---|
| `scripts/diagnose_cbf_writer_path.py` | 核对原 BF16 dev 逐条结果；比较 BF16、仅 head FP32、全读取 FP32；追踪候选/低秩变换/范数限制/BF16 合并中的双生差异；输出查询 hidden 差异、NLL、准确率及错配收益 |
| `scripts/run_cbf_writer_path_diagnostic.sh` | 测试通过后两 GPU 并行精度诊断，记录代码提交及完成标记 |
| `tests/test_cbf_writer_path.py` | 检验投影范数、s=0 共享记忆、s=1 恒等、双生交换指标、原 forward 等价及 head 精度读取 |
| `WRITER_PATH_DIAGNOSTIC.md`、本记录 | 执行前固定检查顺序、网格、数据范围及解释边界 |

配对写入器额外检查 C±sR，s=0/1/8/32，其中 C/R 来自两个双生输入；这属于不可部署的诊断，不是独立测试集结果。s=0 共同投影保证完全相同记忆；s=1 保持原张量；其他点限制为各自原候选范数。FP32 只改变读取精度，候选仍是原 BF16 缓存。禁用 TF32，保留所有预设点，不依结果扩展网格或重新选模型。所有实现为独立诊断入口，不修改 baseline 数值路径；无新依赖。

运行命令（hku-gpu2 仓库内，以 tmux 后台执行）：

```bash
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python bash scripts/run_cbf_writer_path_diagnostic.sh
```

ROOT 默认 `/home/ctj/cbf_ttt_paired_fact_20260930`，OUT 默认 `$ROOT/path_diagnostic_20261003`，输出必须新建。命令行允许显式 `--dtype bfloat16/float32`、`--root`、`--model`、`--output`、`--device`。每组记录时间和峰值显存；逐组记录只留远程/忽略目录，公开聚合与报告。若 FP32 OOM 或原结果不能复现则明确停止记录，不擅自改模型或预算。

#### 返回结果与解释边界（2026-10-05 归档）

两项测试通过，两精度分片均完成 8 个 dev 组；原 BF16 16 条结果逐值复现，最大误差 0。配对 writer 的 BF16/head-FP32/full-FP32 准确率均为 12.5%，正确对错配 NLL 收益分别 +0.007813/+0.001665/−0.000018；普通 writer 也均为 12.5%。s=0/1/8/32 所有预设诊断点均未提高准确率，但查询 hidden 随双生变化而变化，排除完全没有读取响应的解释。原候选相对差异中位数 0.2376%，配对 writer 范数投影前后均约 0.1662%，投影并未明显改变该相对差异。

完整 FP32 只作为配置敏感性结果：额外读取 checkpoint 存储 dtype 的自动审批及唯一重试均超时，操作未执行；尚未核验两种加载参数的逐值等价，不能把所有变化严格归因于算术精度。上述结果来自此前已批准执行的远程日志读取（含完整 summary 输出），没有绕过审批去重新执行被阻断的请求。原始完整逐组文件尚未同步独立审计。head FP32 使用同一 backbone/hidden，原路径复现与准确率结论不受该限制。

BF16+head FP32/全 FP32 峰值 reserved 13.477/19.611 GiB，无 OOM。数值摘录与详细解释新增于 [`writer_path_diagnostic/REPORT.md`](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/writer_path_diagnostic/REPORT.md) 及同目录 `retrieved_summary.json`；逐组数据仍留远程。下一步候选颜色可解码性探针尚未执行；test 继续保持未评分，控制器不扩量。

### 选择性遗忘研究转向（2026-10-08，执行前登记）

按用户的新要求，停止沿写入器/颜色探针方向扩展，依据图片建议制定 [`SELECTIVE_FORGETTING_PLAN.md`](SELECTIVE_FORGETTING_PLAN.md)，先验证全局 + last-2 + last-1 遗忘的收益空间。保留 g=1，已有证据仅支持遗忘优于累积，不支持旧控制器优于全部清除。阶段 A 的双 split 小规模 oracle 通过后，才进入多次遗忘决策和控制器训练；联合端到端训练属于更后阶段。

| 新增文件 | 实现与需求对应 |
|---|---|
| `cbf_ttt/selective.py` | `CohortSession` 保存最近两次实际保留的直接更新贡献，分别控制旧历史/R2/R1；系数为保留率。均匀系数复用原 `commit_both`，g 固定 1；查询独立克隆，不污染轨迹 |
| `tasks/build_cbf_selective.py` | 构建独立自然来源的四类场景；从 LongCrawl64 最后 row group 确定性选择 16 份足够长的文档，8 组、32 场景；扫描实际 1B 混合训练文本的长锚点，匹配即失败；保留元信息和哈希，不重复填充文本 |
| `tasks/cbf_selective.py` | 第四 chunk 单点稀疏干预，0/.5/1 的 27 动作；后续 3 个评分点；另有从头运行的全局清除/半保留/最近 2 和 3 更新窗口；每标签时间与峰值显存；完整性审计、组级置信区间与预设继续门槛 |
| `tests/test_cbf_selective.py` | FP32/BF16 均匀系数与旧更新逐值相等、贡献旋转、clone 独立、模型读取闭环、来源隔离与因果答案、门槛通过/失败及不完整标签拒绝 |
| `scripts/run_cbf_selective_pilot.sh` | 先测试、构建/查重、smoke，再双卡按源组分片运行，审计聚合；输出目录必须全新，任一阶段失败停止 |
| `SELECTIVE_FORGETTING_PLAN.md`、本记录 | 明确假设、对照、预注册门槛、运行命令、后续阶段及解释限制 |

复用原模型加载与候选计算，原 runtime/baseline 配置不修改，无额外依赖。新增构建参数 `--parquet/--tokenizer/--training-data/--output`；采集参数 `--data/--model/--output/--shard/--shards/--smoke`；聚合参数 `--data/--inputs/--output`。脚本支持 ROOT/PYTHON/MODEL/TOKENIZER/PARQUET/TRAINING_DATA 环境变量覆盖路径。默认 4B 最终 1B checkpoint、chunk=4096、完整 KV、BF16；源与模型 chunk 不一致时报错。

```bash
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
ROOT=/home/ctj/cbf_ttt_selective_forgetting_20261008 \
bash scripts/run_cbf_selective_pilot.sh
```

本地语法编译、shell 语法与 diff 空白检查通过；本地没有 torch，张量/模型测试随远程脚本先运行。风险：粗时间来源不是语义模块；清除直接贡献不能撤销 KV/后续 delta 的间接影响；FP32 来源缓存增加显存；27 动作 oracle 有选择容量优势；4 组/split 的 bootstrap 很不稳定；长文本锚点检查非完整语义去重；尚未检验 FineWeb-Edu 和正式下游任务；现有短序列预训练对多更新轨迹覆盖不足。原始文本/token/标签保留服务器，仅代码、文档、来源哈希和聚合结果上传 GitHub。

远程首次运行 14 项测试通过，构建阶段在查重前发现原训练 JSONL 使用 `content_split` 而非 `text`，按预期停止、未评分模型。读取器补充原 VeOmni 字段兼容及回归测试，协议和场景来源不变；保留首次日志，新目录 `/home/ctj/cbf_ttt_selective_forgetting_20261008_r2` 重新执行。

第二次运行 15 项测试通过，但原 16 份尾部候选的长锚点命中了训练文本，故在 GPU 评分前停止。评分前修订构建器为固定 64 文档候选池，完整扫描后剔除所有命中来源，按顺序取前 16 份；保存候选来源哈希/行号/命中行数与最终来源。此修订不使用模型结果，也不放宽重合标准；最终 8 组、32 场景和统计门槛不变。新输出目录 `/home/ctj/cbf_ttt_selective_forgetting_20261008_r3`，前两次日志保留。

r3 完整扫描 162,762 条训练记录；64 候选文档中 12 份来源被剔除，其余 52 份中按顺序选 16 份。独立检查确认选中来源的训练锚点命中和跨组整 chunk 重复均为 0。原构建日志末行误将候选池 140 条命中标为最终来源命中；元数据始终分别正确保存候选池/最终来源数，现修正日志字段命名。聚合审计补充逐来源筛选复核、跨组整 chunk 查重、checkpoint 一致性和分场景动作分布；不改变标签数值或通过判据。

GPU smoke 通过，首轮分支约 3.4–3.6 秒，峰值 allocated/reserved 22.2/26.2 GiB。正式双卡采集已启动，共 864 条反事实标签及 128 条固定策略轨迹；原始目录同上，报告归档目录为 `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/selective_forgetting_pilot/`。完整结果和阶段判据等待最终审计，当前不宣称局部遗忘有效。

在仅查看采集进度/资源日志、未读取任何实际 NLL 时，补充只读诊断：由 pilot 选择固定单点局部动作及全部固定策略中的最优者，锁定后报告 confirm NLL；用于判断自适应控制器是否还有额外空间，不改变主判据。测试加入 confirm 不得反向影响选策的用例。注意：864 个动作评分对应 32 个决策状态，独立统计单位仍只有 8 个来源组，不能把它们当成 864 个独立控制器训练样本。

#### 阶段 A 完成与停止决定

r3 于 2026-10-08 09:35:10–10:13:20（北京时间）完成，流水线 38 分 10 秒。864 次反事实评分及 128 次固定策略评分齐全，最终 16 项测试通过；源码采集提交 b64d4df、审计提交 6451b73，新旧汇总共有字段逐值相同。SSH 审计连接曾断开，核实文件后仅补做未执行的审计，未重跑模型或更改标签。

全部 32 场景、四类情况的局部最优动作均为 `(0,0,0)`，pilot/confirm 局部相对同状态全局 oracle 的 NLL 收益均为 0。单点局部/全局 oracle NLL 分别为 11.412138 / 12.164369，而从头每步清除为 1.882757 / 2.093384；持续累积为 22.338424 / 22.139893。pilot 锁定的最佳单点/全程固定策略在 confirm 上仍分别为单点清除/每步清除。两 split 均未通过门槛，按计划停止扩量，不训练新控制器，也不启动多决策或联合训练。

平均动作评分 3.5309 秒（P95 3.5980），最大 allocated/reserved 22.1824/30.6406 GiB，无 OOM；完整决策状态平均 120.1315 秒。沿用本流程的 3000 状态预计两张 5090 约 50 小时，但当前无扩量依据。注意完整批次 reserved 峰值高于 smoke 的 26.2 GiB。

完整报告与聚合/执行审计新增于 [`selective_forgetting_pilot/REPORT.md`](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/selective_forgetting_pilot/REPORT.md)、同目录 `summary.json` 和 `execution_audit.json`；原始文件仍保留远程。README 增加可选实验入口链接。当前结论限定于此 checkpoint/运行路径/自然代理任务，不能把单点和全程干预的全部差距归因于粒度，也不能直接否定所有局部遗忘设计。下一 TODO 是逐 chunk 快记忆稳定性、原生 TTT 与 CBF 的 BF16 更新路径核验；现有短序列预训练覆盖不足只是待验证解释，未开展长序列再训练。

### 多 chunk 记忆稳定性诊断（2026-10-08，执行前登记）

用户要求继续，按上一轮 TODO 固定 [`MEMORY_STABILITY_PLAN.md`](MEMORY_STABILITY_PLAN.md)。只用已有 4 个 pilot stable 场景、每场景 7 chunk，冻结 4B checkpoint；不评分 confirm、不训练。代码检查发现除了完整权重/独立 M 的累积舍入，原生三操作数 contraction 与 CBF 两次 matmul 的候选计算顺序也不同，需分开核验。

新增 `scripts/diagnose_cbf_memory_stability.py`：原生、CBF、每步清除、不写入、原生运算+BF16 M/W、CBF+FP32 M、原生运算+FP32 M/W 共九条固定路径；逐步记录 NLL、逐层记忆/候选/有效权重变化与对齐；native 与桥接路径比对完整权重/hidden 字节哈希；同输入候选计算比较 FP32 参考；已有 CBF/clear 结果须复现。FP32 仅指累积状态，骨干/候选激活/读出均保持 BF16，原 runtime 和模型文件不改，诊断进程临时 hook 退出自动恢复。

新增 `tests/test_cbf_memory_stability.py`：原生桥接 FP32/BF16 hidden/权重/NLL 一致性、BF16 完整权重与独立 M 舍入差异及 FP32 状态对照、查询克隆与零写入、同输入候选运算参考。新增 `scripts/run_cbf_memory_stability.sh`：先测试，双 GPU 按来源组运行，聚合时校验旧结果并记录完成标记，拒绝覆盖目录。复用已有依赖。

运行：`ROOT=/home/ctj/cbf_ttt_memory_stability_20261008 bash scripts/run_cbf_memory_stability.sh`。脚本可覆盖 ROOT/SOURCE/PYTHON/MODEL；采集 CLI 支持 `--data/--model/--output/--shard`，聚合支持 `--inputs/--reference/--output`。逐步轨迹保留远程，聚合与说明才上传。此为已看过数据上的诊断，不支持独立泛化或训练覆盖不足的因果结论；如桥接不逐值相等，必须报告而不能单独归因于累积形式。

#### 诊断完成与结果归档

采集前修正 FP32 完整 W 的统计定义：存储偏移用 FP32 W−W0，实际读取偏移另用 BF16 有效 W−W0。正式采集提交 `59fbba64badbc2964e3e46e7c8fd9d7f0404088a`，两张 5090 于 2026-10-08 11:04:25–11:06:38 完成，流水线 133 秒；20 项测试通过，4 来源组×9 路径全部 36 轨迹完成。原生/桥接全部 28 个边界 hidden/有效权重哈希及 NLL 一致；已有 CBF/clear 的 24 个评分值重现最大误差 0；冻结骨干检查通过。

最后三点平均 NLL：原生 22.084451，CBF 22.188627，CBF FP32 M 22.197753，原生运算 FP32 M/W 22.192921/22.192738；每步清除 1.849533，不写入 1.802835。原生同样退化且 FP32 累积未改善，不能仅归因于 CBF 增量表示或 BF16 累加；全模型 FP32 等其他数值因素未在本轮排除。这是 pilot stable 子集，和上一轮全场景均值不可直接混比。

原生有效偏移范数比例跨组跨层均值从 0.4304% 增至 3.6742%，第七点最大单组单层为 23.4758%；CBF 候选范数比例从 0.4310% 增至 0.8874%。同输入原生/CBF 候选相对差异均值 0.2169%，运算差异确实存在但无法解释共同退化。最大 allocated/reserved 17.316/22.779 GiB，无 OOM；逐轨迹耗时合计 243.027 秒，原生/桥接额外哈希开销使路径耗时不能直接比较吞吐。

新增归档 `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/memory_stability/{REPORT.md,summary.json,execution_audit.json,.gitignore}`，包含全九路径汇总、逐 chunk 曲线、耗时显存、来源与源码哈希、运行命令和局限；原始逐层/逐步记录与日志仍在远程同名实验目录。计划文档补完成状态，README 补诊断入口。baseline 模型/runtime/训练配置不修改，无额外依赖。

后续 TODO：先审计短序列训练/推理路径，再做长序列训练资源试跑和匹配 token 预算的短/长序列小实验。6144-token 训练、4096 chunk 的现有实现只覆盖一次用于后续 token 的更新，可能不足以支持七次推理更新，但尚未建立因果证据。此前 8192-token 全参数训练发生 OOM，因此须先验证可训练参数冻结/检查点等资源方案，并在短长对照保持一致。只有恢复有益且稳定的快记忆后，才重启局部遗忘 oracle、多决策标签和控制器。上述训练未执行，本轮不扩量，也不宣称研究构思普遍无效。

### 训练/推理前向核验（2026-10-08，执行前登记）

用户要求继续，新增 `TRAIN_INFER_PARITY_PLAN.md`，先 P 前向核验，再根据结果进入 R 资源检查与 L 短长对照。代码审计确认训练端分块并行+cumsum、推理端顺序完整 W 更新；需检查差异对真实模型的影响，不能直接归因于训练长度。

新增 `scripts/diagnose_ttt_train_infer.py`：显式加载训练/原生推理类，逐张量检查 checkpoint 一致；四个 pilot stable 来源的 6144/12288 token，三路径（训练整段/推理整段/推理流式）×原 TTT lr/零 TTT lr，16 比较共 48 前向；报告相同末尾 128 token NLL、hidden 差异、耗时和显存。原模型/runtime 不改，无新依赖。新增 `tests/test_ttt_train_infer.py` 检查 FP32 非零更新下完整/不完整块一致性、writer 因果梯度与零更新系数恢复。新增 `scripts/run_ttt_train_infer.sh` 先测试、双卡分来源采集、完整性审计汇总，拒绝覆盖输出目录。

脚本环境参数 ROOT/SOURCE/PYTHON/MODEL；collect 参数 `--data/--model/--output/--shard`；summarize 参数 `--inputs/--output`。运行 `ROOT=/home/ctj/cbf_ttt_train_infer_20261008 bash scripts/run_ttt_train_infer.sh`。6144/12288 的成对 NLL 差大于 0.1/0.5 触发优先定位，非显著性/等价性判据；仅 pilot 机制诊断。长序列资源检查和训练尚未启动，原始 token/逐来源结果留远程。

P 阶段通过：7 项测试、48 条前向完成，训练/原生整段 hidden 和 NLL 全部逐值相同；训练/流式最大 NLL 差 0.005233，未触发诊断阈值。进入 R，新增 `scripts/probe_ttt_writer_training.py`，只训练现有 conv/proj、FP32 writer/AdamW 配合 BF16 autocast 和非重入梯度检查点。现有训练语料开头按 6144/12288 截取，每长度从相同 checkpoint 独立跑 3 步，测量反向/优化器/梯度/参数修改/显存与耗时，不保存模型。CLI：`--model/--training-data/--tokenizer/--length/--output`。设计和具体命令追加于 `TRAIN_INFER_PARITY_PLAN.md`。此脚本为资源诊断，不能把跨 packed 文档输入或三步 loss 变化当作算法收益；模型与原训练配置不修改。

新增 `scripts/run_ttt_writer_resource.sh` 校验 P 汇总通过后，按长度在两张卡各启动一个资源试跑，记录源码/起止时间并传播失败。环境变量 ROOT/PARITY/PYTHON/MODEL/TRAINING_DATA/TOKENIZER 可覆盖；输出目录必须全新。

#### P/R 完成结果与风险处理

P 于 2026-10-08 11:25:45–11:26:22 完成（37 秒，代码 `ef29877`），四来源组全部 16 次比较/48 前向齐全，7 项测试通过。训练/原生整段 hidden 和 NLL 逐值一致；训练/流式最大 NLL 差 0.005233，未触发门槛，零更新对照有类似 hidden 数值差异。训练/原生模型全部 state_dict 张量一致，权重版本不变。峰值 allocated/reserved 17.891/18.455 GiB。

R 于 11:28:40–11:29:13 完成（33 秒，代码 `91359fe`）。冻结骨干，仅训练 45,964,800 个现有 writer 参数，两长度各 3 步；所有梯度有限非零，14 个张量全部实际更新，冻结骨干未变，AdamW 状态 FP32。6144/12288 峰值 allocated 为 11.169/13.830 GiB，reserved 为 11.945/15.168 GiB；第 2–3 步平均 2.847/9.093 秒，无 OOM。

资源试跑也暴露稳定性风险：长序列 loss 1.9705→1.9188→4.4116；裁剪前梯度范数 5979.30/1442.89/10362.40，每步均执行 clip=1。最大梯度均为第 35 层 conv，平方范数占比 99.50%/97.32%/68.34%。第三步 loss 是第二次 optimizer 更新后的前向，不能误归因于第三次更新。这是单一重复训练输入上的诊断，不能推广为普遍训练失败；也不能据此声称学习率已选择合适。没有保存/替换训练模型。

新增聚合归档 `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/train_infer_parity/{REPORT.md,summary.json,execution_audit.json,.gitignore}`，涵盖 P/R 全部聚合、哈希、耗时显存、测试、解释和复现方式；原始逐来源/逐参数日志留远程。计划和 README 补完成入口。原 baseline 配置、模型和 runtime 不改，无新依赖。

根据新观察到的风险，L 正式短长对照暂不启动：下一步先核验融合 loss 与标准 CE 的梯度，再记录逐层实际相对参数位移，明确第 35 层梯度集中与优化尺度的作用。之后才做同 token/同 optimizer 步数的短长序列适配，并以独立来源比较累积/清除/不写入。当前完成前向核验和资源闭环，尚未完成长序列有效性训练，更未扩大控制器。

### Writer 梯度与 optimizer 位移诊断（2026-10-08，执行前登记）

用户要求继续，新增 `OPTIMIZER_AUDIT_PLAN.md`，固定原三步训练轨迹，核验 Liger 融合 CE/标准 CE 梯度，记录真实相对参数更新，并在第二次更新处干预第 35 层 conv 的 delta。保持原学习率，不训练控制器、不评分保留集。

新增 `scripts/diagnose_ttt_optimizer.py`：标准 CE 按 128 token 分块，同一 hidden 计算上游梯度，两次 VJP 比较各 writer 梯度；恢复融合梯度进行原 AdamW step；记录每层相对位移；完整/撤销末层 conv/仅末层 conv/末层 conv 减半四分支评分后恢复参数。输入哈希、冻结骨干和原 loss 轨迹均核验。新增 `tests/test_ttt_optimizer.py` 检验标准 CE 分块的标签对齐与归一化、梯度一致性和干预隔离；新增 `scripts/run_ttt_optimizer_audit.sh` 先测试再双卡执行。配置和运行命令见计划，复用现有依赖，baseline 模型/runtime/配置不修改。

新增只读聚合 `scripts/summarize_ttt_optimizer.py`：要求两个长度及全部三步/四干预齐全，检查冻结骨干、原轨迹复现、完整干预评分与第三步前向相同；输出逐层更新比例、梯度差与资源统计，不含原始 token/梯度。运行脚本补聚合步骤。此聚合在采集启动后补充，不改变模型路径、干预或判据；首轮采集提交与聚合提交分别记录。

首轮诊断 5 项测试通过但原轨迹复现门槛失败：6144/12288 最大误差 0.0001607/0.0158076，故停止解释。SSH 代理多次短暂超时，重连后核实任务已结束，没有重复启动同一诊断。随后原样重跑资源脚本到 `cbf_ttt_writer_resource_20261008_repeat`，也产生 0.0002980/0.0187774 的跨运行偏差；当前训练设置的固定 seed 不保证逐值重跑，具体非确定性来源未定位。

据此将诊断修订为 v2：先运行普通三步更新，保存每步 CPU before/after 参数快照；全部梯度和干预分析随后在固定快照上执行，无 optimizer 更新。恢复逐值检查，同状态 loss 重放误差仍要求≤1e-5；历史偏差只报告，不把首轮失败改判为通过。梯度/CE 门槛和超参数不变，失败时写完整 JSON。聚合检查同步改为快照/同状态审计；测试去除 detach 警告。新目录 `cbf_ttt_optimizer_audit_20261008_r2` 保留首轮证据。协议修订详见 `OPTIMIZER_AUDIT_PLAN.md`。

#### v2 固定快照诊断完成

v2（源码 `d4659c9`）于 2026-10-08 15:06:57–15:08:31 完成，94 秒；5 项测试通过。普通三步优化先完成，随后两长度×三个快照各做两种 CE 梯度对照和第二步四分支干预。同状态前向/完整更新分支的重放误差均为 0，参数快照逐值恢复，冻结骨干未变。历史跨运行偏差仍记录为短 0.0004668 / 长 0.0132470，没有宣称跨运行精确重现。

融合/标准 CE loss 最大差短 6.929e-7、长 4.284e-7；writer 全局梯度相对 L2 最大短 0.1068%、长 0.7577%，单 writer 最大 1.4036%/6.7710%，全部低于预设门槛。没有发现需更换融合 loss 的明显错误，也不能宣称梯度逐值等价。

长序列首步第 35 层 conv 相对位移 63.64%，其他多个 conv 为约 10%–17%，实际全局 clip=1 并未直接控制 AdamW 的相对参数步长。第二次更新前/完整/撤销末层 conv/仅末层 conv/末层 conv 减半 NLL 为 1.918995/4.398330/3.556957/1.922081/4.009240；仅末层不足以造成跳升，其他 writer 及联合响应需要关注，不能据此建议删去末层。此为单训练输入的局部因果干预，不是泛化效果证据。

新增公开归档 `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/optimizer_audit/{REPORT.md,summary.json,execution_audit.json,.gitignore}`，记录失败/修订/独立复跑/最终结果、全部层位移、资源和复现命令；计划和 README 同步完成入口。峰值 allocated/reserved 短 10.965/11.768 GiB、长 13.773/14.990 GiB，无 OOM，未保存模型。SSH 代理超时后重连读取，未绕过认证。

后续 TODO：在多个训练前缀上比较固定的外层优化尺度设置，验证较小学习率或实际相对位移约束，再启动等 token/等 optimizer 步数短长适配。这里的 Δθ 是外层 conv/proj 参数更新，不是会话 ΔW；本轮 lr=1e-5 试跑与原 1B lr=1e-6 全参数训练不同，不能将结果当作原 checkpoint 长程退化根因。baseline 模型/runtime/配置未改，无新依赖，控制器不扩量。

### 回到图片第二条：随机 1–3 决策点实验（2026-10-08，执行前登记）

按用户要求，直接开展多点遗忘，新增 `MULTIDECISION_FORGETTING_PLAN.md`；此次明确覆盖旧计划 A 失败则暂停 B 的执行安排，但不改变旧 A 结果，不以 optimizer 诊断作为前提。采用已有来源组 0/1/4/5×四场景共 16 场景，第一点 chunk 4 固定、后续点在 5/6/7 预先随机抽取，构成 1/2/3 点嵌套日程。

新增 `tasks/cbf_multidecision.py`：build 写入确定性时间点和候选清单并复核来源；collect 复用 CohortSession，完整穷举全局 3/9/27 序列，局部每个多点日程固定采样 27 序列，另有等候选数对照。跨日程动作编码去重，导入 432 个旧单点评分；新多点轨迹 1144、新清除/不写入对照 32、单点重现桥接 48，共 1224 GPU 轨迹。后续候选各分支重新计算，查询不污染轨迹。半保留/窗口固定对照复用旧标签，逐场景重现三个全局单点动作及每步清除；来源/checkpoint 一致、误差≤1e-5 才继续。

新增 `tests/test_cbf_multidecision.py`：候选预算/日程嵌套、编码与随机复现、完整重放与共享前缀多点分支一致、查询和父分支不被改动、缺失标签拒绝；新增 `scripts/run_cbf_multidecision.sh`：先测试/build/smoke，预计双卡90分钟内且峰值 allocated<30 GiB 才执行全部16场景。CLI/配置/运行命令见计划。原模型/runtime/训练配置不改、无额外依赖；历史 pilot/confirm 均已看过，不宣称新的独立验证。局部结果是采样集合最优值，不能冒称完整局部 oracle；图片第三条联合训练尚未执行。

多决策实验于 2026-10-08 15:36:48 +08:00 启动，代码 `2a57ac1`，tmux `cbf-multidecision-20261008`，远程目录 `/home/ctj/cbf_ttt_multidecision_20261008`。19 项测试通过，固定候选构建完成，smoke 复现与资源门槛通过；每轨迹平均 3.880 秒（含桥接和全程对照）、最大 allocated 22.182 GiB，估计双卡完整批次约49.5分钟（含25%余量）。双卡采集已启动，尚无最终结果；不将启动视为实验完成。

新增启动归档 `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/multidecision_forgetting/{RUN_STATUS.md,design.json,estimate.json,.gitignore}`，仅含固定动作设计、来源哈希、资源预算和状态，不包含原始文本/token/逐场景标签。最终16场景齐全后脚本自动审计聚合，后续需归档完整报告；图片第三条端到端联合训练仍未执行。

### 多决策遗忘实验完成（2026-10-08）

远程完成标记 16:15:42 +08:00，实际墙钟 38 分 54 秒；16/16 场景、4 来源组齐全，两分片骨干未变，64 项参考检查误差均为 0。新计分轨迹 1176 条，另有 48 条复现桥接；平均新计分轨迹 3.594 秒，最大 allocated/reserved 22.183/26.189 GiB。

单/双/三点全局最优 NLL 为 13.703427/5.997364/3.378568；加入局部候选仍为 3.378568。每步清空/不写入为 2.084365/2.025309。预设时间维度改善门槛通过，选择性遗忘超越强对照门槛未通过。此为已观察数据上的 oracle 诊断，不能宣称控制器泛化有效或否定完整局部空间；图片第三条联合训练仍未执行。

归档目录 `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/multidecision_forgetting` 新增最终 `REPORT.md`、聚合 `summary.json`，更新 `RUN_STATUS.md`。未修改 baseline 或新增依赖，未启动后续实验。

### 快速记忆价值验证计划（2026-10-08，尚未执行）

新增 `MEMORY_VALUE_VALIDATION_PLAN.md`。核对多决策、标题 writer 和配对事实历史报告后，明确不能只凭 NLL 下降声称内容记忆有效。固定 V 内容价值、F 遗忘取舍、S 局部动作空间三个问题；24 个新来源组（dev 8 / confirm 16），FineWeb-Edu 和 LongCrawl64 各半，四场景乘双生版本共 192 会话。设计四全程策略、两 KV 条件、self/zero/twin 读取干预；按来源组确认，避免将答案类型偏移当作事实记忆。

计划包含来源排除和 1B 重叠审计、标签与真值、可读性门槛、dev 选型和 confirm 封存、准确率天花板处理、资源上限、逐轨迹耗时显存、待实现文件和 CLI、失败分支及联合训练梯度接口风险。V/F 最多 768 条完整六块轨迹和 7680 次单 token 查询；S 另计，ETA 需要新 smoke 实测。

README 新增计划入口，多决策计划追加最终状态与后续链接。未更改 baseline、模型或依赖，未新增可运行 collector，未启动新 GPU 实验；本次交付是实验计划，不能与已完成的多决策结果混淆。

### 记忆价值验证实现与执行准备（2026-10-09）

新增 `tasks/build_cbf_memory_value.py`、`tasks/cbf_memory_value.py`、`tests/test_cbf_memory_value.py`、`scripts/run_cbf_memory_value.sh`。构造器固定24组、四场景、双生版本和六块4096-token输入；两个来源内标签各自平衡；开发8组、确认16组。双生只差一个事实token，真值从最后一条同键记录审计。来源候选从原始parquet行组末尾倒序读取，排除旧manifest来源及原始上下文/1B训练文本长锚点匹配。只按结构筛选，不使用模型分数。

下载FineWeb-Edu固定revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` 的 `sample/10BT/013_00000.parquet`；复用原LongCrawl64分片。FineWeb末尾行组只有101行，因此按预先固定的行组倒序继续选择候选，不限制在最后一个行组。新增可选 `requirements-memory-value.txt` 的 `pyahocorasick==2.1.0`，用于CPU单遍多锚点扫描；baseline依赖不变。

评测复用CBFSession，四策略从起点推进，两KV条件下独立克隆self/zero/twin记忆，双生完整缓存转CPU以限制显存，查询时只更换M。单次logits计算NLL/八选一/greedy；父缓存完整字节哈希、模型参数版本及权重哈希核验。每条轨迹和评分都记录耗时/allocated/reserved；确认阶段强制读取通过dev的冻结选择。

入口先跑完整dev可读性Q，再做两个dev组的smoke和重复检查，预算通过后跑完整dev。没有KV条件通过V则confirm保持未评分；通过才开启对应条件。S为另外计量的条件式后续，当前脚本不冒称已执行S。Q失败是任务验证未通过，不能当作算法失效。为简化完整性审计，smoke不复用进正式dev，额外计算单列，资源预测按剩余12组/卡上限估计。

本地4项纯协议测试通过；远程新旧共21项测试通过，包含真实tiny Qwen缓存隔离与zero/none一致。此时尚无本轮4B效果结论。待采集完成后另行记录阶段结果，模型/runtime及原训练配置未修改。

#### 正式启动与新来源审计

采集提交 `077d320` 已上传 GitHub 并同步 hku-gpu2；2026-10-09 10:55:33 +08:00 在 `cbf-memory-value-20261009` tmux 启动，ROOT=`/home/ctj/cbf_ttt_memory_value_20261009`。21 项测试通过。场景构建完成192条；128个来源候选经全部162,762条训练记录与21个旧上下文文件扫描，FineWeb/LongCrawl合格94/30，选中72/12篇，84个ID独立、指定锚点匹配为0。

10:59:55 Q阶段16/64会话完成，尚无完整可读性/价值结论。新增运行归档 `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/memory_value_validation/RUN_STATUS.md` 和忽略规则；原始数据和逐条记录仅保留远程。新增只读 `scripts/audit_cbf_memory_value.py`，用于最终核对来源唯一性、阶段完整性、Q门槛、confirm是否被正确封存、权重一致及耗时显存。审计脚本不更改正在运行的采集提交。

#### Q 完成并按预设门槛停止

2026-10-09 11:06:17 +08:00 正常结束，完整流水线644秒。Q共64条dev轨迹/128次查询，两卡各32条。stable/correction/recent1/recent2目标八选一准确率为31.25%/100%/56.25%/62.5%；锚点为87.5%/81.25%/87.5%/93.75%。三类目标不足预设75%，状态`stopped_by_Q`；不运行writer smoke、V/F、confirm或S，128条确认会话保持未评分。该结果检验的是全KV无写入任务可读性，不能宣称快速记忆或遗忘机制无效。

独立审计通过：84个来源ID和两类哈希唯一、指定重叠为0、64行与128查询齐全、Q重算相同、模型权重字节哈希不变、confirm未评分。完整六块rollout平均4.118秒，最大allocated/reserved为14.770/18.828 GiB，两分片墙钟513.418/501.100秒；大部分额外开销包含CPU缓存搬运与状态哈希。无OOM。审计补按来源的Q分层和资源有限值检查，未改采集路径或门槛。

新增完成归档 `memory_value_validation/{REPORT.md,q_summary.json,execution_audit.json,design.json}`，更新RUN_STATUS、README和计划最终状态；原始数据/逐条结果留远程。下一步需单独校准短长时距、键绑定或回答格式，再使用新来源重做价值验证；此次未擅自改场景重跑，V/F实现尚未获得本轮4B实测结果。

### 下一步：开发集可读性校准（2026-10-09，执行前）

新增 `READABILITY_CALIBRATION_PLAN.md`，固定3种时距（短2块、长6块早事实、长6块近事实）×单/双键×QA/cloze格式，另加双键记录槽位交换及原Q路径桥接。仅使用已经观察的8个dev源组、16条stable双生；不评分旧confirm，不训练或写入M。背景从原始parquet按行号重取并验证SHA，不重复filler。新的双键场景只保留目标和锚点两条人工记录，其他块恢复自然背景，因此不能把它的成绩冒充原始四场景均通过。

新增 `tasks/cbf_readability.py`、`tests/test_cbf_readability.py`、`scripts/run_cbf_readability.sh`。128条上下文轨迹/384个查询，查询格式共享同一前缀并独立克隆，父缓存完整哈希和模型权重不变；原stable桥接32个NLL要求误差≤1e-5且预测一致。格式选择只允许long_far/dual默认顺序，目标/锚点在整体和两域均≥75%，否则不推荐完整长程双键协议。新CPU构造测试3项通过，无新依赖，旧模型/runtime不改。

可读性校准采集提交 `c07d98c` 已同步服务器并在 `cbf-readability-20261009` tmux 启动，ROOT=`/home/ctj/cbf_ttt_readability_20261009`。正式入口8项测试通过（2.408秒）；128条上下文/384查询构建通过，来源只含原dev0–7，旧confirm未评分。14:46:50两分片各完成5/64条。新增启动归档 `readability_calibration/RUN_STATUS.md`，完整结果待审计。

采集中独立核验原路径桥接：16条原stable会话、32个查询NLL最大误差0，八选一与greedy预测逐条一致。该检查只读原Q与已生成桥接结果，不改变采集代码或根据新条件成绩选型。固定设计与上下文哈希已归档，最终校准表仍待全部128条齐全。


### 可读性校准完成与审计（2026-10-09）

采集提交 `c07d98c`，14:45:09至14:58:49 +08:00，共820秒；128条轨迹/384查询齐全，8项测试通过，原Q的32桥接评分误差0。新增只读 `scripts/audit_cbf_readability.py`：验证原dev stable来源和桥接输入、文件/上下文哈希、分片归属，重算summary完全一致，检查模型/父缓存审计及confirm未评分，汇总逐条件耗时。最大allocated/reserved为14.770/18.822 GiB；两分片795.830/774.890秒。无OOM，未训练、M为空。

重建自然背景、仅保留目标/锚点记录后QA目标准确率各布局100%，原stable桥接31.25%；移除额外人工记录同时恢复背景，不能独立归因于格式或干扰数量。远事实cloze目标/锚点100%，按预登记窄门槛选中cloze；短程/长程近事实双键cloze目标0%，不能据此推广到四场景或进入V/F。完整分域、分组、NLL、首token和资源表已归档。候选八选一与自由生成不能混称。

新增归档 `readability_calibration/{REPORT.md,summary.json,execution_audit.json}`，更新RUN_STATUS、README、计划完成记录；raw文本/token/逐条评分保留远程。本轮baseline和训练配置不变、无新依赖。附件第一/二条遗忘建议没有新增效应证据，第三条联合训练未启动。

后续TODO：固定QA分离人工记录负载与背景变化；所有四场景通过统一Q后再做新来源V/F。新来源构造前需要修复旧来源排除器目前只识别 `*.meta.json` 和 `ids/context_ids/prefix` 的局限，补充 `design.json/chunks` 并验证来源ID唯一，避免遗漏本轮/上轮新格式来源。此问题不影响本次有意复用dev的校准。没有因单个容易条件通过而自动扩大实验或训练控制器。

### 固定记录槽位干扰实验（2026-10-09，执行前）

按用户继续下一步要求，新增 `RECORD_INTERFERENCE_PLAN.md`，固定QA和原六块输入，仅改动第1/3/4/5/6块的10条无关记录正文槽；背景前缀、header、行尾token、目标/锚点块和所有位置不变。主控制用原背景同绝对位置token替换槽正文，辅助控制填换行；占位文本仍是干预的一部分，不宣称完全无混杂。0/2/4/8/10负载、before/early/late位置及两类原始精确桥接共160条/320查询，仅原dev stable，不评分confirm。

新增 `tasks/cbf_record_interference.py` build/collect/summarize，复用背景来源校验、session/模型/评分/显存计时；加入64评分桥接、逐token槽外不变、双生与标签平衡、父缓存/权重不变检查、分域/分组配对差和错误颜色审计。新增 `tests/test_cbf_record_interference.py` 三项构造与失败拒绝测试；本地新旧6项通过（5.166秒）。新增 `scripts/run_cbf_record_interference.sh`，ROOT/SOURCE/READABILITY/PYTHON/MODEL/TOKENIZER参数与旧入口一致，严格失败标记、不覆盖旧输出。无新依赖，baseline模型/runtime不改。

数量曲线由末段向前扩展，位置和键身份也未完全正交；报告将明确这些限制。natural0可读性门槛为target/anchor整体与每域均≥75%，不自动触发V/F或控制器训练。旧新来源排除器的design/chunks支持问题不影响本次有意复用dev，留待新来源实验前修复。

固定槽位实验采集提交 `f163f17` 已同步hku-gpu2，tmux `cbf-record-interference-20261009`、ROOT `/home/ctj/cbf_ttt_record_interference_20261009`。服务器11项测试通过（5.124秒），真实tokenizer的suffix/行边界检查通过，160条/320查询设计已构建；原始数据SHA `bf171476791db7e20511ff801ddfd73c9e90a3b3636ae00bbf283176ed1eb338`。新增启动归档及design，仅含元数据；结果尚待完成。

采集期间新增只读 `scripts/audit_cbf_record_interference.py`，准备在完成后按原来源重新读取背景并核验每个槽实际内容、两个桥接原输入、原dev身份、summary重算、权重/结果哈希及confirm未评分。另补完整汇总与篡改拒绝测试：合法分片哈希不允许掩盖NLL/logits不一致，4项本地测试通过（5.086秒）。此审计与测试变更不更改正在运行的采集代码或设计。

两类桥接于采集中先行核验：32条上下文、64查询NLL最大误差0，八选一与全词表首token预测均一致。此处只核验桥接，不依据新增条件部分分数修改设计。

同步修复后续新来源构造的已知遗漏（仅本地，尚未用于本次已冻结采集）：`tasks/build_cbf_memory_value.py` 新增 `old_manifests` 同时识别旧meta和新版design并排除自身输出；`excluded_values`识别sources/source_ids字符串列表；新增 `context_texts`支持当前chunks与旧prefix/ids/context_ids；候选去重加入publisher id/url并拒绝重复source_id。新增 `tests/test_cbf_source_exclusion.py` 两项回归测试，证明新旧格式均能排除且输出目录不自排除、chunks正确进入文本扫描。新旧13项本地测试通过（5.866秒），尚未执行新的全语料构造，不能称为完成下一轮数据准备。该修复不修改baseline、旧归档或当前运行条件。

来源排除修复的临时副本在服务器真实归档上只读核验通过：识别46个manifest、713个排除值；原memory_value的84个选中来源，以及readability/record_interference各28个选中来源哈希全部进入排除集；当前6个chunks正确解码。归档 `record_interference/source_guard_audit.json`。没有构建新语料，未修改远程正在运行的模块。


### 固定槽位记录干扰完成（2026-10-09）

15:18:53至15:39:54 +08:00，1261秒；两卡各80条、共160轨迹/320查询。64桥接评分误差0，父缓存/权重不变；独立审计按原parquet重取背景、核验每槽实际token和桥接原输入，重算summary完全一致。最大allocated/reserved 14.770/18.904 GiB，两分片1232.576/1205.033秒；rollout均值4.118秒、query0.103秒，无OOM。

原始10条干扰目标准确率31.25%，natural0与newline0均100%；锚点均87.5%。主移除对比+68.75个百分点，8/8来源组改善，两域目标都100%，最弱域锚点75%达到预登记门槛。natural末段0/2/4/8条目标100/75/62.5/68.75%，并非单调；不同2条位置锚点反应不同，且键身份未正交。newline_late2目标56.25%，说明效应量对占位方式敏感。目标100%是八选一，natural0全词表首token仅12.5%。结论限于输入读取干扰，M为空，不能称为遗忘算法有效。

归档 `record_interference/{REPORT.md,summary.json,execution_audit.json,source_guard_audit.json,design.json,RUN_STATUS.md,.gitignore}`，同步计划和README完成入口。复现/审计命令、文件/配置项、与附件建议关系、风险和TODO见REPORT。当前旧confirm未评分，未启动新来源四场景Q、V/F、控制器或联合训练。后续需保留含干扰任务校准读取，冻结协议后用已修复排除器建立新来源，再按Q→V→F推进。

### 同键值位置对照（2026-10-09，执行前）

新增 `MATCHED_POSITION_PLAN.md`：两组原第3/6块干扰键值，各固定两条及内部顺序，分别移到第1/3/6块；自然占位、原QA、目标/锚点第2块和总长度不变。只复用原dev stable，none加六条件共112轨迹/224查询；三个旧条件精确桥接96评分。每donor内p6−p1/p3及两donor等权配对为主，身份×位置交互单列；不把两donor当独立来源，不自动进入四场景Q/V/F。

新增 `tasks/cbf_matched_position.py` build/collect/summarize，直接搬运donor正文token，长度或行尾token不一致则拒绝；验证固定槽外不变、事实块不变、双生/八类平衡、桥接及权重/缓存审计。旧 `tasks/cbf_record_interference.py` collector只添加可选validator/bridge_cells，默认行为保持原实验路径；模型/runtime不改。新增 `scripts/run_cbf_matched_position.sh`，ROOT/REFERENCE/PYTHON/MODEL可配；新增四项构造、错误输入和完整汇总/篡改拒绝测试。新旧10项本地测试通过（5.888秒），无新依赖。

本轮结论限制为自然占位下相同身份的输入位置干扰；移动仍改变周围上下文与相对距离，不能把它解释为M遗忘收益。后续四场景进入要求和已有QA弱项明确记入计划。采集尚未启动，原始文本/token/逐条分数不上传GitHub。

同键值位置采集提交 `6a06b5d` 已同步并启动tmux `cbf-matched-position-20261009`，ROOT `/home/ctj/cbf_ttt_matched_position_20261009`。服务器15项测试通过（5.635秒），112条/224查询构建通过；数据SHA `c0ca99d886a6761a61ec93199f699df7558614da1bdfe67d711eae7e86c98c7e`。新增启动状态归档。

新增只读 `scripts/audit_cbf_matched_position.py`，完成后从原parquet重建每个干预输入并重算所有指标。启动后、未查看新增位置评分的元数据核查发现donor3目标8/16颜色重合、锚点0/16；donor6目标0/16、锚点16/16。计划追加透明说明：独立审计增加探索性颜色重合/不重合分层及共同不重合目标子集；主采集、预登记对比、样本和规则不变，空组不报告虚假0%或100%。该混杂会限制键绑定解释，后续新任务需预先采用不重合干扰标签。

采集开始时间为2026-10-09 16:13:46 +08:00。三个旧条件的桥接已独立核验：48上下文/96查询，输入/模型/权重哈希一致，NLL最大误差0，八选一和greedy预测逐条一致。辅助颜色重合分层新增空组/共同样本测试，5项matched_position本地测试通过（0.773秒）；该诊断不更改采集代码或主summary规则。


### 同键值位置对照完成（2026-10-09）

16:13:46–16:28:54 +08:00，908秒；112轨迹/224查询齐全，两卡各56条。96桥接评分NLL最大误差0，输入/模型/权重与预测一致；独立审计从原parquet重新构造所有输入并重算summary，通过。峰值allocated/reserved14.770/18.904 GiB；两分片881.691/853.088秒，rollout均值4.117秒、query0.103秒，无OOM。

d3在位置1/3/6的目标准确率87.5/87.5/93.75%，d6为81.25/56.25/75%。两donor等权主差p6−p1为0个百分点、p6−p3为+12.5个百分点；预设“一致末段干扰增强”不成立。锚点和来源域响应不同，不支持统一越新越坏规则，不能解释为位置无效或遗忘机制失效。

探索性诊断：d3重合目标三位置均100%，不重合目标75/75/87.5%；d6所有锚点均重合，缺少不重合锚点对照。固定两donor均不重合的共同8目标/8来源组，d3为75/75/87.5%，d6为75/62.5/62.5%，方向仍不同。该诊断在启动后因输入元数据检查补充，主样本和规则未改，不当独立确认。

新增归档 `matched_position/{REPORT.md,RUN_STATUS.md,design.json,summary.json,execution_audit.json,.gitignore}`，同步README和计划完成入口，完整命令/文件/配置/风险/后续事项见REPORT。没有新来源四场景Q、V/F、控制器或联合训练。下一步优先构建排除正确颜色重合的统一四场景任务，冻结读取协议，再使用新来源按Q→V→F推进；旧confirm未评分。本轮依旧是M=0的输入位置对照，不声称快速记忆遗忘有效。

### 无颜色重合四场景读出校准（2026-10-09，执行前）

新增 `DISJOINT_READOUT_PLAN.md`，保留dev0–7四场景、原记录数/设备键/事实位置/六块背景，干扰palette固定排除两个双生目标值、锚点值和correction旧值。只重编码无关键记录的颜色，要求相同token长度且恰一token/变色记录；合法target/anchor全部记录保持原值。原QA和固定binding指令各评分，不添加答案候选提示。clean64轨迹加两域各一来源组的四场景16桥接，共80轨迹/288查询。

新增 `tasks/cbf_disjoint_readout.py` 的clean_scene/make_rows/validate/build/summarize/choose_format；复用多格式collector，`tasks/cbf_readability.py`仅增加可选validator/bridge_cells/identity_fields参数，旧默认行为不改。新增 `scripts/run_cbf_disjoint_readout.sh`（ROOT/SOURCE/PYTHON/MODEL/TOKENIZER可配）、四项构造/重合拒绝/逐域逐场景门槛/完整汇总与篡改拒绝测试。新旧13项本地测试通过（1.159秒），无新依赖或baseline模型/runtime变化。

每格式4场景×2域×2问题的16格均须≥75%，按最弱格最大选择，平局qa；没有合格格式则停止新来源/M写入，不删难例或改模板追加。若通过，冻结格式再以修复来源排除器构建新来源进行Q，之后才按原协议资源smoke/V/F。此时尚未启动GPU采集，旧confirm不评分。

### 无颜色重合四场景读出校准完成（2026-10-09）

采集提交 `45525be1e59d731b7427f5a566d141455d7bab24` 已同步hku-gpu2并运行结束；ROOT `/home/ctj/cbf_ttt_disjoint_readout_20261009`，tmux `cbf-disjoint-readout-20261009` 正常退出。实际开始/结束18:48:08–18:59:10 +08:00，662秒；两卡各40轨迹，共80上下文/288查询。服务器14项测试通过（2.664秒）。峰值allocated/reserved14.779/18.893 GiB，分片638.672/618.778秒；rollout均值4.114秒、query0.104秒。

新增 `scripts/audit_cbf_disjoint_readout.py`：audit(root, tokenizer)按原输入重编码所有记录/查询，核验源文件哈希、双生、固定背景和位置、624条不重合干扰与144条合法保护记录；实际改变572个颜色token。重算summary完全一致，32桥接查询误差0，权重/父缓存不变，confirm未评分。CLI为--root/--output/--tokenizer，无新依赖；不修改baseline或采集算法。

首次归档传输因自动审批服务额度故障未执行；用户继续后经正常审批重新上传并运行审计通过。另一次SCP连接中断后下载重试成功；本地summary/design哈希与远程execution_audit吻合。没有把未执行的检查计为通过。

原QA四场景目标准确率0/100/25/18.75%，binding为25/100/25/43.75%。原QA失败14/16门槛格，binding失败12/16，最弱均0%，无格式被选中，终态 `stopped_by_disjoint_Q`。按事前规则停止新来源/M写入/V/F/控制器或联合训练。M=0诊断不能证明或否定遗忘机制；四色干扰分布也改变，不能把旧结果差异全部归为颜色重合。

新增归档 `disjoint_readout/{REPORT.md,RUN_STATUS.md,design.json,summary.json,execution_audit.json,.gitignore}`，更新DISJOINT_READOUT_PLAN与README完成入口。REPORT包含目标/假设、逐文件说明、参数、总体及分域准确率/NLL/首token、资源、完整复现/审计命令、附件对应、baseline兼容性与限制。原始语料/token/逐条评分和权重留远程。下一项TODO为冻结任务上的原始Qwen3-4B与最终1B checkpoint能力对照，并核对可比推理路径；本轮没有启动该对照。附件局部遗忘/多分支/端到端联合训练仍需后续Q→V→F验证。

### 原始4B与最终1B checkpoint读取对照（2026-10-09，执行前）

按用户下一步要求，新增CHECKPOINT_READOUT_PLAN.md冻结同一disjoint数据（80上下文/288查询）及两种格式。两个checkpoint共用仓库普通Qwen3类（TTT关闭、M=0、完整KV、BF16/SDPA、TF32关闭），最终checkpoint只允许排除明确的ttt_conv/ttt_proj参数。最终模型全部288查询须复现旧CBF M=0候选NLL≤1e-5与相同预测；原始模型另以Transformers原生类复现固定16桥接上下文/32查询并核对实际主干权重哈希。总176上下文/608查询，不进入新来源或V/F。

新增tasks/cbf_checkpoint_readout.py（PlainSession、严格参数加载、逐查询缓存隔离、collect、validate_scores、compare_scores、summarize及源组配对差）、scripts/run_cbf_checkpoint_readout.sh（双GPU三个固定分支，ROOT/SOURCE/PYTHON/ORIGINAL/FINAL可配，拒绝覆盖），tests/test_cbf_checkpoint_readout.py（三项加载/篡改/全候选路径检查测试）。不更改baseline和旧实验模块，无新依赖。新增与disjoint共7项本地测试通过（0.501秒），shell语法/diff检查通过；本地系统Python缺少torch，完整12项中的缓存测试不能在本地执行，将在远程现有环境作为启动前门槛执行，不把缺依赖计为通过。

当前仅完成方案及代码，尚无本轮模型对照分数。结论将按每模型每格式16格统一门槛及final−original源组配对汇总；八选一/全词表首token不当自由生成准确率。原始权重与训练后checkpoint的任何差异仍需区分训练、加载与导出原因，有限dev不能证明普适训练损伤。

对照实验采集提交 `87866f6` 已同步并启动tmux `cbf-checkpoint-readout-20261009`；ROOT `/home/ctj/cbf_ttt_checkpoint_readout_20261009`，实际启动23:27:34 +08:00。服务器全部12项测试通过（2.123秒），双卡原生原始模型正常加载。新增checkpoint_readout/RUN_STATUS启动归档。

采集期间新增只读 `scripts/audit_cbf_checkpoint_readout.py`：结束后重算完整summary并核验源构造审计、checkpoint源文件哈希、推理模式及完成状态；尚未执行完成审计。另补完整176上下文汇总及不同原生权重拒绝测试，4项checkpoint本地测试通过（0.211秒）。这些改动不修改运行中的采集逻辑或冻结矩阵。

原生路径16轨迹完成后，固定32查询与仓库路径提前数值桥接误差0、预测一致，记录远程early_native_bridge.json；两卡原生主干参数数目4,022,468,096，无任何加载缺失/额外/形状不匹配，实际权重哈希一致。这里只是完整运行结束前的数值核验，主矩阵结论和最终权重/完成审计仍待完成。补充汇总测试后新旧共8项本地测试通过（0.563秒）。

### 原始4B与最终1B checkpoint读取对照完成（2026-10-09）

23:27:34–23:52:12 +08:00，1478秒（24分38秒），176上下文/608查询齐全；tmux已正常退出。原始两分片656.381/634.864秒，最终656.001/639.464秒（含加载/哈希）；两主路径峰值allocated/reserved均14.694/18.957 GiB。原生子样本157.264/154.671秒。逐轨迹/查询时间与显存保留远程，未上传原始数据或权重。

原始加载无缺失/额外/形状不匹配参数；最终仅排除明确的14个TTT conv/proj参数。主干参数数目均4,022,468,096；原始原生/仓库实际主干哈希相同。原始32查询与原生路径、最终288查询与旧CBF M=0全候选NLL误差均0，预测一致。新增审计脚本已在远程实际执行：重算summary完全一致，旧构造审计和源checkpoint文件哈希复核通过，权重/缓存不变；本地summary和审计脚本哈希复核通过。一次SCP连接中断后正常重试成功。

明确binding指令下，原始四场景目标50/100/68.75/75%，最终25/100/25/43.75%；目标准确率均值73.4375%→48.4375%（−25个百分点）。原始锚点四场景均100%，最终62.5/68.75/56.25/50%，均值降40.625个百分点。绑定指令组内全部目标/锚点均值在6/8组下降、2/8持平，没有组改善。原始binding失败3/16格，最终12/16；原QA分别失败9/16和14/16。两者均未通过统一门槛，说明共有任务难点和最终checkpoint额外下降同时存在。不能把M=0对照当遗忘机制有效性测试，也不能直接确定训练算法、优化器或导出是根因。

新增checkpoint_readout/{REPORT.md,RUN_STATUS.md,summary.json,execution_audit.json,.gitignore}正式归档；更新CHECKPOINT_READOUT_PLAN、README和本日志。报告完整包含目标/假设、加载与文件说明、参数、总体与分域/源组配对表、NLL/首token、资源、复现命令、附件对应、baseline兼容性、已完成与TODO及局限。核心新增4项测试（含完整汇总/权重不匹配拒绝）和disjoint共8项本地通过；服务器采集启动前12项通过。

本轮无baseline模型/runtime改动，无新依赖；未做新来源Q、M写入、V/F、标签扩容、控制器或联合训练。下一步建议固定少量长度条件，保留事实键值/干扰记录、缩短自然背景并保持24K对照，验证可读区间；长度变化也影响相对距离，必须另列限制。通过新的冻结Q并在新来源确认后才回到附件局部遗忘/多分支/联合训练。该建议尚未执行。

### 固定记录6K/12K/24K读取对照（2026-10-10，执行前）

按用户下一步要求新增LENGTH_READOUT_PLAN.md：保留原disjoint四场景dev0–7的全部设备/颜色/记录顺序和两个query，每个4096-token段取最后1024/2048/4096 tokens，固定6K/12K/24K。要求header及两条记录完整、不重新分词、不补填充；短窗嵌套，真值/双生/无重合干扰保持。删除自然背景同时改变位置和距离，不能宣称单一纯长度因果效应。

新增tasks/cbf_length_readout.py（make_rows/validate精确尾窗、build/load_data、复用采集、reference_summary重算旧归档、summarize及逐组长度/模型配对差）；scripts/run_cbf_length_readout.sh（双GPU，ROOT/SOURCE/REFERENCE/PYTHON/ORIGINAL/FINAL可配，拒绝覆盖）；tests/test_cbf_length_readout.py四项构造/裁断拒绝/最长合格选择/完整汇总与长度身份篡改拒绝测试。旧tasks/cbf_checkpoint_readout.collect仅增加data_path/validator/identity_fields/expected_data_sha可选参数，原默认不变；baseline模型/runtime/训练配置不修改，无新依赖。

新采集两模型合计288上下文/1152查询：每模型两个短长度各64场景，另组0、2四场景双生的16条24K clean作64查询桥接。24K主分析复用上轮已审计的完整64条/模型，要求权重/配置/源文件一致、桥接所有候选NLL≤1e-5且预测一致。桥接不重复计入主矩阵；三长度两模型主分析384条件/1536查询。逐查询/轨迹耗时与峰值显存保留远程，公开聚合和元数据。

每模型/长度/格式仍须16个域×场景×问题单元全部≥75%；若最终短长度通过，仅选最长合格长度作为新来源Q候选，同长度最弱格最大选格式、平局QA。若均失败停止此版本标签扩容，不加长度/模板或删难例。不自动进入M写入/V/F/控制器/联合训练；后续需要区分当前前向分段与4096-token TTT写入边界。当前尚未GPU采集。

新增只读scripts/audit_cbf_length_readout.py：完成后独立检查实际每段suffix、完整header/记录/查询、144输入的1728条记录及三长度计数，重算summary、重新哈希源checkpoint并核验终态。新旧12项本地测试通过（7.512秒），shell语法、Python审计脚本语法、git diff检查通过；依赖torch的旧缓存测试将在服务器启动前运行。尚无本轮GPU结果。

采集提交a792bfc已同步hku-gpu2；tmux `cbf-length-readout-20261010`、ROOT `/home/ctj/cbf_ttt_length_readout_20261010` 于2026-10-10 00:11:23 +08:00启动。服务器17项测试全部通过（6.184秒），144个新输入/每模型576查询构造校验通过，数据SHA `dfaab6b69f23468d99881fa2a620f3ae4a79de60146caf4521edc6ac1e04aa22`；两个原始模型分片正常加载。新增length_readout/RUN_STATUS和设计归档，完整结果/独立完成审计仍待采集结束。

采集中对原始模型固定16条24K子集先行核验：64查询全部八候选NLL最大误差0，预测一致。该中途检查不替代最终所有分片权重/构造/汇总审计，也不依据短长度部分评分调整矩阵。

最终模型固定16条24K子集也已先行核验：64查询全候选NLL最大误差0、预测一致。两个模型的全部短长度条件仍按冻结矩阵运行；中途只做完整子集复现核验，不调整数据、门槛或格式。

### 固定记录长度对照完成（2026-10-10）

00:11:23–00:26:30 +08:00，907秒（15分07秒）；288新轨迹/1152查询齐全，tmux正常结束。主分析合并已审计24K完整旧数据后为384条件/1536查询，复现子集没有重复计入独立样本。原始两分片445.421/434.010秒，最终443.242/434.557秒；6K/12K/24K峰值allocated约9.346/11.128/14.694 GiB，reserved最高10.498/13.516/18.957 GiB，逐条时间/显存留远程。

独立脚本已实际执行，通过144输入/1728记录的逐段真实尾窗、完整header、事实/干扰/查询不变检查；summary重算完全一致，源checkpoint重新哈希通过。两个模型各64次24K查询的所有八候选NLL最大误差0，候选/全词表首token预测一致；父缓存/权重不变。下载设计及首次SSH审计遇到连接中断，正常重试成功；本地summary/design/审计脚本哈希与远程审计一致。

原始模型binding在6K/12K/24K的目标均值87.5/93.75/73.4375%，锚点98.4375/100/100%；12K通过全部16格，6K有1格失败，24K有3格失败。最终模型binding对应目标51.5625/59.375/48.4375%，锚点92.1875/54.6875/59.375%，失败6/12/12格；最终QA在6K/12K亦失败7/8格，没有合格短长度。短背景不能单独解决最终checkpoint的M=0读取问题，也不是越短越好。

终态stopped_by_length_Q，按预登记停止该版本标签扩容，不追加长度/格式，不删场景，不进入新来源Q、M写入/V/F、控制器或端到端训练。原始12K通过提供了可读正对照，建议下一步冻结该条件，比较中间checkpoint并加原生TTT路径诊断，核验训练/导出及主干变化和对快记忆的依赖；尚未执行。本轮M=0差异不能直接当作原生TTT整体退化或遗忘机制有效性结论。

新增length_readout/{REPORT.md,RUN_STATUS.md,design.json,summary.json,execution_audit.json,.gitignore}正式归档，更新LENGTH_READOUT_PLAN/README/本日志。报告包含完整总体与分域/源组配对表、NLL/首token、资源、构造/参数/函数/文件说明、复现及审计命令、与附件建议关系、baseline兼容性和风险/TODO。无新依赖，baseline模型/runtime/训练配置未改，原始文本/tokens/逐条评分/权重未上传。

正式结果归档提交7f155aa已推送GitHub。最后向hku-gpu2快进同步报告时，复用连接两次中断，随后使用同一SSH配置的新连接也在banner交换阶段超时；因此未确认服务器代码目录更新到报告提交。GPU实验、远程summary/独立审计及本地下载哈希验证均在此前完成，不受影响。下次SSH恢复后仅需在服务器仓库执行git pull --ff-only origin main，无需重跑实验。

### 固定12K checkpoint轨迹与原生TTT（2026-10-10，执行前）

用户要求继续下一步实验。SSH已恢复，服务器已补同步f12e237的上一轮报告；两卡空闲，10000至80000每万步DCP和最终81381均在，磁盘可用1.4T。新增CHECKPOINT_NATIVE_PLAN.md，在新评分前固定原始plain及10000/40000/81381各plain/native，64个既有12K场景；final_zero_lr额外16场景。合计464 rollout/1856查询，binding主指标、QA辅助。数据不增改，将六个2048段无损合并为三个4096块；用原始/最终plain与旧分块桥接，避免分块混淆。仍仅observed dev，不启用新来源/训练/标签/控制器。

新增tasks/cbf_checkpoint_native.py：reblock/build/load_data保存输入及源hash；NativeSession使用原生TTTDynamicCache且拒绝CBF路径，核验完整块/尾缓冲/查询不写入；native_digest隔离核验、native_stats三次更新各层相对位移；load_model严格加载和零lr对照；collect逐查询计时/峰值及前后参数hash；summarize严格完整性和各步/来源/机制聚合。新增scripts/export_cbf_diagnostic_checkpoint.py复用官方DCP转换并逐张量对照原DCP，源文件只读、独立导出目录、保存审计。新增scripts/run_cbf_checkpoint_native.sh负责测试/构造/导出/双卡，环境变量ROOT/SOURCE/PYTHON/ORIGINAL/CHECKPOINTS。新增tests/test_cbf_checkpoint_native.py覆盖无损分块/记录位置/双生真值/上下文不匹配拒绝及非零原生更新、部分块缓存、查询clone隔离、FP32零lr等价。

首次本地构造测试发现源record.chunk是1-based，已按原格式修正重分块映射并验证；没有用错误映射启动GPU。本地macPython无torch，原生张量测试将在服务器执行，失败即停止。baseline模型/runtime/预训练配置未改，无新增依赖。中间导出使用最终相同TTT结构配置，只复制配置不复制权重；风险是只测3个训练点、12K超出原6144训练长度、BF16流式差异与dev适配。不能将普通读取退化直接等同原生TTT退化，更不能视为局部遗忘有效性检验。完整方案/判读阈值/复现命令见计划。

采集代码7077d5b已推送GitHub并同步服务器，tmux cbf-checkpoint-native-20261010于2026-10-10 09:59:10 +08:00启动，ROOT /home/ctj/cbf_ttt_checkpoint_native_20261010。服务器11项测试全部通过（8.399秒），本轮数据已构建，正在中间DCP导出并逐张量复核；无评分结论。另新增scripts/audit_cbf_checkpoint_native.py独立完成审计：重算汇总、核对768条记录绝对位置/查询/标签、重哈希加载模型、复核DCP未变及导出审计来源、核验464/1856矩阵与终态。归档checkpoint_native/RUN_STATUS.md记录实际执行状态；原始语料/逐条结果/导出模型不入Git。

### 固定12K checkpoint/native：完成与结论（2026-10-10）

2026-10-10 09:59:10–10:33:42 +08:00完成464轨迹/1856查询，2072秒；采集代码7077d5b。11项服务器测试全部通过；本地12项中10通过、2无torch跳过，后者服务器已通过。10000/40000 DCP各413模型张量均为BF16，导出逐值相同、源文件未动。独立审计完成：summary重算一致、768条记录绝对位置/内容/查询不变、19模型/config文件重哈希、查询prefix与模型参数hash/version一致。审计SSH曾中断但实际进程已完成，新连接确认audit_passed并下载，本地summary/design/审计脚本SHA一致，无重复评分。

固定binding八候选目标/锚点百分比：原始93.75/100；10000关闭59.375/71.875、原生60.9375/81.25；40000关闭64.0625/50、原生42.1875/40.625；81381关闭60.9375/54.6875、原生50/48.4375。只有原始正对照通过全部16格75%门槛。普通读取下降已出现在首个10000步checkpoint，无法定位更细步骤。原生早期候选均值稍改善，但中期/最终下降，全部训练臂不合格。此为当前构造dev/12K条件的证据，不是自然任务或方法创新总体结论。

特别记录：40000步与最终native的binding正确答案首token均为0/128。事后仅解码已存argmax，均为token151645 <|im_end|>；不能用八选一准确率代替生成成功。10000步native NLL/首token也未与候选改善一致。新增scripts/summarize_cbf_native_greedy.py，从已审计结果聚合首token、保留summary/row/tokenizer/script哈希；无新forward、不调矩阵/阈值。该项为事后描述性诊断，主NLL/首token指标本身已预登记。

最终zero_lr原生与plain64查询所有候选NLL差0、预测/首token全一致。4096 vs 2048分块：原始/最终256查询预测一致率98.4375/97.265625%，最大NLL差0.867893/0.263764，不能逐值混用旧结果。主表统一4096；原始binding不变、最终binding目标从旧59.375变为60.9375。原生峰值allocated约11.871GiB、reserved13.8125GiB；逐条耗时/显存留服务器，rollout约plain1.289s/native1.344s，wall还包括哈希/加载/导出等审计成本。

归档checkpoint_native/{REPORT.md,RUN_STATUS.md,design.json,summary.json,execution_audit.json,greedy_diagnostic.json,.gitignore}，更新计划/README/本日志。报告包含假设、主表/机制/域/8来源配对、NLL/首token、资源、核验、文件/配置/命令、baseline兼容性、附件对应、局限/TODO。下一步建议固定12K对比原生累积、一次写入、清除旧快权重再写、禁写，检查错误终止与多次更新关系；骨干能力保留另需稳定writer训练对照。上述后续均未执行，不自动重跑1B或标签/控制器；仍需合格基座和新来源确认后再回到局部遗忘/多分支/联合训练。

正式报告c6b56d1已推送并快进同步hku-gpu2；服务器确认实验tmux退出、两卡已空闲。首token描述用新增正式脚本从已审计jsonl复现，分布不变；补归档summary/逐分片/tokenizer/script哈希，本地全部核对通过。所有本轮已授权实验及报告归档完成，无后台新训练任务。

### 转向任务训练语料：可验证事件流原型

用户指出连续诊断缺少目标进展，并询问能否通过构造训练语料实现图片中的全局/局部遗忘、多次决策及联合训练。新增TRAINING_CORPUS_PLAN.md，明确先训练可用writer，再收集实际反事实标签训练策略，最后联合训练；当前checkpoint退化不证明语料是唯一原因。现有LowRankWriter对detach候选做变换、CohortSession.commit_cohorts为inference_mode，均不能直接作为原生writer/多步联合训练闭环。计划写明分词/实际4096块装箱、答案mask、可微训练适配器等下一步代码任务，保留原始Qwen3-4B冻结骨干起点。

新增tasks/build_cbf_event_curriculum.py：可执行的赋值/撤销/会话重置规则；保留、撤销last1、撤销last2、全局reset四类；8文本块中4–8边界随机嵌套1/2/3点；同时问当前事实、历史事实、保护anchor；成对样本仅首块anchor值改变。group/entity/batch跨split隔离，三种不同措辞模板；inputs不含family/真值事件，targets教师私有；supervision_records只保留问题时刻之前文本/决策，warmup_records导出单块4事实查询。程序仅给语义答案，显式oracle_actions_generated=false，不能将撤销批次直接当成最优alpha标签。

新增tests/test_cbf_event_curriculum.py五项测试，全通过（0.166秒）；包括用独立反向账本重算全部问题答案、撤销后旧值恢复与anchor保留、reset、未来块/私有标签隔离、twin/完整矩阵、输出hash及禁止覆盖。生成experiments/cbf_ttt/event_curriculum_v1/pilot：train/dev/test8/2/2世界组，192/48/48会话（共288），1152/288/288答案监督（共1728），64/16/16单块warmup（共96）。JSONL本地保留，Git归档manifest/报告/代码；不使用旧诊断集生成训练样本，未模型评分test。

新增实验REPORT.md与.gitignore，计划含图片对应、训练目标/通路、数据规模、复现命令、下一步、baseline兼容性与风险。无旧模型/runtime/config更改、无新依赖。当前仅语义文本原型，未自然背景接入/分词/装箱/训练/动作标签，不声称已训练控制器或方法有效。合成规则容易产生显式指令捷径，后续需自然来源修订/弱提示与真实基准；同世界变体不能算独立样本。建议接下来以100步上限的writer训练验证学习通路，实际GPU方案需在训练适配器完成后固定。

构造代码2ccd5a3已推送GitHub并同步hku-gpu2。服务器CPU复现五项测试通过（0.123秒），在/home/ctj/cbf_ttt_event_curriculum_v1生成同一原型；cmp确认远程manifest与本地归档逐字节一致（其中含全部12份JSONL的SHA256），未启动GPU训练。补明确阶段A仅使用warmup/retain，撤销/reset留到阶段B，避免writer学习与策略学习混淆。

### 事件语料writer首次100步训练（执行前）

用户授权下一步训练。新增EVENT_WRITER_TRAINING_PLAN.md：原始Qwen3-4B骨干冻结、原生7层conv/proj FP32训练，单块4096+fresh-KV答案读取、100步AdamW lr1e-7/clip1/无weight decay、固定0/25/50/100评估。GPU0训练GPU1评估，保留空/错配/fullKV和anchor-twin对照，完整答案EM与全词表NLL，未评分test。阶段A仅warmup，不混入撤销/reset或训练控制器。

新增tasks/pack_cbf_event_warmup.py：train64/dev16样本分词，完整事实后缀+唯一程序生成维修背景装4096块；明确纯合成背景，未接入FineWeb/LongCrawl。联合分词验证答案边界，context/group/twin标识和固定异组错配，manifest哈希；test文件不读取。新增cbf_ttt/event_writer.py：严格加载原始骨干只允许14个新增writer缺失，零conv+正常随机proj；复用cbf_forward_mlp可微写入，freshcache不detach memory，答案/EOS对齐；骨干hash核验。新增tasks/train_cbf_event_writer.py：100步梯度/参数/FP32状态审计、原子checkpoint、异步评估、生成EM、分组对照、固定终点汇总。新增scripts/run_cbf_event_writer.sh：测试/装箱、双进程不同GPU、失败标记与完整终态。

新增tests/test_cbf_event_writer.py：装箱/答案边界/未读test、conv与proj真实梯度、原生非零delta和freshcache读取一致、答案位置/EOS；与事件语料测试本地9项中6通过3无torch跳过（0.486秒）。全部torch测试将在服务器运行后才启动。原baseline/runtime/配置未修改，无新增依赖。当前仅单块可微writer训练，不声称多次控制决策联合训练已完成；数值/范数停止线、dev判据及局限写入计划。

采集代码3a0457e已推送并同步hku-gpu2，tmux cbf-event-writer-100step于2026-10-10 11:01:41 +08:00启动，ROOT /home/ctj/cbf_ttt_event_writer_100step_v1。服务器9项测试全通过（1.649秒），装箱及原始骨干hash验证成功；100步已执行，单步约0.30秒，峰值allocated约10.2GiB，GPU1仍在评估。新增RUN_STATUS跟踪；此时未报告记忆有效结论。

新增scripts/audit_cbf_event_writer.py：完成后独立核对80条实际pack后缀/答案边界、100步固定洗牌序列/答案与EOS loss分解、checkpoint实际FP32参数变化及optimizer状态、272条dev策略生成EM与16条train probe数量、原始模型文件hash和完整状态；test不评分。summarize增加save=False选项，使独立审计重算不覆盖原始summary；默认CLI行为保持不变，不改训练/评分计算。

### 事件语料writer首次100步：完成与负结果

2026-10-10 11:01:41–11:03:58 +08:00，137秒完成测试/装箱/加载/100步优化及0/25/50/100评估。随后独立审计通过：80条实际pack、100步采样与loss分解、FP32保存参数/optimizer、14个writer张量实际更新、冻结骨干及原始文件hash、272次dev策略查询/16次train probe和summary重算；test未tokenize/评分。服务器9测试全通过（1.649秒）。本地下载summary/审计，summary/script/checkpoint引用hash已相互核对。

固定终点正确/空/错配/完整KV答案NLL为2.053219/4.277348/2.235196/0.105974，所有策略严格EM均0/16。正确对错配NLL平均改善0.181977，但两个dev世界增益+0.415777/−0.051823；anchor正确减twin NLL为+0.012688，未显示单事实区分优势。train固定16题probe NLL2.702084、EM0；前10/后10在线训练NLL4.582447/1.937432不是同题配对。passed_memory_gate=false，不追加训练预算或启动控制器。

平均单步0.299468秒，训练峰值allocated10.186427GiB、reserved10.792969GiB；逐步/逐写入/逐查询资源留远程。原始Qwen3-4B骨干未动，当前训练conv/proj共45,964,800参数；100步first im_end概率约1.03e-7，不能把当前失败直接当作此前1B骨干的首token终止问题。

只读查看已有100步生成，发现正确memory的裸code_提前终止/错误编号，以及full-KV正确编号后继续解释或先解释耗尽16token预算。新增scripts/summarize_cbf_event_generation.py用于事后首行/首个数字编号匹配及裸前缀/耗尽预算统计，无新forward、不改变严格EM/门槛。审计完成后SSH出现证书校验失败/连接关闭，多次安全重连仍未成功，因此该事后全量统计尚未在服务器执行；不据部分行推断全量次数，不禁用证书校验。核心100步训练/评估/独立审计已在断线前完成，工件仍保留服务器。

新增event_writer_100step_v1/REPORT.md、summary.json、execution_audit.json，更新RUN_STATUS/训练计划/语料计划/README。报告列出目标/假设、修改文件及函数模块/路径配置、附件对应、参数、复现命令、baseline兼容、负结果/局限/TODO。无baseline代码或新增依赖变动；未实现多步联合训练/全局局部遗忘实验。下一轮建议先固定预算学习完整编号并区分内容/格式指标，待正确vs错配/twin可复现后再进B/C；该后续未执行。

用户再次要求重连后，hku-gpu2连接恢复；服务器从c25c35a快进同步52c5390报告。只读执行summarize_cbf_event_generation，不新增模型调用/test评分：100步正确记忆7/16仅code_，8/16含数字编号但均错误，另1/16无完整数字编号；首行和首个编号正确均0/16。full-KV四checkpoint均首行/首个编号正确12/16，16/16耗尽预算，严格EM仍0。结果补存generation_diagnostic.json并更新REPORT/RUN_STATUS；事后描述不改变预登记门槛。检查时两张5090利用率均0%，显存139/18MiB。未启动额外训练。

### 阶段A第二轮：少量世界的内容训练（执行前）

用户授权继续实验计划。新增EVENT_CONTENT_TRAINING_PLAN.md：两张5090分别运行CE和content_pair；共同从原始冻结Qwen3-4B/相同新writer开始，排序前2个训练世界16问题/8对、原dev2世界16题，复用4096装箱且不读test。固定400步、每问题50次，初始化seed301/顺序seed302、lr1e-7、FP32 writer/state、0/100/400评估，816策略查询。content_pair为0.75数字+0.125前缀+0.125EOS及仅变化anchor的0.5 softplus(.2+正确数字NLL−twin数字NLL)；普通CE臂同样计算交叉前向但成对项权重0。配置/参数/推理g=1不变，不对本轮结果保证收益。

新增tasks/train_cbf_event_content.py：select_rows/split_answer验证数字mask、twin因果对应与两世界错配；training_order固定50轮；regions/objective保留两个记忆的可微梯度；run固定训练/检查/保存/evaluate；extra_score记录预登记首行/首个编号指标、数字NLL及原严格EM；summarize验证全矩阵/初始两臂/冻结控制并按固定400终点判定train/dev内容门槛。新增scripts/run_cbf_event_content.sh双臂tmux内并行；scripts/audit_cbf_event_content.py重建数据/顺序/目标、复核保存参数与FP32 optimizer、解码EM/编号/资源、重算summary、与上一轮dev空/full-KV完全桥接及源文件hash。新增tests/test_cbf_event_content.py验证小数据选择/50次暴露、数字边界拒绝、变化/不变事实对比梯度、真实两记忆原生writer梯度。

本轮新增独立入口，无baseline/runtime/上轮训练修改和新依赖。ROOT/SOURCE/MODEL/PYTHON为路径配置；预算/目标/门槛冻结于计划与代码。训练门槛要求编号≥75%、相对空/错配≥25百分点、两训练组各≥50%、twin数字NLL优势；dev≥25%、相对对照≥12.5百分点、两组数字NLL增益≥.05、twin优势；严格EM仍单独报告，不将新内容指标替代旧EM门槛。只验证阶段A，附件三项遗忘/多边界/联合训练尚未进入。当前两卡空闲、磁盘约1.4T，服务器测试通过后才启动；未根据结果调整超参。

采集代码3901eca已推送并同步服务器，2026-10-10 14:20:09 +08:00 tmux cbf-event-content-v1启动。ROOT=/home/ctj/cbf_ttt_event_content_v1；服务器13项测试全通过（1.895秒），本地13项8通过5无torch跳过（1.037秒）。分词mask/选组准备完成，两个原始模型均已加载；后续按固定计划执行，无中途方法有效结论。

### 阶段A第二轮：双臂400步完成

2026-10-10 14:20:09–14:28:25 +08:00完成，496秒；采集3901eca。每臂400步/每问题50次暴露，合计800步/816策略查询。13服务器测试通过，独立审计通过32行选择/mask/twin重建、目标分解/顺序、实际writer更新及FP32状态、生成EM/编号、summary重算、两臂初始化逐张量相同、旧dev空/full-KV原指标精确桥接、冻结骨干与源文件hash。test未读取评分。本地下载summary/design/audit并核对SHA一致；下载尚未结束时一次本地读取报文件未找到，等待scp成功后重读/验hash通过，未影响远程实验。

固定400终点训练完整编号与严格EM：CE4/16(25%)、content_pair7/16(43.75%)；空0/16，错配分别2/16与3/16。开发两臂全部0/16。训练数字NLL CE0.578284、content_pair0.734783，成对方案虽然多答对3题，但均值NLL更差；anchor的twin减正确数字NLL CE+0.029201、content_pair−0.023363，后者未显示单事实绑定优势。开发数字NLL7.447014/7.665306，高于空2.662896；anchor优势−0.009225/−0.093651。两臂train/dev门槛均false。只支持小训练集部分拟合，不能声称成对机制或选择性遗忘有效。dev世界/值/模板同时变化，泛化失败不能单独归因某一因素。

平均训练step CE0.700420秒/content_pair0.686629秒，allocated峰值两者13.578647GiB、reserved13.902344GiB。每臂800个4096写入块=3,276,800重复token暴露（只有4个不同上下文、2独立世界），非同量独立数据。逐条时间/显存和权重保留服务器。两臂初始checkpoint文件hash不同源于manifest的arm字段，writer张量审计相同。无超参中途更改/延长预算/控制器启动。

归档event_content_v1/{REPORT.md,RUN_STATUS.md,design.json,summary.json,execution_audit.json}，更新EVENT_CONTENT_TRAINING_PLAN/TRAINING_CORPUS_PLAN/README/本日志。报告包含目标假设/全指标分组/门槛/资源审计、修改文件与函数配置、命令/附件映射、baseline兼容性和风险/TODO；无新增依赖或baseline/runtime更改。下一步建议对同一context多事实共同监督，与逐query优化作预算固定对照，验证是否存在事实间优化干扰及双生anchor切换；这是假设和未执行后续，不是本轮已证明原因。

### 逐问题与同上下文多问题监督：干扰对照（执行前）

用户要求比较两种监督并检查不同事实的优化干扰。新增FACT_INTERFERENCE_PLAN.md：固定普通CE、原始冻结4B骨干/新writer初始化seed301，复用前轮32行及4096块；三臂sequential、joint_context、mixed_context（后者区分上下文分组与batch效应）。seed403每轮4×4问题表；50轮每题50次，每臂800问题暴露，更新800/200/200次，写入800/200/800块。0/400/800暴露统一评估，共1224策略查询；不将不同更新数/计算量称为完全配平。

新增tasks/cbf_fact_interference.py：schedules构建顺序/分组；batch_loss在同context共享可微memory、各query fresh KV；run复用原生writer/评分并保存问题暴露及真实optimizer步；probe_context在400/800暴露、4训练context各做4单题与1联合一步AdamW干预，测4事实前后loss及4×4梯度余弦。cpu_copy/equal_state/restore独立保存和逐张量恢复参数/动量/step，回滚后读出漂移≤1e-6；最终状态必须等于干预前checkpoint。0.01数字NLL阈值定义自己改善且其他事实恶化的事件；describe_probes保存矩阵/计数。三臂共120干预，不进入主训练，不访问dev/test作为干预目标。

新增scripts/run_cbf_fact_interference.sh：GPU0 sequential，GPU1 joint后mixed；测试/prepare/汇总/审计/失败终态。新增scripts/audit_cbf_fact_interference.py：数据与固定预算重建、CE分解/步数、FP32状态、初始参数相同、1224评分解码/前轮控制桥接、120干预恢复记录/矩阵形状对称/独立事件计数、源文件hash；不新增forward，余弦向量未另存，独立审计不声称重新计算梯度。新增tests/test_cbf_fact_interference.py验证暴露/分组、干扰定义、共享图与独立平均梯度一致、真实AdamW多分支恢复及读出不漂移。本地17测试10通过7无torch跳过（1.025秒），Python/shell语法通过；服务器测试通过后启动。

ROOT/SOURCE/MODEL/PYTHON为路径配置，预算/lr1e-7/clip1/无decay/阈值执行前固定。所有实现为独立入口，无baseline/runtime/旧训练改动和新依赖。风险：2训练世界/2开发世界/1seed，模板与世界同时变化；AdamW有限步干预不等同原始梯度内积；联合组省写入且优化步数少，必须结合mixed解释。存在干扰不等于遗忘机制有效，未进入B/C。方案/状态/后续实际报告同步Markdown。

077e0a4已同步服务器，2026-10-10 15:15:55 +08:00首次启动在测试阶段停止，尚未prepare/模型评分/训练。17测试中恢复参数/optimizer/读出检查已通过，但微型随机模型的一步更新未产生可分辨的FP32 CE变化，测试的“至少一题loss改变”断言失败。只增强该CPU测试fixture的lm_head/conv/proj尺度和toy优化lr以验证非零干预；正式4B/lr1e-7/预算/阈值未改。保留首次preflight失败日志，重测通过后在原计划新输出目录执行，不重复已有GPU实验。

CPU修正的单项服务器测试通过（1.339秒）。64fc0c7已推送；先cmp确认服务器临时上传的测试文件与origin/main逐字节相同，再清理这一处自有临时改动并快进合并，未覆盖用户改动。失败预检目录移至/home/ctj/cbf_ttt_fact_interference_v1_preflight_failed_077e0a4。2026-10-10 15:22:07 +08:00以64fc0c7正式启动，17服务器测试全通过（2.069秒）；实验配置/预算未改。

### 事实优化干扰：三臂与可回滚干预全部完成

2026-10-10 15:22:07–15:35:36 +08:00，809秒。sequential/joint_context/mixed_context更新800/200/200步，每臂800问题暴露；1224策略评分、120次干预齐全。17服务器测试和独立审计通过。审计重建预算/CE分解、保存FP32 writer/optimizer计数、初始化逐值相同、原控制指标精确桥接、生成EM/编号、120分支恢复记录/无读出漂移、独立事件计数、summary重算、原始骨干/权重文件hash；余弦只独立核验形状/范围/对称性，不声称无原始向量下重算梯度。test未评分。

中点单题干扰事件（自身数字NLL降≥.01且其他事实升≥.01）依次9/16、10/16、13/16；终点0/16、7/16、6/16。中点负梯度对13/24、6/24、3/24，终点5/24、9/24、6/24。逐问题中点具体更新unit_3ca26ad0_0：自身−0.061854、anchor+0.144978。六checkpoint同状态联合一步的四事实均值变化均优于单题一步均值；联合仍有0/2/3等受损事实。sequential终点仍4/48非目标事实变差，只是自身改善条件不达阈值，不能写成完全无干扰。各分支/事实相关，不作统计显著性结论。

固定800暴露终点训练正确/错配EM和编号：sequential14/16对14/16，joint14/16对7/16，mixed13/16对5/16；train门槛分别false/true/true。三臂dev均0/16，全未通过。训练数字NLL0.213237/0.207134/0.102075，开发11.471340/8.280446/9.246504；同context只比mixed多1题但NLL更差，不能称独特效果。sequential高训练成绩不依赖正确记忆，说明仅看训练EM会误判。存在局部优化干扰，但不是泛化失败的充分解释。

训练计时总241.927/88.344/229.789秒，峰值allocated10.188/11.987/16.266GiB。新增scripts/summarize_cbf_fact_resources.py只读聚合训练/评估/干预资源并保存来源hash、new_model_calls=0；通过临时脚本在服务器执行，resources.json已归档。每臂8次context干预合计39.634/37.927/39.108秒，峰值allocated12.851/12.849/12.848GiB；context时间包含分支，不能重复相加。联合复用写入、优化步更少，非严格FLOPs配平。资源JSON源hash与审计/summary本地核对一致。

归档fact_interference_v1/{REPORT.md,RUN_STATUS.md,design.json,summary.json,execution_audit.json,resources.json}，更新FACT_INTERFERENCE_PLAN/TRAINING_CORPUS_PLAN/README/本日志，包含目标假设、全部指标/分组/矩阵例子、文件/函数/配置/复现命令、附件映射、baseline兼容性、已完成/风险/TODO。实验结束两卡空闲（139/18MiB、0%），原始骨干不变，无新增依赖或baseline/runtime改动。后续建议同查询动态重赋值及更多独立世界泛化训练，保留multi-query和负对照；未自动启动后续或控制器。
# 2026-10-10：仅训练遗忘策略的严格复现（执行中）

## 目标与实验假设

论文主线暂时固定 `g=1`，只验证 session 快记忆遗忘。旧控制器相对持续累积有收益，但在旧 test 上没有超过固定完全清除；本轮增加三个随机种子的统一 rollout，以及清空 KV 后的 Correct/Wrong/Empty fast-memory 对照，分别检验“快记忆有内容”和“学习策略优于最佳固定策略”。执行前协议及停止条件见 [`FORGETTING_ONLY_VALIDATION_PLAN.md`](FORGETTING_ONLY_VALIDATION_PLAN.md)。

## 新增文件

| 文件 | 内容 | baseline 影响 |
|---|---|---|
| `tasks/cbf_forgetting_only_validation.py` | 确定性跨来源 wrong-memory 配对、四种记忆条件采集、按历史查询汇总、来源组 bootstrap 和两级门槛 | 独立任务入口，无默认路径改动 |
| `scripts/run_cbf_forgetting_only_validation.sh` | GPU 空闲等待、测试、双卡 smoke、六策略正式采集、汇总和审计 | 独立脚本，不修改训练配置 |
| `scripts/audit_cbf_forgetting_only_validation.py` | 独立核验 300 场景覆盖、donor 对称性、有限损失、非空记忆、汇总均值和旧 rollout 逐场景复现 | 只读审计 |
| `tests/test_cbf_forgetting_only_validation.py` | CPU 检查配对确定性及 historical-only 内容门槛 | 测试辅助 |
| `FORGETTING_ONLY_VALIDATION_PLAN.md` | 研究口径、固定资产、阈值、停止条件与运行命令 | 文档 |

## 实现边界

- `correct_memory/wrong_memory/empty_memory` 都从空 attention KV 开始，避免把上下文 KV 读取误算为快记忆内容。
- 内容门槛只使用 `historical` 查询；`current_or_new` 仍完整记录，因为旧 short-tail 协议中的新事实可能未跨过 TTT 写入边界。
- wrong memory 来自同 regime、相邻的另一来源组，映射互为 donor；六种策略使用同一配对。
- 三个控制器固定为旧 500-source-group 训练的 seed 42/43/44 checkpoint，不读取本轮 test 重新选种子。
- 当前旧控制器包含当前 chunk 的 semantic 特征；history-only 输入消融属于通过本轮门槛后的下一阶段，本轮不提前声称已经验证。
- 尚未产生正式 GPU 结果；hku-gpu2 当前被其他用户的双卡任务占用，运行脚本将在不抢占任务的前提下排队。

## 完成结果

任务使用提交 `b2fd6d3`，17:30:47 入队，因其他用户双卡 27B 推理等待至 20:25:55，20:41:39 完成。11 项服务器测试通过；六策略各覆盖 300 场景、100 来源组；旧 full-context rollout 的四个可复现策略逐场景最大误差均为 0；参数版本不变，独立审计通过。每策略 294–303 秒，峰值 allocated/reserved 为 14.115/16.336 GiB。

六个策略的 Correct-vs-Wrong historical 收益仅 `0.001287–0.004311`，所有来源组 bootstrap 95% 区间跨 0，正场景比例 `49.33%–52.33%`；Correct-vs-Empty 也没有任何策略通过预设阈值。故快记忆内容门槛 0/6 通过。三个控制器在 correct-memory historical 条件下相对 fixed 1 分别改善 `0.010286/0.010393/0.012106`，但正确快记忆本身比空快记忆差；完整上下文中三个控制器相对 fixed 1 分别为 `-0.003888/-0.004321/-0.006142`。总体阶段门槛失败，停止扩大旧标签和 history-only 特征消融。

完整结论、表格、资源、复现与下一步见 [`experiments/cbf_ttt/forgetting_only_validation_v1/REPORT.md`](experiments/cbf_ttt/forgetting_only_validation_v1/REPORT.md)。下一步需先构造动态赋值且可由 memory-only 路径读取的语料，确认新世界上 Correct 显著优于 Wrong/Empty，再重启遗忘策略训练。

# 2026-10-11：动态世界记忆写入与联合监督实验（执行前）

## 目标、理解与实现假设

前一轮仅训练遗忘策略的严格复现表明：Correct/Wrong/Empty fast memory 在 fresh attention KV 下不可区分，因此现阶段不能用该 memory 训练有含义的遗忘控制器。本轮先解决写入前提，在从未见过的随机动态事实上训练原生 conv/proj writer，并以开发世界的 Correct-vs-Wrong/Empty 内容门槛决定是否继续。附件中“整体遗忘、last-1/last-2局部遗忘和端到端联合训练”的建议保留为通过内容门槛后的后续；本轮不训练遗忘策略，也不生成3000条反事实标签。

动态世界沿用已审计事件语义生成器：同一问题的两个 twin session 随机赋不同 anchor 值，其他三个事实、自然背景及问题保持一致；train/dev/test 实体、模板与值由 split 隔离。每个世界用FineWeb-Edu或LongCrawl64的唯一长文本作为4096-token背景，只在尾部写入四项session事实。正式规模预注册为64/16/16世界；test只生成语义真值，不由装箱器读取、分词或评分。单种子pilot只能作阶段筛选，不能给最终泛化置信度。

## 修改文件与需求对应

| 文件 | 具体修改 | 对 baseline 的影响 |
|---|---|---|
| `tasks/pack_cbf_dynamic_memory.py` | 从1B混合预训练语料流式选取唯一长文档，FineWeb-Edu/LongCrawl64交替；把任意规模train/dev动态事实装成4096 token；保存来源行号/hash、twin/wrong映射；封存test | 新入口；不改旧packer和baseline数据 |
| `tasks/train_cbf_dynamic_memory.py` | 两个matched-exposure臂：逐问题更新与同context四问题共享一次可微写入；固定0/中点/终点评估；fresh-KV Correct/Wrong/Empty/twin及Full-KV；来源组bootstrap内容门槛 | 新入口；冻结Qwen3-4B backbone，不改baseline训练 |
| `scripts/audit_cbf_dynamic_memory.py` | 独立复核语料hash/矩阵/twin因果、问题暴露与优化顺序、FP32 writer/optimizer、实际参数变化、评测覆盖与汇总、两臂相同初始化 | 只读审计；不新增模型调用 |
| `scripts/run_cbf_dynamic_memory.sh` | 串起测试、64/16/16构建、8/2/2双卡smoke、显存阈值、64/16正式双臂、汇总与审计；失败写状态 | 独立脚本 |
| `tests/test_cbf_dynamic_memory.py` | 检查test封存、自然背景平衡/唯一性、twin因果、wrong跨世界、两臂相同问题暴露和固定内容门槛 | 测试辅助 |
| `DYNAMIC_MEMORY_WRITER_PLAN.md` | GPU执行前冻结数据、预算、阈值、停止条件和命令 | 文档 |
| `README.md` | 增加本轮计划入口 | 文档 |

## 配置、算法闭环与门槛

- 语义seed `20261012`，writer seed `301`，顺序seed `503`；模型 `/home/ctj/models/Qwen3-4B`，自然语料 `/home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl`。
- 每世界2个context、每context 4题；四轮共2048问题暴露。sequential为2048次optimizer step/write；joint-context为512次step/write，每次对四题CE取均值。两臂问题暴露完全一致，但计算量和更新次数不同。
- checkpoint固定为0/1024/2048暴露；训练probe固定前8个训练世界，开发使用全部16世界。模型主干冻结，writer和AdamW状态均为FP32，BF16 forward，lr `1e-7`、clip 1、无weight decay。
- 内容门槛要求开发Full-KV编号正确率≥80%、Correct-memory≥60%；Correct相对Wrong和Empty均高至少20个百分点；两个对照的来源组平均digit-NLL收益均≥0.10且5000次bootstrap 95%下界>0。任一臂固定终点全通过才允许进入遗忘oracle。

## 运行命令

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_dynamic_memory_v1 \
MODEL=/home/ctj/models/Qwen3-4B \
CORPUS=/home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
bash scripts/run_cbf_dynamic_memory.sh
```

## 当前验证、未完成内容与风险

新增3项标准库测试通过；Python语法、shell语法和`git diff --check`通过。本地完整CBF发现集中的无torch项目通过，但8项旧测试因本机没有PyTorch无法导入，这不是代码断言失败；GPU服务器会运行新增测试后再启动。正式smoke/训练/评测尚未执行，结果将追加到本节并归档独立报告。

主要风险是joint-context一次保留四个answer图，显存高于sequential；smoke的reserved显存达到28 GiB即停止正式训练。自然背景与世界模板同时跨split变化，失败可能来自写入容量、表示绑定或跨模板泛化，不能只归因为事实间优化干扰。单seed与16个开发世界仅适合pilot；即使通过也需多seed复现。Full-KV同样使用正在学习的writer，因此它是端到端可读性对照，会随训练变化；Empty才是严格冻结读出对照。

### 首次smoke与正式预检修正

提交`e7f7ff6`的双臂8/2/2 smoke完成训练、三点评测、汇总和独立审计；随后正式臂在checkpoint 0评测、尚未训练时OOM。原因是初版评测器把train probe与16个dev世界所需的约50份七层dense fast memory同时保留，GPU0只余44–61 MiB并因8.27 GiB allocator碎片无法再申请76 MiB。该失败属于评测实现的规模错误，不是模型/方法结果；失败目录保留。

修正后按来源世界流式评测：每次只构造当前两个twin context与相邻wrong donor的两个context，共4份memory；完成该世界8题全部策略后释放并清理CUDA cache。问题、策略、checkpoint、训练预算和预登记门槛均不变。新增单测固定每世界评测闭包恰为4 contexts；安全tripwire统计显式detach，去除无害的requires-grad转float警告。正式实验从已审计的相同数据和smoke继续，不重跑或选择smoke结果。

新增`scripts/resume_cbf_dynamic_memory_formal.sh`作为可复现恢复入口：先要求smoke独立审计通过，把首次OOM的`formal`非破坏性移动为`formal_failed_e7f7ff6`，记录修正提交，重跑4项新测试，再用完全相同的数据、初始化、训练预算、评测矩阵和门槛启动正式双臂。该恢复只改变评测memory的驻留生命周期。

正式恢复运行中joint-context臂完成2048问题暴露，sequential臂在第1095次更新前触发预登记的`||delta||/||W|| > 1`安全阈值。新增`scripts/finalize_cbf_dynamic_memory_safety_stop.py`，不调用模型，独立核验完整joint臂、sequential固定调度的1094步前缀、0/1024评测、两臂相同初始张量、保存参数dtype/变化、失败日志证据和test封存，并明确生成“一个臂完成、一个臂安全停止”的summary，避免把部分臂伪装成完整正式实验。

## 完成结果

服务器20项相关测试通过；8/2/2 smoke双臂完成且独立审计通过。初版formal评测因同时保留约50份dense memory发生OOM，未开始训练；失败工件保留。按世界流式修正后显存约11GiB，joint完成512更新/2048问题暴露，训练allocated/reserved峰值12.10/12.34GiB、平均update 0.4428秒；sequential完成1094步，第1095步在optimizer更新前触发`||delta||/||W||>1`并停止。终态审计通过640 packed rows、hash/twin/调度/checkpoint/FP32/冻结骨干/聚合/安全日志/test封存，状态为`completed_with_sequential_safety_stop`。

Joint固定终点train probe的Correct/Wrong/Empty digit-NLL为2.2502/2.5073/2.7358，但三者编号正确率均0；dev为2.6992/2.7392/2.7586且编号同样全0。dev Correct相对Wrong/Empty的来源组收益为0.0401（95% CI −0.0963–0.1851）和0.0594（−0.0412–0.1612），未达0.10且区间跨0；anchor twin优势−0.00444。Full-KV dev编号93.75%，说明输入可读。Correct answer-NLL相对Empty从4.3280降到2.3471，但随机数字改善很小，表明主要学习共享格式而非session随机值绑定。Sequential中点dev也为0编号，Correct相对Empty digit-NLL更差0.0366。

预登记内容门槛失败，test未评分，不训练遗忘控制器或3000条反事实标签。完整目标/假设、文件清单、数据、指标、资源、审计、命令、兼容性、风险和TODO已归档于`experiments/cbf_ttt/dynamic_memory_writer_v1/REPORT.md`；小型manifest/summary/audit一并入库，大checkpoint与逐行日志保留远程。下一步建议稳定使用joint-context，在更多独立世界上比较普通CE与仅对变化anchor使用的twin-contrast/数字区域损失；内容门槛通过后才进入遗忘oracle。

# 2026-10-11：动态随机值绑定实验（执行前）

上一轮joint writer降低了共享answer-NLL，但dev随机编号为0；本轮区分格式学习与值绑定。新增`tasks/train_cbf_dynamic_binding.py`，以完全相同初始化/顺序/问题暴露比较`joint_ce`和`joint_binding`。后者对四题使用0.75数字、0.125前缀、0.125 EOS的区域损失，并只对twin间真正变化的anchor加入`0.5*softplus(0.2+correct digit NLL-twin digit NLL)`；正确与twin memory均保持可微。两臂问题暴露配平，但binding每步写两次context，明确不作compute matched声明。

正式数据固定seed 20261013、train/dev/test 128/32/32世界、FineWeb-Edu/LongCrawl64自然背景等量交替、4096 token；四轮共4096问题暴露/1024更新，checkpoint 0/2048/4096。train probe固定8世界，dev全32世界，test不装箱/评分。模型、lr1e-7、FP32 writer/AdamW、BF16 forward、clip1和`||delta||/||W||<=1`均与上一轮一致；内容门槛不变。

新增`scripts/audit_cbf_dynamic_binding.py`独立复核数据、调度、逐步目标分解、对比臂启用、checkpoint/optimizer FP32、参数变化、策略矩阵、summary和初始张量一致；`tests/test_cbf_dynamic_binding.py`检查四问题调度和contrast梯度方向；`scripts/run_cbf_dynamic_binding.sh`串起8/2/2双卡smoke、资源门槛、正式双臂和审计。完整预注册见`DYNAMIC_BINDING_PLAN.md`。这些均为独立入口，不改baseline或上一轮工件；尚未产生GPU结果。

### 动态随机值绑定实验：完成与负结果（2026-10-11）

双5090运行于11:07:50–11:59:29 +08:00完成，包含数据构建、8/2/2 smoke、128/32 formal双臂、三点评测、汇总和独立审计。服务器启动脚本聚焦7项测试全部通过；smoke和formal审计均通过。正式joint CE/joint binding各完成1024 update、4096问题暴露，训练峰值allocated 12.0963/14.1159GiB、平均update 0.4495/0.7290秒、最大相对delta 0.7371/0.7252，未触发OOM/NaN/安全停止。冻结骨干hash不变，writer与AdamW状态FP32；test保持语义文本且未tokenize/评分。

固定终点dev中，joint CE的Correct/Wrong/Empty digit-NLL为2.6741/2.6726/2.6962，binding为2.5776/2.6170/2.6962；binding相对CE将Correct digit-NLL降低0.0965、answer-NLL降低0.0620。Binding的Wrong-minus-Correct来源组收益0.0394（5000次bootstrap 95% CI −0.0917–0.1707），Empty-minus-Correct为0.1186（0.0224–0.2010），anchor correct-minus-twin仅0.0090。两臂Correct/Wrong/Empty完整编号正确率全部为0；Correct首token正确约91.41%/88.28%，说明主要学到`code_`格式。Full-KV编号95.70%，输入可读。

预登记内容门槛失败，不能把小幅NLL改善解释为可靠随机值绑定，也未构建遗忘标签或训练遗忘控制器。两臂dev Correct digit-NLL均在2048暴露优于4096终点，记录为非单调诊断，不事后选checkpoint。下一步应修改writer的实体条件检索/绑定机制，并加入四事实全量key-query或memory permutation对比；只有Correct稳定优于within-world permuted/Wrong/twin且生成门槛通过后，才恢复遗忘oracle。

新增`experiments/cbf_ttt/dynamic_binding_v1/{REPORT.md,RUN_STATUS.md,RUN_STATUS.json,ARTIFACTS_SHA256.txt,.gitignore}`及source/data manifest、smoke/formal summary、execution audit、每臂manifest/complete/checkpoint summary。报告记录目标、假设、文件与函数、配置、附件构思对应、完整指标、资源、命令、baseline兼容性、已完成/未完成、风险和TODO。原始逐步/逐题JSONL及checkpoint保留在服务器`/home/ctj/cbf_ttt_dynamic_binding_v1`；传输tar与合并日志仅本地忽略。baseline代码、预训练配置和入口未修改，无新增依赖。

### 2/5/16候选排序定位（2026-10-11，执行前）

用户要求继续下一步实验。新增`CANDIDATE_RANKING_PLAN.md`，固定复用dynamic binding的joint CE/binding在0/2048/4096暴露六个checkpoint，不重新训练或选点；train probe为原前8世界、dev全32世界、test继续封存。每题构造嵌套候选：变化anchor的correct/twin二选一、同twin世界5个唯一值、加入固定Wrong context与同split哈希负值的16选一。候选集合完全由数据和row id在评分前决定。

新增`tasks/evaluate_cbf_candidate_ranking.py`：一次批量teacher-force全16候选，通过`torch.func.functional_call`仅替换七层有效down-projection，输出digit/full-answer NLL、top1/MRR/rank/best-negative margin；按Correct/twin/Wrong/Empty、split、世界和checkpoint聚合，并做5000次来源世界bootstrap。每个checkpoint用真实Correct/Empty样本与既有CBF fresh-KV regions评分桥接，最大误差阈值`2e-5`；源checkpoint评分前后重哈希。上一轮已审计Full-KV 95.70%及其summary hash作为长上下文正对照，不重复昂贵forward。

新增`scripts/audit_cbf_candidate_ranking.py`，不调用模型地重构候选集合、排名、聚合、CI、初始双臂一致性、checkpoint/data/formal-summary hash、矩阵与test封存；`scripts/run_cbf_candidate_ranking.sh`在两张GPU按臂并行，拒绝覆盖输出并写终态；`tests/test_cbf_candidate_ranking.py`覆盖嵌套候选、digit-NLL排序/边际、聚合和固定strong/weak/absent判读。与旧动态测试合计10项本地8通过、2项无torch跳过；语法、shell和diff检查通过。原baseline/runtime、writer权重及数据未修改，无新增依赖。正式运行尚未启动。

首次服务器启动在模型加载前失败：测试10/10通过后，直接执行`python tasks/evaluate_cbf_candidate_ranking.py`使Python仅把`tasks/`加入模块路径，无法导入`tasks.pack_cbf_event_warmup`。无GPU进程、checkpoint读取或模型评分发生，失败目录保留。运行器改为`python -m tasks.evaluate_cbf_candidate_ranking`和`python -m scripts.audit_cbf_candidate_ranking`，仅修正模块入口，不改协议、候选、阈值或评分。

第二次启动在每checkpoint必做的首个真实桥接处停止，函数式effective-weight无cache评分相对既有CBF fresh-KV路径的answer/digit NLL误差约0.0510/0.0483，超过固定`2e-5`；未写任何ranking行。没有放宽桥接阈值。`cbf_ttt/runtime.py::cbf_forward_mlp`改为仅在`cbf_collect=false`读路径允许batch>1，target batch维仍严格匹配；候选delta收集继续要求batch=1。评分器改为直接使用同一CBF cache批量读出。新增torch单测逐值比较batch读与逐样本读，并确认批量collect仍拒绝。默认batch1 baseline语义不变。

### 2/5/16候选排序定位：完成结果（2026-10-11）

正式运行于22:43:19–22:50:28 +08:00完成。服务器聚焦测试11/11通过；两个臂各在0/2048/4096暴露checkpoint完成320行评分，独立审计通过。批量CBF评分与逐样本路径最大误差为0，checkpoint/data hash未变化，test未tokenize或评分。每臂峰值allocated/reserved为10.1881/10.6641GiB。两次预检失败均未产生正式评分：第一次是直接脚本入口导入错误，第二次是近似评分器未通过固定桥接阈值；失败目录保留，阈值未放宽。

固定终点`joint_binding`的dev Correct-memory top-1为pair2 46.88%、world5 20.31%、global16 6.25%；分别接近2/5/16候选的随机水平。global16的twin和Wrong也均为6.25%，Empty为7.42%。Correct相对Wrong的digit-NLL收益0.0401（5000次来源世界bootstrap 95% CI -0.0892–0.1669），相对twin为0.0011（-0.0012–0.0031）；相对Empty的NLL收益0.1183（0.0217–0.1975），但top-1反而低1.17个百分点。2048中点Correct global16为9.38%，相对Wrong有4.30个百分点的小优势，仍低于预登记weak阈值25%，且终点消失；未事后选点。

固定解释为`no_localized_binding`，strong/weak规则均失败。结果说明writer能形成“有记忆”或输出格式效应，但没有可用的实体—随机值排序；此前0生成正确率不能归因于自由生成终止问题。当前不扩大相同训练、不构建遗忘标签、不训练控制器。下一步先在32/8小规模上加入四事实全量实体条件关联检索、within-world permutation负例及Correct/twin/Wrong/Empty门槛，通过候选门槛后再做新seed完整生成，随后才恢复只训练遗忘策略的主线。

新增归档`experiments/cbf_ttt/candidate_ranking_v1/{REPORT.md,RUN_STATUS.md,RUN_STATUS.json,ARTIFACTS_SHA256.txt,.gitignore}`以及总summary、execution audit、双臂manifest/complete/checkpoint summary。完整报告记录目标、假设、实现映射、两次安全失败、指标、命令、兼容性、风险和TODO；原始ranking JSONL与失败日志保留服务器。baseline配置、数据、训练入口、checkpoint和optimizer均未修改；runtime只在关闭collect时允许batch读，原batch1语义与collect限制不变。

# 2026-10-11：四事实关联绑定与world内置换pilot（执行前）

候选排序定位表明当前writer没有隐藏的实体—值排序能力。本轮固定比较`anchor_twin`和`all_fact_permutation`：前者复现单个变化anchor的twin对比，后者在同一context内循环置换四个被查询值，保持背景、实体和值集合不变，并对四个问题分别要求Correct-memory digit-NLL低于permuted-memory。两臂使用相同初始化、32/8 train/dev世界、四轮、1024问题暴露、256更新、两次可微context write、lr 1e-7和0/512/1024固定checkpoint；test继续封存。

新增`tasks/train_cbf_associative_binding.py`，实现token长度不变的四值derangement、matched schedule、四事实contrast、Correct/permuted/twin/Wrong/Empty/Full-KV生成评测和16候选排名。新增`scripts/audit_cbf_associative_binding.py`独立复核数据hash、固定调度、置换多重集与无固定点、contrast数、初始化一致、checkpoint/evaluation hash、聚合、门槛和test封存；`scripts/run_cbf_associative_binding.sh`使用双GPU并行两臂；`tests/test_cbf_associative_binding.py`覆盖调度、置换、梯度方向和固定门槛；完整预注册见`ASSOCIATIVE_BINDING_PLAN.md`。所有入口均为opt-in，不修改baseline训练配置。

固定主要判定为最终`all_fact_permutation`开发集global16：Correct top-1至少25%，相对permuted/Wrong均高至少10个百分点，且两个control-minus-Correct digit-NLL来源世界bootstrap 95%下界均大于0；另要求Full-KV完整code至少80%。中点只作诊断，不选checkpoint。通过仅授权新seed正式生成实验，不直接进入遗忘控制器。GPU正式结果尚未产生。

服务器预检发现训练环境`cbf_ttt_train_py311`没有pytest，但其PyTorch 2.9.1+cu130和Transformers 4.57.3完整；既有`cbf_ttt_verify_20260923`环境包含相同版本及pytest 9.1.1。运行器因此新增`TEST_PYTHON`，仅用验证环境跑测试，模型训练/评测/审计继续使用原训练环境，不安装或升级依赖。

## 完成结果

双5090运行于23:17:49–23:38:50 +08:00完成。服务器测试11/11通过；80个context全部通过4096-token长度不变、四值无固定点置换预检。两臂各完成256更新/1024问题暴露及0/512/1024三点评测，独立审计通过，初始化张量一致，冻结骨干未变，optimizer/writer为FP32，test未评分。无OOM/NaN/安全停止；anchor/all-fact训练峰值allocated为14.1099/16.0142GiB，平均更新0.7457/0.8805秒，最大相对delta为0.5603/0.5597。

固定终点all-fact开发`global16`的Correct/permuted/twin/Wrong/Empty top-1为4.69/6.25/4.69/4.69/6.25%，Correct低于1/16随机水平。permuted-minus-Correct digit-NLL收益0.0017（95% CI -0.0048–0.0082），Wrong收益-0.0076（-0.1799–0.1549），完整code全部为0；Full-KV code为92.19%。预登记门槛失败。

中点all-fact的permutation收益0.0125（0.0079–0.0183），说明硬负例具备局部辨别力，但两者top-1仍同为6.25%，且终点效应消失。训练首/中/末32步的balanced loss为5.7064/2.0600/2.0514，而contrast为0.7989/0.7985/0.8072；辅助关联目标没有下降。终点train probe虽有17.19% Correct top-1，但permuted同为17.19%，仍未学习实体对应关系。因此不扩大同一目标、不评分test、不进入遗忘控制器。下一步将四值listwise实体检索改为主要目标，并做gradient-balanced与retrieval-only消融；若训练probe也不能拟合，再转向显式关联检索头。

归档`experiments/cbf_ttt/associative_binding_v1/{REPORT.md,RUN_STATUS.md,RUN_STATUS.json,ARTIFACTS_SHA256.txt,.gitignore}`、summary、execution audit、两臂manifest/complete/三点summary。大checkpoint、optimizer、steps和逐题JSONL保留服务器。新功能全部为opt-in，baseline配置与入口未修改。

# 2026-10-12：候选对齐listwise实体绑定（执行前）

上一轮辅助permutation contrast保持在约`softplus(0.2)=0.798`，没有被优化。本轮把实体条件数值选择改为主要目标：每个Correct context写入后，对四个查询分别在该context的四个被查询值中计算batched teacher-forced digit-NLL，以负NLL为logit训练四分类cross-entropy。比较`retrieval_only`和`retrieval_plus_lm`，后者仅保留0.25倍balanced LM loss；两臂均计算相同的四个LM read与四个batch-4 retrieval read，使用一次context write，初始化、顺序和计算图结构配平。

新增`tasks/train_cbf_listwise_binding.py`，包含可微batch候选评分、listwise目标、单更新真实GPU smoke、32/8世界四轮正式训练、0/512/1024生成与global16评测，以及训练拟合+开发泛化联合门槛。新增`scripts/audit_cbf_listwise_binding.py`独立复核smoke、四值候选集合、调度、loss聚合、初始化、checkpoint/evaluation hash、门槛与test封存；`scripts/run_cbf_listwise_binding.sh`双卡依次执行测试、并行smoke、资源阈值、并行formal、汇总和审计；`tests/test_cbf_listwise_binding.py`覆盖候选集合、retrieval-only梯度隔离、混合权重和联合门槛。预注册见`LISTWISE_BINDING_PLAN.md`。

正式primary仍为`retrieval_plus_lm`固定终点dev global16门槛，并新增三项优化sanity：末32步四分类训练准确率>50%、retrieval loss低于`log(4)`、终点train-probe Correct相对permuted top-1至少高10个百分点。中点不用于选模。该实验为opt-in，不改baseline；GPU结果尚未产生。

### 无界listwise安全停止与范数约束恢复

提交`7455ea3`的真实smoke通过，两个臂初始retrieval loss均1.3705、四分类准确率25%，峰值18.37GiB。formal完成checkpoint 0后开始训练；retrieval-only在约第41次更新前触发原有`||delta W||/||W||>1`安全阈值，runner终止两臂并写`failed`，GPU释放，test未评分。停止前retrieval-only第40步loss为1.4112且未优于初始值，说明直接listwise梯度会放大fast memory而未改善检索。失败目录`/home/ctj/cbf_ttt_listwise_binding_v1`保留，不作为正式结果。

新增`cbf_ttt.event_writer.bound_memory`及`write_memory(..., max_relative_norm)`可选参数：逐层把fast-weight delta相对冻结down-projection的范数限制到固定上限，缩放系数detach，从而保留delta方向梯度；默认`None`保持所有baseline和旧实验语义。关联评测器新增可选`memory_cap`并在Correct/permuted/twin/Wrong所有writer memory上统一应用。恢复实验固定cap 0.3，两臂训练与评测一致，其他数据、预算、顺序、lr和门槛不变。新增`NORM_LISTWISE_BINDING_PLAN.md`和独立runner；单测检查范数上限与非零有限梯度。正式恢复尚未执行。

### 范数约束listwise完成结果

双5090正式恢复实验完成：retrieval-only与retrieval-plus-LM各256更新/1024问题暴露，0/512/1024三点评测齐全，服务器14项测试和独立审计通过。冻结骨干hash不变，writer/AdamW状态FP32，test未评分。训练allocated峰值18.8164/18.8200GiB，reserved峰值19.1055/19.1152GiB，平均更新0.6932/0.6807秒，最大测得相对delta 0.30109/0.30087；未触发阈值1的安全停止。

四分类retrieval-only首/末32步loss 1.5732→1.2629、accuracy 26.56%→46.09%；hybrid为1.5010→1.3198、23.44%→37.50%，均低于`log(4)`随机loss但未达>50%门槛。固定终点hybrid train-probe Correct/Permuted global16 top-1为18.75/15.62%，dev为7.81/7.81%；dev permuted-minus-Correct digit-NLL收益0.0200，95% CI 0.0013–0.0411，但top-1无差。Wrong收益-0.0442且CI跨0，Empty收益-0.5377且CI严格为负，完整code仍0；Full-KV dev code 92.19%。retrieval-only训练Correct/Permuted终点均20.31%，dev 4.69/3.12%，同时发生严重LM退化。

预登记stage gate失败。结论限定为：cap解决数值爆炸，listwise目标可以下降，但没有建立可靠的current-memory实体—值绑定。新增归档`experiments/cbf_ttt/norm_listwise_binding_v1/{REPORT.md,RUN_STATUS.md,RUN_STATUS.json,summary.json,execution_audit.json,git_commit.txt,.gitignore}`；原始step/evaluation/checkpoint/optimizer保留服务器`/home/ctj/cbf_ttt_norm_listwise_binding_v1`。baseline默认cap为None，兼容性不变。

### 反事实重赋值episodic binding（执行前）

结果审查发现新的混淆：固定训练query在所有轮次长期对应同一答案，writer慢参数可记住query→value而绕过当前session memory。新增`EPISODIC_BINDING_PLAN.md`，以相同32/8世界、4轮、1024问题暴露、256更新和cap 0.3比较`fixed_assignment`与`episodic_assignment`。后者在每个context中定位四个等长value token span，构造四个长度不变的循环赋值episode；每个query在四轮中均衡对应四个值，背景、实体、query和值集合不变。

新增`tasks/train_cbf_episodic_binding.py`：`rotation_episodes`构造/验证四个赋值，`episode_schedule`保证episodic每context覆盖0/1/2/3且fixed始终0，训练复用四候选listwise+0.25 balanced LM与原生可微writer。评测固定assignment shift 0/2，Correct memory与同实体下一赋值Counterfactual、跨世界Wrong、Empty比较，同时报告context4/global16排序和完整生成，Full-KV作为可读性对照。最终episodic门槛要求dev Full-KV code≥80%、Correct context4 top-1≥60%、相对Counterfactual至少+25点且NLL gain bootstrap 95%下界>0、Correct global16 top-1≥25%。

新增`scripts/run_cbf_episodic_binding.sh`双卡测试/smoke/formal/汇总/审计流程，`scripts/audit_cbf_episodic_binding.py`独立重建token替换、schedule/target、训练聚合、初始张量、checkpoint/evaluation hash、门槛及test封存；`tests/test_cbf_episodic_binding.py`覆盖循环赋值、均衡调度和因果门槛。与既有相关测试本地18项中13通过、5项因本地无torch跳过；Python及shell语法通过。所有新行为为opt-in，不改baseline配置或旧工件，无新增依赖。风险包括仅四个循环排列、8个dev来源组和单seed；即使通过仍需新排列族与新seed确认。

首次服务器启动18/18测试与双卡smoke通过，峰值reserved约18.62GiB；正式训练前的checkpoint-0流式评测开始后主动停止。原因是代码审查发现artifact审计将BF16 cap 0.3按0.3001检查，而上一正式实验已观测到舍入后0.30109；若不修正会在完整训练后误报失败。该启动尚无optimizer step，目录保留为`/home/ctj/cbf_ttt_episodic_binding_v1_preflight_cf42a6e`。runner/audit固定改为0.305数值容差，实际cap、训练安全阈值1.0、数据、预算、loss和效果门槛均不改；计划补记该容差后从新目录重启。

### 反事实重赋值episodic binding：完成结果

正式重启于commit `1a152a1`完成；18/18服务器测试、双臂smoke、各256更新/1024暴露、0/1024两点评测和独立审计全部通过。固定/episodic首末32步retrieval loss为1.5526→1.3163、1.5387→1.4316，准确率23.44→36.72%、25.00→26.56%。两臂相同初始化/数据/曝光/计算路径下，固定关系明显更易拟合，支持固定query→value捷径会夸大writer学习；但这不是参数内部机制的直接证明。

episodic终点dev Correct/Counterfactual/Wrong/Empty context4 top-1为25.00/22.66/22.66/26.56%，global16为7.03/4.69/10.16/5.47%。Counterfactual-minus-Correct digit-NLL 0.0073（95% CI -0.0033–0.0188），Wrong 0.0601（-0.1263–0.2585），Empty 0.0943（-0.0057–0.1933）；完整code仍0，Full-KV code95.31%。train probe Correct/Counterfactual也仅25.00/24.22%，表明并非只有开发泛化失败。固定臂dev有极小正NLL区间但top-1方向相反，验证联合门槛必要。预登记stage gate失败，test未评分，两卡释放。

归档`experiments/cbf_ttt/episodic_binding_v1/{REPORT.md,RUN_STATUS.md,RUN_STATUS.json,summary.json,execution_audit.json,git_commit.txt,.gitignore}`；大工件保留服务器。下一步新增`PAIRED_BINDING_PLAN.md`：在同一optimizer step放入相反赋值，消除跨step梯度漂移。

新增`tasks/train_cbf_paired_binding.py`：固定(0,2)/(1,3)两对，每context覆盖四赋值一次；`paired_listwise`同时优化两赋值listwise CE，`paired_swap`在同一批候选NLL上增加对称`0.5*softplus(0.2+own-cross)`，无需额外前向。每臂128更新、每步2次4096写入/8问题、总1024暴露，去除LM项以隔离关联检索；两臂计算完全匹配。复用episodic的Correct/Counterfactual/Wrong/Empty/Full-KV与固定门槛。

新增`scripts/run_cbf_paired_binding.sh`、`scripts/audit_cbf_paired_binding.py`和`tests/test_cbf_paired_binding.py`，覆盖四赋值配对、swap梯度方向、schedule/target/聚合/checkpoint/初始张量/test封存。真实双写+八候选图先做双卡smoke，reserved≥28GiB或cap舍入>0.305立即停止；通过才正式运行。所有入口opt-in，无baseline修改和新依赖。

### 同一步反事实配对：完成结果

commit `3b758a5`的双卡实验完成，22项服务器测试与独立审计通过；两臂各128更新/1024问题暴露。paired-listwise首末16步retrieval 1.5900→1.3955、accuracy25.00→24.22%；paired-swap 1.5887→1.3920、25.00→28.13%。关键swap loss为0.79807→0.79864，没有低于`softplus(0.2)`无区分值。训练峰值allocated/reserved 24.385/24.752GiB，平均更新约0.97–0.99秒，cap舍入最大0.30109。

primary swap终点dev Correct/Counterfactual/Wrong/Empty context4 top-1为21.09/21.88/27.34/26.56%，global16为6.25/6.25/4.69/5.47%。Counterfactual-minus-Correct NLL为-0.00539，95% CI -0.00842–-0.00204，显著方向错误；正确memory完整code仍0。去掉LM后Correct digit-NLL升至15.8077，Empty保持2.7108，说明相对候选目标伴随严重绝对语言退化。门槛失败、test未评分、两卡释放。

归档`experiments/cbf_ttt/paired_binding_v1`报告/summary/audit/status/hash来源；大工件保留服务器。结论是同一步配对也未使现有dense delta+冻结LM head产生可用绑定，停止继续调listwise/permutation/swap权重。

### 显式候选检索头pilot（执行前）

新增`RETRIEVAL_HEAD_PLAN.md`与`tasks/train_cbf_retrieval_head.py`。检索头把fresh-KV查询末端hidden和候选数字token的冻结平均embedding分别投影到256维，L2归一化后按temperature0.1余弦评分。`head_only`冻结零初始化writer只训练head，作为平衡赋值下query-prior捷径控制；`joint_head_writer`同时训练相同head和七层writer。两臂head seed1301、writer seed301、四个平衡pair cycle、512更新/4096暴露；head lr1e-4、writer lr1e-7，分别clip1，cap0.3，骨干/embedding冻结。

评测固定shift0/2的Correct/同实体Counterfactual/Wrong/Empty，报告context4/global16并增加Full-KV head排序。门槛要求dev Correct context4≥60%、相对Counterfactual≥25点且NLL bootstrap下界>0、Correct global16≥25%、Full-KV global16≥50%。新增`scripts/run_cbf_retrieval_head.sh`、`scripts/audit_cbf_retrieval_head.py`和`tests/test_cbf_retrieval_head.py`；audit重建schedule、检查head-only writer零变化/joint writer全变化、相同head+writer初始化、FP32 optimizer、evaluation聚合和test封存。新head只在opt-in task中创建，不改baseline logits或模型类。

### 显式候选检索头pilot：完成结果

commit `3d3457f`的实验完成：25项服务器聚焦测试、双臂smoke、每臂512次更新/4096问题暴露、两点评测和独立工件审计全部通过。骨干hash不变，test未评分，显存已释放。head-only/joint首末32步loss为1.3949→1.3868、1.3979→1.3847，accuracy为25.78→22.66%、25.00→26.56%，都没有稳定优于四分类随机水平。

primary joint的dev Correct/Counterfactual/Wrong/Empty context4 top-1为25.00/25.00/24.22/25.00%，global16为13.28/13.28/7.81/3.12%。Counterfactual-minus-Correct NLL仅`0.00001196`，来源组bootstrap 95% CI `[-0.00000062, 0.00002883]`，top-1差为0；global16相对Empty的升高不能归因于正确session绑定。终点Full-KV head dev global16仅3.91%，因该head没有直接在Full-KV上训练，不能据此单独判定writer失败。五项门槛全失败。

归档`experiments/cbf_ttt/retrieval_head_v1`的完整报告、状态、summary、audit、manifest、complete和两点评测摘要；大checkpoint、optimizer、steps、逐题JSONL和log保留在服务器。下一步用完全上下文KV直接训练同一head作为正控制：通过则锁定dense writer为瓶颈，失败则改用candidate-conditioned自回归表征或显式结构化key-value memory。

### Full-KV直训检索头正控制（执行前）

新增`FULLKV_HEAD_CONTROL_PLAN.md`、`tasks/train_cbf_fullkv_head_control.py`、`scripts/run_cbf_fullkv_head_control.sh`、`scripts/audit_cbf_fullkv_head_control.py`和`tests/test_cbf_fullkv_head_control.py`。两臂`same_lr/high_lr`分别使用1e-4/1e-3 head学习率，共享seed1301初始化、32/8 train/dev世界、四轮反事实均衡调度、512次更新和4096问题暴露。writer/骨干/embedding全冻结；先对完整4096-token context+query提取固定末位hidden特征，再只优化与上软pilot完全相同的256维余弦head。评测shift0/2的context4/global16，primary high-lr门槛要求末32步训练accuracy≥80%、dev context4≥75%、dev global16≥50%且来源组bootstrap 95%下界高于6.25%随机水平。

审计重建schedule、逐行聚合、gate、feature/step hash，并核对两臂head初始张量一致、参数确实更新和test封存。本地7项相关测试6通过、1项因本地无torch跳过；Python语法、shell语法和diff检查通过。所有行为opt-in，未改baseline模型或默认训练。

首次服务器启动11/11测试和双卡smoke通过，正式双臂各提取完1152条Full-KV特征，但在checkpoint-0首个排序前因`_scores`把候选list误作单个dict key而停止；尚无optimizer step或效果结果，GPU已释放。修正为按候选顺序stack value tensor，并增加直接覆盖列表候选的单测；不改数据、门槛、学习率或预算。失败目录保留为preflight来源。

修正后commit `c2e37ae`的正式实验完成，12/12服务器测试、双臂smoke、各512更新/4096暴露、固定评测和独立审计通过。same/high-lr末32步训练accuracy为48.83/38.67%；dev context4为28.13/24.22%，global16为4.69/7.81%。high-lr dev global16来源组bootstrap 95% CI `[2.34%,13.28%]`，same-lr为`[0.78%,9.38%]`；两臂四项门槛全失败。

两臂1152条特征的SHA-256一致，提取271–279秒，峰值allocated/reserved 8.597/8.865GiB；head更新仅约1.3ms。结论是单个query末位hidden对齐五个数字embedding均值的表征无法解码未见多token code，提高学习率不能修复。停止调该pooled cosine head；下一步使用可读的结构化session key-value memory，独立训练retain/last1/last2/global遗忘控制器，再考虑向TTT可微快记忆蒸馏。完整归档位于`experiments/cbf_ttt/fullkv_head_control_v1`，test未评分，baseline不受影响。

### 基于物理不变量的结构化遗忘控制器（执行前）

新增`STRUCTURED_FORGETTING_PLAN.md`、`tasks/train_cbf_structured_forgetting.py`、`scripts/run_cbf_structured_forgetting.sh`、`scripts/audit_cbf_structured_forgetting.py`和`tests/test_cbf_structured_forgetting.py`。固定更新门`g=1`，在可读的batch式session key-value memory上只学习`retain/last1/last2/global`四种遗忘动作。使用已有event-curriculum生成器扩展64/16/16个实体互斥且模板互斥的train/dev/test世界，test只在dev gate通过后一次性解封。

每个决策点对四动作进行真实当前查询反事实回放，以最小遗忘的最大正确率动作为监督。线性softmax控制器只读memory size、已知batch引用及其recency、新写入前行数、block位置、fact数和active-key数；不读family/split/ID/答案/未来block。history-blind消融清零last1/last2/deeper特征。比较学习策略、消融、四个固定动作、确定随机、物理规则和反事实oracle，报告分family/role查询正确率、动作正确率、保留batch数和来源组bootstrap。

dev门槛固定为学习动作正确率≥95%、查询正确率≥98%、相对最优固定动作提升≥15点且成对来源组bootstrap 95%下界>10点。本地5项新单测全通过，Python/shell语法和diff检查通过。实验仅验证结构化memory上的遗忘策略，不预先声称TTT fast-weight端到端有效。

首次服务器预检10/10测试通过，64/16/16世界、2304 episodes/13824 queries语料构建完成。控制器训练后在评测故意错误的固定策略时停止：错误策略可能已删掉后续撤销事件引用的batch，旧parser依赖当前memory判定新batch，把缺失旧引用误判为第二个新batch。无dev汇总或test评分。修正为按可见block结构取最后一个batch引用为紧邻facts的新写入，保留更早引用为撤销目标；physics在错误轨迹上找不到目标时安全retain。增加策略状态分歧回归测试，不改语料、特征、标签、预算或门槛。

修正后commit `4596028`的正式实验完成：11项服务器测试、64/16/16互斥世界、3072训练决策样本、dev gate、按gate一次性test解封和独立重算审计全通过。学习控制器dev查询/动作正确率99.91/99.74%，最优固定last2查询75.00%；提升24.91点，成对来源组bootstrap 95% CI `[24.74,25.00]`，四项预注册门槛全通过。

解封后test上学习控制器查询/动作正确率99.74/99.22%，最优固定仍为75.00%；提升24.74点，95% CI `[24.48,25.00]`。history-blind为90.28%，恢复batch recency后提升9.46点，95% CI `[8.94,9.98]`；revoke-last1/last2分支从72.92/89.24%升至100/100%。这是修订后遗忘主线的首个正向预验证：在可读结构化session memory上，状态依赖的局部/全局遗忘显著优于任一固定策略。

限制：3072训练决策中1114个有反事实并列最优，按最小遗忘顺序解除，因此主效果以query accuracy和成对基线差为准；当前语料有显式结构语法，且结果不等于原姆TTT dense fast-weight端到端有效。完整报告和小工件归档于`experiments/cbf_ttt/structured_forgetting_v1`；下一步是加入噪声/重复/更长延迟撤销、多语料seed复现，再向可读TTT-compatible memory蒸馏。

### 低秩fast-weight连续+局部遗忘（执行前）

新增`LOWRANK_FORGETTING_PLAN.md`、`tasks/train_cbf_lowrank_forgetting.py`、`scripts/run_cbf_lowrank_forgetting.sh`、`scripts/audit_cbf_lowrank_forgetting.py`和`tests/test_cbf_lowrank_forgetting.py`。每个可见实体-值写入构造实体key与value向量的外积，batch delta为这些外积之和，session memory为带权batch delta的加和。该受控代理保留`M_t=(1-alpha)M_{t-1}+DeltaW_t`的加性fast-weight形式，同时允许审计每个分量。

反事实动作为`retain(alpha=0)`、`half(alpha=0.5)`、`local(删除被引用batch)`和`global(alpha=1)`，所有分支写入完全相同的当前delta；以当前三类查询平均NLL选标签，并列按实测记忆改变最小解除。控制器输入包括memory/delta范数、对齐度、引用分量范数/时间距离、词面结构、实体重叠/重复和capacity统计，不读答案、未来、family/split或ID。

新语料固定128/32/32 train/dev/test世界，使用三个独立低秩投影seed 1701/1702/1703。比较完整控制器、reference-blind、仅`0/0.5/1`的global controller、四个固定策略、随机、物理规则和oracle。dev gate要求oracle≥90%、learned平均≥88%且每seed≥85%、距oracle≤3点、相对最强固定/global-only控制至少+8点且来源组bootstrap 95%下界>5点。只有聚合dev gate通过才一次性读test。本地非torches单测通过，torch真实外积/反事实测试因环境无torch跳过；语法、shell和diff检查通过。

服务器13/13测试、128/32/32新语料、三投影seed、dev gate、按gate一次性test和独立全量重算审计通过。learned的dev/test top-1为96.82/96.68%，NLL为0.3842/0.3895；oracle为96.93/96.85%。最强固定local为82.07/82.13%；learned在dev提升14.75点，95% CI `[14.06,15.35]`，test提升14.55点，95% CI `[13.98,15.12]`。相对只允许global alpha 0/0.5/1的learned controller，test提升18.32点，95% CI `[17.80,18.85]`。三个test投影seed的learned为97.85/95.75/96.74%，稳定贴近oracle。CPU全流程139.3秒。

但每seed 6144个训练决策中，alpha=0.5仅出现14–17次（0.23–0.28%），local约3060次，alpha=0约1530次，alpha=1固定1536次。因此结果强力支持选择性local/global遗忘，但不支持“中间连续alpha普遍有用”。归档`experiments/cbf_ttt/lowrank_forgetting_v1`；下一实验专门构造旧事实保留与新更新接受的连续权衡场景，用更密alpha网格验证内点最优和连续回归。

### 连续alpha权衡场景分布预检（执行前）

新增`CONTINUOUS_FORGETTING_PLAN.md`、`tasks/train_cbf_continuous_forgetting.py`和`tests/test_cbf_continuous_forgetting.py`。每个场景先写6–12个低秩旧事实，当前delta以0.55–1.55的强度覆盖其中变动数量的key、重复部分事实并加0–4个新key；未被更新的历史事实和新值同时纳入固定查询集。对`alpha={0,0.25,0.5,0.75,1}`在同一`(1-alpha)M+Delta`状态上比较查询NLL，并列选较小alpha。

控制特征只包括memory/delta范数和对齐、conflict/duplicate/new/retained比例、fact数、当前写入强度和key相关性，不包含值ID、答案或任何未来损失。正式实验前先对40组×20场景×3个memory seed做label-distribution probe；只有内点总比例≥30%且至少两个不同内点各占≥5%才冻结正式协议。本地确定性/互斥性测试通过，torch网格最优测试因本地无torch跳过；语法和diff检查通过。

### 连续alpha正式控制器实验（执行前）

在尚未读取服务器PyTorch探针结果时冻结正式协议：128/32/32个来源组、每组20场景、三个独立fast-weight投影seed；test场景只在聚合dev门通过后生成一次。主控制器为两隐层sigmoid回归器，以五点反事实最优alpha作MSE监督。对照包括五分类网格控制器、去除conflict/duplicate/new/retained/count/scale结构特征的盲回归器、只允许alpha 0/1的端点控制器、固定物理启发式、train均值、五个固定alpha及grid oracle。

新增`formal`子命令、`scripts/run_cbf_continuous_forgetting.sh`和`scripts/audit_cbf_continuous_forgetting.py`。正式评测对每个策略回放真实alpha并重算NLL/top-1，按来源组做bootstrap。dev门固定为：alpha MAE≤0.15；距grid oracle的NLL≤0.03；相对最佳固定alpha的NLL收益≥0.05且95% CI下界>0.02；相对端点控制器收益≥0.02且下界>0.005；三个投影seed均优于最佳固定alpha。审计会重建训练标签、dev记录、控制器消融与条件test结果，并校验probe/checkpoint/raw记录hash。

本机无PyTorch，因此新增的两个真实张量测试跳过；其余3项连续实验测试、Python语法、shell语法和diff检查通过。另用不入库的Python标准库随机向量实现做机制预检，内点比例57.42%，三个内点均超过5%；该数值只用于排除明显场景设计错误，不能替代服务器PyTorch正式probe。baseline入口、配置和既有实验工件均未修改。

### 连续alpha新投影与分布外压力测试（执行前）

正式结果通过后，在生成任何压力场景前新增`CONTINUOUS_STRESS_PLAN.md`、`tasks/evaluate_cbf_continuous_stress.py`、`scripts/run_cbf_continuous_stress.sh`、独立审计与测试。冻结上一轮checkpoint且不再训练，换用未见投影seed 2901/2902/2903，并固定`long_memory`、`weak_update`、`strong_update`、`extreme_overlap`四类训练支持外场景，每类32组×20场景。

每类主对照是在该压力集结果上事后选择的最优固定alpha，因此比部署时固定常数更强。通过标准冻结为：四类中连续控制器相对事后最优固定alpha和端点控制器的来源组bootstrap 95%下界均大于0；宏平均alpha MAE≤0.20；三个新投影seed均优于宏平均最佳固定alpha。审计重构全部场景、策略回放、聚合与CI并校验原checkpoint/summary hash。该实验不修改baseline或已有checkpoint。

### 域增强连续遗忘控制器（执行前）

压力测试中连续控制器在四类上均显著优于事后最优固定alpha，但`strong_update`的端点控制器NLL低0.00117，预注册总门失败且不事后修改。新增`CONTINUOUS_ROBUST_PLAN.md`、`tasks/train_cbf_continuous_robust.py`、运行器、独立审计与复合偏移测试。保持控制器结构不变，以base/long/weak/strong/extreme五域等量训练；dev使用未见投影2901/2902/2903并同时比较原冻结控制器。

dev全门通过后才生成最终评测：换用投影3901/3902/3903、五个新组内测试域和未在训练中出现的long+weak、long+strong、extreme+scale三种复合偏移。门槛包括相对固定、端点和原控制器的来源组CI、强写入端点非劣、各域/各seed一致性。所有大工件保留远程，新入口不改baseline。

### 域增强连续遗忘控制器：完成结果

服务器9项测试通过，五域训练共19,200个投影后反事实样本。七项dev门全部通过：相对原冻结控制器NLL改善0.01023，来源组bootstrap 95% CI `[0.00710,0.01388]`；strong-update相对端点仅差0.000539，满足0.002非劣界。随后脚本首次生成最终test，换用投影3901/3902/3903及八个域。

最终宏平均continuous NLL/accuracy为0.7591/87.78%，事后最佳固定alpha为0.9139/76.80%；NLL收益0.1548，95% CI `[0.1476,0.1620]`，accuracy提升10.98点。端点控制为0.8165/87.07%；continuous NLL收益0.05737，CI `[0.05048,0.06445]`。long+weak、long+strong、extreme+scale三个未见组合相对固定和端点均显著为正。孤立strong-update仍由端点alpha0微弱领先0.00067，符合强写入无需中间遗忘的机制边界。

独立审计重建训练标签、5个dev域、8个test域、184,320条test策略记录、全部bootstrap和hash并通过。归档`experiments/cbf_ttt/continuous_robust_v2`，新增`PAPER_EVIDENCE_SUMMARY.md`明确论文可支持的受控机制结论与不能声称的Qwen端到端效果。baseline、预训练checkpoint及既有实验工件未修改。

### 原生 TTT 快权重累积机制实验（执行前）

新增`NATIVE_ACCUMULATION_PLAN.md`、`tasks/cbf_native_accumulation.py`、`scripts/run_cbf_native_accumulation.sh`、`scripts/audit_cbf_native_accumulation.py`和`tests/test_cbf_native_accumulation.py`。实验固定最终Qwen3-4B 1B-token checkpoint、既有12K/binding场景和完整KV，只改变三个4096-token边界的快权重规则：禁写、仅首块写入、仅末块写入、每块写入前清除旧快权重、原生三次累积。

主指标为正确答案NLL、正确答案相对`<|im_end|>`的logit margin及全词表首token命中；八候选和QA只作辅助。所有比较按8个来源组配对bootstrap。预登记的累积损伤门要求受限写入臂相对累积臂同时改善NLL、answer-vs-EOS margin和首token命中；另与禁写臂比较，只有NLL配对CI下界大于0才声称写入具有内容收益。清除只替换7层fast weight并保留KV；查询使用prefix深拷贝，审计父cache、参数、数据及checkpoint hash。无baseline代码路径或训练配置变更。

### 原生 TTT 快权重累积机制实验：完成结果

数据采集提交`546707d`在hku-gpu2两张RTX5090完成320 rollout/1280查询，9项服务器测试通过，总墙钟1453秒，最高allocated/reserved为11.872/13.889GiB。`cumulative`的binding答案NLL为6.8965，128/128查询首token均为`<|im_end|>`；`reset_each`/`last_only`的NLL为1.4877、EOS率4.69%、正确首token53.13%。相对累积，NLL改善5.4088，来源组bootstrap 95% CI `[3.4846,7.4589]`；margin提升8.7588，CI `[6.8975,10.9180]`，预登记累积损伤门通过。

完全禁写的NLL为1.4243，仍显著优于`reset_each` 0.06339，95% CI `[0.02448,0.10206]`；因此受限写入内容收益门失败。`reset_each`与`last_only`全部分数逐值相同，证明块尾更新在下一块前清除后，早期delta没有被使用。增强后的独立审计提交`547f2c5`将本轮cumulative/zero与上一轮native/plain逐值桥接，所有256查询最大误差0。聚合工件和完整报告归档于`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/native_accumulation`；原始JSONL、语料和模型留服务器。

### 单次原生 Fast Weight 内容因果实验（执行前）

新增`FASTWEIGHT_CONTENT_PLAN.md`、`tasks/cbf_fastweight_content.py`、`scripts/run_cbf_fastweight_content.sh`、`scripts/audit_cbf_fastweight_content.py`和`tests/test_cbf_fastweight_content.py`。复用从未评分的16个confirm来源组，组8–15只做`short/single/qa`完整KV读取门，组16–23在Q通过后才评分。V固定第一块禁写、事实所在第二块单次写入；每个查询保持self KV不变，只替换为Correct、Wrong-twin、Zero或三个逐层等范数Rademacher随机fast weight。

内容绑定与内容效用分开判定：前者要求Correct相对Wrong/Random同时改善答案NLL并提升own-vs-twin margin，后者要求Correct相对Zero改善NLL且候选准确率不降；全部差值按8个来源组bootstrap。构造器验证Q/V来源互斥、twin只差一个事实token和分域颜色平衡；collector验证单次写入、父cache/骨干不变和随机范数；独立审计从原始parquet重建场景并重算Q/V结果。baseline入口和配置不变。

v1实际Q的两个域八选一均100%，8/8组双twin正确，但全词表首token仅31.25%，低于预登记50%门槛；流程以`stopped_by_content_Q`结束，V文件不存在，独立重建审计通过。事后只解码已存argmax，6条为` The`、5条为首字母大写颜色、5条为正确小写颜色，提示开放词表格式是门槛失败的重要组成。

新增`FASTWEIGHT_CONTENT_V2_PLAN.md`、`tasks/cbf_fastweight_content_v2.py`及对应runner/audit/test。v2不降低门槛：在已观察dev组0–7比较三个预设小写单词模板，按最差域首token固定选择；只将唯一模板送到组8–15 Q，Q通过后才首次评分组16–23 V。V的干预、内容绑定/效用阈值与v1相同。

v2提交`a1984b2`在两张RTX5090完成。dev选择options，独立Q的候选与首token均100%；随后首次解封V。Correct/Wrong/Zero/Random的答案NLL为0.12824/0.13041/0.10799/0.11032。Correct相对Wrong收益0.00216，95% CI `[-0.00064,0.00493]`；特异性margin提升0.01953，CI `[-0.00781,0.04297]`。Zero与Random均显著优于Correct，内容绑定和内容效用门失败。独立审计从parquet重建48场景并重算三阶段和bootstrap通过；墙钟164秒，峰值allocated/reserved 10.867/11.746GiB。完整归档位于`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/fastweight_content_v2`。

### 最终checkpoint writer内容绑定恢复（执行前）

新增`FINAL_WRITER_BINDING_PLAN.md`、`tasks/train_final_writer_binding.py`、`scripts/run_final_writer_binding.sh`、`scripts/audit_final_writer_binding.py`及`tests/test_final_writer_binding.py`。与此前从Qwen基座新初始化writer的实验不同，本轮加载1B-token最终checkpoint中的既有`ttt_conv/ttt_proj`，冻结全部非writer参数。已审计v2场景只替换唯一颜色token，为每个来源组生成同一查询对应8种反事实值；组0–7用于嵌套调参与最终训练，组8–15为Q，组16–23保持封存至Q通过。

`cbf_ttt.event_writer.write_memory_after_prefix`新增可选的两块写入路径：第一块只构造冻结KV，第二个完整块生成可微delta，查询使用fresh KV。训练采用八选一CE、全词表CE、同一步对称counterfactual swap和Correct-vs-Zero margin，逐层delta cap固定为0.3。调参只比较预登记的`3e-7/1e-6`，随后从原最终checkpoint重新初始化、用组选定学习率训练；Q门要求Correct同时优于Wrong/Zero/等范数Random且source-group bootstrap下界大于0，通过后才允许一次性V。所有入口均为opt-in，baseline默认配置和原生公式未修改。

正式实验在提交`672f9a9`完成。双学习率开发门均失败，冻结规则选择`3e-7`后从最终checkpoint重新训练256步。独立Q的Correct/Wrong/Zero/Random准确率为10.94%/12.50%/12.50%/14.06%；Wrong-minus-Correct NLL为-0.01074，95% CI `[-0.02051,-0.00293]`，而Zero/Random-minus-Correct为正。该组合说明writer产生通用memory效应但没有当前值绑定。Q失败后V保持封存；独立审计全部通过。总墙钟14分09秒，最终训练峰值allocated/reserved 12.558/13.643GiB。完整报告与紧凑工件位于`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/final_writer_binding_v1`。

### Qwen低秩key-value memory桥接（执行前）

新增`QWEN_LOWRANK_BRIDGE_PLAN.md`、`tasks/cbf_qwen_lowrank_bridge.py`及对应runner/audit/test。表示将实体字符串的Qwen输入embedding均值作为归一化key，将观测值token的Qwen LM-head方向作为归一化value，以rank-one外积形成session-local fast memory；query读出的残差加到冻结Qwen的fresh-KV末位hidden后仍由原LM head评分。dev仅在`[0,1,2,4,8,16]`选择残差scale，Q/V来源隔离并沿用Correct/Wrong/Zero/等范数Random因果门，Q失败不读取V。该桥接明确标记为可读memory substrate，不声称原生dense writer已经学到此表示；目标是通过后接回多块累积和遗忘策略。

提交`8217ee3`的正式桥接在37秒内完成，dev按冻结规则选择scale16，独立Q和解封V均通过。Q/V Correct候选及全词表首token均100%，Wrong均0%，Zero均12.5%；V Correct相对Wrong/Zero/Random的答案NLL收益为11.7500/2.3364/2.3784，bootstrap下界均大于0，两域均100%，EOS为0。独立审计重算全部阶段通过。该结果只证明Qwen-grounded低秩单组件memory可读；后续将冻结表示并验证多组件累积下的学习局部/全局遗忘，同时报告scale4/8/16敏感性。

### Qwen表征多组件遗忘（执行前）

新增`QWEN_LOWRANK_FORGETTING_PLAN.md`、`tasks/train_cbf_qwen_lowrank_forgetting.py`及对应runner/audit/test。使用新seed构建128/32/32世界event curriculum；实体key来自最终Qwen输入embedding，value来自最终Qwen LM-head token方向，分别用train+dev统计中心化，再以三个固定Gaussian投影映射到128维。测试命名空间只在dev聚合门通过后读取，并严格复用train+dev中心和投影。每个assignment写入value-key外积，比较四个固定动作、随机、history-blind、global-only learned、full learned、物理规则和oracle；门槛沿用三seed低秩正式实验。模型checkpoint冻结，只训练小型遗忘控制器。

### Qwen表征多组件遗忘：完成结果与下一轮计划

提交`3b1d6e6`在hku-gpu2完成正式运行。生成4,608个episode和27,648个query，三个投影seed的控制器训练与开发评测耗时102.67秒，总墙钟117秒。full learned准确率/NLL为27.73%/4.0998，最强固定策略global reset为32.20%/4.7087；尽管learned NLL更低，准确率差值为-4.48点，来源组95% CI `[-6.35,-2.60]`。greedy oracle准确率只有28.49%，可读性、learned准确率、全seed、控制增益和置信区间五项门槛失败；test保持封存。独立审计验证数据/表示/raw记录hash、聚合、门控和test lock全部通过。runner未记录GPU峰值，因此不报告该数值。

新增归档`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/qwen_lowrank_forgetting_v1`，包含报告、summary、training、manifest、审计和运行状态；原始JSONL和checkpoint保留服务器。新增`QWEN_LOWRANK_GEOMETRY_PLAN.md`，下一轮先在完美语义状态下比较`rp128`、`rp128_whiten`、`rp256_whiten`与随机码上限，量化1–32组件cross-talk。只有mean≥90%、最差seed/事件族/16组件≥85%的Qwen表示才能进入v2遗忘控制器；否则停止并预注册contextual hidden-state表示实验。baseline入口和配置均未修改。

### Qwen低秩表示几何：完成结果

提交`34a2e70`的正式G1/G2诊断在单张RTX5090完成，墙钟54秒、核心诊断49.27秒，峰值allocated/reserved 7.633/7.643GiB。`rp128`、128维白化、256维白化的完美语义状态dev准确率分别为32.65%、31.62%、32.20%，均未通过90%门；随机码控制为100%。白化仍将通用16组件load sweep从69.16%提高到87.03%，32组件从36.36%提高到约67%，说明其改善全局容量但没有解决同一世界共享前缀key的局部冲突。test未读取，控制器未训练，独立审计与`rp128`逐bit旧结果桥均通过。

新增结果归档`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/qwen_lowrank_geometry_v1`。事后仅分解已存dev行：256维白化的known/unknown为37.93%/3.56%，current/historical/anchor为41.67%/17.38%/37.57%，标记为探索性定位。新增`QWEN_CONTEXTUAL_GEOMETRY_PLAN.md`及contextual诊断代码：先做Qwen/random key-value二因素替换，再比较final-layer last-token与目标span mean两种256维白化表示；白化正则与unknown阈值只在train选择，dev门槛不变，test继续封存。所有入口opt-in，baseline未修改。

### Qwen contextual key/value几何：通过

提交`d445a8e`的C0/C1正式实验完成，墙钟150秒、任务时间145.99秒，峰值allocated/reserved 7.867/8.277GiB。二因素dev对照中mean-Qwen/mean-Qwen为32.20%，Qwen-key/random-value为54.88%，random-key/Qwen-value为98.18%，random/random为100%，将旧表示的主要瓶颈定位到同世界相似entity key。`context_last`经train选择regularizer0.01与unknown_raw0.75后，train 99.89%，dev所有seed、family及1/2/4/8/16/32组件均100%；`context_mean`仅76.51%，未通过。独立审计通过，test未读取。

新增归档`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/qwen_contextual_geometry_v1`。新增`tasks/train_cbf_qwen_contextual_forgetting.py`及runner/audit/test，冻结通过的context-last表示后复用原四动作、三seed、300 epoch控制器、全部固定/随机/盲/global-only/physics/oracle对照和原dev门槛；只有门槛通过才首次读取原封存test。`train_cbf_lowrank_forgetting.py`仅增加默认0.35不变的可选`unknown_raw`参数传递，既有baseline行为不变。

### Contextual Qwen遗忘控制器v2：通过

提交`30c70ce`的正式v2完成，dev全部六项门通过后首次解封原32组test。test full learned准确率/NLL为89.05%/0.4321，最强固定local为79.01%/0.8979；准确率提升10.04点、来源组95% CI `[9.35,10.82]`，NLL收益0.4657、CI `[0.4437,0.4908]`。global-only为76.50%，reference-blind与fixed-local同为79.01%；full learned最差投影seed为88.80%，动作oracle匹配率97.79%。retain/revoke-last1/revoke-last2/reset分别98.32%/93.23%/91.44%/73.21%。

独立审计验证contextual来源、transform和embedding逐bit桥、60个raw策略文件、聚合、dev门与条件test解封全部通过。墙钟575秒、任务时间570.04秒，峰值allocated/reserved 7.868/8.277GiB。归档位于`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/qwen_contextual_forgetting_v2`。限制是physics与greedy oracle的test准确率同为89.05%；当前结果证明状态依赖遗忘优于固定/global-only策略，但没有证明学习优于直接读取显式revoke/reset规则。下一步必须使用自动activation/text evidence并迁移到ZsRE等自然顺序更新。

## 2026-10-13: Sparse long-session forgetting gate V1 implementation

Implemented `tasks/train_long_session_sparse_gate_v1.py`,
`scripts/run_long_session_sparse_gate_v1.sh`, and regression tests for the
registered long-session correction stage. The implementation freezes the final
Qwen3-4B checkpoint and the three contextual transforms, embeds the 2,136
audited corpus texts once, and trains only a rank-16 binary deletion gate. Its
observable features cover three projected cosine maxima, dispersion, candidate
margin, projection agreement, fixed similarity fractions, lexical/relation
overlap, unit counts, candidate age, and memory size. Chain identity, source
subject, split, and labels are not model inputs.

To make the 16/32/48/64-session causal grid tractable, all session-pair ranking
scores are precomputed once per episode on GPU, retained as compact CPU score
tables, and detailed top-pair features use a bounded LRU cache. An initial
single-CPU precompute attempt was stopped before any development summary after
15 minutes; no result from that attempt is used. Calibration-fit trains three fixed seeds; calibration-tune
selects epoch and threshold; fit+tune refitting precedes a one-time
subject-disjoint development replay. Controls include retain-all, always-delete
top candidate, tune-selected cosine, exact action-count-matched deterministic
random deletion, and oracle local deletion. Candidate top-1 and all release
checks are evaluated on development. LongMemEval-S remains locked and LoCoMo is
not loaded. Passing every registered check is required before a separate
LoCoMo preregistration; failure stops V1 and is retained as an architecture
diagnostic. `scripts/audit_long_session_sparse_gate_v1.py` independently
recomputes raw hashes, aggregates, fit/tune selections, candidate accuracy,
matched-random counts, clustered bootstrap comparison, release gate, source
hashes, and claim locks. Baseline training and inference entry points are
unchanged.

Run on `hku-gpu2` with:

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
bash scripts/run_long_session_sparse_gate_v1.sh
```

The audited run completed in 164.52 seconds at 8.090/10.023 GiB peak
allocated/reserved. Candidate top-1 was 33.99%; the learned three-seed mean was
55.20% action accuracy, 55.18% strict stale/current pair state, 73.75% stale
deletion, 77.81% current retention, and 58.43% non-target retention. The
primary seed beat exact action-count-matched random in pair state by 27.25
points with 95% CI `[22.66,31.65]`, but only that comparison check passed.
Oracle local deletion reached 100% on all lengths. The result rejects flat
maximum-cosine candidate selection under multi-fact sessions rather than local
forgetting itself. LoCoMo stayed sealed. Compact artifacts and the full report
are archived at
`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/long_session_sparse_gate_v1`;
61 MiB of hash-covered raw trajectories remain on `hku-gpu2`.

Registered and implemented `LONG_SESSION_CANDIDATE_V2_PLAN.md`,
`tasks/train_long_session_candidate_v2.py`, its immutable runner, independent
auditor, and regression tests. V2 isolates candidate retrieval before another
causal deletion experiment. It uses listwise supervision over all active
sessions, order-neutral per-projection max/top-k/mean/dispersion geometry,
projection agreement, similarity-density, lexical, and relation evidence. Its
primary ranker cannot observe age or unit position, preventing exploitation of
the synthetic event schedule and anchor serialization. Three seeds select only
on calibration-tune and are refit on fit+tune before development. V1 maximum
cosine, top-4, full-mean, consensus, and an ineligible anchor-only diagnostic
are fixed controls. A 95% mean top-1 gate with seed, length, MRR, and clustered
gain checks must pass before a separately registered sparse deletion gate.

V2 completed in 38.73 seconds at 8.090/10.023 GiB peak allocated/reserved and
passed all nine independent audit checks. The learned resolver reached 67.38%
top-1 and 73.16% MRR versus 33.99%/47.18% for V1 maximum cosine. Primary-seed
top-1 gain was 33.33 points with 95% CI `[28.93,37.64]`, and the three seeds
were stable within 0.19 points. Only the gain check passed; causal deletion did
not run and LoCoMo remained sealed. The ineligible anchor-only result of 79.21%
showed that the combined prompt-plus-changed-value key remains a bottleneck.
Compact results are archived under
`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/long_session_candidate_v2`.

Registered `FACTORIZED_SCOPE_CANDIDATE_V3_PLAN.md` and implemented its trainer,
runner, auditor, and tests. V3 mechanically splits each existing unit at the
audited `Value:` delimiter, embeds only the prompt/rephrase as the retrieval
scope key, and keeps the value as excluded payload. It reuses V2's fixed
listwise protocol without age or position features and hash-verifies the V2
paired baseline. The release gate additionally requires fixed scope-key maximum
cosine to beat the original full-text maximum by 30 points, so a passing claim
must demonstrate the factorization mechanism independently of the learned MLP.

V3 completed in 32.22 seconds at 8.026/8.988 GiB peak allocated/reserved and
passed all nine independent audit checks. Learned top-1/MRR reached
97.07/98.38%; all four lengths were at least 95.63% top-1, and the minimum seed
was 96.63%. Primary gain over V2 was 29.40 points with 95% CI
`[24.72,34.18]`. Fixed scope-only maximum cosine rose from 33.99% full-text to
93.63%, establishing key/value factorization independently of the learned
ranker. Compact artifacts are archived under
`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/factorized_scope_candidate_v3`.

Registered `FACTORIZED_CAUSAL_FORGETTING_V4_PLAN.md` and implemented its causal
trainer, runner, auditor, and tests. V4 freezes V3, uses a scope-max top-4
cascade followed by the listwise ranker, and trains a rank-16 deletion gate on
scope evidence plus value-payload conflict. Payload remains excluded from
candidate ranking; age and unit position remain excluded entirely. Complete
causal tune replay imposes 98% current, 97% non-target, 97% hard-retain, and
85% update constraints before maximizing strict pair state. The development
release gate checks shortlist/candidate recall, cumulative safety, stale
deletion, every length, and clustered gains over both V1 and exact
deletion-count-matched random. LoCoMo remains sealed until every check passes.

V4 completed in 383.54 seconds at 8.026/8.988 GiB peak allocated/reserved and
passed all ten independent audit checks. Candidate recall@4/top-1 remained
99.25/96.63%. The three-seed gate averaged 98.85% current retention, 97.34%
non-target retention, 97.32% hard-retain accuracy, 83.71% stale deletion, and
82.74% strict pair state. Six of ten release checks passed. Seed 31102 reached
91.48% pair state and 92.42% stale deletion with all safety metrics and every
length above 90%, while seed 31103 reached only 70.04% pair state. Inspection
of the frozen tune grids showed no fit-only threshold jointly met update and
safety constraints; refitting changed probability calibration. This supports
cross-fit calibration rather than another scope representation change.

Registered `CROSSFIT_CALIBRATED_FORGETTING_V5_PLAN.md` and implemented its
trainer, conditional runner, auditor, and tests. V5 uses deterministic nested
five-fold train/calibrate/validate subject splits, fixed 50-epoch gates, no
post-calibration refit, three gate seeds, and fold-specific causal thresholds.
The deployment controller takes the median signed margin of 15 calibrated
models. Absolute listwise ranker logits are removed from gate features, while
the frozen three-ranker V3 ensemble still selects a shared candidate. An OOF
gate must pass before development is loaded; the unchanged V4 ten-check gate
then controls any LoCoMo release.

V5 completed on 1,068 nested-OOF episodes and passed all ten audit checks.
Current/non-target/hard-retain safety reached 99.81/99.70/99.86%, while stale
deletion and pair state reached 83.99/83.90%. The latter missed the unchanged
85% prerequisite by roughly one point, so subject-disjoint development was not
loaded and no development files exist. Compact artifacts are archived under
`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/crossfit_calibrated_forgetting_v5`.

Registered `EPISODE_BARRIER_FORGETTING_V6_PLAN.md` and implemented V6 as a
fixed-objective variant of the audited V5 cross-fit engine. It changes only the
gate loss to `softplus(0.5 - (positive - max_retain)) + 0.25 * BCE`, directly
optimizing the per-episode update-versus-worst-retain safety barrier. Folds,
features, candidate resolver, epochs, calibration thresholds, OOF prerequisite,
conditional development gate, and stop rule remain unchanged. The runner and
auditor add an explicit objective lock; no V5 threshold or dev result is used.

V6 completed on the same 1,068 nested-OOF episodes and passed all eleven audit
checks. Current/non-target/hard-retain safety was 99.81/99.98/100.00%, but stale
deletion and strict pair state were 83.52/83.43%, slightly below V5 and below
the unchanged 85% prerequisite. Subject-disjoint development was not loaded;
the archive contains no dev records. This rejects the registered fixed-margin
episode barrier as the missing ingredient and activates the stop rule for any
further loss, epoch, margin, or threshold iteration on this constructed ZsRE
corpus. Compact artifacts are archived under
`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/episode_barrier_forgetting_v6`.

## 2026-10-12: ZsRE contextual readout and relation-aware key alignment

### Completed natural readout

Added a subject-disjoint natural factual-memory protocol from the local `ZsRE-test-all.json`: one deterministic row per unique subject, 267/267/534 calibration/development/test subjects and eight facts per session. The final 1B-token Qwen3-4B checkpoint, contextual last-token representation, three frozen 256-dimensional projections, whitening, and unknown score were selected using calibration only. Development exact accuracy was 100%, rephrase accuracy was 98.75/98.63%, and combined locality accuracy was 88.26%. The fixed 90% locality gate failed, so test was neither loaded nor scored. The independent audit passed and the compact artifacts are archived at `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/zsre_contextual_readout_v1`.

This protocol is a constructed candidate-memory readout study. It does not claim official ZsRE editing, open-vocabulary generation, or native dense TTT writer performance.

### Registered contrastive follow-up

Added `ZSRE_CONTRASTIVE_KEY_PLAN.md`, `tasks/train_zsre_contrastive_keys.py`, `scripts/run_zsre_contrastive_keys.sh`, `scripts/audit_zsre_contrastive_keys.py`, and `tests/test_zsre_contrastive_keys.py`. The experiment compares an exact identity bridge, prompt/rephrase-only InfoNCE, and a primary relation-aware arm that adds same-subject locality hard negatives. Each seed uses a zero-initialized rank-16 residual key map; values, source transforms, memory construction, split, Qwen backbone, and development gates stay frozen. Training uses only the 267 calibration subjects. The 534-subject test is read once only if the relation-aware arm passes all unchanged gates.

Run on `hku-gpu2` with:

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
bash scripts/run_zsre_contrastive_keys.sh
```

No baseline entry point or default configuration is changed. The learned head is an opt-in diagnostic module and introduces no new dependency. If it fails development, the registered stop rule keeps test sealed and requires a joint reference resolver/retrieval objective rather than lowering the locality threshold.

### Contrastive alignment result

Commit `f2c6ed9` completed on one RTX5090. All six server tests and the independent artifact audit passed; the identity arm reproduced every v1 development JSONL byte for byte. The primary relation-aware arm raised combined locality from 88.26% to 99.09%, while original/updated rephrase fell to 69.66/69.29%, below the unchanged 70% gates. Positive-only reached 99.97% locality but only 70.66/68.29% rephrase. Exact accuracy stayed at 100.00/99.63% for relation-aware. The test split remained sealed.

All trained seeds reached 100% calibration prompt-to-rephrase retrieval, so this is subject-disjoint overfitting rather than under-optimization. Source inspection clarified that ZsRE locality may be nearly synonymous with the edited relation but carries an unchanged ground truth; `unknown` in this constructed memory benchmark means deferring to the frozen model, not claiming the factual answer is unknown. The next registered method will learn pairwise query–memory scope instead of applying one global key transform.

### Pairwise scope resolver (before execution)

Added `ZSRE_SCOPE_RESOLVER_PLAN.md`, `tasks/train_zsre_scope_resolver.py`, `scripts/run_zsre_scope_resolver.sh`, `scripts/audit_zsre_scope_resolver.py`, and `tests/test_zsre_scope_resolver.py`. The primary resolver encodes each query jointly with each of the eight session memory prompts using the frozen final Qwen checkpoint, L2-normalizes the final-token hidden state, and trains only a linear score plus a `defer` logit. It receives no subject ID, relation label, answer, future event, split name, or row identifier.

The 34 calibration sessions are deterministically divided into 27 fit and seven checkpoint-selection sessions. Three initialization seeds train for 200 full-batch epochs; only epochs 10/25/50/100/200 can be selected by calibration-tune accuracy/NLL. A structured frozen bi-encoder cosine resolver is the control. The unchanged 267-subject dev split is used only for a six-part release gate, including ≥90% exact, ≥70% rephrase, ≥90% locality, ≥80% minimum-seed overall, ≥5-point gain over the control, and a positive session-bootstrap lower bound. Test remains conditionally sealed.

Run with `bash scripts/run_zsre_scope_resolver.sh` on `hku-gpu2`. All behavior is opt-in and does not modify baseline inference, training, `ttt_proj`, or `ttt_conv`.

The first formal launch at commit `f601834` passed all eight server tests and loaded the frozen checkpoint, then stopped before training because the final calibration session contains three facts and the initial tensor builder assumed every session contained eight. No optimizer step, development load, or test load occurred. The failed preflight remains at `/home/ctj/cbf_ttt_zsre_scope_resolver_v1`. The recovery pads each session to eight keys, masks invalid key logits to negative infinity, fixes `defer` as class nine, and adds a three-key shape/mask regression test. Data, pair template, model, optimizer, epochs, checkpoint selection, and release gates are unchanged.

Commit `00f949f` recovery completed and passed the independent audit. Structured bi-cosine reached 94.51% overall, 100% exact, 98.38% rephrase and 89.83% locality. The cross-linear primary reached 82.12/84.64/53.56/95.13%, respectively; it improved locality by 5.62 points but lost 12.42 points overall versus cosine, with bootstrap CI `[-14.61,-10.00]`. Exact, rephrase and control gates failed, so test remained sealed. Error decomposition found only two wrong-key rephrase errors for cosine but 163 locality false retrieves; the next method will retain cosine candidate ranking and train only retrieve/defer scope.

### Decomposed scope gate (before execution)

Added `ZSRE_DECOMPOSED_SCOPE_GATE_PLAN.md`, `tasks/train_zsre_decomposed_scope_gate.py`, `scripts/run_zsre_decomposed_scope_gate.sh`, `scripts/audit_zsre_decomposed_scope_gate.py`, and `tests/test_zsre_decomposed_scope_gate.py`. The candidate key is fixed by the mean cosine across all three frozen contextual projections. A binary linear gate sees only the top pair's frozen joint-Qwen hidden plus top score, cross-projection standard deviation, top1–top2 margin, and projection agreement; it can retrieve that key or defer but cannot rerank it.

Calibration fit/tune sessions, epochs, checkpoint epochs and optimizer stay fixed. The threshold-only ensemble is the control. The stronger development gate requires ≥95% exact, ≥90% rephrase, ≥92% locality, ≥93% minimum-seed overall, ≥1 point overall gain over control, and a positive source-session bootstrap lower bound. Only a full pass releases test once. No baseline path or dependency changes.

### Decomposed scope gate result

Commit `043462c` completed with 13/13 server tests and a fully passing independent audit. Dev decomposed gate/control accuracy was 98.50/91.20%; all six gates passed and released the original 534-subject test once. Test accuracy was 98.47% versus 88.58%, a gain of 9.90 points with 67-session bootstrap 95% CI `[8.57,11.24]`. Test exact/rephrase/locality were 100.00/97.75/98.06%, minimum gate-seed overall was 98.46%, and NLL was 0.0901 versus 0.2781. The candidate-selection upper bound was 99.95%.

This establishes automatic scope evidence on natural subject-disjoint text and removes the explicit symbolic-reference limitation of the preceding controlled controller. It does not yet establish natural sequential forgetting: the gate decides whether a query should read a structured update, not which accumulated memory components should be deleted before future queries.

### Natural duplicate-chain forgetting benchmark (construction stage)

Added `ZSRE_NATURAL_FORGETTING_PLAN.md`, `tasks/build_zsre_natural_forgetting.py`, and `tests/test_zsre_natural_forgetting.py`. Previously excluded duplicate ZsRE rows are assigned to their original canonical subject split, grouped by subject and normalized old answer, deduplicated by distinct new target, and deterministically ordered into counterfactual update chains. Up to eight chains are interleaved per session: initial versions require retain, later versions require local forgetting. The planned calibration-only pilot compares fixed retain/global/top deletion, cosine local deletion, the frozen learned scope gate, and hidden-chain oracle; development and test remain unopened until feasibility and gates are frozen.

The generated manifest at `/home/ctj/cbf_ttt_zsre_natural_forgetting_v1_data` records 26/31/52 subject-disjoint chains, 56/73/120 distinct-target versions, 30/42/68 local transitions and 4/4/7 packed sessions for calibration/dev/test. Added `tasks/evaluate_zsre_natural_forgetting.py`, a calibration-only runner, independent audit and replay tests. Frozen Qwen entity and joint-pair features are extracted once, then all six policies and three frozen gate seeds replay identical updates and current/historical/locality queries. The pilot records runtime, peak memory, stale reads, action correctness and chain-bootstrap policy comparisons; it has no code path that loads development or test.

Commit `ce1e9f8` completed in 16.16 seconds at 8.231/9.576 GiB peak allocated/reserved; 10 server tests and the independent calibration-only audit passed. Cosine local reached 97.21% query and 98.21% action accuracy, close to oracle 97.77/100%. Learned local reached only 84.43/53.57% and was 11.41 points below cosine, with chain-bootstrap CI `[-16.04,-6.82]`. It correctly classified only 13/26 retain and 17/30 local events. V1 is too easy for cosine because retain always introduces another subject. Stage B is cancelled; a v2 calibration pilot will add each row's consistent two-alias locality ground truth as an independently retained same-subject fact.

Added `ZSRE_HARD_RETAIN_FORGETTING_PLAN.md`, `tasks/build_zsre_hard_retain_forgetting.py`, and construction tests. V2 pairs every primary chain with one same-subject auxiliary locality fact whose two aliases share an identical source ground-truth set; all calibration/dev/test chains pass this schema condition. Sessions contain at most four primary plus four auxiliary chains. Initial primary and auxiliary writes are retain events, later primary versions are local-forget events. The A2 calibration pilot must show a nontrivial cosine–oracle gap and learned or hybrid improvement before any development run.

Added the calibration-only evaluator, independent recomputation audit, and immutable remote runner for hard-retain A2. Primary and auxiliary chains share one paired-subject bootstrap unit. Metrics separate primary-initial retain, same-subject hard retain, and primary-update local deletion, plus primary versus auxiliary readout. The preregistered A2 gate requires a cosine-to-oracle gap of at least three points and a learned gain over the best fixed baseline whose paired bootstrap 95% interval excludes zero. The builder records that all splits are transformed mechanically while explicitly asserting that held-out labels are not used for method selection; the evaluator has no development or test loading path.

Commit `ac6795d` completed hard-retain A2 in 12.44 seconds at 8.231/9.727 GiB peak allocated/reserved; 14 server tests and the independent audit passed. Cosine local reached 74.02% query and 91.46% action accuracy versus oracle 77.66/100%, establishing the preregistered 3.64-point gap. The transferred query-scope gate reached 69.07% and was 2.46 points below cosine with paired CI `[-8.75,3.72]`, so development remained sealed.

Added `ZSRE_FORGETTING_CONTROLLER_PLAN.md`, `tasks/train_zsre_forgetting_controller.py`, controller tests, an independent audit, and an immutable runner. The new update-time controller scores each active candidate versus retain-all using frozen Qwen pair features, bidirectional cosine and lexical features without chain identity. Seven outer leave-one-session-out folds each use a deterministic inner tune session for epoch selection, refit on all six remaining sessions, and causally replay the unseen session. Three controller seeds are paired with three frozen readout seeds. A deployable checkpoint is refit on all calibration sessions using median selected epochs, while development and test remain inaccessible.

Commit `21b4a57` completed the first nested-LOSO controller in 15.40 seconds; the independent audit passed. Its 2,571-dimensional head overfit: query/action accuracy was 54.64/54.88% versus cosine's 74.02/91.46%, with paired action gain -36.86 points and CI `[-51.07,-21.79]`. A calibration-only oracle-state diagnostic found that the best individual low-dimensional lexical feature reached 76.00% action accuracy. Added V2 as a residual binary gate over the frozen cosine top candidate. It uses only 13 cosine/lexical/relation features, preserves cosine candidate ordering, and repeats the nested 5/1/1 session protocol for epoch and threshold selection.

Commit `6a47dda` completed residual gate V2 in 13.20 seconds and passed its independent audit. All three seeds were identical: query/action accuracy reached 76.45/95.12%, versus cosine 74.02/91.46%. Hard-retain rose from 88.46% to 96.15% and primary-update action from 86.67% to 90.00%. Paired gains were +2.75 points query (CI `[0.00,6.63]`) and +4.17 points action (CI `[-1.92,12.82]`), so development remained sealed. Added V3 to augment binary gate training with 267 contextual calibration subjects, one same-scope rephrase positive and two locality negatives per fact. Every nested fold excludes auxiliary examples from held-out subjects, preserving outer subject independence.

Commit `11af360` completed augmented V3 in 15.48 seconds and passed the subject-exclusion audit. It matched V2 action accuracy (95.12%) and slightly reduced query gain to +2.50 points; confidence intervals remained inconclusive, so the 801 auxiliary labels did not solve the 26-unit power limit. Added a full counterfactual sequence builder over the canonical 267/267/534 subject-disjoint ZsRE split. Each unit writes the source truth, retains one independent same-subject locality fact, then locally replaces the source truth with the original `target_new` under the source rephrase. The expanded calibration runner reuses the frozen cosine selector and low-dimensional gate under 67-session nested LOSO; development and test remain unscored.

The first expanded build stopped before training because 9/11/18 calibration/development/test records have identical normalized source and target values. The builder now excludes and hashes these no-op updates explicitly, leaving 258/256/516 valid subject units and 65 calibration sessions. The nested calibration audit expects 774 updates, 709 candidate-bearing decisions and 258 paired bootstrap units.

Commit `c2a312c` completed the 258-unit calibration in 62.91 seconds at 8.513/11.254 GiB peak allocated/reserved; the independent audit passed. The controller reached 77.46% query and 96.12% action accuracy versus cosine 76.53/94.44%. Paired query gain was significant at +0.91 points (CI `[0.13,1.77]`); action gain was +1.68 points with borderline CI `[-0.13,3.75]`. Discordant actions were 26 controller wins versus 13 losses (exact McNemar p=0.0533), led by hard-retain at 11 wins versus one loss. Added a five-fold grouped inner OOF threshold protocol with fixed 200-epoch training, selecting thresholds by causal query then action accuracy before evaluating each untouched outer session.

The first grouped-OOF implementation was manually stopped after eight minutes before scoring because it redundantly refit an identical convex action head for each readout seed and replayed all queries for all 17 thresholds. Added an equivalent efficient protocol: train one head per inner fold, collect all grouped OOF action logits once, select the threshold by action accuracy, refit on the full outer-training fold, and reuse the action decisions with the three frozen readout seeds. This preserves outer LOSO independence while removing redundant training and query replay.

Commit `5c230f3` completed efficient action-OOF calibration in 60.88 seconds; the independent audit passed. The controller reached 77.53% query and 96.77% action accuracy versus cosine 76.53/94.44%. Paired query gain was +1.03 points (CI `[0.34,1.76]`) and action gain +2.33 points (CI `[0.65,4.13]`); both preregistered gates passed. Primary-initial/hard-retain/local-update action accuracies were 99.61/96.51/94.19%, and 53/65 outer folds selected threshold 0.4. Added a frozen formal evaluator whose six-check development gate conditionally releases one-time test scoring; it reuses the immutable calibration checkpoint and records all source hashes.

Commit `d7757ae` added and executed the frozen formal evaluator on 256 subject-disjoint development units. It records three controller/readout seeds, three cosine baselines, paired subject bootstrap intervals, source/checkpoint/data hashes, and conditionally loads test only after six development checks. Development query/action gains were +3.51/+7.03 points with 95% CIs `[2.27,4.82]`/`[4.56,9.77]`, but overall action, hard-retain, and primary-update accuracies were 91.41/87.89/86.72%, below the frozen 96/95/92% floors. Test was neither loaded nor scored; the independent audit passed and verified its absence. Runtime was 41.69 seconds with 8.233/11.297 GiB peak allocated/reserved. Compact results and a report are archived under `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/zsre_counterfactual_forgetting_formal_v1`; raw records remain on the server. The next stage uses development only for threshold diagnostics and V2 selection, keeps the write gate fixed at one, and preserves test for a single frozen evaluation.

Added a development-only causal threshold diagnostic for the frozen forgetting head. It replays thresholds 0.1--0.9 rather than relabeling decisions offline, so each threshold receives its own evolving memory state and query outcomes. The diagnostic reuses hash-verified formal cosine records, writes three controller/readout-seed traces per threshold, reports the original six-check gate, and ranks thresholds by action reliability before query accuracy. Its independent auditor recomputes every aggregate, paired bootstrap interval, source hash, selection, and test-sealed invariant. The immutable runner records tests, resources, terminal status, and failure state. This experiment decides whether calibration alone is sufficient before any V2 architecture change.

The coarse diagnostic completed in 86.83 seconds at 8.233/11.297 GiB peak allocated/reserved and passed its independent audit. No coarse threshold passed all six checks. Threshold 0.8 maximized overall action at 97.01% but update accuracy was 91.02%; threshold 0.7 reached 96.61% overall, 98.05% hard-retain, and 91.80% update accuracy, missing the 92% update floor by one event. Added an explicit custom-grid option and generalized the auditor to a manifest-recorded grid. A preregistered sparse boundary sweep over 0.625/0.650/0.675/0.725/0.750/0.775 will decide scalar-calibration sufficiency without reopening test.

The sparse boundary sweep completed and passed independent audit. Thresholds 0.625 and 0.675 passed all original development checks; the registered reliability ordering selected 0.675. It reached 77.91% query and 96.74% action accuracy, with hard-retain/update at 98.05/92.19%. Paired gains over cosine-local were +6.27 query points (CI `[4.70,7.95]`) and +12.37 action points (CI `[8.98,15.89]`). Test remained sealed. Added a one-time final evaluator that accepts only a hash-matched, independently audited passing boundary result, freezes the calibration-trained head and selected threshold, loads the 516-unit test once, writes all raw controller/baseline traces, and applies the unchanged six checks. A separate auditor recomputes every test aggregate, interval, gate, source hash, threshold, and unit count.

Commit `7505257` executed the one-time 516-unit held-out test at the frozen threshold 0.675. The independent audit passed every raw trace, source hash, aggregation, bootstrap, threshold, and unit-count check. Learned forgetting reached 77.23% query and 95.16% action accuracy versus cosine-local 70.67/82.43%; paired gains were +6.59 points (CI `[5.43,7.79]`) and +12.73 points (CI `[10.27,15.25]`). Hard-retain improved to 96.12% and update to 89.34%, but the unchanged 96% overall-action and 92% update floors failed. Current-exact/rephrase/historical query accuracy was 98.51/50.65/95.27%. Runtime was 82.47 seconds at 8.281/10.350 GiB peak allocated/reserved. Compact coarse, boundary, and final artifacts plus reports are archived under the final-checkpoint experiment directory; raw traces remain on hku-gpu2. The next architecture stage must use readable Qwen low-rank fast memory and a new source-disjoint final benchmark because the ZsRE test is now observed.

Added a preregistered CounterFact zero-shot transfer stage using the official ROME dataset. The downloader records the official URL and file hash; the observed server file has 21,919 rows and SHA256 `d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f`. The deterministic builder selects one primary and one different-relation auxiliary record for each eligible same-subject group, producing the same retain/hard-retain/counterfactual-update topology as ZsRE without using model scores. The evaluator freezes the ZsRE-trained controller and threshold 0.675, performs no CounterFact training or tuning, and compares learned forgetting with cosine-local plus fixed retain/global/top and oracle controls. The audit independently rebuilds all policy summaries, paired intervals, hashes, zero-shot invariants, and subject-unit count.

The one-time CounterFact run scored 785 unique subjects and passed every independent audit check. Learned forgetting reached 48.12% query and 67.77% action accuracy versus cosine-local 43.45/56.26%; subject-paired gains were +4.57 points (CI `[3.90,5.24]`) and +11.51 points (CI `[9.51,13.50]`). The registered absolute transfer gate failed: hard-retain was 78.73% and update was 30.70%, while fixed retain reached 49.44% query accuracy. Added a deterministic failure diagnostic that verifies all controller raw hashes, confirms identical actions across readout seeds, measures event-wise probability distributions and AUC, and reports a clearly marked noncausal fixed-trajectory threshold grid. AUC was 0.759, but the best diagnostic threshold reached only 72.61% binary action and the selected old chain was correct for only 49.08% of triggered updates. These results rule out threshold-only repair and motivate a decomposed pairwise candidate ranker plus conflict gate; CounterFact is now development-only.

Added the preregistered cross-domain forgetting-reranker development stage. It keeps the current write gate fixed at one, adds a residual pairwise ranker on top of the frozen cosine prior, and trains a separate safe-delete gate from the selected pair and rank geometry. The runner evaluates a ZsRE-only transfer arm and deterministic five-fold CounterFact grouped OOF arm; all method selection stays on already observed ZsRE/CounterFact. LongMemEval-S and LoCoMo source hashes are frozen before method scoring, with their model scores sealed for a later final evaluation. New construction tests verify deterministic balanced folds and identity namespacing without changing model-visible text.

Reranker V1 processed 4,990 decision groups and improved CounterFact OOF candidate top-1 to 78.47%, causal query to 49.66%, and action to 84.93%; query/action gains over the frozen controller were +1.15 points (CI `[0.49,1.84]`) and +17.15 points (CI `[15.24,19.15]`). It passed candidate, hard-retain, and fixed-retain query checks but failed overall action and update (63.69%). V2 then used complete outer-training folds, fixed short ranker training, class-balanced semantic-conflict targets, and nested inner OOF calibration. Candidate top-1 improved to 79.62% and oracle-state update to 78.22%, but thresholds 0.20–0.25 caused cascading false deletion: causal query/action fell to 46.79/78.26%, hard-retain to 80.25%, and update to 61.15%. Its independent audit passed every raw, fold, checkpoint, source, aggregate, comparison, and sealing check. Added a preregistered no-weight-change constraint-calibration replay that chooses inner thresholds satisfying hard-retain ≥85% and update ≥75% before any fallback.

Constraint calibration selected fold thresholds 0.50/0.45/0.40/0.45/0.40 without changing weights. It recovered query/action to 48.64/84.29% and hard-retain to 88.66%, but causal update was 67.52% versus 76.82% on oracle states, exposing a 9.30-point state-distribution gap. Added V3 one-round on-policy collection, variable-length memory support, asymmetric retain loss, and causal inner-fold threshold selection. V3 processed roughly 4.1k policy-induced groups per fold, including 51–86 local events whose old chain was already missing. It reached 49.15% query, 82.34% action, 88.92% hard-retain, and 59.75% update; all five inner folds required the minimum-deficit fallback. The registered stop rule therefore fired: scalar-gate iteration on the same explicit candidate representation ends, and the next method must improve semantic relation/scope representation before any sealed benchmark is scored.

Added the preregistered Scoped Delta Ranker V4 stage. `tasks/build_scoped_delta_corpus.py` converts the existing sequential ZsRE and CounterFact sessions into an audited pair corpus with same-chain positives, same-subject/different-chain hard negatives, and different-subject negatives; it records subject-span coverage and source hashes. `tasks/train_scoped_delta_ranker_v4.py` freezes the final Qwen3-4B checkpoint, combines the existing full-pair features with subject-masked relation-pair features, and evaluates `structured_full`, `no_relation`, and `no_subject` arms under deterministic five-fold CounterFact OOF and three initialization seeds. The primary gate requires ≥88% top-1, ≥93% MRR, ≥5-point gain over V2 with positive paired-bootstrap lower bound, and per-negative-family floors. Added construction/ranker tests, a remote runner, and `SCOPED_DELTA_RANKER_V4_PLAN.md`. The subject span is explicitly marked as an oracle-segmentation diagnostic, while chain identity, relation IDs, age, LongMemEval-S, and LoCoMo remain outside model features and selection.

V4 processed 1,030 ZsRE and 785 CounterFact local-update groups. The primary arm reached 82.93% top-1 and 90.91% MRR, improving over V2 by 3.31 points with paired-bootstrap CI `[0.64,5.86]`; the registered gate still failed. Relation features contributed 2.04 points and the subject bit 0.38 points. Error audit found 95 same-subject errors, all involving different CounterFact relation labels, plus 39 different-subject errors. Archived the compact V4 result and report. Added `RELATION_SUPERVISED_SCOPE_V5_PLAN.md`, `tasks/train_relation_supervised_scope_v5.py`, its runner and tests. V5 uses 34 CounterFact relation IDs only as class targets, excludes every outer/inner subject from the raw supervision pool, selects among four fixed relation-compatibility rules on an inner fold, and keeps inference text-only apart from the explicitly acknowledged oracle subject span. The corpus now records source-row hashes for independent relation-label auditing.

V5 encoded 43,853 unique relation windows from 65,757 raw examples and reached 97.83% candidate top-1 / 98.92% MRR. The 14.90-point gain over V4 had paired-bootstrap CI `[12.23,17.58]`. Exact 34-way query relation classification was 87.64%, below its preregistered 90% auxiliary floor, while 82/97 class errors still produced the correct pairwise candidate; the complete gate therefore failed and causal deletion remained stopped. Archived the compact V5 result and report. Added `AUTOMATIC_FACTORIZED_SCOPE_V6_PLAN.md`, `tasks/evaluate_automatic_factorized_scope_v6.py`, a runner, and pure tests. V6 removes subject metadata from inference by locating the maximum-IDF shared contiguous span, forms pair-specific relation windows, and combines V5 posterior compatibility with normalized entity evidence. IDF and combination weights use fold-disjoint text and inner-fold selection; relation IDs, chain IDs, ages, targets, and future queries remain unavailable at inference.

V6 reached 93.25% text-only candidate top-1 and 96.37% MRR. It improved over V4 by 10.32 points with paired-bootstrap CI `[7.52,13.25]`, but trailed oracle-segmented V5 by 4.59 points and failed the 95%/97%/within-three-points gate. Archived its compact result and report. All five inner folds selected the registered maximum entity/relation weight of 8, with monotonic accuracy at the boundary. Added the one-shot `AUTOMATIC_SCOPE_BOUNDARY_V7_PLAN.md` and runner; the shared V6 evaluator now accepts an explicit protocol and weight list without changing the original default. V7 tests only `{8,12,16,24,32,48,64}` under the unchanged gate and has a mandatory stop rule against further scalar expansion.

V7 reached 93.63% top-1 and 96.76% MRR, only 0.38/0.39 points above V6; it failed the unchanged gate, so scalar entity/relation weight expansion stopped permanently. Archived its compact result and report. Added `TEXT_ONLY_SCOPE_CALIBRATOR_V8_PLAN.md`, `tasks/train_text_only_scope_calibrator_v8.py`, its runner, and a pure gate test. V8 recomputes OOF V5/V6 features and trains a rank-16 residual over 13 text-only features using nested five-fold epoch selection and three seeds. It excludes subject/relation/chain metadata, values, age, candidate position, and future queries at inference. The V8 gate requires ≥95% top-1, ≥97% MRR, ≥10 points over V4, and a statistically positive ≥1-point gain over V7 before causal evaluation can resume.

V8 reached 94.01% top-1 and 96.97% MRR. Its 0.38-point gain over V7 had CI `[-0.76,1.66]`; the gate failed and fixed-feature calibration stopped. Archived its compact result and report. Added `LEARNED_ENTITY_SCOPE_V9_PLAN.md`, `tasks/train_learned_entity_scope_v9.py`, its runner, and tests. V9 builds balanced positive and same-relation/different-subject hard-negative text pairs from raw CounterFact, excludes every outer subject from both sides of entity training pairs, trains three frozen-Qwen linear entity heads, and combines `log P(entity)` with the automatic `log P(relation)`. Subject/relation identifiers are supervision and audit labels only; inference receives raw query/key text.

V9 processed 87,676 balanced examples and 93,149 unique entity pairs. Its entity-pair AUC was 96.57%, but the fixed entity/relation probability product reached only 92.48% candidate top-1 and 95.79% MRR. It was 1.53 points below V8 with CI `[-3.31,0.25]`; the gate failed. Archived the compact result and report. The entity probability remains an OOF text-only signal, but the rigid product rule is rejected. At most one nested-OOF fusion with the V8 feature set is allowed before CounterFact score calibration stops and work moves to token-level entity-span extraction.

Added the final registered `ENTITY_FUSION_SCOPE_V10_PLAN.md`, `tasks/train_entity_fusion_scope_v10.py`, its runner, and a pure gate test. V10 loads the hashed fold-specific V9 entity heads, encodes only evaluation candidate pairs, appends entity probability/log-probability/logit to the V8 feature vector, and retains nested five-fold epoch selection plus three-seed ensembling. Its gate requires ≥95% top-1, ≥97% MRR, a statistically positive ≥1-point gain over V8, and ≥12 points over V4. Failure permanently ends CounterFact score fusion and redirects work to token-level span extraction.

V10 completed at 94.65% top-1 and 97.29% MRR. The +0.64-point gain over V8 had paired-bootstrap CI `[-0.25,1.66]`, so the registered gate failed. Archived the prediction trace, manifest, metrics, runtime, GPU peak, and report. A post-hoc join of all 42 errors to the frozen corpus found 41 `same_subject_different_chain` selections and one different-subject selection; oracle-segmented V5 corrected 32/42. Added `TOKEN_SPAN_RELATION_SCOPE_V11_PLAN.md`. V11 uses subject labels only to train a subject-disjoint frozen-Qwen token classifier, masks the predicted span before relation encoding, and evaluates fully text-only OOF ranking. This corrects the earlier entity-matching interpretation: the learned span is primarily needed for relation de-identification.

Implemented V11 in `tasks/train_token_span_relation_scope_v11.py`, with `scripts/run_token_span_relation_scope_v11.sh` and pure tests. The data builder combines raw CounterFact prompts/paraphrases with constructed sequential texts, labels exact subject-overlapping tokens, and deduplicates text/subject pairs. Frozen Qwen token and whole-text hidden states are projected deterministically and combined with surface features. A three-seed MLP head selects epoch and threshold on the next subject-disjoint inner fold. Its predicted span is replaced by `<ENTITY>` before the frozen V5 relation classifier is applied. Nested selection chooses only between a predicted-span exact-match constraint and V7's fixed overlap weight. The output records span exact match/token F1, relation-only and fixed-arm ablations, primary five-fold predictions, paired confidence intervals, hashes, runtime, and GPU peak. No inference-time subject or relation metadata is passed to the resolver.

V11 completed on 785 CounterFact OOF decisions at 97.20% top-1 and 98.60% MRR. It improved over V10 by 2.55 points with paired-bootstrap CI `[1.02,4.20]` and came within 0.64 points of oracle-segmented V5. Span exact match/token F1 were 93.27%/98.14%; all folds selected epoch 15, threshold 0.3, and `overlap8`. The complete registered gate passed. Archived its compact evidence and report, and added `CAUSAL_TOKEN_SPAN_FORGETTING_V12_PLAN.md` to test whether the frozen resolver improves memory causally under sequential replay.

Implemented V12 in `tasks/train_causal_token_span_forgetting_v12.py`, with its runner and protocol tests. V12 reloads the hashed V11 span heads and V5 relation classifiers, reconstructs text-only span predictions for every visible update, and freezes the resulting `log relation + 8 * span-overlap` candidate resolver. A three-seed linear retain/local head uses only resolver confidence, margins, entropy, overlap, exact-span agreement, and memory count. Epoch and threshold are selected through causal inner-fold replay under 95% hard-retain and 90% update floors; outer heads are refit on four folds and replayed once on the untouched outer fold. The old query readout and archived fixed-retain/oracle traces are retained for causal attribution and paired comparison. The current write gate remains one.

V12 completed with 97.07% static candidate top-1 and 50.92% causal query accuracy. It beat fixed retain by 1.46 points with paired-bootstrap CI `[0.99,1.96]`, providing positive causal evidence, but failed the strict action gate at 90.40% overall, 93.50% hard-retain, and 80.00% update. The audit attributed update errors to 121 false retains and 36 wrong deletions, so the bottleneck is the conservative linear gate. Archived V12 and added the final `CAUSAL_ONPOLICY_SCOPE_V13_PLAN.md`, which keeps V11 frozen and tests a nonlinear posterior-geometry gate with one on-policy aggregation round plus an oracle-state-only ablation.

Implemented V13 in `tasks/train_causal_onpolicy_scope_v13.py`, with its runner and gate test. V13 adds query/candidate posterior confidence and entropy, class agreement, posterior cosine, and Jensen-Shannon divergence to the ten V12 physical features. Its rank-16 three-seed gate is trained either on oracle states alone or on the registered union of oracle and V12-induced states. Each arm uses the same nested causal epoch/threshold selection, outer refit, frozen candidate resolver, and frozen readout. The primary arm is compared pairwise with fixed retain, oracle local, V12, and the oracle-state MLP ablation. This is registered as the last CounterFact gate-architecture iteration.

V13 completed in 756.28 seconds with 9.026/10.744 GiB peak allocated/reserved memory. The frozen candidate resolver remained at 97.07% top-1. The on-policy nonlinear arm reached 51.52% query accuracy, 95.29% overall action accuracy, 96.56% hard-retain, and 89.55% primary-update accuracy. It improved query accuracy over fixed retain by 1.949 points, subject-bootstrap CI `[1.547,2.368]`, and over V12 by 0.489 points, CI `[0.245,0.752]`. Five of six preregistered checks passed; primary-update missed the 90% floor by 0.446 points, so the development gate failed and CounterFact controller tuning stops. On-policy aggregation was not significantly different from the oracle-state-only nonlinear ablation; the supported improvement is posterior geometry. Added the complete V13 report and archived summary/training/manifest/terminal artifacts. The next natural-memory work must use the already readable contextual low-rank Qwen substrate or an independently validated writer; previous audited experiments rule out repeating the same failed dense `ttt_proj`/`ttt_conv` content-binding path.

Added `tasks/synthesize_crossdomain_forgetting.py` and focused tests to combine frozen CounterFact V13 and the one-time ZsRE held-out result without any model fitting, threshold selection, or benchmark rescore. The two domains independently show positive clustered query/action gains and agree within 0.44 points on every action slice. Descriptively, 3,717 of 3,903 actions are correct (95.23%): primary-initial 99.85%, hard-retain 96.39%, and primary-update 89.47%. The report preserves CounterFact's development-only status, keeps domain-specific confidence intervals primary, rejects pooling query accuracy, and records that native dense fast weights, LongMemEval-S, and LoCoMo remain unproven/unscored.

Preregistered and implemented `LONGMEMEVAL_TRANSFER_V1_PLAN.md` and `tasks/evaluate_longmemeval_forgetting_transfer.py`. The runner hashes the fixed LongMemEval-S cleaned file, selects all 78 `knowledge-update` rows, and scores them exactly once with the frozen ZsRE contextual transforms, binary forgetting head, and held-out threshold 0.675. Every dated role-labelled session is split into non-overlapping 1024-token windows without truncation; maximum window-pair cosine supplies the three frozen semantic scores. Six arms compare retain-all, an eight-session window, unconditional top deletion, action-rate-matched deterministic random deletion, the frozen controller, and an answer-ID-only oracle upper bound. Primary state evidence is whether the stale answer session is deleted while the current answer session remains; question hit@1/5, stale hit@1/5, non-answer retention, memory size, latency, and peak GPU memory are also recorded. The gate uses paired 10,000-resample question bootstrap and is frozen before scoring. Added a two-RTX-5090 shard runner and pure-Python construction/policy tests. This is a zero-shot mechanism-transfer study, not the official LongMemEval generation score; LoCoMo remains unscored.

The first LongMemEval transfer launch stopped before embedding or scoring because the strict renderer encountered three role-labelled turns with empty `content`. A schema-only audit confirmed that no session consists entirely of empty turns. Before any shard result existed, the frozen normalization was amended to skip empty-content placeholders while still rejecting nonempty turns without roles and all-empty sessions; a focused regression test covers this case. The failed output directory is retained as provenance and the formal retry uses a fresh directory.

The one-time LongMemEval-S knowledge-update transfer completed on all 78 questions and failed its registered gate. The frozen controller reduced stale hit@5 from 61.54% to 7.69%, a 53.85-point reduction with question-bootstrap CI `[41.03,66.67]`, but over-deleted: current answer-session retention was 50.00%, non-answer retention 41.69%, current hit@5 50.00% versus 60.26% for retain-all, and strict stale/current pair state 44.87% versus 51.28% for action-rate-matched random (difference CI `[-20.51,7.69]`). The answer-ID oracle reached 100% pair state, current hit@5 66.67%, and stale hit@5 0, demonstrating that local deletion is feasible while the zero-shot session controller is miscalibrated. Both RTX 5090 shards processed 39 questions and 10,101 total windows in about 407 seconds wall time; peak allocated/reserved memory was 7.983/8.184 GiB. Added compact raw metrics, resource records, a full report, and `scripts/audit_longmemeval_forgetting_transfer.py`; seven focused tests pass and the independent audit recomputes all hashes, metrics, intervals, locks, and the failed gate. This is not an official LongMemEval generation score, the 78 outcomes are locked against retuning, and LoCoMo remains unscored.

Added `LONG_SESSION_SPARSE_GATE_V1_PLAN.md` and `tasks/build_long_session_sparse_corpus.py` for the independent corrective stage. The builder uses only the existing ZsRE contextual-readout calibration/dev splits, deterministically splits calibration source sessions into fit/tune, and constructs one 16/32/48/64-session episode per primary fact. Each episode has exactly one old/current update pair, two same-subject locality hard-retain sessions, unrelated sessions, and four factual units per session. It stores source/chain metadata only for labels and auditing, verifies subject disjointness, and records that LongMemEval and LoCoMo are not used. Added deterministic construction tests; no LongMemEval outcome is an input to corpus construction or later fit/tune selection.

## 2026-10-13: MQuAKE-T natural temporal transfer V7

The constructed ZsRE loss/threshold loop is closed after V6. V7 tests whether the
factorized forgetting rule transfers to naturally occurring temporal changes.
Official MQuAKE-T is an independent development source; repeated cases for one edited
subject remain in the same split. LoCoMo stays sealed, and MQuAKE-CF is not used.

- Added `MQUAKE_TEMPORAL_TRANSFER_V7_PLAN.md` with frozen data access, method, metrics,
  and release criteria.
- Added `tasks/build_mquake_temporal_sessions.py`. It verifies the one-update schema,
  creates a deterministic subject-disjoint 60/20/20 split, and builds 16/32/48/64
  event episodes containing an old fact, its real update, natural multi-hop context,
  two different-subject hard retains (same relation when available, with coverage
  recorded), and distractors. The manifest
  records official source/license URLs and all source/output hashes.
- Added `tasks/evaluate_mquake_temporal_transfer_v7.py`. It reads only MQuAKE dev,
  freezes the V3 resolver and V5 calibrated ensemble, and compares against retain-all,
  exact deletion-count matched random, and oracle policies.
- Added `scripts/audit_mquake_temporal_transfer_v7.py` to recompute metrics and gates,
  verify data/checkpoint/output hashes, split separation, exact matched deletions, and
  the absence of MQuAKE fitting, MQuAKE-CF, LongMemEval, and LoCoMo access.
- Added `scripts/run_mquake_temporal_transfer_v7.sh` and focused construction/gate
  tests. Updated `DATA_DOWNLOAD_GUIDE.md` and `README.md`.

Run with `bash scripts/run_mquake_temporal_transfer_v7.sh`. The feature is opt-in and
does not alter baseline model, optimizer, dataset, or evaluation paths. If zero-shot
transfer fails, the registered next stage can fit only on MQuAKE train, calibrate only
on tune, and then score the entity-disjoint dev once. Risks include only 96 unique
temporal scopes, skewed subject frequencies, repeated contexts, and the remaining gap
between structured memory units and raw conversational sessions.

The first server launch stopped before model loading because the V5 runner default
used the historical `development/` layout while the archived cross-fit run stores its
checkpoint under `result/`. The default was corrected; no MQuAKE embedding, prediction,
or metric existed in the failed directory.

V7 then completed on all 1,496 entity-disjoint dev episodes in 156.89 seconds at
7.867/8.115 GiB peak allocated/reserved memory. Candidate shortlist/top-1 were
31.28/26.34%. The frozen controller reached 78.01% stale deletion but only 31.82%
current retention, 29.16% non-target retention, and 9.83% strict pair state. Exact
deletion-count matched random reached 51.14% pair state; controller-minus-random was
-41.31 points with subject-bootstrap CI `[-45.52,-39.18]`. Oracle was 100%. The audit
passed all checks, the gate failed, and LoCoMo remained unscored. Compact evidence and
`REPORT.md` are archived under
`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/mquake_temporal_transfer_v7`.

Added `MQUAKE_TEMPORAL_ADAPTATION_V8_PLAN.md`,
`tasks/train_mquake_temporal_adaptation_v8.py`, its runner, audit, and focused tests.
V8 fits three 50-epoch factorized candidate heads and three 50-epoch BCE deletion
heads on MQuAKE train. Tune selects only deletion thresholds; no model is refit after
calibration. A six-check tune prerequisite conditionally permits entity-disjoint dev,
where an independently recomputed nine-check gate requires scope, safety, deletion,
every-length, and matched-random confidence criteria. MQuAKE-CF, LongMemEval-S, and
LoCoMo are locked out. The baseline remains unchanged because this is a separate
opt-in runner.

V8 completed in 892.01 seconds at 7.867/8.115 GiB peak allocated/reserved memory and
passed the independent audit. All three heads found safety-feasible tune thresholds
(0.85, 0.85, 0.75). Current/non-target/hard-retain accuracy reached
100.00/98.86/97.49%, but candidate shortlist/top-1 was 28.82% and pair state 15.88%.
The tune prerequisite failed, so dev artifacts are absent. Added the compact result and
full report under `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/mquake_temporal_adaptation_v8`.

Added `MQUAKE_CANDIDATE_DIAGNOSTIC_V9_PLAN.md`,
`tasks/diagnose_mquake_candidate_scope_v9.py`, its independent audit, runner, and gate
tests. The tune-only diagnostic reuses V8's frozen train-fitted candidate heads and
compares full-list learned ranking, all-unit maximum and explicit scope-anchor
shortlists at widths 4/8/16/32. It records raw target ranks and selects an arm only at
>=90% recall and >=80% top-1. No deletion model is trained and dev, MQuAKE-CF,
LongMemEval-S, and LoCoMo are inaccessible.

V9 completed on all 1,492 tune episodes and passed the independent audit. Raw all-unit
maximum cosine achieved 31.84% top-1 with mean target rank 7.25. All-unit shortlist
recall rose from 34.52% at width 4 to 92.09% at width 16 and 100% at width 32. Raw
explicit-anchor cosine, every anchor shortlist, and learned full-list ranking were
100% top-1 at all four session lengths. The registered selector chose `anchor_4`.
Compact evidence and `REPORT.md` are archived under
`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/mquake_candidate_diagnostic_v9`.

Added `MQUAKE_ANCHOR_FORGETTING_V10_PLAN.md`,
`tasks/train_mquake_anchor_forgetting_v10.py`, its independent audit, runner, and gate
tests. V10 freezes the declared anchor-field shortlist and V8 listwise reranker, trains
only three new 50-epoch deletion heads on train causal states, and calibrates tune
thresholds without refitting. A six-check tune prerequisite conditionally releases a
nine-check entity-disjoint dev evaluation. All outputs are separate and baseline paths
remain unchanged.

V10 completed in 935.92 seconds at 7.867/8.115 GiB peak allocated/reserved memory and
passed its independent audit. Tune passed all prerequisites at 100% candidate top-1,
99.20% pair state, 100% current retention, and 98.42% non-target retention. The fixed
method then reached 99.67% candidate top-1, 96.19% stale deletion, 99.87% current
retention, and 96.12% pair state on entity-disjoint dev. Every length exceeded 95%
pair state and matched-random gain was +84.49 points, CI `[81.71,88.18]`. Non-target
and hard-retain safety were 91.80/91.81%, below the registered 95% floors, so the gate
failed and LoCoMo stayed sealed. Added the compact result and report under
`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/mquake_anchor_forgetting_v10`.

Added `MQUAKE_ROBUST_CALIBRATION_V11_PLAN.md`. It freezes the successful `anchor_4`
resolver and runs five-fold subject OOF gate calibration on MQuAKE-T train+tune with
stricter 99% calibration safety constraints and no refit. V10 dev will not be rescored
or used for option selection. Only an OOF pass may release one-time MQuAKE-CF-3k-v2
safety confirmation; LoCoMo remains a later confirmation set.

Implemented V11 in `tasks/train_mquake_robust_calibration_v11.py`, with
`scripts/run_mquake_robust_calibration_v11.sh`,
`scripts/audit_mquake_robust_calibration_v11.py`, and focused tests. Subjects are
assigned intact to five episode-count-balanced folds. Each outer fold trains three
fixed 50-epoch gates on three folds, calibrates on the next fold under 99% current,
non-target, and hard-retain constraints, and causally replays the untouched fold.
Calibration never refits. The run writes pooled OOF controller/candidate/random traces,
per-fold metrics and thresholds, clustered paired intervals, runtime, GPU peak, hashes,
and a cross-fit checkpoint. The auditor independently reconstructs fold membership,
threshold selection, metrics, gate decisions, matched deletion counts, and the absence
of V10 dev, MQuAKE-CF, LongMemEval-S, and LoCoMo access.

### MQuAKE-T V11 鲁棒校准完成与 V12 预注册

V11 在 hku-gpu2 完成五个主体互斥外折、每折三个种子训练、内层校准、完整因果回放和独立审计。总耗时 2784.49 秒，峰值 CUDA allocated/reserved 为 7.867/8.115 GiB；5976 个 OOF episode、239040 个动作，控制器与匹配随机各删除 6954 次。独立审计的指标重算、阈值选择、主体划分、删除数、原始输出哈希、来源 checkpoint 哈希和数据访问锁七项全部通过。

OOF candidate top-1 为 96.82%，current/non-target/hard-retain 为 100.00/99.05/99.51%，stale deletion 与 strict pair 均为 83.08%；五折 pair 为 83.28/85.62/79.68/96.91/69.88%。相对匹配随机的增益为 79.53 个百分点，主体聚类 95% CI `[72.69,86.49]`。安全门槛和随机对照通过，candidate、删除、pair 与最差折门槛失败；MQuAKE-CF、LongMemEval、LoCoMo 未访问。

新增 `scripts/analyze_mquake_robust_calibration_v11.py`，只读分析终态 OOF 轨迹，不调用模型、不选择阈值。5976 次 primary update 中：4912 次正确删除（82.20%）、832 次候选正确但阈值拒绝（13.92%）、61 次错误候选被删除（1.02%）、171 次错误候选被拒绝（2.86%）。oracle-memory candidate 只有 3/68 主体不完美；其中一个 700-episode 主体的 anchor-4 shortlist/top-1 均为 73.71%，主导 fold-0 失败。这证明 episode 频率不平衡与跨主体分数漂移是下一轮应处理的两个具体问题，不能将结果概括成候选或遗忘机制整体无效。

新增完整归档 `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/mquake_robust_calibration_v11/`，包含 REPORT、summary、training、manifest、cross-fit checkpoint、audit、diagnostic、日志、时间戳和终态；约 122 MiB 的原始 OOF JSONL 保留服务器并由 training.json 哈希锁定。新增 `MQUAKE_SUBJECT_RISK_V12_PLAN.md`：继续只用 MQuAKE-T train+tune，固定 anchor-16，在每个外折内部训练候选器，候选和 gate 均做主体等权优化；阈值通过主体聚类安全下界选择。V12 通过 OOF 门后才运行消融并一次性开放 MQuAKE-CF。

实现 V12：`tasks/train_mquake_subject_risk_v12.py` 在每个外折内部以三个 fit 折训练三种候选 seed，固定 anchor-16，再以主体总权重完全相等的 BCE 训练三个 gate seed。校准逐点完整因果回放 0.05–0.95、0.975、0.99、0.995，并以每主体 current/non-target/hard-retain 三者最小值计算宏平均和单侧 95% 正态下界；只在聚合安全≥97%、主体宏安全≥98.5%、下界≥97%的可行集合中最大化主体宏 pair。输出增加主体宏 candidate、pair、安全下界、每折指标、匹配随机、完整 checkpoint 与访问锁。

新增 `scripts/audit_mquake_subject_risk_v12.py`，独立重算 OOF/主体聚类/candidate/随机对照/门槛，复原每个阈值选择、主体折和身份覆盖，并核对删除数、原始行、来源、checkpoint、任务源文件 hash 与 benchmark 锁。新增 runner 和 3 项聚焦测试；与 V11 测试合计 5 项本地及服务器均通过，远程真实 PyTorch ranker/gate 小规模训练 smoke 通过。警告中的 requires-grad 张量转标量改为显式 detach，不改变 loss。

由于自动审批认为 GitHub 仓库公开范围未确认，包含实验结果和 checkpoint 的提交推送被拒绝；未绕过审核。V11 归档与 V12 计划已本地提交 `81b336e`，V12 源文件经 SSH 直接同步用于已授权远程实验。正式运行于 2026-10-13 13:32:15 +08:00 在 tmux `cbf_mquake_v12` 启动，root `/home/ctj/cbf_ttt_mquake_subject_risk_v12`，PID 995204；启动测试通过、Qwen 特征提取完成，MQuAKE-CF/LongMemEval/LoCoMo 未访问。

首版 V12 运行 32 分 54 秒仍未完成 fold 0，进程持续约 31 核计算、无 OOM。静态分析确认 anchor-16 的单个校准折约产生 75 万个唯一 pair，超过 `PairEvidence` 原 40 万 LRU 上限；22 阈值×3 seed 的顺序回放会反复驱逐并重算相同的确定性特征。该运行通过 tmux `C-c` 安全停止，完整保留为 `/home/ctj/cbf_ttt_mquake_subject_risk_v12_slow_incomplete_20261013`，未产生或冒充正式指标。

`PairEvidence` 新增可选 `cache_limit`，默认仍为 400000，所有旧调用和 baseline 数值/内存行为保持不变；V12 显式使用 1000000，并在 manifest、training 与独立审计中记录/核验 limit、entries、hits、misses、evictions。新增真实多-unit 数值等价测试，服务器 6 项测试通过，证明小/大缓存的 scope/payload feature 逐值相同，只改变 eviction。加速正式重跑于 2026-10-13 14:14:38 +08:00 启动，tmux `cbf_mquake_v12`、PID 997567；外部确认集仍封存。

### MQuAKE-T V12 完成与 V13 最终集成校准

V12 加速重跑在 hku-gpu2 完成并通过独立七项审计。5976 个主体互斥 OOF
episode 上，anchor-16 shortlist recall、candidate top-1 和主体宏 top-1 分别为
100.00%、98.93% 和 97.34%；current/non-target/hard-retain 为
99.82/99.51/99.72%。然而 stale deletion/strict pair 只有 37.18/37.01%，主体宏
pair 44.66%，五折 pair 为 13.38/20.32/46.15/93.06/12.08%。相对精确匹配删除
数的随机策略提升 34.84 个百分点，主体聚类 95% CI `[23.19,49.46]`。总耗时
10366.71 秒，峰值 CUDA allocated/reserved 7.867/8.115 GiB。

新增只读 `scripts/analyze_mquake_subject_risk_v12.py`。它确认 61.96% 更新已
选中正确旧记忆却被最终阈值拒绝，仅 0.35% 更新错误删除；部署分数 pooled
AUROC 0.9898，各折 0.9871--0.9968。primary-update 中位概率 0.346，而
hard-retain 中位数/90 分位仅 0.0021/0.0186。结果支持“最终阈值组合过度保守”
这一具体诊断，不支持将 V12 概括为表征没有学到遗忘。

新增 `MQUAKE_ENSEMBLE_CALIBRATION_V13_PLAN.md` 并锁定 V12 四个来源哈希。新增
`tasks/evaluate_mquake_ensemble_calibration_v13.py`：冻结每折 anchor-16 候选器和
三个 Gate，忽略旧的成员阈值，先对原始 Gate logit 取中位数，再在独立
calibration fold 上通过完整因果回放选择一次最终阈值。22 点低阈值扩展网格、
主体宏安全 98.5%、单侧 LCB 97% 和原 V12 release gate 均在运行前锁定；V10
dev、MQuAKE-CF、LongMemEval-S、LoCoMo 继续封存。

新增 `scripts/audit_mquake_ensemble_calibration_v13.py`，独立复算阈值选择、折划分、
OOF/主体指标、匹配随机区间、来源/checkpoint/原始输出哈希与数据访问锁；新增
runner 和四项纯 Python 测试，覆盖原始 logit 中位数、可行/不可行阈值选择以及
V12 来源哈希锁。所有改动是独立 opt-in 实验路径，不修改 baseline 的模型、
训练、优化器、数据或评测入口。

### MQuAKE-T V13 完成与 V14 在线相对 margin

V13 在 hku-gpu2 完成全部五折、matched-random 对照和独立审计，runner exit=0。
训练/评测耗时 4015.68 秒，完整 runner 从 17:43:34 到 18:50:47 +08:00；峰值
CUDA allocated/reserved 为 7.867/8.115 GiB。独立审计的指标、单次最终阈值、
主体折、身份、删除数、原始输出、V12 来源/checkpoint 哈希和访问锁七项全部通过。

V13 strict pair/stale 为 37.048/37.216%，current/non-target/hard-retain 为
99.799/99.493/99.699%，主体宏 pair 44.724%。五折阈值为
0.975/0.95/0.95/0.60/0.95，validation pair 为
13.21/20.32/46.15/93.39/12.08%。完整 release gate 失败，MQuAKE-CF、V10 dev、
LongMemEval-S、LoCoMo 均未访问。

新增 `scripts/analyze_mquake_ensemble_calibration_v13.py` 做只读 V12→V13 配对归因。
5976 个 episode 中仅 4 个 pair 从错变对、2 个从对变错，净增 2 个；V13 相对
V12 为 +0.033 个百分点，主体聚类 95% CI `[-0.050,0.145]`。primary update 正确
删除仅从 2191 增至 2192，安全动作略有回退。该结果否定“成员阈值组合是主要
瓶颈”的假设。

V13 raw-logit pooled AUROC 仍为 0.9933，各折 0.9871--0.9969。只读架构筛选显示，
减去同一 episode 历史分数中位数后，各折 AUROC 为 0.9875--0.9978，在 0.5%
经验 retain FPR 下 update recall 相对 raw score 在所有折提高，但现有 BCE Gate
仍只有 53.0--92.4%，因此不能把静态筛选冒充因果成功。完整 V13 归档和报告保存于
`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/mquake_ensemble_calibration_v13/`；
大体积 OOF JSONL 留在服务器并由 training hash 锁定。

新增 `MQUAKE_ONLINE_MARGIN_V14_PLAN.md`。V14 冻结 V12 fold-specific anchor-16
候选器，在每个外折的三个 fit 折训练三种 subject-equal episode-listwise rank-16
Gate；每个 episode 的唯一 update 与全部 retain 动作共同进入 listwise CE。成员
用 fit score 中位数/IQR 对齐尺度，部署分数减去当前 episode 全部历史动作分数的
中位数，只在 margin 超过已校准阈值时删除。规则不输入位置、长度、主体、未来
分数或正确性，首个 scored action 不能删除。

实现 `tasks/train_mquake_online_margin_v14.py`、独立审计、runner 和五项测试。审计
额外逐动作验证 `history_size == max(0, step-1)`，锁定历史仅含过去分数、每
episode 重置、无 future/position/identity feature。20 点 margin 网格通过完整
因果回放选择，使用原 cluster-risk 安全约束和 release gate。Mac 上 4 项通过、
1 项因无 torch 跳过；hku-gpu2 真实 PyTorch listwise 拟合和精确在线回放共 5 项
全部通过。所有功能仍为独立 opt-in 路径，baseline 不变。

### MQuAKE-T V14 完成、失败归因与 V15 jackknife 共识

V14 在 hku-gpu2 完成五折、匹配随机对照和独立八项审计，runner exit=0。正式
计算耗时 7458.63 秒，峰值 CUDA allocated/reserved 为 7.867/8.115 GiB。5976
个 OOF episode 上 strict pair/stale 为 49.866/50.000%，current/non-target/
hard-retain 为 99.866/99.850/99.833%，主体宏 pair 为 51.490%，安全 LCB95 为
97.301%。相对 matched-random 增益为 48.042 个百分点，主体 bootstrap 95% CI
`[37.468,59.046]`。五折 pair 为 46.405/43.144/63.545/94.732/1.342%，因此
完整 release gate 失败；外部集继续封存。

新增 `scripts/analyze_mquake_online_margin_v14.py`。它只在 OOF 结束后用锁定的原始
MQuAKE-T relation ID 做归因，不改变任何决策。7 个 relation 中约 99% episode 的
relation 在 fit 已有 500+ episode；用其他折同 relation 表现预测当前折只有
46.12% 准确率，否定 relation 稀缺解释。fold 4 update margin 中位数 3.011 而
阈值为 4.0；fold 3 中位数 3.663 而阈值为 2.5，定位为跨主体校准漂移。

新增 `scripts/screen_mquake_causal_normalizers_v14.py`，在不改变 V14 轨迹的前提下
筛查 max-gap、MAD/IQR/mean z-score、upper-IQR、percentile 和原 median margin。
MAD/IQR/mean 归一化显著降低安全召回；percentile 的非执行动作 FPR 约 4.8%。
原 median margin 在跨折阈值下仍最好：校准 FPR 0.25% 时 pair proxy 73.41%、
最差折 62.46%；FPR 0.5% 时 88.28%、最差折 81.80%。这些是固定轨迹架构筛查，
不冒充精确因果结果。V14 紧凑归档包含报告、checkpoint、独立 audit、relation
diagnostic 和 normalizer screen；原始 OOF JSONL 由 hash 锁定在服务器。

新增 `MQUAKE_JACKKNIFE_CONSENSUS_V15_PLAN.md` 与
`tasks/train_mquake_jackknife_consensus_v15.py`。每个 outer subject fold 内保留四个
inner calibration fold；每个 inner 模型用其余三折训练三个 listwise seed，在未见
inner 折上以非执行动作经验 FPR 0.05/0.1/0.25/0.5% 生成四个阈值，并通过完整
因果回放及原 cluster safety 约束选定阈值。outer 推理部署这四个未 refit 模型，
各自维护历史，至少 3/4 同意才删除共享候选。V15 不加入 relation、主体、位置、
长度或未来分数，保持 V12 candidate 与 V14 gate feature/loss 冻结。

新增 V15 独立 auditor、runner 与四项因果单测。审计将复算每个 inner 阈值、OOF/
主体/candidate/随机指标、fold 身份、删除数、来源和 raw hash，并逐动作核对四路
history size、threshold、vote count 与 3/4 删除规则。本地四项测试通过；服务器
与 V13/V14 组合共 14 项真实 PyTorch 测试通过。所有改动为 opt-in，baseline 模型、
训练、优化器、数据和评测默认路径不变。

### V15 运行期间冻结后续论文实验分支

在 V15 尚未产生任何 `inner_complete`/`outer_complete` 指标时新增
`POST_V15_PAPER_EXPERIMENT_PLAN.md`，避免根据 V15 部分结果临时选择后续路线。计划
明确 V15 使用的是已通过可读性门槛的 contextual additive Qwen memory；原论文
`ttt_conv`/`ttt_proj` writer 的 Correct-vs-Wrong/Zero/Random 内容门此前已失败，
因此不能把控制器通过直接表述为原生 dense fast-weight 成功。

条件分支为：V15 通过后先做真实 Qwen answer accuracy/NLL 的精确因果回放，与
no-memory、retain-all、global-reset、last-1/2、matched-random、V14 和 oracle
同输入比较；只有 query-level 增益、三个安全切片、所有外折和独立审计同时通过，
才一次性开放 MQuAKE-CF、LongMemEval-S、LoCoMo、MMLU、RULER。V15 失败则停止
listwise score + scalar threshold/normalizer/quorum 搜索，先检验 query-level causal
oracle 是否存在至少 5 点可控空间，再决定是否用真实 future-NLL 差构造训练标签。
原生 writer 作为独立桥接分支，需先通过单组件与 2/4/8/16 多组件内容门，之后才可
接入遗忘策略。新增内容仅为预注册文档与 README 链接，不改变正在运行的 V15、
baseline 默认配置、模型、优化器、数据或评测入口。

随后在 V15 仍未产生首个 inner/outer 指标时核对数据和读出代码：MQuAKE episode
只存 session/unit，现有 contextual memory 在 256 维候选空间评分，没有经过审计的
Qwen-vocabulary decoder。因此把下一阶段直接称为“Qwen answer accuracy”会越过证据。
计划补充固定的无训练 readout：将每个策略保留的原始 `Memory statement`/`Value`
按时间顺序序列化，使用 fresh KV 的冻结 Qwen 自回归评分 current/stale 完整答案；
它只证明文本证据选择的下游效用，仍不冒充 native fast-weight。远程只读统计覆盖
train+tune 全部 5,976 个 episode：完整未过滤文本为 1,202--6,137 tokens，64-session
最大 6,137；checkpoint `max_position_embeddings=40960`、无 sliding window，因此
协议可不截断地评分所有 retained units。正式入口仍需逐行检查加上 prompt/answer 后
不越界，并由 auditor 重建 prompt 和 token 长度。

同样在 V15 首个 outer 指标出现前锁定 Qwen outcome 的采样与查询：primary old/current
匹配单元提供两个固定 query phrasing 和 stale/current 候选；另取一个确定性的
same-relation hard-retain 单元与一个无关单元检查安全性。资源 pilot 为 5 folds ×
4 lengths × 每格 2 个 hash 最小 episode，共 40 条，只公开 oracle/no-memory 可读性、
耗时、显存和长度，V15-vs-fixed 指标保持遮蔽；oracle current-over-stale ≥80%、全行
有限且不越界才开放 formal。Formal 预定每个 `subject x length` 取 hash 最小 episode；
现有 train+tune 为 68 subjects × 4 lengths，预计 272 条，并按 subject 聚类 bootstrap。
episode ID 顺序需在首次 formal 模型评分前落盘并哈希。

实现 `tasks/build_mquake_qwen_outcome_v16.py` 与三项纯 Python 测试。构建器只加载
MQuAKE-T train+tune 和已哈希 `texts.json`，复用 V11--V15 完全相同的 subject-fold
分配；按冻结 hash 规则生成 `pilot.jsonl` 与 `formal.jsonl`。每行显式记录 primary
old/current、same-relation hard-retain、unrelated 四个 anchor unit 的原始 statement/
value 及 session 身份，并核验 update 指向 old。manifest 锁定来源文件、fold loads、
选择规则、行数和输出 hash，同时声明所有外部集未加载、模型未评分。本地三项测试
通过，覆盖 anchor 抽取、40 行 fold×length pilot 完整性/输入顺序不变性和每个
subject×length 唯一 formal 行。

首次远程构建在写出清单或模型评分前因少数 episode 没有 same-relation hard-retain
而停止，失败目录保留为
`/home/ctj/cbf_ttt_mquake_qwen_outcome_v16_failed_ineligible_safety_20261014`。只读
审计显示 5,976 条中 5,928 条至少有一个 same-relation hard，68×4=272 个
subject×length cell 中 252 个有此类 episode、20 个没有。由于原 corpus 明确将后者
标为 hard fallback，修订为每格优先 same-relation、缺失时使用标记 fallback，并分层
报告安全指标；pilot 仍只从 same-relation episode 抽取。该修订发生在任何 V15 outer
指标、Qwen 分数或方法对照可见前。测试增加 fallback 身份记录，总数变为四项。

修订后远程构建成功：pilot/formal 为 40/272 行，matched-hard 为 40/252，formal
fallback 为 20。三个 artifact SHA-256 分别为 manifest `6dbe6154...`, pilot
`e9a61d5f...`, formal `352c2cab...`；完整值归档于
`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/mquake_qwen_outcome_v16/RUN_STATUS.md`
及 `manifest.json`。本地同时保存两份有序 JSONL，使正式 episode 集合可独立核验。
当前 `model_scored=false`，Qwen outcome pilot 未启动。

在实现后续回放接口时发现首版清单的 unrelated safety unit 从整条 episode 选择，可能
位于 primary update 之后，违反因果可见性。模型仍未评分且 V15 尚无 outer 指标时，
将唯一查询时点固定为“primary current 提交后立即评分”，hard/unrelated 只能从
`sessions[:query_step]` 选择，并在每行记录 `query_step`；新增未来 distractor 排除
测试，构建测试共五项。首版成功清单在远程改名保留，修订版重新构建并产生新 hash，
不得混用旧清单。

因果可见性修订后的最终预评分清单保持 40/272 行与 40/252 matched-hard，SHA-256
更新为 manifest `9eea25d6aa253014d47553e0e45fb01107546a3ff14b86bc3881b6c60b208be1`、
pilot `4a66ff3e2adbc1c22466f876149855d62fbf2863a124ea8d292e2861359bbe38`、
formal `4c184db9568ad93e04c88f343bd3372d2cb3796d4035e4834938db9bbbaa9a11`。
本地归档已覆盖为修订版，RUN_STATUS 明确旧远程目录只作失败/修订历史，不能评分。

新增独立 `scripts/audit_mquake_qwen_outcome_v16_data.py`，不导入构建器选择函数，
从 train/tune/texts 重新计算 V11--V15 subject folds、matched-first hash 选择、40/272
有序 ID、query_step 可见性、old/current/hard/unrelated anchor 文本和值、matched-hard
计数、来源/产物 hash 及所有外部访问锁。新增两项 helper 测试；与构建器合计七项
本地通过。该 auditor 只验证预评分数据，不能替代后续 Qwen score/replay audit。

服务器七项测试通过；独立数据审计的 12 项检查全部为 true，`audit_passed=true`。
审计文件 SHA-256 为
`9840a50489a3cd9fe1dd7a39071af535b078b2aa4d3f172cf1a58f1d1375d744`，已归档至
V16 实验目录。审计再次确认 `model_scored=false`、`external_sets_loaded=false`。

新增 `tasks/evaluate_mquake_qwen_outcome_v16.py` 的 pilot-only 入口及五项测试。入口在
加载模型前强制要求 V15 `jackknife_consensus_supported=true`、终态完成且独立 audit
通过，因此 V15 失败时不能运行。Pilot 只暴露 `no_memory` 与 causal `oracle`，在
primary-current 提交后用 chronological 原始 unit 文本构造 fresh-KV prompt；两个
primary phrasing 比较 current/stale，hard/unrelated 各在 old/current/hard/unrelated
四个候选中评分。每个答案验证 tokenizer 前缀不变、总长度不超过 checkpoint limit、
NLL 有限；summary 只以 oracle primary preference≥80% 和完整性决定是否可进入
formal，不读取 V15/fixed 方法效果。与数据构建/审计合计 12 项本地测试通过。

服务器 12 项测试通过。使用真实 Qwen3 tokenizer 对 40 条 pilot、两种可读性臂和
四类 query 的 928 个候选逐一验证 prompt-prefix 边界；最大总长度为 oracle 4,509、
no-memory 63 tokens，均低于 40,960。结果归档为
`prompt_tokenization_audit.json`，明确 `model_scored=false`，不能当作 Qwen 效果。

### V16 双卡资源 Pilot runner 与独立结果审计（2026-10-14，模型评分前）

V15 运行至 outer-1 期间补齐 `scripts/run_mquake_qwen_outcome_v16_pilot.sh` 与
`scripts/audit_mquake_qwen_outcome_v16_pilot.py`，未启动 Qwen 评分。Runner 只有在
V15 终态、独立 audit 通过且 `jackknife_consensus_supported=true` 时才允许双卡各跑
20 条；否则在加载模型前停止。Formal 272 条和所有外部数据集不在该入口中。

`tasks/evaluate_mquake_qwen_outcome_v16.py` 现为每个 policy-query 标签单独记录耗时、
候选评分数、起始/峰值 allocated 与 reserved 显存；40×2×4 共 320 条资源记录，汇总
mean/median/p95/max latency 和两类峰值显存。独立 auditor 不调用 scorer 的聚合函数，
从 frozen sample、train/tune/texts 和 raw shard 重建因果可见 memory session、prompt
SHA-256、query/candidates、预测、正确性、answer NLL、token 上限、两卡奇偶 shard
覆盖、V15 依赖 hash、资源记录、汇总和外部集锁。新增三项审计 helper 测试；与现有
scorer/data tests 合计 15 项本地通过，Python 编译、shell 语法与 diff whitespace
检查通过。当前状态仍为 `model_scored=false`；这些改动只提高即将运行 Pilot 的
可审计性，不构成实验效果证据。

运行前代码复核发现需要显式锁住 4,096-token 边界：oracle prompt 最长 4,509，若走
默认原生 TTT cache，长 prompt 可能触发一次 native writer，而 no-memory prompt 不会。
实际 `CBFSession` 空 memory 读路径虽然不会 commit delta，但为消除实现解释歧义，
Pilot 改用仓库已验证的 `load_plain + PlainSession`：保留最终 checkpoint 的 slow Qwen
权重，将 `config.ttt_mode=false`，严格核验并排除 `ttt_conv/ttt_proj` adapter，每个
答案候选新建标准 DynamicCache。Shard audit 保存完整 slow-weight safetensors、config、
排除 adapter 和 tokenizer 文件 hash；独立 auditor 重新哈希并要求两卡 provenance
完全一致。测试总数更新为 16 项且全部通过。由此两臂唯一实验变量是序列化的 session
文本，不能将结果归因于 native fast-weight 更新。

远程 GPU1 随后完成不读取数据的 loader smoke：最终 checkpoint 以 plain 路径得到
4,022,468,096 个参数、`ttt_mode=false`、0 missing keys，且严格只有七层各一组
`ttt_conv/ttt_proj` 共 14 个预期 excluded/unexpected keys；RTX 5090 峰值
allocated/reserved 为 7.545/7.547 GiB。该检查没有加载 pilot row、没有计算答案 NLL，
所以 `model_scored=false` 保持不变。

新增 `scripts/chain_mquake_v15_to_v16_pilot.sh` 作为一次性 tmux 衔接器。它每 30 秒
只检查 V15 的终态、runner exit、固定 PID、summary 与独立 audit；进程连续三次缺失
且无终态会报错。只有 protocol/terminal/audit 均正确且
`jackknife_consensus_supported=true` 才调用 V16 Pilot runner。V15 正常结束但 gate
不支持时写入 `v16_not_launched_v15_unsupported` 并停止，不创建 Pilot 结果目录、
不加载模型。该 watcher 不改变 V15、V16 指标或阈值。

### V16 formal 九臂状态冻结器（2026-10-14，V15/Pilot 终态前）

在 V15 仅完成 outer-0/1、V16 尚未评分时实现
`tasks/build_mquake_qwen_outcome_v16_formal_states.py`。它只有在 V15 supported 且
独立 audit 通过、40 条 Pilot gate 与独立 audit 均通过后才运行；不加载 Qwen。
对 frozen 272 条 formal sample，在 primary-current 后重建并哈希九臂 memory session
IDs：no-memory、retain-all、global-reset、last-1、last-2、V15 deletion-count-matched
random、V14 OOF、V15 OOF、primary-old deletion oracle。V14/V15/random 都从完整 OOF
逐步 `deleted session_id` 因果回放，删除发生在当前 event admission 之前；任何未来
session、非可见删除、trace identity/step/event mismatch、当前 session 丢失或来源 raw
hash 变化都会拒绝。注册查询点下 global-reset 与 last-1 数学上相同，manifest 显式
记录此重复基线而不隐瞒。新增四项纯 Python 测试，结合现有 V16 tests 共 20 项通过；
正式状态尚未生成，`model_scored=false` 与外部集封存保持不变。

新增完全分离的 `scripts/audit_mquake_qwen_outcome_v16_formal_states.py`，不导入状态
builder 的 replay/policy 函数。它重新加载 5,976 条 train+tune episode、272 条 frozen
sample、V14/V15/random 三份完整 OOF trace，逐 event 独立执行 delete-before-admit，
并逐行逐策略比较有序 session IDs；同时检查 V14/V15/Pilot 终态和 audit、所有依赖与
raw OOF hash、完整 episode 覆盖、当前 session 保留、未来 session 排除、重复基线披露
和外部访问锁。新增三项独立 helper 测试，V16 相关纯 Python 测试合计 23 项通过。
Auditor 也只有在 formal states 实际生成后才运行；当前仍没有 formal 模型分数。

新增 `scripts/run_mquake_qwen_outcome_v16_formal_states.sh`：在独立新目录依次执行
builder/auditor 单测、272 条状态冻结、独立重建审计、exit/terminal/timestamp 记录。
默认输入固定为当前 MQuAKE-T、V14 OOF、V15 OOF、V16 sample 与 V16 Pilot 路径；
任何上游 gate/hash/audit 不符都会在写入完成终态前停止。该 runner 目前仅完成 shell
语法验证，没有启动，因 V15 与 Pilot 尚未满足解锁条件。

### V16 候选评分共享只读 prompt KV（2026-10-14，Pilot 前）

审查发现原 scorer 会为同一 query 的每个答案候选重复完整 1--4.5K-token prompt
前向，资源 Pilot 因而会高估 formal 成本。改为每个 policy-query 只前向一次 prompt，
随后为各 answer 克隆独立标准 DynamicCache branch：第一个答案 token 使用同一只读
prompt-last hidden，后续 tokens 只在各自 branch 前向；父 cache 长度前后必须相同。
每个不同 query/policy 仍从全新 cache 开始，不跨臂共享。Auditor 要求 shard metadata
明确记录此模式；新增候选前缀/重复候选测试，V16 tests 合计 24 项本地和远程通过。

新增固定容差 `scripts/check_mquake_qwen_outcome_v16_prefix.py`，只用合成问句和
Paris/London 两候选比较共享-prefix 与原逐候选 full-forward。首次远程调用因 CLI
`model/tokenizer` 名称未显式映射到函数参数而在加载模型前报错，失败目录保留；修复
CLI help/compile 后在全新 `_retry1` 目录运行。两候选 NLL 误差均为 0，预测一致，
通过预设 `2e-5` 门槛；耗时 0.328s、峰值 allocated/reserved 7.562/7.576 GiB。
归档 `prefix_scoring_bridge.json`，SHA-256
`a81c98a535ea6bd961395357d0b98df7a8055b209cc92ef0c948b74952ec6a49`。
脚本与结果均声明 `pilot_rows_loaded=0`、实验 `model_scored=false`、外部集未加载。
# 2026-10-14: V16 frozen-Qwen nine-arm formal outcome scorer (before outcomes)

While the registered V15 jackknife-consensus run was still computing its
fourth outer fold and before any V15 terminal result existed, the downstream
formal protocol was completed without inspecting formal outcomes.  Added
`tasks/evaluate_mquake_qwen_outcome_v16_formal.py`,
`scripts/audit_mquake_qwen_outcome_v16_formal.py`,
`scripts/run_mquake_qwen_outcome_v16_formal.sh` and focused unit tests.  The
runner is hard-gated by a passed/audited V15, passed/audited 40-episode V16
resource pilot and passed/audited 272-row frozen-state matrix.

The scorer evaluates no-memory, retain-all, global-reset, last-1, last-2,
deletion-count-matched random, V14, V15 and causal oracle on both primary
phrasings plus hard-retain and unrelated queries.  It uses the previously
bridged one-prompt candidate NLL path and now optionally performs an independent
greedy decode from a cloned prompt cache, reporting strict and conservative
case/whitespace-normalized exact match.  Within an episode, arms with identical
retained session IDs are scored once and copied exactly, with score provenance
recorded; this covers the registered global-reset/last-1 identity and any
additional coincident states without changing an arm's result.

The strongest fixed baseline is selected separately for each held-out outer
fold from the other four folds by primary candidate accuracy, lower NLL and
registered arm order.  V15-minus-fixed accuracy and fixed-minus-V15 NLL use
subject-cluster bootstrap with 20,000 draws (seeds 1611/1612).  Release still
requires both CI lower bounds above zero, current/unrelated/hard-retain accuracy
at least 97%, no negative outer-fold accuracy gain, finite in-context scores,
and an independent audit.  The auditor reconstructs every causal memory and
prompt, independently tokenizes candidates and generated IDs, verifies score
deduplication/model/tokenizer/resource provenance, and reimplements all policy,
fold, bootstrap and gate aggregation without importing the formal scorer.
External sets remain locked.  These additions are opt-in and do not alter the
baseline configuration or native `ttt_conv`/`ttt_proj` path.

Added `scripts/check_mquake_qwen_outcome_v16_generation.py`, a synthetic-only
real-checkpoint preflight that exercises candidate scoring plus greedy decoding
without loading any pilot/formal/external row.  The formal summary and auditor
require its passed artifact and exact hash; the formal runner executes it on
GPU 0 before launching the two scoring shards.

Remote commit `8c40ee9` passed all 22 V16 evaluator/state/auditor tests.  The
synthetic generation bridge then passed all eight checks on RTX 5090: greedy
output `Paris`, EOS stop, strict/normalized EM 1, Paris/London NLL
0.560188/7.685188, 0.43485 seconds, and 7.56597/7.58008 GiB peak
allocated/reserved.  The bridge loaded zero experiment rows and no external
set.  Its archived result is
`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/mquake_qwen_outcome_v16/generation_scoring_bridge.json`
with SHA-256 `f43dfb473789bfc999c1d72e40182320fd2455c3ce5aad2cc979cd68b63ec0d9`.

Added `scripts/chain_mquake_v16_pilot_to_formal.sh` to make the registered
gates operational rather than manual.  It waits for the existing V15-to-pilot
chain, exits cleanly when V15 is unsupported or the pilot gate fails, and only
then runs the audited formal-state freezer followed by the two-GPU formal Qwen
runner.  State/audit or runner failures produce distinct terminal markers; it
never launches an external benchmark.

Before V15 produced any terminal metric, added the conditional Branch-B state
freezer `tasks/build_mquake_qwen_headroom_v17_states.py` and focused tests.  It
can run only when V15 is terminal, independently audited, and explicitly
unsupported.  For the frozen 40-row pilot and 272-row formal samples it records
the five fixed arms and every causal single deletion of a visible historical
session, excluding current and future sessions.  The exact oracle tie-break,
cross-fold fixed comparator, resource-pilot prerequisite and five-point
subject-macro headroom gate are now frozen in
`POST_V15_PAPER_EXPERIMENT_PLAN.md`.  No Qwen row is scored by this builder.

Added `scripts/chain_mquake_v15_to_v17_states.sh`.  It observes the same V15
PID and terminal artifacts as the Branch-A watcher, exits without V17 work when
V15 is supported, and launches only the model-free V17 state runner when V15 is
audited and explicitly unsupported.  Missing/failed/invalid V15 states receive
distinct terminal markers.

V15 completed with a passed independent audit but failed the registered gate,
so Branch A correctly remained locked and the conditional V17 state runner
started.  Its builder wrote 40/272 rows with 1,200/8,160 exhaustive historical
single-deletion states and stable hashes, but the first independent audit
failed only `ordered_sample_coverage_and_identity`: it compared the iteration
order of a JSON object to the registered fixed-arm order even though the
builder deliberately uses `sort_keys=True`.  All state reconstructions, causal
checks, counts, hashes, dependencies, V15 gate and access locks passed.  The
failed root is preserved.  The auditor now treats JSON object order as
non-semantic, requires the exact key set with no extras, and still verifies the
registered order separately from the manifest.  A regression test covers
shuffled and extra keys; recovery must use a fresh `_retry1` root.

### MQuAKE-T V17 causal-headroom resource pilot

Added `tasks/evaluate_mquake_qwen_headroom_v17.py` to score the frozen 40-row
V17 pilot on two GPUs. For every episode it evaluates the five registered
fixed arms and all 30 causal single-session deletions. Identical fixed-memory
states are scored once and copied with an auditable source-arm identity. All
four registered query kinds use the final Qwen3-4B slow checkpoint with native
TTT adapters disabled, a fresh KV cache, and independent candidate-answer
branches. Each query records latency and CUDA start/peak allocated and
reserved memory.

The pilot constructs its exact per-episode oracle from `retain_all` plus every
single deletion, ordered by primary candidate accuracy, lower primary NLL,
retain-all, and chronological deletion order. The fixed comparator is selected
inside each outer fold using only the other folds. Pilot headroom remains
descriptive: only complete finite scoring, the 40,960-token limit, complete
resource records, peak reserved memory no greater than 31.5 GiB, a projected
two-GPU formal time no greater than 48 hours, and a passed independent audit
can release formal scoring. These thresholds were added to
`POST_V15_PAPER_EXPERIMENT_PLAN.md` before model scoring.

Added `scripts/audit_mquake_qwen_headroom_v17_pilot.py` to reconstruct sample,
state, arm, prompt, prediction, resource, model, tokenizer, shard and summary
identities independently. Added
`scripts/run_mquake_qwen_headroom_v17_pilot.sh` and focused unit tests for arm
ordering, the registered oracle tie-break, outer-fold fixed selection and
deterministic subject-cluster bootstrap. External benchmarks and the 272-row
formal set remain locked until this resource pilot completes and passes.

The V17 resource pilot completed on two RTX 5090 GPUs with runner exit code 0
and passed all ten independent audit checks. It scored 1,360 unique memory
states and 5,440 query labels. The two shards took 646.86 and 631.41 seconds;
mean/median/p95 query latency was 0.2160/0.2155/0.3401 seconds. Peak allocated
and reserved memory was 8.91/9.46 GiB, and the registered two-GPU formal
projection was 1.22 hours. All resource-gate checks passed. The frozen pilot's
descriptive oracle and outer-fold fixed comparator both achieved 100% primary
candidate accuracy, so the accuracy headroom estimate was 0.0 points (subject
cluster 95% interval `[0.0, 0.0]`). This is a ceiling-effect warning, not a
pilot efficacy stop, because the preregistered pilot releases formal scoring
solely on resources and integrity.

During the live pilot, row logs showed that episode history lengths differ.
The first independent auditor version incorrectly expected exactly 680 unique
states in each shard even though only the total 1,360 was registered. The
scorer was unaffected. The auditor was corrected before aggregation to derive
each shard's exact work from `pilot_states.jsonl`; a regression test covers
unequal shard workloads. The corrected audit passed.

Added `tasks/evaluate_mquake_qwen_headroom_v17_formal.py`, its independent
auditor, two-GPU runner and focused tests for the frozen 272-row formal stage.
Formal release requires the passed pilot terminal marker, runner code, summary
hash and independent audit plus the frozen formal-state and sample hashes. It
will score all 8,160 single deletions and fixed arms, select the exact causal
oracle by the registered tie-break, select fixed comparators using only other
outer folds, and apply the preregistered subject-macro gain >=5 points with
subject-cluster 95% interval lower bound >0. A failed formal gate records
`stop_controller_training_no_causal_headroom`; it cannot train another
controller or open external benchmarks.

Before reading any V17 formal outcome, added
`POST_V17_INTERFERENCE_CORPUS_PLAN.md` in response to the unsealed pilot fixed
arms. The pilot showed that the primary-only fixed selector chose
`global_reset` at 100% primary accuracy while that arm retained only 65.0% of
hard history and 52.5% of unrelated facts; `retain_all` was simultaneously at
the candidate-accuracy ceiling. The new preregistered controlled corpus uses
opaque unseen facts, balanced interference mechanisms, free generation and a
joint update-plus-retention episode endpoint. Its 40-row construction gate,
320-row source/template-disjoint formal gate, sparse `0/0.5/1` boundary study,
cluster bootstrap, controller admission rule and external-data lock are frozen
without access to the running 272-row V17 formal metrics.

### V17 formal terminal result

The 272-row formal exact-headroom run completed with exit code 0 and passed all
eleven independent audit checks. It evaluated 9,248 unique memory states and
36,992 query labels. The slower shard took 5,722.22 seconds; peak allocated and
reserved memory was 8.96/9.94 GiB. The summary and audit SHA-256 values are
`e96ab43c5e8e4ab4bd5bab5959784bd8bb1ab7d769747ffc0881ccadab346a20`
and `80cb31a1dde5aa19fb62a048fe3e8149401a41106b03f60dc74c04f76abee3ce`.

The registered subject-macro primary accuracy gain was exactly 0.0 points with
95% interval `[0.0,0.0]`: exact oracle and outer-fold `global_reset` both
achieved 100% primary candidate and pair accuracy in every fold. Oracle primary
NLL was worse by 0.0746 on the subject-macro scale. Both formal headroom checks
failed, so `causal_headroom_supported=false` and the current MQuAKE controller
line stops. No new controller or external benchmark was launched.

The complete matrix also quantified the endpoint defect anticipated from the
pilot. Exact selective deletion retained 100% hard history and 98.53% unrelated
facts, versus 70.59% and 55.15% for the primary-selected global reset. This is
reported as a safety trade-off and does not override the registered negative
primary result. Added the complete raw-shard archive, report and artifact hash
list under `experiments/.../mquake_qwen_headroom_v17_formal`.

### V18 controlled-interference pilot construction

Added `tasks/build_controlled_interference_v18.py` to materialize only the
preregistered 40-row pilot. It balances lengths 8/16/32/64 and clean update,
stale repetition, irrelevant near duplicate and useful repetition mechanisms.
Every episode uses pilot-only opaque entities, keys and values, six persistent
keys, one final target update, two update phrasings, two retained-key queries
and an inline-reference locality query. The formal prefix, value pool and
templates remain ungenerated and inaccessible.

The builder freezes no-memory, retain-all, global-reset, last-1 and last-2,
every causal single historical-session deletion, and every historical key
group deletion. Current-session deletion is prohibited. Added an independent
model-free auditor for split, balance, opacity, uniqueness, content binding,
query answers/candidates, mechanism-specific roles, state reconstruction,
hashes and access locks; focused tests exercise every mechanism, deterministic
rebuild, complete counts and audit pass. The build runner records code and
terminal provenance before any Qwen scoring.

Before model scoring, a read-only Qwen tokenizer audit found that the first
opaque hex format varied from 7 to 11 answer tokens. That root is preserved as
an invalid construction attempt and is not scored. Replaced values with a
spaced four-digit base-16 code (`Z A A A A` form); an exhaustive check of 400
codes produced exactly five tokens for every value. The builder and independent
auditor now freeze this format and the recovery uses a fresh
`_tokmatch_retry1` root.

Added `tasks/evaluate_controlled_interference_v18_pilot.py` and an independent
outcome auditor. Two GPUs score every unique fixed, single-deletion and
key-group-deletion memory state on five registered queries. Candidate branches
share only an immutable prompt cache; each query also performs independent
13-token greedy generation and records latency plus CUDA start/peak allocated
and reserved memory. The scorer rechecks all 320 opaque values and every
prompt-context answer at exactly five tokens before evaluation.

Pilot aggregation selects the exact oracle by joint update-plus-retention
success, update accuracy, retained-key accuracy, lower joint NLL, retain-all
and registered deletion order. The corpus admission checks are the frozen
no-memory, retain-all, reset-retention, oracle-update, oracle-retention and
oracle-minus-fixed joint-success bounds from
`POST_V17_INTERFERENCE_CORPUS_PLAN.md`. The independent auditor reconstructs
all prompts, arms, deduplicated score sources, predictions, generation flags,
resources, model/tokenizer provenance and admission metrics. Focused tests
cover a passing nontrivial corpus and reject a retain-all ceiling.

### V18 controlled-interference Qwen pilot terminal result

The two-GPU 40-episode pilot completed with exit code 0 and passed all twelve
independent audit checks.  It evaluated 1,435 unique memory states and 7,175
query measurements in 1,029.61 seconds on the slower RTX 5090 shard.  Peak
allocated/reserved memory was 8.30/8.75 GiB and mean query latency was 0.2694
seconds.  The summary and independent-audit SHA-256 values are
`3487fdf2fef70a9aa65276ffd81e9495992f8385bb0127ec955b99679f6fa0d2`
and `397b20382e4328c6f4bab0061cd2e5c495e302e4f6981419e24436c7162e2bf6`.

All six preregistered corpus gates passed.  The best fixed policy was
`retain_all`, with 75.0% candidate and free-generation joint success, 80.0%
update accuracy, 100% retained-key accuracy and 100% locality.  The exact
selective-deletion oracle reached 100% on joint success, free-generation joint
success, update accuracy, retained-key accuracy and locality, for a 25-point
joint gain.  `global_reset` reached 100% update accuracy but 0% retained-key
and joint success, directly exposing the update/retention trade-off that V17's
primary-only endpoint missed.

After pilot admission and before generating any formal row, froze the formal
design in `POST_V17_INTERFERENCE_CORPUS_PLAN.md`: 320 rows, 80 subjects, 20
formal-only template groups, five subject/template-disjoint outer folds and
20,000-draw subject and template cluster bootstraps.  Added the gated formal
builder, independent construction auditor, two-GPU scorer, independent outcome
auditor, runners and focused tests.  Formal code keeps native TTT adapters
disabled for the causal-headroom study, records candidate NLL plus independent
greedy generation and does not load external benchmarks.  Controller training
remains locked until the formal headroom gate passes.

The first formal-construction runner attempt exited before writing any corpus
row because the shell script created `result/` before invoking the builder,
while the builder correctly requires a nonexistent output directory. The
failed root is preserved. Removed the premature `mkdir`; recovery uses a new
`_retry1` root with the unchanged seed and registered protocol.

The `_retry1` formal construction completed with exit code 0 and passed all
eleven independent audit checks. It froze 320 episodes, 80 subjects, 20
template groups and five 64-row folds, balanced at 80 rows for each length and
mechanism. The state matrix contains 9,280 single-session and 1,920 key-group
deletions. Manifest, episode and state SHA-256 values are
`8312c232a10e8e7f66cb069bd4738c99dbc2dcfe57dd01297f65c6e80a129de6`,
`1975f8c5ca221d707f25665b8974cf1435eae63a62285a57ae8076efd0b365f2`
and `87b4727324282e6f608e642674d9e5521924bae454a384633cf148959f940192`.
The two-GPU formal scorer was then launched against this exact frozen root.

While the formal scorer was running and before any formal outcome was read,
added `POST_V18_FORGETTING_POLICY_PREREGISTRATION.md`. It separates the current
explicit-text causal diagnosis from the required low-rank Qwen fast-parameter
bridge, makes exact same-key conflict resolution a mandatory strong baseline,
freezes the three controller seeds and training grid, and requires held-out
semantic-alias improvement before natural benchmarks. This prevents a positive
text-memory oracle result from being overstated as native dense-writer or
learned-controller efficacy.

Added the gated Stage-B implementation:
`tasks/evaluate_v18_lowrank_fastmemory_bridge.py`, its independent auditor,
two-GPU runner and focused tests. Each observed session becomes five
position-specific Qwen rank-one key/value components, the query uses fresh KV,
and causal interventions remove those fast-parameter components rather than
editing prompt text. Scales 4/8/16 are scored together with scale 8 primary;
correct/wrong/zero single-component controls distinguish content binding from
a generic residual effect. Both collectors refuse to start unless the V18
formal summary and independent audit pass the frozen headroom gate.

Added `scripts/chain_v18_formal_to_lowrank_bridge.sh`. The detached watcher
observes the formal runner's terminal exit file, independently checks the
positive formal summary, audit hash and recomputed gate, and starts Stage B
only when every release condition is true. Failed or unsupported formal runs
produce terminal stop markers without creating a bridge result directory.

### Preregistered Stage-C physical conflict rules

While Stage A remained live and before reading its outcome, added
`tasks/evaluate_v18_physical_conflict_rules.py`, an independent reconstruction
auditor, a two-GPU runner and focused tests. This stage is hard-gated by the
positive independently audited Stage-B low-rank bridge, so it cannot be used
to rescue a failed causal-headroom or content-binding result.

The implementation causally suppresses an older component only when its value
disagrees with a newer component and either its literal entity-slot key matches
or its frozen Qwen-key cosine crosses a registered threshold. Agreeing repeats
are retained. A separate most-recent-same-key rule deliberately removes older
agreeing repeats as an ablation. Every rule preserves the current update.
Matched-random controls delete the same number of historical components using
seeds 1901/1907/1913.

The Qwen-cosine grid is frozen at 0.70/0.80/0.90/0.95/0.975/0.99. Threshold
and rule-family selection happen only on the training side of each outer fold,
using joint success, update, retention, locality and answer NLL in that order;
higher thresholds win exact ties. Collection records every component state,
pairwise key similarity, candidate score, independent greedy generation,
latency and peak CUDA memory. The auditor reconstructs every rule and random
state from the formal corpus, checks fresh-KV scores and bridge fixed/oracle
references, and independently recomputes all outer-fold results. Nine focused
tests currently pass together with the Stage-B tests.

Added `scripts/chain_v18_bridge_to_physical_rules.sh`. It waits on the actual
Stage-B runner exit handle, verifies the positive summary, independent audit,
summary hash and terminal marker, and starts Stage C only after all release
conditions pass. A failed or unsupported bridge produces an explicit terminal
stop marker and never creates the physical-rule result root.

### Stage-D causal deletion example construction

Before any Stage-A outcome was available, tightened the Stage-D policy to a
write-time historical-memory decision. The newest component is always written;
the controller has no current-update acceptance output and receives no query
text, answer, future loss, fold identity, episode length or absolute position.
The registered target is 0/0.5/1 for harmful/neutral/helpful deletion under
joint success. Exact-key group deletion supplies interaction credit to stale
repetitions when deleting any single member is insufficient.

Added `tasks/build_v18_causal_deletion_examples.py`, its independent auditor,
runner, chain and focused tests. Each historical component receives thirteen
normalized write-time features covering newest-event key cosine, exact key and
value agreement/conflict, relative age, repetition fractions and similarity
summaries. The builder joins the audited Stage-B single/key counterfactual arms
to the audited Stage-C key geometry and emits all 9,280 historical components.
The independent auditor separately reconstructs every feature, three-point
target, group credit, NLL tie evidence, fold count, identity and source hash.
The physical-to-example chain creates this dataset only after the physical
summary and independent audit pass.

Added the nested-OOF Stage-D trainer, independent audit, runner and release
chain. For each outer fold, the next fold is a fixed inner validation fold;
25/50/100 epochs and thresholds 0.4/0.5/0.6 are selected from ensemble
predictions of seeds 18031/18037/18043. Models are two-layer 64-unit GELU MLPs
with layer normalization. AdamW, learning rate 1e-3, weight decay 1e-4 and
gradient clipping at 1 follow the frozen protocol. The loss combines Huber
utility regression, causal positive-versus-conservation ranking, unrelated
memory conservation and useful-repetition consistency at weights
1.0/0.5/0.25/0.25.

The trainer saves all inner-fold ensemble scores, selection grids, three-seed
outer-fold checkpoints, learning curves and per-episode OOF retained/deleted
component states. Its auditor reloads every checkpoint, independently
recomputes inner selection and all 320 OOF actions, and verifies that the
current update is retained in every seed and episode. GPU training time and
peak allocated/reserved memory are recorded. Model-level policy efficacy is
still gated on the later Qwen rescoring stage.

Added the model-level OOF policy evaluator, independent auditor, two-GPU runner
and policy-to-evaluation release chain. The evaluator applies each seed's
retained component IDs to the frozen low-rank Qwen fast memory, uses fresh KV,
and scores candidate accuracy, answer NLL and independent greedy generation.
It compares the three-seed mean with the outer-fold fixed arm, strongest
physical rule, exact-match rule and exact oracle, including subject/template
cluster intervals, every-fold gains and fixed-to-oracle gap closure.

The controlled policy gate implements all registered exact-V18 criteria. The
summary keeps `learned_policy_claim_supported=false` until a separately built
semantic-alias set beats exact string matching; controlled-set success cannot
silently satisfy that requirement. The auditor reconstructs all seed-specific
memory states from the saved OOF policy artifact, rechecks every prompt, score,
generation and resource record, and independently recomputes the full gate.

### Frozen Stage-E semantic-alias confirmation

While Stage A was still running and before any controlled result was read,
added `POST_V18_SEMANTIC_ALIAS_CONFIRMATION_PLAN.md` and froze a gated 320-row
semantic-alias confirmation builder. Twenty latent-slot families each contain
three fixed surface forms. The primary historical fact, current update and
current query use different forms, so exact string matching cannot identify
the conflict while the latent causal graph remains known. Alias-only entities,
values, templates and folds are disjoint from pilot and formal V18.

The builder preserves the four lengths and mechanisms, 80 subjects, 20
template groups and five 64-row folds. It explicitly materializes retain-all,
global reset, exact-conflict and latent semantic-oracle states; exact conflict
deletes zero primary components by construction, while the oracle removes all
stale components of the latent target slot and always retains the newest
update. The independent auditor checks three-surface identities, query keys,
mechanism roles, states, split balance, source disjointness and exhaustive
five-token value encoding with the frozen Qwen tokenizer.

Construction is hard-gated on a positive independently audited controlled
policy evaluation. A sixth watcher stops without creating the alias root when
that gate fails. The registered alias claim requires a five-point gain over
exact matching with positive subject/template interval lower bounds, at least
90% update and retention, 97% locality, no negative fold or seed, and at least
40% exact-to-semantic-oracle gap closure.

A clearly marked `/tmp` integration smoke used a synthetic release marker only
to exercise construction before the real gate. With the real formal/pilot
corpora and Qwen3-4B tokenizer, all nine independent construction checks
passed: 320 rows, 80 subjects, 20 template groups, balanced 64-row folds,
five-token source-disjoint values, zero exact-rule deletions and 1,280 latent
semantic-oracle deletions. Manifest/data/state SHA-256 values were
`03047c86723ea27e92225a43b6e39f0bc12430e48a062c9e46ed2f3221dba9c3`,
`e82d7923d048e5db63787bdf1e0084c303897955049941e05b289cad585d3f4a` and
`e01f3c900c5387f4fec20af9dbb6717a9e66d2fac543ea740546779d1243e9b1`.
This smoke root is not an official result and is not consumed by any chain.

### V18 formal outcome and locality-constrained diagnosis

The 320-row V18 formal scorer completed with exit code 0 and passed all twelve
independent audit checks. Retain-all was selected in every outer fold and
reached 43.75% joint success, 56.72% update, 100% retention and 93.44%
locality. The unconstrained exact deletion oracle reached 90.63% joint,
94.84% update and 100% retention, with a 46.88-point joint gain and positive
subject/template interval lower bounds in every fold. Its 91.88% locality
failed the frozen absolute 97% gate, so the original V18 result remains failed
and Stage B was correctly not created.

Because the best fixed arm itself missed 97% locality, added the explicitly
post-hoc V18-LC Pareto diagnostic without rescoring the model. Dynamic
programming maximizes the registered endpoint under fixed 100/99/97/95/90%
locality budgets. At the primary 97% budget it used seven of nine allowed
failures and reached 90.63% joint, 100% update, 91.25% retention, 97.81%
locality and 73.13% generation joint success. Joint gain over fixed remained
46.88 points; subject/template 95% lower bounds were 41.56/41.25 points and all
five fold gains were positive. All eight diagnostic checks passed. This is
exploratory evidence for a safety-constrained objective and requires an
independent V19 confirmation; it does not change V18 or release Stage B.

The repaired independent V18-LC auditor was rerun on `hku-gpu2`. All four
source, coverage/hash, exact-DP recomputation and no-release-lock checks
passed. The immutable summary and audit SHA-256 values are
`54a87c57c65c3b36fc3fb13fef7b6c1c1e940cfc8765b5c507572c2f261ccc96`
and `abe9b2078d31fee137a60f0ad648dfe8b5189e3b11b35b6196c3b47d89050677`.
The completed diagnostic, logs, provenance and checksums are archived under
`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/`.

Five downstream V18 chain sessions had remained asleep while waiting for the
bridge that the failed original V18 gate correctly refused to create. Each
chain now has terminal status `stopped_upstream_formal_gate_failed`, exit code
0, completion time and an explicit stop reason; the corresponding tmux
sessions were closed. No model result or gate was changed.

### Preregistered V19 independent confirmation

Added `POST_V19_LOCALITY_CONFIRMATION_PLAN.md` before V19 model scoring. It
freezes the post-hoc-to-confirmatory transition: 320 new episodes, 80 subjects,
20 template groups, five outer folds, a sole primary 3% aggregate locality
budget (at most nine failures), 20,000 subject/template cluster bootstrap
draws, a ten-point minimum joint-success gain and the previously used
90%/90%/97% update/retention/locality safeguards. Failure stops the synthetic
claim and controller training; success releases only causal forgetting-example
construction.

Added `tasks/build_controlled_interference_v19.py`. Construction is hard-gated
on the passing audited V18-LC artifact while preserving
`stage_b_released=false`. V19 uses seed 1901, an unused opaque-value range
starting at 8192, new entity/key namespaces and new fact/query surfaces. It
reuses the causal state builder, keeps the current update undeletable and
materializes fixed, single-session and whole-key deletion arms. Added
`scripts/audit_controlled_interference_v19.py`, which independently checks the
release chain, all 320 episode/state identities, balance, nested folds,
mechanism semantics, action states, registered budget/gate and zero overlap
with both V18 corpora.

Added `tasks/evaluate_controlled_interference_v19.py` and the two-GPU runner.
The scorer reuses the frozen Qwen3-4B candidate/generation readout and records
per-query latency and peak memory. The primary exact dynamic program maximizes
joint success over all causal actions while enforcing at most nine locality
failures. Budgets 0/1/5/10% and the unconstrained oracle are diagnostic only.
The fixed comparator is chosen without the held-out outer fold. Added
`scripts/audit_controlled_interference_v19_outcome.py`, which independently
reconstructs prompts, arms, scores, generation, resources, model/tokenizer
provenance, the budgeted DP, cluster intervals and the complete gate.

Six V19 unit tests pass, including exact agreement between the production and
independent-audit DP tables. A real-dependency construction smoke using the
audited V18-LC release and both archived V18 corpora passed all eleven corpus
audit checks. Its deterministic manifest/data/state hashes are
`e620e75f1e73d1ec3f09510c784785a53c09a4209ce80fb767b259d7b0abbb6b`,
`384d94f24820bc11244d6d0c2525fc02887d456a066d68e0cbade0bcd7763cb9`
and `2459ca4c48d2a341e277b875150cf532736d6fbf5ccd7a4e91c736ee9ec0bf3f`.
This local smoke is not the official frozen remote result and is not used as
model evidence.

The first remote construction root exited with code 1 before producing data
because its runner still named a legacy V18 formal-data directory. The path was
corrected in a separate commit and the failed root was not reused. The official
`_retry1` construction then passed all three build tests and all eleven
independent corpus-audit checks. Its manifest/data/state hashes exactly match
the preregistration smoke hashes above. The official data, states, manifest,
audit, logs, provenance and checksums are archived in
`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/memory_interference_v19_data`.

The formal two-RTX-5090 scorer was launched only after that audit passed, using
remote code commit `ae99cfc7689cedb0210d5b3b8934edb363a4a480` and root
`/home/ctj/cbf_ttt_controlled_interference_v19_qwen_20261014`. Both shards
loaded the frozen Qwen3-4B checkpoint, entered model scoring and wrote their
first episode. Outcome aggregation, audit and any downstream release remain
pending until both shards terminate successfully.

### V19 causal-rule explanation frozen before outcome

While both V19 shards were still running and before reading any arm outcome,
added `POST_V19_CAUSAL_RULE_DIAGNOSTIC_PLAN.md`. The diagnostic asks whether a
fully causal exact-conflict rule already explains the registered oracle gain.
It freezes retain-all/reset/last-1/last-2, newest/oldest exact single deletion
and whole-key exact deletion. These rules see only session content through the
current update, never a query, answer, score or future event, and cannot delete
the current update.

Added `tasks/analyze_v19_causal_rules.py`, its independent auditor, runner and
two tests. The decision rule is fixed: locality at least 97%, positive
subject/template interval lower bounds, no negative fold and at least 90% of
the positive 3%-budget oracle gap closed. Passing means that V19 demonstrates
headroom but is too easy to justify controller learning, so the next study must
remove exact key identity with a source-disjoint semantic-alias V20. Failing
this explanation while V19 passes permits nested OOF controller training. The
production and independent recomputations match exactly on a 320-row fixture.

### Preregistered V20 semantic-alias headroom experiment

While the independent V19 scorer was still running and before reading any V19
arm outcome, added `POST_V20_SEMANTIC_ALIAS_HEADROOM_PLAN.md` and froze V20.
V20 is released only when the independently audited V19 causal-rule diagnostic
shows that whole-key exact deletion closes at least 90% of the positive V19
oracle gap. It then removes exact surface identity: historical facts use
surface A, the current update uses surface B and both update/query readouts
include a fresh surface C for the same latent slot. The primary comparison is
the latent semantic oracle against exact-string conflict deletion. This study
uses contextual evidence memory and makes no claim that the original dense
`ttt_conv/ttt_proj` writer learned a readable content-bound representation.

Added `tasks/build_controlled_interference_v20_alias.py` with 320 episodes, 80
subjects, 20 template groups, five folds, the four registered lengths and
mechanisms, seed 2003, and a fresh opaque-value range beginning at 16000. The
builder materializes the five fixed arms, an exact-string arm that removes
zero sessions by construction, and a latent semantic arm that removes every
stale slot-0 session while retaining the current update. Added
`scripts/audit_controlled_interference_v20_alias.py`, which independently
checks the V19 release, hashes, folds, balance, V19 disjointness, three-surface
identity, query semantics, causal states and the registered gate.

Added `tasks/evaluate_controlled_interference_v20_alias.py`. The two-GPU
scorer deduplicates identical memory states, verifies five-token opaque values,
records candidate likelihood, independent greedy generation, latency and peak
GPU memory, and selects the descriptive fixed comparator without its held-out
outer fold. The sole registered gate compares semantic deletion with exact
matching and requires a ten-point subject-macro joint gain, positive
subject/template clustered 95% lower bounds, at least 90% update and
retention, 97% locality, no negative fold, and exact matching to be bitwise
identical to retain-all. It cannot select a controller or threshold.

Added `scripts/audit_controlled_interference_v20_alias_outcome.py`, which
independently reconstructs all states, prompts, scores, predictions,
generation, resources, model/tokenizer provenance, outer-fold selection,
cluster intervals and the full gate. The build, scoring and result-audit
runners are connected by `scripts/chain_v19_rules_to_v20_alias.sh`; it stops
without creating V20 data whenever the audited V19 exact-rule explanation
gate does not pass. Six V20 tests pass, including exact equality between the
production and independent 20,000-draw bootstrap tables. The full V19/V20
14-test regression set passes. No V20 model result exists yet.

A 320-row integration smoke against the archived official V19 corpus and a
clearly synthetic upstream release marker passed all ten independent V20 data
checks. It confirmed 1,280 semantic deletions, zero exact-string deletions and
deterministic manifest/data/state hashes
`726550b3e145c2d9cbd3e0f3df6d44ca2bb1f6dea0ba7dede6b8f233cba0a2a0`,
`567cdcf69702bc02c51bc13d4266351393a28dbde6d16bb61af709507bee2787`
and `85b8bbaf5f4a241ae08d623a2ab5b831816074220edcbf0119bd8a32b03b762e`.
This smoke is not an experimental result and is not consumed by the chain.

### V19 nested-OOF policy fallback frozen before outcome

At 84/320 V19 scored rows and before reading any arm outcome, added
`POST_V19_NESTED_OOF_POLICY_FALLBACK_PLAN.md`. This branch is mutually
exclusive with V20: it is released only if the audited V19 confirmation passes
but the exact-key rule explanation fails. It freezes three seeds, a two-head
64-unit MLP, nested subject/template-disjoint selection, epoch/threshold/risk
penalty grids and the final model-level gate before labels exist.

Added `tasks/build_v19_nested_policy_examples.py`. Each historical component
receives a three-point causal utility label from retain-all versus its
single-deletion arm and, for exact conflicts, its registered group-deletion
arm. A separate target records locality risk. Its 17 inference features use
only history through the current update: exact key/value conflict, age,
repetition counts, character-key similarity, conflict context and memory
length. Query, answer, score, fold and oracle-selection features are excluded;
the current update is never a deletion candidate.

Added `scripts/audit_v19_nested_policy_examples.py`, which independently
recomputes string features, intervention choice, utility/risk labels, ordering,
coverage, hashes and access locks. Five focused tests pass, including exact
agreement between production and independent example reconstruction. The
runner and complementary automatic chain create examples only when the V19
rule diagnostic selects `train_v19_nested_oof_causal_policy`; otherwise the
chain records an explicit stopped terminal state.

Added `tasks/train_v19_nested_forgetting_policy.py` and its independent audit.
The shared 64-unit MLP predicts utility and locality risk; the registered score
subtracts a nested-selected risk penalty from utility. Three seeds, three
epochs, three thresholds and four penalties are selected on the fixed inner
fold under positive-recall, conservation and locality-risk constraints, then
refit without the held-out outer fold. The auditor reloads every checkpoint,
recomputes the complete inner grid and verifies all three-seed OOF memory
states and current-update retention. The complementary chain now trains and
audits this policy after an audited example release. Combined-state Qwen
rescoring remains conditional on that policy artifact and is required before
any efficacy claim.

Added `tasks/evaluate_v19_nested_forgetting_policy.py`, the two-GPU runner and
an independent model-outcome auditor. The evaluator applies each outer-fold
checkpoint's three OOF memory states, deduplicates identical states, and
rescales none of the model outputs. Every combined state is scored anew by the
frozen Qwen3-4B checkpoint with fresh KV. Comparators are reconstructed from
the immutable V19 raw rows: outer-fold fixed, causal whole-key exact deletion
and the registered 3% oracle. It reports three-seed means, subject/template
cluster intervals, every fold and seed, oracle-gap closure, generation,
latency and peak GPU memory.

The registered model gate requires five-point gain over fixed with positive
subject/template interval lower bounds, no negative fold or seed, 90% update
and retention, 97% locality, 40% oracle-gap closure, and a two-point gain over
whole-key exact deletion with positive subject-cluster lower bound. The
independent auditor reconstructs every policy memory state, prompt, score,
generation, resource record, model/tokenizer provenance, comparator, interval
and gate. Production and audit aggregation match exactly on a 320-row fixture.
The fallback chain now proceeds through model rescoring only after both label
and checkpoint audits pass.

### Post-V20 semantic-controller protocol frozen before outcomes

Added `POST_V20_SEMANTIC_CONTROLLER_NATURAL_TRANSFER_PLAN.md` while V19 was
still scoring and before V20 construction. It explicitly limits V20 to a
semantic-oracle headroom claim because its 20 alias families recur across
folds. V20 rows and latent IDs cannot train the learned controller.

The next learned stage uses already observed CounterFact/MQuAKE-T development
sources with relation- and subject-disjoint outer folds, subject/value-masked
frozen Qwen relation states, a rank-16 contrastive projection and a nested OOF
utility-risk gate. The source inventory, overlap matrix, model grids,
comparators and five-point/90%/97%/40%-gap release criteria are frozen. Only a
passing independently audited development chain may open one-time LoCoMo,
RULER/RULER-QA and MMLU confirmation. Observed LongMemEval-S remains diagnostic
and cannot tune or release the stage. The native dense-writer negative result
is preserved.

### V19 final archive, audit recovery and post-V19 decision

Archived the complete terminal V19 evidence in
`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/memory_interference_v19_final`.
The archive contains independently audited confirmation, exact-rule,
causal-label, nested-policy-training and model-evaluation summaries. Its
`REPORT.md` records the causal headroom result (96.25% oracle versus 39.38%
fixed), the exact-rule result (86.56%, 82.97% oracle-gap closure) and the
negative learned-over-exact result (86.67%, +0.10 points, subject-cluster 95%
CI [-1.56, 1.88]).

Fixed `scripts/audit_v19_nested_forgetting_policy_outcome.py` so shard resource
counts join policy states to immutable collection order by `episode_id`.
Policy states are serialized by outer fold and therefore cannot be counted by
their file position. Added an order-permutation regression test and
`scripts/finalize_v19_policy_eval_after_audit_order_fix.sh`. Local tests pass;
all six corresponding server tests pass with real PyTorch. The recovery did
not rescore Qwen. The scientific summary hash remains
`a5d7659ea585994ac8731742792bf4e4e324ab8523304c21e3f924abd012227d`,
the corrected independent audit passes all twelve checks, and the recovery
runner exits zero.

Frozen `POST_V19_NATURAL_SEMANTIC_HEADROOM_PLAN.md` before constructing or
scoring its data. It stops V19 tuning, preserves the audited negative native
dense-writer conclusion, and introduces a new relation/subject-disjoint
CounterFact/MQuAKE pilot plus sealed confirmation. The first gate tests a
relation-aware semantic deletion oracle against exact surface deletion. Only
a passing audited headroom result can release an exact-rule residual
controller; LoCoMo, RULER/RULER-QA and MMLU remain conditional one-time tests.

Implemented the data-only first stage in
`tasks/build_post_v19_natural_semantic_headroom.py`. It verifies the immutable
inventory hashes, filters malformed natural rows, selects 16 pilot and 64
formal targets per registered fold without subject/source-row overlap, and
constructs length 8/16/32/64 episodes covering single conflicts, stale
repetition, useful repetition and same-relation other-subject distractors.
Each episode preserves the current update and materializes fixed, exact-surface
and relation-aware semantic-oracle states. Exact matching is intentionally
relation-ID free and removes no paraphrased stale component.

Added `scripts/audit_post_v19_natural_semantic_headroom.py`, the corresponding
data runner and four focused tests. The auditor deterministically regenerates
all 400 episodes and states, checks source hashes, five-fold balance,
pilot/formal isolation, query candidates, current-update retention and both
causal state definitions. This stage does not load Qwen outcomes or any sealed
benchmark.

The first construction attempt stopped before producing a manifest because a
pilot relation had only one selected target and could not supply the required
same-relation other-subject distractor. The deterministic selector now chooses
targets in relation pairs inside each split; it does not relax the distractor
requirement. A fresh retry produced 80 pilot and 320 formal rows with 16/64 per
fold, equal length and mechanism counts, 36 relations, zero exact-surface
deletions and 488/1,952 semantic deletions. All nine independent data checks
pass. Manifest/audit SHA256 values are
`d5cecdde72c2080c2c9eef0b3a52cb8cb23197ce2affde4372aa55205f903185`
and `91fa2276adc31c95331d0fd6cdec579e0b2a7b277c0790f95f0deae4883bb3d0`.

Added the variable-length Qwen scorer, outcome auditor, pilot runner and
aggregation tests. The scorer accepts the observed 1--10-token natural values,
uses 16-token free generation, deduplicates equivalent causal states and keeps
formal scoring locked behind an audited passing pilot. The outcome auditor
reconstructs prompt hashes, state sharing, predictions, resource counts,
clustered intervals and the complete scientific gate. The longest pilot
retain-all prompt is 1,826 tokens.

### Natural semantic headroom model results

Ran the 80-row pilot on two RTX 5090 GPUs. One initial attempt stopped before a
completed row because a one-token candidate entered the inherited multi-token
branch and produced an empty float input tensor. The scorer now uses the
prompt's final hidden state directly for one-token answers. The successful
model scoring completed all rows. Its summary passed every scientific gate;
an auditor-only missing `QUERY_KINDS` import was repaired without rescoring,
and the recovery audit passes all eight checks.

The pilot semantic oracle reached 95.00% joint success versus 51.25% for exact
surface deletion, with subject-cluster 95% CI [32.50, 55.00] points and every
fold positive. The pilot MQuAKE stratum contained only seven rows and had zero
gain, so the conclusion remained provisional.

The audited pilot release unlocked the previously sealed 320-row confirmation.
The formal semantic oracle reached 92.19% versus 36.56% for exact/retain-all
and 48.75% for the outer-fold fixed comparator. Semantic-minus-exact gain was
55.63 points; subject and relation cluster intervals were [50.00, 61.25] and
[49.17, 61.74] points. All folds were positive, and CounterFact/MQuAKE gains
were 56.86/38.10 points. Update, retention and locality were
95.31%/99.22%/100%. All ten scientific and eight independent audit checks
passed. `experiments/cbf_ttt/qwen3_4b_final_1b_20260927/natural_semantic_headroom_v1`
archives the data, pilot and formal terminal summaries plus a full report.

Frozen `POST_NATURAL_HEADROOM_SEMANTIC_RESIDUAL_PLAN.md` after the audited
headroom release and before learned-stage embedding extraction. It defines a
relation/subject-disjoint pair corpus, frozen-Qwen masked-surface embeddings,
rank-16 contrastive projection, nested three-seed utility/risk controller,
exact-rule residual inference and a model-level release gate. Relation IDs are
training labels only and are forbidden at inference. Failed inner constraints
or a failure to exceed projected cosine stops the branch before natural
benchmarks.

### Learned natural-semantic residual: pair corpus and representation stages

Added `tasks/build_natural_semantic_relation_pairs.py`, its independent audit,
runner and tests. The frozen CounterFact/MQuAKE inventory yields 63,782
balanced relation pairs over 15,198 unique masked surfaces: within-row and
cross-row positives plus lexical-hard and random negatives. Relations remain
isolated across five folds and relation IDs are training labels only. The
audited release contains 31,886 positives, 31,896 negatives and at least 500
examples of each class in every fold. The pair manifest and audit SHA256 values
are `66b21a27d437f250d66f0b3daee6dc3869c89954c1f211d5093c600c06162d01`
and `09f63610747dddf23b9aae0ba1a86d9cebc9e7971f6270ac9bec2e900164eb1a`.

Added two-GPU frozen-Qwen surface extraction in
`tasks/extract_natural_semantic_surface_embeddings.py`, a full artifact audit
with fixed-sample model re-encoding, and recovery logic that distinguishes
strict same-batch reproduction from BF16 padding-shape stability. All 15,198
surfaces fit the registered 256-token limit; the maximum is 164 tokens. The
two RTX 5090 shards each took about 20 seconds and peaked at 7.73/7.94 GiB
allocated memory. Every vector is finite, FP32 and unit normalized. Same-batch
re-encoding has zero maximum absolute error; a deliberately different padding
shape has minimum cosine 0.99992. All eight audit checks pass; the audit SHA256
is `4ff61768ac749b6479f9209acb39b04463fb25b6e48a7d3aa811ca223e981649`.

Added a nested-OOF raw-cosine baseline and an efficient exact threshold search.
Mean-pooled Qwen embeddings are strongly anisotropic for this task: OOF AUROC
is 0.3933, balanced accuracy 54.10% and positive recall 11.97%; no inner fold
satisfies the registered recall/specificity constraints. A rank-16 symmetric
InfoNCE projection with the registered hard-negative margin improves AUROC to
0.6643 and balanced accuracy to 63.33%, but lexical-hard-negative specificity
is only 34.45%. It therefore fails the frozen projection diagnostic and is not
treated as a learned-controller result.

Added a leakage-safe geometry diagnostic. It discovered that 19/15,198 masked
surface strings occur in more than one relation fold, then restricted all
normalization statistics to surfaces whose complete fold ownership is within
the three training folds. Train-only centering improves balanced accuracy by
only 0.14 points and diagonal standardization does not help. The overlap rate
is 0.125%, so neither anisotropy correction nor cross-fold duplicate strings
explain the projection failure.

Added audited multiview extraction for global mean, last token, last-eight-token
mean and an eight-token subject-context window. Two source surfaces lack the
registered `[SUBJECT]` marker; they use an explicit last-eight-token fallback
and affect only five pairs. The independent audit registers exactly these two
fallbacks, reproduces all four views with zero maximum absolute error, and
passes all eight checks. Extraction takes about 22 seconds per RTX 5090 shard;
the audit SHA256 is
`ea817bab098ce41446309fa79a76f5eb4cc6afa6037a78c481f775e8057f28fb`.

Nested inner-fold view selection chooses last-token in all five folds. Its raw
OOF balanced accuracy/AUROC are 61.15%/0.6596; excluding the five marker-fallback
pairs changes balanced accuracy by 0.0015 points. Training the unchanged
rank-16 projection on last-token features raises these to 69.94%/0.7754, with
all three seeds above 0.75 AUROC. The result remains below the registered
constraint because lexical-hard-negative specificity is 50.71% and folds 1
and 4 are weak.

Added `tasks/train_natural_semantic_residual_controller.py`, which implements
the registered shared 64-unit utility/risk heads with five-fold nested
selection, three seeds and the fixed threshold/risk-penalty grid. Inputs are
four inference-safe cosine views, projected last-token cosine and character
trigram similarity. It reads no model outcome or sealed benchmark. The
controller improves OOF balanced accuracy/AUROC to 74.08%/0.8112, and all
three seeds exceed 73% balanced accuracy. It still fails the safety constraint:
negative specificity is 73.55% overall and 61.20% on lexical hard negatives;
no inner fold reaches 80% recall together with 95% specificity and at most 3%
risk violation. The 320 model-level outcomes remain unopened for controller
selection.

Added a symmetric frozen-Qwen pair cross-encoder to address the remaining
interaction error. It encodes both statement orders, averages their normalized
last-token states and records a one-token zero-shot yes-minus-no margin. The
63,782-pair extraction took 192/191 seconds on two RTX 5090 cards and peaked at
8.02/7.82 GiB allocated memory. All seven feature-audit checks pass with exact
fixed-sample vector and margin reproduction; audit SHA256 is
`b0ff99e28a13e8744527b86ccf9b70a20504e9b1e6ccbf403effb3fd482cfcc3`.
The zero-shot margin reached 64.03% balanced accuracy and 0.6798 AUROC.

The nested three-seed cross-feature controller reached 83.56% OOF balanced
accuracy, 0.8796 AUROC, 86.77% positive recall and 80.35% specificity. Every
fold exceeded 0.76 balanced accuracy and every seed exceeded 0.833. This is a
9.48-point balanced-accuracy and 0.0685 AUROC improvement over the independent
embedding controller. The registered safety gate nevertheless failed:
lexical-hard-negative specificity was 72.89%, and no inner seed/ensemble
selection satisfied the frozen recall/specificity/risk constraints. An
independent auditor reconstructed all 63,782 OOF rows, aggregate/fold/kind
metrics, gate logic, dependencies and checkpoint hashes; all 20 audit checks
pass and its SHA256 is
`30d793dec40f0480c4852a9e33d354f9611648fb8fa933ff85c2ec5b046123b7`.
The registered stop on CounterFact/MQuAKE controller tuning is therefore in
force. Their 320 model outcomes and external benchmarks remain unused.

### Source-disjoint FewRel selective-deletion development

Added `FEWREL_SELECTIVE_FORGETTING_PLAN.md` before any FewRel feature or model
score was produced. It moves development to the official THU FewRel commit
`278a2315d2138810a379cd8d5718914dc56e2582`, separates 48/8/8 training,
development and internal-test relations, and reserves all 16 official
validation relations as external test. Every subject/head entity and masked
surface is partition-disjoint. Relation IDs are supervision/audit metadata
only. `scripts/download_fewrel_official.sh` pins the repository and the three
source file hashes.

The first model-free corpus build revealed a protocol implementation mismatch:
the plan specified SHA256 of `2311:relation:<id>`, while the generic stable-hash
helper inserted `:.:` separators. No model features or scores had been
computed. That build is excluded. The corrected builder uses the exact
registered byte string, and its strengthened auditor independently recomputes
the relation partition from the official files. The corrected manifest/audit
SHA256 values are
`6f1dc9c9bacc934c5e44693b59a841220e18772ef0250eb64a5afb76aee77a1d`
and `960e706573916846fa46194bc6323dea1756a15be0cb4dc87d0e99d32ced7a3f`;
all 16 audit checks pass.

The release retains 24,000 instances, exactly 300 per relation, and constructs
95,354 class-balanced pairs. Train/development contain 57,104/9,554 pairs;
internal/external test contain 9,570/19,126 pairs and remain unscored. Excluded
rows and reasons are fully recorded, including 3,678 cross-partition subject
rows, 485 duplicate surfaces, 22 ambiguous/cross-partition surfaces and 27,815
relation-cap rows.

Added a train/development-only two-GPU cross-feature extractor. It encodes
66,658 pairs and writes no internal/external test feature. The two RTX 5090
shards took 367/363 seconds and peaked at 7.91/7.92 GiB allocated memory.
Prompt lengths are 44--154 tokens. All nine audit checks pass; eight independent
re-encodings have minimum cosine 0.99999994 and zero maximum absolute vector
or margin error. Feature-audit SHA256 is
`9fd6a12b5a1c28f676aa1febb44af30c9dcd654c6b3941afaaf1f2c79a396857`.
The next step trains three fixed controller seeds, reports full binary and
conservative selective-deletion gates, and opens internal-test features once
only if a preregistered development gate passes.

That controller development completed in 5.35 seconds. The three selected
25-epoch heads have development AUROC 0.7621--0.7637. Their ensemble reaches
70.76% balanced accuracy and 0.7680 AUROC, versus 55.59%/0.5856 for the
zero-shot margin and 50.00%/0.3894 for character trigrams. The full binary
gate fails every registered check: recall/specificity are 72.20%/69.31%, well
below the 80%/95% constraint. At a relation-bootstrap upper false-deletion
bound of zero, the selective ensemble can delete only 9/4,777 positives
(0.19% recall), far below the 25% floor. All seeds fail similarly. Internal
and external test features were neither generated nor scored, and the branch
stops as registered.

The independent controller audit reconstructs all 9,554 development rows,
full metrics, selective actions, 20,000-draw relation bootstrap, baselines,
gate logic, feature/checkpoint hashes and sealing state. All 15 checks pass;
the summary and audit SHA256 values are
`ff5a36e551a3155ed3877fc97f80d54d1bc4f543153e3895e475cecb7674c129`
and `9b257ba36a7df6f9489940866eeb394c79949b31010c5dd30434d8a9859916fa`.
A threshold-free diagnostic shows relation-macro balanced accuracy/AUROC of
70.75%/0.7702, ranging from 61.33%--82.71% and 0.6595--0.9024 across the eight
relations. Random and lexical-hard negatives have nearly identical
specificity (69.33%/69.48%), so the failure is broad unseen-relation separation
rather than a lexical-negative-only artifact. Same-subject negatives number
only eight and cannot support a separate claim. Diagnostic SHA256 is
`8e24b5b8c505dd1dd21fb9888c931633b2cbe8ddf515d59f41fed9030271ca0e`.

### DocRED relation-description representation branch

Added `DOCRED_RELATION_SPACE_PLAN.md` before any DocRED model feature was
extracted. The official DocRED train/dev/relationship files are hash-pinned;
all relations assigned to either FewRel test partition are excluded before
eligibility. SHA256 assignment of 67 eligible relations creates 51
representation-training, eight development and eight sealed source-test
relations. Instances use evidence-only sentences with `[SUBJECT]`/`[VALUE]`
masking. The first two build attempts are excluded: one exposed that 392
overlapping entity mentions must be rejected as specified, and one exposed an
implementation-only post-filter training floor absent from the frozen plan.
Neither attempt wrote a manifest or any feature. The corrected release at
`/home/ctj/cbf_ttt_docred_relation_space_data_20261015_retry2` has manifest
SHA256 `cc6731d694a988fcf7e79af54dab89319436ccfa81181de85e5665c76a928cd0`
and audit SHA256
`ddfc013b4dc635057c5dafeecc397aaba4a476878140bbf4aa5f76b36feabeff`;
all 18 data checks pass.

The corpus contains 10,952/492/356 train/development/source-test instances and
43,598/1,922/1,390 exactly balanced pairs. Only 51 training relation
descriptions are published; instance rows contain no description field.
Source-test features and scores remain absent. The build took 216 seconds.

Added a two-GPU frozen-Qwen last-token extractor for train/development
statements and training descriptions only. It produced 11,495 unit-normalized
vectors (11,444 statements and 51 descriptions) in 54--55 seconds per RTX
5090 shard, with peak allocated memory 8.28--8.29 GiB and token lengths 8--313.
All 11 independent feature checks pass; fixed-sample same-batch reproduction
has minimum cosine 0.99999970 and zero maximum absolute error. Feature-audit
SHA256 is
`8170b5908d6659565bb8a99e71e4f004dff7675f02af0771c8d499980d44a5bc`.

Added the frozen three-seed rank-64 statement/description projection with
relation-balanced symmetric InfoNCE, within-relation contrastive loss and a
hard-negative margin. The 230-second development run improved balanced
accuracy by 8.48 points over raw cosine (63.37% versus 54.89%) but reached only
0.6564 AUROC; all three seed AUROCs are 0.6474--0.6563. P31 and P527 AUROCs are
0.4666/0.4193, lexical-hard-negative specificity is 46.91%, and full/selective
safety gates fail. The registered branch therefore stops without generating
source-test or FewRel-test features. Independent reconstruction passes all 11
checks. Summary/checkpoint/prediction/audit SHA256 values are
`35114445a1f9994eeb67371d29c0cc7154afe57e1291561914d8fa4421e665d6`,
`586398646e98744ed97e677098fcb60291ba1d053fa301d69507887aa01278e0`,
`e8e3ac25e57b5d2e55b2e56b1c02a22afeaa65299e8e89b5331abb9e81f61652`
and `89624d89f7822498af7e4594598b90931af6988c5c09b0c1d3da9f732c09ff77`.

### Qwen continuous cohort forgetting and sparse-reader recovery

Added `QWEN_COHORT_FORGETTING_PLAN.md`,
`tasks/train_qwen_cohort_forgetting.py`, its runner, focused tests and an
independent row-level auditor. The experiment keeps the current write and
learns three continuous retention coefficients for history, last-2 and last-1
from 27-grid counterfactual labels. A fresh 128/32/64-group corpus and three
projection seeds were used. The controller reached 88.35% development
accuracy versus 66.41% for retain-all, a 21.94-point gain with 95% CI
`[20.15,23.80]`; NLL gain was 0.7271 with CI `[0.6822,0.7752]`. It recovered
96.9% of dense-reader oracle headroom, but failed the unchanged absolute role,
family and retention-MAE gates. Physics/exact deletion reached 88.54% and the
oracle only 89.06%, exposing a reader ceiling. Test remained unencoded. All 13
audit checks pass; summary/training/manifest/checkpoint/audit SHA256 values are
`b8de681ae8ff62af0eb2a974724a2a6555d18c1b8c42bfafae39847e65591f67`,
`33ad664dd628df8ee3b3c68d1dcb3eac6d1fd96a33d6a4a865e01c76ddcf8d47`,
`8d3446d6959686e51b7ae1c5da4b1e3bc44d58ac2ee73fa0cba75fb4b24a732e`,
`bd96395c622f4747625e8dbfaf35da2dc5eb77b9c52342a92dad4a61c77c6d5b`
and `92949bdef771e25ad6509a3ee94e292ddacdcb37cdc116809fd7dc17abdd949f`.

Added `QWEN_SPARSE_READER_DIAGNOSTIC_PLAN.md`,
`tasks/diagnose_qwen_sparse_reader.py`, its GPU-selectable runner, six tests
and `scripts/audit_qwen_sparse_reader.py`. Two attempts are retained but
excluded: the first used an algebraically equivalent floating reduction order
that changed near-tied oracle actions, and retry1 extracted development names
without the registered train+development batching. Retry2 exactly matches all
4,608 dense seed-2301 queries, actions and NLL values. Across three seeds,
top-1 key routing raises oracle accuracy from 89.06% to 99.33%, historical
accuracy from 67.90% to 98.00%, and minimum family accuracy from 71.82% to
97.34%. Top-2 reaches 97.12%; exact-key upper reaches 100%. Top-1 is the
registered selection. All 11 audit checks pass; summary/training/manifest/audit
SHA256 values are
`c281b79253dd7ab13bf4a30467a751af3f3985e4bd2729a9176b5a0552153836`,
`f106805b049db2237e4623209d051479d9e1abea8e601c8dd110702d781653c1`,
`ace92f1ef2293ec39ca38b9f17fb705eebed1fe91c00c143bd75b8289a8545db`
and `684f4e2b9e99074f3daf08b08b605a05cbe6d78818eca09585f7bb6f7066256e`.

Added and froze `QWEN_SPARSE_COHORT_FORMAL_PLAN.md` before its fresh corpus,
plus `tasks/train_qwen_sparse_cohort_forgetting.py`, its runner, gate tests and
an independent auditor. It uses seed `202610152`, 256/64/128 source groups and
the audited top-1 reader. The primary release test requires at least 97%
accuracy, a three-point gain over physics with positive paired intervals,
every-seed improvement, at most two points to oracle, and strict role/family
floors. A 24-episode end-to-end smoke covers label construction, controller
training and fixed/learned/physics/oracle replay. The formal run is recorded
separately and may access test once only if every development gate passes.

### Conditional discrete cohort-state confirmation

The first completed predecessor projection showed a qualitative mismatch
between continuous retention and the sparse reader: the continuous controller
reached 83.30% despite low retention MAE, whereas an already registered
27-class grid controller reached 98.86%.  A small nonzero coefficient leaves a
component active under top-1 routing, so exact zero is a physical state rather
than an interchangeable approximation.  Before observing predecessor seeds
2302/2303, `QWEN_DISCRETE_COHORT_CONFIRMATION_PLAN.md` froze a conditional
confirmation on corpus seed `202610153` with 256/64/128 source groups.  It may
run only if the independently audited predecessor keeps test sealed and the
grid controller meets four fixed accuracy, per-seed, physics-gain and oracle-gap
prerequisites.

Added `tasks/train_qwen_discrete_cohort_confirmation.py` and
`scripts/run_qwen_discrete_cohort_confirmation.sh`.  The primary controller
selects one of the 27 exact `{0, 0.5, 1}^3` cohort actions.  Comparators include
the continuous controller, its independently snapped output, a reference-blind
controller, fixed/random policies, explicit physics, exact-reference deletion
and the causal grid oracle.  Development must pass all 12 registered checks
before test identifiers are loaded or encoded.  The implementation passed a
24-episode remote end-to-end smoke and four focused local/remote tests.

Added `scripts/audit_qwen_discrete_cohort_confirmation.py`, which independently
reconstructs source views, every row and aggregate, bootstrap comparisons,
gates, checkpoint structure, dependency hashes and test sealing.  Gate parity
tests require the training and audit implementations to return identical
decisions.  Added `scripts/chain_sparse_to_discrete_confirmation.sh` to wait for
the predecessor's real EXIT-trap status, audit it, recompute the frozen
prerequisites and launch the confirmation on GPU 1 only if all prerequisites
pass; the confirmation is independently audited after completion.  The chain
records each exit code and cannot relax a failed prerequisite.  Baseline
training and inference defaults remain unchanged.

Before the confirmation corpus was generated, the task also gained an atomic
`result/progress.json`.  It records completed projection seeds, the current
policy, test-loading state and terminal gate state without changing any model,
data, optimizer, action, metric or release decision.  This operational marker
lets long counterfactual-label phases be distinguished from a stopped process;
the independent scientific audit deliberately ignores it.

Added `scripts/summarize_qwen_sparse_action_diagnostic.py` and two focused
tests.  After all predecessor seeds close, the chain now hashes and summarizes
the complete grid-controller development rows by exact endpoint versus
interior-oracle action, reporting action accuracy, query accuracy, retention
MAE and label distributions.  The artifact is explicitly post-hoc and is
generated after the integrity audit but before the conditional prerequisite;
it is never read by either release gate.

Before the discrete confirmation corpus was generated, added the conditional
`QWEN_DISCRETE_COHORT_OOD_PLAN.md`.  If and only if the confirmation's
independently audited one-time test passes, it evaluates the frozen controller
without retraining on four freshly generated lifecycle shifts: longer sessions,
distant revocations, mixed reset/revoke streams and duplicate/no-op writes.
The suites, 64-group sample sizes, comparisons, group bootstrap and eleven
success checks are fixed in advance.  The plan explicitly limits its claim to
observable application/session lifecycle events and keeps native dense writing
and implicit natural contradiction discovery as separate required stages.

Added `tasks/build_qwen_discrete_cohort_ood.py` and three pure-Python tests,
without generating the formal corpus.  The deterministic builder implements
the four frozen block/boundary schedules, two single-anchor twins per group,
causal replay, template rendering, explicit semantic no-op revocations and
hash manifests.  Validation requires exact causal answers, complete twins,
suite-disjoint identifiers and values, query/render parity and absence of
weight-oracle labels.  Formal materialization remains conditional on the
confirmation audit and test gate.

The builder also provides a hash-verifying loader and a strict visible-text
parser with the baseline parser interface.  The parser reconstructs only the
rendered new record, facts, reset prelude and at most one revoked-record
reference; it ignores private state and is checked against every generated
block.  Four tests now cover all suites, semantic no-ops, parser parity,
deterministic materialization and hash-verified loading.

At the time of this entry the predecessor formal run remains active on
`hku-gpu2`: seed 2301 has completed all 12 development policies, seed 2302 is
in progress, seed 2303 has not started and no test row exists.  The chained
confirmation monitor is active but consumes no GPU while waiting.  Formal
metrics and hashes will be added only after both producer and independent audit
reach terminal state.

The predecessor subsequently completed in 3,024.20 seconds with zero exit code
and all 16 independent audit checks passing.  Its registered continuous primary
failed and test remained unencoded.  The auxiliary exact 27-class controller
reached 98.9258% development accuracy versus 93.1351% for physics and 99.3634%
for the oracle; minimum seed, historical and minimum-family accuracy were
98.8607/96.8207/95.7031%.  All four separately frozen conditional prerequisites
passed, so the chain generated the fresh seed-`202610153` corpus and started the
confirmation on GPU 1 with `test_loaded=false`.  Compact audited artifacts,
controller checkpoint, action diagnostic, prerequisite decision and a complete
report are archived under
`experiments/cbf_ttt/qwen3_4b_final_1b_20260927/qwen_sparse_cohort_v1/`.

### Frozen-controller lifecycle OOD execution and audit

Added `tasks/evaluate_qwen_discrete_cohort_ood.py`.  The evaluator refuses to
create its result directory until the fresh confirmation audit, one-time test
release and registered test gate all pass with matching summary and controller
hashes.  It then verifies the conditionally generated seed-`202610154` corpus,
extracts the frozen Qwen3-4B contextual representations once, restores the
confirmation controllers without training or calibration, and replays all 13
registered policies across all four lifecycle suites and three projection
seeds.  It records every raw row, suite and macro aggregates, source-group
bootstrap comparisons, endpoint/interior action diagnostics, elapsed time,
peak GPU memory and dependency hashes.  An atomic progress file exposes the
current seed, suite and policy but is not used by any scientific decision.

Added `scripts/audit_qwen_discrete_cohort_ood.py`, which independently reloads
the corpus and recomputes all 156 policy/suite/seed files, source alignment,
aggregates, 20,000-draw paired bootstrap intervals, action diagnostics and the
eleven frozen success checks.  It also verifies confirmation, controller,
geometry, corpus and raw-result hashes and requires explicit runner completion.
Added `scripts/run_qwen_discrete_cohort_ood.sh` and
`scripts/run_qwen_discrete_cohort_ood_chain.sh`; dependency verification occurs
before formal corpus generation, and the independent audit runs only after the
workload's EXIT trap records zero.  Seven focused pure-Python tests cover corpus
causality, deterministic hashes, visible parsing, dependency hash binding,
training/audit gate parity and independent endpoint/interior accounting.  The
formal OOD corpus and result directory remain absent until confirmation passes,
so the pre-registered conditional release remains intact.  These additions do
not change baseline model, training or inference defaults.

Added `scripts/chain_discrete_confirmation_to_ood.sh` for the live transition.
It waits for the confirmation chain's EXIT-trap file rather than inferring
completion from logs or missing processes, requires its independent audit and
one-time test gate to pass, records a separate release decision, and only then
invokes the OOD runner plus auditor.  A failed confirmation leaves the formal
OOD root uncreated.

Before confirmation seed 2303, hidden test, or lifecycle OOD outcomes were
available, froze `QWEN_DISCRETE_RESIDUAL_READOUT_PLAN.md`.  This conditional
bridge uses a fresh seed-`202610155` corpus and applies the frozen discrete
controller to an actual final-hidden-state Qwen low-rank residual read by the
original LM head.  It begins with the requested eight-group resource pilot,
full `{0,0.5,1}^3` labels and sparse boundaries, records per-label time and peak
memory, and chooses only among predeclared 16/32/64-group formal sizes.  The
protocol fixes content interventions, all policies, actual-Qwen counterfactual
oracle, scales 4/8/16, source-group inference and ten release checks.  It
explicitly preserves the negative native dense `ttt_conv`/`ttt_proj` result and
cannot run unless both confirmation and lifecycle OOD audits pass.

Before the formal OOD release, ran an implementation-only end-to-end smoke on
GPU 0 with an unregistered seed, one group per suite and a temporary protocol
copy of the already audited predecessor controller.  It loaded the final
Qwen3-4B checkpoint, extracted all four suites, restored three projection
controllers, evaluated all 13 policies and wrote the expected 156 raw files
plus aggregates without error.  Its artifact explicitly records
`scientific_evidence=false` and `formal_seed_used=false`; its metrics are not a
pilot, are not used by any gate and cannot influence the registered formal
run.
