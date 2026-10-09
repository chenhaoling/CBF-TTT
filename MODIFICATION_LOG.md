# CBF-TTT 实现记录

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
