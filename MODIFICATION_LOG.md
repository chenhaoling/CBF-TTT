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
