# 训练/推理一致性与 writer 训练资源检查

## 执行结论

1. 四个已看过的 pilot 来源、两种长度、开/关更新下，训练整段与原生整段推理的最终 hidden 和 NLL 全部逐值相同。流式差异较小，未触发预登记门槛。因此本轮没有发现足以解释此前严重累积退化的训练/推理前向不一致。
2. 冻结骨干、只训练现有 TTT writer 时，单张 5090 可以完成 12288-token 的前向、反向和 AdamW 更新；峰值 allocated 13.830 GiB。资源不是这一配置的阻塞因素。
3. **资源可行不等于学习稳定。** 长序列同一输入的三次训练 loss 为 1.9705→1.9188→4.4116，裁剪前总梯度最大约 10362。暂不启动正式短/长训练扩量；先核验优化更新尺度及梯度集中问题。没有保存、替换或发布新训练模型。

## P：冻结前向比较

方案在执行前写入 [TRAIN_INFER_PARITY_PLAN.md](../../../../TRAIN_INFER_PARITY_PLAN.md)，源码提交 `ef29877af33e8aff594a9e2be2ac77728abf797c`。2026-10-08 11:25:45–11:26:22（北京时间），37 秒，两张 5090 按来源分片。

- 最终 Qwen3-4B / 1B checkpoint；统一 BF16、SDPA、关闭 TF32；两套模型加载后所有 state_dict 张量完全相同，参数版本未变。
- 使用此前四个 pilot stable 场景，各取前 6144/12288 token；三条路径×正常/零 TTT lr，共 16 次成对比较、48 条前向。
- 训练前向在 eval/inference_mode 下检查，没有执行训练。零 lr 禁用快权重更新，正常 attention/KV 保留。
- 相同末尾 128 个 token 的 next-token NLL；以下均值跨四来源组，最后一列是逐来源最大差。

| 长度/更新 | 训练整段 | 原生整段 | 原生流式 | 最大训练/流式 NLL 差 |
|---|---:|---:|---:|---:|
| 6144_zero_lr | 2.308517 | 2.308517 | 2.309806 | 0.003029 |
| 6144_ttt | 2.326322 | 2.326322 | 2.325960 | 0.001170 |
| 12288_zero_lr | 1.790051 | 1.790051 | 1.789129 | 0.005054 |
| 12288_ttt | 2.085283 | 2.085283 | 2.086714 | 0.005233 |

训练整段/原生整段最终 hidden 的逐值相同比例均为 100%。训练/流式全序列 hidden 相对 L2 均值约 1.4%–1.5%，零更新对照也出现类似差异；不能声称流式逐值等价。最大 NLL 差 0.005233，低于预登记 6144 的 0.1 / 12288 的 0.5 门槛。这些阈值是诊断触发条件，不是统计等价或显著性检验。

7 项测试通过，包括 FP32 非零候选时完整/不完整块的一致性、writer 获得未来块梯度而后续 target 不影响先前块、TTT lr 还原及此前原生桥接检查。测试中显式初始化非零 conv，避免因零候选而空泛通过。

峰值 allocated/reserved 为 17.891/18.455 GiB；比较计时总和 52.931 秒。数据仍为机制诊断子集，未评分 confirm/test，不得宣称泛化结论。

## R：三步 writer 训练资源检查

源码提交 `91359fea83baead2d81fb8c33702bf4cfb884927`。2026-10-08 11:28:40–11:29:13，两卡并行完成，33 秒。

每长度从同一最终 checkpoint 独立初始化；训练现有 7 层 conv/proj，共 14 个参数张量、45,964,800 参数。其余骨干/head 冻结。writer 参数与 AdamW moments 为 FP32，计算为 BF16 autocast；SDPA、非重入梯度检查点、batch=1、lr=1e-5、weight_decay=.01、全局 clip=1。复用已安装 Liger loss。

输入来自实际 `mixed_1b.jsonl` 开头的 packed 训练记录，Qwen tokenizer 编码并加 EOS，截取指定长度。两种长度前缀相同，但训练 token 数不同，且可能跨文档边界；本轮只检查资源，不能作为公平学习效果比较。

| 长度 | 第 2–3 步平均耗时 | 峰值 allocated / reserved | 三步训练 loss |
|---:|---:|---:|---|
| 6144 | 2.847 s | 11.169 / 11.945 GiB | 2.2156 → 2.1990 → 2.2095 |
| 12288 | 9.093 s | 13.830 / 15.168 GiB | 1.9705 → 1.9188 → 4.4116 |

两长度所有 writer 梯度均有限非零、全部 writer 权重确实改变、冻结权重版本不变、优化器状态为 FP32。包含首步状态分配，未发生 OOM。

### 稳定性风险

- 短序列裁剪前梯度范数为 72.41、974.62、468.35；长序列为 5979.30、1442.89、10362.40。所有步实际都执行全局 clip=1，不能误称这些大范数未经裁剪进入优化器。
- 每一步最大梯度张量都是第 35 层的 `ttt_conv.weight`。其平方梯度范数占全部 writer 的比例：短序列 98.66%/98.56%/89.37%，长序列 99.50%/97.32%/68.34%。这是定位线索，不是证明该层导致损失波动的因果结果。
- 每步记录的 loss 是该步优化前计算的，所以第三步损失跳升发生在第二次更新之后。当前只观察一个重复训练输入，不能推广为所有数据或所有学习率不稳定。
- FP32 master 参数保证小更新可保存，但这不意味着每层的相对参数变化合适。全局梯度裁剪也不直接约束 AdamW 每层的实际相对更新。

## 下一步：先定位优化稳定性，再进入 L

按 R 的实测风险调整执行顺序，L 尚未启动。下一轮应预先固定并执行：

1. 在相同训练输入上核验融合 loss 与分块标准交叉熵的 loss/梯度，确认梯度集中是模型/数据响应还是训练算术问题。
2. 记录各层 conv/proj 的初始范数、梯度、实际 optimizer 位移及相对位移；尤其关注第 35 层。使用固定的小规模诊断比较，而非在保留集上反复选择超参数。
3. 只有确认梯度正确并有稳定更新设置后，才做短/长序列适配：同 checkpoint、同 trainable 集合、同 token 预算与 optimizer 步数（短序列通过梯度累积匹配）、相同数据 token 顺序，明确长序列允许跨段记忆而短序列重置。
4. 验证保留来源上的七 chunk 原生累积、每步清除和不写入。仍无有益快记忆则不扩大局部遗忘标签或控制器。

这些都是后续 TODO，未在本轮执行。不能把三步资源检查当作长序列适配已完成，也不能把 P 通过当作完全排除所有实现或精度问题。

## 文件、配置和复现

- `scripts/diagnose_ttt_train_infer.py`、`tests/test_ttt_train_infer.py`、`scripts/run_ttt_train_infer.sh`：P 的比较、测试与双卡运行。
- `scripts/probe_ttt_writer_training.py`、`scripts/run_ttt_writer_resource.sh`：R 的训练链路和双卡运行，必须读取通过的 P 汇总，拒绝覆盖目录。
- [summary.json](summary.json)：P 的全部聚合；[execution_audit.json](execution_audit.json)：P/R 的起止时间、源码/日志/结果哈希、R 资源和梯度聚合。
- [TRAIN_INFER_PARITY_PLAN.md](../../../../TRAIN_INFER_PARITY_PLAN.md)、[MODIFICATION_LOG.md](../../../../MODIFICATION_LOG.md)：预登记、参数、实现说明与完成状态。

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_train_infer_repeat bash scripts/run_ttt_train_infer.sh
ROOT=/home/ctj/cbf_ttt_writer_resource_repeat \
PARITY=/home/ctj/cbf_ttt_train_infer_repeat/summary.json \
bash scripts/run_ttt_writer_resource.sh
```

P 环境变量 ROOT/SOURCE/PYTHON/MODEL；R 环境变量 ROOT/PARITY/PYTHON/MODEL/TRAINING_DATA/TOKENIZER。默认路径指向当前服务器已存在的数据、环境和最终 4B checkpoint。各 CLI 参数详见计划及 `--help`；本轮无新增 baseline 配置项、无新增依赖，原模型、runtime、预训练配置均不修改。

逐来源比较、逐参数梯度和原始日志留在 `/home/ctj/cbf_ttt_train_infer_20261008` 与 `/home/ctj/cbf_ttt_writer_resource_20261008`；公开归档只包含代码、聚合数值、来源/结果哈希和解释。
