# Qwen3-4B 原生 TTT 快权重累积机制实验：完成报告

## 结论

本轮完成预登记的五臂因果对照和独立审计。结果支持一个明确但有限的结论：**在当前最终 checkpoint 和固定 12K 任务上，连续三次原生快权重相加会造成严重的输出终止偏置；限制为单次末块写入可以消除大部分偏置，但仍未产生超过完全禁写的内容收益。**

- 原生三次累积的正确答案 NLL 为 **6.8965**，全部 128 个 binding 查询的全词表首 token 都是 `<|im_end|>`，正确首 token 为 0%。
- `reset_each`/`last_only` 的正确答案 NLL 为 **1.4877**，EOS 首 token 率降到 **4.69%**，正确首 token 恢复到 **53.13%**。
- 相对累积，预登记选择的 `reset_each` 改善正确答案 NLL **5.4088**，8 来源组 bootstrap 95% CI **[3.4846, 7.4589]**；answer-vs-EOS margin 提升 **8.7588**，95% CI **[6.8975, 10.9180]**；首 token 正确率提升 53.13 个百分点。三项门槛全部通过。
- 但完全禁写 `zero` 的 NLL 更低，为 **1.4243**，正确首 token 为 **57.81%**，EOS 首 token 为 0%。`reset_each` 相对 zero 的 NLL“收益”为 **-0.06339**，95% CI **[-0.10206, -0.02448]**。因此“受限写入有内容收益”门槛失败。

该结果证明当前读出崩溃的直接原因之一是**跨块快权重累积**，没有证明当前 writer 能可靠保存事实，也没有证明遗忘控制器已经可以改善自然任务。

## 五臂主结果

主指标固定为 binding 的 target+anchor，共 128 查询/臂。NLL 使用完整词表概率；margin=`eos_nll-answer_nll`，正值表示答案 logit 高于 EOS。候选准确率是八选一，不能替代全词表首 token。

| 快权重规则 | 答案 NLL | answer-vs-EOS margin | 正确首 token | EOS 首 token | 八选一准确率 |
|---|---:|---:|---:|---:|---:|
| zero | 1.4243 | +3.2754 | 57.81% | 0.00% | 57.81% |
| first_only | 1.5718 | +1.3306 | 46.88% | 25.00% | 58.59% |
| last_only | 1.4877 | +1.9253 | 53.13% | 4.69% | 56.25% |
| reset_each | 1.4877 | +1.9253 | 53.13% | 4.69% | 56.25% |
| cumulative | 6.8965 | -6.8335 | 0.00% | 100.00% | 49.22% |

`first_only` 的 EOS 率为 25%，高于 `last_only` 的 4.69%，说明写入位置也影响偏置。两种单次写入都没有优于 zero；八选一数值接近时，全词表行为已明显不同，再次说明仅用候选排名会掩盖生成失败。

### Target 与 anchor

| 规则 | target NLL / 首token | anchor NLL / 首token | target EOS | anchor EOS |
|---|---:|---:|---:|---:|
| zero | 1.3970 / 60.94% | 1.4516 / 54.69% | 0.00% | 0.00% |
| first_only | 1.5330 / 45.31% | 1.6107 / 48.44% | 25.00% | 25.00% |
| last_only / reset_each | 1.4594 / 57.81% | 1.5160 / 48.44% | 1.56% | 7.81% |
| cumulative | 6.8985 / 0.00% | 6.8945 / 0.00% | 100.00% | 100.00% |

累积问题同时影响当前目标与应保留锚点，不是单一场景或单一问题类型造成。完整四场景、两域和八来源组数据见 `summary.json`。

## 机制核验

### `reset_each` 与 `last_only` 精确相同

两臂的 64 个上下文、256 个 binding/QA 查询在候选 NLL、答案 NLL、EOS NLL、margin、候选预测和全词表 argmax 上全部逐值相同，最大误差为 0。

原因与仓库更新时序一致：每个完整块先用进入该块前的 `W_fast` 产生输出和 KV，再在块尾形成 delta。`reset_each` 在第 2、3 块前恢复骨干权重，因此前两次块尾写入在被任何后续块使用前就被清除；查询只看到第三块的 delta，等价于 `last_only`。这既验证了干预实现，也说明后续若要研究 last-1/last-2 局部记忆，必须显式保存可寻址的分块 delta，单个覆盖式 `W_fast` 无法同时读取多个局部分量。

### 更新尺度与偏置

以第 35 层 `||W_fast-W_base||/||W_base||` 为例：单次首块为 0.00932，单次末块为 0.00894，三次累积依次为 0.00932、0.01729、0.02797。偏置随累积出现，但当前实验只比较离散时序，不能仅凭相关性断言某个范数阈值是根因；还需做归一化/衰减对照。

## 精确桥接与审计

- `cumulative` 对上一轮 `step81381_native`：64 上下文/256 查询，候选及答案 NLL 最大误差 0，候选/首 token 一致率 100%。
- `zero` 对上一轮 `step81381_plain`：64/256，所有同类指标误差 0、一致率 100%。
- `last_only` 对 `reset_each`：64/256，答案、候选、EOS 和 margin 最大误差均为 0。
- 五臂使用同一模型权重 hash；各分片前后参数 hash/version 不变；每个查询使用 prefix 深拷贝，父 cache hash 不变。
- 9 项服务器 PyTorch 测试通过。独立脚本重新加载原始逐条结果、重算 summary、检查源 audit/hash、矩阵完整性和三组精确桥接，`audit_passed=true`。
- 数据采集提交 `546707d`；增强只读桥接审计提交 `547f2c5`。summary SHA-256 `5fb33b079697e85deb822e3073507c644637152accb8b5a8a9ed51e77d9a53d9`。

## 规模、时间与显存

- 64 个已观察 dev 场景 × 5 臂 = **320 rollout**；每个场景四个查询，共 **1280 查询**。
- hku-gpu2 两张 RTX 5090；2026-10-12 11:23:21–11:47:34 +08:00，总墙钟 **1453 秒（24 分 13 秒）**，含测试、五次模型加载/卡、采集、汇总及首次审计。
- 各臂平均 rollout 1.341–1.346 秒；平均 query 0.0583–0.0586 秒。
- 最高 allocated **11.872 GiB**，最高 reserved **13.889 GiB**。这是 PyTorch 峰值，不是整机总显存。

## 实现文件与兼容性

| 文件 | 内容 |
|---|---|
| `NATIVE_ACCUMULATION_PLAN.md` | 评分前冻结五臂、指标、门槛与停止规则 |
| `tasks/cbf_native_accumulation.py` | 写入调度、只清 fast weight 的干预、EOS 诊断、来源组 bootstrap 与汇总 |
| `tests/test_cbf_native_accumulation.py` | zero/first/last/reset/cumulative 时序及统计测试 |
| `scripts/run_cbf_native_accumulation.sh` | 双卡固定分片、拒绝覆盖、汇总和审计 |
| `scripts/audit_cbf_native_accumulation.py` | 独立重算、逐值桥接和完成审计 |
| 本目录 | 设计、聚合结果、审计、状态、时间戳和报告；逐条 JSONL/模型留服务器 |

所有入口均为新增 opt-in 文件；baseline 训练、推理默认配置和 checkpoint 未修改，无新增依赖。

## 复现

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_native_accumulation_repro \
SOURCE=/home/ctj/cbf_ttt_checkpoint_native_20261010 \
bash scripts/run_cbf_native_accumulation.sh

/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
  -m scripts.audit_cbf_native_accumulation \
  --root /home/ctj/cbf_ttt_native_accumulation_repro \
  --output /home/ctj/cbf_ttt_native_accumulation_repro/execution_audit_recheck.json
```

复现依赖服务器上最终 checkpoint、上一轮已审计 source 及其逐条结果；公开聚合文件不含原始语料/token/权重。

## 与遗忘主线的关系和下一步

本轮给出了设计约束：原生 `W_fast <- W_fast + delta` 在三个块后发生强终止偏置，直接训练“保留/遗忘”控制器会把 writer 缺陷混入策略标签。当前不应扩大反事实标签规模。

下一项关键实验应先验证**单次 fast weight 是否携带事实内容**：在最终 checkpoint 上构造新的、Full-KV 读取通过的短程单键 twin 场景；两条 twin 仅改变一个随机事实值。在保持同一查询和同一 KV 不变时，分别替换 `W_fast` 为 Correct、Wrong-twin、Zero，并加 norm-matched random 对照。先以未见来源 Full-KV Q gate 封存任务，再检验 Correct 是否在答案 NLL 上稳定优于 Wrong/Zero。若该内容门通过，再做衰减/归一化多块累积以及 last-1/last-2 可寻址分量；若不通过，则应先改 writer 目标，而不是训练遗忘策略。

## 限制

- 仅 8 个已观察来源组和构造颜色绑定，不是独立确认或自然 benchmark。
- 当前最终 checkpoint 的 zero 基线本身未通过此前全 16 格 75% 读取门槛；本轮因果结论限于分布偏置与相对 NLL。
- `reset_each` 的前两次写入在块间被清除，不能代表一个可读取多层级局部记忆的完整实现。
- 只测试 lr=0.3 和三次写入，尚未定位学习率、delta 归一化、层选择或训练目标的贡献。

