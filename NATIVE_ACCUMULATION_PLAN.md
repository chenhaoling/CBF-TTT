# 原生 TTT 快权重累积机制实验（执行前冻结）

## 研究问题

最终 1B checkpoint 在固定 12K/binding 任务上，原生三次 TTT 写入使正确答案首 token 命中降为 0%，并在全部 128 个 binding 查询上偏向 `<|im_end|>`。本实验只改变快权重的写入与清除时序，检验该现象是否由多次快权重累积造成。它不训练遗忘控制器，也不把基础读取失败解释成遗忘策略无效。

## 固定数据、模型与五个臂

- 数据复用已审计 `checkpoint_native_v1` 的 64 个 dev 场景：8 个来源组、FineWeb-Edu/LongCrawl64 各 4 组、4 个机制、2 个 twin；每个场景保留 target/anchor × binding/qa 四个查询。
- 模型固定为 Qwen3-4B 经过约 1B token 预训练的 `global_step_81381/hf_ckpt`，BF16、SDPA、TF32 关闭；7 个 TTT 层、chunk=4096、写入学习率 0.3。骨干参数冻结。
- 输入固定为三个连续 4096-token 块，总计 12288 token；完整 KV 始终保留。查询短于一个块，不触发额外写入。
- 五臂仅改变三个块的写入规则：
  - `zero`：学习率 `[0,0,0]`，原生路径但不写入；
  - `first_only`：`[0.3,0,0]`，只写第一块，随后保留该快权重；
  - `last_only`：`[0,0,0.3]`，只写最后一块；
  - `reset_each`：每块均以 0.3 写入，但第 2、3 块前把各 TTT 层快权重恢复为冻结骨干的 down-proj，KV 不变；查询只看到最后一块产生的快权重；
  - `cumulative`：`[0.3,0.3,0.3]`，仓库原生三次累积路径。
- 两张 RTX 5090 按固定来源组分片。五臂共 320 条轨迹、1280 次查询。每条记录 rollout/query 耗时及 PyTorch allocated/reserved 峰值。

## 指标与事前判读

主分析固定为 binding 的 target+anchor，不按结果切换格式。逐臂报告：正确答案 NLL、`<|im_end|>` NLL、正确答案相对结束 token 的 logit margin（`eos_nll-answer_nll`）、全词表首 token 命中、八候选准确率；另报告 target/anchor、场景、域和来源组结果。QA 只作辅助。

对 `first_only/last_only/reset_each/zero` 分别计算相对 `cumulative` 的 8 来源组配对差，并以固定 seed 的来源组 bootstrap 给出 95% CI。仅在三个仍有写入的受限臂中，按 binding 正确答案 NLL 最低选择诊断臂，平局顺序为 `reset_each`、`first_only`、`last_only`。

“多次累积造成读出损伤”的预登记门槛为所选受限臂同时满足：

1. 相对 cumulative 的正确答案 NLL 改善至少 0.25，且来源组 bootstrap 95% CI 下界大于 0；
2. answer-vs-EOS margin 提升至少 0.5，且 95% CI 下界大于 0；
3. 全词表首 token 命中率提升至少 10 个百分点。

另将所选受限臂与 `zero` 比较。只有正确答案 NLL 相对 zero 的来源组 CI 下界大于 0，才记为“受限写入有内容收益”；否则结论限于累积有害或写入无稳定收益。八候选改善不能替代全词表指标。

## 隔离、审计与停止规则

- 每个场景新建 cache；每个查询深拷贝 prefix；查询前后父 cache hash 必须一致。
- 清除操作只替换 7 层 `ttt_states[*][2]`，要求块边界的 partial `h/t` 均为空；KV 长度和内容不变。记录每块后各层相对骨干位移。
- 五臂加载同一 checkpoint；参数 hash/version 在每分片前后必须不变。验证矩阵、模型来源、旧 summary/audit hash、结果行 hash 和所有有限性。
- 任一隔离、来源、加载或数值检查失败即停止并保留失败工件，不解释效果。当前 64 个场景是已观察 dev，结论是机制诊断，不是独立泛化结果，也不自动启动控制器训练。

## 运行入口

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_native_accumulation_20261012 \
SOURCE=/home/ctj/cbf_ttt_checkpoint_native_20261010 \
bash scripts/run_cbf_native_accumulation.sh
```

该文件在读取本实验任何新评分之前冻结。完成后追加实际结果，不回改上述臂、指标或门槛。
