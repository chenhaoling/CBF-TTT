# 动态世界 fast-memory 写入实验报告

## 1. 目标与结论

前一轮只训练遗忘策略时，fresh attention KV 下的 Correct/Wrong/Empty fast memory 不可区分。本实验先检验更基础的前提：原始 Qwen3-4B 上新初始化的 In-Place TTT conv/proj writer，能否把未见过的随机 session 事实写入 fast memory，并在新世界中通过正确记忆恢复随机值。

**固定内容门槛未通过。** 完整的 `joint_context` 臂在 2048 次问题暴露后，dev 的 Correct/Wrong/Empty `code_correct` 均为 0/128；Correct 相对 Wrong 和 Empty 的 source-group 平均 digit-NLL 改善只有 `0.0401` 和 `0.0594`，两个 95% bootstrap 区间都跨 0，也低于预设 `0.10`。Full-KV `code_correct=93.75%`，证明输入事实和问题可读。`sequential` 臂在第 1095 次更新时触发 `||ΔW||/||W|| > 1` 安全阈值，按计划停止，未产生终点结果。

当前证据说明 writer 学到了共享输出格式，但没有学到可泛化的随机值绑定：joint dev 的 Correct answer-NLL 从 Empty 的 `4.3280` 降到 `2.3471`，而真正承载随机内容的 digit-NLL 仅从 `2.7586` 降到 `2.6992`，生成的完整编号仍为 0。**现在不应训练遗忘控制器或扩大到 3000 条反事实标签。**

## 2. 研究构思与实现假设

- 本阶段固定 `g=1`，不决策是否提交新 `ΔW`，也不训练 `alpha` 遗忘策略；只有内容可读后才进入遗忘研究。
- 同一来源世界包含两个 twin session。问题、自然背景和三个保护事实相同，只改变 anchor 的随机 `code_#####`，因此可检查同一 query 是否依赖正确 session memory。
- train/dev/test 的实体和值和语言模板均分离。test 只生成语义真值；装箱器不读取、不分词、不评分 test。
- 每个世界使用一个唯一 FineWeb-Edu 或 LongCrawl64 长文档作背景，动态事实置于 4096-token block 尾部。两来源等量交替。
- 两训练臂按问题暴露配平，不按 optimizer step 或写入次数配平。`joint_context` 一次写入并对同 context 四题平均 CE；`sequential` 每题重新写入并单独更新。
- Qwen3-4B backbone 冻结；七层 conv/proj writer 与 AdamW 状态为 FP32，forward 为 BF16。writer 从新初始化开始，并非已经通过大规模 In-Place TTT 预训练的 writer。

执行前的完整预算、门槛与停止规则见 [`DYNAMIC_MEMORY_WRITER_PLAN.md`](../../../DYNAMIC_MEMORY_WRITER_PLAN.md)。

## 3. 数据与实验矩阵

| 项目 | 数值 |
|---|---:|
| 语义世界 | train/dev/test = 64/16/16 |
| packed rows | train 512，dev 128，test 0 |
| packed contexts | train 128，dev 32 |
| 自然背景 | FineWeb-Edu 40，LongCrawl64 40 |
| context 长度 | 4096 token |
| 动态事实后缀 | 113–152 token |
| 答案长度 | 7 token |
| smoke | 8/2 train/dev 世界，1 round，64 问题暴露/臂 |
| formal | 64/16 train/dev 世界，4 rounds，2048 问题暴露/臂 |
| checkpoint | 0 / 1024 / 2048 问题暴露 |
| train probe | 固定前 8 个训练世界，64 题 |
| dev | 全部 16 个开发世界，128 题 |

数据 hash：`cc62cd72134334532a5250152a74c5e0a52442419a5ade265f40d7c31f0da48d`；背景 manifest hash：`5235c118f999c9e83ae767338210a66eb4b8ca5063a131ee548e7b2760005059`。原始自然语料为 1B 预训练混合文件，动态随机事实不来自该语料。

## 4. 正式结果

### 4.1 Joint-context 完整终点

| split / policy | code correct | first-line exact | digit NLL | answer NLL |
|---|---:|---:|---:|---:|
| train probe / Correct | 0.00% | 0.00% | 2.2502 | 1.6233 |
| train probe / Wrong | 0.00% | 0.00% | 2.5073 | 1.8255 |
| train probe / Empty | 0.00% | 0.00% | 2.7358 | 4.8409 |
| train probe / Full-KV | 100.00% | 98.44% | 0.00004 | 0.0104 |
| dev / Correct | 0.00% | 0.00% | 2.6992 | 2.3471 |
| dev / Wrong | 0.00% | 0.00% | 2.7392 | 2.3738 |
| dev / Empty | 0.00% | 0.00% | 2.7586 | 4.3280 |
| dev / Full-KV | 93.75% | 93.75% | 0.0309 | 0.0668 |

Dev source-group 对照：

| 对照 | Wrong/Empty − Correct digit-NLL | 95% bootstrap CI | 正收益组比例 | 固定要求 |
|---|---:|---:|---:|---:|
| Wrong | 0.0401 | [-0.0963, 0.1851] | 43.75% | ≥0.10 且下界>0 |
| Empty | 0.0594 | [-0.0412, 0.1612] | 56.25% | ≥0.10 且下界>0 |

Anchor 的 twin-memory − correct-memory digit-NLL 为 `-0.00444`，方向也不支持 session-specific anchor 绑定。Joint 的前 50/后 50 update 平均 loss 为 `5.0046/1.5351`；loss 下降主要可由格式、前缀和 EOS 学习解释，不能据此声称随机事实写入成功。

### 4.2 Sequential 安全停止

Sequential 在固定 1024 暴露中点的 dev `code_correct` 仍为 0。Correct/Wrong/Empty digit-NLL 为 `2.7952/2.8572/2.7586`：Correct 相对 Wrong 改善 `0.0620`，bootstrap CI `[-0.5801, 0.6842]`；相对 Empty 反而差 `0.0366`。完成的 1094 步中最大 `||ΔW||/||W||=0.99018`；第 1095 步候选超过 1，训练入口在 optimizer step 前抛错并停止。该臂没有 2048 终点，不能与 joint 作终点优劣比较。

### 4.3 固定门槛

| Joint dev 条件 | 结果 | 是否通过 |
|---|---:|---:|
| Full-KV code ≥ 0.80 | 0.9375 | 是 |
| Correct code ≥ 0.60 | 0 | 否 |
| Correct−Wrong code ≥ 0.20 | 0 | 否 |
| Correct−Empty code ≥ 0.20 | 0 | 否 |
| Wrong−Correct digit NLL ≥0.10 且 CI 下界>0 | 0.0401；下界−0.0963 | 否 |
| Empty−Correct digit NLL ≥0.10 且 CI 下界>0 | 0.0594；下界−0.0412 | 否 |

`stage_passed=false`。Test 未评分，未用中间 checkpoint 选择结果。

## 5. 资源、失败修正与审计

- 20 项相关服务器测试全部通过；新增动态实验测试 4 项通过。
- 首次 8/2/2 smoke 两臂完成并通过独立审计。Sequential/joint 训练 reserved 峰值为 `19.06/19.55 GiB`。
- 初版 formal checkpoint-0 评测一次常驻约 50 份 dense fast memory，发生 OOM；没有开始正式训练。失败工件保留为远程 `formal_failed_e7f7ff6`。
- 修正仅改变评测驻留生命周期：按来源世界保留当前 twin 与 wrong donor 共四份 memory，完成 8 题后释放。正式流式评测稳定约 11 GiB。
- Joint 正式训练 allocated/reserved 峰值为 `12.10/12.34 GiB`，平均 update `0.4428 s`；最大单策略评测 `0.970 s`，allocated 峰值 `10.43 GiB`。
- 独立终态审计通过：640 packed rows、来源/背景 hash、twin 因果、2048 joint 暴露、1094 sequential 固定调度前缀、checkpoint FP32 与实际变化、AdamW FP32、两臂相同初始化、冻结 backbone hash、评测聚合重算、安全停止日志和 test 封存均一致。

审计文件为 [`formal/execution_audit.json`](formal/execution_audit.json)，汇总为 [`formal/summary.json`](formal/summary.json)。原始大 checkpoint、逐步日志和逐行评分保留在 hku-gpu2 的 `/home/ctj/cbf_ttt_dynamic_memory_v1`，仓库只归档小型 summary/manifest/audit。

## 6. 修改文件与模块

| 文件 | 作用 |
|---|---|
| `tasks/pack_cbf_dynamic_memory.py` | 自然背景选择、任意规模4096-token装箱、twin/wrong映射、test封存 |
| `tasks/train_cbf_dynamic_memory.py` | matched-exposure训练、流式分组评测、内容门槛和常规双臂汇总 |
| `scripts/audit_cbf_dynamic_memory.py` | 完整双臂数据/训练/checkpoint/评测独立审计 |
| `scripts/run_cbf_dynamic_memory.sh` | 数据、smoke、formal、汇总与审计主入口 |
| `scripts/resume_cbf_dynamic_memory_formal.sh` | 保留首次OOM工件并从已审计smoke恢复formal |
| `scripts/finalize_cbf_dynamic_memory_safety_stop.py` | 无模型调用地审计完整joint与部分sequential，生成诚实终态 |
| `tests/test_cbf_dynamic_memory.py` | 数据封存、twin因果、暴露配平、门槛、四context流式闭包测试 |
| `DYNAMIC_MEMORY_WRITER_PLAN.md` | 执行前冻结的预算、门槛和停止规则 |

这些都是独立入口；原始 baseline 模型、预训练配置、训练脚本和评测入口未修改。

## 7. 复现命令

完整新运行：

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_dynamic_memory_v1_new \
MODEL=/home/ctj/models/Qwen3-4B \
CORPUS=/home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
bash scripts/run_cbf_dynamic_memory.sh
```

若一个完整 joint 臂和一个 tripwire 停止的 sequential 臂需要按本次协议归档：

```bash
python -m scripts.finalize_cbf_dynamic_memory_safety_stop \
  --root /home/ctj/cbf_ttt_dynamic_memory_v1/formal \
  --data-root /home/ctj/cbf_ttt_dynamic_memory_v1/data \
  --run-log /home/ctj/cbf_ttt_dynamic_memory_v1/run.log
```

## 8. 下一步与风险

本轮不支持继续遗忘策略。下一阶段应保留稳定的 joint-context 训练，直接针对“随机值绑定”设计 writer 学习信号：

1. 在更多独立动态世界上比较 `joint CE` 与 `joint CE + anchor twin contrast`，对随机数字区域加权；不再使用发生尺度失稳的 sequential 配方。
2. 先做 128/32 世界的固定小规模剂量实验，检查 digit-NLL、Correct-vs-Wrong/Empty 和 delta scale 随暴露增长的曲线；通过趋势门槛后再扩到 512/128 世界。
3. 若内容门槛通过，再固定 writer，构造 retain、last-1、last-2、global-reset 的遗忘 oracle；之后才收集控制器标签。

风险包括：46M writer 参数相对 64 个训练世界和 2048 次暴露过少；train/dev 同时更换实体与模板，失败来源可能混合；只有一个 seed；两臂按问题暴露而非计算/更新次数配平；Full-KV 的高分只证明文本路径可读，不证明 fast memory；本轮模型是官方 base 上新初始化 writer，结论不能直接外推到充分预训练的 In-Place TTT writer。

