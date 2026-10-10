# 仅训练遗忘策略：独立复现与记忆内容门槛

## 研究问题与执行顺序

固定 `g=1`，不训练写入门，不修改 backbone 或已有 writer。第一阶段只回答两个问题：

1. 旧实验中的 session 快记忆是否真的携带可区分的历史事实内容；
2. 学习到的遗忘策略是否在三个随机种子上优于最强固定遗忘率。

任一问题未通过，不进入大规模动态世界、反事实标签或端到端控制器实验。旧结果中控制器优于持续累积，但固定完全清除更强；因此本轮不把“遗忘有价值”和“自适应遗忘有效”视为同一个结论。

## 固定资产

- 服务器：hku-gpu2，两张 RTX 5090；已有任务占卡时排队等待，不抢占其他用户进程。
- 模型：Qwen3-4B / 1B-token In-Place TTT 最终 checkpoint。
- 数据：旧控制器实验封存的 300 个 test 场景、100 个互斥来源组，`correction/stable/topic_shift` 各 100 个。
- 每场景 8192-token context、一个短尾 future、两个查询：`historical` 与 `current_or_new`。
- 控制器：固定 500 个训练来源组，seed 42/43/44。三个 checkpoint 已经存在，本轮不按 test 重新选 epoch、种子或训练规模。

旧控制器的输入包含当前 chunk 语义和历史/候选统计，因此本轮验证的是“只控制遗忘、不控制写入”，尚不是“控制器完全不看当前输入”的特征消融。只有本轮通过后，才比较 history-only 与包含当前语义的控制器。

## 策略与系数语义

旧 `forget` 实现中的系数是旧记忆的遗忘率：

`M_t = (1 - alpha_t) M_{t-1} + delta_t`

- `fixed_0`：持续累积旧快记忆；
- `fixed_0.5`：固定半衰减；
- `fixed_1`：每次提交新更新时完全清除旧快记忆，但保留当前 `delta_t`；
- `controller_seed42/43/44`：三个既有学习策略。

所有策略保持正常 attention KV；`g=1`，当前候选始终写入。

## Correct / Wrong / Empty memory 对照

每个策略先正常消费 context 与 future tail，然后对同一个查询构造四种条件：

1. `full_context`：保留正常 KV 与快记忆，精确复现旧 rollout；
2. `correct_memory`：清空 KV，只保留本场景快记忆；
3. `wrong_memory`：清空 KV，换入同 regime、不同来源组的快记忆；
4. `empty_memory`：清空 KV 与快记忆。

来源组按排序后相邻两两配对，wrong-memory 映射互为 donor；同一配对在六个策略中固定。内容门槛只使用 `historical` 查询，因为 `current_or_new` 事实可能只出现在不足一个 TTT chunk 的 future tail 中，只能进入 KV、不能写入快记忆。两类查询仍完整记录并分别报告。

## 预注册判据

### A. 快记忆内容门槛

对每个策略分别计算：

- `NLL(wrong historical) - NLL(correct historical)`；
- `NLL(empty historical) - NLL(correct historical)`。

两项均须满足：平均收益大于 0.005 NLL、以 100 个来源组重采样的 95% bootstrap 区间下界大于 0、至少 60% 场景收益为正。只有这样才认为对应策略的快记忆携带可复用的历史内容。

### B. 自适应遗忘门槛

以旧 test 上最强的 `fixed_1` 为对照。对三个控制器分别比较 `correct_memory` 的 historical NLL。至少 2/3 个随机种子平均收益大于 0.005，且至少 2/3 个种子的来源组 bootstrap 区间下界大于 0，才认为自适应遗忘通过第一阶段。

同时报告 `full_context` 的配对差值，但不以 KV 可见的当前事实替代 session 快记忆证据。

## 完整性与停止条件

- 六个策略必须各覆盖完全相同的 300 场景、100 来源组；
- wrong-memory donor 必须跨来源组、同 regime、互为配对；
- 快记忆范数必须非零，全部 NLL 有限；
- `fixed_0/fixed_0.5/fixed_1/controller_seed42` 的 `full_context` NLL 必须逐场景复现旧封存 rollout，最大误差不超过 `1e-5`；
- 模型参数版本在采集前后保持不变；
- 任一审计失败，运行状态记为 failed，不解释效果。

若内容门槛失败，下一步先重新构造能被快记忆读取的动态赋值语料。若内容门槛通过而自适应门槛失败，保留“遗忘有价值”的结论，但不训练更大控制器；下一步改用损失敏感排序目标或构造同时需要保留与清除的动态世界。只有两项都通过，才进行 history-only 特征消融与 32/8/8 独立世界实验。

## 运行方式

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_forgetting_only_validation_v1 \
bash scripts/run_cbf_forgetting_only_validation.sh
```

脚本拒绝覆盖已有 ROOT。若 GPU 被其他用户占用，会等待两张卡连续三次低于 4 GiB 后再启动 smoke 和双卡正式采集。
