# 单次原生 Fast Weight 内容因果实验（执行前冻结）

## 问题与两阶段门控

上一轮证明最终 Qwen3-4B checkpoint 的三次原生 TTT 累积造成强 `<|im_end|>` 偏置，单次末块写入能解除大部分崩溃，但没有优于禁写。下一问题是：**单次 `Delta W` 是否携带产生它的事实内容，还是只造成与内容无关的分布移动？**

复用 `memory_value_v1` 中从未评分的 16 个 confirm 来源组。只取每组 stable twin，并按已在 dev 选定的 `short/single/qa` 协议重建两块上下文。组 8–15 固定为 Q gate，组 16–23 固定为 V；Q/V 各含 FineWeb-Edu 与 LongCrawl64 各 4 组、16 个 twin 上下文，来源互斥。不得根据 V 分数修改任务。

### Q：基础读取门

最终 checkpoint 关闭 TTT，以完整 KV 读取 Q 的 16 条上下文。门槛同时要求：

1. FineWeb 和 LongCrawl 的八选一准确率各不低于 75%；
2. 总体八选一准确率不低于 75%；
3. 总体全词表首 token 正确率不低于 50%；
4. 至少 6/8 来源组的两个 twin 均八选一正确。

Q 不通过则 V 保持未评分，终态 `stopped_by_content_Q`。

### V：保持同一 KV，只替换 W_fast

Q 通过后才读取 V。每个 twin 上下文均保留自己的完整 KV；第一块学习率为 0，事实所在第二块学习率为 0.3，因此只产生一次块尾写入。查询不满 4096 token，不产生新 delta。对同一个查询和同一份 KV 顺序评分：

- `correct`：本上下文产生的 `W_fast`；
- `wrong`：只替换为 twin 上下文产生的 `W_fast`；twin 与本上下文仅有一个事实值 token 不同；
- `zero`：各 TTT 层恢复冻结的 down-proj 权重；
- `random0/1/2`：对本上下文的每层 delta 使用三个固定 seed 做独立 Rademacher 符号随机化，再加回骨干权重；逐元素绝对值和逐层 Frobenius 范数与 correct delta 相同。

所有干预只替换 7 个 TTT 层的 `ttt_states[layer][2]`；KV、query、位置、骨干、delta 尺度及其他 cache 状态保持不变。

## 指标与预登记判读

每个策略报告正确答案 NLL、twin 答案 NLL、own-vs-twin logit margin、八选一准确率、全词表首 token 和 EOS 率。三个 random 先在每条样本内平均，再按 8 个来源组配对 bootstrap（20,000 次，固定 seed）。

内容绑定门同时要求：

1. `wrong_nll - correct_nll >= 0.02` 且来源组 bootstrap 95% CI 下界大于 0；
2. `random_mean_nll - correct_nll >= 0.02` 且 CI 下界大于 0；
3. correct 相对 wrong 的 own-vs-twin margin 提升至少 0.05，且 CI 下界大于 0。

内容效用单独判定：`zero_nll - correct_nll >= 0.02`、CI 下界大于 0，且 correct 八选一准确率不低于 zero。内容绑定通过但效用失败表示 `Delta W` 含事实方向信息，却没有改善完整系统；两者均失败表示当前 writer 尚无可检出的内容选择性。

## 隔离、审计和资源

- 每对 twin 独立创建 cache；每个干预从同一 self prefix 深拷贝。查询前后父 cache SHA-256 必须一致。
- 记录两块后各层相对位移、随机 delta 的逐层范数比、查询/rollout 时间和 allocated/reserved 峰值。
- 两卡按来源组固定分片；参数 hash/version 在每分片前后不变。独立审计重建数据、Q/V 矩阵、随机 seed/范数、summary、来源和模型 hash。
- 原始语料、token、逐条评分与权重留服务器；GitHub 只归档计划、代码、聚合 summary、审计和报告。baseline 默认路径不修改。

## 运行入口

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_fastweight_content_20261012 \
SOURCE=/home/ctj/cbf_ttt_memory_value_20261009 \
bash scripts/run_cbf_fastweight_content.sh
```

本文件在本轮任何 Q/V 新评分产生前冻结。
