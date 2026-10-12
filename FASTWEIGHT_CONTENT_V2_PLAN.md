# Fast Weight 内容实验 v2：显式小写单词读出（执行前冻结）

## v1 失败与本轮修复

v1 的新来源 Q 在八选一上为 16/16，但全词表首 token 仅 5/16，按预登记停止且 V 未评分。事后只解码已存 argmax：6 条为 ` The`，5 条为首字母大写颜色，其余 5 条为正确小写颜色。这表明失败包含输出格式问题，不能据此评价 fast weight 内容。

本轮不降低 Q 门槛，也不使用 V 调模板。固定在已观察 dev 组 0–7 比较三个模板：

1. `lowercase`：原问题后要求“只输出一个小写颜色词”；
2. `options`：同时列出八个小写合法词，并要求只输出其中一个；
3. `completion`：用“latest access color ... is”单词补全。

选择规则按 dev 的最差域全词表首 token 准确率、总体首 token、较低 NLL、固定模板顺序依次最大化。只有最差域首 token ≥62.5%且总体八选一 ≥75%才允许进入 Q。组 8–15 的 Q 仍要求两个域八选一各≥75%、总体首 token≥50%、至少6/8组双 twin 正确。Q 通过后，组16–23才首次评分。

## V 与判读

V 完全沿用 v1 已冻结机制：两块8192 token，第一块禁写、事实所在第二块单次lr=0.3写入；同一 self KV 上比较 Correct、Wrong-twin、Zero、三个逐层等范数 Rademacher random。内容绑定和内容效用门槛、8来源组bootstrap、隔离检查均不变。

本轮新增格式只改变 query token，不改变背景、事实、答案、候选、写入位置或 checkpoint。任何 dev/Q 门失败均停止，不继续搜索模板或评分V。

## 运行

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_fastweight_content_v2_20261012 \
SOURCE=/home/ctj/cbf_ttt_memory_value_20261009 \
bash scripts/run_cbf_fastweight_content_v2.sh
```

本方案在 v2 的任何模型评分前冻结。

