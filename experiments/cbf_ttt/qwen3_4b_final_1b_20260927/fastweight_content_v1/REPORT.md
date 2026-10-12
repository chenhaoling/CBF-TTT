# 单次 Fast Weight 内容实验 v1：Q 门停止报告

## 结论

v1 按预登记在基础读取门停止，**没有评分 V，也没有得到关于 Correct/Wrong/Zero fast weight 的效果结论**。16 条新来源 `short/single/qa` 上，八选一准确率为 100%，FineWeb/LongCrawl 均为100%，8/8 来源组的两个 twin 都正确；但全词表首 token 仅31.25%，低于50%门槛。

这说明事实的候选排序可读，但开放词表输出仍不稳定。事后只解码已保存的 argmax：6/16 为句子开头 ` The`，5/16 为首字母大写颜色，5/16 为正确小写颜色。该诊断没有新增 forward，也没有用于降低门槛。

## 隔离与审计

- Q 使用原 `memory_value_v1` 中从未评分的 confirm 组8–15；预留V为组16–23，来源完全不重合。
- 原始parquet重建、32个场景哈希、twin单token差异、颜色平衡、两分片结果和summary重算全部通过。
- `v_0.jsonl`、`v_1.jsonl` 和最终 `summary.json` 均不存在；终态 `stopped_by_content_Q`。
- 12项服务器测试通过；Q采集双分片各8条。最高allocated/reserved为10.145/10.961GiB；总流程55秒。
- 数据采集提交`fb484b5`；Q summary SHA-256 `24799f85ba4cfcf2a9faffb3e374e1414933b23980456e9e4cfd85591ffdea0d`。

## 后续

v2 不修改本轮结果或门槛。在已观察dev组0–7比较三个固定的“小写单词”提示，固定唯一格式后再用组8–15重新做Q；只有Q通过才首次评分仍未触碰的组16–23 V。详见 `FASTWEIGHT_CONTENT_V2_PLAN.md`。

