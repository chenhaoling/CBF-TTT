# 记忆价值验证 v1：Q 可读性未通过，按协议终止

## 执行结论

**完整执行了数据构建、来源审计和 Q 可读性检查；Q 未过门，V/F 对照与确认集均未运行。** 这不是“记忆价值无效”的结论，也不是实现崩溃。按照执行前协议，每类场景的目标与锚点八选一准确率均须至少 75%；三类目标未达标，不能把后续机制验证视为已完成。

采集代码 `077d32059f11bad2bdef9a9687f07be3a6454e83`；2026-10-09 10:55:33–11:06:17 +08:00，墙钟 **10 分 44 秒**。hku-gpu2 双 RTX 5090，最终 Qwen3-4B / 1B checkpoint；21 项测试通过。服务器 tmux 已正常退出，状态 `stopped_by_Q`，没有采集异常。

## 数据与来源

构建 24 个来源组：dev 8 / confirm 16，各 split 的 FineWeb-Edu 和 LongCrawl64 各半。每组四种场景、两个只差一个目标事实 token 的版本，共 192 会话；每会话六个 4096-token 块，两个独立查询（目标和旧锚点）。标签在每 split、场景和数据来源内均衡。

FineWeb-Edu 使用固定 revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` 的 `sample/10BT/013_00000.parquet`；LongCrawl64 复用原始 `train/0-of-256.parquet`。候选池96/32，经扫描合格94/30，固定种子选出72/12篇，共84篇。FineWeb每组六篇提供自然背景；LongCrawl每组一篇提供六段，不混称为相同文档布局。

核对39个旧manifest、21个旧原始上下文文件，扫描全部162,762条打包训练记录。最终来源ID、原始/规范化哈希唯一，指定长锚点匹配为0；这不等于语义去重或Qwen基座预训练去重。数据与原始分片哈希见 [design.json](design.json)。

全部192个会话已构建，但只评分64个dev会话。**128个confirm会话没有被评分，也没有用于选择提示、策略或门槛。**

## Q 完整结果

全部保留正常 attention KV，全程不写入快速记忆；不是任何遗忘策略的效果。每行16个目标和16个锚点答案，来自8个dev来源组、每组两版本。

| 场景 | 目标事实正确数/准确率 | 旧锚点正确数/准确率 | 两项都≥75% |
|---|---:|---:|---|
| 历史持续有效 stable | 5/16，31.25% | 14/16，87.50% | 否 |
| 同键事实更正 correction | 16/16，100% | 13/16，81.25% | 是 |
| 最近一次是干扰 | 9/16，56.25% | 14/16，87.50% | 否 |
| 倒数第二次是干扰 | 10/16，62.50% | 15/16，93.75% | 否 |

目标整体40/64（62.5%），锚点56/64（87.5%）。这都是预设八个颜色之间的分类准确率，不是自由生成准确率。全词表首token greedy也记录在原始输出/审计分层中，不应把受限准确率冒充自然回答成功率。

按数据来源的目标准确率（各8答案/场景）：

| 来源 | stable | correction | 最近一次干扰 | 倒数第二次干扰 |
|---|---:|---:|---:|---:|
| FineWeb-Edu | 37.5% | 100% | 50.0% | 62.5% |
| LongCrawl64 | 25.0% | 100% | 62.5% | 62.5% |

两个来源均存在旧目标读取困难，不能简单归结为其中一个数据集。更新事实位于最后一块，与较早事实的差异提示时距、键绑定或提示/读取格式值得校准，但本轮没有做这些因素的因果消融，不能确定原因。

## 资源与审计

- 完成64条六块轨迹、128个查询前向，两个GPU各32条；没有反事实标签或新训练模型。
- 六块rollout平均4.118秒；全过程最大allocated/reserved为 **14.770/18.828 GiB**（包含评分克隆）。
- 两分片采集墙钟513.418/501.100秒，另有数据构建、加载和其他开销。GPU前向时间不包含缓存CPU搬运与完整字节哈希，不等于整个流程耗时。
- 独立审计重算Q分数并核验行数/身份/标签、来源唯一性与零匹配、相同checkpoint权重哈希、父缓存无变化和确认集未评分；审计通过。
- 全部精确数值、源/输出哈希见 [q_summary.json](q_summary.json) 和 [execution_audit.json](execution_audit.json)。

## 实现与复现

新增独立构造器 `tasks/build_cbf_memory_value.py`、采集/汇总 `tasks/cbf_memory_value.py`、双卡入口 `scripts/run_cbf_memory_value.sh`、完成审计 `scripts/audit_cbf_memory_value.py` 和测试 `tests/test_cbf_memory_value.py`。新增可选CPU扫描依赖 `requirements-memory-value.txt`（pyahocorasick 2.1.0）。模型定义、原runtime、baseline训练配置未修改。

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
# 已有运行的只读审计
/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python -m scripts.audit_cbf_memory_value \
  --root /home/ctj/cbf_ttt_memory_value_20261009 \
  --output /home/ctj/cbf_ttt_memory_value_20261009/execution_audit.json
# 新一次完整构建/运行必须使用全新ROOT，且会重新排除旧来源
ROOT=/home/ctj/cbf_ttt_memory_value_NEW_RUN bash scripts/run_cbf_memory_value.sh
```

若要精确重评分同一批数据，应使用归档设计对应的服务器 `data/scenes.jsonl` 与 `data/design.json`，新建输出文件，执行 `python -m tasks.cbf_memory_value collect --stage q --shard 0|1 --model ... --data ... --design ... --output ...`。不可覆盖已有结果；从源重新构建会把本批纳入旧来源排除，因此不保证相同场景。

服务器原始目录 `/home/ctj/cbf_ttt_memory_value_20261009`。GitHub仅保存代码、设计/来源元数据、聚合和审计，不保存原文、token或逐条结果。

## 已完成、未完成与后续

已完成：新场景与数据审计、21项测试、完整Q、独立结果审计及记录。按预注册Q失败分支结束，不下调门槛、不筛掉难场景。

未执行：writer smoke/重复资源检查、V内容价值、F遗忘取舍、封存confirm、S局部动作、控制器训练和端到端联合训练。相关V/F代码已实现但没有本轮4B效果验证，不能称作实验通过。

下一步应另行登记小规模任务可读性校准：用开发来源做短/长时距、单键/双键与回答格式的对照，确定能够可靠读取事实的协议，再用新来源冻结完整价值验证。该建议尚未执行；不应把本轮Q失败直接归因于快速记忆或遗忘机制。
