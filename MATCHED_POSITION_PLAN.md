# 同键值位置对照 v1（2026-10-09，执行前登记）

## 问题与范围

上一轮证明局部移除无关记录提高dev stable的八选一读取，但位置比较使用不同键值。本轮固定干扰身份与数量，检验相同两条记录在不同位置的作用。仅原dev0–7、16个stable双生，完整KV/M=0、原QA、Qwen3-4B最终1B checkpoint。无新来源、训练、confirm或V/F。沿用natural占位；鉴于上轮占位敏感，结论不外推到所有占位方式。

## 固定矩阵与构造

从原第3块和第6块各取两条记录作为两套donor，保持键、颜色、文字token及内部顺序完全相同。分别放到第1/3/6块的原记录槽，其他所有无关槽使用上一轮natural0原样输入。七个条件为none、d3_p1/d3_p3/d3_p6、d6_p1/d6_p3/d6_p6，共112轨迹、224个QA查询。

目标与锚点在第2块整块不动；每块4096、共6块不动；header、行尾token、槽外背景不动。正文token直接搬运而非重新措辞；目标槽长度和行尾token必须与donor相同，不能截断或补齐。不同位置使用当地原槽，周围自然文本、相对顺序和距离随位置一起改变，不能视为只改变一个抽象时间变量。

每个来源组×版本×donor在三位置是配对；两个donor同属一个来源组，不作为额外独立样本。目标标签每域/条件八类各一次，双生只差一个目标颜色token。

三个精确桥接：none对应上一轮natural0、d3_p3对应natural_early2、d6_p6对应natural_late2，共48轨迹/96评分。桥接优先，模型/权重、输入/QA必须一致，NLL误差≤1e-5且八选一/首token预测相同。失败保留结果但不解释新条件。

## 预定汇总与解释

报告两个问题的八选一、NLL、全词表首token；分条件、域、来源组；逐轨迹/查询耗时、allocated/reserved、父缓存和权重哈希。主对比在每个donor内计算p6−p1、p6−p3，按两个donor等权平均后在8个源组汇总。次要报告每位置的d6−d3与位置×donor交互，以及每条件相对none的差；不选择最好的donor/域/位置冒充主结果。

仅当两个donor、两个域的目标准确率p6−p1和p6−p3全为负，才称本设计下“一致的末段干扰增强”；任一零/反向则不成立。锚点结果单列，不把目标规律推广到其他查询。此描述性规则不是显著性检验。样本为8来源组，不因条件数增加夸大独立样本量。

已有d3_p3锚点未达到此前全条件可读要求；本轮不把位置对照当成四场景QA修复。后续进入条件：先冻结包含干扰的读出协议与四场景Q，使用修复来源排除器构建新来源；target/anchor各场景各域≥75%才进入V；V先验内容干预成立再验证F及联合训练。此次只完成位置矩阵与报告，不自动切换新协议或开启确认集。

## 修改与运行

新增 `tasks/cbf_matched_position.py`、`tests/test_cbf_matched_position.py`、`scripts/run_cbf_matched_position.sh`、完成审计与归档。仅给旧collector新增可选validator和bridge_cells参数，默认值保持原实验路径；复用模型/session/评分/计时，无新依赖或baseline修改。

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_matched_position_20261009 bash scripts/run_cbf_matched_position.sh
```

ROOT必须新目录，REFERENCE指向上一轮record_interference；MODEL/PYTHON沿用旧入口。测试和构造先通过，双卡各56条。约15分钟是上轮吞吐估计，实际时间和峰值另记。公开元数据/聚合，原始文本/token/逐条结果保留远程。
