# 固定槽位的记录干扰实验 v1（2026-10-09，执行前登记）

## 目的与解释范围

上一轮原stable目标QA八选一31.25%，重建仅两条记录背景后100%，存在背景变化混杂。本轮固定QA、原始六块4096-token布局和目标/锚点，局部改变其他设备的记录，区分记录负载与位置影响。只复用原dev0–7的16个stable双生版本，不评分confirm，不训练、不写M，不声称四场景或记忆价值已通过。

## 固定构造

原stable每块两条人工记录，目标和锚点均在第2块；可干预槽为第1/3/4/5/6块各两条，共10条。原背景前缀、每块header、所有位置、目标与锚点整块保持逐token相同。通过原suffix重编码确认每行token边界，保留每行最后一个token（可能包含标点/换行）；只替换该行正文token。任何边界不匹配则构建失败，不能静默调整。

主占位方式natural：用同一来源原始背景相同绝对位置的token替换该槽正文，等长；其他位置不变。它改变了预定干预槽内容，不意味着全部自然文本逐token相同。辅占位newline：相同位置填等量换行token，检测主结果是否依赖填充方式；换行分布本身是混杂，不能当成不存在信息的理想空白。

固定10个条件，每个16会话，共160轨迹/320个QA查询：

| 条件 | 保留的干扰块 | 干扰条数 | 占位方式 |
|---|---|---:|---|
| original10 | 1,3,4,5,6 | 10 | 原始输入精确桥接 |
| natural0 | 无 | 0 | natural |
| natural_before2 | 1 | 2 | natural |
| natural_early2 | 3 | 2 | natural |
| natural_late2 | 6 | 2 | natural |
| natural_late4 | 5,6 | 4 | natural |
| natural_after8 | 3,4,5,6 | 8 | natural |
| newline0 | 无 | 0 | newline |
| newline_late2 | 6 | 2 | newline |
| rebuilt_bridge | 上轮long_far/dual/target_last | 0 | 上轮输入精确桥接，不用于固定槽位因果差 |

0→2→4→8是从末段向前扩展的嵌套负载，数量和覆盖位置共同变化，不能称为位置不变的纯数量曲线。同为2条的before/early/late比较也使用各位置原有键值，位置与键身份未完全正交；不作唯一机制归因。original10与after8只差第1块两条干扰。

## 检查、指标与停止规则

- 从原始parquet按行号读取并复核来源SHA；原始/上轮设计与数据SHA固定。按组分片，两个版本仅差一个目标颜色token，八选一标签每条件/来源域平衡。
- 160条完整性、每块长度、槽外token不变、目标块不变、查询一致；所有条件只使用dev stable。
- 两类桥接各16会话×2查询，合计64个评分；旧model/weights一致，NLL误差≤1e-5，八选一和全词表首token预测必须相同。失败则不解释新条件。
- 无写入完整KV；每query克隆父缓存，缓存字节哈希和模型权重/版本不变；逐轨迹/查询记录耗时、allocated/reserved。
- 分条件、域、来源组报告目标/锚点的八选一、NLL、全词表首token；逐组配对差，不能将版本/布局当独立样本。补八类预测混淆与错误是否落在干扰颜色集合，后者仅线索，不等于定位到特定记录。
- 主比较natural0−original10；位置比较late2−before2、late2−early2；负载比较late2−0、late4−late2、after8−late4、original10−after8；占位稳健性比较newline0−natural0及两占位的late2−0差。八选一正值有利、NLL负值有利。
- 描述性“去除干扰后可读”门槛：natural0的target/anchor整体和每域均≥75%。仅表示此dev条件可读，不开新V/F；主收益须两域方向一致才称跨域一致线索。newline结果单独报告，不事后择优占位。
- 本轮只运行固定矩阵，结束后报告，不自动训练控制器或联合模型。

## 实现与运行

新增 `tasks/cbf_record_interference.py` build/collect/summarize，复用原背景读取、模型加载、session和评分；新增构造/负测试、独立汇总校验及双GPU shell入口。baseline代码/配置不改，无新依赖。原始文本/token/逐条结果仅远程保存，设计/聚合/资源公开归档。

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_record_interference_20261009 bash scripts/run_cbf_record_interference.sh
```

环境变量SOURCE/READABILITY分别指向原memory_value和readability目录；MODEL为1B训练最终Qwen3-4B。ROOT必须新目录。两卡预计约20分钟仅参考上轮耗时；不以估计代替真实墙钟。构造/单元测试先通过，再双卡采集；任何失败保留标记和日志。
