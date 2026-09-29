# 当前内容驱动低秩写入器关键试点（2026-09-29）

训练写入器在独立 test 上降低了标题 NLL，但错配论文更新取得几乎相同收益，训练集平均更新更好。因此本轮**未通过内容相关写入阶段门**。这表明单看“比不写入好”会把通用任务适配误判为会话信息写入。双门控控制器和正式反事实扩量继续暂停。

## 预设方法与数据

执行前协议为 [`TASK_AWARE_WRITER_PILOT.md`](../../../../TASK_AWARE_WRITER_PILOT.md)。使用最终 Qwen3-4B 1B-token checkpoint，冻结全部骨干与原始 TTT 参数。每篇新论文从零会话状态读入前 4096 token，缓存原始候选 `D_l`。7 个 TTT 层各训练 rank-8 变换 `D'_l=D_l+B_l(A_l D_l)`，随后限制每层 `||D'_l||≤||D_l||`。初始 B=0，与原始更新等价；总可训练参数 **286,720**。写入器只接收当前论文候选 D，不接收问题、标题元数据或答案。训练用标题答案 NLL 监督 A/B；测试时仅前向生成更新。

为隔离写入能力，本轮是单论文、空旧记忆、KV 清空后的固定标题问题，与之前双论文四条件试点不同；它不能直接证明遗忘门或双门控有效。原始 baseline 运行时和配置保持兼容，所有新行为由 `task_writer_v1` 及新 CLI 显式启用。

从原官方 arXiv feed 前 300 条排除前三批的 **136 个唯一论文 ID**，以种子 121 打乱后筛选。106 篇尝试得到 96 篇：7 篇不足 4256 token、3 篇首块未匹配标题。首投时间为 2026-09-21 06:18:11 UTC 至 2026-09-24 11:38:40 UTC。以种子 122 分为 **64 train /16 dev /16 test**，每篇一个独立源组；与旧源组交集为 0，1B 续训语料精确标题匹配为 0/96。候选特征只由上下文产生，96 份缓存均记录来源、上下文和模型配置哈希。公开来源信息见 [`documents.jsonl.meta.json`](documents.jsonl.meta.json)、[`episodes.jsonl.meta.json`](episodes.jsonl.meta.json) 与 [`title_audit.json`](title_audit.json)。

## 训练与模型冻结

固定 seed=123、AdamW lr=0.001、weight_decay=0.01、batch=1、梯度裁剪 1、5 epochs，共 **320 步**。每 epoch 只用 dev 在 `g=.5/1` 选择 checkpoint 和门值；原始更新对照也只在 dev 选门值。无写入 dev NLL 4.95451；最佳 writer 位于 **epoch 1、g=1**，NLL **4.37044**，15/16 篇改善 >0.005，平均较无写入改善 0.58407，较最佳原始更新改善 0.60361，满足预设 dev 门槛。

后续训练损失继续下降，但 dev 变差，故按原规则选 epoch 1，未按 test 调参。所选 checkpoint SHA256 为 `4fd6be58ae7bfa00c74911e4d4dc24c7ec31ba56cd8d98e003208561b2c05487`；原始对照固定 `g=.5`。选择、曲线与资源见 [`selection.json`](train_seed123/selection.json)、[`history.json`](train_seed123/history.json)、[`profile_summary.json`](train_seed123/profile_summary.json)。

## 独立 test 结果

dev 通过后，首次评分全部 16 篇 test。writer 和错配/平均更新使用冻结的 `g=1`，原始更新使用 dev 选定的 `g=.5`。错配更新为按 test ID 排序后循环错配下一篇论文；平均更新只由 64 篇 train 的 writer 输出计算。

| 固定方法 | 平均标题 NLL（越低越好） |
|---|---:|
| 无写入 `g=0` | **5.15830** |
| 原始 In-Place TTT 候选 | **5.15812** |
| 当前论文的训练写入器 | **4.44029** |
| 另一篇论文的训练写入器 | **4.44186** |
| 训练集平均更新 | **4.42475** |

当前论文 writer 较无写入平均改善 **0.71801 NLL**，16/16 篇改善超过 0.005。然而它对错配论文仅改善 **0.00157 NLL**，低于预设 0.005；相比训练集平均更新则差 **0.01554 NLL**。因此独立 test 阶段门为 **false**。补充网格中 writer `g=.5` 为 4.58468、`g=1` 为 4.44029；这些 test 数字没有用于重选模型或门值。完整聚合见 [`test summary`](test_seed123/summary.json)。

可支持的结论是：任务监督能够学到跨论文的通用标题问答适配；本轮 rank-8 候选变换尚未证明收益需要来自**当前论文的信息**。不能把 16/16 改善解读为双门控成功，或据此训练控制器。固定标题问题、共同更新分量、小样本与低秩容量可能共同造成这一结果；本轮未进一步在已评分 test 上试新损失或超参数。

## 校验与资源

远程 tiny 模型及数据测试通过，覆盖初始等价、范数上限、候选 detach、梯度连接、骨干无梯度、dev 失败拒绝 test，以及成功 gate 后的所有固定对照。真实 smoke 初始 writer `.5/1` 与原始更新 NLL 逐值一致，训练更新可运行且无 OOM。

两张 5090 分片提取候选，平均每篇 **0.894 秒**，最大 reserved **9.096 GiB**；缓存约 **32 GiB**。单张 5090 完成 320 步训练，平均每步 **0.237 秒**，累计步内计时 **75.73 秒**，最大 allocated/reserved **10.793/11.631 GiB**。步内计时不包括模型加载、dev 评分和保存。test 含全部对照平均每篇 **0.503 秒**，最大 reserved **11.938 GiB**。每步/每篇原始日志在远程保留，聚合见 [`resources.json`](resources.json)。

新增 [`audit_cbf_writer_results.py`](../../../../scripts/audit_cbf_writer_results.py) 从同步的逐条数据独立核对源组无交叉、训练仅用 train、320 步预算、96 份候选缓存来源、初始等价、dev 最优 epoch/门值和全部 test 均值/阶段门。全部检查通过，详见 [`audit.json`](audit.json)。test summary SHA256 为 `4f28dc3bacea74cec1623e76e3b15050f2337e3e5c76928e483681127221d3bc`。

## 复现与保存位置

完整构造、提取、smoke、训练及有门槛的 test 命令见 [`MODIFICATION_LOG.md`](../../../../MODIFICATION_LOG.md) 的“内容驱动低秩写入器关键试点”段。远程代码目录为 `/home/ctj/cbf_ttt_joint_exp_20260927`，模型为 `/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt`，环境为 `/home/ctj/miniconda3/envs/cbf_ttt_train_py311`。本轮数据、特征、模型和逐条结果保存于 `/home/ctj/cbf_ttt_writer_pilot_20260929`；writer checkpoint 为 `train_seed123/best.pt`。再次运行需使用新输出目录，代码拒绝覆盖已有结果。

GitHub 只保存实现、预注册协议、公开论文元数据、特征清单哈希和聚合结果。PDF、全文、候选矩阵、writer checkpoint、逐条 dev/test 结果和训练日志不上传。没有新增第三方依赖。

## 后续工作与限制

下一轮应在训练目标中要求“正确论文更新优于错配更新”，并采用答案类型相同但事实内容不同的查询，减少通用标题输出偏移的捷径。可研究仅从训练集估计并去除候选公共分量，或加入正确/错配记忆的配对损失；这些都是待检验方案。必须使用新源组重新固定预算与阶段门，先获得内容相关写入，再恢复四种 `(α,g)` 情况的联合实验。

本轮是单种子、16 篇独立 test、teacher-forced 标题 NLL、人为 KV 清空与单论文输入，尚无自由生成、自然长会话或公开基准的结论。精确标题扫描也不等于全文去重。low-rank writer 为新增算法模块，不能将其结果归为原始 CBF 双门控构思本身的成功或失败。
