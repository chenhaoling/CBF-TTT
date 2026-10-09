# 无颜色重合四场景读出校准：完成报告

## 结论与决策

实验与独立审计均完成。两种固定读出格式均未达到预登记的全部16格准确率≥75%门槛，未选定格式，状态 `stopped_by_disjoint_Q`。原QA失败14/16格，binding失败12/16格，两者最差格均为0%。依照事前计划，本轮到此停止：未构建新来源、未写入快速记忆M、未进入记忆价值V/遗忘F测试、未训练控制器或启动联合训练，旧confirm仍未评分。

明确设备绑定指令改善部分读取，但四场景中的旧事实保留仍不可靠。本轮使用完整KV、M=0，不能检验或否定遗忘机制收益。它暴露了当前checkpoint在此输入/读出协议下的能力瓶颈。干扰颜色分布也从原分布变为四色循环，因此不能把与旧实验的全部差异归因为偶然颜色重合，更不能据此断言以前所有正确答案都是猜中。

## 设计与实现假设

- 最终1B训练checkpoint的Qwen3-4B，BF16，TF32关闭，冻结backbone；hku-gpu2两张RTX5090，每卡40条。
- 沿用已观察dev0–7：FineWeb-Edu、LongCrawl64各4组，四场景×双生=64个clean上下文。双生共享来源，不能将64会话视为64个独立来源样本；这是开发集校准，无独立确认或显著性主张。
- 保留原六块4096-token背景、设备键、记录数量/顺序和事实位置。每组保护双生的两个目标值、锚点值、correction旧值；无关键记录只能用剩余四色。自然背景中的颜色词未排除。
- 重编码只允许颜色token变化，不删除难例。独立审计检查624条干扰记录、144条合法记录，实际572个颜色token发生变化，人工干扰的受保护颜色重合为0。
- clean上下文分别查询原QA与固定binding指令，目标/锚点各一次；另用原组0、2的16个上下文做32查询精确桥接。共80上下文/288查询。桥接子样本不等同于全部clean样本，不能直接用两个总体均值构造完整配对效应。
- 每格式4场景×2域×2问题=16格，每格8会话/4来源组，全部≥75%才合格；按最弱格最大选择，平局保留QA。没有事后新增模板、改门槛或删样本。

## 八选一正确率

各总体单元16会话/8来源组。这里是固定八种颜色的候选分类，不能称为自由生成问答准确率。

| 场景 | 目标 QA | 目标 binding | 锚点 QA | 锚点 binding |
|---|---:|---:|---:|---:|
| stable | 0.00% | 25.00% | 31.25% | 62.50% |
| correction | 100.00% | 100.00% | 31.25% | 68.75% |
| recent1_distractor | 25.00% | 25.00% | 31.25% | 56.25% |
| recent2_distractor | 18.75% | 43.75% | 12.50% | 50.00% |

### 分域门槛表

| 域 | 场景 | 目标 QA | 目标 binding | 锚点 QA | 锚点 binding |
|---|---|---:|---:|---:|---:|
| fineweb | stable | 0.00% | 25.00% | 37.50% | 75.00% |
| fineweb | correction | 100.00% | 100.00% | 25.00% | 100.00% |
| fineweb | recent1_distractor | 0.00% | 0.00% | 37.50% | 62.50% |
| fineweb | recent2_distractor | 25.00% | 62.50% | 0.00% | 50.00% |
| longcrawl | stable | 0.00% | 25.00% | 25.00% | 50.00% |
| longcrawl | correction | 100.00% | 100.00% | 37.50% | 37.50% |
| longcrawl | recent1_distractor | 50.00% | 50.00% | 25.00% | 50.00% |
| longcrawl | recent2_distractor | 12.50% | 25.00% | 25.00% | 50.00% |

### NLL与全词表首token

NLL为正确颜色token的负对数似然；首token率只检查全词表最大概率token是否等于正确颜色token，不是多token语义生成评测。

| 场景 | 问题 | QA NLL | binding NLL | QA 首 token 正确率 | binding 首 token 正确率 |
|---|---|---:|---:|---:|---:|
| stable | target | 4.7309 | 2.1974 | 0.00% | 25.00% |
| stable | anchor | 3.0297 | 1.5175 | 0.00% | 62.50% |
| correction | target | 2.0015 | 0.6727 | 25.00% | 100.00% |
| correction | anchor | 3.0842 | 1.6911 | 0.00% | 68.75% |
| recent1_distractor | target | 4.0079 | 1.9235 | 18.75% | 25.00% |
| recent1_distractor | anchor | 2.9665 | 1.5272 | 0.00% | 56.25% |
| recent2_distractor | target | 3.3681 | 1.5953 | 0.00% | 43.75% |
| recent2_distractor | anchor | 3.0568 | 1.5598 | 0.00% | 50.00% |

逐来源组结果和binding−QA配对差见 `summary.json` 的 `group_means` 与 `binding_minus_qa_group_differences`，不把格式或双生当独立复制。

## 时间、显存与审计

| 项目 | 实测 |
|---|---:|
| 开始（UTC+8） | 2026-10-09 18:48:08 |
| 结束（UTC+8） | 2026-10-09 18:59:10 |
| 总墙钟 | 662秒（11分02秒） |
| 两分片采集 | 638.672 / 618.778秒 |
| rollout均值 | 4.114秒 |
| query均值 | 0.104秒 |
| 单卡最高allocated | 14.779 GiB |
| 单卡最高reserved | 18.893 GiB |
| 旧结果桥接 | 32查询，NLL最大误差0，预测相同 |

每轨迹和每查询的时间/显存保存在远程逐条JSONL，公开归档仅保留聚合。没有失败标记，两个分片各40条；以上内存为PyTorch峰值，不是整机总占用。完整运行时长包括构造/加载/测试；query、rollout均值不含全部这些开销。

采集代码提交 `45525be1e59d731b7427f5a566d141455d7bab24`。服务器14项测试通过（2.664秒）。独立脚本重新编码全部干预记录、验证保护记录/精确查询/双生/哈希/桥接，重新汇总原始评分，结果与summary完全相同。模型权重与父缓存不变检查通过，confirm未评分。

最终审计脚本在采集结束后上传执行，不改变已完成的采集代码。此前自动审批服务额度故障阻止了传输；用户继续后正常审批重试成功。一次SCP连接中断后重新下载成功；本地summary/design哈希与远程审计吻合。

## 文件与配置

| 文件 | 实现或变更 |
|---|---|
| `tasks/cbf_disjoint_readout.py` | clean_scene/make_rows/validate/build/summarize/choose_format；受保护颜色排除、固定两格式、逐域门槛和精确桥接 |
| `tasks/cbf_readability.py` | collector增加可选validator/bridge_cells/identity_fields，默认保持旧调用行为 |
| `tests/test_cbf_disjoint_readout.py` | 构造、重合拒绝、16格门槛、完整汇总/篡改拒绝四项测试 |
| `scripts/run_cbf_disjoint_readout.sh` | 双卡构造/采集/汇总、失败标记、完成状态；拒绝覆盖输出目录 |
| `scripts/audit_cbf_disjoint_readout.py` | 已有结果独立重建检查和汇总重算，生成execution_audit.json |
| `DISJOINT_READOUT_PLAN.md` | 执行前协议及完成记录 |
| 本目录 | REPORT、RUN_STATUS、design、summary、execution_audit及排除原始数据的.gitignore |
| `README.md` / `MODIFICATION_LOG.md` | 完成入口及完整修改/结果记录 |

新增运行参数为环境变量 `ROOT`（新的输出目录）、`SOURCE`（原memory_value归档）、`PYTHON`、`MODEL`、`TOKENIZER`；默认值见脚本。审计参数 `--root`、`--output`、`--tokenizer`。两种格式及75%门槛属于冻结协议，不通过调参绕过失败。无baseline模型/runtime/训练配置变更，无新依赖，旧collector默认参数保持兼容。

## 复现与只读审计

在保有原checkpoint、tokenizer、原memory_value逐条数据/设计/评分的hku-gpu2环境执行；公开聚合文件不足以独立重建原始背景。使用全新ROOT，避免覆盖本次结果。

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_disjoint_readout_repro \
SOURCE=/home/ctj/cbf_ttt_memory_value_20261009 \
bash scripts/run_cbf_disjoint_readout.sh
```

重做现有结果审计，不运行模型训练：

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python -m scripts.audit_cbf_disjoint_readout \
  --root /home/ctj/cbf_ttt_disjoint_readout_20261009 \
  --output /home/ctj/cbf_ttt_disjoint_readout_20261009/execution_audit_recheck.json \
  --tokenizer /home/ctj/models/Qwen3-4B
```

运行目录 `/home/ctj/cbf_ttt_disjoint_readout_20261009`；tmux `cbf-disjoint-readout-20261009` 已正常结束。原始文本、token、逐条评分和权重保留远程，不上传GitHub。审计原件的脚本哈希来自同内容临时脚本，重跑输出路径可以不同。

## 与研究构思的关系和下一步

附件第一条（全局+last-1/last-2局部遗忘）及第二条（随机1–3处多分支）要求先能可靠读取被保留/清除的事实。本轮只完成该前提的四场景含干扰诊断，没有新增两种机制的效果证据。第三条（独立初始化后端到端联合训练）未启动。背景取自既有预训练语料，绑定事实是构建的人工任务；本轮复用dev，不能当新的独立测试集。

下一项建议先冻结同一批disjoint输入与两种查询，对照原始Qwen3-4B和最终1B checkpoint，保持tokenizer、候选、精度、KV和M=0可比，并核对checkpoint加载/推理路径。该对照能帮助定位checkpoint变化与共同任务/读出瓶颈；在实现不同推理路径时须单列其混杂，不能把差异全算作预训练损伤。本轮没有启动此新实验。

若两者均失败，优先重新验证可测的任务协议与真实键值读取；若原始模型通过而最终模型失败，再定位训练或加载差异。完成能力诊断后才制定新的冻结Q协议/新来源确认，再按Q→V→F门槛开展局部遗忘、多分支和联合训练。当前不扩大标签构造或控制器规模。
