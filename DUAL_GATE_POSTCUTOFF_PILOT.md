# Qwen3 发布后文档：候选写入收益试点

预先固定于 2026-09-27，**在读取本轮标签之前**。目标是核查 `ΔW` 对下一个未见 continuation 是否有正收益，并把当前写入 `g` 与旧会话记忆保留 `α` 分开。沿用最终 Qwen3-4B In-Place TTT 1B-token checkpoint、4096-token TTT chunk、两张 5090 和稀疏第二边界的 `0/0.5/1` 网格。该协议独立标识为 `joint_postcutoff_v1`，原 baseline 与旧标签不变。

## 数据来源和构造

使用 arXiv 官方 API 检索 **2026-09-01 至 2026-09-26 首次提交**、`cs.CL` 类别论文，下载官方 PDF，由 `pdftotext` 提取正文。Qwen 团队于 2025-04-29 发布 Qwen3；首次提交时间晚于该日期，因此这些版本不可能进入已发布的基础模型训练。仍不能排除作者此前公开过部分相似内容。把论文标题在本次 1B 混合续训语料中作一次精确扫描，并保存每篇的 arXiv ID、首投时间、PDF URL、原 PDF 与提取文本 SHA256、Qwen token 长度和扫描结果。原始全文、PDF、逐样本场景与标签只保留本机和 hku-gpu2，不上传 GitHub。

从 API 按首投时间降序返回的前 300 篇候选中，先以固定种子 118 打乱顺序，再从满足至少 4256 Qwen tokens 的**不同论文**依次取 48 篇，组成 12 个独立源组，每组四篇 A/B/C/D。每篇前 4096 token 为已见 chunk，后 32 token 作查询前缀，再后 128 token 为评分目标。`both_relevant` 用 A/B、评 A/B 续写；`old_only` 用 A/C、评 A 续写；`new_only` 用 A/B、评 B 续写；`neither_relevant` 用 A/C、评 D 续写。按源组 train/dev/test=8/2/2；只在第二个完整 chunk 后取 3×3 标签，前 4096 token 按原始 `11` 写入。先做一条 smoke，记录每条耗时和峰值显存，再运行 48 条。不按结果挑论文或删除失败样本；下载/解析失败和不足长度的论文计入构造漏斗。

## 判据

主要写入效应固定旧记忆保留 `α=1`：`G_write=J(1,0)−J(1,1)`，正值表示写入当前 `ΔW` 降低 NLL。每个源组只计一次：

- 有用写入：`new_only` 的 `G_write>0.005` NLL 至少 **3/12** 源组，且 12 组平均 `G_write>0`。
- 有害写入：`old_only` 或 `neither_relevant` 的 `G_write<−0.005` 至少 **3/12** 个不同源组。

只有两项均通过，才考虑扩大写入门控标签。旧记忆保留另在 `g=0` 下比较 `G_keep=J(0,0)−J(1,0)`：`old_only` 中正值对应保留有益，`neither_relevant` 中负值对应清除有益。两种方向都在至少 3/12 源组超过绝对 0.005 NLL，才有双输出控制器的初步监督依据。此处阈值是工程筛选，不是显著性判定。先验还需检查 `11`、`10` 和中间点的相对表现、每条标签耗时/峰值显存，以及同组条件是否严格共享文档。若写入阶段门失败，暂停正式标签及控制器训练，先查候选更新的任务相关性与优化尺度。

## 限制

PDF 转文本含公式、排版和引用噪声，可能影响语言模型 NLL；论文体裁也不能代表日常会话。首投日期和本地标题扫描是可核验的来源代理，不能形式化证明全文与所有预训练语料零重合。dev/test 在这个探索试点中会被查看，以后正式测试必须新取论文。两条件可能共享 B 的续写，阶段门按独立源组去重。

## 运行命令

在 hku-gpu2 的 `/home/ctj/cbf_ttt_joint_exp_20260927` 和 `cbf_ttt_train_py311` 环境执行，先下载、审计，再单场景 smoke。完整标签仅在 smoke 达标后用两个 tmux worker 分别运行 shard 0/1：

```bash
ROOT=/home/ctj/cbf_ttt_postcutoff_random_pilot_20260927
FEED=/home/ctj/cbf_ttt_postcutoff_pilot_20260927/api_feed.xml
MODEL=/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt
mkdir -p "$ROOT"
python -m scripts.download_postcutoff_arxiv --tokenizer /home/ctj/models/Qwen3-4B --output "$ROOT/documents.jsonl" --category cs.CL --start 202609010000 --end 202609262359 --target 48 --max-results 300 --min-tokens 4256 --selection-seed 118 --api-feed "$FEED" --pdf-cache /home/ctj/cbf_ttt_postcutoff_pilot_20260927/pdf_cache
python -m scripts.audit_postcutoff_corpus --metadata "$ROOT/documents.jsonl.meta.json" --corpus /home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl --output "$ROOT/title_audit.json"
python -m tasks.build_cbf_natural_scenarios --tokenizer /home/ctj/models/Qwen3-4B --data "$ROOT/documents.jsonl" --output "$ROOT/scenarios.jsonl" --train-groups 8 --dev-groups 2 --test-groups 2 --context-tokens 4096 --query-tokens 32 --answer-tokens 128 --seed 118 --protocol joint_postcutoff_v1 --group-prefix postcutoff-paper
python -m scripts.shard_cbf_scenarios --input "$ROOT/scenarios.jsonl" --output-dir "$ROOT/shards" --shards 2
head -n1 "$ROOT/shards/train_shard0.jsonl" > "$ROOT/smoke_one.jsonl"
CUDA_VISIBLE_DEVICES=0 python -m tasks.cbf_ttt collect-joint --model "$MODEL" --dtype bfloat16 --data "$ROOT/smoke_one.jsonl" --split train --output "$ROOT/smoke_one_labels.jsonl" --grid 0,0.5,1 --every 2
bash scripts/run_cbf_joint_pilot.sh "$MODEL" "$ROOT" 0 0
bash scripts/run_cbf_joint_pilot.sh "$MODEL" "$ROOT" 1 1
python -m scripts.summarize_cbf_joint_pilot --labels "$ROOT"/{train,dev,test}_shard{0,1}_joint_labels.jsonl --output "$ROOT/summary.json"
python -m scripts.evaluate_cbf_postcutoff_gate --labels "$ROOT"/{train,dev,test}_shard{0,1}_joint_labels.jsonl --output "$ROOT/gate.json"
```

两条 `run` 命令的最后参数分别指定 GPU0 和 GPU1，须在不同会话中并发执行。首次 smoke 输出文件与正式分片文件不同。所有采集脚本拒绝覆盖已存在的正式标签文件；复现实验时应使用新目录。

远程 hku-gpu2 对官方 arXiv 元数据 API 返回 HTTP 406。实际运行时先在本机用上面相同查询 URL 保存 Atom XML，传到 `FEED` 路径；脚本核对类别和日期，并在元数据中保存原始 XML 的 SHA256。PDF 仍由 hku-gpu2 从官方地址下载。上述阶段门不变。

第一轮链路预检的下载器曾直接取时间排序后前 48 篇合格论文，与本节预设的随机选取不一致。该轮 48 条标签虽已采集，但**在未查看损失结果前发现偏差**，其标签只作为资源与代码链路预检；不用于阶段门结论。修正后另建 `cbf_ttt_postcutoff_random_pilot_20260927` 目录，用同一 API XML、`--selection-seed 118` 及独立标签文件执行本节主要试点；第一轮的来源排序不得用于决定阈值或挑选样本。

## 执行状态

随机主试点已于 2026-09-28 完成。写入有益仅 1/12 源组、平均收益 −0.01193 NLL，未达预设 3/12 且均值为正的要求；保留轴也未通过。正式标签和控制器暂停。完整结果与精度复核见 [`主试点报告`](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/joint_postcutoff_random_pilot/REPORT.md)。
