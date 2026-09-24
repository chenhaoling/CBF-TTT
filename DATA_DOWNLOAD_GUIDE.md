# CBF-TTT 训练与评测数据下载指南

本文针对 Qwen3-4B、FineWeb-Edu + LongCrawl64 等量 Qwen token 的 1B 短程继续预训练，以及 ZsRE、MMLU、RULER、RULER/QA、LongMemEval-S、LoCoMo 评测。下载的数据、模型和检查点应存放在仓库之外；GitHub 仓库只保存代码和小型实验记录。

## 0. 网络受限的 8 卡 NPU 机器

在 NPU 机器上从用户仓库拉取代码：

```bash
git clone https://github.com/chenhaoling/CBF-TTT.git
cd CBF-TTT
git rev-parse HEAD
```

再在能访问 Hugging Face 的机器上运行第 1–3 节下载与构建数据。如果 NPU 机器只能访问 GitHub，**`git clone` 只能得到代码，得不到数 GB 的模型或数据**；将模型目录、训练 JSONL 和评测目录通过允许的内网文件共享、对象存储或可达机器的 `rsync` 复制到 NPU 机器。网络访问策略由机器管理员决定，不应把这些文件提交到 Git。

下文在仓库根目录运行命令，并使用示例目录 `/data/cbf_ttt`。按实际挂载点修改路径。需要 Python 3.11，以及与目标机器匹配的 PyTorch；数据准备依赖 `datasets`、`huggingface_hub`、`transformers`、`pyarrow`、`tiktoken`。**不要在 NPU 机器运行 [`scripts/setup_qwen3_4b_train_env.sh`](scripts/setup_qwen3_4b_train_env.sh)**：它安装的是 CUDA 13 版 PyTorch。NPU 训练还需另行配置匹配 CANN 的 `torch_npu` / VeOmni 环境，并先做短程兼容性与吞吐验证。

```bash
python -m pip install datasets huggingface_hub transformers==4.57.3 pyarrow tiktoken
mkdir -p /data/cbf_ttt/models /data/cbf_ttt/train /data/cbf_ttt/eval
```

## 1. Qwen3-4B 基座模型

从 [Qwen 官方 Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) 下载权重和 tokenizer。脚本默认固定到本项目试跑使用的 revision `1cfa9a7208912126459214e8b04321603b3df60c`；模型约 8 GB，下载前检查空间和网络。项目没有使用 7B 模型。

```bash
python -m scripts.download_qwen3_4b --output /data/cbf_ttt/models/Qwen3-4B
```

## 2. 训练语料：FineWeb-Edu + LongCrawl64

公开来源分别为 [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 的 `sample-10BT` 配置和 [manifestai/longcrawl64](https://huggingface.co/datasets/manifestai/longcrawl64)。它们是用户指定的公开替代语料，**不是**原论文未公开的原始 20B token 数据。这里的 `1B` 是用 Qwen3-4B tokenizer 计数的总训练 token，两个来源各约 0.5B。

FineWeb-Edu 由脚本按需流式读取，不需要先下载整个数据集。LongCrawl64 先下载发布方的 parquet 分片；下例取首个 `train/0-of-256.parquet`，约 11 GB。如实际构建时不足 81,381 条记录，可继续下载 `train/1-of-256.parquet` 等，并按顺序传给构建脚本的 `--input`。优先使用发布方的文本列；若某分片只有 GPT-2 TikToken ID，脚本会先解码，再使用 Qwen tokenizer 分词，不能把原 ID 直接用于 Qwen 训练。

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="manifestai/longcrawl64",
    repo_type="dataset",
    filename="train/0-of-256.parquet",
    local_dir="/data/cbf_ttt/train/longcrawl64_source",
)
print(path)
PY

python -m scripts.build_fineweb_pretrain_pilot \
  --tokenizer /data/cbf_ttt/models/Qwen3-4B \
  --output /data/cbf_ttt/train/fineweb_edu.jsonl \
  --max-records 81381 --max-documents 10000000 \
  --seq-len 6144 --chunk-size 4096

python -m scripts.build_longcrawl_pretrain_pilot \
  --input /data/cbf_ttt/train/longcrawl64_source/train/0-of-256.parquet \
  --tokenizer /data/cbf_ttt/models/Qwen3-4B \
  --output /data/cbf_ttt/train/longcrawl64.jsonl \
  --max-records 81381 --max-documents 100000 \
  --seq-len 6144 --chunk-size 4096

python -m scripts.merge_pretrain_pilot \
  --fineweb /data/cbf_ttt/train/fineweb_edu.jsonl \
  --longcrawl /data/cbf_ttt/train/longcrawl64.jsonl \
  --output /data/cbf_ttt/train/mixed_1b.jsonl
```

每个来源应得到 81,381 行；合并文件 162,762 行，按最大序列长度计算为 `162762 × 6144 = 1,000,009,728` 个**名义** token。实际应以两个 `*.jsonl.meta.json` 的 `total_token_count_with_eos` 之和为准，并检查 `records`、最短/最长记录以及 `wc -l /data/cbf_ttt/train/*.jsonl`。在 hku-gpu2 的一次真实构建中，少量记录经 decode/re-encode 后为 6143 token，实测总数为 1,000,009,680，仍高于 1B。构建脚本会重新编码验证记录长度；元数据中的计数比源站以其他 tokenizer 标注的“10B”更适合本实验。脚本目前不能从中断处精确续写，建议在 `tmux` 中运行并保留源分片。

两卡 5090 已试跑的 [`configs/pretrain/qwen3_4b_1b.yaml`](configs/pretrain/qwen3_4b_1b.yaml) 使用固定路径 `/home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl`。若在其他目录训练，覆盖 `--data.train_path /data/cbf_ttt/train/mixed_1b.jsonl` 和 `--model.model_path /data/cbf_ttt/models/Qwen3-4B`。**8 卡 NPU 不应直接沿用这份两卡配置的 `global_batch_size=2`、`max_steps=81381` 和 CUDA 环境**；先完成 NPU 移植和小规模测试，再按实际全局 batch 重新计算步数及数据边界。

## 3. 测试与评测数据

运行下载脚本会在 `/data/cbf_ttt/eval/MANIFEST.json` 写出来源、文件大小和 SHA256，便于在转运到 NPU 机器后校验。

```bash
python -m scripts.download_cbf_eval_data --output /data/cbf_ttt/eval
```

| 评测集 | 下载入口 | 本项目得到的文件或方式 |
|---|---|---|
| ZsRE | [KnowEdit 的 ZsRE 测试文件](https://huggingface.co/datasets/zjunlp/KnowEdit) | `zsre/benchmark/ZsRE/ZsRE-test-all.json`；这是知识编辑版本，需要固定评测协议 |
| MMLU | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | `mmlu/dev.jsonl`、`mmlu/test.jsonl`，配置 `all` |
| LongMemEval-S | [longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | `longmemeval_s/longmemeval_s_cleaned.json`；约 277 MB |
| LoCoMo | [snap-research/locomo](https://github.com/snap-research/locomo) | `locomo/locomo10.json`；本项目文本评测使用文字和图片 caption |
| RULER | [OpenCompass RULER 数据配置](https://github.com/open-compass/opencompass/tree/main/opencompass/configs/datasets/ruler) | 按 `eval_config/ruler_*.py` 在评测时生成，不由下载脚本取固定 JSON |
| RULER/QA | 同上 | 使用 RULER 的 SQuAD/HotpotQA 子任务；入口 [`eval_config/ruler_qa_4k.py`](eval_config/ruler_qa_4k.py) |

```bash
python -m tasks.eval_public_benchmarks --dataset mmlu \
  --data-root /data/cbf_ttt/eval --inspect-only --max-examples 2
python -m tasks.eval_public_benchmarks --dataset zsre \
  --data-root /data/cbf_ttt/eval --inspect-only --max-examples 2
```

RULER 运行需单独安装 OpenCompass；项目的 [`scripts/setup_cbf_eval_env.sh`](scripts/setup_cbf_eval_env.sh) 是针对已建好的 **CUDA** 训练环境克隆的脚本，不能直接作为 NPU 安装方案。下载完成并不表示这些基准已经按官方协议完成评测。尤其完整 LongMemEval-S 的长上下文可能超出当前 6144-token 短程训练模型的有效范围；评测脚本会标记超限，不应把跳过或截断的结果当作正式分数。

## 4. 在网络受限机器上核验文件

有网络的机器下载完成后，可生成文件清单，再用可用的内网传输方式拷贝整个 `/data/cbf_ttt`。GitHub 只用于同步本仓库代码。下列校验命令在源机器与目标机器分别执行，确认传输一致：

```bash
cd /data/cbf_ttt
find models train eval -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
sha256sum -c SHA256SUMS
```

移动目录后，`MANIFEST.json` 中记录的绝对路径可能仍指向旧机器；SHA256 校验以当前文件实际位置为准。模型和数据下载需要满足各来源的许可及使用条款。
