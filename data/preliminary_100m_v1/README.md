# CBF-TTT preliminary 100M public corpus v1

This directory publishes the independently audited `preliminary_100m`
FineWeb-Edu/LongCrawl64 corpus used by the Qwen3-4B-Base matched baseline.
The compressed archive is split into ordinary Git objects below GitHub's
100 MiB per-file limit. Git LFS is not required.

## Contents

| Stage | Length | Records | Training tokens |
|---|---:|---:|---:|
| Stage 1 | 32,768 | 1,728 | 56,623,104 |
| Stage 2 | 131,072 | 320 | 41,943,040 |
| Total | — | 2,048 | 98,566,144 |

Each stage is exactly 50% FineWeb-Edu and 50% LongCrawl64 by Qwen training
tokens. Stage 2 continues both source cursors from Stage 1. The independent
audit passes all ten registered checks, including artifact hashes, ordered
tokens, cross-stage continuity, tokenizer assets and absence of plaintext in
provenance. The corpus contains zero synthetic EOS tokens.

Source revisions and the Qwen3-4B-Base tokenizer identity are bound in the
included `manifest.json`. Source content remains subject to the original
FineWeb-Edu and LongCrawl64 terms.

## Download and unpack

```bash
git clone https://github.com/chenhaoling/CBF-TTT.git
cd CBF-TTT
bash data/preliminary_100m_v1/unpack.sh /data/cbf_ttt/preliminary_100m_v1
```

The unpacker verifies every part, reconstructs the archive, verifies the full
archive SHA256, extracts into a new destination and then verifies all four
JSONL/provenance artifact hashes from `manifest.json`.

Required system commands: `sha256sum`, `tar`, `zstd`, `python3`.

After unpacking:

```text
/data/cbf_ttt/preliminary_100m_v1/
├── independent_audit.json
├── manifest.json
├── stage1_32k.jsonl
├── stage1_32k.provenance.jsonl
├── stage2_128k.jsonl
└── stage2_128k.provenance.jsonl
```

## 8xA100 Stage-1 launch

The launcher detects the eight visible CUDA devices. Keep global batch size
64, so eight ranks perform eight accumulation microsteps per optimizer step.
Replace the model and output paths below.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
MODEL=/data/models/Qwen3-4B-Base
DATA=/data/cbf_ttt/preliminary_100m_v1
OUT=/data/cbf_ttt/runs

bash train.sh tasks/train_torch.py \
  configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage1_no_ttt.yaml \
  --model.model_path "$MODEL" \
  --data.train_path "$DATA/stage1_32k.jsonl" \
  --train.output_dir "$OUT/100m_stage1_no_ttt"

bash train.sh tasks/train_torch.py \
  configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage1_ttt.yaml \
  --model.model_path "$MODEL" \
  --data.train_path "$DATA/stage1_32k.jsonl" \
  --train.output_dir "$OUT/100m_stage1_ttt"
```

Run the two arms sequentially so they do not contend for the same GPUs. Stage
2 must start each arm from its own converted and audited Stage-1 HF checkpoint;
do not cross the checkpoints between arms. The `2x5090` substring records the
original validation hardware, but the YAML does not fix world size: `train.sh`
uses `NPROC_PER_NODE=8` on the A100 host while preserving the registered
global batch and token budget.

This dataset supports preliminary matched validation. It is not the paper's
private corpus and is not a 35B-token paper-scale reproduction.
