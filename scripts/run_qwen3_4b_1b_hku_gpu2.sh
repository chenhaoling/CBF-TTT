#!/usr/bin/env bash
# Build the equal-token 1B corpus, validate it, then launch the tested 2x5090 recipe.
set -euo pipefail

CBF_PROJECT_ROOT=${CBF_PROJECT_ROOT:-/home/ctj/cbf_ttt_verify_20260923}
CBF_DATA_ROOT=${CBF_DATA_ROOT:-/home/ctj/data/cbf_ttt_1b}
CBF_MODEL_DIR=${CBF_MODEL_DIR:-/home/ctj/models/Qwen3-4B}
CBF_LONGCRAWL_SHARD=${CBF_LONGCRAWL_SHARD:-/home/ctj/data/cbf_ttt_pilot/longcrawl64/train/0-of-256.parquet}

source /home/ctj/miniconda3/etc/profile.d/conda.sh
conda activate cbf_ttt_train_py311
cd "$CBF_PROJECT_ROOT"
export PYTHONPATH="$CBF_PROJECT_ROOT"
mkdir -p "$CBF_DATA_ROOT"

echo "$(date -Is) Building 1B Qwen-token corpus"
if [[ ! -s "$CBF_DATA_ROOT/fineweb_edu.jsonl.meta.json" ]]; then
  python -m scripts.build_fineweb_pretrain_pilot \
    --tokenizer "$CBF_MODEL_DIR" \
    --output "$CBF_DATA_ROOT/fineweb_edu.jsonl" \
    --max-records 81381 --max-documents 10000000 \
    --seq-len 6144 --chunk-size 4096
fi
if [[ ! -s "$CBF_DATA_ROOT/longcrawl64.jsonl.meta.json" ]]; then
  python -m scripts.build_longcrawl_pretrain_pilot \
    --input "$CBF_LONGCRAWL_SHARD" \
    --tokenizer "$CBF_MODEL_DIR" \
    --output "$CBF_DATA_ROOT/longcrawl64.jsonl" \
    --max-records 81381 --max-documents 100000 \
    --seq-len 6144 --chunk-size 4096
fi

python - "$CBF_DATA_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name in ("fineweb_edu", "longcrawl64"):
    source = root / f"{name}.jsonl"
    metadata = json.loads((root / f"{name}.jsonl.meta.json").read_text())
    assert source.is_file() and source.stat().st_size > 0, source
    assert metadata["records"] == 81381, (name, metadata["records"])
    assert metadata["min_token_count_with_eos"] == 6144, (name, metadata)
    assert metadata["max_token_count_with_eos"] == 6144, (name, metadata)
    assert metadata["total_token_count_with_eos"] == 500004864, (name, metadata)
    with source.open() as stream:
        rows = sum(1 for _ in stream)
    assert rows == 81381, (name, rows)
print("Both sources contain 81,381 records and 500,004,864 Qwen tokens each")
PY

if [[ ! -s "$CBF_DATA_ROOT/mixed_1b.jsonl" ]]; then
  python -m scripts.merge_pretrain_pilot \
    --fineweb "$CBF_DATA_ROOT/fineweb_edu.jsonl" \
    --longcrawl "$CBF_DATA_ROOT/longcrawl64.jsonl" \
    --output "$CBF_DATA_ROOT/mixed_1b.jsonl"
fi

python - "$CBF_DATA_ROOT/mixed_1b.jsonl" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open() as stream:
    rows = sum(1 for _ in stream)
assert rows == 162762, rows
print(f"Validated {rows} merged records, {rows * 6144} nominal Qwen tokens")
PY

echo "$(date -Is) Starting two-GPU Qwen3-4B In-Place TTT pretraining"
CUDA_VISIBLE_DEVICES=0,1 bash train.sh tasks/train_torch.py \
  configs/pretrain/qwen3_4b_1b.yaml \
  --data.train_path "$CBF_DATA_ROOT/mixed_1b.jsonl" \
  --model.model_path "$CBF_MODEL_DIR"
echo "$(date -Is) Pretraining finished"
