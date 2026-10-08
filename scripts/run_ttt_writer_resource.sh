#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_writer_resource_20261008}
PARITY=${PARITY:-/home/ctj/cbf_ttt_train_infer_20261008/summary.json}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TRAINING_DATA=${TRAINING_DATA:-/home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
"$PYTHON" - "$PARITY" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p['protocol'] == 'train_infer_parity_v1'
assert p['forward_paths'] == 48 and not p['needs_mismatch_investigation']
PY
mkdir "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date -Is > "$ROOT/started_at.txt"
pids=()
for gpu in 0 1; do
  length=$((6144 * (gpu + 1)))
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m scripts.probe_ttt_writer_training \
    --model "$MODEL" --training-data "$TRAINING_DATA" --tokenizer "$TOKENIZER" \
    --length "$length" --output "$ROOT/probe_${length}.json" > "$ROOT/probe_${length}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status != 0 )); then exit 1; fi
date -Is > "$ROOT/completed_at.txt"
