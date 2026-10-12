#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_zsre_contrastive_keys_v1}
SOURCE_STUDY_ROOT=${SOURCE_STUDY_ROOT:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1}
DATA_ROOT=${DATA_ROOT:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/data}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
if [[ -e "$ROOT" ]]; then echo "ROOT already exists: $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"; git rev-parse HEAD > "$ROOT/code_commit.txt"; date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT
"$PYTHON" -m unittest tests.test_zsre_contrastive_keys tests.test_zsre_contextual_readout
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_zsre_contrastive_keys \
  --data-root "$DATA_ROOT" --source-root "$SOURCE_STUDY_ROOT/formal" \
  --output "$ROOT/formal" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --epochs 300 --seed 6151 --batch-size 128 > "$ROOT/formal.log" 2>&1
"$PYTHON" -m scripts.audit_zsre_contrastive_keys \
  --root "$ROOT/formal" --data-root "$DATA_ROOT" --source-study-root "$SOURCE_STUDY_ROOT" \
  --output "$ROOT/execution_audit.json"
echo completed_zsre_contrastive_keys_v1 > "$ROOT/terminal_status.txt"; date -Is > "$ROOT/completed_at.txt"
