#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_zsre_forgetting_controller_loso_v2}
DATA_ROOT=${DATA_ROOT:-/home/ctj/cbf_ttt_zsre_hard_retain_forgetting_v1_data}
SOURCE_ROOT=${SOURCE_ROOT:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal}
SCOPE_ROOT=${SCOPE_ROOT:-/home/ctj/cbf_ttt_zsre_decomposed_scope_gate_v1/formal}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

if [[ -e "$ROOT" ]]; then echo "ROOT already exists: $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"; git rev-parse HEAD > "$ROOT/code_commit.txt"; date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT

"$PYTHON" -m unittest tests.test_zsre_forgetting_controller \
  tests.test_zsre_forgetting_controller_v2 tests.test_zsre_hard_retain_forgetting

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_zsre_forgetting_controller_v2 \
  --data-root "$DATA_ROOT" --source-root "$SOURCE_ROOT" --scope-root "$SCOPE_ROOT" \
  --output "$ROOT/formal" --model "$MODEL" --tokenizer "$TOKENIZER" --batch-size 128 \
  > "$ROOT/formal.log" 2>&1

"$PYTHON" -m scripts.audit_zsre_forgetting_controller_v2 \
  --root "$ROOT/formal" --data-root "$DATA_ROOT" --source-root "$SOURCE_ROOT" \
  --scope-root "$SCOPE_ROOT" --output "$ROOT/execution_audit.json"

echo completed_zsre_forgetting_controller_loso_v2 > "$ROOT/terminal_status.txt"
date -Is > "$ROOT/completed_at.txt"
