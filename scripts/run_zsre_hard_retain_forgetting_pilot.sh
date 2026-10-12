#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_zsre_hard_retain_forgetting_pilot_v1}
DATA_ROOT=${DATA_ROOT:-/home/ctj/cbf_ttt_zsre_hard_retain_forgetting_v1_data}
V1_ROOT=${V1_ROOT:-/home/ctj/cbf_ttt_zsre_natural_forgetting_v1_data}
SOURCE_DATA=${SOURCE_DATA:-/home/ctj/data/cbf_ttt_eval/zsre/benchmark/ZsRE/ZsRE-test-all.json}
SOURCE_ROOT=${SOURCE_ROOT:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal}
SCOPE_ROOT=${SCOPE_ROOT:-/home/ctj/cbf_ttt_zsre_decomposed_scope_gate_v1/formal}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

if [[ -e "$ROOT" ]]; then
  echo "ROOT already exists: $ROOT" >&2
  exit 2
fi
if [[ ! -e "$DATA_ROOT" ]]; then
  "$PYTHON" -m tasks.build_zsre_hard_retain_forgetting \
    --source "$SOURCE_DATA" --v1-root "$V1_ROOT" --output "$DATA_ROOT"
fi

mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT

"$PYTHON" -m unittest \
  tests.test_zsre_hard_retain_forgetting \
  tests.test_zsre_natural_forgetting_pilot \
  tests.test_zsre_decomposed_scope_gate

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.evaluate_zsre_hard_retain_forgetting \
  --data-root "$DATA_ROOT" --source-root "$SOURCE_ROOT" --scope-root "$SCOPE_ROOT" \
  --output "$ROOT/formal" --model "$MODEL" --tokenizer "$TOKENIZER" --batch-size 128 \
  > "$ROOT/formal.log" 2>&1

"$PYTHON" -m scripts.audit_zsre_hard_retain_forgetting_pilot \
  --root "$ROOT/formal" --data-root "$DATA_ROOT" --source-root "$SOURCE_ROOT" \
  --scope-root "$SCOPE_ROOT" --output "$ROOT/execution_audit.json"

echo completed_zsre_hard_retain_forgetting_pilot_v1 > "$ROOT/terminal_status.txt"
date -Is > "$ROOT/completed_at.txt"
