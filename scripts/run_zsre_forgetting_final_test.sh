#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_zsre_forgetting_final_test_v1}
BOUNDARY_ROOT=${BOUNDARY_ROOT:-/home/ctj/cbf_ttt_zsre_forgetting_dev_boundary_v1/diagnostic}
BOUNDARY_AUDIT=${BOUNDARY_AUDIT:-/home/ctj/cbf_ttt_zsre_forgetting_dev_boundary_v1/execution_audit.json}
DATA_ROOT=${DATA_ROOT:-/home/ctj/cbf_ttt_zsre_counterfactual_forgetting_v1_data}
CONTROLLER_ROOT=${CONTROLLER_ROOT:-/home/ctj/cbf_ttt_zsre_counterfactual_forgetting_controller_loso_v3/formal}
SOURCE_ROOT=${SOURCE_ROOT:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal}
SCOPE_ROOT=${SCOPE_ROOT:-/home/ctj/cbf_ttt_zsre_decomposed_scope_gate_v1/formal}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

if [[ -e "$ROOT" ]]; then echo "ROOT already exists: $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"; git rev-parse HEAD > "$ROOT/code_commit.txt"; date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT

"$PYTHON" -m unittest tests.test_zsre_forgetting_final_test \
  tests.test_zsre_forgetting_dev_threshold tests.test_zsre_counterfactual_forgetting_formal \
  > "$ROOT/tests.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.evaluate_zsre_forgetting_final_test \
  --data-root "$DATA_ROOT" --boundary-root "$BOUNDARY_ROOT" --boundary-audit "$BOUNDARY_AUDIT" \
  --controller-root "$CONTROLLER_ROOT" --source-root "$SOURCE_ROOT" --scope-root "$SCOPE_ROOT" \
  --output "$ROOT/test" --model "$MODEL" --tokenizer "$TOKENIZER" --batch-size 128 \
  > "$ROOT/test.log" 2>&1

"$PYTHON" -m scripts.audit_zsre_forgetting_final_test \
  --root "$ROOT/test" --boundary-root "$BOUNDARY_ROOT" --boundary-audit "$BOUNDARY_AUDIT" \
  --controller-root "$CONTROLLER_ROOT" --source-root "$SOURCE_ROOT" --scope-root "$SCOPE_ROOT" \
  --output "$ROOT/execution_audit.json"

echo completed_zsre_forgetting_final_test_v1 > "$ROOT/terminal_status.txt"
date -Is > "$ROOT/completed_at.txt"
