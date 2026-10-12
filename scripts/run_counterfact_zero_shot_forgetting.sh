#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_counterfact_zero_shot_v1}
SOURCE=${SOURCE:-/home/ctj/data/cbf_ttt_eval/counterfact/counterfact.json}
BOUNDARY_ROOT=${BOUNDARY_ROOT:-/home/ctj/cbf_ttt_zsre_forgetting_dev_boundary_v1/diagnostic}
BOUNDARY_AUDIT=${BOUNDARY_AUDIT:-/home/ctj/cbf_ttt_zsre_forgetting_dev_boundary_v1/execution_audit.json}
CONTROLLER_ROOT=${CONTROLLER_ROOT:-/home/ctj/cbf_ttt_zsre_counterfactual_forgetting_controller_loso_v3/formal}
CONTEXT_SOURCE=${CONTEXT_SOURCE:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal}
SCOPE_ROOT=${SCOPE_ROOT:-/home/ctj/cbf_ttt_zsre_decomposed_scope_gate_v1/formal}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

if [[ -e "$ROOT" ]]; then echo "ROOT already exists: $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"; git rev-parse HEAD > "$ROOT/code_commit.txt"; date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT

"$PYTHON" -m unittest tests.test_counterfact_zero_shot_forgetting \
  tests.test_zsre_forgetting_final_test tests.test_zsre_counterfactual_forgetting \
  > "$ROOT/tests.log" 2>&1

"$PYTHON" -m tasks.build_counterfact_zero_shot_forgetting \
  --source "$SOURCE" --output "$ROOT/data" > "$ROOT/build.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.evaluate_counterfact_zero_shot_forgetting \
  --data-root "$ROOT/data" --boundary-root "$BOUNDARY_ROOT" --boundary-audit "$BOUNDARY_AUDIT" \
  --controller-root "$CONTROLLER_ROOT" --source-root "$CONTEXT_SOURCE" --scope-root "$SCOPE_ROOT" \
  --output "$ROOT/evaluation" --model "$MODEL" --tokenizer "$TOKENIZER" --batch-size 128 \
  > "$ROOT/evaluation.log" 2>&1

"$PYTHON" -m scripts.audit_counterfact_zero_shot_forgetting \
  --root "$ROOT/evaluation" --data-root "$ROOT/data" --boundary-root "$BOUNDARY_ROOT" \
  --boundary-audit "$BOUNDARY_AUDIT" --controller-root "$CONTROLLER_ROOT" \
  --source-root "$CONTEXT_SOURCE" --scope-root "$SCOPE_ROOT" --output "$ROOT/execution_audit.json"

echo completed_counterfact_zero_shot_forgetting_v1 > "$ROOT/terminal_status.txt"
date -Is > "$ROOT/completed_at.txt"
