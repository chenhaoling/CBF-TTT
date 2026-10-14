#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_v18_semantic_alias_data_v1}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
CONTROLLED_ROOT=${CONTROLLED_ROOT:-/home/ctj/cbf_ttt_v18_causal_deletion_policy_eval_20261014}
FORMAL_DATA=${FORMAL_DATA:-/home/ctj/cbf_ttt_controlled_interference_v18_formal_data_20261014_retry1/result}
PILOT_DATA=${PILOT_DATA:-/home/ctj/cbf_ttt_controlled_interference_v18_pilot_data_tokmatch_retry1/result}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}

cd "$REPO"; test ! -e "$ROOT"; mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"; git rev-parse HEAD > "$ROOT/code_commit.txt"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -m unittest tests.test_build_v18_semantic_alias_confirmation
"$PYTHON" -m tasks.build_v18_semantic_alias_confirmation --output "$ROOT/data" \
  --controlled-root "$CONTROLLED_ROOT" > "$ROOT/build.log" 2>&1
"$PYTHON" -m scripts.audit_v18_semantic_alias_confirmation --root "$ROOT/data" \
  --controlled-root "$CONTROLLED_ROOT" --formal-data-root "$FORMAL_DATA" \
  --pilot-data-root "$PILOT_DATA" --tokenizer "$TOKENIZER" --output "$ROOT/audit.json" \
  > "$ROOT/audit.log" 2>&1
echo completed_v18_semantic_alias_build > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"; cat "$ROOT/data/manifest.json"
