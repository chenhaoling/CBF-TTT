#!/usr/bin/env bash
set -euo pipefail

RULE_ROOT=${RULE_ROOT:-/home/ctj/cbf_ttt_v19_causal_rule_diagnostic_20261014}
UPSTREAM_CHAIN=${UPSTREAM_CHAIN:-/home/ctj/cbf_ttt_v19_outcome_to_rules_chain_20261014}
DATA_ROOT=${DATA_ROOT:-/home/ctj/cbf_ttt_controlled_interference_v20_alias_data_20261014}
OUTCOME_ROOT=${OUTCOME_ROOT:-/home/ctj/cbf_ttt_controlled_interference_v20_alias_qwen_20261014}
CHAIN_ROOT=${CHAIN_ROOT:-/home/ctj/cbf_ttt_v19_rules_to_v20_alias_chain_20261014}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$CHAIN_ROOT"
mkdir -p "$CHAIN_ROOT"
trap 'code=$?; echo "$code" > "$CHAIN_ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$CHAIN_ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$CHAIN_ROOT/code_commit.txt"

while [[ ! -f "$RULE_ROOT/runner_exit_code.txt" ]]; do
  if [[ -f "$UPSTREAM_CHAIN/runner_exit_code.txt" && ! -d "$RULE_ROOT" ]]; then
    echo stopped_upstream_did_not_create_rule_diagnostic > "$CHAIN_ROOT/terminal_status.txt"
    date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
    exit 0
  fi
  date --iso-8601=seconds > "$CHAIN_ROOT/last_wait_at.txt"
  sleep 60
done

if [[ "$(cat "$RULE_ROOT/runner_exit_code.txt")" != "0" ]]; then
  echo stopped_v19_rule_runner_failed > "$CHAIN_ROOT/terminal_status.txt"
  date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
  exit 0
fi

if ! "$PYTHON" - "$RULE_ROOT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
summary_path = root / "summary.json"
summary = json.loads(summary_path.read_text())
audit = json.loads((root / "audit.json").read_text())
checks = (
    summary.get("protocol") == "v19_causal_rule_explanation_v1",
    summary.get("explicit_key_explanation_gate", {}).get("passed") is True,
    summary.get("next_stage") == "build_v20_semantic_alias_confirmation",
    summary.get("query_or_answer_features_used") is False,
    summary.get("model_rescored") is False,
    audit.get("audit_passed") is True,
    audit.get("summary_sha256") == hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    audit.get("recomputed_explanation_gate") == summary.get("explicit_key_explanation_gate"),
    (root / "terminal_status.txt").read_text().strip()
        == "completed_v19_causal_rule_diagnostic",
)
raise SystemExit(0 if all(checks) else 19)
PY
then
  echo stopped_v19_explicit_key_explanation_gate_not_passed > "$CHAIN_ROOT/terminal_status.txt"
  date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
  exit 0
fi

cd "$REPO"
echo audited_v19_rule_gate_launching_v20_data > "$CHAIN_ROOT/release_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/data_released_at.txt"
ROOT="$DATA_ROOT" RULE_ROOT="$RULE_ROOT" \
  bash scripts/run_controlled_interference_v20_alias_build.sh \
  > "$CHAIN_ROOT/data_runner.log" 2>&1
echo audited_v20_data_launching_model_scoring > "$CHAIN_ROOT/release_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/model_released_at.txt"
ROOT="$OUTCOME_ROOT" DATA="$DATA_ROOT/result" RULE_ROOT="$RULE_ROOT" \
  bash scripts/run_controlled_interference_v20_alias_qwen.sh \
  > "$CHAIN_ROOT/model_runner.log" 2>&1
echo completed_v20_semantic_alias_after_v19_rule_gate > "$CHAIN_ROOT/terminal_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
