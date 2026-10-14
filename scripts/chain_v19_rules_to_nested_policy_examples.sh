#!/usr/bin/env bash
set -euo pipefail

RULE_ROOT=${RULE_ROOT:-/home/ctj/cbf_ttt_v19_causal_rule_diagnostic_20261014}
UPSTREAM_CHAIN=${UPSTREAM_CHAIN:-/home/ctj/cbf_ttt_v19_outcome_to_rules_chain_20261014}
EXAMPLE_ROOT=${EXAMPLE_ROOT:-/home/ctj/cbf_ttt_v19_nested_policy_examples_20261014}
POLICY_ROOT=${POLICY_ROOT:-/home/ctj/cbf_ttt_v19_nested_forgetting_policy_20261014}
EVAL_ROOT=${EVAL_ROOT:-/home/ctj/cbf_ttt_v19_nested_forgetting_policy_qwen_20261014}
CHAIN_ROOT=${CHAIN_ROOT:-/home/ctj/cbf_ttt_v19_rules_to_nested_policy_chain_20261014}
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
    summary.get("explicit_key_explanation_gate", {}).get("passed") is False,
    summary.get("next_stage") == "train_v19_nested_oof_causal_policy",
    summary.get("query_or_answer_features_used") is False,
    summary.get("model_rescored") is False,
    audit.get("audit_passed") is True,
    audit.get("summary_sha256") == hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    audit.get("recomputed_explanation_gate") == summary.get("explicit_key_explanation_gate"),
    (root / "terminal_status.txt").read_text().strip()
        == "completed_v19_causal_rule_diagnostic",
)
raise SystemExit(0 if all(checks) else 23)
PY
then
  echo stopped_v19_nested_policy_fallback_not_released > "$CHAIN_ROOT/terminal_status.txt"
  date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
  exit 0
fi

cd "$REPO"
echo audited_v19_fallback_launching_policy_examples > "$CHAIN_ROOT/release_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/released_at.txt"
ROOT="$EXAMPLE_ROOT" RULE_ROOT="$RULE_ROOT" \
  bash scripts/run_v19_nested_policy_examples.sh \
  > "$CHAIN_ROOT/example_runner.log" 2>&1
echo audited_policy_examples_launching_nested_oof_training > "$CHAIN_ROOT/release_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/policy_released_at.txt"
ROOT="$POLICY_ROOT" EXAMPLES="$EXAMPLE_ROOT" \
  bash scripts/run_v19_nested_forgetting_policy.sh \
  > "$CHAIN_ROOT/policy_runner.log" 2>&1
echo audited_nested_policy_launching_qwen_rescore > "$CHAIN_ROOT/release_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/eval_released_at.txt"
ROOT="$EVAL_ROOT" EXAMPLES="$EXAMPLE_ROOT" POLICY="$POLICY_ROOT" \
  RULE_ROOT="$RULE_ROOT" \
  bash scripts/run_v19_nested_forgetting_policy_qwen.sh \
  > "$CHAIN_ROOT/eval_runner.log" 2>&1
echo completed_v19_nested_policy_qwen_after_rule_gate > "$CHAIN_ROOT/terminal_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
