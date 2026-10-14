#!/usr/bin/env bash
set -euo pipefail

OUTCOME=${OUTCOME:-/home/ctj/cbf_ttt_controlled_interference_v19_qwen_20261014}
RULE_ROOT=${RULE_ROOT:-/home/ctj/cbf_ttt_v19_causal_rule_diagnostic_20261014}
CHAIN_ROOT=${CHAIN_ROOT:-/home/ctj/cbf_ttt_v19_outcome_to_rules_chain_20261014}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$CHAIN_ROOT"
mkdir -p "$CHAIN_ROOT"
trap 'code=$?; echo "$code" > "$CHAIN_ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$CHAIN_ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$CHAIN_ROOT/code_commit.txt"

while [[ ! -f "$OUTCOME/runner_exit_code.txt" ]]; do
  date --iso-8601=seconds > "$CHAIN_ROOT/last_wait_at.txt"
  sleep 60
done

if [[ "$(cat "$OUTCOME/runner_exit_code.txt")" != "0" ]]; then
  echo stopped_v19_outcome_runner_failed > "$CHAIN_ROOT/terminal_status.txt"
  date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
  exit 0
fi

if ! "$PYTHON" - "$OUTCOME" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
summary_path = root / "summary.json"
summary = json.loads(summary_path.read_text())
audit = json.loads((root / "audit.json").read_text())
digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
checks = (
    summary.get("protocol") == "controlled_memory_interference_v19_qwen_confirmation",
    summary.get("rows") == 320,
    summary.get("primary_budget") == .03,
    summary.get("terminal_status") in {
        "v19_locality_safe_headroom_confirmed",
        "v19_locality_safe_headroom_not_confirmed"},
    audit.get("audit_passed") is True,
    audit.get("summary_sha256") == digest,
    audit.get("recomputed_confirmation_gate") == summary.get("confirmation_gate"),
    (root / "terminal_status.txt").read_text().strip()
        == "completed_controlled_interference_v19_qwen",
)
raise SystemExit(0 if all(checks) else 17)
PY
then
  echo stopped_v19_outcome_audit_unsupported > "$CHAIN_ROOT/terminal_status.txt"
  date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
  exit 0
fi

test ! -e "$RULE_ROOT"
echo audited_v19_outcome_launching_rules > "$CHAIN_ROOT/release_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/released_at.txt"
cd "$REPO"
ROOT="$RULE_ROOT" OUTCOME="$OUTCOME" \
  bash scripts/run_v19_causal_rule_diagnostic.sh \
  > "$CHAIN_ROOT/rule_runner.log" 2>&1
echo completed_rules_after_v19_outcome > "$CHAIN_ROOT/terminal_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
