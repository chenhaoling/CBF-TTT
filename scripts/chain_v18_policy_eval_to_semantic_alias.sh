#!/usr/bin/env bash
set -euo pipefail

CONTROLLED_ROOT=${CONTROLLED_ROOT:-/home/ctj/cbf_ttt_v18_causal_deletion_policy_eval_20261014}
ALIAS_ROOT=${ALIAS_ROOT:-/home/ctj/cbf_ttt_v18_semantic_alias_data_20261014}
CHAIN_ROOT=${CHAIN_ROOT:-/home/ctj/cbf_ttt_v18_policy_eval_to_alias_chain_20261014}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$CHAIN_ROOT"; mkdir -p "$CHAIN_ROOT"
trap 'code=$?; echo "$code" > "$CHAIN_ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$CHAIN_ROOT/started_at.txt"; git -C "$REPO" rev-parse HEAD > "$CHAIN_ROOT/code_commit.txt"
while [[ ! -f "$CONTROLLED_ROOT/runner_exit_code.txt" ]]; do
  date --iso-8601=seconds > "$CHAIN_ROOT/last_wait_at.txt"; sleep 60
done
if [[ "$(cat "$CONTROLLED_ROOT/runner_exit_code.txt")" != "0" ]]; then
  echo stopped_controlled_eval_runner_failed > "$CHAIN_ROOT/terminal_status.txt"; exit 0
fi
if "$PYTHON" - "$CONTROLLED_ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]); summary_path = root / "summary.json"
summary = json.loads(summary_path.read_text()); audit = json.loads((root / "audit.json").read_text())
checks = (summary.get("controlled_policy_supported") is True,
          summary.get("controlled_policy_gate", {}).get("passed") is True,
          audit.get("audit_passed") is True,
          audit.get("summary_sha256") == hashlib.sha256(summary_path.read_bytes()).hexdigest(),
          (root / "terminal_status.txt").read_text().strip()
              == "completed_v18_causal_deletion_policy_eval")
raise SystemExit(0 if all(checks) else 37)
PY
then released=1; else released=0; fi
if [[ "$released" -ne 1 ]]; then
  echo stopped_controlled_policy_unsupported > "$CHAIN_ROOT/terminal_status.txt"; exit 0
fi
test ! -e "$ALIAS_ROOT"; echo controlled_policy_passed_launching_alias_build > "$CHAIN_ROOT/release_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/released_at.txt"; cd "$REPO"
ROOT="$ALIAS_ROOT" CONTROLLED_ROOT="$CONTROLLED_ROOT" \
  bash scripts/run_v18_semantic_alias_build.sh > "$CHAIN_ROOT/alias_runner.log" 2>&1
echo completed_alias_build_after_controlled_release > "$CHAIN_ROOT/terminal_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
