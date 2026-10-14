#!/usr/bin/env bash
set -euo pipefail

FORMAL_ROOT=${FORMAL_ROOT:-/home/ctj/cbf_ttt_controlled_interference_v18_qwen_formal_20261014}
BRIDGE_ROOT=${BRIDGE_ROOT:-/home/ctj/cbf_ttt_v18_lowrank_fastmemory_bridge_20261014}
PHYSICAL_ROOT=${PHYSICAL_ROOT:-/home/ctj/cbf_ttt_v18_physical_conflict_rules_20261014}
EXAMPLES_ROOT=${EXAMPLES_ROOT:-/home/ctj/cbf_ttt_v18_causal_deletion_examples_20261014}
CHAIN_ROOT=${CHAIN_ROOT:-/home/ctj/cbf_ttt_v18_physical_to_examples_chain_20261014}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$CHAIN_ROOT"
mkdir -p "$CHAIN_ROOT"
trap 'code=$?; echo "$code" > "$CHAIN_ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$CHAIN_ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$CHAIN_ROOT/code_commit.txt"
while [[ ! -f "$PHYSICAL_ROOT/runner_exit_code.txt" ]]; do
  date --iso-8601=seconds > "$CHAIN_ROOT/last_wait_at.txt"
  sleep 60
done
if [[ "$(cat "$PHYSICAL_ROOT/runner_exit_code.txt")" != "0" ]]; then
  echo stopped_physical_runner_failed > "$CHAIN_ROOT/terminal_status.txt"
  exit 0
fi
if "$PYTHON" - "$PHYSICAL_ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]); summary_path = root / "summary.json"
summary = json.loads(summary_path.read_text()); audit = json.loads((root / "audit.json").read_text())
checks = (summary.get("protocol") == "v18_physical_conflict_rules_v1",
          summary.get("terminal_status") == "physical_conflict_rules_evaluated",
          audit.get("audit_passed") is True,
          audit.get("summary_sha256") == hashlib.sha256(summary_path.read_bytes()).hexdigest(),
          (root / "terminal_status.txt").read_text().strip()
              == "completed_v18_physical_conflict_rules")
raise SystemExit(0 if all(checks) else 23)
PY
then released=1; else released=0; fi
if [[ "$released" -ne 1 ]]; then
  echo stopped_physical_audit_unsupported > "$CHAIN_ROOT/terminal_status.txt"
  exit 0
fi
test ! -e "$EXAMPLES_ROOT"
echo physical_audit_passed_launching_examples > "$CHAIN_ROOT/release_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/released_at.txt"
cd "$REPO"
ROOT="$EXAMPLES_ROOT" FORMAL_ROOT="$FORMAL_ROOT" BRIDGE_ROOT="$BRIDGE_ROOT" \
  PHYSICAL_ROOT="$PHYSICAL_ROOT" bash scripts/run_v18_causal_deletion_examples.sh \
  > "$CHAIN_ROOT/examples_runner.log" 2>&1
echo completed_examples_after_physical_release > "$CHAIN_ROOT/terminal_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
