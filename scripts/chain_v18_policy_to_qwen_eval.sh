#!/usr/bin/env bash
set -euo pipefail

FORMAL_ROOT=${FORMAL_ROOT:-/home/ctj/cbf_ttt_controlled_interference_v18_qwen_formal_20261014}
BRIDGE_ROOT=${BRIDGE_ROOT:-/home/ctj/cbf_ttt_v18_lowrank_fastmemory_bridge_20261014}
PHYSICAL_ROOT=${PHYSICAL_ROOT:-/home/ctj/cbf_ttt_v18_physical_conflict_rules_20261014}
POLICY_ROOT=${POLICY_ROOT:-/home/ctj/cbf_ttt_v18_causal_deletion_policy_20261014}
EVAL_ROOT=${EVAL_ROOT:-/home/ctj/cbf_ttt_v18_causal_deletion_policy_eval_20261014}
CHAIN_ROOT=${CHAIN_ROOT:-/home/ctj/cbf_ttt_v18_policy_to_eval_chain_20261014}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$CHAIN_ROOT"; mkdir -p "$CHAIN_ROOT"
trap 'code=$?; echo "$code" > "$CHAIN_ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$CHAIN_ROOT/started_at.txt"; git -C "$REPO" rev-parse HEAD > "$CHAIN_ROOT/code_commit.txt"
while [[ ! -f "$POLICY_ROOT/runner_exit_code.txt" ]]; do
  date --iso-8601=seconds > "$CHAIN_ROOT/last_wait_at.txt"; sleep 60
done
if [[ "$(cat "$POLICY_ROOT/runner_exit_code.txt")" != "0" ]]; then
  echo stopped_policy_runner_failed > "$CHAIN_ROOT/terminal_status.txt"; exit 0
fi
if "$PYTHON" - "$POLICY_ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]); summary_path = root / "result/summary.json"
summary = json.loads(summary_path.read_text()); audit = json.loads((root / "audit.json").read_text())
checks = (summary.get("protocol") == "v18_causal_deletion_policy_v1",
          summary.get("terminal_status") == "trained_oof_deletion_policy",
          audit.get("audit_passed") is True,
          audit.get("summary_sha256") == hashlib.sha256(summary_path.read_bytes()).hexdigest(),
          (root / "terminal_status.txt").read_text().strip()
              == "completed_v18_causal_deletion_policy")
raise SystemExit(0 if all(checks) else 31)
PY
then released=1; else released=0; fi
if [[ "$released" -ne 1 ]]; then
  echo stopped_policy_audit_unsupported > "$CHAIN_ROOT/terminal_status.txt"; exit 0
fi
test ! -e "$EVAL_ROOT"; echo policy_audit_passed_launching_qwen_eval > "$CHAIN_ROOT/release_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/released_at.txt"; cd "$REPO"
ROOT="$EVAL_ROOT" FORMAL_ROOT="$FORMAL_ROOT" BRIDGE_ROOT="$BRIDGE_ROOT" \
  PHYSICAL_ROOT="$PHYSICAL_ROOT" POLICY_ROOT="$POLICY_ROOT" \
  bash scripts/run_v18_causal_deletion_policy_eval.sh > "$CHAIN_ROOT/eval_runner.log" 2>&1
echo completed_qwen_eval_after_policy_release > "$CHAIN_ROOT/terminal_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
