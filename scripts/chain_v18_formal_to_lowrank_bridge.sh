#!/usr/bin/env bash
set -euo pipefail

FORMAL_ROOT=${FORMAL_ROOT:-/home/ctj/cbf_ttt_controlled_interference_v18_qwen_formal_20261014}
BRIDGE_ROOT=${BRIDGE_ROOT:-/home/ctj/cbf_ttt_v18_lowrank_fastmemory_bridge_20261014}
CHAIN_ROOT=${CHAIN_ROOT:-/home/ctj/cbf_ttt_v18_formal_to_lowrank_chain_20261014}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$CHAIN_ROOT"
mkdir -p "$CHAIN_ROOT"
trap 'code=$?; echo "$code" > "$CHAIN_ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$CHAIN_ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$CHAIN_ROOT/code_commit.txt"

while [[ ! -f "$FORMAL_ROOT/runner_exit_code.txt" ]]; do
  date --iso-8601=seconds > "$CHAIN_ROOT/last_wait_at.txt"
  sleep 60
done

if [[ "$(cat "$FORMAL_ROOT/runner_exit_code.txt")" != "0" ]]; then
  echo stopped_formal_runner_failed > "$CHAIN_ROOT/terminal_status.txt"
  exit 0
fi

if "$PYTHON" - "$FORMAL_ROOT" <<'PY'
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
    summary.get("protocol") == "controlled_memory_interference_v18_qwen_formal",
    summary.get("causal_headroom_supported") is True,
    summary.get("headroom_gate", {}).get("passed") is True,
    summary.get("terminal_status") == "formal_headroom_supported",
    audit.get("audit_passed") is True,
    audit.get("summary_sha256") == digest,
    audit.get("recomputed_headroom_gate", {}).get("passed") is True,
    (root / "terminal_status.txt").read_text().strip()
        == "completed_controlled_interference_v18_qwen_formal",
)
raise SystemExit(0 if all(checks) else 17)
PY
then
  released=1
else
  released=0
fi
if [[ "$released" -ne 1 ]]; then
  echo stopped_formal_headroom_or_audit_unsupported > "$CHAIN_ROOT/terminal_status.txt"
  exit 0
fi

test ! -e "$BRIDGE_ROOT"
echo formal_headroom_passed_launching_bridge > "$CHAIN_ROOT/release_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/released_at.txt"
cd "$REPO"
ROOT="$BRIDGE_ROOT" FORMAL_ROOT="$FORMAL_ROOT" \
  bash scripts/run_v18_lowrank_fastmemory_bridge.sh \
  > "$CHAIN_ROOT/bridge_runner.log" 2>&1
echo completed_bridge_after_formal_release > "$CHAIN_ROOT/terminal_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
