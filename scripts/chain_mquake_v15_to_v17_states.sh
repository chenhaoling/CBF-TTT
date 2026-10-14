#!/usr/bin/env bash
set -euo pipefail

CHAIN_ROOT=${CHAIN_ROOT:-/home/ctj/cbf_ttt_mquake_v15_to_v17_chain}
V15_ROOT=${V15_ROOT:-/home/ctj/cbf_ttt_mquake_jackknife_consensus_v15}
V15_PID=${V15_PID:-1022027}
V17_ROOT=${V17_ROOT:-/home/ctj/cbf_ttt_mquake_qwen_headroom_v17_states}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$CHAIN_ROOT"
mkdir -p "$CHAIN_ROOT"
trap 'code=$?; echo "$code" > "$CHAIN_ROOT/monitor_exit_code.txt"' EXIT
date --iso-8601=seconds > "$CHAIN_ROOT/started_at.txt"

missing_polls=0
while [[ ! -f "$V15_ROOT/terminal_status.txt" ]]; do
  if [[ -f "$V15_ROOT/runner_exit_code.txt" ]] \
      && [[ "$(cat "$V15_ROOT/runner_exit_code.txt")" != "0" ]]; then
    echo "v17_not_launched:v15_runner_failed" > "$CHAIN_ROOT/terminal_status.txt"
    exit 10
  fi
  if kill -0 "$V15_PID" 2>/dev/null; then
    missing_polls=0
  else
    missing_polls=$((missing_polls + 1))
    if [[ "$missing_polls" -ge 3 ]]; then
      echo "v17_not_launched:v15_missing_without_terminal" > "$CHAIN_ROOT/terminal_status.txt"
      exit 11
    fi
  fi
  sleep 30
done

set +e
verdict=$("$PYTHON" -c 'import json,sys; from pathlib import Path; r=Path(sys.argv[1]); s=json.loads((r/"result/summary.json").read_text()); a=json.loads((r/"audit.json").read_text()); assert s.get("terminal_status")=="completed_mquake_jackknife_consensus_v15" and a.get("audit_passed") is True; print("supported" if s.get("jackknife_consensus_supported") is True else "unsupported" if s.get("jackknife_consensus_supported") is False else "invalid")' "$V15_ROOT" 2>/dev/null)
verdict_code=$?
set -e
if [[ "$verdict_code" -ne 0 ]] || [[ "$verdict" == "invalid" ]]; then
  echo "v17_not_launched:v15_terminal_or_audit_invalid" > "$CHAIN_ROOT/terminal_status.txt"
  exit 12
fi
if [[ "$verdict" == "supported" ]]; then
  echo "v17_not_launched:v15_supported" > "$CHAIN_ROOT/terminal_status.txt"
  date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
  exit 0
fi

echo "v15_unsupported_launching_v17_states" > "$CHAIN_ROOT/stage.txt"
cd "$REPO"
ROOT="$V17_ROOT" bash scripts/run_mquake_qwen_headroom_v17_states.sh \
  > "$CHAIN_ROOT/v17_states.log" 2>&1
echo "completed_v15_to_v17_states_chain" > "$CHAIN_ROOT/terminal_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
