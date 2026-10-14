#!/usr/bin/env bash
set -euo pipefail

CHAIN_ROOT=${CHAIN_ROOT:-/home/ctj/cbf_ttt_mquake_v15_to_v16_chain}
V15_ROOT=${V15_ROOT:-/home/ctj/cbf_ttt_mquake_jackknife_consensus_v15}
V15_PID=${V15_PID:-1022027}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$CHAIN_ROOT"
mkdir -p "$CHAIN_ROOT"
trap 'code=$?; echo "$code" > "$CHAIN_ROOT/monitor_exit_code.txt"' EXIT
date --iso-8601=seconds > "$CHAIN_ROOT/started_at.txt"

missing_polls=0
while [[ ! -f "$V15_ROOT/terminal_status.txt" ]]; do
  if [[ -f "$V15_ROOT/runner_exit_code.txt" ]]; then
    code=$(cat "$V15_ROOT/runner_exit_code.txt")
    if [[ "$code" != "0" ]]; then
      echo "v15_runner_failed:$code" > "$CHAIN_ROOT/terminal_status.txt"
      exit 10
    fi
  fi
  if kill -0 "$V15_PID" 2>/dev/null; then
    missing_polls=0
  else
    missing_polls=$((missing_polls + 1))
    if [[ "$missing_polls" -ge 3 ]]; then
      echo "v15_process_missing_without_terminal" > "$CHAIN_ROOT/terminal_status.txt"
      exit 11
    fi
  fi
  sleep 30
done

set +e
"$PYTHON" -c 'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); s=json.loads((root/"result/summary.json").read_text()); a=json.loads((root/"audit.json").read_text()); ok=(s.get("protocol")=="mquake_jackknife_consensus_v15" and s.get("terminal_status")=="completed_mquake_jackknife_consensus_v15" and s.get("jackknife_consensus_supported") is True and a.get("audit_passed") is True); raise SystemExit(0 if ok else 20)' "$V15_ROOT"
gate_code=$?
set -e
if [[ "$gate_code" -ne 0 ]]; then
  echo "v16_not_launched_v15_unsupported" > "$CHAIN_ROOT/terminal_status.txt"
  date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
  exit 0
fi

echo "v15_supported_launching_v16_pilot" > "$CHAIN_ROOT/gate_status.txt"
cd "$REPO"
bash scripts/run_mquake_qwen_outcome_v16_pilot.sh \
  > "$CHAIN_ROOT/v16_pilot.log" 2>&1
echo "completed_v15_to_v16_pilot_chain" > "$CHAIN_ROOT/terminal_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
