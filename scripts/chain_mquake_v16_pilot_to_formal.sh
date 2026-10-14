#!/usr/bin/env bash
set -euo pipefail

CHAIN_ROOT=${CHAIN_ROOT:-/home/ctj/cbf_ttt_mquake_v16_pilot_to_formal_chain}
UPSTREAM_CHAIN=${UPSTREAM_CHAIN:-/home/ctj/cbf_ttt_mquake_v15_to_v16_chain}
PILOT_ROOT=${PILOT_ROOT:-/home/ctj/cbf_ttt_mquake_qwen_outcome_v16_pilot}
STATE_ROOT=${STATE_ROOT:-/home/ctj/cbf_ttt_mquake_qwen_outcome_v16_formal_states}
FORMAL_ROOT=${FORMAL_ROOT:-/home/ctj/cbf_ttt_mquake_qwen_outcome_v16_formal}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$CHAIN_ROOT"
mkdir -p "$CHAIN_ROOT"
trap 'code=$?; echo "$code" > "$CHAIN_ROOT/monitor_exit_code.txt"' EXIT
date --iso-8601=seconds > "$CHAIN_ROOT/started_at.txt"

while [[ ! -f "$PILOT_ROOT/runner_exit_code.txt" ]]; do
  if [[ -f "$UPSTREAM_CHAIN/terminal_status.txt" ]]; then
    upstream=$(cat "$UPSTREAM_CHAIN/terminal_status.txt")
    if [[ "$upstream" != "completed_v15_to_v16_pilot_chain" ]]; then
      echo "formal_not_launched:$upstream" > "$CHAIN_ROOT/terminal_status.txt"
      date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
      exit 0
    fi
  fi
  if [[ -f "$UPSTREAM_CHAIN/monitor_exit_code.txt" ]] \
      && [[ "$(cat "$UPSTREAM_CHAIN/monitor_exit_code.txt")" != "0" ]]; then
    echo "formal_not_launched:upstream_monitor_failed" > "$CHAIN_ROOT/terminal_status.txt"
    exit 10
  fi
  sleep 30
done

if [[ "$(cat "$PILOT_ROOT/runner_exit_code.txt")" != "0" ]]; then
  echo "formal_not_launched:pilot_runner_failed" > "$CHAIN_ROOT/terminal_status.txt"
  exit 11
fi
set +e
"$PYTHON" -c 'import json,sys; from pathlib import Path; r=Path(sys.argv[1]); s=json.loads((r/"summary.json").read_text()); a=json.loads((r/"audit.json").read_text()); ok=(s.get("terminal_status")=="pilot_passed" and s.get("pilot_gate",{}).get("passed") is True and a.get("audit_passed") is True); raise SystemExit(0 if ok else 1)' "$PILOT_ROOT"
pilot_gate=$?
set -e
if [[ "$pilot_gate" -ne 0 ]]; then
  echo "formal_not_launched:pilot_gate_failed" > "$CHAIN_ROOT/terminal_status.txt"
  date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
  exit 0
fi

cd "$REPO"
echo "pilot_passed_launching_formal_states" > "$CHAIN_ROOT/stage.txt"
ROOT="$STATE_ROOT" bash scripts/run_mquake_qwen_outcome_v16_formal_states.sh \
  > "$CHAIN_ROOT/formal_states.log" 2>&1
if [[ "$(cat "$STATE_ROOT/runner_exit_code.txt")" != "0" ]]; then
  echo "formal_failed:state_runner" > "$CHAIN_ROOT/terminal_status.txt"
  exit 12
fi
set +e
"$PYTHON" -c 'import json,sys; from pathlib import Path; r=Path(sys.argv[1]); m=json.loads((r/"result/manifest.json").read_text()); a=json.loads((r/"audit.json").read_text()); ok=(m.get("terminal_status")=="frozen_after_passed_v16_pilot_before_formal_scoring" and a.get("audit_passed") is True); raise SystemExit(0 if ok else 1)' "$STATE_ROOT"
state_gate=$?
set -e
if [[ "$state_gate" -ne 0 ]]; then
  echo "formal_failed:state_audit" > "$CHAIN_ROOT/terminal_status.txt"
  exit 13
fi

echo "formal_states_passed_launching_qwen_outcomes" > "$CHAIN_ROOT/stage.txt"
ROOT="$FORMAL_ROOT" STATES="$STATE_ROOT/result" \
  bash scripts/run_mquake_qwen_outcome_v16_formal.sh \
  > "$CHAIN_ROOT/formal_outcome.log" 2>&1
if [[ "$(cat "$FORMAL_ROOT/runner_exit_code.txt")" != "0" ]]; then
  echo "formal_failed:outcome_runner" > "$CHAIN_ROOT/terminal_status.txt"
  exit 14
fi
echo "completed_v16_pilot_to_formal_chain" > "$CHAIN_ROOT/terminal_status.txt"
date --iso-8601=seconds > "$CHAIN_ROOT/completed_at.txt"
