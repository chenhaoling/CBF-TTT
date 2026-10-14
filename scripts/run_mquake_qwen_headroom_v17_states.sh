#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_mquake_qwen_headroom_v17_states}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
DATA=${DATA:-/home/ctj/cbf_ttt_mquake_temporal_sessions_v1}
SAMPLE=${SAMPLE:-/home/ctj/cbf_ttt_mquake_qwen_outcome_v16}
V15=${V15:-/home/ctj/cbf_ttt_mquake_jackknife_consensus_v15/result}

cd "$REPO"
test ! -e "$ROOT"
mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -m unittest \
  tests.test_build_mquake_qwen_headroom_v17_states \
  tests.test_audit_mquake_qwen_headroom_v17_states
"$PYTHON" -m tasks.build_mquake_qwen_headroom_v17_states \
  --data-root "$DATA" --sample-root "$SAMPLE" --v15-root "$V15" \
  --output "$ROOT/result" > "$ROOT/build.log" 2>&1
"$PYTHON" -m scripts.audit_mquake_qwen_headroom_v17_states \
  --data-root "$DATA" --sample-root "$SAMPLE" --v15-root "$V15" \
  --state-root "$ROOT/result" --output "$ROOT/audit.json" \
  > "$ROOT/audit.log" 2>&1
echo completed_mquake_qwen_headroom_v17_states > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
cat "$ROOT/result/manifest.json"
