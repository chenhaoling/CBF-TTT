#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=${SOURCE_ROOT:-/home/ctj/cbf_ttt_post_v19_natural_semantic_pilot_20261015_retry1}
ROOT=${ROOT:-/home/ctj/cbf_ttt_post_v19_natural_semantic_pilot_20261015_retry2_audit_recovery}
DATA_ROOT=${DATA_ROOT:-/home/ctj/cbf_ttt_post_v19_natural_semantic_data_20261015_retry1/data}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$ROOT"
test "$(cat "$SOURCE_ROOT/runner_exit_code.txt")" = "1"
"$PYTHON_BIN" - "$SOURCE_ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
assert summary["terminal_status"] == "pilot_semantic_headroom_supported"
assert summary["pilot_gate"]["passed"] is True
assert "NameError: name 'QUERY_KINDS' is not defined" in (
    root / "audit.log").read_text()
PY

mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$ROOT/code_commit.txt"
printf '%s\n' "$SOURCE_ROOT" > "$ROOT/source_failed_audit_root.txt"
for name in shard0.jsonl shard1.jsonl shard0.jsonl.audit.json \
            shard1.jsonl.audit.json summary.json; do
  cp "$SOURCE_ROOT/$name" "$ROOT/$name"
done

cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" -m unittest tests.test_evaluate_post_v19_natural_semantic_headroom \
  > "$ROOT/tests.log" 2>&1
"$PYTHON_BIN" -m scripts.audit_post_v19_natural_semantic_outcome \
  --data-root "$DATA_ROOT" --split pilot \
  --inputs "$ROOT/shard0.jsonl" "$ROOT/shard1.jsonl" \
  --summary-path "$ROOT/summary.json" --output "$ROOT/audit.json" \
  > "$ROOT/audit.log" 2>&1
echo completed_post_v19_natural_semantic_pilot > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
