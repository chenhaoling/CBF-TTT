#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_continuous_forgetting_v1}
PYTHON=${PYTHON:-python}
TEST_PYTHON=${TEST_PYTHON:-$PYTHON}

if [[ -e "$ROOT" ]]; then echo "Refusing to overwrite existing root: $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"
printf '{"state":"running","stage":"tests"}\n' > "$ROOT/RUN_STATUS.json"
fail() { code=$?; printf '{"state":"failed","exit_code":%s}\n' "$code" > "$ROOT/RUN_STATUS.json"; exit "$code"; }
trap fail ERR

cd "$(dirname "$0")/.."
"$TEST_PYTHON" -m pytest -q tests/test_cbf_continuous_forgetting.py tests/test_cbf_lowrank_forgetting.py

printf '{"state":"running","stage":"probe"}\n' > "$ROOT/RUN_STATUS.json"
"$PYTHON" -m tasks.train_cbf_continuous_forgetting probe --output "$ROOT/probe.json" \
  --groups 40 --per-group 20 --seed 20261101 --dim 32
"$PYTHON" - "$ROOT/probe.json" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))
if not row['passed']:
    raise SystemExit('registered label-distribution probe failed')
PY

printf '{"state":"running","stage":"formal"}\n' > "$ROOT/RUN_STATUS.json"
"$PYTHON" -m tasks.train_cbf_continuous_forgetting formal --probe "$ROOT/probe.json" \
  --output "$ROOT/formal" --seed 20261101 --controller-seed 1941 \
  --train-groups 128 --dev-groups 32 --test-groups 32 --per-group 20 --dim 32 --epochs 300
printf '{"state":"running","stage":"audit"}\n' > "$ROOT/RUN_STATUS.json"
"$PYTHON" -m scripts.audit_cbf_continuous_forgetting --root "$ROOT/formal" --probe "$ROOT/probe.json"
git rev-parse HEAD > "$ROOT/formal/git_commit.txt"
"$PYTHON" - "$ROOT" <<'PY'
import datetime,json,sys
from pathlib import Path
root=Path(sys.argv[1]); summary=json.loads((root/'formal'/'summary.json').read_text())
(root/'RUN_STATUS.json').write_text(json.dumps({
  'state':'completed','stage_passed':summary['stage_passed'],
  'test_scored':summary['test_scored'],
  'updated_at':datetime.datetime.now().astimezone().isoformat()})+'\n')
PY
trap - ERR
