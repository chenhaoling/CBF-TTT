#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_structured_forgetting_v1}
PYTHON=${PYTHON:-python}
TEST_PYTHON=${TEST_PYTHON:-$PYTHON}

if [[ -e "$ROOT" ]]; then echo "Refusing to overwrite existing root: $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"
printf '{"state":"running","stage":"tests"}\n' > "$ROOT/RUN_STATUS.json"
fail() { code=$?; printf '{"state":"failed","exit_code":%s}\n' "$code" > "$ROOT/RUN_STATUS.json"; exit "$code"; }
trap fail ERR

cd "$(dirname "$0")/.."
"$TEST_PYTHON" -m pytest -q tests/test_cbf_structured_forgetting.py tests/test_cbf_event_curriculum.py

printf '{"state":"running","stage":"corpus"}\n' > "$ROOT/RUN_STATUS.json"
"$PYTHON" -m tasks.build_cbf_event_curriculum --output "$ROOT/data" --seed 20261011 \
  --train-groups 64 --dev-groups 16 --test-groups 16

printf '{"state":"running","stage":"training"}\n' > "$ROOT/RUN_STATUS.json"
"$PYTHON" -m tasks.train_cbf_structured_forgetting --data-root "$ROOT/data" \
  --output "$ROOT/formal" --seed 1517 --epochs 500
"$PYTHON" -m scripts.audit_cbf_structured_forgetting --root "$ROOT/formal" --data-root "$ROOT/data"
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
