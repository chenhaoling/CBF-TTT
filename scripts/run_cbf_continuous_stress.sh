#!/usr/bin/env bash
set -euo pipefail

FORMAL=${FORMAL:-/home/ctj/cbf_ttt_continuous_forgetting_v1/formal}
ROOT=${ROOT:-/home/ctj/cbf_ttt_continuous_stress_v1}
PYTHON=${PYTHON:-python}
TEST_PYTHON=${TEST_PYTHON:-$PYTHON}

if [[ -e "$ROOT" ]]; then echo "Refusing to overwrite existing root: $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"
printf '{"state":"running","stage":"tests"}\n' > "$ROOT/RUN_STATUS.json"
fail() { code=$?; printf '{"state":"failed","exit_code":%s}\n' "$code" > "$ROOT/RUN_STATUS.json"; exit "$code"; }
trap fail ERR
cd "$(dirname "$0")/.."
"$TEST_PYTHON" -m pytest -q tests/test_cbf_continuous_forgetting.py tests/test_cbf_continuous_stress.py
printf '{"state":"running","stage":"stress"}\n' > "$ROOT/RUN_STATUS.json"
"$PYTHON" -m tasks.evaluate_cbf_continuous_stress --formal "$FORMAL" --output "$ROOT/formal" \
  --groups 32 --per-group 20 --seed 20261103 --bootstrap-seed 2951
printf '{"state":"running","stage":"audit"}\n' > "$ROOT/RUN_STATUS.json"
"$PYTHON" -m scripts.audit_cbf_continuous_stress --root "$ROOT/formal" --formal "$FORMAL"
git rev-parse HEAD > "$ROOT/formal/git_commit.txt"
"$PYTHON" - "$ROOT" <<'PY'
import datetime,json,sys
from pathlib import Path
root=Path(sys.argv[1]); summary=json.loads((root/'formal'/'summary.json').read_text())
(root/'RUN_STATUS.json').write_text(json.dumps({
 'state':'completed','passed':summary['passed'],
 'updated_at':datetime.datetime.now().astimezone().isoformat()})+'\n')
PY
trap - ERR
