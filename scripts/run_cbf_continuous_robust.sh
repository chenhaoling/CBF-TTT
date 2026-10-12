#!/usr/bin/env bash
set -euo pipefail

FROZEN=${FROZEN:-/home/ctj/cbf_ttt_continuous_forgetting_v1/formal}
ROOT=${ROOT:-/home/ctj/cbf_ttt_continuous_robust_v2}
PYTHON=${PYTHON:-python}
TEST_PYTHON=${TEST_PYTHON:-$PYTHON}
if [[ -e "$ROOT" ]]; then echo "Refusing to overwrite existing root: $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"
printf '{"state":"running","stage":"tests"}\n' > "$ROOT/RUN_STATUS.json"
fail() { code=$?; printf '{"state":"failed","exit_code":%s}\n' "$code" > "$ROOT/RUN_STATUS.json"; exit "$code"; }
trap fail ERR
cd "$(dirname "$0")/.."
"$TEST_PYTHON" -m pytest -q tests/test_cbf_continuous_forgetting.py tests/test_cbf_continuous_stress.py tests/test_cbf_continuous_robust.py
printf '{"state":"running","stage":"robust_formal"}\n' > "$ROOT/RUN_STATUS.json"
"$PYTHON" -m tasks.train_cbf_continuous_robust --frozen "$FROZEN" --output "$ROOT/formal" \
  --train-groups 64 --dev-groups 16 --test-groups 32 --per-group 20 \
  --train-seed 20261105 --dev-seed 20261106 --test-seed 20261107 \
  --controller-seed 3941 --bootstrap-seed 3951 --dim 32 --epochs 300
printf '{"state":"running","stage":"audit"}\n' > "$ROOT/RUN_STATUS.json"
"$PYTHON" -m scripts.audit_cbf_continuous_robust --root "$ROOT/formal" --frozen "$FROZEN"
git rev-parse HEAD > "$ROOT/formal/git_commit.txt"
"$PYTHON" - "$ROOT" <<'PY'
import datetime,json,sys
from pathlib import Path
root=Path(sys.argv[1]); s=json.loads((root/'formal'/'summary.json').read_text())
(root/'RUN_STATUS.json').write_text(json.dumps({'state':'completed','stage_passed':s['stage_passed'],
 'test_scored':s['test_scored'],'updated_at':datetime.datetime.now().astimezone().isoformat()})+'\n')
PY
trap - ERR
