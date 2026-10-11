#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_paired_binding_v1}
DATA=${DATA:-/home/ctj/cbf_ttt_dynamic_binding_v1/data}
MODEL=${MODEL:-/home/ctj/models/Qwen3-4B}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
TEST_PYTHON=${TEST_PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_verify_20260923/bin/python}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite existing root: $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
printf '{"state":"running","stage":"tests"}\n' > "$ROOT/RUN_STATUS.json"
fail() {
  code=$?
  printf '{"state":"failed","exit_code":%s}\n' "$code" > "$ROOT/RUN_STATUS.json"
  exit "$code"
}
trap fail ERR

cd "$(dirname "$0")/.."
"$TEST_PYTHON" -m pytest -q \
  tests/test_cbf_paired_binding.py \
  tests/test_cbf_episodic_binding.py \
  tests/test_cbf_listwise_binding.py \
  tests/test_cbf_associative_binding.py \
  tests/test_cbf_candidate_ranking.py

printf '{"state":"running","stage":"smoke"}\n' > "$ROOT/RUN_STATUS.json"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_cbf_paired_binding smoke \
  --data-root "$DATA" --output "$ROOT/smoke_paired_listwise.json" --model "$MODEL" \
  --arm paired_listwise --memory-cap .3 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m tasks.train_cbf_paired_binding smoke \
  --data-root "$DATA" --output "$ROOT/smoke_paired_swap.json" --model "$MODEL" \
  --arm paired_swap --memory-cap .3 &
pid1=$!
wait "$pid0"
wait "$pid1"

"$PYTHON" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for arm in ('paired_listwise','paired_swap'):
    row=json.loads((root/f'smoke_{arm}.json').read_text())
    if row['peak_reserved_gib'] >= 28 or row['max_relative_delta'] > .305:
        raise SystemExit(f'unsafe paired smoke for {arm}: {row}')
PY

printf '{"state":"running","stage":"formal"}\n' > "$ROOT/RUN_STATUS.json"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_cbf_paired_binding run \
  --data-root "$DATA" --output "$ROOT/paired_listwise" --model "$MODEL" \
  --arm paired_listwise --rounds 2 --train-groups 32 --dev-groups 8 --memory-cap .3 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m tasks.train_cbf_paired_binding run \
  --data-root "$DATA" --output "$ROOT/paired_swap" --model "$MODEL" \
  --arm paired_swap --rounds 2 --train-groups 32 --dev-groups 8 --memory-cap .3 &
pid1=$!
wait "$pid0"
wait "$pid1"

"$PYTHON" -m tasks.train_cbf_paired_binding summarize --root "$ROOT"
"$PYTHON" -m scripts.audit_cbf_paired_binding --root "$ROOT" --data-root "$DATA" --model "$MODEL"
git rev-parse HEAD > "$ROOT/git_commit.txt"
"$PYTHON" - "$ROOT" <<'PY'
import datetime,json,sys
from pathlib import Path
root=Path(sys.argv[1])
(root/'RUN_STATUS.json').write_text(json.dumps({
    'state':'completed', 'updated_at':datetime.datetime.now().astimezone().isoformat(),
})+'\n')
PY
trap - ERR
