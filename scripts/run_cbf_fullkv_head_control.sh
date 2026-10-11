#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_fullkv_head_control_v1}
DATA=${DATA:-/home/ctj/cbf_ttt_dynamic_binding_v1/data}
MODEL=${MODEL:-/home/ctj/models/Qwen3-4B}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
TEST_PYTHON=${TEST_PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_verify_20260923/bin/python}

if [[ -e "$ROOT" ]]; then echo "Refusing to overwrite existing root: $ROOT" >&2; exit 2; fi
mkdir -p "$ROOT"
printf '{"state":"running","stage":"tests"}\n' > "$ROOT/RUN_STATUS.json"
fail() { code=$?; printf '{"state":"failed","exit_code":%s}\n' "$code" > "$ROOT/RUN_STATUS.json"; exit "$code"; }
trap fail ERR

cd "$(dirname "$0")/.."
"$TEST_PYTHON" -m pytest -q tests/test_cbf_fullkv_head_control.py tests/test_cbf_retrieval_head.py tests/test_cbf_episodic_binding.py

printf '{"state":"running","stage":"smoke"}\n' > "$ROOT/RUN_STATUS.json"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_cbf_fullkv_head_control smoke --data-root "$DATA" --output "$ROOT/smoke_same_lr.json" --model "$MODEL" --arm same_lr &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m tasks.train_cbf_fullkv_head_control smoke --data-root "$DATA" --output "$ROOT/smoke_high_lr.json" --model "$MODEL" --arm high_lr &
pid1=$!
wait "$pid0"; wait "$pid1"
"$PYTHON" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for arm in ('same_lr','high_lr'):
    row=json.loads((root/f'smoke_{arm}.json').read_text())
    if row['peak_reserved_gib'] >= 28 or not 0 <= row['accuracy'] <= 1:
        raise SystemExit(f'unsafe fullkv head smoke: {row}')
PY

printf '{"state":"running","stage":"formal"}\n' > "$ROOT/RUN_STATUS.json"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_cbf_fullkv_head_control run --data-root "$DATA" --output "$ROOT/same_lr" --model "$MODEL" --arm same_lr --cycles 4 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m tasks.train_cbf_fullkv_head_control run --data-root "$DATA" --output "$ROOT/high_lr" --model "$MODEL" --arm high_lr --cycles 4 &
pid1=$!
wait "$pid0"; wait "$pid1"

"$PYTHON" -m tasks.train_cbf_fullkv_head_control summarize --root "$ROOT"
"$PYTHON" -m scripts.audit_cbf_fullkv_head_control --root "$ROOT" --data-root "$DATA"
git rev-parse HEAD > "$ROOT/git_commit.txt"
"$PYTHON" - "$ROOT" <<'PY'
import datetime,json,sys
from pathlib import Path
root=Path(sys.argv[1]); (root/'RUN_STATUS.json').write_text(json.dumps({
  'state':'completed','updated_at':datetime.datetime.now().astimezone().isoformat()})+'\n')
PY
trap - ERR
