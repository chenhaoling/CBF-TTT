#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_dynamic_binding_v1}
MODEL=${MODEL:-/home/ctj/models/Qwen3-4B}
CORPUS=${CORPUS:-/home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite existing experiment root: $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
exec > >(tee -a "$ROOT/run.log") 2>&1

status() {
  "$PYTHON" - "$ROOT/RUN_STATUS.json" "$1" <<'PY'
import datetime,json,sys
path,state=sys.argv[1:]
with open(path,'w') as stream:
    json.dump({'state':state,'updated_at':datetime.datetime.now().astimezone().isoformat()},stream,indent=2)
    stream.write('\n')
PY
}
wait_pair() {
  local first=$1 second=$2 first_status second_status
  set +e
  wait "$first"; first_status=$?
  wait "$second"; second_status=$?
  set -e
  if [[ $first_status -ne 0 || $second_status -ne 0 ]]; then return 1; fi
}
trap 'status failed' ERR
status preflight
git rev-parse HEAD > "$ROOT/git_commit.txt"
"$PYTHON" -m unittest tests.test_cbf_dynamic_memory tests.test_cbf_dynamic_binding -v

status building_data
"$PYTHON" -m tasks.build_cbf_event_curriculum \
  --output "$ROOT/source" --seed 20261013 --train-groups 128 --dev-groups 32 --test-groups 32
"$PYTHON" -m tasks.pack_cbf_dynamic_memory \
  --source "$ROOT/source" --output "$ROOT/data" --tokenizer "$MODEL" --corpus "$CORPUS" --chunk 4096

status smoke
mkdir "$ROOT/smoke"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_cbf_dynamic_binding run \
  --data-root "$ROOT/data" --output "$ROOT/smoke/joint_ce" --model "$MODEL" \
  --arm joint_ce --rounds 1 --train-groups 8 --dev-groups 2 &
smoke_ce=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m tasks.train_cbf_dynamic_binding run \
  --data-root "$ROOT/data" --output "$ROOT/smoke/joint_binding" --model "$MODEL" \
  --arm joint_binding --rounds 1 --train-groups 8 --dev-groups 2 &
smoke_binding=$!
wait_pair "$smoke_ce" "$smoke_binding"
"$PYTHON" -m tasks.train_cbf_dynamic_binding summarize --root "$ROOT/smoke"
"$PYTHON" -m scripts.audit_cbf_dynamic_binding --root "$ROOT/smoke" --data-root "$ROOT/data"
"$PYTHON" - "$ROOT/smoke" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for arm in ('joint_ce','joint_binding'):
    complete=json.loads((root/arm/'complete.json').read_text())
    if complete['peak_reserved_gib'] >= 28 or complete['max_relative_delta'] > 1:
        raise SystemExit(f'{arm} smoke exceeded a resource/safety limit')
PY

status formal
mkdir "$ROOT/formal"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_cbf_dynamic_binding run \
  --data-root "$ROOT/data" --output "$ROOT/formal/joint_ce" --model "$MODEL" \
  --arm joint_ce --rounds 4 &
formal_ce=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m tasks.train_cbf_dynamic_binding run \
  --data-root "$ROOT/data" --output "$ROOT/formal/joint_binding" --model "$MODEL" \
  --arm joint_binding --rounds 4 &
formal_binding=$!
wait_pair "$formal_ce" "$formal_binding"
"$PYTHON" -m tasks.train_cbf_dynamic_binding summarize --root "$ROOT/formal"
"$PYTHON" -m scripts.audit_cbf_dynamic_binding --root "$ROOT/formal" --data-root "$ROOT/data"
status completed
