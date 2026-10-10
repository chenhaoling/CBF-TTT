#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_dynamic_memory_v1}
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
  if [[ $first_status -ne 0 || $second_status -ne 0 ]]; then
    return 1
  fi
}
trap 'status failed' ERR
status preflight
git rev-parse HEAD > "$ROOT/git_commit.txt"
"$PYTHON" -m unittest tests/test_cbf_dynamic_memory.py -v

status building_data
"$PYTHON" -m tasks.build_cbf_event_curriculum \
  --output "$ROOT/source" --seed 20261012 --train-groups 64 --dev-groups 16 --test-groups 16
"$PYTHON" -m tasks.pack_cbf_dynamic_memory \
  --source "$ROOT/source" --output "$ROOT/data" --tokenizer "$MODEL" --corpus "$CORPUS" --chunk 4096

status smoke
mkdir "$ROOT/smoke"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_cbf_dynamic_memory run \
  --data-root "$ROOT/data" --output "$ROOT/smoke/sequential" --model "$MODEL" \
  --arm sequential --rounds 1 --train-groups 8 --dev-groups 2 &
smoke_sequential=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m tasks.train_cbf_dynamic_memory run \
  --data-root "$ROOT/data" --output "$ROOT/smoke/joint_context" --model "$MODEL" \
  --arm joint_context --rounds 1 --train-groups 8 --dev-groups 2 &
smoke_joint=$!
wait_pair "$smoke_sequential" "$smoke_joint"
"$PYTHON" -m tasks.train_cbf_dynamic_memory summarize --root "$ROOT/smoke"
"$PYTHON" -m scripts.audit_cbf_dynamic_memory --root "$ROOT/smoke" --data-root "$ROOT/data"
"$PYTHON" - "$ROOT/smoke" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for arm in ('sequential','joint_context'):
    complete=json.loads((root/arm/'complete.json').read_text())
    if complete['peak_reserved_gib'] >= 28:
        raise SystemExit(f'{arm} smoke exceeded 28 GiB reserve limit')
PY

status formal
mkdir "$ROOT/formal"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_cbf_dynamic_memory run \
  --data-root "$ROOT/data" --output "$ROOT/formal/sequential" --model "$MODEL" \
  --arm sequential --rounds 4 &
formal_sequential=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m tasks.train_cbf_dynamic_memory run \
  --data-root "$ROOT/data" --output "$ROOT/formal/joint_context" --model "$MODEL" \
  --arm joint_context --rounds 4 &
formal_joint=$!
wait_pair "$formal_sequential" "$formal_joint"
"$PYTHON" -m tasks.train_cbf_dynamic_memory summarize --root "$ROOT/formal"
"$PYTHON" -m scripts.audit_cbf_dynamic_memory --root "$ROOT/formal" --data-root "$ROOT/data"
status completed
