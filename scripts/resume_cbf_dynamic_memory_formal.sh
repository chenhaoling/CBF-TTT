#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_dynamic_memory_v1}
MODEL=${MODEL:-/home/ctj/models/Qwen3-4B}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
FAILED_TAG=${FAILED_TAG:-formal_failed_e7f7ff6}

exec >> >(tee -a "$ROOT/run.log") 2>&1
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
trap 'status failed_after_streaming_fix' ERR

"$PYTHON" - "$ROOT/smoke/execution_audit.json" <<'PY'
import json,sys
if not json.load(open(sys.argv[1]))['passed']:
    raise SystemExit('smoke audit did not pass')
PY
if [[ -e "$ROOT/$FAILED_TAG" ]]; then
  echo "Preserved failure target already exists: $ROOT/$FAILED_TAG" >&2
  exit 2
fi
if [[ -e "$ROOT/formal" ]]; then mv "$ROOT/formal" "$ROOT/$FAILED_TAG"; fi
git rev-parse HEAD > "$ROOT/formal_resume_git_commit.txt"
"$PYTHON" -m unittest tests.test_cbf_dynamic_memory -v

status formal_streaming_evaluation
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
