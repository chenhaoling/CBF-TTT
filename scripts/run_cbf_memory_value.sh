#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_memory_value_20261009}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
FINEWEB=${FINEWEB:-/home/ctj/data/cbf_ttt_memory_value_sources/sample/10BT/013_00000.parquet}
LONGCRAWL=${LONGCRAWL:-/home/ctj/data/cbf_ttt_pilot/longcrawl64/train/0-of-256.parquet}
TRAINING_DATA=${TRAINING_DATA:-/home/ctj/data/cbf_ttt_1b/mixed_1b.jsonl}
mkdir "$ROOT"
date -Is > "$ROOT/started_at.txt"
git rev-parse HEAD > "$ROOT/code_commit.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT
"$PYTHON" -m unittest tests.test_cbf_memory_value tests.test_cbf_selective tests.test_cbf_ttt > "$ROOT/tests.log" 2>&1
"$PYTHON" -m tasks.build_cbf_memory_value --fineweb "$FINEWEB" --longcrawl "$LONGCRAWL" \
  --training-data "$TRAINING_DATA" --tokenizer "$TOKENIZER" --old-root /home/ctj --output "$ROOT/data" > "$ROOT/build.log" 2>&1
DATA="$ROOT/data/scenes.jsonl"
DESIGN="$ROOT/data/design.json"
run_stage() {
  local stage=$1
  shift
  local pids=()
  for gpu in 0 1; do
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m tasks.cbf_memory_value collect \
      --model "$MODEL" --data "$DATA" --design "$DESIGN" --stage "$stage" \
      --shard "$gpu" --output "$ROOT/${stage}_${gpu}.jsonl" "$@" > "$ROOT/${stage}_${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  if (( status != 0 )); then return 1; fi
}
summarize() {
  local stage=$1
  shift
  "$PYTHON" -m tasks.cbf_memory_value summarize --data "$DATA" --design "$DESIGN" --stage "$stage" \
    --inputs "$ROOT/${stage}_0.jsonl" "$ROOT/${stage}_1.jsonl" --output "$ROOT/${stage}_summary.json" "$@" > "$ROOT/${stage}_summary.log" 2>&1
}
finish() {
  echo "$1" > "$ROOT/terminal_status.txt"
  date -Is > "$ROOT/completed_at.txt"
}
# Readability is checked before costly counterfactual writer paths.
run_stage q
summarize q
if ! "$PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["Q"]["passed"] else 1)' "$ROOT/q_summary.json"; then
  finish stopped_by_Q
  exit 0
fi
run_stage smoke
summarize smoke
run_stage repeat
summarize repeat --reference "$ROOT/smoke_0.jsonl" "$ROOT/smoke_1.jsonl"
if ! "$PYTHON" -c 'import json,sys; a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); sys.exit(0 if a["resource_passed"] and b["repeat_passed"] else 1)' "$ROOT/smoke_summary.json" "$ROOT/repeat_summary.json"; then
  finish stopped_by_resource_or_replay
  exit 0
fi
run_stage dev
summarize dev --q-summary "$ROOT/q_summary.json"
if ! "$PYTHON" -c 'import json,sys; sys.exit(0 if any(v["passed"] for v in json.load(open(sys.argv[1]))["V"].values()) else 1)' "$ROOT/dev_summary.json"; then
  finish stopped_by_V_dev
  exit 0
fi
run_stage confirm --selection "$ROOT/dev_summary.json"
summarize confirm --q-summary "$ROOT/q_summary.json" --selection "$ROOT/dev_summary.json"
# S is a separately budgeted conditional stage; never claim it has run here.
"$PYTHON" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);s=json.loads((p/'confirm_summary.json').read_text())
need_s=any(v['passed'] and s['F'][mode]['passed'] for mode,v in s['V'].items())
(p/'next_stage.json').write_text(json.dumps({'S_required':need_s,'S_executed':False})+'\n')
PY
finish V_F_complete
