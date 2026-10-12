#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_final_writer_binding_v1}
SOURCE=${SOURCE:-/home/ctj/cbf_ttt_fastweight_content_v2_20261012}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
if [[ -e "$ROOT" ]]; then echo "ROOT already exists: $ROOT" >&2; exit 2; fi

"$PYTHON" -m unittest tests.test_final_writer_binding tests.test_cbf_event_writer tests.test_cbf_fastweight_content_v2
"$PYTHON" -m tasks.train_final_writer_binding build --source "$SOURCE" --root "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT

run_tune () {
  gpu=$1; arm=$2; lr=$3
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m tasks.train_final_writer_binding train \
    --root "$ROOT" --model "$MODEL" --output "$ROOT/$arm" --stage tune --arm "$arm" \
    --groups 0,1,2,3,4,7 --rounds 8 --lr "$lr" > "$ROOT/${arm}_train.log" 2>&1
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m tasks.train_final_writer_binding evaluate \
    --root "$ROOT" --model "$MODEL" --checkpoint "$ROOT/$arm/writer.pt" \
    --groups 5,6 --stage dev --output "$ROOT/$arm/dev.jsonl" > "$ROOT/${arm}_dev.log" 2>&1
  "$PYTHON" -m tasks.train_final_writer_binding summarize --root "$ROOT" --stage dev \
    --inputs "$ROOT/$arm/dev.jsonl" --output "$ROOT/$arm/dev_summary.json"
}

run_tune 0 lr3e7 3e-7 & p0=$!
run_tune 1 lr1e6 1e-6 & p1=$!
status=0; wait "$p0" || status=1; wait "$p1" || status=1
if (( status != 0 )); then exit 1; fi

"$PYTHON" -m tasks.train_final_writer_binding select-tune --root "$ROOT" \
  --arms lr3e7 lr1e6 --output "$ROOT/tune_selection.json"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.train_final_writer_binding train \
  --root "$ROOT" --model "$MODEL" --output "$ROOT/final" --stage final --arm selected \
  --groups 0,1,2,3,4,5,6,7 --rounds 8 --selection "$ROOT/tune_selection.json" \
  > "$ROOT/final_train.log" 2>&1

run_eval () {
  stage=$1; groups=$2; pids=()
  for gpu in 0 1; do
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m tasks.train_final_writer_binding evaluate \
      --root "$ROOT" --model "$MODEL" --checkpoint "$ROOT/final/writer.pt" \
      --groups "$groups" --stage "$stage" --shard "$gpu" --output "$ROOT/final/${stage}_${gpu}.jsonl" \
      > "$ROOT/${stage}_${gpu}.log" 2>&1 & pids+=("$!")
  done
  status=0; for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  if (( status != 0 )); then return 1; fi
  "$PYTHON" -m tasks.train_final_writer_binding summarize --root "$ROOT" --stage "$stage" \
    --inputs "$ROOT/final/${stage}_0.jsonl" "$ROOT/final/${stage}_1.jsonl" \
    --output "$ROOT/final/${stage}_summary.json"
}

run_eval q 8,9,10,11,12,13,14,15
if "$PYTHON" - "$ROOT/final/q_summary.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['passed'] else 1)
PY
then
  run_eval v 16,17,18,19,20,21,22,23
  echo completed_final_writer_binding_v1 > "$ROOT/terminal_status.txt"
else
  echo stopped_by_final_writer_Q > "$ROOT/terminal_status.txt"
fi
"$PYTHON" -m scripts.audit_final_writer_binding --root "$ROOT" --output "$ROOT/execution_audit.json"
date -Is > "$ROOT/completed_at.txt"
