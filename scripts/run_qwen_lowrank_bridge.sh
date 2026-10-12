#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_lowrank_bridge_v1}
SOURCE=${SOURCE:-/home/ctj/cbf_ttt_fastweight_content_v2_20261012}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
if [[ -e "$ROOT" ]]; then echo "ROOT already exists: $ROOT" >&2; exit 2; fi
"$PYTHON" -m unittest tests.test_qwen_lowrank_bridge tests.test_final_writer_binding
"$PYTHON" -m tasks.cbf_qwen_lowrank_bridge build --source "$SOURCE" --root "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"; date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT

run_stage () {
  stage=$1; selection=${2:-}; pids=()
  for gpu in 0 1; do
    args=(--root "$ROOT" --model "$MODEL" --tokenizer "$TOKENIZER" --stage "$stage" --shard "$gpu" --output "$ROOT/${stage}_${gpu}.jsonl")
    if [[ -n "$selection" ]]; then args+=(--selection "$selection"); fi
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m tasks.cbf_qwen_lowrank_bridge collect "${args[@]}" > "$ROOT/${stage}_${gpu}.log" 2>&1 & pids+=("$!")
  done
  status=0; for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  if (( status != 0 )); then return 1; fi
}

run_stage dev
"$PYTHON" -m tasks.cbf_qwen_lowrank_bridge summarize-dev --root "$ROOT" \
  --inputs "$ROOT/dev_0.jsonl" "$ROOT/dev_1.jsonl" --output "$ROOT/dev_selection.json"
run_stage q "$ROOT/dev_selection.json"
"$PYTHON" -m tasks.cbf_qwen_lowrank_bridge summarize --root "$ROOT" --stage q \
  --selection "$ROOT/dev_selection.json" --inputs "$ROOT/q_0.jsonl" "$ROOT/q_1.jsonl" --output "$ROOT/q_summary.json"
if "$PYTHON" - "$ROOT/q_summary.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['passed'] else 1)
PY
then
  run_stage v "$ROOT/dev_selection.json"
  "$PYTHON" -m tasks.cbf_qwen_lowrank_bridge summarize --root "$ROOT" --stage v \
    --selection "$ROOT/dev_selection.json" --inputs "$ROOT/v_0.jsonl" "$ROOT/v_1.jsonl" --output "$ROOT/v_summary.json"
  echo completed_qwen_lowrank_bridge_v1 > "$ROOT/terminal_status.txt"
else
  echo stopped_by_qwen_lowrank_Q > "$ROOT/terminal_status.txt"
fi
"$PYTHON" -m scripts.audit_qwen_lowrank_bridge --root "$ROOT" --output "$ROOT/execution_audit.json"
date -Is > "$ROOT/completed_at.txt"
