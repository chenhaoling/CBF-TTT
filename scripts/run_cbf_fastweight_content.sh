#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_fastweight_content_20261012}
SOURCE=${SOURCE:-/home/ctj/cbf_ttt_memory_value_20261009}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}

if [[ -e "$ROOT" ]]; then echo "ROOT already exists: $ROOT" >&2; exit 2; fi
"$PYTHON" -m unittest tests.test_cbf_fastweight_content tests.test_cbf_native_accumulation tests.test_cbf_checkpoint_native tests.test_ttt_train_infer
"$PYTHON" -m tasks.cbf_fastweight_content build --source "$SOURCE" --tokenizer "$TOKENIZER" --root "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT

pids=()
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m tasks.cbf_fastweight_content collect-q \
    --root "$ROOT" --model "$MODEL" --shard "$gpu" --output "$ROOT/q_${gpu}.jsonl" \
    > "$ROOT/q_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status != 0 )); then exit 1; fi
"$PYTHON" -m tasks.cbf_fastweight_content summarize-q --root "$ROOT" --output "$ROOT/q_summary.json"

if ! "$PYTHON" - "$ROOT/q_summary.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['passed'] else 1)
PY
then
  "$PYTHON" -m scripts.audit_cbf_fastweight_content --root "$ROOT" --tokenizer "$TOKENIZER" --output "$ROOT/execution_audit.json"
  echo stopped_by_content_Q > "$ROOT/terminal_status.txt"
  date -Is > "$ROOT/completed_at.txt"
  exit 0
fi

pids=()
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m tasks.cbf_fastweight_content collect-v \
    --root "$ROOT" --model "$MODEL" --selection "$ROOT/q_summary.json" --shard "$gpu" \
    --output "$ROOT/v_${gpu}.jsonl" > "$ROOT/v_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status != 0 )); then exit 1; fi
"$PYTHON" -m tasks.cbf_fastweight_content summarize --root "$ROOT" --output "$ROOT/summary.json"
"$PYTHON" -m scripts.audit_cbf_fastweight_content --root "$ROOT" --tokenizer "$TOKENIZER" --output "$ROOT/execution_audit.json"
"$PYTHON" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);a=json.loads((r/'execution_audit.json').read_text());s=json.loads((r/'summary.json').read_text())
if not a['audit_passed']:raise SystemExit('audit failed')
(r/'terminal_status.txt').write_text(s['terminal_status']+'\n')
PY
date -Is > "$ROOT/completed_at.txt"

