#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_disjoint_readout_20261009}
SOURCE=${SOURCE:-/home/ctj/cbf_ttt_memory_value_20261009}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
mkdir "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT
"$PYTHON" -m unittest tests.test_cbf_disjoint_readout tests.test_cbf_readability tests.test_cbf_memory_value tests.test_cbf_source_exclusion > "$ROOT/tests.log" 2>&1
"$PYTHON" -m tasks.cbf_disjoint_readout build --source "$SOURCE" --tokenizer "$TOKENIZER" --output "$ROOT/data" > "$ROOT/build.log" 2>&1
pids=()
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m tasks.cbf_disjoint_readout collect --model "$MODEL" \
    --data "$ROOT/data/scenes.jsonl" --shard "$gpu" --output "$ROOT/rows_${gpu}.jsonl" > "$ROOT/collect_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status != 0 )); then exit 1; fi
"$PYTHON" -m tasks.cbf_disjoint_readout summarize --data "$ROOT/data/scenes.jsonl" \
  --inputs "$ROOT/rows_0.jsonl" "$ROOT/rows_1.jsonl" --reference "$SOURCE/q_0.jsonl" "$SOURCE/q_1.jsonl" \
  --output "$ROOT/summary.json" > "$ROOT/summary.log" 2>&1
"$PYTHON" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);s=json.loads((r/'summary.json').read_text())
(r/'terminal_status.txt').write_text(s['terminal_status']+'\n')
PY
date -Is > "$ROOT/completed_at.txt"
