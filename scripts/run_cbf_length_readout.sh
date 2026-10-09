#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_length_readout_20261010}
SOURCE=${SOURCE:-/home/ctj/cbf_ttt_disjoint_readout_20261009}
REFERENCE=${REFERENCE:-/home/ctj/cbf_ttt_checkpoint_readout_20261009}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
ORIGINAL=${ORIGINAL:-/home/ctj/models/Qwen3-4B}
FINAL=${FINAL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
mkdir "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT
"$PYTHON" -m unittest tests.test_cbf_length_readout tests.test_cbf_checkpoint_readout tests.test_cbf_disjoint_readout tests.test_cbf_memory_value > "$ROOT/tests.log" 2>&1
"$PYTHON" -m tasks.cbf_length_readout build --source "$SOURCE" --reference "$REFERENCE" --output "$ROOT/data" > "$ROOT/build.log" 2>&1
pids=()
for gpu in 0 1; do
  (
    for arm in original_repo final_repo; do
      model=$ORIGINAL
      if [[ "$arm" == final_repo ]]; then model=$FINAL; fi
      CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m tasks.cbf_length_readout collect \
        --arm "$arm" --shard "$gpu" --root "$ROOT" --model "$model" \
        --output "$ROOT/${arm}_${gpu}.jsonl" > "$ROOT/${arm}_${gpu}.log" 2>&1
    done
  ) &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status != 0 )); then exit 1; fi
"$PYTHON" -m tasks.cbf_length_readout summarize --root "$ROOT" --output "$ROOT/summary.json" > "$ROOT/summary.log" 2>&1
"$PYTHON" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);(r/'terminal_status.txt').write_text(json.loads((r/'summary.json').read_text())['terminal_status']+'\n')
PY
date -Is > "$ROOT/completed_at.txt"
