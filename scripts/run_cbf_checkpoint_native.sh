#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_checkpoint_native_20261010}
SOURCE=${SOURCE:-/home/ctj/cbf_ttt_length_readout_20261010}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
ORIGINAL=${ORIGINAL:-/home/ctj/models/Qwen3-4B}
CHECKPOINTS=${CHECKPOINTS:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints}
FINAL=$CHECKPOINTS/global_step_81381/hf_ckpt
mkdir "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT
"$PYTHON" -m unittest tests.test_cbf_checkpoint_native tests.test_ttt_train_infer tests.test_cbf_length_readout > "$ROOT/tests.log" 2>&1
"$PYTHON" -m tasks.cbf_checkpoint_native build --source "$SOURCE" --root "$ROOT" > "$ROOT/build.log" 2>&1
for step in 10000 40000; do
  CUDA_VISIBLE_DEVICES='' "$PYTHON" -m scripts.export_cbf_diagnostic_checkpoint \
    --source "$CHECKPOINTS/global_step_$step" --output "$ROOT/exports/step$step" \
    --config "$FINAL/config.json" > "$ROOT/export_$step.log" 2>&1
done
pids=()
for gpu in 0 1; do
  (
    for arm in original_plain step10000_plain step10000_native step40000_plain step40000_native step81381_plain step81381_native final_zero_lr; do
      case "$arm" in
        original_plain) model=$ORIGINAL ;;
        step10000_*) model=$ROOT/exports/step10000 ;;
        step40000_*) model=$ROOT/exports/step40000 ;;
        *) model=$FINAL ;;
      esac
      CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m tasks.cbf_checkpoint_native collect \
        --root "$ROOT" --arm "$arm" --shard "$gpu" --model "$model" \
        --output "$ROOT/${arm}_${gpu}.jsonl" > "$ROOT/${arm}_${gpu}.log" 2>&1
    done
  ) &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status != 0 )); then exit 1; fi
"$PYTHON" -m tasks.cbf_checkpoint_native summarize --root "$ROOT" --output "$ROOT/summary.json" > "$ROOT/summary.log" 2>&1
"$PYTHON" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);(r/'terminal_status.txt').write_text(json.loads((r/'summary.json').read_text())['terminal_status']+'\n')
PY
date -Is > "$ROOT/completed_at.txt"
