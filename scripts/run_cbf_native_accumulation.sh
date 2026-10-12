#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ctj/cbf_ttt_native_accumulation_20261012}
SOURCE=${SOURCE:-/home/ctj/cbf_ttt_checkpoint_native_20261010}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}

if [[ -e "$ROOT" ]]; then echo "ROOT already exists: $ROOT" >&2; exit 2; fi
"$PYTHON" -m unittest tests.test_cbf_native_accumulation tests.test_cbf_checkpoint_native tests.test_ttt_train_infer
"$PYTHON" -m tasks.cbf_native_accumulation build --source "$SOURCE" --root "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date -Is > "$ROOT/started_at.txt"
trap 'code=$?; if (( code != 0 )); then echo "$code" > "$ROOT/failed_exit_code.txt"; date -Is > "$ROOT/failed_at.txt"; fi' EXIT

pids=()
for gpu in 0 1; do
  (
    for arm in zero first_only last_only reset_each cumulative; do
      CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -m tasks.cbf_native_accumulation collect \
        --root "$ROOT" --model "$MODEL" --arm "$arm" --shard "$gpu" \
        --output "$ROOT/${arm}_${gpu}.jsonl" > "$ROOT/${arm}_${gpu}.log" 2>&1
    done
  ) &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status != 0 )); then exit 1; fi

"$PYTHON" -m tasks.cbf_native_accumulation summarize --root "$ROOT" --output "$ROOT/summary.json" > "$ROOT/summary.log" 2>&1
"$PYTHON" -m scripts.audit_cbf_native_accumulation --root "$ROOT" --output "$ROOT/execution_audit.json" > "$ROOT/audit.log" 2>&1
"$PYTHON" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]); summary=json.loads((r/'summary.json').read_text()); audit=json.loads((r/'execution_audit.json').read_text())
if not audit['audit_passed']: raise SystemExit('audit failed')
(r/'terminal_status.txt').write_text(summary['terminal_status']+'\n')
PY
date -Is > "$ROOT/completed_at.txt"

