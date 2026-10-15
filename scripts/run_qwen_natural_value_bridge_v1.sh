#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_natural_value_bridge_v1_20261015}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
SOURCE_ROOT=${SOURCE_ROOT:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/data}
OOD_ROOT=${OOD_ROOT:-/home/ctj/cbf_ttt_qwen_causal_ledger_ood_v1_20261015}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"
status=failed_qwen_natural_value_bridge_v1
code=1
finish() {
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

"$PYTHON_BIN" -m unittest tests.test_qwen_natural_value_bridge > "$ROOT/tests.log" 2>&1
"$PYTHON_BIN" - "$OOD_ROOT" "$SOURCE_ROOT" "$ROOT/dependencies.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
ood,source,output=map(Path,sys.argv[1:])
summary=json.loads((ood/'summary.json').read_text()); audit=json.loads((ood/'execution_audit.json').read_text())
manifest=json.loads((source/'manifest.json').read_text())
if not summary['gate']['passed'] or not audit['passed']:
    raise SystemExit('lifecycle OOD dependency did not pass')
if manifest['test_loaded'] or manifest['test_scored']:
    raise SystemExit('ZsRE contextual source test is not sealed')
paths=(ood/'summary.json',ood/'execution_audit.json',source/'manifest.json',source/'calibration.jsonl',source/'dev.jsonl')
output.write_text(json.dumps({str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},indent=2)+'\n')
PY
"$PYTHON_BIN" -m tasks.build_qwen_natural_value_bridge_v1 \
  --source-root "$SOURCE_ROOT" --output "$ROOT/data" > "$ROOT/build.log" 2>&1

declare -a pids
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=$shard "$PYTHON_BIN" -m tasks.evaluate_qwen_natural_value_bridge_v1 collect \
    --data-root "$ROOT/data" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --stage calibration --shard "$shard" --num-shards 2 \
    --output "$ROOT/calibration_shard${shard}.jsonl" \
    --resource-output "$ROOT/calibration_resource_shard${shard}.json" \
    > "$ROOT/calibration_shard${shard}.log" 2>&1 &
  pids[$shard]=$!
done
wait "${pids[0]}"
wait "${pids[1]}"
"$PYTHON_BIN" -m tasks.evaluate_qwen_natural_value_bridge_v1 summarize-calibration \
  --root "$ROOT" --output "$ROOT/calibration_summary.json" \
  > "$ROOT/calibration_summarize.log" 2>&1

if "$PYTHON_BIN" - "$ROOT/calibration_summary.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['gate']['passed'] else 1)
PY
then
  unset pids
  declare -a pids
  for shard in 0 1; do
    CUDA_VISIBLE_DEVICES=$shard "$PYTHON_BIN" -m tasks.evaluate_qwen_natural_value_bridge_v1 collect \
      --data-root "$ROOT/data" --model "$MODEL" --tokenizer "$TOKENIZER" \
      --stage dev --selection "$ROOT/calibration_summary.json" \
      --shard "$shard" --num-shards 2 \
      --output "$ROOT/dev_shard${shard}.jsonl" \
      --resource-output "$ROOT/dev_resource_shard${shard}.json" \
      > "$ROOT/dev_shard${shard}.log" 2>&1 &
    pids[$shard]=$!
  done
  wait "${pids[0]}"
  wait "${pids[1]}"
  "$PYTHON_BIN" -m tasks.evaluate_qwen_natural_value_bridge_v1 summarize-development \
    --root "$ROOT" --calibration "$ROOT/calibration_summary.json" \
    --output "$ROOT/development_summary.json" > "$ROOT/development_summarize.log" 2>&1
  status=completed_qwen_natural_value_bridge_v1
else
  status=completed_qwen_natural_value_bridge_v1_calibration_stop
fi
code=0
