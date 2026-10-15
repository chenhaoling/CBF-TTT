#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_native_writer_ledger_bridge_20261015}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
CONFIRMATION_ROOT=${CONFIRMATION_ROOT:-/home/ctj/cbf_ttt_qwen_causal_ledger_confirmation_v3_20261015}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"
status=failed_qwen_native_writer_ledger_bridge
code=1
finish() {
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

"$PYTHON_BIN" -m unittest tests.test_qwen_causal_ledger_residual > "$ROOT/tests.log" 2>&1
"$PYTHON_BIN" - "$CONFIRMATION_ROOT" <<'PY' > "$ROOT/confirmation_dependency.json"
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); summary=json.loads((root/'summary.json').read_text()); audit=json.loads((root/'execution_audit.json').read_text())
if not summary['gate']['passed'] or not audit['passed']:
    raise SystemExit('confirmation dependency did not pass')
print(json.dumps({'summary_sha256':hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),'audit_sha256':hashlib.sha256((root/'execution_audit.json').read_bytes()).hexdigest()},indent=2))
PY

declare -a pids
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=$shard "$PYTHON_BIN" -m tasks.pilot_qwen_native_writer_ledger_bridge collect \
    --model "$MODEL" --tokenizer "$TOKENIZER" --shard "$shard" --num-shards 2 \
    --output "$ROOT/bridge_shard${shard}.jsonl" \
    --resource-output "$ROOT/bridge_resource_shard${shard}.json" \
    > "$ROOT/shard${shard}.log" 2>&1 &
  pids[$shard]=$!
done
wait "${pids[0]}"
wait "${pids[1]}"
"$PYTHON_BIN" -m tasks.pilot_qwen_native_writer_ledger_bridge summarize \
  --root "$ROOT" --output "$ROOT/summary.json" > "$ROOT/summarize.log" 2>&1
status=completed_qwen_native_writer_ledger_bridge
code=0
