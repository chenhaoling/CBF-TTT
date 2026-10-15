#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_causal_ledger_confirmation_v3_20261015}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
RESIDUAL_V2_ROOT=${RESIDUAL_V2_ROOT:-/home/ctj/cbf_ttt_qwen_causal_ledger_residual_v2_retry_20261015}
SCALE_RESCUE_ROOT=${SCALE_RESCUE_ROOT:-/home/ctj/cbf_ttt_qwen_causal_ledger_scale_rescue_v3_20261015}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"
status=failed_qwen_causal_ledger_confirmation_v3
code=1
finish() {
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

"$PYTHON_BIN" -m unittest tests.test_qwen_causal_ledger_residual > "$ROOT/tests.log" 2>&1
"$PYTHON_BIN" - "$SCALE_RESCUE_ROOT" "$ROOT/scale_rescue_v3_dependency.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
prior=Path(sys.argv[1]); output=Path(sys.argv[2])
summary=json.loads((prior/'summary.json').read_text())
audit=json.loads((prior/'execution_audit.json').read_text())
if not summary['gate']['passed'] or not audit['passed']:
    raise SystemExit('scale-rescue V3 dependency did not pass')
files=('summary.json','execution_audit.json')
hashes={name:hashlib.sha256((prior/name).read_bytes()).hexdigest() for name in files}
output.write_text(json.dumps({'hashes':hashes},indent=2)+'\n')
PY
"$PYTHON_BIN" - "$RESIDUAL_V2_ROOT" <<'PY' > "$ROOT/v2_dependency.log"
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); summary=json.loads((root/'summary.json').read_text()); audit=json.loads((root/'execution_audit.json').read_text())
if not summary['gate']['passed'] or not audit['passed']:
    raise SystemExit('V2 content dependency did not pass')
print(json.dumps({'summary_sha256':hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),'audit_sha256':hashlib.sha256((root/'execution_audit.json').read_bytes()).hexdigest()},indent=2))
PY
"$PYTHON_BIN" -m tasks.build_qwen_causal_ledger_confirmation_v3 \
  --output "$ROOT/data" > "$ROOT/build.log" 2>&1

declare -a pids
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=$shard "$PYTHON_BIN" -m tasks.evaluate_qwen_causal_ledger_confirmation_v3 collect \
    --data-root "$ROOT/data" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --shard "$shard" --num-shards 2 --output "$ROOT/development_shard${shard}.jsonl" \
    --resource-output "$ROOT/development_resource_shard${shard}.json" \
    > "$ROOT/shard${shard}.log" 2>&1 &
  pids[$shard]=$!
done
wait "${pids[0]}"
wait "${pids[1]}"
"$PYTHON_BIN" -m tasks.evaluate_qwen_causal_ledger_confirmation_v3 summarize \
  --root "$ROOT" --output "$ROOT/summary.json" > "$ROOT/summarize.log" 2>&1
status=completed_qwen_causal_ledger_confirmation_v3
code=0
