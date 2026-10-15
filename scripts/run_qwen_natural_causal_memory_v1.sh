#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_natural_causal_memory_v1_20261015}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
SOURCE_ROOT=${SOURCE_ROOT:-/home/ctj/cbf_ttt_zsre_counterfactual_forgetting_v1_data}
CONTEXT_ROOT=${CONTEXT_ROOT:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal}
SCOPE_ROOT=${SCOPE_ROOT:-/home/ctj/cbf_ttt_zsre_decomposed_scope_gate_v1/formal}
CONTROLLER_ROOT=${CONTROLLER_ROOT:-/home/ctj/cbf_ttt_zsre_counterfactual_forgetting_controller_loso_v3/formal}
BOUNDARY_ROOT=${BOUNDARY_ROOT:-/home/ctj/cbf_ttt_zsre_forgetting_dev_boundary_v1}
VALUE_ROOT=${VALUE_ROOT:-/home/ctj/cbf_ttt_qwen_natural_value_bridge_v1_20261015_r1}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date --iso-8601=seconds > "$ROOT/started_at.txt"
status=failed_qwen_natural_causal_memory_v1
code=1
finish() {
  echo "$status" > "$ROOT/terminal_status.txt"
  echo "$code" > "$ROOT/runner_exit_code.txt"
  date --iso-8601=seconds > "$ROOT/completed_at.txt"
}
trap finish EXIT

"$PYTHON_BIN" -m unittest tests.test_qwen_natural_causal_memory_v1 \
  tests.test_qwen_natural_value_bridge > "$ROOT/tests.log" 2>&1
"$PYTHON_BIN" - "$SOURCE_ROOT" "$CONTEXT_ROOT" "$SCOPE_ROOT" \
  "$CONTROLLER_ROOT" "$BOUNDARY_ROOT" "$VALUE_ROOT" "$ROOT/dependencies.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source,context,scope,controller,boundary,value,output=map(Path,sys.argv[1:])
value_summary=json.loads((value/'development_summary.json').read_text())
value_audit=json.loads((value/'execution_audit.json').read_text())
boundary_summary=json.loads((boundary/'diagnostic/summary.json').read_text())
boundary_audit=json.loads((boundary/'execution_audit.json').read_text())
source_manifest=json.loads((source/'manifest.json').read_text())
selected=boundary_summary['selected_passing_original_gate']
if not value_summary['gate']['passed'] or not value_audit['passed'] or value_summary['selected_scale'] != 8.0:
    raise SystemExit('natural-value dependency did not pass at scale 8')
if not boundary_audit['audit_passed'] or not selected['development_gate']['passed'] \
        or float(selected['threshold']) != .675:
    raise SystemExit('forgetting-boundary dependency did not pass at threshold 0.675')
if source_manifest['test_scored']:
    raise SystemExit('counterfactual source test is not sealed')
paths=(
    source/'manifest.json',source/'calibration.jsonl',source/'dev.jsonl',
    context/'summary.json',context/'transforms.pt',
    scope/'training.json',scope/'scope_gates.pt',
    controller/'training.json',controller/'counterfactual_forgetting_action_oof_gate.pt',
    boundary/'diagnostic/summary.json',boundary/'execution_audit.json',
    value/'development_summary.json',value/'execution_audit.json',
)
output.write_text(json.dumps({str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in paths},indent=2)+'\n')
PY
"$PYTHON_BIN" -m tasks.build_qwen_natural_causal_memory_v1 \
  --source-root "$SOURCE_ROOT" --output "$ROOT/data" > "$ROOT/build.log" 2>&1
"$PYTHON_BIN" - "$ROOT/data" "$TOKENIZER" > "$ROOT/tokenization_preflight.log" <<'PY'
import json
import sys
from transformers import AutoTokenizer
from tasks.build_qwen_natural_causal_memory_v1 import load
from tasks.evaluate_mquake_qwen_outcome_v16 import encode_candidates
from tasks.evaluate_qwen_natural_value_bridge_v1 import render_query

data_root,tokenizer_path=sys.argv[1:]
tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,use_fast=True)
counts={}; lengths=[]
for split in ('calibration','dev'):
    _,sessions=load(data_root,split); queries=0
    for session in sessions:
        for chain in session['chains']:
            for version in chain['versions']:
                for query in (version['prompt'],version['rephrase_prompt']):
                    _,encoded=encode_candidates(tokenizer,render_query(query),session['candidates'])
                    lengths.extend(len(ids) for ids in encoded.values()); queries += 1
    counts[split]=queries
print(json.dumps({'passed':True,'unique_registered_queries':counts,
                  'candidate_tokenizations':len(lengths),
                  'minimum_answer_tokens':min(lengths),
                  'maximum_answer_tokens':max(lengths)},indent=2))
PY

run_stage() {
  local stage=$1
  unset pids 2>/dev/null || true
  declare -a pids
  for shard in 0 1; do
    CUDA_VISIBLE_DEVICES=$shard "$PYTHON_BIN" -m tasks.evaluate_qwen_natural_causal_memory_v1 collect \
      --data-root "$ROOT/data" --model "$MODEL" --tokenizer "$TOKENIZER" \
      --context-root "$CONTEXT_ROOT" --scope-root "$SCOPE_ROOT" \
      --controller-root "$CONTROLLER_ROOT" --stage "$stage" \
      --shard "$shard" --num-shards 2 --batch-size 128 \
      --output "$ROOT/${stage}_shard${shard}.jsonl" \
      --state-output "$ROOT/${stage}_states_shard${shard}.jsonl" \
      --resource-output "$ROOT/${stage}_resource_shard${shard}.json" \
      > "$ROOT/${stage}_shard${shard}.log" 2>&1 &
    pids[$shard]=$!
  done
  wait "${pids[0]}"
  wait "${pids[1]}"
  "$PYTHON_BIN" -m tasks.evaluate_qwen_natural_causal_memory_v1 summarize \
    --root "$ROOT" --stage "$stage" --output "$ROOT/${stage}_summary.json" \
    > "$ROOT/${stage}_summarize.log" 2>&1
}

run_stage calibration
if "$PYTHON_BIN" - "$ROOT/calibration_summary.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['gate']['passed'] else 1)
PY
then
  run_stage dev
  status=completed_qwen_natural_causal_memory_v1
else
  status=completed_qwen_natural_causal_memory_v1_calibration_stop
fi
code=0
