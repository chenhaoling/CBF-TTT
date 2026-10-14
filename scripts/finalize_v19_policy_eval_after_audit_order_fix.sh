#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=${SOURCE_ROOT:-/home/ctj/cbf_ttt_v19_nested_forgetting_policy_qwen_20261014_retry1}
ROOT=${ROOT:-/home/ctj/cbf_ttt_v19_nested_forgetting_policy_qwen_20261014_retry2_audit_recovery}
DATA=${DATA:-/home/ctj/cbf_ttt_controlled_interference_v19_data_20261014_retry1/result}
OUTCOME=${OUTCOME:-/home/ctj/cbf_ttt_controlled_interference_v19_qwen_20261014}
RULE_ROOT=${RULE_ROOT:-/home/ctj/cbf_ttt_v19_causal_rule_diagnostic_20261014}
EXAMPLES=${EXAMPLES:-/home/ctj/cbf_ttt_v19_nested_policy_examples_20261014}
POLICY=${POLICY:-/home/ctj/cbf_ttt_v19_nested_forgetting_policy_20261014_retry1}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
TOKENIZER=${TOKENIZER:-/home/ctj/models/Qwen3-4B}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test ! -e "$ROOT"
test "$(cat "$SOURCE_ROOT/runner_exit_code.txt")" = "1"
"$PYTHON" - "$SOURCE_ROOT" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
audit = json.loads((root / "audit.json").read_text())
checks = audit["checks"]
assert summary["terminal_status"] == "v19_nested_policy_not_supported"
assert audit["audit_passed"] is False
assert checks["shard_counts_dependencies_and_resources"] is False
assert all(value is True for key, value in checks.items()
           if key != "shard_counts_dependencies_and_resources")
PY

mkdir -p "$ROOT"
trap 'code=$?; echo "$code" > "$ROOT/runner_exit_code.txt"' EXIT
date --iso-8601=seconds > "$ROOT/started_at.txt"
git -C "$REPO" rev-parse HEAD > "$ROOT/code_commit.txt"
printf '%s\n' "$SOURCE_ROOT" > "$ROOT/source_failed_audit_root.txt"
for name in shard0.jsonl shard1.jsonl shard0.jsonl.audit.json shard1.jsonl.audit.json summary.json; do
  cp "$SOURCE_ROOT/$name" "$ROOT/$name"
done

cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -m unittest tests.test_audit_v19_nested_forgetting_policy_outcome \
  > "$ROOT/tests.log" 2>&1
"$PYTHON" -m scripts.audit_v19_nested_forgetting_policy_outcome \
  --data-root "$DATA" --outcome-root "$OUTCOME" --rule-root "$RULE_ROOT" \
  --examples-root "$EXAMPLES" --policy-root "$POLICY" --model "$MODEL" \
  --tokenizer "$TOKENIZER" --inputs "$ROOT/shard0.jsonl" "$ROOT/shard1.jsonl" \
  --summary-path "$ROOT/summary.json" --output "$ROOT/audit.json" \
  > "$ROOT/audit.log" 2>&1
echo completed_v19_nested_policy_eval_audit_recovery > "$ROOT/terminal_status.txt"
date --iso-8601=seconds > "$ROOT/completed_at.txt"
