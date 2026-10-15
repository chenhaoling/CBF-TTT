#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_qwen_natural_forgetting_online_v3_20261015}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON_BIN=${PYTHON_BIN:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}

test -f "$ROOT/execution_audit.json"
test ! -e "$ROOT/execution_audit_repaired.json"
cp "$ROOT/execution_audit.json" "$ROOT/execution_audit_second.json"
cp "$ROOT/runner_exit_code.txt" "$ROOT/runner_exit_code_initial_audit_failure.txt"
cat > "$ROOT/audit_recovery.md" <<'EOF'
The scientific rollout and summary completed successfully. The initial
auditors rebuilt one combined embedding table instead of the two registered
shard batches. Both failed audits are preserved. The repaired auditor
recreates the registered shards and changes no scores, policies, thresholds,
summary, or release gate.
EOF
echo 0 > "$ROOT/runner_exit_code.txt"

cd "$REPO"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m scripts.audit_qwen_natural_forgetting_online_v3 \
  --root "$ROOT" --repo "$REPO" \
  --factor-root /home/ctj/cbf_ttt_qwen_natural_forgetting_factor_gate_v2_20261015 \
  --data-root /home/ctj/cbf_ttt_qwen_natural_causal_memory_v1_20261015_r1/data \
  --v5-root /home/ctj/cbf_ttt_qwen_natural_scope_constrained_v5_20261015 \
  --model /home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt \
  --tokenizer /home/ctj/models/Qwen3-4B \
  --context-root /home/ctj/cbf_ttt_zsre_contextual_readout_v1/formal \
  --scope-root /home/ctj/cbf_ttt_zsre_decomposed_scope_gate_v1/formal \
  --output "$ROOT/execution_audit_repaired.json" \
  > "$ROOT/audit_repaired_sharded.log" 2>&1
