#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ctj/cbf_ttt_forgetting_only_validation_v1}
REPO=${REPO:-/home/ctj/cbf_ttt_joint_exp_20260927}
PYTHON=${PYTHON:-/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python}
MODEL=${MODEL:-/home/ctj/cbf_ttt_pretrain_qwen3_4b_1b/checkpoints/global_step_81381/hf_ckpt}
STUDY=${STUDY:-/home/ctj/cbf_ttt_1b_labels/controller_study_20260927}
DATA=${DATA:-$STUDY/test_scenarios.jsonl}
WAIT_FOR_GPUS=${WAIT_FOR_GPUS:-1}
GPU_MEMORY_THRESHOLD_MIB=${GPU_MEMORY_THRESHOLD_MIB:-4000}
WAIT_POLL_SECONDS=${WAIT_POLL_SECONDS:-60}

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite existing ROOT: $ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT/logs" "$ROOT/results"
cd "$REPO"
git rev-parse HEAD > "$ROOT/code_commit.txt"
date -Is > "$ROOT/started_at.txt"
printf '%s\n' running > "$ROOT/status.txt"

fail() {
  code=$?
  printf '%s\n' failed > "$ROOT/status.txt"
  printf '%s\n' "$code" > "$ROOT/exit_code.txt"
  date -Is > "$ROOT/failed_at.txt"
  exit "$code"
}
trap fail ERR INT TERM

"$PYTHON" -m unittest \
  tests.test_cbf_forgetting_only_validation \
  tests.test_cbf_ttt \
  > "$ROOT/logs/tests.log" 2>&1

wait_for_gpu_pair() {
  if [[ "$WAIT_FOR_GPUS" != 1 ]]; then
    return
  fi
  stable=0
  while (( stable < 3 )); do
    mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [[ ${#used[@]} -ge 2 && ${used[0]} -lt $GPU_MEMORY_THRESHOLD_MIB && ${used[1]} -lt $GPU_MEMORY_THRESHOLD_MIB ]]; then
      stable=$((stable + 1))
    else
      stable=0
    fi
    printf '%s gpu_used_mib=%s,%s stable=%s/3\n' \
      "$(date -Is)" "${used[0]:-missing}" "${used[1]:-missing}" "$stable" \
      >> "$ROOT/logs/gpu_wait.log"
    if (( stable < 3 )); then
      sleep "$WAIT_POLL_SECONDS"
    fi
  done
}

collect_one() {
  gpu=$1
  name=$2
  policy=$3
  controller=${4:-}
  args=(
    -m tasks.cbf_forgetting_only_validation collect
    --model "$MODEL" --data "$DATA" --split test
    --policy "$policy" --policy-name "$name"
    --output "$ROOT/results/${name}.jsonl"
  )
  if [[ -n "$controller" ]]; then
    args+=(--controller "$controller")
  fi
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "${args[@]}" \
    > "$ROOT/logs/${name}.log" 2>&1
}

wait_for_gpu_pair

# A one-pair smoke checks both a fixed policy and a trained controller before the formal run.
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m tasks.cbf_forgetting_only_validation collect \
  --model "$MODEL" --data "$DATA" --split test --policy 1 --policy-name smoke_fixed_1 \
  --limit-pairs 1 --output "$ROOT/results/smoke_fixed_1.jsonl" \
  > "$ROOT/logs/smoke_fixed_1.log" 2>&1
CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m tasks.cbf_forgetting_only_validation collect \
  --model "$MODEL" --data "$DATA" --split test --policy controller --policy-name smoke_controller_seed42 \
  --controller "$STUDY/controller_g500_s42.pt" --limit-pairs 1 \
  --output "$ROOT/results/smoke_controller_seed42.jsonl" \
  > "$ROOT/logs/smoke_controller_seed42.log" 2>&1

(
  collect_one 0 fixed_0 0
  collect_one 0 fixed_1 1
  collect_one 0 controller_seed42 controller "$STUDY/controller_g500_s42.pt"
) > "$ROOT/logs/gpu0.log" 2>&1 &
pid0=$!
(
  collect_one 1 fixed_0.5 0.5
  collect_one 1 controller_seed43 controller "$STUDY/controller_g500_s43.pt"
  collect_one 1 controller_seed44 controller "$STUDY/controller_g500_s44.pt"
) > "$ROOT/logs/gpu1.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

"$PYTHON" -m tasks.cbf_forgetting_only_validation summarize \
  --inputs \
    "$ROOT/results/fixed_0.jsonl" \
    "$ROOT/results/fixed_0.5.jsonl" \
    "$ROOT/results/fixed_1.jsonl" \
    "$ROOT/results/controller_seed42.jsonl" \
    "$ROOT/results/controller_seed43.jsonl" \
    "$ROOT/results/controller_seed44.jsonl" \
  --output "$ROOT/summary.json" \
  > "$ROOT/logs/summarize.log" 2>&1

printf '%s\n' completed > "$ROOT/status.txt"
"$PYTHON" -m scripts.audit_cbf_forgetting_only_validation \
  --root "$ROOT" \
  --reference "$STUDY/test_all_rollouts.jsonl" \
  --output "$ROOT/execution_audit.json" \
  > "$ROOT/logs/audit.log" 2>&1
date -Is > "$ROOT/completed_at.txt"
printf '%s\n' 0 > "$ROOT/exit_code.txt"
trap - ERR INT TERM
