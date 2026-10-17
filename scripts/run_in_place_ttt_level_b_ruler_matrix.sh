#!/usr/bin/env bash
set -Eeuo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
PYTHON_BIN=${PYTHON_BIN:?set the isolated OpenCompass Python}
TOKENIZER=${TOKENIZER:?set the original Qwen3-4B tokenizer directory}
NO_TTT_MODEL=${NO_TTT_MODEL:?set the audited no-TTT Stage-2 HF model}
TTT_MODEL=${TTT_MODEL:?set the audited TTT Stage-2 HF model}
NO_TTT_TRAINING_ROOT=${NO_TTT_TRAINING_ROOT:?set the no-TTT Stage-2 training root}
TTT_TRAINING_ROOT=${TTT_TRAINING_ROOT:?set the TTT Stage-2 training root}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new formal RULER matrix root}
RULER_GPUS=${RULER_GPUS:?select seven comma-separated 80GB GPUs}
PLAN=${PLAN:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_PUBLIC_CORPUS_PLAN.md}
EVALUATION_ADDENDUM=${EVALUATION_ADDENDUM:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_LEVEL_B_RULER_ADDENDUM.md}

IFS=',' read -r -a gpus <<< "$RULER_GPUS"
if [[ ${#gpus[@]} -ne 7 ]]; then echo "exactly seven GPUs are required" >&2; exit 2; fi
if [[ "$(printf '%s\n' "${gpus[@]}" | sort -u | wc -l | tr -d ' ')" -ne 7 ]]; then
  echo "RULER GPU selection contains duplicates" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.partial" ]]; then
  echo "refusing to overwrite output or partial path" >&2; exit 3
fi
test -z "$(git -C "$TOOL_REPO" status --short)"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"

staging="${OUTPUT_ROOT}.partial"
mkdir -p "$staging"
status_file=$staging/status.txt
finish() {
  rc=$?
  trap - EXIT
  if [[ $rc -eq 0 ]]; then
    printf 'COMPLETED\nfinished_utc=%s\nexit_code=0\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_file"
    mv "$staging" "$OUTPUT_ROOT"
  else
    printf 'FAILED\nfinished_utc=%s\nexit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" > "$status_file"
  fi
  exit "$rc"
}
trap finish EXIT
printf 'RUNNING\nstarted_utc=%s\ngpus=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RULER_GPUS" > "$status_file"

contexts=(4096 8192 16384 32768 65536 131072 262144)
suffixes=(4k 8k 16k 32k 64k 128k 256k)
for arm in no_ttt ttt; do
  if [[ "$arm" == no_ttt ]]; then
    model=$NO_TTT_MODEL
    training_root=$NO_TTT_TRAINING_ROOT
  else
    model=$TTT_MODEL
    training_root=$TTT_TRAINING_ROOT
  fi
  pids=()
  for index in "${!contexts[@]}"; do
    context=${contexts[$index]}
    suffix=${suffixes[$index]}
    gpu=${gpus[$index]}
    (
      export UPSTREAM_REPO TOOL_REPO PYTHON_BIN TOKENIZER PLAN EVALUATION_ADDENDUM
      export MODEL="$model"
      export TRAINING_ROOT="$training_root"
      export OUTPUT_ROOT="$staging/$arm/$suffix"
      bash "$TOOL_REPO/scripts/run_in_place_ttt_level_b_ruler_single.sh" \
        "$context" "$arm" "$gpu"
    ) > "$staging/${arm}_${suffix}.launcher.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if [[ $failed -ne 0 ]]; then
    echo "$arm RULER wave failed" >&2; exit 4
  fi
done

"$PYTHON_BIN" "$TOOL_REPO/scripts/summarize_in_place_ttt_level_b_ruler_matrix.py" \
  --root "$staging" --output "$staging/matrix_summary.json" > "$staging/matrix_summary.log"
"$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_level_b_ruler_matrix.py" \
  --root "$staging" --output "$staging/independent_matrix_audit.json" \
  > "$staging/independent_matrix_audit.log"
