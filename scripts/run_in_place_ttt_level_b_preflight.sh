#!/usr/bin/env bash
set -Eeuo pipefail

stage=${1:?usage: run_in_place_ttt_level_b_preflight.sh stage1|stage2}
if [[ "$stage" != stage1 && "$stage" != stage2 ]]; then
  echo "stage must be stage1 or stage2" >&2
  exit 2
fi

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the CBF-TTT checkout}
PYTHON_BIN=${PYTHON_BIN:?set the official training Python}
CORPUS_ROOT=${CORPUS_ROOT:?set the independently audited full corpus}
BASE_MODEL=${BASE_MODEL:?set the original Qwen3-4B directory}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new preflight output root}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?select exactly eight GPUs explicitly}
export CUDA_VISIBLE_DEVICES
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1

IFS=',' read -r -a selected_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#selected_gpus[@]} -ne 8 ]]; then
  echo "exactly eight CUDA_VISIBLE_DEVICES are required" >&2
  exit 3
fi
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.partial" ]]; then
  echo "refusing to overwrite $OUTPUT_ROOT or its partial path" >&2
  exit 4
fi

case "$stage" in
  stage1)
    seq_len=32768
    one_step_tokens=2097152
    data_path="$CORPUS_ROOT/stage1_32k.jsonl"
    no_ttt_model="$BASE_MODEL"
    ttt_model="$BASE_MODEL"
    ;;
  stage2)
    seq_len=131072
    one_step_tokens=8388608
    data_path="$CORPUS_ROOT/stage2_128k.jsonl"
    no_ttt_model=${STAGE1_NO_TTT_MODEL:?set converted Stage-1 no-TTT HF weights}
    ttt_model=${STAGE1_TTT_MODEL:?set converted Stage-1 TTT HF weights}
    ;;
esac

test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test -x "$PYTHON_BIN"
test -s "$data_path"
test -s "$BASE_MODEL/config.json"

staging="${OUTPUT_ROOT}.partial"
mkdir -p "$staging"
status_file="$staging/status.txt"
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
printf 'RUNNING_PREFLIGHT\nstarted_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_file"

config_no_ttt="$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_public_${stage}_no_ttt.yaml"
config_ttt="$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_public_${stage}_ttt.yaml"
test -s "$config_no_ttt"; test -s "$config_ttt"

"$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_level_b_configs.py" \
  --config-root "$TOOL_REPO/configs/baseline_reproduction" \
  --corpus-root "$CORPUS_ROOT" --output "$staging/config_audit.json" \
  > "$staging/config_audit.log"

nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader,nounits > "$staging/gpu_inventory_raw.csv"

"$PYTHON_BIN" - "$CUDA_VISIBLE_DEVICES" "$staging/gpu_inventory_raw.csv" \
  "$staging/gpu_inventory.json" <<'PY'
import csv, json, pathlib, subprocess, sys
selected = [int(value) for value in sys.argv[1].split(',')]
if len(selected) != 8 or len(set(selected)) != 8:
    raise SystemExit('GPU selection is not eight unique indices')
rows = {}
with open(sys.argv[2], newline='') as handle:
    for row in csv.reader(handle):
        index = int(row[0].strip())
        rows[index] = {
            'index': index, 'name': row[1].strip(), 'uuid': row[2].strip(),
            'memory_total_mib': int(row[3].strip()),
            'memory_used_mib': int(row[4].strip()),
            'utilization_gpu_percent': int(row[5].strip()),
        }
result = []
for index in selected:
    if index not in rows:
        raise SystemExit(f'GPU {index} is absent')
    item = rows[index]
    query = subprocess.run(
        ['nvidia-smi', '-i', str(index), '--query-compute-apps=pid',
         '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True)
    processes = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    item['compute_process_count'] = len(processes)
    if item['memory_total_mib'] < 80000 or item['memory_used_mib'] > 512 or processes:
        raise SystemExit(f'GPU gate failed: {item}')
    result.append(item)
pathlib.Path(sys.argv[3]).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
PY

disk_free_bytes=$(df -PB1 "$staging" | awk 'NR==2 {print $4}')
if [[ "$disk_free_bytes" -lt 429496729600 ]]; then
  echo "less than 400 GiB is free at the output filesystem" >&2
  exit 5
fi

"$PYTHON_BIN" - "$UPSTREAM_REPO" "$data_path" "$BASE_MODEL" \
  "$config_no_ttt" "$config_ttt" "$staging/config_audit.json" \
  "$staging/gpu_inventory.json" "$staging/run_metadata.json" "$stage" \
  "$seq_len" "$one_step_tokens" "$disk_free_bytes" <<'PY'
import hashlib, json, pathlib, subprocess, sys
(upstream, data, base_model, no_config, ttt_config, config_audit,
 gpu_inventory, output, stage, seq_len, tokens, disk_free) = sys.argv[1:]
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
metadata = {
    'protocol': 'in_place_ttt_level_b_preflight_v1',
    'stage': stage,
    'seq_len': int(seq_len),
    'one_step_tokens': int(tokens),
    'selected_gpus': [int(value) for value in __import__('os').environ['CUDA_VISIBLE_DEVICES'].split(',')],
    'gpu_inventory': json.loads(pathlib.Path(gpu_inventory).read_text()),
    'disk_free_bytes': int(float(disk_free)),
    'upstream_commit': subprocess.check_output(['git', '-C', upstream, 'rev-parse', 'HEAD'], text=True).strip(),
    'upstream_clean': not subprocess.check_output(['git', '-C', upstream, 'status', '--short'], text=True).strip(),
    'data_path': data,
    'data_sha256': sha(data),
    'base_model_config_sha256': sha(pathlib.Path(base_model) / 'config.json'),
    'config_sha256': {'no_ttt': sha(no_config), 'ttt': sha(ttt_config)},
    'config_audit_sha256': sha(config_audit),
}
pathlib.Path(output).write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
PY

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$UPSTREAM_REPO${PYTHONPATH:+:$PYTHONPATH}"
export NPROC_PER_NODE=8
export TOKENIZERS_PARALLELISM=false

run_arm() {
  arm=$1
  config=$2
  model=$3
  arm_root="$staging/$arm"
  mkdir "$arm_root"
  "$PYTHON_BIN" - "$arm" "$stage" "$model" "$data_path" "$config" \
    "$arm_root/metadata.json" <<'PY'
import hashlib, json, pathlib, sys
arm, stage, model, data, config, output = sys.argv[1:]
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
pathlib.Path(output).write_text(json.dumps({
    'arm': arm, 'stage': stage, 'model_path': model,
    'model_config_sha256': sha(pathlib.Path(model) / 'config.json'),
    'data_sha256': sha(data), 'config_sha256': sha(config),
}, indent=2, sort_keys=True) + '\n')
PY
  printf 'timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu\n' > "$arm_root/gpu.csv"
  (
    while [[ ! -e "$arm_root/monitor.stop" ]]; do
      timestamp=$(date +%s.%N)
      nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits | while IFS=',' read -r index used total util; do
          printf '%s,%s,%s,%s,%s\n' "$timestamp" "${index// /}" "${used// /}" \
            "${total// /}" "${util// /}" >> "$arm_root/gpu.csv"
        done
      sleep 0.5
    done
  ) &
  monitor_pid=$!
  set +e
  (
    cd "$UPSTREAM_REPO"
    bash train.sh tasks/train_torch.py "$config" \
      --model.model_path "$model" \
      --data.train_path "$data_path" --data.train_size "$one_step_tokens" \
      --train.output_dir "$arm_root/output" --train.max_steps 1 \
      --train.num_train_epochs 1 --train.save_steps 0 --train.save_epochs 0 \
      --train.save_hf_weights false --train.use_wandb false
  ) > "$arm_root/train.log" 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$arm_root/exit_code.txt"
  touch "$arm_root/monitor.stop"
  wait "$monitor_pid" 2>/dev/null || true
  if [[ $rc -ne 0 ]]; then
    return "$rc"
  fi
}

run_arm no_ttt "$config_no_ttt" "$no_ttt_model"
run_arm ttt "$config_ttt" "$ttt_model"

"$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_level_b_preflight.py" \
  --root "$staging" --stage "$stage" --output "$staging/independent_audit.json" \
  > "$staging/independent_audit.log"
