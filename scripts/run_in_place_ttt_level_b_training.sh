#!/usr/bin/env bash
set -Eeuo pipefail

stage=${1:?usage: run_in_place_ttt_level_b_training.sh stage1|stage2 no_ttt|ttt}
arm=${2:?usage: run_in_place_ttt_level_b_training.sh stage1|stage2 no_ttt|ttt}
if [[ "$stage" != stage1 && "$stage" != stage2 ]]; then echo "invalid stage" >&2; exit 2; fi
if [[ "$arm" != no_ttt && "$arm" != ttt ]]; then echo "invalid arm" >&2; exit 2; fi

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
PYTHON_BIN=${PYTHON_BIN:?set the official training Python}
CORPUS_ROOT=${CORPUS_ROOT:?set the independently audited full corpus}
START_MODEL=${START_MODEL:?set Qwen3-4B or the arm-specific converted Stage-1 HF path}
PREFLIGHT_ROOT=${PREFLIGHT_ROOT:?set the completed matched stage preflight}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new formal training output root}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?select exactly eight GPUs explicitly}
TRAINING_ADDENDUM=${TRAINING_ADDENDUM:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_LEVEL_B_TRAINING_ADDENDUM.md}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1
export CUDA_VISIBLE_DEVICES

IFS=',' read -r -a selected_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#selected_gpus[@]} -ne 8 ]]; then echo "exactly eight GPUs are required" >&2; exit 3; fi
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.partial" ]]; then
  echo "refusing to overwrite output or partial path" >&2; exit 4
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test -z "$(git -C "$TOOL_REPO" status --short)"
test -x "$PYTHON_BIN"; test -s "$START_MODEL/config.json"; test -s "$TRAINING_ADDENDUM"
test -s "$PREFLIGHT_ROOT/independent_audit.json"; test -s "$PREFLIGHT_ROOT/run_metadata.json"
"$PYTHON_BIN" - "$PREFLIGHT_ROOT/independent_audit.json" \
  "$PREFLIGHT_ROOT/run_metadata.json" "$stage" "$arm" "$CUDA_VISIBLE_DEVICES" <<'PY'
import json, socket, sys
audit = json.load(open(sys.argv[1]))
metadata = json.load(open(sys.argv[2]))
stage, arm = sys.argv[3:5]
gpus = [int(value) for value in sys.argv[5].split(',')]
if (not audit.get('passed') or audit.get('stage') != stage
        or not audit.get('arms', {}).get(arm, {}).get('passed')):
    raise SystemExit('matched preflight did not release this stage/arm')
if (metadata.get('hostname') != socket.gethostname()
        or metadata.get('stage') != stage or metadata.get('selected_gpus') != gpus):
    raise SystemExit('preflight host/stage/GPU binding differs from this run')
PY

case "$stage" in
  stage1)
    seq_len=32768; train_size=20000538624; expected_steps=9537
    data_path="$CORPUS_ROOT/stage1_32k.jsonl"
    if [[ "$arm" == no_ttt ]]; then min_disk_bytes=3298534883328; else min_disk_bytes=1374389534720; fi
    ;;
  stage2)
    seq_len=131072; train_size=14998831104; expected_steps=1788
    data_path="$CORPUS_ROOT/stage2_128k.jsonl"
    if [[ "$arm" == no_ttt ]]; then min_disk_bytes=879609302221; else min_disk_bytes=439804651110; fi
    ;;
esac
config="$TOOL_REPO/configs/baseline_reproduction/qwen3_4b_public_${stage}_${arm}.yaml"
test -s "$config"; test -s "$data_path"

staging="${OUTPUT_ROOT}.partial"
mkdir -p "$staging"
status_file="$staging/status.txt"
monitor_pid=""
stop_monitor() {
  if [[ -n "$monitor_pid" ]]; then
    touch "$staging/monitor.stop"
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=""
  fi
}
finish() {
  rc=$?
  trap - EXIT
  stop_monitor
  if [[ $rc -eq 0 ]]; then
    printf 'COMPLETED\nfinished_utc=%s\nexit_code=0\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$status_file"
    mv "$staging" "$OUTPUT_ROOT"
  else
    printf 'FAILED\nfinished_utc=%s\nexit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" > "$status_file"
  fi
  exit "$rc"
}
trap finish EXIT
printf 'RUNNING\nstarted_utc=%s\nstage=%s\narm=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$stage" "$arm" > "$status_file"

disk_free_bytes=$(df -PB1 "$staging" | awk 'NR==2 {print $4}')
if [[ "$disk_free_bytes" -lt "$min_disk_bytes" ]]; then
  echo "insufficient free disk: $disk_free_bytes < $min_disk_bytes" >&2; exit 5
fi

"$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_level_b_configs.py" \
  --config-root "$TOOL_REPO/configs/baseline_reproduction" \
  --corpus-root "$CORPUS_ROOT" --output "$staging/config_audit.json" \
  > "$staging/config_audit.log"

nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader,nounits > "$staging/gpu_before.csv"
"$PYTHON_BIN" - "$CUDA_VISIBLE_DEVICES" "$staging/gpu_before.csv" <<'PY'
import csv, subprocess, sys
selected = [int(value) for value in sys.argv[1].split(',')]
rows = {int(row[0].strip()): row for row in csv.reader(open(sys.argv[2]))}
if len(selected) != 8 or len(set(selected)) != 8:
    raise SystemExit('invalid GPU selection')
for index in selected:
    row = rows.get(index)
    if row is None or int(row[3].strip()) < 80000 or int(row[4].strip()) > 512:
        raise SystemExit(f'GPU capacity/idle gate failed for {index}')
    query = subprocess.run(['nvidia-smi', '-i', str(index), '--query-compute-apps=pid',
                           '--format=csv,noheader,nounits'], text=True,
                           capture_output=True, check=True)
    if query.stdout.strip():
        raise SystemExit(f'GPU {index} has a compute process')
PY

"$PYTHON_BIN" - "$UPSTREAM_REPO" "$TOOL_REPO" "$CORPUS_ROOT" "$data_path" \
  "$START_MODEL" "$config" "$PREFLIGHT_ROOT/independent_audit.json" \
  "$PREFLIGHT_ROOT/run_metadata.json" \
  "$staging/config_audit.json" "$TRAINING_ADDENDUM" "$staging/run_metadata.json" \
  "$stage" "$arm" "$seq_len" "$train_size" "$expected_steps" \
  "$CUDA_VISIBLE_DEVICES" "$disk_free_bytes" <<'PY'
import hashlib, json, pathlib, socket, subprocess, sys
(upstream, tool, corpus, data, model, config, preflight, preflight_metadata, config_audit,
 addendum, output, stage, arm, seq_len, train_size, steps, gpus, disk_free) = sys.argv[1:]
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
corpus_manifest = pathlib.Path(corpus) / 'manifest.json'
corpus_audit = pathlib.Path(corpus) / 'independent_audit.json'
metadata = {
    'protocol': 'in_place_ttt_level_b_training_v1', 'hostname': socket.gethostname(),
    'stage': stage, 'arm': arm, 'seq_len': int(seq_len),
    'train_size': int(train_size), 'expected_steps': int(steps),
    'selected_gpus': [int(value) for value in gpus.split(',')],
    'disk_free_bytes_before': int(disk_free),
    'upstream_commit': subprocess.check_output(['git','-C',upstream,'rev-parse','HEAD'], text=True).strip(),
    'upstream_clean': not subprocess.check_output(['git','-C',upstream,'status','--short'], text=True).strip(),
    'tool_commit': subprocess.check_output(['git','-C',tool,'rev-parse','HEAD'], text=True).strip(),
    'tool_clean': not subprocess.check_output(['git','-C',tool,'status','--short'], text=True).strip(),
    'data_path': data, 'data_sha256': sha(data),
    'start_model': model, 'start_model_config_sha256': sha(pathlib.Path(model)/'config.json'),
    'config_path': config, 'config_sha256': sha(config),
    'preflight_audit_path': str(pathlib.Path(preflight).resolve()),
    'preflight_audit_sha256': sha(preflight),
    'preflight_run_metadata_path': str(pathlib.Path(preflight_metadata).resolve()),
    'preflight_run_metadata_sha256': sha(preflight_metadata),
    'config_audit_sha256': sha(config_audit),
    'training_addendum_sha256': sha(addendum),
    'corpus_manifest_sha256': sha(corpus_manifest),
    'corpus_audit_sha256': sha(corpus_audit),
}
pathlib.Path(output).write_text(json.dumps(metadata, indent=2, sort_keys=True)+'\n')
PY

printf 'timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu\n' > "$staging/gpu.csv"
printf 'timestamp,memory_used_bytes,memory_available_bytes,swap_used_bytes\n' > "$staging/host.csv"
(
  while [[ ! -e "$staging/monitor.stop" ]]; do
    timestamp=$(date +%s.%N)
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | while IFS=',' read -r index used total util; do
        printf '%s,%s,%s,%s,%s\n' "$timestamp" "${index// /}" "${used// /}" \
          "${total// /}" "${util// /}" >> "$staging/gpu.csv"
      done
    awk -v timestamp="$timestamp" \
      '/MemTotal/{t=$2}/MemAvailable/{a=$2}/SwapTotal/{s=$2}/SwapFree/{f=$2}END{print timestamp "," (t-a)*1024 "," a*1024 "," (s-f)*1024}' \
      /proc/meminfo >> "$staging/host.csv"
    sleep 1
  done
) &
monitor_pid=$!

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$UPSTREAM_REPO:$TOOL_REPO${PYTHONPATH:+:$PYTHONPATH}"
export NPROC_PER_NODE=8
export TOKENIZERS_PARALLELISM=false
set +e
(
  cd "$UPSTREAM_REPO"
  bash train.sh tasks/train_torch.py "$config" \
    --model.model_path "$START_MODEL" \
    --data.train_path "$data_path" \
    --train.output_dir "$staging/training_output"
) > "$staging/train.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$staging/exit_code.txt"
stop_monitor
if [[ $rc -ne 0 ]]; then exit "$rc"; fi

"$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_level_b_training.py" \
  --root "$staging" --stage "$stage" --arm "$arm" \
  --training-addendum "$TRAINING_ADDENDUM" \
  --output "$staging/independent_audit.json" > "$staging/independent_audit.log"
