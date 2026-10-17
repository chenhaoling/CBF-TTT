#!/usr/bin/env bash
set -Eeuo pipefail

context=${1:?usage: run_in_place_ttt_level_b_ruler_single.sh CONTEXT no_ttt|ttt PHYSICAL_GPU}
arm=${2:?usage: run_in_place_ttt_level_b_ruler_single.sh CONTEXT no_ttt|ttt PHYSICAL_GPU}
physical_gpu=${3:?usage: run_in_place_ttt_level_b_ruler_single.sh CONTEXT no_ttt|ttt PHYSICAL_GPU}
case "$context" in
  4096) suffix=4k ;;
  8192) suffix=8k ;;
  16384) suffix=16k ;;
  32768) suffix=32k ;;
  65536) suffix=64k ;;
  131072) suffix=128k ;;
  262144) suffix=256k ;;
  *) echo "unregistered context: $context" >&2; exit 2 ;;
esac
if [[ "$arm" != no_ttt && "$arm" != ttt ]]; then echo "invalid arm: $arm" >&2; exit 2; fi
if [[ ! "$physical_gpu" =~ ^[0-9]+$ ]]; then echo "physical GPU must be an integer" >&2; exit 2; fi

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
PYTHON_BIN=${PYTHON_BIN:?set the isolated OpenCompass Python}
MODEL=${MODEL:?set the audited arm-specific Stage-2 HF model}
TOKENIZER=${TOKENIZER:?set the original Qwen3-4B tokenizer directory}
TRAINING_ROOT=${TRAINING_ROOT:?set the completed formal Stage-2 training root}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new single-context output root}
PLAN=${PLAN:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_PUBLIC_CORPUS_PLAN.md}
EVALUATION_ADDENDUM=${EVALUATION_ADDENDUM:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_LEVEL_B_RULER_ADDENDUM.md}
CONFIG=$TOOL_REPO/eval_config/baseline_reproduction/qwen3_4b_formal_ruler.py
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1
TRAINING_AUDIT=$TRAINING_ROOT/independent_audit.json

if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.partial" ]]; then
  echo "refusing to overwrite output or partial path" >&2; exit 3
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test -z "$(git -C "$TOOL_REPO" status --short)"
test -x "$PYTHON_BIN"
for path in "$MODEL/config.json" "$MODEL/tokenizer.json" "$TOKENIZER/tokenizer.json" \
  "$TOKENIZER/tokenizer_config.json" "$TRAINING_AUDIT" "$PLAN" \
  "$EVALUATION_ADDENDUM" "$CONFIG"; do test -s "$path"; done

"$PYTHON_BIN" - "$TRAINING_AUDIT" "$MODEL" "$arm" "$TOKENIZER" <<'PY'
import hashlib, json, pathlib, sys
audit = json.load(open(sys.argv[1]))
model = pathlib.Path(sys.argv[2]).resolve()
arm = sys.argv[3]
tokenizer = pathlib.Path(sys.argv[4]).resolve()
if not audit.get("passed") or audit.get("stage") != "stage2" or audit.get("arm") != arm:
    raise SystemExit("formal Stage-2 training audit did not release this arm")
if pathlib.Path(audit.get("final_hf_path", "")).resolve() != model:
    raise SystemExit("MODEL differs from the audited Stage-2 final HF path")
config = json.loads((model / "config.json").read_text())
if bool(config.get("ttt_mode", False)) != (arm == "ttt"):
    raise SystemExit("model ttt_mode differs from arm")
if config.get("max_position_embeddings") != 131072:
    raise SystemExit("formal RULER requires a Stage-2 128K model")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if sha(model / "tokenizer.json") != sha(tokenizer / "tokenizer.json"):
    raise SystemExit("model tokenizer differs from the original evaluation tokenizer")
PY

gpu_row=$(nvidia-smi -i "$physical_gpu" \
  --query-gpu=index,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader,nounits)
"$PYTHON_BIN" - "$physical_gpu" "$gpu_row" <<'PY'
import csv, io, subprocess, sys
index = int(sys.argv[1])
row = next(csv.reader(io.StringIO(sys.argv[2])))
if int(row[0].strip()) != index or int(row[1].strip()) < 80000 or int(row[2].strip()) > 512:
    raise SystemExit("selected GPU does not satisfy the 80GB idle gate")
query = subprocess.run(
    ["nvidia-smi", "-i", str(index), "--query-compute-apps=pid",
     "--format=csv,noheader,nounits"],
    text=True, capture_output=True, check=True,
)
if query.stdout.strip():
    raise SystemExit("selected GPU already has a compute process")
PY

staging="${OUTPUT_ROOT}.partial"
mkdir -p "$staging/opencompass"
status_file=$staging/status.txt
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
printf 'RUNNING\nstarted_utc=%s\narm=%s\ncontext=%s\nphysical_gpu=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$arm" "$context" "$physical_gpu" > "$status_file"

model_name="qwen3_4b_level_b_${arm}_${suffix}"
"$PYTHON_BIN" - "$UPSTREAM_REPO" "$TOOL_REPO" "$MODEL" "$TOKENIZER" \
  "$TRAINING_AUDIT" "$PLAN" "$EVALUATION_ADDENDUM" "$CONFIG" \
  "$staging/run_metadata.json" "$arm" "$context" "$suffix" "$physical_gpu" \
  "$model_name" <<'PY'
import hashlib, importlib, importlib.metadata, inspect, json, pathlib, socket, subprocess, sys
from opencompass.utils import get_data_path
(upstream, tool, model_text, tokenizer_text, training_audit_text, plan_text,
 addendum_text, config_text, output_text, arm, context, suffix, gpu, model_name) = sys.argv[1:]
model = pathlib.Path(model_text).resolve()
tokenizer = pathlib.Path(tokenizer_text).resolve()
training_audit = pathlib.Path(training_audit_text).resolve()
plan = pathlib.Path(plan_text).resolve()
addendum = pathlib.Path(addendum_text).resolve()
config = pathlib.Path(config_text).resolve()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
tool_root = pathlib.Path(tool).resolve()
tool_paths = [
    config,
    tool_root / "scripts/summarize_in_place_ttt_level_b_ruler.py",
    tool_root / "scripts/audit_in_place_ttt_level_b_ruler.py",
]
ruler_modules = (
    "opencompass.configs.datasets.ruler.ruler_qa_gen",
    "opencompass.configs.datasets.ruler.ruler_niah_gen",
    "opencompass.configs.datasets.ruler.ruler_vt_gen",
    "opencompass.configs.datasets.ruler.ruler_fwe_gen",
    "opencompass.configs.datasets.ruler.ruler_cwe_gen",
    "opencompass.datasets.ruler.ruler_qa",
    "opencompass.datasets.ruler.ruler_niah",
    "opencompass.datasets.ruler.ruler_vt",
    "opencompass.datasets.ruler.ruler_fwe",
    "opencompass.datasets.ruler.ruler_cwe",
)
generator_files = {}
for module_name in ruler_modules:
    module_path = pathlib.Path(inspect.getfile(importlib.import_module(module_name))).resolve()
    generator_files[module_name] = {"path": str(module_path), "sha256": sha(module_path)}
ruler_source_specs = (
    "data/ruler/dev-v2.0.json",
    "data/ruler/hotpotqa.json",
    "data/ruler/PaulGrahamEssays.jsonl",
)
ruler_source_paths = {
    spec: pathlib.Path(get_data_path("./" + spec, local_mode=True)).resolve()
    for spec in ruler_source_specs
}
if not all(path.is_file() and path.stat().st_size > 0 for path in ruler_source_paths.values()):
    raise SystemExit("resolved RULER source data are missing")
model_files = {}
for path in sorted(item for item in model.rglob("*") if item.is_file()):
    relative = str(path.relative_to(model))
    model_files[relative] = {"size_bytes": path.stat().st_size, "sha256": sha(path)}
metadata = {
    "protocol": "in_place_ttt_level_b_ruler_run_v1",
    "hostname": socket.gethostname(),
    "arm": arm,
    "context_length": int(context),
    "context_suffix": suffix,
    "extrapolation": int(context) == 262144,
    "num_samples_per_task": 100,
    "ruler_random_seed": 42,
    "physical_gpu": int(gpu),
    "model_name": model_name,
    "model_path": str(model),
    "model_config_sha256": sha(model / "config.json"),
    "model_files": model_files,
    "tokenizer_path": str(tokenizer),
    "tokenizer_policy": "qwen_original_tokenizer_mistral_fix_false_v1",
    "fix_mistral_regex": False,
    "tokenizer_json_sha256": sha(tokenizer / "tokenizer.json"),
    "tokenizer_config_sha256": sha(tokenizer / "tokenizer_config.json"),
    "training_audit_path": str(training_audit),
    "training_audit_sha256": sha(training_audit),
    "plan_sha256": sha(plan),
    "evaluation_addendum_sha256": sha(addendum),
    "config_sha256": sha(config),
    "tool_artifacts": {
        str(path.relative_to(tool_root)): sha(path) for path in tool_paths
    },
    "dependency_versions": {
        package: importlib.metadata.version(package)
        for package in ("opencompass", "transformers", "datasets", "torch")
    },
    "ruler_generator_files": generator_files,
    "ruler_source_files": {
        spec: {"path": str(path), "sha256": sha(path)}
        for spec, path in ruler_source_paths.items()
    },
    "upstream_commit": subprocess.check_output(
        ["git", "-C", upstream, "rev-parse", "HEAD"], text=True).strip(),
    "upstream_clean": not subprocess.check_output(
        ["git", "-C", upstream, "status", "--short"], text=True).strip(),
    "tool_commit": subprocess.check_output(
        ["git", "-C", tool, "rev-parse", "HEAD"], text=True).strip(),
    "tool_clean": not subprocess.check_output(
        ["git", "-C", tool, "status", "--short"], text=True).strip(),
}
pathlib.Path(output_text).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
PY

printf 'timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu\n' > "$staging/gpu.csv"
(
  while [[ ! -e "$staging/monitor.stop" ]]; do
    timestamp=$(date +%s.%N)
    nvidia-smi -i "$physical_gpu" \
      --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits | while IFS=',' read -r index used total util; do
        printf '%s,%s,%s,%s,%s\n' "$timestamp" "${index// /}" "${used// /}" \
          "${total// /}" "${util// /}" >> "$staging/gpu.csv"
      done
    sleep 1
  done
) &
monitor_pid=$!

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$UPSTREAM_REPO:$TOOL_REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export BASELINE_EVAL_MODEL="$MODEL"
export BASELINE_EVAL_TOKENIZER="$TOKENIZER"
export BASELINE_EVAL_NAME="$model_name"
export BASELINE_EVAL_CONTEXT="$context"
export BASELINE_EVAL_NUM_SAMPLES=100
export TOKENIZERS_PARALLELISM=false
set +e
(
  cd "$UPSTREAM_REPO"
  /usr/bin/time -v "$PYTHON_BIN" "$TOOL_REPO/tasks/run_opencompass.py" \
    "$CONFIG" --work-dir "$staging/opencompass" --debug
) > "$staging/eval.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$staging/exit_code.txt"
stop_monitor
if [[ $rc -ne 0 ]]; then exit "$rc"; fi

"$PYTHON_BIN" "$TOOL_REPO/scripts/summarize_in_place_ttt_level_b_ruler.py" \
  --root "$staging" --output "$staging/summary.json" > "$staging/summary.log"
"$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_level_b_ruler.py" \
  --root "$staging" --plan "$PLAN" --evaluation-addendum "$EVALUATION_ADDENDUM" \
  --output "$staging/independent_audit.json" > "$staging/independent_audit.log"
