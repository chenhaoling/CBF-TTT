#!/usr/bin/env bash
set -Eeuo pipefail

context=${1:?usage: run_in_place_ttt_2x5090_ruler_single.sh CONTEXT no_ttt|ttt PHYSICAL_GPU}
arm=${2:?usage: run_in_place_ttt_2x5090_ruler_single.sh CONTEXT no_ttt|ttt PHYSICAL_GPU}
physical_gpu=${3:?usage: run_in_place_ttt_2x5090_ruler_single.sh CONTEXT no_ttt|ttt PHYSICAL_GPU}
RUN_KIND=${RULER_RUN_KIND:-formal}
NUM_SAMPLES=${RULER_NUM_SAMPLES:-100}
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
if [[ "$physical_gpu" != 0 && "$physical_gpu" != 1 ]]; then
  echo "physical GPU must be 0 or 1 on hku-gpu2" >&2; exit 2
fi
if [[ "$RUN_KIND" == formal && "$NUM_SAMPLES" != 100 ]]; then
  echo "formal RULER requires 100 samples per task" >&2; exit 2
fi
if [[ "$RUN_KIND" == resource && "$NUM_SAMPLES" != 1 ]]; then
  echo "resource RULER requires one sample per task" >&2; exit 2
fi
if [[ "$RUN_KIND" != formal && "$RUN_KIND" != resource ]]; then
  echo "RULER_RUN_KIND must be formal or resource" >&2; exit 2
fi

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
TOOL_REPO=${TOOL_REPO:?set the clean CBF-TTT checkout}
TOOL_COMMIT=${TOOL_COMMIT:?set the exact tool commit}
PYTHON_BIN=${PYTHON_BIN:?set the isolated OpenCompass Python}
MODEL=${MODEL:?set the audited arm-specific Stage-2 HF model}
TOKENIZER=${TOKENIZER:?set the original Qwen3-4B tokenizer directory}
TRAINING_ROOT=${TRAINING_ROOT:?set the completed paired 2x5090 formal Stage-2 root}
OUTPUT_ROOT=${OUTPUT_ROOT:?set a new single-context output root}
PUBLIC_CORPUS_PLAN=${PUBLIC_CORPUS_PLAN:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_PUBLIC_CORPUS_PLAN.md}
EVALUATION_PLAN=${EVALUATION_PLAN:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_2X5090_RULER_PLAN.md}
PAPER_INFERENCE_ADDENDUM=${PAPER_INFERENCE_ADDENDUM:-$TOOL_REPO/IN_PLACE_TTT_BASELINE_PAPER_INFERENCE_CLIP_ADDENDUM.md}
CONFIG=$TOOL_REPO/eval_config/baseline_reproduction/qwen3_4b_2x5090_ruler.py
OVERLAY_BUILDER=$TOOL_REPO/scripts/build_in_place_ttt_paper_inference_overlay.py
OVERLAY_VALIDATOR=$TOOL_REPO/scripts/validate_in_place_ttt_paper_inference_overlay.py
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1
TRAINING_AUDIT=$TRAINING_ROOT/independent_audit.json
EXPECTED_HOSTNAME=${EXPECTED_HOSTNAME:-user-Super-Server}

if [[ "$(hostname)" != "$EXPECTED_HOSTNAME" ]]; then
  echo "2x5090 RULER is restricted to hku-gpu2" >&2; exit 2
fi
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.partial" ]]; then
  echo "refusing to overwrite output or partial path" >&2; exit 3
fi
test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test "$(git -C "$TOOL_REPO" rev-parse HEAD)" = "$TOOL_COMMIT"
test -z "$(git -C "$TOOL_REPO" status --short)"
test -x "$PYTHON_BIN"
for path in "$MODEL/config.json" "$MODEL/tokenizer.json" "$TOKENIZER/tokenizer.json" \
  "$TOKENIZER/tokenizer_config.json" "$TRAINING_AUDIT" "$PUBLIC_CORPUS_PLAN" \
  "$EVALUATION_PLAN" "$PAPER_INFERENCE_ADDENDUM" "$CONFIG" \
  "$OVERLAY_BUILDER" "$OVERLAY_VALIDATOR"; do test -s "$path"; done

"$PYTHON_BIN" - "$TRAINING_AUDIT" "$MODEL" "$arm" "$TOKENIZER" <<'PY'
import hashlib, json, pathlib, sys
audit = json.load(open(sys.argv[1]))
model = pathlib.Path(sys.argv[2]).resolve()
arm = sys.argv[3]
tokenizer = pathlib.Path(sys.argv[4]).resolve()
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()
if (audit.get("protocol") != "in_place_ttt_2x5090_100m_stage2_audit_v1"
        or audit.get("passed") is not True):
    raise SystemExit("paired formal Stage-2 audit did not pass")
record = audit.get("arms", {}).get(arm, {})
expected = (pathlib.Path(sys.argv[1]).resolve().parent / arm
            / record.get("final_hf_relative_path", "")).resolve()
if expected != model:
    raise SystemExit("MODEL differs from the audited Stage-2 final HF path")
manifest = {
    str(path.relative_to(model)): {
        "sha256": sha(path),
        "size_bytes": path.stat().st_size,
    }
    for path in model.rglob("*") if path.is_file()
}
if manifest != record.get("hf_manifest"):
    raise SystemExit("MODEL file manifest differs from formal Stage-2 audit")
config = json.loads((model / "config.json").read_text())
if bool(config.get("ttt_mode", False)) != (arm == "ttt"):
    raise SystemExit("model ttt_mode differs from arm")
if config.get("max_position_embeddings") != 131072:
    raise SystemExit("formal RULER requires a Stage-2 128K model")
if sha(model / "tokenizer.json") != sha(tokenizer / "tokenizer.json"):
    raise SystemExit("model tokenizer differs from the original evaluation tokenizer")
PY

gpu_row=$(nvidia-smi -i "$physical_gpu" \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader,nounits)
"$PYTHON_BIN" - "$physical_gpu" "$gpu_row" <<'PY'
import csv, io, subprocess, sys
index = int(sys.argv[1])
row = next(csv.reader(io.StringIO(sys.argv[2])))
if (int(row[0].strip()) != index or "RTX 5090" not in row[1]
        or int(row[2].strip()) < 32000 or int(row[3].strip()) > 128):
    raise SystemExit("selected GPU does not satisfy the 32GB RTX5090 idle gate")
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
paper_overlay=$staging/paper_inference_overlay
printf '%s\n' "$gpu_row" > "$staging/gpu_before.csv"
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

CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "$OVERLAY_BUILDER" \
  --upstream-root "$UPSTREAM_REPO" --output "$paper_overlay" \
  --clip-norm 1e-5 > "$staging/paper_inference_overlay_build.log"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$paper_overlay" \
  "$PYTHON_BIN" "$OVERLAY_VALIDATOR" --overlay "$paper_overlay" \
  --output "$staging/paper_inference_overlay_validation.json" \
  > "$staging/paper_inference_overlay_validation.log"

cp /proc/meminfo "$staging/preflight_host.txt"
: > "$staging/preflight_conflicts.txt"
"$PYTHON_BIN" - "$staging/preflight_host.txt" \
  "$staging/preflight_conflicts.txt" <<'PY'
import pathlib, shlex, sys
host, output = map(pathlib.Path, sys.argv[1:])
available = None
for line in host.read_text().splitlines():
    if line.startswith("MemAvailable:"):
        available = int(line.split()[1])
if available is None or available < 32 * 1024 * 1024:
    raise SystemExit("host has less than 32 GiB available")
targets = {"build_in_place_ttt_public_corpus.py", "audit_in_place_ttt_public_corpus.py"}
conflicts = []
for process in pathlib.Path("/proc").glob("[0-9]*"):
    try:
        argv = [item.decode(errors="replace") for item in
                (process / "cmdline").read_bytes().split(b"\0") if item]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if any(pathlib.Path(argument).name in targets for argument in argv):
        conflicts.append((int(process.name), argv))
output.write_text("".join(
    f"{pid} {shlex.join(argv)}\n" for pid, argv in sorted(conflicts)))
if conflicts:
    raise SystemExit("public-corpus builder/auditor is active")
PY

model_name="qwen3_4b_2x5090_100m_${RUN_KIND}_${arm}_${suffix}"
"$PYTHON_BIN" - "$UPSTREAM_REPO" "$TOOL_REPO" "$MODEL" "$TOKENIZER" \
  "$TRAINING_AUDIT" "$PUBLIC_CORPUS_PLAN" "$EVALUATION_PLAN" "$CONFIG" \
  "$staging/run_metadata.json" "$arm" "$context" "$suffix" "$physical_gpu" \
  "$model_name" "$RUN_KIND" "$NUM_SAMPLES" "$PAPER_INFERENCE_ADDENDUM" \
  "$paper_overlay/paper_inference_overlay_manifest.json" \
  "$staging/paper_inference_overlay_validation.json" <<'PY'
import hashlib, importlib, importlib.metadata, inspect, json, pathlib, socket, subprocess, sys
from opencompass.utils import get_data_path
(upstream, tool, model_text, tokenizer_text, training_audit_text,
 public_plan_text, evaluation_plan_text, config_text, output_text, arm,
 context, suffix, gpu, model_name, run_kind, num_samples,
 paper_addendum_text, overlay_manifest_text,
 overlay_validation_text) = sys.argv[1:]
model = pathlib.Path(model_text).resolve()
tokenizer = pathlib.Path(tokenizer_text).resolve()
training_audit = pathlib.Path(training_audit_text).resolve()
public_plan = pathlib.Path(public_plan_text).resolve()
evaluation_plan = pathlib.Path(evaluation_plan_text).resolve()
paper_addendum = pathlib.Path(paper_addendum_text).resolve()
overlay_manifest_path = pathlib.Path(overlay_manifest_text).resolve()
overlay_validation_path = pathlib.Path(overlay_validation_text).resolve()
config = pathlib.Path(config_text).resolve()
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()
tool_root = pathlib.Path(tool).resolve()
tool_paths = [
    config,
    tool_root / "scripts/run_in_place_ttt_2x5090_ruler_single.sh",
    tool_root / "scripts/build_in_place_ttt_paper_inference_overlay.py",
    tool_root / "scripts/validate_in_place_ttt_paper_inference_overlay.py",
    tool_root / "scripts/audit_in_place_ttt_paper_inference_overlay.py",
]
if run_kind == "formal":
    tool_paths.extend([
        tool_root / "scripts/summarize_in_place_ttt_2x5090_ruler.py",
        tool_root / "scripts/audit_in_place_ttt_2x5090_ruler.py",
    ])
else:
    tool_paths.append(
        tool_root / "scripts/audit_in_place_ttt_2x5090_ruler_resource.py"
    )
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
overlay_manifest = json.loads(overlay_manifest_path.read_text())
overlay_validation = json.loads(overlay_validation_path.read_text())
if (overlay_manifest.get("protocol") != "in_place_ttt_paper_inference_overlay_v1"
        or overlay_manifest.get("clip_norm") != 1e-5
        or overlay_manifest.get("clip_norm_type") != "frobenius"
        or overlay_validation.get("protocol")
        != "in_place_ttt_paper_inference_overlay_validation_v1"
        or overlay_validation.get("passed") is not True
        or overlay_validation.get("cuda_initialized") is not False
        or overlay_validation.get("from_pretrained_override") is not True
        or overlay_validation.get("clip_norm") != 1e-5):
    raise SystemExit("paper inference overlay did not pass its CPU validation")
metadata = {
    "protocol": "in_place_ttt_2x5090_ruler_run_v1",
    "baseline_tier": "2x5090_100m",
    "run_kind": run_kind,
    "hostname": socket.gethostname(),
    "arm": arm,
    "context_length": int(context),
    "context_suffix": suffix,
    "extrapolation": int(context) == 262144,
    "num_samples_per_task": int(num_samples),
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
    "public_corpus_plan_sha256": sha(public_plan),
    "evaluation_plan_sha256": sha(evaluation_plan),
    "paper_inference_addendum_sha256": sha(paper_addendum),
    "paper_inference_semantics": "per_update_delta_frobenius_clip",
    "ttt_update_clip_norm": 1e-5,
    "paper_inference_overlay_manifest_relative_path":
        "paper_inference_overlay/paper_inference_overlay_manifest.json",
    "paper_inference_overlay_manifest_sha256": sha(overlay_manifest_path),
    "paper_inference_overlay_validation_relative_path":
        "paper_inference_overlay_validation.json",
    "paper_inference_overlay_validation_sha256": sha(overlay_validation_path),
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
export PYTHONPATH="$paper_overlay:$UPSTREAM_REPO:$TOOL_REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export BASELINE_EVAL_MODEL="$MODEL"
export BASELINE_EVAL_TOKENIZER="$TOKENIZER"
export BASELINE_EVAL_NAME="$model_name"
export BASELINE_EVAL_CONTEXT="$context"
export BASELINE_EVAL_NUM_SAMPLES="$NUM_SAMPLES"
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

if [[ "$RUN_KIND" == formal ]]; then
  "$PYTHON_BIN" "$TOOL_REPO/scripts/summarize_in_place_ttt_2x5090_ruler.py" \
    --root "$staging" --output "$staging/summary.json" > "$staging/summary.log"
  "$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_ruler.py" \
    --root "$staging" --public-corpus-plan "$PUBLIC_CORPUS_PLAN" \
    --evaluation-plan "$EVALUATION_PLAN" \
    --paper-inference-addendum "$PAPER_INFERENCE_ADDENDUM" \
    --output "$staging/independent_audit.json" > "$staging/independent_audit.log"
else
  "$PYTHON_BIN" "$TOOL_REPO/scripts/audit_in_place_ttt_2x5090_ruler_resource.py" \
    --root "$staging" --public-corpus-plan "$PUBLIC_CORPUS_PLAN" \
    --evaluation-plan "$EVALUATION_PLAN" \
    --paper-inference-addendum "$PAPER_INFERENCE_ADDENDUM" \
    --output "$staging/independent_resource_audit.json" \
    > "$staging/independent_resource_audit.log"
fi
