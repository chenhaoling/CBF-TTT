#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO=${UPSTREAM_REPO:?set the clean official checkout}
PYTHON_BIN=${PYTHON_BIN:?set the baseline Python}
OUTPUT_ROOT=${OUTPUT_ROOT:?set the completed baseline resource root}
UPSTREAM_COMMIT=be2324829b0e91c8fd10a74d4b43714fde6676e1
CONVERTER="$UPSTREAM_REPO/scripts/merge_dcp_to_hf.py"

test "$(git -C "$UPSTREAM_REPO" rev-parse HEAD)" = "$UPSTREAM_COMMIT"
test -z "$(git -C "$UPSTREAM_REPO" status --short)"
test -s "$CONVERTER"
test "$(cat "$OUTPUT_ROOT/state.txt")" = completed_both_arms

printf '%s\n' running > "$OUTPUT_ROOT/conversion_state.txt"
date --iso-8601=seconds > "$OUTPUT_ROOT/conversion_started_at.txt"
sha256sum "$0" "$CONVERTER" > "$OUTPUT_ROOT/conversion_code.sha256"

validate_summary() {
  local summary=$1
  "$PYTHON_BIN" - "$summary" <<'PY'
import json
import math
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
if payload.get("exit_code") != 0 or not payload.get("completed"):
    raise SystemExit("resource arm did not complete")
for key in ("last_loss", "last_gradient_norm"):
    value = payload.get(key)
    if value is None or not math.isfinite(float(value)):
        raise SystemExit(f"missing or non-finite {key}")
PY
}

validate_hf() {
  local output=$1
  local arm=$2
  "$PYTHON_BIN" - "$output" "$arm" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
arm = sys.argv[2]
config = json.loads((root / "config.json").read_text())
index_path = root / "model.safetensors.index.json"
if not index_path.is_file():
    raise SystemExit("missing safetensors index")
index = json.loads(index_path.read_text())
weight_map = index.get("weight_map", {})
if not weight_map:
    raise SystemExit("empty weight map")
missing = sorted({name for name in weight_map.values() if not (root / name).is_file()})
if missing:
    raise SystemExit(f"missing shards: {missing}")
ttt_keys = [key for key in weight_map if ".ttt_proj." in key or ".ttt_conv." in key]
if arm == "ttt" and not ttt_keys:
    raise SystemExit("TTT checkpoint has no native writer weights")
if arm == "no_ttt" and ttt_keys:
    raise SystemExit("no-TTT checkpoint unexpectedly has native writer weights")
foundation = config.get("foundation", config)
expected_mode = arm == "ttt"
if bool(foundation.get("ttt_mode", False)) != expected_mode:
    raise SystemExit("config ttt_mode does not match arm")
payload = {
    "arm": arm,
    "tensor_count": len(weight_map),
    "ttt_tensor_count": len(ttt_keys),
    "total_size": index.get("metadata", {}).get("total_size"),
    "shards": sorted(set(weight_map.values())),
    "ttt_mode": foundation.get("ttt_mode"),
}
(root / "conversion_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
}

for arm in no_ttt ttt; do
  arm_root="$OUTPUT_ROOT/resource_8k_${arm}"
  summary="$arm_root/summary.json"
  load_dir="$arm_root/checkpoint/checkpoints/global_step_1"
  assets_dir="$arm_root/checkpoint/model_assets"
  save_dir="$arm_root/hf_ckpt"
  validate_summary "$summary"
  test -d "$load_dir"
  test -s "$assets_dir/config.json"
  if [[ -e "$save_dir" ]]; then
    echo "refusing to overwrite $save_dir" >&2
    exit 3
  fi
  date --iso-8601=seconds > "$arm_root/conversion_started_at.txt"
  set +e
  /usr/bin/time -v "$PYTHON_BIN" "$CONVERTER" \
    --load-dir "$load_dir" \
    --save-dir "$save_dir" \
    --model-assets-dir "$assets_dir" \
    --shard-size 2000000000 \
    > "$arm_root/conversion.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$arm_root/conversion_exit_code.txt"
  if [[ "$status" -ne 0 ]]; then
    printf 'failed_%s\n' "$arm" > "$OUTPUT_ROOT/conversion_state.txt"
    exit "$status"
  fi
  validate_hf "$save_dir" "$arm"
  find "$save_dir" -maxdepth 1 -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$arm_root/hf_files.sha256"
  date --iso-8601=seconds > "$arm_root/conversion_completed_at.txt"
done

date --iso-8601=seconds > "$OUTPUT_ROOT/conversion_completed_at.txt"
printf '%s\n' completed_both_arms > "$OUTPUT_ROOT/conversion_state.txt"
