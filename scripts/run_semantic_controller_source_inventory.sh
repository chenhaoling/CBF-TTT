#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?usage: $0 OUTPUT_ROOT [COUNTERFACT] [MQUAKE] [ZSRE_CALIBRATION]}"
COUNTERFACT="${2:-/home/ctj/data/cbf_ttt_eval/counterfact/counterfact.json}"
MQUAKE="${3:-/home/ctj/data/mquake_official/MQuAKE-T.json}"
ZSRE_CALIBRATION="${4:-/home/ctj/cbf_ttt_zsre_contextual_readout_v1/data/calibration.jsonl}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m tasks.build_semantic_controller_source_inventory \
  --counterfact "$COUNTERFACT" \
  --mquake "$MQUAKE" \
  --zsre-calibration "$ZSRE_CALIBRATION" \
  --output "$ROOT/inventory"

"$PYTHON_BIN" -m scripts.audit_semantic_controller_source_inventory \
  --counterfact "$COUNTERFACT" \
  --mquake "$MQUAKE" \
  --zsre-calibration "$ZSRE_CALIBRATION" \
  --inventory "$ROOT/inventory" \
  --output "$ROOT/audit.json"
