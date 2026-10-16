#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
LABEL_ROOT=${LABEL_ROOT:?set LABEL_ROOT to independently audited V4 sealed-test labels}
CONTROLLER_ROOT=${CONTROLLER_ROOT:?set CONTROLLER_ROOT to independently audited released controllers}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT to a new sealed-test evaluation directory}

if [[ -e "$OUTPUT_ROOT" ]]; then echo "refusing to overwrite $OUTPUT_ROOT" >&2; exit 2; fi
"$PYTHON_BIN" -m tasks.evaluate_native_eos_safe_layerwise_write_gate_v4 \
  --label-root "$LABEL_ROOT" --controller-root "$CONTROLLER_ROOT" --output-root "$OUTPUT_ROOT"
"$PYTHON_BIN" -m scripts.audit_evaluate_native_eos_safe_layerwise_write_gate_v4 \
  --label-root "$LABEL_ROOT" --controller-root "$CONTROLLER_ROOT" --evaluation-root "$OUTPUT_ROOT"
