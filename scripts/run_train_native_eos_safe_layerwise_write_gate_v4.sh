#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
LABEL_ROOT=${LABEL_ROOT:?set LABEL_ROOT to independently audited V4 train/development labels}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT to a new controller directory}

if [[ -e "$OUTPUT_ROOT" ]]; then echo "refusing to overwrite $OUTPUT_ROOT" >&2; exit 2; fi
"$PYTHON_BIN" -m tasks.train_native_eos_safe_layerwise_write_gate_v4 \
  --label-root "$LABEL_ROOT" --output-root "$OUTPUT_ROOT"
"$PYTHON_BIN" -m scripts.audit_train_native_eos_safe_layerwise_write_gate_v4 \
  --label-root "$LABEL_ROOT" --controller-root "$OUTPUT_ROOT"
