#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN=${CONDA_BIN:-/home/ctj/miniconda3/bin/conda}
SOURCE_ENV=${SOURCE_ENV:-/home/ctj/miniconda3/envs/in_place_ttt_official_py311}
ENV_DIR=${ENV_DIR:-/home/ctj/miniconda3/envs/in_place_ttt_official_eval}

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$CONDA_BIN" create --yes --clone "$SOURCE_ENV" --prefix "$ENV_DIR"
fi

PYTHON_BIN="$ENV_DIR/bin/python"
# OpenCompass 0.5.4 requires NumPy <2.  PyArrow 26 wheels require NumPy >=2,
# so pin the last working PyArrow line used by the independently verified
# OpenCompass environment.
"$PYTHON_BIN" -m pip install \
  opencompass==0.5.4 wonderwords==3.0.1 numpy==1.26.4 pyarrow==25.0.1

"$PYTHON_BIN" - <<'PY'
import json
import opencompass
import pyarrow
import torch
import transformers
import wonderwords
import flash_attn

print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "flash_attn": flash_attn.__version__,
    "opencompass": opencompass.__version__,
    "pyarrow": pyarrow.__version__,
    "wonderwords": getattr(wonderwords, "__version__", "installed"),
}, indent=2))
PY
