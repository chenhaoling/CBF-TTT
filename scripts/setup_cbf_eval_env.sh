#!/usr/bin/env bash
# Isolate OpenCompass dependencies from the working VeOmni pretraining env.
set -euo pipefail

CONDA_ROOT=${CONDA_ROOT:-/home/ctj/miniconda3}
TRAIN_ENV=${TRAIN_ENV:-cbf_ttt_train_py311}
EVAL_ENV=${EVAL_ENV:-cbf_ttt_eval_opencompass}
EVAL_PYTHON="$CONDA_ROOT/envs/$EVAL_ENV/bin/python"

if [[ ! -x "$EVAL_PYTHON" ]]; then
  "$CONDA_ROOT/bin/conda" create -y -n "$EVAL_ENV" --clone "$TRAIN_ENV"
fi
"$EVAL_PYTHON" -m pip install 'opencompass==0.5.4' 'wonderwords==3.0.1'
"$EVAL_PYTHON" - <<'PY'
import opencompass
import mmengine
import wonderwords
print("OpenCompass evaluation environment ready", opencompass.__file__)
PY
