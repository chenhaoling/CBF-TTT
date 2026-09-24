#!/usr/bin/env bash
# Prepare a separate Python 3.11 environment for the baseline VeOmni trainer.
set -euo pipefail

CONDA_ROOT=${CONDA_ROOT:-/home/ctj/miniconda3}
ENV_NAME=${ENV_NAME:-cbf_ttt_train_py311}
PYTHON="$CONDA_ROOT/envs/$ENV_NAME/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  "$CONDA_ROOT/bin/conda" create -y -n "$ENV_NAME" python=3.11
fi

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu130
"$PYTHON" -m pip install \
  'veomni @ git+https://github.com/ByteDance-Seed/VeOmni.git@9b91e164bea9e17f17ed490aab5e076c2335ca25' \
  'transformers==4.57.3' liger-kernel opt_einsum einops tiktoken zarr

"$PYTHON" - <<'PY'
import torch
import transformers
import veomni
import liger_kernel
print("python training environment ready", torch.__version__, transformers.__version__, veomni.__file__)
print("CUDA devices:", torch.cuda.device_count())
PY
