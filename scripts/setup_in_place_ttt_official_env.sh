#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT=${CONDA_ROOT:-/home/ctj/miniconda3}
ENV_NAME=${ENV_NAME:-in_place_ttt_official_py311}
ENV_DIR=$CONDA_ROOT/envs/$ENV_NAME
PYTHON_BIN=$ENV_DIR/bin/python
VEOMNI_COMMIT=9b91e164bea9e17f17ed490aab5e076c2335ca25
FLASH_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

if [[ -e "$ENV_DIR" && ! -x "$PYTHON_BIN" ]]; then
  echo "$ENV_DIR exists but is not a usable conda environment" >&2; exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  "$CONDA_ROOT/bin/conda" create -y -n "$ENV_NAME" python=3.11 pip
fi

"$PYTHON_BIN" -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$PYTHON_BIN" -m pip install "$FLASH_WHEEL"
"$PYTHON_BIN" -m pip install \
  "veomni @ git+https://github.com/ByteDance-Seed/VeOmni.git@$VEOMNI_COMMIT"
"$PYTHON_BIN" -m pip install \
  liger-kernel torchdata blobfile 'datasets<=2.21.0' diffusers tiktoken timm \
  transformers==4.57.3 opt_einsum einops byted-wandb

"$PYTHON_BIN" - <<'PY'
import json
import pathlib
import sys
import torch
import transformers
import flash_attn
import veomni

direct = pathlib.Path(veomni.__file__).resolve().parents[1] / "veomni-0.1.0.dist-info/direct_url.json"
metadata = json.loads(direct.read_text())
expected = "9b91e164bea9e17f17ed490aab5e076c2335ca25"
assert torch.__version__ == "2.8.0+cu128", torch.__version__
assert transformers.__version__ == "4.57.3", transformers.__version__
assert metadata["vcs_info"]["commit_id"] == expected, metadata
print(json.dumps({
    "python": sys.version,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "flash_attn": flash_attn.__version__,
    "veomni": metadata,
}, indent=2))
PY

