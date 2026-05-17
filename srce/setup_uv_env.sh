#!/bin/bash
set -euo pipefail

# Run this once from the project directory on Supek, preferably in an
# interactive gpu-test session or on a login node where package downloads work.
#
#   bash srce/setup_uv_env.sh
#
# It creates .venv with CUDA-enabled PyTorch plus the project dependencies.

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not on PATH."
  echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "Then start a new shell or source ~/.bashrc and rerun this script."
  exit 1
fi

uv venv --python 3.11 .venv

# Install PyTorch separately from the CUDA wheel index. A100 nodes should work
# with CUDA 12.x wheels as long as the driver is recent enough.
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch

uv pip install --python .venv/bin/python -r srce/requirements.txt

.venv/bin/python - <<'PY'
import torch
print("Python/Torch environment ready")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
