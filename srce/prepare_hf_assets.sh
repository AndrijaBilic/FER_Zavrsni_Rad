#!/bin/bash
set -euo pipefail

# Run this before submitting PBS jobs, from a node with internet access.
# It prepares:
# - outputs/df_webq_balanced.csv
# - Hugging Face cache entries for the selected model
#
# Usage:
#   MODEL_CHOICE=mistral bash srce/prepare_hf_assets.sh

cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  echo "Missing .venv. Run first: bash srce/setup_uv_env.sh"
  exit 2
fi

export MODEL_CHOICE="${MODEL_CHOICE:-mistral}"
export DRIVE_PATH="${DRIVE_PATH:-$PWD/outputs}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

mkdir -p "$DRIVE_PATH" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

.venv/bin/python srce/prepare_hf_assets.py
