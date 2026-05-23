#!/bin/bash
# Load a Hugging Face token from a local, git-ignored file when present.
#
# Preferred usage on SRCE:
#   printf '%s\n' 'hf_your_token_here' > .hf_token
#   chmod 600 .hf_token
#
# You can also set HF_TOKEN directly in the shell or via qsub -v.

HF_TOKEN_FILE="${HF_TOKEN_FILE:-$PWD/.hf_token}"

if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_TOKEN_FILE" ]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
  export HF_TOKEN
fi

if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
fi
