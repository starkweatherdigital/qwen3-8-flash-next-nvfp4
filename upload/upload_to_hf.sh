#!/bin/bash
# BCN-79 publish: upload the Flash-Next NVFP4 checkpoint to Hugging Face.
# Prereq: `hf auth login` with a WRITE token. Then run this script.
# Resumable: re-running continues where it left off.
set -euo pipefail
REPO="${1:-qwen3.8-flash-next-nvfp4}"            # repo name under your account
STAGE="${2:-$HOME/bcn79_hf_stage}"               # staged clean checkpoint dir
USER=$(hf auth whoami 2>/dev/null | head -1 | awk '{print $NF}')
[ -z "$USER" ] && { echo "not logged in — run: hf auth login"; exit 1; }
echo "uploading $STAGE -> $USER/$REPO (public)"
hf repo create "$REPO" --repo-type model 2>/dev/null || true
hf upload-large-folder "$USER/$REPO" "$STAGE" --repo-type model
echo "DONE: https://huggingface.co/$USER/$REPO"
