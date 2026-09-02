#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="lonestar6"
REMOTE_DIR="/work/11426/yzfx0416/ls6/slowrun/"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
LOCAL_DIR="${SCRIPT_DIR}/"

EXCLUDES=(
  --exclude ".git/"
  --exclude ".DS_Store"
  --exclude "__pycache__/"
  --exclude ".pytest_cache/"
  --exclude ".mypy_cache/"
  --exclude ".venv/"
  --exclude "venv/"
  --exclude "/slowrun/"
  --exclude "logs/"
  --exclude "wandb/"
  --exclude "*.pt"
  --exclude "*.pth"
  --exclude "*.ckpt"
  --exclude "*.pyc"
)

mkdir -p "$LOCAL_DIR"

echo "Pulling newer files from Lonestar -> local"
rsync -avzu "${EXCLUDES[@]}" \
  "${REMOTE_HOST}:${REMOTE_DIR}" \
  "${LOCAL_DIR}"

echo "Pushing newer files from local -> Lonestar"
rsync -avzu "${EXCLUDES[@]}" \
  "${LOCAL_DIR}" \
  "${REMOTE_HOST}:${REMOTE_DIR}"

echo "Sync complete."
