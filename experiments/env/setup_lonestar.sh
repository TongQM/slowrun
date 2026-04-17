#!/bin/bash
# One-time environment setup on Lonestar6 (TACC).
# Creates a venv at $REPO_ROOT/.venv and installs all deps, including
# flash-attn (FA2) so sliding-window attention stays enabled on A100.
#
# Run on a compute node, not the login node (flash-attn build is heavy):
#   idev -p gpu-a100-small -A dms26007 -t 1:00:00
#   bash experiments/env/setup_lonestar.sh
#
# Safe to re-run; uses venv + pip, no conda.

set -euo pipefail

REPO_ROOT=/work/11426/yzfx0416/ls6/slowrun
cd "$REPO_ROOT"

module load cuda/12.8
module load python/3.12.11

if [ ! -d .venv ]; then
    echo "Creating venv at $REPO_ROOT/.venv ..."
    python -m venv .venv
fi
source .venv/bin/activate

python -m pip install --upgrade pip wheel

# Core deps (torch==2.10 per requirements.txt pins the rest)
pip install -r requirements.txt

# FA2 for A100 (sm_80). Without this the code silently falls back to SDPA
# and disables sliding-window attention -> worse perf + more memory.
# The wheel build takes ~10 min; pin a known-good version.
pip install packaging ninja
pip install flash-attn --no-build-isolation

python - <<'PY'
import torch
print(f"torch     : {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device    : {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability()}")
try:
    import flash_attn
    print(f"flash-attn: {flash_attn.__version__} (FA2 available)")
except Exception as e:
    print(f"flash-attn: NOT available ({e}) -> will fall back to SDPA (slower)")
PY

echo
echo "Setup done. Next:"
echo "  echo YOUR_WANDB_KEY > ~/.wandb_key && chmod 600 ~/.wandb_key"
echo "  python prepare_data.py        # tokenize 100M FineWeb tokens into fineweb_data/"
echo "  bash experiments/parallel/launch.sh"
