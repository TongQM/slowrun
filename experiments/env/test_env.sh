#!/bin/bash
#SBATCH -J env_test
#SBATCH -p gpu-a100-dev
#SBATCH -A DMS26010
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH -o experiments/logs/env_test_%j.out
#SBATCH -e experiments/logs/env_test_%j.err
#
# Quick environment validation on a compute node. Checks that all deps
# load cleanly, CUDA is visible, and a 1-step training forward pass works.
#
# Usage:
#   cd /work/11426/yzfx0416/ls6/slowrun
#   sbatch experiments/env/test_env.sh
#   # then check: cat experiments/logs/env_test_<JOBID>.out

set -euo pipefail

REPO_ROOT=/work/11426/yzfx0416/ls6/slowrun
cd "$REPO_ROOT"

# --- Same environment block as train_array.sh ---
module load cuda/12.8
export LD_LIBRARY_PATH=/opt/apps/python/3.12.11/lib:${LD_LIBRARY_PATH:-}
source "$REPO_ROOT/.venv/bin/activate"

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT=slowrun_lonestar
mkdir -p experiments/logs

PASS=0
FAIL=0

check() {
    local desc="$1"
    shift
    if "$@" > /dev/null 2>&1; then
        echo "  PASS  $desc"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  $desc"
        FAIL=$((FAIL + 1))
    fi
}

echo "============================================================"
echo "Environment test on $(hostname) at $(date)"
echo "============================================================"

echo
echo "--- Python & shared libs ---"
check "python binary"       which python
check "libpython3.12.so"    python -c "import ctypes; ctypes.CDLL('libpython3.12.so.1.0')"

echo
echo "--- Core packages ---"
check "torch"               python -c "import torch; assert torch.__version__"
check "numpy (single load)" python -c "import numpy; print(numpy.__version__)"
check "tiktoken"            python -c "import tiktoken"
check "wandb"               python -c "import wandb"
check "datasets"            python -c "import datasets"
check "tqdm"                python -c "import tqdm"
check "kernels"             python -c "import kernels"

echo
echo "--- CUDA ---"
check "torch.cuda available" python -c "import torch; assert torch.cuda.is_available()"
python -c "
import torch
print(f'  device: {torch.cuda.get_device_name(0)}')
print(f'  capability: {torch.cuda.get_device_capability()}')
print(f'  memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
" 2>&1

echo
echo "--- Flash Attention ---"
python -c "
try:
    from flash_attn import flash_attn_func
    print('  PASS  flash-attn (FA2) available')
except ImportError:
    print('  WARN  flash-attn not installed — SDPA fallback (sliding window disabled)')
" 2>&1

echo
echo "--- Wandb auth ---"
python -c "
import os, wandb
key = os.environ.get('WANDB_API_KEY', '')
if key:
    print('  PASS  WANDB_API_KEY set')
    try:
        wandb.login(key=key, verify=True, relogin=True)
        print('  PASS  wandb login verified')
    except Exception as e:
        print(f'  FAIL  wandb login failed: {e}')
else:
    print('  FAIL  WANDB_API_KEY not set (create ~/.wandb_key)')
" 2>&1

echo
echo "--- Data ---"
check "fineweb_train.pt exists" test -f fineweb_data/fineweb_train.pt
check "fineweb_val.pt exists"   test -f fineweb_data/fineweb_val.pt

echo
echo "--- Training smoke (1 step, no wandb) ---"
WANDB_MODE=disabled torchrun --standalone --nproc_per_node=1 \
    -- unlimited/train.py \
    --n_layer=2 --n_head=2 --n_embd=64 \
    --num-models=1 --ensemble-type=init --num-epochs=1 \
    --optimizer=hybrid --ensemble-mode=logit \
    --data-fraction=0.005 --val-every-n-steps=999 \
    --num-epochs-model-0=1 --compile-mode=eager --no-compile \
    --completep \
    --run=env_test 2>&1 | tail -10

if [ $? -eq 0 ]; then
    echo "  PASS  training smoke test"
    PASS=$((PASS + 1))
else
    echo "  FAIL  training smoke test"
    FAIL=$((FAIL + 1))
fi

echo
echo "============================================================"
echo "Results: $PASS passed, $FAIL failed"
if [ $FAIL -gt 0 ]; then
    echo "FIX FAILURES BEFORE LAUNCHING EXPERIMENTS"
    exit 1
else
    echo "ALL CHECKS PASSED — ready to launch"
fi
echo "============================================================"
