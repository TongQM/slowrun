#!/bin/bash
#SBATCH --job-name=ckpt_cleanup
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260161p
#SBATCH --gpus=h100-80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/logs/%x_%A_%a.out
#SBATCH --error=experiments/logs/%x_%A_%a.err
#
# Per-(cell, strategy) cleanup: after replay finishes, prune transient ckpts.
# Keeps:
#   - model_*_step_{S}.pt where S % PERMANENT_EVERY_N_STEPS == 0
#   - model_*_epoch_{k}.pt where k % PERMANENT_EVERY_N_EPOCHS == 0
# Deletes everything else.
#
# Required env vars (set by launch_grid_v2.sh):
#   SHARED_TIMESTAMP  (per-cell timestamp, identifies the run dir)
#   PERMANENT_EVERY_N_STEPS  (stride for permanent step ckpts; e.g. 760)
#
# Optional:
#   PERMANENT_EVERY_N_EPOCHS  (stride for permanent per-epoch ckpts; default 5)
#   CHECKPOINT_BASE  (default "checkpoints")
#
# This is a 2-task array: index 0 = init_ens, 1 = init_shuffle_ens.

set -euo pipefail

: "${SHARED_TIMESTAMP:?SHARED_TIMESTAMP not set}"
: "${PERMANENT_EVERY_N_STEPS:?PERMANENT_EVERY_N_STEPS not set}"

PERMANENT_EVERY_N_EPOCHS="${PERMANENT_EVERY_N_EPOCHS:-5}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-checkpoints}"
cd /ocean/projects/cis260161p/ymiao6/scaling/slowrun

case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) STRATEGY_NAME="init_ens" ;;
    1) STRATEGY_NAME="init_shuffle_ens" ;;
    *) echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac

CKPT_DIR="${CHECKPOINT_BASE}/parallel_${STRATEGY_NAME}_${SHARED_TIMESTAMP}"
echo "============================================================"
echo "Cleanup: ${CKPT_DIR}"
echo "  Permanent stride: every ${PERMANENT_EVERY_N_STEPS} steps  (step ckpts)"
echo "  Permanent stride: every ${PERMANENT_EVERY_N_EPOCHS} epochs (epoch ckpts)"
echo "============================================================"

if [ ! -d "$CKPT_DIR" ]; then
    echo "WARNING: $CKPT_DIR missing — nothing to clean"
    exit 0
fi

del=0; del_bytes=0; keep=0; keep_bytes=0

# --- step ckpts ---
while IFS= read -r f; do
    base=$(basename "$f")
    S=$(echo "$base" | sed -E 's/^model_[0-9]+_step_([0-9]+)\.pt$/\1/')
    if ! [[ "$S" =~ ^[0-9]+$ ]]; then
        echo "  skip (regex mismatch): $base"; continue
    fi
    sz=$(stat -L -c %s "$f" 2>/dev/null || echo 0)
    if [ $((S % PERMANENT_EVERY_N_STEPS)) -eq 0 ]; then
        keep=$((keep+1)); keep_bytes=$((keep_bytes+sz))
    else
        rm -f "$f" && del=$((del+1)) && del_bytes=$((del_bytes+sz))
    fi
done < <(find -L "$CKPT_DIR" -maxdepth 1 -name 'model_*_step_*.pt' 2>/dev/null)

# --- per-epoch ckpts ---
ep_del=0; ep_del_bytes=0; ep_keep=0; ep_keep_bytes=0
while IFS= read -r f; do
    base=$(basename "$f")
    K=$(echo "$base" | sed -E 's/^model_[0-9]+_epoch_([0-9]+)\.pt$/\1/')
    if ! [[ "$K" =~ ^[0-9]+$ ]]; then
        echo "  skip (regex mismatch): $base"; continue
    fi
    sz=$(stat -L -c %s "$f" 2>/dev/null || echo 0)
    if [ $((K % PERMANENT_EVERY_N_EPOCHS)) -eq 0 ]; then
        ep_keep=$((ep_keep+1)); ep_keep_bytes=$((ep_keep_bytes+sz))
    else
        rm -f "$f" && ep_del=$((ep_del+1)) && ep_del_bytes=$((ep_del_bytes+sz))
    fi
done < <(find -L "$CKPT_DIR" -maxdepth 1 -name 'model_*_epoch_*.pt' 2>/dev/null)

echo
echo "Cleanup summary for ${STRATEGY_NAME}:"
echo "  step  ckpts deleted: ${del}    ($((del_bytes/1024/1024)) MB)"
echo "  step  ckpts kept:    ${keep}   ($((keep_bytes/1024/1024)) MB)"
echo "  epoch ckpts deleted: ${ep_del} ($((ep_del_bytes/1024/1024)) MB)"
echo "  epoch ckpts kept:    ${ep_keep} ($((ep_keep_bytes/1024/1024)) MB)"

n_epoch=$(find -L "$CKPT_DIR" -maxdepth 1 -name 'model_*_epoch_*.pt' 2>/dev/null | wc -l)
n_step=$(find -L "$CKPT_DIR"  -maxdepth 1 -name 'model_*_step_*.pt'  2>/dev/null | wc -l)
echo "  remaining files: ${n_epoch} epoch ckpts, ${n_step} step ckpts"
