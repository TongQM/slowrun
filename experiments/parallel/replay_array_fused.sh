#!/bin/bash
#SBATCH --job-name=replay_fused
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260161p
#SBATCH --gpus=h100-80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --array=0-1
#SBATCH --output=experiments/logs/%x_%A_%a.out
#SBATCH --error=experiments/logs/%x_%A_%a.err
#
# Fused replay: 2 array tasks (one per strategy). Each task evaluates ALL
# ensemble sizes in a single pass over the saved per-epoch checkpoints.
#
# Required env vars (set by launcher):
#   SHARED_TIMESTAMP  per-cell run id suffix
#   NUM_MODELS        e.g. 20
#   ENS_SIZES_STR     space-separated, e.g. "2 5 10 15 20"
#   NUM_EPOCHS        end-of-training epoch count
#   END_EPOCH         cap (defaults to NUM_EPOCHS)
#   WANDB_GROUP
#   CHECKPOINT_BASE   defaults to "checkpoints"
#   ENSEMBLE_MODE     defaults to "logit"

set -euo pipefail

if [ -z "${SHARED_TIMESTAMP:-}" ]; then
    echo "ERROR: SHARED_TIMESTAMP env var not set."
    exit 1
fi

module load anaconda3/2024.10-1
conda activate slowrun

cd /ocean/projects/cis260161p/ymiao6/scaling/slowrun

if [ -f /ocean/projects/cis260161p/ymiao6/.wandb_key ]; then
    export WANDB_API_KEY=$(cat /ocean/projects/cis260161p/ymiao6/.wandb_key)
fi
mkdir -p experiments/logs
export PYTHONUNBUFFERED=1

NUM_MODELS="${NUM_MODELS:-20}"
NUM_EPOCHS="${NUM_EPOCHS:-25}"
END_EPOCH="${END_EPOCH:-$NUM_EPOCHS}"
ENS_SIZES_STR="${ENS_SIZES_STR:-2 5 10 15 20}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-checkpoints}"
ENSEMBLE_MODE="${ENSEMBLE_MODE:-logit}"

case "$SLURM_ARRAY_TASK_ID" in
    0) STRATEGY_NAME="init_ens" ;;
    1) STRATEGY_NAME="init_shuffle_ens" ;;
    *) echo "Invalid SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

RUN_ID="parallel_${STRATEGY_NAME}_${SHARED_TIMESTAMP}"
CKPT_DIR="${CHECKPOINT_BASE}/${RUN_ID}"
RUN_PREFIX="${WANDB_GROUP}_${STRATEGY_NAME}"

echo "============================================================"
echo "Fused replay  strategy=$STRATEGY_NAME  ckpt_dir=$CKPT_DIR"
echo "  num_models=$NUM_MODELS  ens_sizes=$ENS_SIZES_STR  epochs 1..$END_EPOCH"
echo "  wandb group=$WANDB_GROUP  run prefix=$RUN_PREFIX"
echo "============================================================"

python experiments/parallel/replay_fused.py \
    --checkpoint-dir="$CKPT_DIR" \
    --num-models="$NUM_MODELS" \
    --ens-sizes $ENS_SIZES_STR \
    --num-epochs="$NUM_EPOCHS" \
    --end-epoch="$END_EPOCH" \
    --ensemble-mode="$ENSEMBLE_MODE" \
    --wandb-group="$WANDB_GROUP" \
    --wandb-run-name-prefix="$RUN_PREFIX"

echo "Done: $STRATEGY_NAME fused replay (exit $?)"
