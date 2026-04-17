#!/bin/bash
#SBATCH -J parallel_train
#SBATCH -p gpu-a100-small
#SBATCH -A DMS26010
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o experiments/logs/%x_%A_%a.out
#SBATCH -e experiments/logs/%x_%A_%a.err
#
# Lonestar6 / TACC: 1 A100-40GB slice per array task.
# Each task trains ONE model for all epochs. Array indices map to model
# indices within a SINGLE strategy (set via ENSEMBLE_TYPE env var).
# The orchestrator (launch.sh) submits one array per strategy.
#
# Submit via: bash experiments/parallel/launch.sh   (NOT directly with sbatch)

set -euo pipefail

if [ -z "${SHARED_TIMESTAMP:-}" ]; then
    echo "ERROR: SHARED_TIMESTAMP env var not set. Use experiments/parallel/launch.sh."
    exit 1
fi
if [ -z "${ENSEMBLE_TYPE:-}" ] || [ -z "${STRATEGY_NAME:-}" ]; then
    echo "ERROR: ENSEMBLE_TYPE and STRATEGY_NAME must be set by launch.sh."
    exit 1
fi

REPO_ROOT=/work/11426/yzfx0416/ls6/slowrun
cd "$REPO_ROOT"

module load cuda/12.8 python/3.12.11
source "$REPO_ROOT/.venv/bin/activate"

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT=slowrun_lonestar
mkdir -p experiments/logs

# --- Configuration (must match experiments/parallel/replay_array.sh) ---
N_LAYER=12
N_HEAD=12
N_EMBD=768
NUM_MODELS=20
NUM_EPOCHS=20
DATA_FRACTION=0.2
OPTIMIZER="hybrid"
ENSEMBLE_MODE="logit"
WANDB_GROUP="parallel_d${N_LAYER}_w${N_EMBD}_df${DATA_FRACTION}_${SHARED_TIMESTAMP}"

MODEL_IDX=$SLURM_ARRAY_TASK_ID

RUN_ID="parallel_${STRATEGY_NAME}_${SHARED_TIMESTAMP}"
RUN_NAME="${WANDB_GROUP}_${STRATEGY_NAME}_model${MODEL_IDX}"

echo "============================================================"
echo "Lonestar6 array task $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "  Strategy: $STRATEGY_NAME (ensemble_type=$ENSEMBLE_TYPE)"
echo "  Model index: $MODEL_IDX of $NUM_MODELS"
echo "  Run ID (shared dir): $RUN_ID"
echo "  Wandb group: $WANDB_GROUP"
echo "============================================================"

torchrun \
    --standalone \
    --nproc_per_node=1 \
    -- unlimited/train.py \
    --n_layer=$N_LAYER \
    --n_head=$N_HEAD \
    --n_embd=$N_EMBD \
    --num-models=$NUM_MODELS \
    --single-model-idx=$MODEL_IDX \
    --ensemble-type=$ENSEMBLE_TYPE \
    --num-epochs=$NUM_EPOCHS \
    --optimizer=$OPTIMIZER \
    --ensemble-mode=$ENSEMBLE_MODE \
    --data-fraction=$DATA_FRACTION \
    --val-every-n-steps=10 \
    --num-epochs-model-0=$NUM_EPOCHS \
    --compile-mode=inductor \
    --resume=$RUN_ID \
    --run=$RUN_NAME \
    --wandb_group=$WANDB_GROUP

echo "Done: model $MODEL_IDX of $STRATEGY_NAME (exit $?)"
