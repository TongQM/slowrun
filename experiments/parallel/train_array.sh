#!/bin/bash
#SBATCH -J parallel_train
#SBATCH -p gpu-a100-small
#SBATCH -A DMS26010
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --array=0-9
#SBATCH -o experiments/logs/%x_%A_%a.out
#SBATCH -e experiments/logs/%x_%A_%a.err
#
# Lonestar6 / TACC: 1 A100-40GB slice per array task.
# 10 tasks = 5 models × 2 ensemble strategies (init, init_shuffle).
# Each task trains ONE model and writes into a strategy-shared checkpoint dir
# keyed by SHARED_TIMESTAMP (set by experiments/parallel/launch.sh).
#
# Submit via: bash experiments/parallel/launch.sh   (NOT directly with sbatch)

set -euo pipefail

if [ -z "${SHARED_TIMESTAMP:-}" ]; then
    echo "ERROR: SHARED_TIMESTAMP env var not set. Use experiments/parallel/launch.sh."
    exit 1
fi

REPO_ROOT=/work/11426/yzfx0416/ls6/slowrun
cd "$REPO_ROOT"

# --- Environment (one-time setup via experiments/env/setup_lonestar.sh) ---
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
NUM_MODELS=5
NUM_EPOCHS=30
DATA_FRACTION=0.2
OPTIMIZER="hybrid"
ENSEMBLE_MODE="logit"
WANDB_GROUP="parallel_d${N_LAYER}_w${N_EMBD}_df${DATA_FRACTION}_${SHARED_TIMESTAMP}"

# --- Map array index to (strategy, model_idx) ---
STRATEGY_IDX=$((SLURM_ARRAY_TASK_ID / NUM_MODELS))
MODEL_IDX=$((SLURM_ARRAY_TASK_ID % NUM_MODELS))

case $STRATEGY_IDX in
    0) ENSEMBLE_TYPE="init";         STRATEGY_NAME="init_ens" ;;
    1) ENSEMBLE_TYPE="init_shuffle"; STRATEGY_NAME="init_shuffle_ens" ;;
    *) echo "Invalid STRATEGY_IDX=$STRATEGY_IDX"; exit 1 ;;
esac

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
