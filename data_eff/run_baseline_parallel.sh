#!/bin/bash
#SBATCH --job-name=parallel_baseline
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260095p
#SBATCH --gpus=h100-80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-9
#SBATCH --output=data_eff/logs/%x_%A_%a.out
#SBATCH --error=data_eff/logs/%x_%A_%a.err
#
# Parallel single-model training: 5 models × 2 ensemble strategies = 10 jobs
#   Array 0..4: init ensemble        models 0..4
#   Array 5..9: init+shuffle ensemble models 0..4
#
# All 5 jobs of one strategy write to a SHARED checkpoint dir, identified by
# the env var SHARED_TIMESTAMP (set by data_eff/launch_parallel.sh).
#
# Submit via: bash data_eff/launch_parallel.sh   (NOT directly with sbatch)
#

set -euo pipefail

if [ -z "${SHARED_TIMESTAMP:-}" ]; then
    echo "ERROR: SHARED_TIMESTAMP env var not set. Use data_eff/launch_parallel.sh."
    exit 1
fi

# --- Environment ---
module load anaconda3/2024.10-1
conda activate slowrun

cd /ocean/projects/cis260095p/ymiao6/scaling/slowrun

if [ -f /ocean/projects/cis260095p/ymiao6/.wandb_key ]; then
    export WANDB_API_KEY=$(cat /ocean/projects/cis260095p/ymiao6/.wandb_key)
fi
mkdir -p data_eff/logs

# --- Configuration (must match run_baseline.sh / launch_parallel.sh) ---
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
    0)
        ENSEMBLE_TYPE="init"
        STRATEGY_NAME="init_ens"
        ;;
    1)
        ENSEMBLE_TYPE="init_shuffle"
        STRATEGY_NAME="init_shuffle_ens"
        ;;
    *)
        echo "Invalid STRATEGY_IDX=$STRATEGY_IDX"; exit 1
        ;;
esac

RUN_ID="parallel_${STRATEGY_NAME}_${SHARED_TIMESTAMP}"
RUN_NAME="${WANDB_GROUP}_${STRATEGY_NAME}_model${MODEL_IDX}"

echo "============================================================"
echo "Parallel job (array $SLURM_ARRAY_TASK_ID)"
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
