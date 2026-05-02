#!/bin/bash
#SBATCH --job-name=parallel_baseline
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260161p
#SBATCH --gpus=h100-80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-39
#SBATCH --output=experiments/logs/%x_%A_%a.out
#SBATCH --error=experiments/logs/%x_%A_%a.err
#
# Parallel single-model training: 20 models × 2 ensemble strategies = 40 jobs
#   Array 0..19:  init ensemble        models 0..19
#   Array 20..39: init+shuffle ensemble models 0..19
#
# All jobs of one strategy write to a SHARED checkpoint dir, identified by
# the env var SHARED_TIMESTAMP (set by experiments/parallel/launch.sh).
# Jobs skip training if their final checkpoint already exists (safe to resubmit).
#
# Submit via: bash experiments/parallel/launch.sh   (NOT directly with sbatch)
#

set -euo pipefail

if [ -z "${SHARED_TIMESTAMP:-}" ]; then
    echo "ERROR: SHARED_TIMESTAMP env var not set. Use experiments/parallel/launch.sh."
    exit 1
fi

# --- Environment ---
module load anaconda3/2024.10-1
conda activate slowrun

cd /ocean/projects/cis260161p/ymiao6/scaling/slowrun

if [ -f /ocean/projects/cis260161p/ymiao6/.wandb_key ]; then
    export WANDB_API_KEY=$(cat /ocean/projects/cis260161p/ymiao6/.wandb_key)
fi
mkdir -p experiments/logs

# --- Configuration (overridable via env vars from experiments/parallel/launch.sh) ---
N_LAYER="${N_LAYER:-12}"
N_HEAD="${N_HEAD:-12}"
N_EMBD="${N_EMBD:-768}"
NUM_MODELS="${NUM_MODELS:-5}"
NUM_EPOCHS="${NUM_EPOCHS:-25}"
DATA_FRACTION="${DATA_FRACTION:-0.2}"
OPTIMIZER="${OPTIMIZER:-adamw}"
ENSEMBLE_MODE="${ENSEMBLE_MODE:-logit}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-131072}"   # tokens per optimizer step
# --compile-mode: inductor is fastest but requires H100; eager works on all GPUs
COMPILE_MODE="${COMPILE_MODE:-inductor}"
# --val-every-n-steps: 0 = only at epoch boundary (fast). >0 gives denser val curves but much slower.
VAL_EVERY_N_STEPS="${VAL_EVERY_N_STEPS:-0}"
# CompleteP defaults: ALWAYS on for this project. NO_VE_PROJS=1 → adds --no-ve-projs.
COMPLETEP="${COMPLETEP:-1}"
NO_VE_PROJS="${NO_VE_PROJS:-1}"
MUP_BASE_WIDTH="${MUP_BASE_WIDTH:-768}"
MUP_BASE_DEPTH="${MUP_BASE_DEPTH:-12}"
MUP_BASE_HEAD_DIM="${MUP_BASE_HEAD_DIM:-64}"
WANDB_GROUP="${WANDB_GROUP:-parallel_d${N_LAYER}_w${N_EMBD}_df${DATA_FRACTION}_${SHARED_TIMESTAMP}}"

EXTRA_FLAGS=()
if [ "$COMPLETEP" = "1" ]; then
    EXTRA_FLAGS+=(--completep
                  --mup-base-width=$MUP_BASE_WIDTH
                  --mup-base-depth=$MUP_BASE_DEPTH
                  --mup-base-head-dim=$MUP_BASE_HEAD_DIM)
fi
if [ "$NO_VE_PROJS" = "1" ]; then
    EXTRA_FLAGS+=(--no-ve-projs)
fi
if [ "${NO_WARMDOWN:-0}" = "1" ]; then
    EXTRA_FLAGS+=(--no-warmdown)
fi
if [ "${CHECKPOINT_EVERY_N_STEPS:-0}" -gt 0 ] 2>/dev/null; then
    EXTRA_FLAGS+=(--checkpoint-every-n-steps=$CHECKPOINT_EVERY_N_STEPS)
fi
if [ -n "${CHECKPOINT_BASE:-}" ]; then
    EXTRA_FLAGS+=(--checkpoint-base="$CHECKPOINT_BASE")
fi

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
_CKPT_BASE="${CHECKPOINT_BASE:-checkpoints}"
FINAL_CKPT="${_CKPT_BASE}/${RUN_ID}/model_${MODEL_IDX}_epoch_${NUM_EPOCHS}.pt"

echo "============================================================"
echo "Parallel job (array $SLURM_ARRAY_TASK_ID)"
echo "  Strategy: $STRATEGY_NAME (ensemble_type=$ENSEMBLE_TYPE)"
echo "  Model index: $MODEL_IDX of $NUM_MODELS"
echo "  Run ID (shared dir): $RUN_ID"
echo "  Wandb group: $WANDB_GROUP"
echo "============================================================"

# Skip if this model has already completed all epochs
if [ -f "$FINAL_CKPT" ]; then
    echo "SKIP: final checkpoint already exists: $FINAL_CKPT"
    exit 0
fi

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
    --val-every-n-steps=$VAL_EVERY_N_STEPS \
    --total-batch-size=$TOTAL_BATCH_SIZE \
    --num-epochs-model-0=$NUM_EPOCHS \
    --compile-mode=$COMPILE_MODE \
    "${EXTRA_FLAGS[@]}" \
    --resume=$RUN_ID \
    --run=$RUN_NAME \
    --wandb_group=$WANDB_GROUP

echo "Done: model $MODEL_IDX of $STRATEGY_NAME (exit $?)"
