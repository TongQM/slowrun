#!/bin/bash
#SBATCH --job-name=replay_ensemble
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260095p
#SBATCH --gpus=h100-80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-1
#SBATCH --output=experiments/logs/%x_%A_%a.out
#SBATCH --error=experiments/logs/%x_%A_%a.err
#
# Post-hoc ensemble eval. One job per ensemble strategy:
#   Array 0: replay init_ens
#   Array 1: replay init_shuffle_ens
# Submit via experiments/parallel/launch.sh with --dependency=afterok on the training array.

set -euo pipefail

if [ -z "${SHARED_TIMESTAMP:-}" ]; then
    echo "ERROR: SHARED_TIMESTAMP env var not set."
    exit 1
fi

module load anaconda3/2024.10-1
conda activate slowrun

cd /ocean/projects/cis260095p/ymiao6/scaling/slowrun

if [ -f /ocean/projects/cis260095p/ymiao6/.wandb_key ]; then
    export WANDB_API_KEY=$(cat /ocean/projects/cis260095p/ymiao6/.wandb_key)
fi
mkdir -p experiments/logs

# --- Configuration (must match experiments/parallel/train_array.sh) ---
N_LAYER=12
N_EMBD=768
NUM_MODELS=5
NUM_EPOCHS=30
DATA_FRACTION=0.2
ENSEMBLE_MODE="logit"
WANDB_GROUP="parallel_d${N_LAYER}_w${N_EMBD}_df${DATA_FRACTION}_${SHARED_TIMESTAMP}"

case $SLURM_ARRAY_TASK_ID in
    0) STRATEGY_NAME="init_ens" ;;
    1) STRATEGY_NAME="init_shuffle_ens" ;;
    *) echo "Invalid array task $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

RUN_ID="parallel_${STRATEGY_NAME}_${SHARED_TIMESTAMP}"
CKPT_DIR="checkpoints/${RUN_ID}"
RUN_NAME="${WANDB_GROUP}_${STRATEGY_NAME}_replay"

echo "============================================================"
echo "Replay $STRATEGY_NAME from $CKPT_DIR"
echo "============================================================"

python experiments/parallel/replay.py \
    --checkpoint-dir=$CKPT_DIR \
    --num-models=$NUM_MODELS \
    --num-epochs=$NUM_EPOCHS \
    --ensemble-mode=$ENSEMBLE_MODE \
    --wandb-run-name=$RUN_NAME \
    --wandb-group=$WANDB_GROUP

echo "Done: $STRATEGY_NAME replay (exit $?)"
