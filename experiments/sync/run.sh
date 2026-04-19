#!/bin/bash
#SBATCH -J sync_baseline
#SBATCH -p gpu-a100-small
#SBATCH -A DMS26010
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH --array=0-1
#SBATCH -o experiments/logs/%x_%A_%a.out
#SBATCH -e experiments/logs/%x_%A_%a.err
#
# Lonestar6 / TACC: synchronized in-process ensemble training (path a).
# Each array task trains all N models co-located on one A100 slice,
# evaluating per-model + ensemble val loss at each epoch boundary.
#   Array 0: init ensemble     (5 models, shared per-epoch data permutation)
#   Array 1: init+shuffle      (5 models, independent per-epoch permutations)
# The no-ensemble baseline comes free as model_1/* in either run (same seed=42).
#
# One-time setup: bash experiments/env/setup_lonestar.sh
# Submit:         sbatch experiments/sync/run.sh

set -euo pipefail

REPO_ROOT=/work/11426/yzfx0416/ls6/slowrun
cd "$REPO_ROOT"

module purge 2>/dev/null
module load cuda/12.8
unset PYTHONPATH PYTHONHOME
export LD_LIBRARY_PATH=/opt/apps/python/3.12.11/lib:${LD_LIBRARY_PATH:-}
source "$REPO_ROOT/.venv/bin/activate"

if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat "$HOME/.wandb_key")
fi
export WANDB_PROJECT=slowrun_lonestar
mkdir -p experiments/logs

# --- Configuration ---
N_LAYER=12
N_HEAD=12
N_EMBD=768
NUM_MODELS=5
NUM_EPOCHS=30              # multi-epoch dynamics
DATA_FRACTION=0.2          # 20M tokens per epoch (20% of 100M)
OPTIMIZER="hybrid"
ENSEMBLE_MODE="logit"
WANDB_GROUP="baseline_d${N_LAYER}_w${N_EMBD}_df${DATA_FRACTION}"

case $SLURM_ARRAY_TASK_ID in
    0) ENSEMBLE_TYPE="init";         RUN_NAME="${WANDB_GROUP}_init_ens" ;;
    1) ENSEMBLE_TYPE="init_shuffle"; RUN_NAME="${WANDB_GROUP}_init_shuffle_ens" ;;
    *) echo "Invalid array task $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "============================================================"
echo "Lonestar6 sync-ensemble array task $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "  Run: $RUN_NAME"
echo "  Config: d${N_LAYER} w${N_EMBD} h${N_HEAD}"
echo "  Ensemble: type=$ENSEMBLE_TYPE, models=$NUM_MODELS"
echo "============================================================"

torchrun \
    --standalone \
    --nproc_per_node=1 \
    -- unlimited/train.py \
    --n_layer=$N_LAYER \
    --n_head=$N_HEAD \
    --n_embd=$N_EMBD \
    --num-models=$NUM_MODELS \
    --ensemble-type=$ENSEMBLE_TYPE \
    --num-epochs=$NUM_EPOCHS \
    --optimizer=$OPTIMIZER \
    --ensemble-mode=$ENSEMBLE_MODE \
    --data-fraction=$DATA_FRACTION \
    --val-every-n-steps=10 \
    --num-epochs-model-0=$NUM_EPOCHS \
    --completep \
    --compile-mode=inductor \
    --run=$RUN_NAME \
    --wandb_group=$WANDB_GROUP

echo "Done: $RUN_NAME (exit $?)"
