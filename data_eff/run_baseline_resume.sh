#!/bin/bash
#SBATCH --job-name=baseline_resume
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260095p
#SBATCH --gpus=h100-80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=1-2
#SBATCH --output=data_eff/logs/%x_%A_%a.out
#SBATCH --error=data_eff/logs/%x_%A_%a.err
#
# Resume baseline jobs from model 2 onwards, with --no-distill
#   Array 1: init ensemble     (resume run_id=20260413_105339)
#   Array 2: init+shuffle      (resume run_id=20260413_105444)
#

set -euo pipefail

module load anaconda3/2024.10-1
conda activate /ocean/projects/cis260095p/ymiao6/scaling/slowrun/.conda_env

cd /ocean/projects/cis260095p/ymiao6/scaling/slowrun

if [ -f /ocean/projects/cis260095p/ymiao6/.wandb_key ]; then
    export WANDB_API_KEY=$(cat /ocean/projects/cis260095p/ymiao6/.wandb_key)
fi
mkdir -p data_eff/logs

# --- Configuration ---
N_LAYER=12
N_HEAD=12
N_EMBD=768
NUM_MODELS=5
NUM_EPOCHS=12
OPTIMIZER="hybrid"
ENSEMBLE_MODE="logit"
WANDB_GROUP="baseline_d${N_LAYER}_w${N_EMBD}"
DUPE_START=$((N_LAYER / 2))
DUPE_END=$((N_LAYER - 2))

# --- Map array index to ensemble config ---
case $SLURM_ARRAY_TASK_ID in
    1)
        ENSEMBLE_TYPE="init"
        RUN_NAME="${WANDB_GROUP}_init_ens"
        RESUME_ID="20260413_105339"
        ;;
    2)
        ENSEMBLE_TYPE="init_shuffle"
        RUN_NAME="${WANDB_GROUP}_init_shuffle_ens"
        RESUME_ID="20260413_105444"
        ;;
esac

echo "============================================================"
echo "Resume job (array $SLURM_ARRAY_TASK_ID) run_id=$RESUME_ID"
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
    --dupe-layers-start=$DUPE_START \
    --dupe-layers-end=$DUPE_END \
    --num-epochs-model-0=$NUM_EPOCHS \
    --no-distill \
    --resume=$RESUME_ID \
    --run=$RUN_NAME \
    --wandb_group=$WANDB_GROUP
