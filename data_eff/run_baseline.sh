#!/bin/bash
#SBATCH --job-name=completep_baseline
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260095p
#SBATCH --gpus=h100-80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-2
#SBATCH --output=data_eff/logs/%x_%A_%a.out
#SBATCH --error=data_eff/logs/%x_%A_%a.err
#
# Baseline: 3 ensemble strategies on d12 model (n_layer=12, n_embd=768, n_head=12, ~125M params)
#   Array 0: no ensemble       (1 model)
#   Array 1: init ensemble     (5 models, same data order)
#   Array 2: init+shuffle      (5 models, different data orders)
#
# Submit: sbatch data_eff/run_baseline.sh
#

set -euo pipefail

# --- Environment ---
module load anaconda3/2024.10-1
conda activate /ocean/projects/cis260095p/ymiao6/scaling/slowrun/.conda_env

cd /ocean/projects/cis260095p/ymiao6/scaling/slowrun

# wandb API key (create this file with: echo "YOUR_KEY" > /ocean/projects/cis260095p/ymiao6/.wandb_key)
if [ -f /ocean/projects/cis260095p/ymiao6/.wandb_key ]; then
    export WANDB_API_KEY=$(cat /ocean/projects/cis260095p/ymiao6/.wandb_key)
fi

# Create log directory
mkdir -p data_eff/logs

# --- Configuration ---
N_LAYER=12
N_HEAD=12
N_EMBD=768
NUM_MODELS=5
NUM_EPOCHS=12
OPTIMIZER="hybrid"
ENSEMBLE_MODE="logit"
DUPE_START=$((N_LAYER / 2))       # decoder start
DUPE_END=$((N_LAYER - 2))         # leave last 2 layers unduplicated
WANDB_GROUP="baseline_d${N_LAYER}_w${N_EMBD}"

# --- Map array index to ensemble config ---
case $SLURM_ARRAY_TASK_ID in
    0)
        RUN_MODELS=1
        ENSEMBLE_TYPE="init_shuffle"
        RUN_NAME="${WANDB_GROUP}_no_ensemble"
        ;;
    1)
        RUN_MODELS=$NUM_MODELS
        ENSEMBLE_TYPE="init"
        RUN_NAME="${WANDB_GROUP}_init_ens"
        ;;
    2)
        RUN_MODELS=$NUM_MODELS
        ENSEMBLE_TYPE="init_shuffle"
        RUN_NAME="${WANDB_GROUP}_init_shuffle_ens"
        ;;
esac

echo "============================================================"
echo "Job: $SLURM_JOB_NAME (array $SLURM_ARRAY_TASK_ID)"
echo "Node: $(hostname), GPUs: ${SLURM_GPUS_ON_NODE:-4}"
echo "Run: $RUN_NAME"
echo "Config: d${N_LAYER} w${N_EMBD} h${N_HEAD}"
echo "Ensemble: type=$ENSEMBLE_TYPE, models=$RUN_MODELS"
echo "============================================================"

torchrun \
    --standalone \
    --nproc_per_node=1 \
    -- unlimited/train.py \
    --n_layer=$N_LAYER \
    --n_head=$N_HEAD \
    --n_embd=$N_EMBD \
    --num-models=$RUN_MODELS \
    --ensemble-type=$ENSEMBLE_TYPE \
    --num-epochs=$NUM_EPOCHS \
    --optimizer=$OPTIMIZER \
    --ensemble-mode=$ENSEMBLE_MODE \
    --dupe-layers-start=$DUPE_START \
    --dupe-layers-end=$DUPE_END \
    --num-epochs-model-0=$NUM_EPOCHS \
    --no-distill \
    --run=$RUN_NAME \
    --wandb_group=$WANDB_GROUP

echo "Done: $RUN_NAME (exit $?)"
