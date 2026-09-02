#!/bin/bash
#SBATCH --job-name=single_1p8b
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260095p
#SBATCH --gpus=h100-80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err
#
# Worker for launch_single_1p8b.sh. Trains ONE ~1.8B looped model. Resumable
# (constant LR / --no-warmdown): resubmit with the same RUN_ID to continue from
# the latest per-epoch checkpoint.
set -euo pipefail

module load anaconda3/2024.10-1
conda activate slowrun
cd /ocean/projects/cis260161p/ymiao6/scaling/slowrun

if [ -f /ocean/projects/cis260161p/ymiao6/.wandb_key ]; then
    export WANDB_API_KEY=$(cat /ocean/projects/cis260161p/ymiao6/.wandb_key)
fi
mkdir -p experiments/logs
export TIKTOKEN_CACHE_DIR=/ocean/projects/cis260161p/ymiao6/.tiktoken_cache
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1

# config (from launcher --export; with fallbacks)
N_LAYER="${N_LAYER:-30}"; N_HEAD="${N_HEAD:-16}"; N_EMBD="${N_EMBD:-2048}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"; DF="${DF:-1.0}"
OPTIMIZER="${OPTIMIZER:-hybrid}"; LR_MULTIPLIER="${LR_MULTIPLIER:-0.25}"; WEIGHT_DECAY="${WEIGHT_DECAY:-1.3}"
DUPE_START="${DUPE_START:-15}"; DUPE_END="${DUPE_END:-25}"; DUPE_FRACTION="${DUPE_FRACTION:-0.0}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-131072}"; DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"
VAL_EVERY_N_STEPS="${VAL_EVERY_N_STEPS:-152}"; COMPILE_MODE="${COMPILE_MODE:-inductor}"
MUP_BASE_WIDTH="${MUP_BASE_WIDTH:-768}"; MUP_BASE_DEPTH="${MUP_BASE_DEPTH:-12}"; MUP_BASE_HEAD_DIM="${MUP_BASE_HEAD_DIM:-64}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-checkpoints}"
RUN_ID="${RUN_ID:?RUN_ID must be set by launcher}"
WANDB_GROUP="${WANDB_GROUP:-$RUN_ID}"

echo "============================================================"
echo "single 1.8B  RUN_ID=$RUN_ID  d${N_LAYER}/w${N_EMBD}/h${N_HEAD}"
echo "  loop ${DUPE_START}-$((DUPE_END-1)) frac=$DUPE_FRACTION  opt=$OPTIMIZER lr_mult=$LR_MULTIPLIER wd=$WEIGHT_DECAY"
echo "  completep=ON ve_projs=ON warmdown=OFF  batch=$TOTAL_BATCH_SIZE/dev$DEVICE_BATCH_SIZE  epochs=$NUM_EPOCHS"
echo "============================================================"

torchrun --standalone --nproc_per_node=1 -- unlimited/train.py \
    --n_layer=$N_LAYER --n_head=$N_HEAD --n_embd=$N_EMBD --num-models=1 \
    --num-epochs=$NUM_EPOCHS --data-fraction=$DF \
    --optimizer=$OPTIMIZER --lr_multiplier=$LR_MULTIPLIER --weight-decay=$WEIGHT_DECAY \
    --dupe-layers-start=$DUPE_START --dupe-layers-end=$DUPE_END --dupe-fraction=$DUPE_FRACTION \
    --completep --mup-base-width=$MUP_BASE_WIDTH --mup-base-depth=$MUP_BASE_DEPTH --mup-base-head-dim=$MUP_BASE_HEAD_DIM \
    --total-batch-size=$TOTAL_BATCH_SIZE --device-batch-size=$DEVICE_BATCH_SIZE \
    --val-every-n-steps=$VAL_EVERY_N_STEPS --compile-mode=$COMPILE_MODE --no-warmdown \
    --ensemble-mode=logit --checkpoint-base=$CHECKPOINT_BASE \
    --resume=$RUN_ID --run=$RUN_ID --wandb_group=$WANDB_GROUP

echo "Done: $RUN_ID (exit $?)"
