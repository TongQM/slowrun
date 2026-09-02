#!/bin/bash
#SBATCH --job-name=coordcheck
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260161p
#SBATCH --gpus=h100-80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --array=0-71
#SBATCH --output=experiments/logs/%x_%A_%a.out
#SBATCH --error=experiments/logs/%x_%A_%a.err
#
# CompleteP LR-transfer coordinate check (muP-style).
# Iso-aspect-ratio ladder (w/L = 64, head_dim = 64 fixed) x lr_multiplier x 3 seeds.
# 1 epoch, df=0.2 (~152 steps), wd=0, constant LR (no-warmdown). Goal: verify the
# loss-vs-lr_multiplier MINIMUM aligns across model sizes (absolute loss is expected
# to fall with size and is NOT the deliverable).
#
# 72 tasks = 4 sizes x 6 LRs x 3 seeds. Index decode:
#   seed   = t % 3 ;  lr_i = (t/3) % 6 ;  size_i = t / 18
# Submit via experiments/parallel/launch_coord_check.sh (NOT directly).
set -euo pipefail

GRID_TAG="${GRID_TAG:?set by launcher}"
CKPT_BASE="${CKPT_BASE:?set by launcher}"

module load anaconda3/2024.10-1
conda activate slowrun
cd /ocean/projects/cis260161p/ymiao6/scaling/slowrun
[ -f /ocean/projects/cis260161p/ymiao6/.wandb_key ] && export WANDB_API_KEY=$(cat /ocean/projects/cis260161p/ymiao6/.wandb_key)
export TIKTOKEN_CACHE_DIR=/ocean/projects/cis260161p/ymiao6/.tiktoken_cache
export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p experiments/logs

# --- iso-aspect-ratio ladder (w/L=64, head_dim=64) ---
SIZES_L=(6 12 18 24)
SIZES_W=(384 768 1152 1536)
SIZES_H=(6 12 18 24)          # n_head = w/64  -> head_dim = 64 at every rung
# LR grid is overridable so we can extend it downward without touching the original
# 72-run sweep dirs. Default = the original grid; the launcher passes LR_LIST.
LRS=(${LR_LIST:-0.0625 0.125 0.25 0.5 1.0 2.0})
NSEED=3

t=$SLURM_ARRAY_TASK_ID
SEED=$(( t % NSEED ))
LR_I=$(( (t / NSEED) % ${#LRS[@]} ))
SIZE_I=$(( t / (NSEED * ${#LRS[@]}) ))

L=${SIZES_L[$SIZE_I]}; W=${SIZES_W[$SIZE_I]}; H=${SIZES_H[$SIZE_I]}
LR=${LRS[$LR_I]}

GROUP="coordcheck_${GRID_TAG}"
RUN_ID="${GROUP}_d${L}_w${W}_lr${LR}"          # checkpoint dir (3 seeds share it)
RUN_NAME="${RUN_ID}_s${SEED}"

echo "COORDCHECK cell=d${L}_w${W} n_head=${H} lr_multiplier=${LR} seed=${SEED} (task ${t})"

# --resume expects the run dir to exist (start-fresh if empty). mkdir -p is race-safe
# across the 3 concurrent seeds that share this dir.
mkdir -p "${CKPT_BASE}/${RUN_ID}"

FINAL_CKPT="${CKPT_BASE}/${RUN_ID}/model_${SEED}_epoch_1.pt"
if [ -f "$FINAL_CKPT" ]; then
    echo "SKIP: $FINAL_CKPT exists"; exit 0
fi

torchrun --standalone --nproc_per_node=1 -- unlimited/train.py \
    --n_layer=$L --n_head=$H --n_embd=$W \
    --num-models=$NSEED --single-model-idx=$SEED \
    --ensemble-type=init \
    --num-epochs=1 --num-epochs-model-0=1 \
    --optimizer=adamw --ensemble-mode=logit \
    --data-fraction=0.2 \
    --lr_multiplier=$LR \
    --weight-decay=0 \
    --completep --mup-base-width=768 --mup-base-depth=12 --mup-base-head-dim=64 \
    --no-ve-projs --no-warmdown \
    --val-every-n-steps=38 \
    --total-batch-size=131072 \
    --compile-mode=inductor \
    --checkpoint-base="$CKPT_BASE" \
    --resume=$RUN_ID --run=$RUN_NAME --wandb_group=$GROUP

echo "Done: d${L}_w${W} lr=${LR} seed=${SEED} (exit $?)"
