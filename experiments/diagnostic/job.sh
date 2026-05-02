#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260161p
#SBATCH --gpus=h100-80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err
#
# CompleteP diagnostic single-model trainer. Configurable via env vars from launch.sh.
#   Required: N_LAYER, N_HEAD, N_EMBD, RUN_ID, RUN_NAME, WANDB_GROUP
#   Optional: NUM_EPOCHS (20), DATA_FRACTION (0.1), TOTAL_BATCH_SIZE (524288),
#             OPTIMIZER (adamw), COMPILE_MODE (inductor),
#             MUP_BASE_WIDTH (768), MUP_BASE_DEPTH (12), MUP_BASE_HEAD_DIM (64)

set -euo pipefail

module load anaconda3/2024.10-1
conda activate slowrun

cd /ocean/projects/cis260161p/ymiao6/scaling/slowrun

if [ -f /ocean/projects/cis260161p/ymiao6/.wandb_key ]; then
    export WANDB_API_KEY=$(cat /ocean/projects/cis260161p/ymiao6/.wandb_key)
fi
mkdir -p experiments/logs

: "${N_LAYER:?N_LAYER not set}"
: "${N_HEAD:?N_HEAD not set}"
: "${N_EMBD:?N_EMBD not set}"
: "${RUN_ID:?RUN_ID not set}"
: "${RUN_NAME:?RUN_NAME not set}"
: "${WANDB_GROUP:?WANDB_GROUP not set}"

NUM_EPOCHS="${NUM_EPOCHS:-20}"
DATA_FRACTION="${DATA_FRACTION:-0.1}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"
OPTIMIZER="${OPTIMIZER:-adamw}"
COMPILE_MODE="${COMPILE_MODE:-inductor}"
MUP_BASE_WIDTH="${MUP_BASE_WIDTH:-768}"
MUP_BASE_DEPTH="${MUP_BASE_DEPTH:-12}"
MUP_BASE_HEAD_DIM="${MUP_BASE_HEAD_DIM:-64}"
NO_VE_PROJS="${NO_VE_PROJS:-0}"  # 1 -> add --no-ve-projs (disable value-embedding projections)

NO_VE_FLAG=""
if [ "$NO_VE_PROJS" = "1" ]; then
    NO_VE_FLAG="--no-ve-projs"
fi

echo "============================================================"
echo "CompleteP diagnostic"
echo "  Config: d${N_LAYER}_h${N_HEAD}_w${N_EMBD}"
echo "  Epochs: $NUM_EPOCHS  data_fraction=$DATA_FRACTION  batch=$TOTAL_BATCH_SIZE"
echo "  Optimizer: $OPTIMIZER  compile=$COMPILE_MODE"
echo "  CompleteP base: w=$MUP_BASE_WIDTH L=$MUP_BASE_DEPTH d_head=$MUP_BASE_HEAD_DIM  no_ve_projs=$NO_VE_PROJS"
echo "  Run ID: $RUN_ID"
echo "  Wandb group: $WANDB_GROUP / run: $RUN_NAME"
echo "============================================================"

torchrun --standalone --nproc_per_node=1 \
    -- unlimited/train.py \
    --completep \
    --mup-base-width=$MUP_BASE_WIDTH \
    --mup-base-depth=$MUP_BASE_DEPTH \
    --mup-base-head-dim=$MUP_BASE_HEAD_DIM \
    $NO_VE_FLAG \
    --n_layer=$N_LAYER --n_head=$N_HEAD --n_embd=$N_EMBD \
    --num-models=1 --single-model-idx=0 --num-epochs-model-0=$NUM_EPOCHS \
    --num-epochs=$NUM_EPOCHS \
    --data-fraction=$DATA_FRACTION \
    --total-batch-size=$TOTAL_BATCH_SIZE \
    --optimizer=$OPTIMIZER \
    --compile-mode=$COMPILE_MODE \
    --resume=$RUN_ID \
    --run=$RUN_NAME \
    --wandb_group=$WANDB_GROUP

echo "Done: $RUN_NAME (exit $?)"
