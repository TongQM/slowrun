#!/bin/bash
#SBATCH --job-name=replay_ensemble
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260095p
#SBATCH --gpus=h100-80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --array=0-9
#SBATCH --output=experiments/logs/%x_%A_%a.out
#SBATCH --error=experiments/logs/%x_%A_%a.err
#
# Post-hoc ensemble eval with varying ensemble sizes.
# 2 strategies × 5 ensemble sizes = 10 replay jobs
#   Array 0..4: init_ens          ensemble sizes 2, 4, 8, 16, 20
#   Array 5..9: init_shuffle_ens  ensemble sizes 2, 4, 8, 16, 20
# (size=1 skipped: individual model val is already logged during training)
#
# All jobs share the same wandb group so the curves overlay.

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

# Flush Python stdout per line so SLURM .out shows live progress (not block-buffered).
export PYTHONUNBUFFERED=1

# --- Configuration (overridable via env vars from experiments/parallel/launch.sh) ---
N_LAYER="${N_LAYER:-12}"
N_EMBD="${N_EMBD:-768}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
END_EPOCH="${END_EPOCH:-$NUM_EPOCHS}"   # cap replay at this epoch (for partial training runs)
SKIP_INDIV_VAL="${SKIP_INDIV_VAL:-1}"   # 1 = don't re-evaluate per-model val (training already logs it); 0 = do eval
DATA_FRACTION="${DATA_FRACTION:-0.2}"
ENSEMBLE_MODE="${ENSEMBLE_MODE:-logit}"
WANDB_GROUP="parallel_d${N_LAYER}_w${N_EMBD}_df${DATA_FRACTION}_${SHARED_TIMESTAMP}"

# Ensemble sizes to sweep over (size=1 omitted; covered by per-model val during training)
# Override via ENS_SIZES_STR env var (space-separated), e.g. ENS_SIZES_STR="2 3 4 5"
ENS_SIZES_STR="${ENS_SIZES_STR:-2 4 8 16 20}"
read -ra ENS_SIZES <<< "$ENS_SIZES_STR"
NUM_SIZES=${#ENS_SIZES[@]}

# --- Map array index to (strategy, ensemble_size) ---
STRATEGY_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SIZES))
SIZE_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SIZES))
ENS_SIZE=${ENS_SIZES[$SIZE_IDX]}

case $STRATEGY_IDX in
    0) STRATEGY_NAME="init_ens" ;;
    1) STRATEGY_NAME="init_shuffle_ens" ;;
    *) echo "Invalid STRATEGY_IDX=$STRATEGY_IDX"; exit 1 ;;
esac

RUN_ID="parallel_${STRATEGY_NAME}_${SHARED_TIMESTAMP}"
CKPT_DIR="checkpoints/${RUN_ID}"
RUN_NAME="${WANDB_GROUP}_${STRATEGY_NAME}_ens${ENS_SIZE}_replay"
PROGRESS_FILE="${CKPT_DIR}/replay_progress_${STRATEGY_NAME}_ens${ENS_SIZE}.json"

echo "============================================================"
echo "Replay $STRATEGY_NAME with ensemble_size=$ENS_SIZE from $CKPT_DIR"
echo "  Wandb group: $WANDB_GROUP"
echo "  Wandb run name: $RUN_NAME"
echo "  Progress file: $PROGRESS_FILE"
echo "============================================================"

REPLAY_ARGS=(
    --checkpoint-dir=$CKPT_DIR
    --num-models=$ENS_SIZE
    --num-epochs=$NUM_EPOCHS
    --end-epoch=$END_EPOCH
    --ensemble-mode=$ENSEMBLE_MODE
    --wandb-run-name=$RUN_NAME
    --wandb-group=$WANDB_GROUP
    --progress-file=$PROGRESS_FILE
)
if [ "$SKIP_INDIV_VAL" = "1" ]; then
    REPLAY_ARGS+=(--skip-individual-val)
fi

python experiments/parallel/replay.py "${REPLAY_ARGS[@]}"

echo "Done: $STRATEGY_NAME ens$ENS_SIZE replay (exit $?)"
