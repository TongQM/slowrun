#!/bin/bash
# CompleteP correctness diagnostic — width/depth stability sweep.
# Submits 8 single-model SLURM jobs, all with --completep, same LR / batch / data / num_epochs.
# Configs span both width and depth axes, halving and doubling from the base (d12_w768):
#   original      d12_h12_w768    (base; all CompleteP multipliers = 1)
#   halfwidth     d12_h6_w384     (width 1/2)
#   quarterwidth  d12_h3_w192     (width 1/4)
#   doublewidth   d12_h24_w1536   (width 2x)
#   halfdepth     d6_h12_w768     (depth 1/2)
#   thirddepth    d4_h12_w768     (depth 1/3 — n_layer=3 breaks the U-Net skip indexing,
#                                    so we use 4 as the next-smallest even value)
#   doubledepth   d24_h12_w768    (depth 2x)
#   halfboth      d6_h6_w384      (depth 1/2 + width 1/2 — tests compounded scaling)
#
# Under correct CompleteP, val-loss curves should track each other in step-space,
# ordered by capacity (smaller models above larger models, no crossings, similar slope).
#
# Usage:
#   bash experiments/diagnostic/launch.sh
#
# Optional env overrides (passed through to job.sh):
#   NUM_EPOCHS=20  DATA_FRACTION=0.1  TOTAL_BATCH_SIZE=524288
#   OPTIMIZER=adamw  COMPILE_MODE=inductor
#   GPU_SPEC=h100-80:1   (must be H100 if COMPILE_MODE=inductor)

set -euo pipefail
cd "$(dirname "$0")/../.."

# TS=<existing_timestamp> bash launch.sh  -> reuse existing checkpoint dirs and
# wandb runs (resume training from the latest per-epoch checkpoint, continue
# logging on the same wandb run). Otherwise a fresh timestamp is generated.
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
WANDB_GROUP="completep_diag_df${DATA_FRACTION:-0.1}_${TS}"
GPU_SPEC="${GPU_SPEC:-h100-80:1}"
TIME_LIMIT="${TIME_LIMIT:-2:00:00}"

mkdir -p experiments/logs

echo "============================================================"
echo "CompleteP correctness diagnostic"
echo "  Timestamp:   $TS"
echo "  Wandb group: $WANDB_GROUP"
echo "  GPU spec:    $GPU_SPEC"
echo "  Pass-through: NUM_EPOCHS=${NUM_EPOCHS:-20} DATA_FRACTION=${DATA_FRACTION:-0.1}"
echo "                TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-524288}"
echo "                OPTIMIZER=${OPTIMIZER:-adamw} COMPILE_MODE=${COMPILE_MODE:-inductor}"
echo "============================================================"

submit() {
    local TAG=$1; local L=$2; local H=$3; local W=$4
    local RUN_ID="diag_${TAG}_${TS}"
    local RUN_NAME="${WANDB_GROUP}_${TAG}"
    mkdir -p "checkpoints/${RUN_ID}"

    # Build a comma-separated --export list, including any forwarded overrides.
    local EXPORTS="ALL,N_LAYER=$L,N_HEAD=$H,N_EMBD=$W,RUN_ID=$RUN_ID,RUN_NAME=$RUN_NAME,WANDB_GROUP=$WANDB_GROUP"
    for v in NUM_EPOCHS DATA_FRACTION TOTAL_BATCH_SIZE OPTIMIZER COMPILE_MODE \
             MUP_BASE_WIDTH MUP_BASE_DEPTH MUP_BASE_HEAD_DIM NO_VE_PROJS; do
        if [ -n "${!v:-}" ]; then EXPORTS+=",$v=${!v}"; fi
    done

    JOB=$(sbatch --parsable \
        --job-name=completep_diag_${TAG} \
        --gpus=$GPU_SPEC \
        --time=$TIME_LIMIT \
        --export="$EXPORTS" \
        experiments/diagnostic/job.sh)
    echo "  submitted $TAG  (d${L}_h${H}_w${W})  job=$JOB  run=$RUN_NAME"
}

SUITE="${SUITE:-all}"   # "widths", "depths", or "all"

case "$SUITE" in
    widths|all)
        submit original     12 12 768
        submit halfwidth    12 6  384
        submit quarterwidth 12 3  192
        submit doublewidth  12 24 1536
        ;;
esac
case "$SUITE" in
    depths|all)
        submit halfdepth    6  12 768
        submit thirddepth   4  12 768
        submit doubledepth  24 12 768
        submit halfboth     6  6  384
        ;;
esac

echo
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f experiments/logs/completep_diag_*.out"
echo "Wandb:"
echo "  group=$WANDB_GROUP"
