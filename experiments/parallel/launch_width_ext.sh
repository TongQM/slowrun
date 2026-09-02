#!/bin/bash
# Width-extension cells for the df=1.0 (full-data) Panel-B width sweep.
#
# Panel B ("ensembling beats width-scaling at matched compute") needs d=12, E=1
# (single individual) min val loss across widths. The fulldata grid only trained
# w384 and w768; this adds w1152 (2.25x compute) and w1536 (4x compute) so the
# width curve reaches the ensemble's 4x point.
#
# Only E=1 is needed -> train ONE init model (array=0 with NUM_MODELS=5 keeps the
# seed derivation identical to the existing fulldata cells). No replay/cleanup.
# No step checkpoints (val loss is read straight from the .out log); per-epoch
# ckpts only (resume safety), routed to cis260095p, deletable after extraction.
#
# Recipe matches fulldata_20260528_150057 EXACTLY: df=1.0, wd=0, no-warmdown,
# completep, no-ve-projs, adamw, batch 131072, n_head=12, muP base w768/d12/h64,
# 40 epochs.  Compute charged to cis260161p (user directive).
#
#   DRY_RUN=1 bash experiments/parallel/launch_width_ext.sh   # preview
#   bash experiments/parallel/launch_width_ext.sh             # submit
set -euo pipefail
cd "$(dirname "$0")/../.."

GRID_TAG="${GRID_TAG:-fulldata_widthext_20260618}"
DRY_RUN="${DRY_RUN:-0}"

ACCOUNT=cis260161p
GPU_SPEC=h100-80:1
COMPILE_MODE=inductor
COMPLETEP=1
NO_VE_PROJS=1
NO_WARMDOWN=1
WEIGHT_DECAY=0
TOTAL_BATCH_SIZE=131072
NUM_MODELS=5            # virtual ensemble size (seed parity); we train only model 0
NUM_EPOCHS=40
DATA_FRACTION=1.0
OPTIMIZER=adamw
ENSEMBLE_MODE=logit
VAL_EVERY_N_STEPS=152
CHECKPOINT_EVERY_N_STEPS=0     # NO step ckpts (width curve needs only logged val)
MUP_BASE_WIDTH=768
MUP_BASE_DEPTH=12
MUP_BASE_HEAD_DIM=64

# Transient per-epoch ckpts -> cis260095p (845 GB free), keep cis260161p clear.
CKPT_BASE=/ocean/projects/cis260095p/ymiao6/scaling/slowrun/checkpoints
mkdir -p "$CKPT_BASE" experiments/logs

# (width, n_head, walltime) — n_head=12 fixed at d12 per grid convention.
declare -a WIDTHS=(1152 1536)
declare -a TIMES=("14:00:00" "24:00:00")
L=12; H=12

for i in "${!WIDTHS[@]}"; do
    W=${WIDTHS[$i]}
    TL=${TIMES[$i]}
    CELL_TS="${GRID_TAG}_d${L}_w${W}"
    GROUP="${GRID_TAG}_d${L}_w${W}"
    mkdir -p "$CKPT_BASE/parallel_init_ens_${CELL_TS}"

    exports="ALL,N_LAYER=$L,N_HEAD=$H,N_EMBD=$W,SHARED_TIMESTAMP=$CELL_TS,WANDB_GROUP=$GROUP"
    exports+=",NUM_MODELS=$NUM_MODELS,NUM_EPOCHS=$NUM_EPOCHS,TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
    exports+=",DATA_FRACTION=$DATA_FRACTION,OPTIMIZER=$OPTIMIZER,ENSEMBLE_MODE=$ENSEMBLE_MODE"
    exports+=",VAL_EVERY_N_STEPS=$VAL_EVERY_N_STEPS,COMPILE_MODE=$COMPILE_MODE"
    exports+=",COMPLETEP=$COMPLETEP,NO_VE_PROJS=$NO_VE_PROJS,NO_WARMDOWN=$NO_WARMDOWN"
    exports+=",WEIGHT_DECAY=$WEIGHT_DECAY"
    exports+=",MUP_BASE_WIDTH=$MUP_BASE_WIDTH,MUP_BASE_DEPTH=$MUP_BASE_DEPTH,MUP_BASE_HEAD_DIM=$MUP_BASE_HEAD_DIM"
    exports+=",CHECKPOINT_EVERY_N_STEPS=$CHECKPOINT_EVERY_N_STEPS,CHECKPOINT_BASE=$CKPT_BASE"

    sb=(--parsable --account=$ACCOUNT --gpus=$GPU_SPEC --time="$TL"
        --array=0 --job-name="fd_widthext_d${L}_w${W}" --export="$exports")
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY: sbatch ${sb[*]} experiments/parallel/train_array.sh"
    else
        JOB=$(sbatch "${sb[@]}" experiments/parallel/train_array.sh)
        echo "submitted d${L}/w${W}: job=$JOB  time=$TL  group=$GROUP  ckpt=$CKPT_BASE"
    fi
done

echo "Width-extension submission done. GRID_TAG=$GRID_TAG"
