#!/bin/bash
# Is the non-standard residual machinery worth its cost?
#
# The x0 injection (x0_lambdas) and the U-Net skips (skip_weights) are the two
# paths that bypass CompleteP's depth_scale and break depth HP transfer:
# residual-stream RMS grows 5.45x (toy) / 7.21x (production, real FineWeb) from
# the shallowest to the deepest cell, versus EXACTLY 1.00 once both are ablated.
#
# If a plain pre-LN transformer (both paths off, plus the already-default
# --no-ve-projs) reaches comparable validation loss, we can drop them and do the
# whole scaling study on an architecture CompleteP is actually derived for --
# removing the depth-transfer confound at its source rather than correcting it.
#
# Phase 1 (this script): d12/w768, the BASE depth. Chosen deliberately -- at
# L = L_base = 12 the depth factor is 1.0 both pre- and post-fix, so the
# full-architecture reference numbers are themselves uncontaminated. This
# isolates "do these paths help performance?" from "do they break transfer?".
#
# Reference, full architecture, identical settings (d12/w768, cooldown, df=1.0):
#     lambda=0.1 -> 3.5733    lambda=0.15 -> 3.5487    lambda=0.2 -> 3.5537
# lambda is re-swept because the optimum need not be the same without those paths.
#
# Cost: 3 x ~5.1h x 2 SU/hr ~= 31 SU.
#
#   DRY_RUN=1 bash experiments/parallel/launch_plain_resid.sh
#   bash experiments/parallel/launch_plain_resid.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

GRID_TAG="${GRID_TAG:-plainresid_$(date +%Y%m%d)}"
DRY_RUN="${DRY_RUN:-0}"
ACCOUNT="${ACCOUNT:-cis260161p}"
GPU_SPEC=h100-80:1
COMPILE_MODE=inductor
COMPLETEP=1
NO_VE_PROJS=1
PLAIN_RESID=1                  # <-- the point of this experiment
NO_WARMDOWN=0                  # cooldown ON, matching the wdsize probes
TOTAL_BATCH_SIZE=131072
NUM_MODELS=5
NUM_EPOCHS=40
DATA_FRACTION=1.0
OPTIMIZER=adamw
ENSEMBLE_MODE=logit
VAL_EVERY_N_STEPS=152
CHECKPOINT_EVERY_N_STEPS=0
MUP_BASE_WIDTH=768; MUP_BASE_DEPTH=12; MUP_BASE_HEAD_DIM=64

CKPT_BASE="${CKPT_BASE:-/ocean/projects/cis260161p/ymiao6/scaling/slowrun/checkpoints}"
mkdir -p "$CKPT_BASE" experiments/logs

L=12; W=768; H=12
for WD in ${WDLIST:-0.1 0.15 0.2}; do
    CELL_TS="${GRID_TAG}_d${L}_w${W}_wd${WD}"
    mkdir -p "$CKPT_BASE/parallel_init_ens_${CELL_TS}"
    exports="ALL,N_LAYER=$L,N_HEAD=$H,N_EMBD=$W,SHARED_TIMESTAMP=$CELL_TS,WANDB_GROUP=$CELL_TS"
    exports+=",NUM_MODELS=$NUM_MODELS,NUM_EPOCHS=$NUM_EPOCHS,TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
    exports+=",DATA_FRACTION=$DATA_FRACTION,OPTIMIZER=$OPTIMIZER,ENSEMBLE_MODE=$ENSEMBLE_MODE"
    exports+=",VAL_EVERY_N_STEPS=$VAL_EVERY_N_STEPS,COMPILE_MODE=$COMPILE_MODE"
    exports+=",COMPLETEP=$COMPLETEP,NO_VE_PROJS=$NO_VE_PROJS,PLAIN_RESID=$PLAIN_RESID"
    exports+=",NO_WARMDOWN=$NO_WARMDOWN,WEIGHT_DECAY=$WD"
    exports+=",MUP_BASE_WIDTH=$MUP_BASE_WIDTH,MUP_BASE_DEPTH=$MUP_BASE_DEPTH,MUP_BASE_HEAD_DIM=$MUP_BASE_HEAD_DIM"
    exports+=",CHECKPOINT_EVERY_N_STEPS=$CHECKPOINT_EVERY_N_STEPS,CHECKPOINT_BASE=$CKPT_BASE"
    sb=(--parsable --account=$ACCOUNT --gpus=$GPU_SPEC --time=08:00:00
        --array=0 --job-name="plain_d${L}_w${W}_wd${WD}" --export="$exports")
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY: sbatch ${sb[*]} experiments/parallel/train_array.sh"
    else
        JOB=$(sbatch "${sb[@]}" experiments/parallel/train_array.sh)
        echo "submitted PLAIN d${L}/w${W} wd=${WD}: job=$JOB"
    fi
done
echo "GRID_TAG=$GRID_TAG  (compare against full-arch d12/w768: 0.1->3.5733 0.15->3.5487 0.2->3.5537)"
