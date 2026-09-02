#!/bin/bash
# P1: does the optimal weight decay TRANSFER across model size? (wd x size probe)
#
# The base-cell wd sweep found wd_opt=0.1 (cooldown ON) at d12/w768. Bigger models
# overfit the fixed 100M tokens harder, so wd_opt may drift UP with size. We probe
# this along the DEPTH axis (the only size axis that helps at 100M; width scaling is
# flat-to-harmful in the data-limited regime) at two depths that bracket the region
# the compute-optimal frontier lands in:
#   d24/w768 (2x compute)  and  d48/w768 (4x compute)
# each swept over wd in {0.1, 0.2, 0.3}, E=1 single model. 2 x 3 = 6 jobs.
#
# Recipe = the base wd sweep EXACTLY except size: COOLDOWN ON (warmdown_ratio=0.2,
# final_lr_frac=0 -> train.py defaults; we simply do NOT pass --no-warmdown), df=1.0,
# completep, no-ve-projs, adamw, batch 131072, lr_multiplier 0.25 (default), 40 ep.
# So delta(wd) here is directly comparable to the base-cell delta(wd).
#
#   DRY_RUN=1 bash experiments/parallel/launch_wd_size.sh   # preview
#   bash experiments/parallel/launch_wd_size.sh             # submit
set -euo pipefail
cd "$(dirname "$0")/../.."

GRID_TAG="${GRID_TAG:-wdsize_20260717}"
DRY_RUN="${DRY_RUN:-0}"

ACCOUNT="${ACCOUNT:-cis260161p}"
GPU_SPEC=h100-80:1
COMPILE_MODE=inductor
COMPLETEP=1
NO_VE_PROJS=1
NO_WARMDOWN=0                  # <-- COOLDOWN ON (the whole point vs the fd grid)
TOTAL_BATCH_SIZE=131072
NUM_MODELS=5                   # virtual ensemble size (seed parity); train model 0 only
NUM_EPOCHS=40
DATA_FRACTION=1.0
OPTIMIZER=adamw
ENSEMBLE_MODE=logit
VAL_EVERY_N_STEPS=152
CHECKPOINT_EVERY_N_STEPS=0     # per-epoch ckpts only (resume safety), deletable
MUP_BASE_WIDTH=768
MUP_BASE_DEPTH=12
MUP_BASE_HEAD_DIM=64

CKPT_BASE=/ocean/projects/cis260095p/ymiao6/scaling/slowrun/checkpoints
mkdir -p "$CKPT_BASE" experiments/logs

# (n_layer, n_embd, n_head, walltime)
declare -a LAYERS=(24   48)
declare -a EMBDS=( 768  768)
declare -a HEADS=( 12   12)
declare -a TIMES=("18:00:00" "30:00:00")
declare -a WDS=(0.1 0.2 0.3)

for i in "${!LAYERS[@]}"; do
    L=${LAYERS[$i]}; W=${EMBDS[$i]}; H=${HEADS[$i]}; TL=${TIMES[$i]}
    for WD in "${WDS[@]}"; do
        CELL_TS="${GRID_TAG}_d${L}_w${W}_wd${WD}"
        GROUP="${GRID_TAG}_d${L}_w${W}_wd${WD}"
        mkdir -p "$CKPT_BASE/parallel_init_ens_${CELL_TS}"

        exports="ALL,N_LAYER=$L,N_HEAD=$H,N_EMBD=$W,SHARED_TIMESTAMP=$CELL_TS,WANDB_GROUP=$GROUP"
        exports+=",NUM_MODELS=$NUM_MODELS,NUM_EPOCHS=$NUM_EPOCHS,TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
        exports+=",DATA_FRACTION=$DATA_FRACTION,OPTIMIZER=$OPTIMIZER,ENSEMBLE_MODE=$ENSEMBLE_MODE"
        exports+=",VAL_EVERY_N_STEPS=$VAL_EVERY_N_STEPS,COMPILE_MODE=$COMPILE_MODE"
        exports+=",COMPLETEP=$COMPLETEP,NO_VE_PROJS=$NO_VE_PROJS,NO_WARMDOWN=$NO_WARMDOWN"
        exports+=",WEIGHT_DECAY=$WD"
        exports+=",MUP_BASE_WIDTH=$MUP_BASE_WIDTH,MUP_BASE_DEPTH=$MUP_BASE_DEPTH,MUP_BASE_HEAD_DIM=$MUP_BASE_HEAD_DIM"
        exports+=",CHECKPOINT_EVERY_N_STEPS=$CHECKPOINT_EVERY_N_STEPS,CHECKPOINT_BASE=$CKPT_BASE"

        sb=(--parsable --account=$ACCOUNT --gpus=$GPU_SPEC --time="$TL"
            --array=0 --job-name="wdsize_d${L}_w${W}_wd${WD}" --export="$exports")
        if [ "$DRY_RUN" = "1" ]; then
            echo "DRY: sbatch ${sb[*]} experiments/parallel/train_array.sh"
        else
            JOB=$(sbatch "${sb[@]}" experiments/parallel/train_array.sh)
            echo "submitted d${L}/w${W} wd=${WD}: job=$JOB  time=$TL  group=$GROUP"
        fi
    done
done

echo "wd-size probe submitted. GRID_TAG=$GRID_TAG  account=$ACCOUNT  ckpt=$CKPT_BASE"
