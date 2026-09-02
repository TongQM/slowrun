#!/bin/bash
# Deep-depth cells to complete the DEPTH Figure-2 matched-compute comparison:
#   d48/w768 (4x compute -> matches ensemble E=4)
#   d60/w768 (5x compute -> matches ensemble E=5)
# Extends the depth curve {d6,d12,d18,d24} up to the E=4/E=5 compute points so the
# depth-vs-ensemble Delta can be read at 4x (parallel to the WIDTH figure's 4x Delta).
#
# E=1 single-model, no replay/deps -> parallel. Recipe identical to the 2026-07-06
# grid-fill (df=1.0, wd=0, no-warmdown, completep, no-ve-projs, adamw, batch 131072,
# n_head=12 -> head_dim=64, muP base w768/d12/h64, 40 epochs). Job-name fd_gridfill_*
# so the depth plotter globs it. Compute cis260161p; ckpts -> cis260095p (deletable).
#
#   DRY_RUN=1 bash experiments/parallel/launch_depth_deep.sh   # preview
#   bash experiments/parallel/launch_depth_deep.sh             # submit
set -euo pipefail
cd "$(dirname "$0")/../.."

GRID_TAG="${GRID_TAG:-fulldata_gridfill_20260706}"
DRY_RUN="${DRY_RUN:-0}"

ACCOUNT=cis260161p
GPU_SPEC=h100-80:1
COMPILE_MODE=inductor
COMPLETEP=1
NO_VE_PROJS=1
NO_WARMDOWN=1
WEIGHT_DECAY=0
TOTAL_BATCH_SIZE=131072
NUM_MODELS=5
NUM_EPOCHS=40
DATA_FRACTION=1.0
OPTIMIZER=adamw
ENSEMBLE_MODE=logit
VAL_EVERY_N_STEPS=152
CHECKPOINT_EVERY_N_STEPS=0
MUP_BASE_WIDTH=768
MUP_BASE_DEPTH=12
MUP_BASE_HEAD_DIM=64

CKPT_BASE=/ocean/projects/cis260095p/ymiao6/scaling/slowrun/checkpoints
mkdir -p "$CKPT_BASE" experiments/logs

# (n_layer, n_embd, n_head, walltime) — 4x and 5x compute
declare -a LAYERS=(48   60)
declare -a EMBDS=( 768  768)
declare -a HEADS=( 12   12)
declare -a TIMES=("30:00:00" "40:00:00")

for i in "${!LAYERS[@]}"; do
    L=${LAYERS[$i]}; W=${EMBDS[$i]}; H=${HEADS[$i]}; TL=${TIMES[$i]}
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
        --array=0 --job-name="fd_gridfill_d${L}_w${W}" --export="$exports")
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY: sbatch ${sb[*]} experiments/parallel/train_array.sh"
    else
        JOB=$(sbatch "${sb[@]}" experiments/parallel/train_array.sh)
        echo "submitted d${L}/w${W} (n_head=$H): job=$JOB  time=$TL  group=$GROUP"
    fi
done

echo "Deep-depth submission done. GRID_TAG=$GRID_TAG  ckpt=$CKPT_BASE"
