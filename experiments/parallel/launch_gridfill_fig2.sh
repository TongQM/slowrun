#!/bin/bash
# Grid-fill cells for (a) the DEPTH version of Figure 2 and (b) the d6 line of the
# val-loss-vs-params figure. All E=1 single-model runs -> no replay, no dependency,
# fully parallel.
#
#   (a) depth Fig-2 : fixed w768, add d18 (1.5x compute) + d24 (2x) to the existing
#                     d6/w768, d12/w768. n_head=12 -> head_dim=64 (matches the w768
#                     column). Ensemble baseline (d12/w768 E1-5) already exists.
#   (b) Q1 d6 line  : fixed d6, add w1152 + w1536 to the existing d6/w384, d6/w768.
#                     n_head = w/64 -> head_dim=64 (matches the d6 row convention).
#
# Recipe matches fulldata_20260528_150057 EXACTLY: df=1.0, wd=0, no-warmdown,
# completep, no-ve-projs, adamw, batch 131072, muP base w768/d12/h64, 40 epochs.
# Train ONE init model (array=0, NUM_MODELS=5 for seed parity). No step ckpts;
# per-epoch ckpts only (resume safety) -> cis260095p, deletable after extraction.
# Compute on cis260161p.
#
#   DRY_RUN=1 bash experiments/parallel/launch_gridfill_fig2.sh   # preview
#   bash experiments/parallel/launch_gridfill_fig2.sh             # submit
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
NUM_MODELS=5            # virtual ensemble size (seed parity); train only model 0
NUM_EPOCHS=40
DATA_FRACTION=1.0
OPTIMIZER=adamw
ENSEMBLE_MODE=logit
VAL_EVERY_N_STEPS=152
CHECKPOINT_EVERY_N_STEPS=0     # no step ckpts (curve needs only logged val)
MUP_BASE_WIDTH=768
MUP_BASE_DEPTH=12
MUP_BASE_HEAD_DIM=64

CKPT_BASE=/ocean/projects/cis260095p/ymiao6/scaling/slowrun/checkpoints
mkdir -p "$CKPT_BASE" experiments/logs

# per-cell table: (n_layer, n_embd, n_head, walltime)
declare -a LAYERS=(18   24   6    6)
declare -a EMBDS=( 768  768  1152 1536)
declare -a HEADS=( 12   12   18   24)
declare -a TIMES=("12:00:00" "18:00:00" "12:00:00" "18:00:00")

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

    # widthext naming for w768? no -> use fd_gridfill_* so the plotters can glob it.
    sb=(--parsable --account=$ACCOUNT --gpus=$GPU_SPEC --time="$TL"
        --array=0 --job-name="fd_gridfill_d${L}_w${W}" --export="$exports")
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY: sbatch ${sb[*]} experiments/parallel/train_array.sh"
    else
        JOB=$(sbatch "${sb[@]}" experiments/parallel/train_array.sh)
        echo "submitted d${L}/w${W} (n_head=$H): job=$JOB  time=$TL  group=$GROUP"
    fi
done

echo "Grid-fill submission done. GRID_TAG=$GRID_TAG  ckpt=$CKPT_BASE"
