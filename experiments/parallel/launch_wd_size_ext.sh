#!/bin/bash
# P1-EXT: resolve the d48 weight-decay optimum, which sat at the EDGE of the P1 grid.
#
# P1 swept d48/w768 over wd {0.1, 0.2, 0.3} and found L* monotone DECREASING to the
# boundary (3.5378 / 3.4751 / 3.4667) -- so the true optimum is >= 0.3, unresolved.
#
# Hypothesis under test: the optimal decoupled lambda scales with depth as
#     lambda_opt(L) = lambda_base * L / L_base,   lambda_base ~= 0.1, L_base = 12
# i.e. the SAME L_base/L factor the model already applies to the residual branch
# (unlimited/train.py:469), but which is NOT applied to AdamW's weight decay.
# Fit so far: d12 -> 0.1 (measured), d24 -> 0.2 (measured, exact hit).
#
#   PREDICTION for d48: lambda_opt = 0.4, and wd=0.5 turns back UP.
#   FALSIFIED if wd=0.4 is no better than the wd=0.3 already measured (3.4667),
#   which would mean 0.3 was the optimum and the drift is a genuine capacity effect.
#
# Across-seed sigma of L* at this cell size is 0.0072 (measured, n=5 at d12/w768) and
# the predicted effects are 0.03-0.07, so single seeds are sufficient.
#
# Config is IDENTICAL to experiments/parallel/launch_wd_size.sh except WDS and LAYERS,
# and reuses the same GRID_TAG so log names / checkpoint dirs / wandb groups stay in
# the same family and the existing parsers pick these up unchanged. No training-code
# changes: weight decay is passed through the existing WEIGHT_DECAY env var.
#
#   DRY_RUN=1 bash experiments/parallel/launch_wd_size_ext.sh   # preview
#   bash experiments/parallel/launch_wd_size_ext.sh             # submit
set -euo pipefail
cd "$(dirname "$0")/../.."

GRID_TAG="${GRID_TAG:-wdsize_20260717}"   # same family as P1
DRY_RUN="${DRY_RUN:-0}"

ACCOUNT="${ACCOUNT:-cis260161p}"
GPU_SPEC=h100-80:1
COMPILE_MODE=inductor
COMPLETEP=1
NO_VE_PROJS=1
NO_WARMDOWN=0                  # COOLDOWN ON -- matches P1 and the base wd sweep
TOTAL_BATCH_SIZE=131072
NUM_MODELS=5
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

# (n_layer, n_embd, n_head, walltime) -- d48 only; P1 measured it at 16.5h, keep 30h
declare -a LAYERS=(48)
declare -a EMBDS=( 768)
declare -a HEADS=( 12)
declare -a TIMES=("30:00:00")
declare -a WDS=(0.4 0.5)

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

echo "wd-size EXT submitted. GRID_TAG=$GRID_TAG  account=$ACCOUNT  ckpt=$CKPT_BASE"
