#!/bin/bash
# CONTROL: lambda = 0 WITH cooldown, at d{12,24,48}/w768, df=1.0.
#
# Why: the capacity figure currently contrasts
#     (a) lambda=0 + constant LR      vs   (b) tuned lambda + cooldown
# which differ in TWO things at once, so the ~0.34 nat gap at d12 cannot be
# attributed to weight decay alone. This run supplies the missing corner —
# lambda=0 + cooldown — turning the comparison into a clean decomposition:
#     schedule effect = (a) - (control)
#     weight-decay effect = (control) - (b)
#
# No such point exists today: `launch_wd_sweep.sh` did run wd=0 at d12, but under
# constant LR (job 41609119: min at 22% of the run, then +0.92 nats of overfit —
# the constant-LR signature, not the cooldown one).
#
# Recipe identical to launch_wd_size{,_fill}.sh except WEIGHT_DECAY=0:
# cooldown ON, df=1.0, 40 epochs, completep, no-ve-projs, adamw, batch 131072,
# E=1 single model at index 0, per-epoch ckpts only.
#
# Cost (measured runtimes x 2 SU/H100-hr): d12 ~3.5h, d24 ~8.9h, d48 ~16.5h
#   -> ~58 SU total.
#
#   DRY_RUN=1 bash experiments/parallel/launch_wd0_cooldown.sh
#   bash experiments/parallel/launch_wd0_cooldown.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

GRID_TAG="${GRID_TAG:-wd0cd_$(date +%Y%m%d)}"
DRY_RUN="${DRY_RUN:-0}"

ACCOUNT="${ACCOUNT:-cis260161p}"
GPU_SPEC=h100-80:1
COMPILE_MODE=inductor
COMPLETEP=1
NO_VE_PROJS=1
NO_WARMDOWN=0                  # COOLDOWN ON -- the point of this control
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
WD=0.0

CKPT_BASE="${CKPT_BASE:-/ocean/projects/cis260161p/ymiao6/scaling/slowrun/checkpoints}"
mkdir -p "$CKPT_BASE" experiments/logs

declare -a LAYERS=(12         24         48)
declare -a TIMES=("08:00:00" "18:00:00" "30:00:00")
W=768; H=12

echo "=== lambda=0 + cooldown control  GRID_TAG=$GRID_TAG  account=$ACCOUNT ==="

for i in "${!LAYERS[@]}"; do
    L=${LAYERS[$i]}; TL=${TIMES[$i]}
    CELL_TS="${GRID_TAG}_d${L}_w${W}_wd${WD}"
    GROUP="$CELL_TS"
    mkdir -p "$CKPT_BASE/parallel_init_ens_${CELL_TS}"

    exports="ALL,N_LAYER=$L,N_HEAD=$H,N_EMBD=$W,SHARED_TIMESTAMP=$CELL_TS,WANDB_GROUP=$GROUP"
    exports+=",NUM_MODELS=$NUM_MODELS,NUM_EPOCHS=$NUM_EPOCHS,TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
    exports+=",DATA_FRACTION=$DATA_FRACTION,OPTIMIZER=$OPTIMIZER,ENSEMBLE_MODE=$ENSEMBLE_MODE"
    exports+=",VAL_EVERY_N_STEPS=$VAL_EVERY_N_STEPS,COMPILE_MODE=$COMPILE_MODE"
    exports+=",COMPLETEP=$COMPLETEP,NO_VE_PROJS=$NO_VE_PROJS,NO_WARMDOWN=$NO_WARMDOWN"
    exports+=",WEIGHT_DECAY=$WD"
    exports+=",MUP_BASE_WIDTH=$MUP_BASE_WIDTH,MUP_BASE_DEPTH=$MUP_BASE_DEPTH,MUP_BASE_HEAD_DIM=$MUP_BASE_HEAD_DIM"
    exports+=",CHECKPOINT_EVERY_N_STEPS=$CHECKPOINT_EVERY_N_STEPS,CHECKPOINT_BASE=$CKPT_BASE"

    # distinct job-name prefix so the analysis can tell this control apart from
    # the tuned-wd probes, which all use the wdsize_ prefix
    sb=(--parsable --account=$ACCOUNT --gpus=$GPU_SPEC --time="$TL"
        --array=0 --job-name="wd0cd_d${L}_w${W}_wd${WD}" --export="$exports")
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY: sbatch ${sb[*]} experiments/parallel/train_array.sh"
    else
        JOB=$(sbatch "${sb[@]}" experiments/parallel/train_array.sh)
        echo "submitted d${L}/w${W} wd=${WD} (cooldown): job=$JOB  time=$TL"
    fi
done

echo "control submitted. GRID_TAG=$GRID_TAG  ckpt=$CKPT_BASE"
