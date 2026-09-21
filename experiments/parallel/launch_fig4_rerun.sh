#!/bin/bash
# Figure 4 rerun: ensemble-size sweep up to E=20 at L12/W768, P=20M (df=0.2), in the
# dynamics recipe of every other figure: lambda=0, constant LR, 40 epochs, corrected
# CompleteP. 20 individuals x 2 strategies; fused replays at E in {2,5,10,15,20};
# 10 bootstrap resamples per strategy. Per-epoch checkpoints are kept (546 MB each,
# 40 x 40 = ~875 GB); NO cleanup job is submitted.
#
# Usage:  bash experiments/parallel/launch_fig4_rerun.sh        (DRY_RUN=1 to preview)
set -euo pipefail
cd "$(dirname "$0")/../.."

GRID_TAG="${GRID_TAG:-fig4_$(date +%Y%m%d)}"
DRY_RUN="${DRY_RUN:-0}"

ACCOUNT=cis260161p
GPU_SPEC=h100-80:1
COMPILE_MODE=inductor
COMPLETEP=1
NO_VE_PROJS=1
NO_WARMDOWN=1
TOTAL_BATCH_SIZE=131072
NUM_MODELS=20
NUM_EPOCHS=40
WEIGHT_DECAY=0.0
DATA_FRACTION=0.2
VAL_EVERY_N_STEPS=152
CHECKPOINT_EVERY_N_STEPS=0       # per-epoch ckpts only; at df=0.2, 1 epoch = 20M tokens already
PERMANENT_EVERY_N_STEPS=0
PERMANENT_EVERY_N_EPOCHS=1
OPTIMIZER=adamw
ENSEMBLE_MODE=logit
MUP_BASE_WIDTH=768
MUP_BASE_DEPTH=12
MUP_BASE_HEAD_DIM=64
ENS_SIZES_STR="2 5 10 15 20"

DEST=/ocean/projects/cis260161p/ymiao6/scaling/slowrun/checkpoints
mkdir -p "$DEST" experiments/logs

L=12; H=12; W=768
df="$DATA_FRACTION"
CELL_TS="${GRID_TAG}_d${L}_w${W}_df${df}"
GROUP="fig4_ensemble_size_${GRID_TAG}_d${L}_w${W}_df${df}"

mkdir -p "$DEST/parallel_init_ens_${CELL_TS}" "$DEST/parallel_init_shuffle_ens_${CELL_TS}"

echo "============================================================"
echo "Q2 ensemble-size sweep: d${L}/w${W}, df=${df}, N=${NUM_MODELS}"
echo "  NUM_EPOCHS=$NUM_EPOCHS, ens_sizes={$ENS_SIZES_STR}"
echo "  ACCOUNT=$ACCOUNT, DEST=$DEST"
echo "  CompleteP=$COMPLETEP, no_ve_projs=$NO_VE_PROJS, no_warmdown=$NO_WARMDOWN (constant LR)"
echo "============================================================"

submit_one() {
    local name=$1 dep=$2 timelim=$3 arr=$4 exports=$5 script=$6
    local args=(--parsable --account=$ACCOUNT --gpus=$GPU_SPEC --time="$timelim"
                --array=$arr --job-name=$name --export="$exports")
    if [ -n "$dep" ]; then args+=("$dep"); fi
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY: sbatch ${args[*]} $script" >&2
        echo "DRYJOB$RANDOM"; return
    fi
    sbatch "${args[@]}" "$script"
}

NUM_SIZES=$(echo $ENS_SIZES_STR | wc -w)

exports="ALL,N_LAYER=$L,N_HEAD=$H,N_EMBD=$W,SHARED_TIMESTAMP=$CELL_TS,WANDB_GROUP=$GROUP"
exports+=",NUM_MODELS=$NUM_MODELS,NUM_EPOCHS=$NUM_EPOCHS,TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
exports+=",DATA_FRACTION=$df,OPTIMIZER=$OPTIMIZER,ENSEMBLE_MODE=$ENSEMBLE_MODE"
exports+=",VAL_EVERY_N_STEPS=$VAL_EVERY_N_STEPS,COMPILE_MODE=$COMPILE_MODE"
exports+=",COMPLETEP=$COMPLETEP,NO_VE_PROJS=$NO_VE_PROJS,NO_WARMDOWN=$NO_WARMDOWN"
exports+=",MUP_BASE_WIDTH=$MUP_BASE_WIDTH,MUP_BASE_DEPTH=$MUP_BASE_DEPTH,MUP_BASE_HEAD_DIM=$MUP_BASE_HEAD_DIM"
exports+=",CHECKPOINT_EVERY_N_STEPS=$CHECKPOINT_EVERY_N_STEPS,CHECKPOINT_BASE=$DEST"
exports+=",PERMANENT_EVERY_N_STEPS=$PERMANENT_EVERY_N_STEPS,PERMANENT_EVERY_N_EPOCHS=$PERMANENT_EVERY_N_EPOCHS"
exports+=",ENS_SIZES_STR=$ENS_SIZES_STR,SKIP_INDIV_VAL=1,END_EPOCH=$NUM_EPOCHS,WEIGHT_DECAY=$WEIGHT_DECAY"

# 20 ind × 2 strats = 40 train tasks.
TRAIN_RANGE="0-$((2*NUM_MODELS - 1))"
REPLAY_RANGE="${REPLAY_RANGE:-0-$((2*NUM_SIZES - 1))}"
BOOT_RANGE="${BOOT_RANGE:-0-19}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"   # 1: training already done; resubmit replays/bootstraps only
CLEANUP_RANGE="0-1"

DEP=""
if [ "$SKIP_TRAIN" != "1" ]; then
TJOB=$(submit_one "fig4_train_d12_w768" "" "06:00:00" "$TRAIN_RANGE" "$exports" experiments/parallel/train_array.sh)
echo "  train  array=$TRAIN_RANGE  job=$TJOB  group=$GROUP" >&2
DEP="--dependency=afterok:$TJOB"
fi

RJOB=$(submit_one "fig4_replay_d12_w768" "$DEP" "12:00:00" "$REPLAY_RANGE" "$exports" experiments/parallel/replay_array.sh)
echo "  replay array=$REPLAY_RANGE  job=$RJOB  ${DEP:+($DEP)}" >&2

bexports="$exports,CKPT_PREFIX=$DEST,CKPT_TAG=${CELL_TS},BOOT_OUT=experiments/figures/02_ensemble_scaling/bootstrap_${GRID_TAG}"
BJOB=$(submit_one "fig4_bootstrap_d12_w768" "$DEP" "10:00:00" "$BOOT_RANGE" "$bexports" experiments/parallel/replay_bootstrap.sh)
echo "  boot   array=0-19  job=$BJOB  ${DEP:+($DEP)}" >&2

echo
echo "Figure 4 rerun submitted. Wandb group: $GROUP"
