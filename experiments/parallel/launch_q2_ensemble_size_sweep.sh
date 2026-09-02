#!/bin/bash
# Q2: Ensemble-size sweep up to N=20 at d12/w768, df=0.2.
# 20 individuals × 2 strategies × 25 epochs, fresh from scratch (constant LR
# matters for comparability with df=0.4 grid; can't reuse the warmdown-LR
# 5-individual data from grid_20260430_152533).
#
# Replays at ensemble sizes {2, 5, 10, 15, 20} per strategy.
# CompleteP + no-ve-projs + constant LR throughout.
# Charged to cis260161p, stored on cis260095p (transient peak ~500 GB).
# Cleanup keeps per-epoch ckpts at every 5th epoch.
#
# Usage:
#   bash experiments/parallel/launch_q2_ensemble_size_sweep.sh
#   GRID_TAG=<custom> bash experiments/parallel/launch_q2_ensemble_size_sweep.sh
#   DRY_RUN=1 bash ...

set -euo pipefail
cd "$(dirname "$0")/../.."

GRID_TAG="${GRID_TAG:-q2_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

ACCOUNT=cis260161p
GPU_SPEC=h100-80:1
COMPILE_MODE=inductor
COMPLETEP=1
NO_VE_PROJS=1
NO_WARMDOWN=1
TOTAL_BATCH_SIZE=131072
NUM_MODELS=20
NUM_EPOCHS=25
DATA_FRACTION=0.2
VAL_EVERY_N_STEPS=152
CHECKPOINT_EVERY_N_STEPS=0       # per-epoch ckpts only; at df=0.2, 1 epoch = 20M tokens already
PERMANENT_EVERY_N_STEPS=0
PERMANENT_EVERY_N_EPOCHS=5
OPTIMIZER=adamw
ENSEMBLE_MODE=logit
MUP_BASE_WIDTH=768
MUP_BASE_DEPTH=12
MUP_BASE_HEAD_DIM=64
ENS_SIZES_STR="2 5 10 15 20"

CIS095=/ocean/projects/cis260095p/ymiao6/scaling/slowrun/checkpoints
DEST="$CIS095"
mkdir -p "$DEST" experiments/logs

L=12; H=12; W=768
df="$DATA_FRACTION"
CELL_TS="${GRID_TAG}_d${L}_w${W}_df${df}"
GROUP="q2_ensemble_size_${GRID_TAG}_d${L}_w${W}_df${df}"

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
exports+=",ENS_SIZES_STR=$ENS_SIZES_STR,SKIP_INDIV_VAL=1,END_EPOCH=$NUM_EPOCHS"

# 20 ind × 2 strats = 40 train tasks.
TRAIN_RANGE="0-$((2*NUM_MODELS - 1))"
REPLAY_RANGE="0-$((2*NUM_SIZES - 1))"
CLEANUP_RANGE="0-1"

TJOB=$(submit_one "q2_train_d12_w768_df${df}" "" "06:00:00" "$TRAIN_RANGE" "$exports" experiments/parallel/train_array.sh)
echo "  train  array=$TRAIN_RANGE  job=$TJOB  group=$GROUP" >&2

RJOB=$(submit_one "q2_replay_d12_w768_df${df}" "--dependency=afterok:$TJOB" "08:00:00" "$REPLAY_RANGE" "$exports" experiments/parallel/replay_array.sh)
echo "  replay array=$REPLAY_RANGE  job=$RJOB  (after $TJOB)" >&2

CJOB=$(submit_one "q2_cleanup_d12_w768_df${df}" "--dependency=afterok:$RJOB" "00:30:00" "$CLEANUP_RANGE" "$exports" experiments/parallel/cleanup_array.sh)
echo "  clean  array=$CLEANUP_RANGE  job=$CJOB  (after $RJOB)" >&2

echo
echo "Q2 submission complete. Wandb group: $GROUP"
