#!/bin/bash
# launch_wd_sweep.sh — weight-decay sweep for ONE cell + ONE strategy, storage-safe.
#
# Pilots the effect of AdamW weight decay on multi-epoch overfit. For each WD value
# it submits a train -> replay -> cleanup chain, and chains the WD values
# *sequentially* (each WD's training waits afterany on the previous WD's cleanup).
# Because cleanup deletes a finished run's checkpoints before the next run starts,
# the peak on-disk footprint is ONE run (~640 GB for d12_w768) instead of the sum.
#
# Default scope: d12/w768, init_shuffle_ens only, df=1.0, 40 epochs, 20M resolution
# (checkpoint/val every 152 steps), E in {2,3,4,5}. Same as the wd=0 baseline grid
# except --weight-decay. The val-loss data is parsed from the SLURM .out logs by the
# export-valloss skill (NOT from checkpoints), so deleting checkpoints loses nothing.
#
# Storage: routed to cis260095p (846 GB free). Compute charged to cis260095p too
# (cis260009p access was revoked 2026-06). cis260161p holds the durable baseline+data.
#
# Usage:
#   bash experiments/parallel/launch_wd_sweep.sh                 # submit wd in {0.1,0.3,1.0}
#   DRY_RUN=1 bash experiments/parallel/launch_wd_sweep.sh       # print sbatch lines only
#   WD_LIST="0.1 0.5" bash experiments/parallel/launch_wd_sweep.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

# ---- scope (overridable) ----
WD_LIST="${WD_LIST:-0.1 0.3 1.0}"
N_LAYER="${N_LAYER:-12}"; N_HEAD="${N_HEAD:-12}"; N_EMBD="${N_EMBD:-768}"
DF="${DF:-1.0}"
NUM_EPOCHS="${NUM_EPOCHS:-40}"
NUM_MODELS="${NUM_MODELS:-5}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-131072}"
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-152}"   # ~20M tokens
VAL_EVERY_N_STEPS="${VAL_EVERY_N_STEPS:-152}"
ENS_SIZES_STR="${ENS_SIZES_STR:-2 3 4 5}"
EVAL_MODE="${EVAL_MODE:-step}"                                # 20M-resolution ensemble replay
ENSEMBLE_MODE="${ENSEMBLE_MODE:-logit}"
OPTIMIZER="${OPTIMIZER:-adamw}"
# project always-on defaults
COMPLETEP=1; NO_VE_PROJS=1; NO_WARMDOWN="${NO_WARMDOWN:-1}"; COMPILE_MODE="${COMPILE_MODE:-inductor}"
MUP_BASE_WIDTH=768; MUP_BASE_DEPTH=12; MUP_BASE_HEAD_DIM=64
# init_shuffle_ens only: train strategy-idx 1 -> array 5..9 ; replay array task 1
TRAIN_ARRAY="$(( NUM_MODELS ))-$(( NUM_MODELS * 2 - 1 ))"     # 5-9
REPLAY_ARRAY="1"
# SLURM / storage
ACCOUNT="${ACCOUNT:-cis260095p}"                              # compute (cis260009p revoked 2026-06)
GPU_SPEC="${GPU_SPEC:-h100-80:1}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-checkpoints}"             # storage on cis260161p (3.4 TB free)
TRAIN_TIME="${TRAIN_TIME:-06:00:00}"
# fresh step-replay over all 200 20M-points @ ~224 s/point ~= 12.4 h; budget 14 h.
# (extend mode with START_STEP recomputes only the new region -> can be much shorter.)
REPLAY_TIME="${REPLAY_TIME:-14:00:00}"
# CLEANUP=1 deletes each run's ckpts after replay AND chains WD values sequentially
# (peak ~1 run on disk). CLEANUP=0 (default; safe now that cis260161p has 3.4 TB free)
# runs all WD values concurrently and leaves ckpts for manual pruning after export.
CLEANUP="${CLEANUP:-0}"
TAG="${TAG:-wdsweep_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

echo "============================================================"
echo "WD sweep  TAG=$TAG"
echo "  cell d${N_LAYER}/w${N_EMBD}  strategy=init_shuffle_ens  df=$DF  epochs=$NUM_EPOCHS"
echo "  wd_list=[$WD_LIST]  res=${CHECKPOINT_EVERY_N_STEPS}step(~20M)  eval_mode=$EVAL_MODE"
echo "  account=$ACCOUNT  ckpt_base=$CHECKPOINT_BASE"
if [ "$CLEANUP" = "1" ]; then
    echo "  CLEANUP=1: sequential chain + rm-after-replay => peak ~1 run on disk"
else
    echo "  CLEANUP=0: all WD values run concurrently; ckpts kept for manual prune after export"
fi
echo "============================================================"

submit() {  # name dep timelim array exports script -> jobid
    local name=$1 dep=$2 tl=$3 arr=$4 exp=$5 script=$6
    local a=(--parsable --account="$ACCOUNT" --gpus="$GPU_SPEC" --time="$tl"
             --array="$arr" --job-name="$name" --export="$exp")
    [ -n "$dep" ] && a+=("$dep")
    if [ "$DRY_RUN" = "1" ]; then echo "DRY: sbatch ${a[*]} $script" >&2; echo "DRY$RANDOM"; return; fi
    sbatch "${a[@]}" "$script"
}
submit_cleanup() {  # name dep target_dir -> jobid
    local name=$1 dep=$2 tgt=$3
    local a=(--parsable --account="$ACCOUNT" --time=00:30:00 --partition=RM-shared
             --ntasks=1 --cpus-per-task=1 --mem=1G --job-name="$name"
             --output=experiments/logs/%x_%A.out)
    [ -n "$dep" ] && a+=("$dep")
    if [ "$DRY_RUN" = "1" ]; then echo "DRY: sbatch ${a[*]} --wrap 'rm -rf $tgt'" >&2; echo "DRY$RANDOM"; return; fi
    sbatch "${a[@]}" --wrap "echo 'cleanup: rm -rf $tgt'; rm -rf '$tgt'; echo done"
}

common="NUM_MODELS=$NUM_MODELS,NUM_EPOCHS=$NUM_EPOCHS,TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE,DATA_FRACTION=$DF"
common+=",OPTIMIZER=$OPTIMIZER,ENSEMBLE_MODE=$ENSEMBLE_MODE"
common+=",COMPILE_MODE=$COMPILE_MODE,COMPLETEP=$COMPLETEP,NO_VE_PROJS=$NO_VE_PROJS,NO_WARMDOWN=$NO_WARMDOWN"
common+=",MUP_BASE_WIDTH=$MUP_BASE_WIDTH,MUP_BASE_DEPTH=$MUP_BASE_DEPTH,MUP_BASE_HEAD_DIM=$MUP_BASE_HEAD_DIM"
common+=",CHECKPOINT_EVERY_N_STEPS=$CHECKPOINT_EVERY_N_STEPS,VAL_EVERY_N_STEPS=$VAL_EVERY_N_STEPS,CHECKPOINT_BASE=$CHECKPOINT_BASE"
common+=",N_LAYER=$N_LAYER,N_HEAD=$N_HEAD,N_EMBD=$N_EMBD"

prev_cleanup=""
for wd in $WD_LIST; do
    CELL_TS="${TAG}_wd${wd}_d${N_LAYER}_w${N_EMBD}"
    GROUP="$CELL_TS"
    CKPT_DIR="${CHECKPOINT_BASE}/parallel_init_shuffle_ens_${CELL_TS}"
    echo; echo "=== wd=$wd  (tag $CELL_TS) ==="
    [ "$DRY_RUN" = "1" ] || mkdir -p "$CKPT_DIR"

    texp="ALL,SHARED_TIMESTAMP=$CELL_TS,WANDB_GROUP=$GROUP,WEIGHT_DECAY=$wd,$common"
    # sequential chain only when cleaning up (to bound disk); else concurrent
    tdep=""; [ "$CLEANUP" = "1" ] && [ -n "$prev_cleanup" ] && tdep="--dependency=afterany:$prev_cleanup"
    TJOB=$(submit "wd${wd}_train_d${N_LAYER}_w${N_EMBD}" "$tdep" "$TRAIN_TIME" "$TRAIN_ARRAY" "$texp" experiments/parallel/train_array.sh)
    echo "  train   job=$TJOB  array=$TRAIN_ARRAY ${tdep:+($tdep)}"

    rexp="ALL,SHARED_TIMESTAMP=$CELL_TS,WANDB_GROUP=$GROUP,NUM_MODELS=$NUM_MODELS,NUM_EPOCHS=$NUM_EPOCHS,END_EPOCH=$NUM_EPOCHS"
    rexp+=",ENS_SIZES_STR=$ENS_SIZES_STR,CHECKPOINT_BASE=$CHECKPOINT_BASE,ENSEMBLE_MODE=$ENSEMBLE_MODE,EVAL_MODE=$EVAL_MODE,START_STEP=0"
    RJOB=$(submit "wd${wd}_replay_d${N_LAYER}_w${N_EMBD}" "--dependency=afterok:$TJOB" "$REPLAY_TIME" "$REPLAY_ARRAY" "$rexp" experiments/parallel/replay_array_fused.sh)
    echo "  replay  job=$RJOB  (afterok $TJOB)"

    if [ "$CLEANUP" = "1" ]; then
        CJOB=$(submit_cleanup "wd${wd}_cleanup_d${N_LAYER}_w${N_EMBD}" "--dependency=afterok:$RJOB" "$CKPT_DIR")
        echo "  cleanup job=$CJOB  (afterok $RJOB) -> rm $CKPT_DIR"
        prev_cleanup="$CJOB"
    fi
done

echo; echo "Submitted WD sweep TAG=$TAG. Monitor: squeue -u $USER | grep -E 'wd0|wd1'"
echo "After replays finish, export curves with the export-valloss skill (reads .out logs)."
