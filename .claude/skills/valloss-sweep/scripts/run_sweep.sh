#!/bin/bash
# run_sweep.sh — launch or extend the slowrun training+replay pipeline to produce
# individual + ensemble validation-loss curves at a chosen token resolution.
#
# For each (depth, head, width) cell it submits a chain:
#     train_array (num_models x 2 strategies)  ->  replay_array_fused (2 strategies)
# with the replay gated `afterok` on training. The backbone scripts live in
# experiments/parallel/ (train_array.sh, replay_array_fused.sh, replay_fused.py).
#
# MODE=fresh   : new run. GRID_TAG auto-generated if unset. Replays evaluate the
#                full step grid (START_STEP=0).
# MODE=extend  : resume an EXISTING run to more epochs. Requires GRID_TAG of the
#                existing run and a higher NUM_EPOCHS. Training resumes from the
#                latest per-epoch checkpoint (constant-LR / --no-warmdown makes
#                this exact). Replays compute ONLY the new region: START_STEP is
#                auto-derived as PREV_EPOCHS*STEPS_PER_EPOCH+1 (override directly).
#                New replay points merge with the existing ones by token.
#
# Resolution: CHECKPOINT_EVERY_N_STEPS controls per-step ckpt cadence
#   152 steps @ batch 131072 = ~20M tokens (fine).  EVAL_MODE=step uses these
#   (~20M ensemble resolution); EVAL_MODE=epoch uses per-epoch ckpts (~100M).
#
# Usage:
#   bash run_sweep.sh                                   # fresh df=1.0 2x2 grid, 31 epochs, 20M res
#   DRY_RUN=1 bash run_sweep.sh                         # print sbatch lines only
#   MODE=extend GRID_TAG=<tag> NUM_EPOCHS=40 PREV_EPOCHS=31 bash run_sweep.sh
#   CELLS="6:6:384 12:12:768" NUM_EPOCHS=25 bash run_sweep.sh
set -euo pipefail

# Repo root = four levels up from this script.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO"

# ---- parameters (env-overridable) ----
MODE="${MODE:-fresh}"                               # fresh | extend
# (depth:head:width) cells, smallest first
CELLS="${CELLS:-6:6:384 6:12:768 12:12:384 12:12:768}"
DF="${DF:-1.0}"
NUM_EPOCHS="${NUM_EPOCHS:-31}"                      # target epochs (extend: the new, larger total)
NUM_MODELS="${NUM_MODELS:-5}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-131072}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-763}"           # round(df*tokens_per_epoch/batch); df=1.0 -> 763
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-152}"   # 152 => ~20M-token ckpts
VAL_EVERY_N_STEPS="${VAL_EVERY_N_STEPS:-152}"
EVAL_MODE="${EVAL_MODE:-step}"                      # step (~20M) | epoch (~100M) | both
ENS_SIZES_STR="${ENS_SIZES_STR:-2 3 4 5}"
ENSEMBLE_MODE="${ENSEMBLE_MODE:-logit}"
OPTIMIZER="${OPTIMIZER:-adamw}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
# project always-on defaults
COMPLETEP="${COMPLETEP:-1}"; NO_VE_PROJS="${NO_VE_PROJS:-1}"; NO_WARMDOWN="${NO_WARMDOWN:-1}"
COMPILE_MODE="${COMPILE_MODE:-inductor}"
MUP_BASE_WIDTH="${MUP_BASE_WIDTH:-768}"; MUP_BASE_DEPTH="${MUP_BASE_DEPTH:-12}"; MUP_BASE_HEAD_DIM="${MUP_BASE_HEAD_DIM:-64}"
# SLURM
ACCOUNT="${ACCOUNT:-cis260095p}"                   # compute charged here (cis260009p access revoked 2026-06)
GPU_SPEC="${GPU_SPEC:-h100-80:1}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-checkpoints}"  # checkpoints stored here (under REPO)
TRAIN_TIME="${TRAIN_TIME:-03:00:00}"
REPLAY_TIME="${REPLAY_TIME:-06:00:00}"
DRY_RUN="${DRY_RUN:-0}"

# GRID_TAG: required for extend; auto for fresh.
if [ "$MODE" = "extend" ]; then
    if [ -z "${GRID_TAG:-}" ]; then echo "ERROR: extend mode needs GRID_TAG of the existing run."; exit 1; fi
    if [ "$EVAL_MODE" = "step" ] && [ -z "${START_STEP:-}" ]; then
        if [ -z "${PREV_EPOCHS:-}" ]; then
            echo "ERROR: extend+step needs START_STEP, or PREV_EPOCHS to derive it."; exit 1; fi
        START_STEP=$(( PREV_EPOCHS * STEPS_PER_EPOCH + 1 ))
        echo "derived START_STEP=$START_STEP (PREV_EPOCHS=$PREV_EPOCHS x SPE=$STEPS_PER_EPOCH +1)"
    fi
else
    GRID_TAG="${GRID_TAG:-sweep_$(date +%Y%m%d_%H%M%S)}"
    START_STEP="${START_STEP:-0}"
fi
START_STEP="${START_STEP:-0}"

echo "============================================================"
echo "run_sweep  MODE=$MODE  GRID_TAG=$GRID_TAG"
echo "  cells=[$CELLS]  df=$DF  num_epochs=$NUM_EPOCHS  num_models=$NUM_MODELS"
echo "  ckpt_every=$CHECKPOINT_EVERY_N_STEPS (res), eval_mode=$EVAL_MODE, start_step=$START_STEP"
echo "  account=$ACCOUNT  ckpt_base=$CHECKPOINT_BASE  wd=$WEIGHT_DECAY"
echo "============================================================"

submit() {  # name dep timelim arr exports script -> jobid
    local name=$1 dep=$2 tl=$3 arr=$4 exp=$5 script=$6
    local a=(--parsable --account="$ACCOUNT" --gpus="$GPU_SPEC" --time="$tl"
             --array="$arr" --job-name="$name" --export="$exp")
    [ -n "$dep" ] && a+=("$dep")
    if [ "$DRY_RUN" = "1" ]; then echo "DRY: sbatch ${a[*]} $script" >&2; echo "DRY$RANDOM"; return; fi
    sbatch "${a[@]}" "$script"
}

common="NUM_MODELS=$NUM_MODELS,NUM_EPOCHS=$NUM_EPOCHS,TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE,DATA_FRACTION=$DF"
common+=",OPTIMIZER=$OPTIMIZER,ENSEMBLE_MODE=$ENSEMBLE_MODE,WEIGHT_DECAY=$WEIGHT_DECAY"
common+=",COMPILE_MODE=$COMPILE_MODE,COMPLETEP=$COMPLETEP,NO_VE_PROJS=$NO_VE_PROJS,NO_WARMDOWN=$NO_WARMDOWN"
common+=",MUP_BASE_WIDTH=$MUP_BASE_WIDTH,MUP_BASE_DEPTH=$MUP_BASE_DEPTH,MUP_BASE_HEAD_DIM=$MUP_BASE_HEAD_DIM"
common+=",CHECKPOINT_EVERY_N_STEPS=$CHECKPOINT_EVERY_N_STEPS,VAL_EVERY_N_STEPS=$VAL_EVERY_N_STEPS,CHECKPOINT_BASE=$CHECKPOINT_BASE"

for cell in $CELLS; do
    L="${cell%%:*}"; rest="${cell#*:}"; H="${rest%%:*}"; W="${rest##*:}"
    CELL_TS="${GRID_TAG}_d${L}_w${W}"; GROUP="$CELL_TS"
    echo; echo "=== cell d${L}/w${W} (tag $CELL_TS) ==="
    mkdir -p "$CHECKPOINT_BASE/parallel_init_ens_${CELL_TS}" "$CHECKPOINT_BASE/parallel_init_shuffle_ens_${CELL_TS}"

    texp="ALL,N_LAYER=$L,N_HEAD=$H,N_EMBD=$W,SHARED_TIMESTAMP=$CELL_TS,WANDB_GROUP=$GROUP,$common"
    arr_end=$(( NUM_MODELS * 2 - 1 ))
    TJOB=$(submit "fd_train_d${L}_w${W}" "" "$TRAIN_TIME" "0-${arr_end}" "$texp" experiments/parallel/train_array.sh)
    echo "  train  job=$TJOB"

    rexp="ALL,SHARED_TIMESTAMP=$CELL_TS,WANDB_GROUP=$GROUP,NUM_MODELS=$NUM_MODELS,NUM_EPOCHS=$NUM_EPOCHS,END_EPOCH=$NUM_EPOCHS"
    rexp+=",ENS_SIZES_STR=$ENS_SIZES_STR,CHECKPOINT_BASE=$CHECKPOINT_BASE,ENSEMBLE_MODE=$ENSEMBLE_MODE,EVAL_MODE=$EVAL_MODE,START_STEP=$START_STEP"
    RJOB=$(submit "fd_replay20m_d${L}_w${W}" "--dependency=afterok:$TJOB" "$REPLAY_TIME" "0-1" "$rexp" experiments/parallel/replay_array_fused.sh)
    echo "  replay job=$RJOB  (after $TJOB)"
done

echo; echo "Submitted. Monitor with: bash $HERE/monitor_sweep.sh '$GRID_TAG'"
