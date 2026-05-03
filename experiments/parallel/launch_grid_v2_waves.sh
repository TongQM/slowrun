#!/bin/bash
# Wave-parallel launcher for cells 4-9 of the df=0.4 grid.
#
# Plan (assuming cells 1-3 are already done):
#   Wave A: cells 4 (d6/w768), 5 (d12/w768), 6 (d24/w768) — all parallel on cis260161p
#   Wave B: cell 7 (d6/w1536) on cis260161p + cell 8 init_ens on cis260095p — parallel
#   Wave C: cell 8 init_shuffle on cis260161p + cell 9 init_ens on cis260095p — parallel
#   Wave D: cell 9 init_shuffle on cis260161p — alone
#
# Storage budget per allocation:
#   cis260161p ≤ 1000 GB additional transient
#   cis260095p ≤  600 GB additional transient
#
# Compute charged to cis260009p (which has SU budget).
#
# Usage:
#   GRID_TAG=20260502_013702 PREV_CLEANUP_JOB=40522102 \
#       bash experiments/parallel/launch_grid_v2_waves.sh
#
# (PREV_CLEANUP_JOB is cell 3's cleanup array job ID.)

set -euo pipefail
cd "$(dirname "$0")/../.."

GRID_TAG="${GRID_TAG:-20260502_013702}"
PREV_CLEANUP_JOB="${PREV_CLEANUP_JOB:-}"
DRY_RUN="${DRY_RUN:-0}"

ACCOUNT="${ACCOUNT:-cis260009p}"
GPU_SPEC="${GPU_SPEC:-h100-80:1}"
COMPILE_MODE="${COMPILE_MODE:-inductor}"

NUM_MODELS=5
NUM_EPOCHS=25
DATA_FRACTION=0.4
TOTAL_BATCH_SIZE=131072
VAL_EVERY_N_STEPS=152
ENS_SIZES_STR="2 3 4 5"
ENSEMBLE_MODE=logit
OPTIMIZER=adamw

COMPLETEP=1
NO_VE_PROJS=1
NO_WARMDOWN=1
MUP_BASE_WIDTH=768
MUP_BASE_DEPTH=12
MUP_BASE_HEAD_DIM=64

CIS161=/ocean/projects/cis260161p/ymiao6/scaling/slowrun/checkpoints
CIS095=/ocean/projects/cis260095p/ymiao6/scaling/slowrun/checkpoints

TIME_TRAIN="${TIME_LIMIT:-12:00:00}"
TIME_REPLAY="${REPLAY_TIME_LIMIT:-08:00:00}"
TIME_CLEANUP="00:30:00"

NUM_SIZES=$(echo $ENS_SIZES_STR | wc -w)

# ---- helpers ----
_submit() {
    # _submit name extra_exports dep_flag time_limit array_range script
    local name=$1 extra=$2 dep=$3 timelim=$4 arr=$5 script=$6
    local exports="ALL,N_LAYER=$L,N_HEAD=$H,N_EMBD=$W,SHARED_TIMESTAMP=$CELL_TS,WANDB_GROUP=$GROUP"
    exports+=",NUM_MODELS=$NUM_MODELS,NUM_EPOCHS=$NUM_EPOCHS,TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
    exports+=",DATA_FRACTION=$DATA_FRACTION,OPTIMIZER=$OPTIMIZER,ENSEMBLE_MODE=$ENSEMBLE_MODE"
    exports+=",VAL_EVERY_N_STEPS=$VAL_EVERY_N_STEPS,COMPILE_MODE=$COMPILE_MODE"
    exports+=",COMPLETEP=$COMPLETEP,NO_VE_PROJS=$NO_VE_PROJS,NO_WARMDOWN=$NO_WARMDOWN"
    exports+=",MUP_BASE_WIDTH=$MUP_BASE_WIDTH,MUP_BASE_DEPTH=$MUP_BASE_DEPTH,MUP_BASE_HEAD_DIM=$MUP_BASE_HEAD_DIM"
    exports+=",CHECKPOINT_EVERY_N_STEPS=$CADENCE,CHECKPOINT_BASE=$DEST"
    exports+=",ENS_SIZES_STR=$ENS_SIZES_STR,SKIP_INDIV_VAL=1,END_EPOCH=$NUM_EPOCHS"
    exports+=",PERMANENT_EVERY_N_STEPS=$PERM,$extra"

    local args=(--parsable --account=$ACCOUNT --gpus=$GPU_SPEC --time="$timelim"
                --array=$arr --job-name=$name --export="$exports")
    if [ -n "$dep" ]; then args+=("$dep"); fi

    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY: sbatch ${args[*]} $script" >&2
        echo "DRYJOB$RANDOM"
        return
    fi
    sbatch "${args[@]}" "$script"
}

# Submit a full chain (train→replay→cleanup) for a (cell, strategy_mode) — returns cleanup job ID.
# strategy_mode: "parallel" (tasks 0-9, both strats), "init" (tasks 0-4), "shuffle" (tasks 5-9)
submit_chain() {
    local cell_tag=$1 ll=$2 hh=$3 ww=$4 dest=$5 cadence=$6 perm=$7 mode=$8 dep=$9
    L=$ll; H=$hh; W=$ww; DEST=$dest; CADENCE=$cadence; PERM=$perm

    CELL_TS="${GRID_TAG}_${cell_tag}"
    GROUP="grid_${GRID_TAG}_${cell_tag}_df0.4"
    mkdir -p "$DEST/parallel_init_ens_${CELL_TS}" "$DEST/parallel_init_shuffle_ens_${CELL_TS}"

    local DEP_TRAIN=""
    if [ -n "$dep" ]; then DEP_TRAIN="--dependency=afterok:$dep"; fi

    local TRAIN_RANGE REPLAY_RANGE CLEANUP_RANGE strat_label
    case "$mode" in
        parallel) TRAIN_RANGE="0-$((2*NUM_MODELS - 1))"; REPLAY_RANGE="0-$((2*NUM_SIZES - 1))"; CLEANUP_RANGE="0-1"; strat_label="both" ;;
        init)     TRAIN_RANGE="0-$((NUM_MODELS - 1))";   REPLAY_RANGE="0-$((NUM_SIZES - 1))";   CLEANUP_RANGE="0";   strat_label="init_ens" ;;
        shuffle)  TRAIN_RANGE="$NUM_MODELS-$((2*NUM_MODELS - 1))"; REPLAY_RANGE="$NUM_SIZES-$((2*NUM_SIZES - 1))"; CLEANUP_RANGE="1"; strat_label="init_shuffle_ens" ;;
        *) echo "Invalid mode: $mode" >&2; return 1 ;;
    esac

    local TJOB
    TJOB=$(_submit "train_${cell_tag}_${strat_label}" "" "$DEP_TRAIN" "$TIME_TRAIN" "$TRAIN_RANGE" experiments/parallel/train_array.sh)
    echo "  train  [$strat_label]  array=$TRAIN_RANGE  job=$TJOB" >&2

    local RJOB
    RJOB=$(_submit "replay_${cell_tag}_${strat_label}" "" "--dependency=afterok:$TJOB" "$TIME_REPLAY" "$REPLAY_RANGE" experiments/parallel/replay_array.sh)
    echo "  replay [$strat_label]  array=$REPLAY_RANGE  job=$RJOB  (after $TJOB)" >&2

    local CJOB
    CJOB=$(_submit "cleanup_${cell_tag}_${strat_label}" "" "--dependency=afterok:$RJOB" "$TIME_CLEANUP" "$CLEANUP_RANGE" experiments/parallel/cleanup_array.sh)
    echo "  clean  [$strat_label]  array=$CLEANUP_RANGE  job=$CJOB  (after $RJOB)" >&2

    echo "$CJOB"
}

# ============ Waves ============
echo "============================================================"
echo "Wave launcher: GRID_TAG=$GRID_TAG  ACCOUNT=$ACCOUNT  PREV_CLEANUP=$PREV_CLEANUP_JOB"
echo "============================================================"

# --- Wave A: cells 4, 5, 6 parallel on cis260161p, all chained on PREV_CLEANUP ---
echo
echo "=== Wave A: d6/w768 + d12/w768 + d24/w768 parallel on cis260161p ==="
JOB_A4=$(submit_chain "d6_w768"   6  12 768  "$CIS161" 152 760 parallel "$PREV_CLEANUP_JOB"); JOB_A4=$(echo "$JOB_A4" | tail -1)
JOB_A5=$(submit_chain "d12_w768" 12  12 768  "$CIS161" 152 760 parallel "$PREV_CLEANUP_JOB"); JOB_A5=$(echo "$JOB_A5" | tail -1)
JOB_A6=$(submit_chain "d24_w768" 24  12 768  "$CIS161" 152 760 parallel "$PREV_CLEANUP_JOB"); JOB_A6=$(echo "$JOB_A6" | tail -1)
WAVE_A_DEPS="$JOB_A4:$JOB_A5:$JOB_A6"

# --- Wave B: cell 7 on cis260161p, cell 8 init_ens on cis260095p ---
echo
echo "=== Wave B: d6/w1536 (cis260161p) + d12/w1536 init_ens (cis260095p) parallel ==="
JOB_B7=$(submit_chain "d6_w1536"   6  24 1536 "$CIS161" 152 760 parallel "$WAVE_A_DEPS"); JOB_B7=$(echo "$JOB_B7" | tail -1)
JOB_B8a=$(submit_chain "d12_w1536" 12 24 1536 "$CIS095" 152 760 init     "$WAVE_A_DEPS"); JOB_B8a=$(echo "$JOB_B8a" | tail -1)
WAVE_B_DEPS="$JOB_B7:$JOB_B8a"

# --- Wave C: cell 8 init_shuffle on cis260161p + cell 9 init_ens on cis260095p ---
echo
echo "=== Wave C: d12/w1536 init_shuffle (cis260161p) + d24/w1536 init_ens (cis260095p) ==="
JOB_C8b=$(submit_chain "d12_w1536" 12 24 1536 "$CIS161" 152 760 shuffle "$WAVE_B_DEPS"); JOB_C8b=$(echo "$JOB_C8b" | tail -1)
JOB_C9a=$(submit_chain "d24_w1536" 24 24 1536 "$CIS095"   0   0 init    "$WAVE_B_DEPS"); JOB_C9a=$(echo "$JOB_C9a" | tail -1)
WAVE_C_DEPS="$JOB_C8b:$JOB_C9a"

# --- Wave D: cell 9 init_shuffle on cis260161p ---
echo
echo "=== Wave D: d24/w1536 init_shuffle (cis260161p) alone ==="
JOB_D9b=$(submit_chain "d24_w1536" 24 24 1536 "$CIS161" 0 0 shuffle "$WAVE_C_DEPS"); JOB_D9b=$(echo "$JOB_D9b" | tail -1)

echo
echo "Final cleanup job: $JOB_D9b"
echo
echo "Wandb groups: grid_${GRID_TAG}_d{6,12,24}_w{768,1536}_df0.4"
echo "Monitor: squeue -u \$USER --format=\"%.12i %.30j %.8T %.10M %R\""
