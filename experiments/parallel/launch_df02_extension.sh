#!/bin/bash
# df=0.2 grid extension: add d18 row + w1152 column (7 new cells) for scaling-law fit.
#
# Aligned with the original df=0.2 grid (grid_20260430_152533) for direct comparison:
#   - LR schedule: WARMUP=0 + flat 80% + linear warmdown to 0 (trapezoidal, NOT --no-warmdown)
#   - completep ON, no-ve-projs ON, weight_decay=0.1, optimizer adamw
#   - data_fraction=0.2, num_epochs=25, total_batch_size=131072 (small batch)
#   - 5 individuals × 2 strategies = 10 train tasks per cell
#   - replay sizes {2,3,4,5}
#   - val once per epoch (val-every-n-steps=152), per-epoch ckpts only (no step ckpts)
#   - cleanup keeps every-5-epoch ckpt (5, 10, 15, 20, 25)
#
# 7 new cells:
#   d18 row : d18/{w384, w768, w1152, w1536}    (4 cells)
#   w1152 col: {d6, d12, d18, d24}/w1152          (4 cells, d18/w1152 shared with d18 row)
#
# Compute: ~630 SU total (smallest 12 SU, largest 198 SU). Fits cis260161p.
# Storage: largest transient peak ~640 GB. cis260161p has ~2 TB free.
#
# Usage:
#   bash experiments/parallel/launch_df02_extension.sh
#   GRID_TAG=<custom> bash ...
#   START_CELL=<idx> bash ...   (1-7)
#   DRY_RUN=1 bash ...

set -euo pipefail
cd "$(dirname "$0")/../.."

GRID_TAG="${GRID_TAG:-df02ext_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
START_CELL="${START_CELL:-1}"
END_CELL="${END_CELL:-7}"

ACCOUNT="${ACCOUNT:-cis260161p}"
GPU_SPEC="${GPU_SPEC:-h100-80:1}"
COMPILE_MODE="${COMPILE_MODE:-inductor}"

NUM_MODELS=5
NUM_EPOCHS=25
DATA_FRACTION=0.2
TOTAL_BATCH_SIZE=131072
VAL_EVERY_N_STEPS=152               # once per epoch at df=0.2
ENS_SIZES_STR="2 3 4 5"
ENSEMBLE_MODE=logit
OPTIMIZER=adamw
WEIGHT_DECAY=0.1                     # original df=0.2 grid setup
# NO_WARMDOWN intentionally NOT set → trapezoidal LR (matches original grid)

COMPLETEP=1
NO_VE_PROJS=1
MUP_BASE_WIDTH=768
MUP_BASE_DEPTH=12
MUP_BASE_HEAD_DIM=64

CIS161=/ocean/projects/cis260161p/ymiao6/scaling/slowrun/checkpoints

# Per-cell config: "tag:L:H:W:DEST"
# Smallest-to-largest by d × w² (sequential cells, each waits for prev cleanup).
# n_head = w / 64 (head_dim_base=64).
# Costs (1× ref d12/w768 = 33 SU):
#   d18/w384  : 12 SU
#   d6/w1152  : 37 SU
#   d18/w768  : 50 SU
#   d12/w1152 : 74 SU
#   d18/w1152 : 111 SU
#   d24/w1152 : 148 SU
#   d18/w1536 : 198 SU
declare -a CELLS=(
    "d18_w384:18:6:384:$CIS161"
    "d6_w1152:6:18:1152:$CIS161"
    "d18_w768:18:12:768:$CIS161"
    "d12_w1152:12:18:1152:$CIS161"
    "d18_w1152:18:18:1152:$CIS161"
    "d24_w1152:24:18:1152:$CIS161"
    "d18_w1536:18:24:1536:$CIS161"
)

echo "============================================================"
echo "df=0.2 grid extension: $((END_CELL - START_CELL + 1))-cell scan, ACCOUNT=$ACCOUNT"
echo "  GRID_TAG=$GRID_TAG"
echo "  trapezoidal LR (NOT --no-warmdown), wd=$WEIGHT_DECAY  (matches grid_20260430_152533)"
echo "  CompleteP=ON, no_ve_projs=ON, $NUM_MODELS ind × 2 strats × $NUM_EPOCHS ep"
echo "  ens sizes: {$ENS_SIZES_STR}"
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

submit_cell() {
    local idx=$1 spec=$2 prev_cleanup=$3
    IFS=':' read -r tag L H W DEST <<< "$spec"
    local CELL_TS="${GRID_TAG}_${tag}"
    local GROUP="grid_${GRID_TAG}_${tag}_df${DATA_FRACTION}"

    mkdir -p "$DEST/parallel_init_ens_${CELL_TS}" "$DEST/parallel_init_shuffle_ens_${CELL_TS}"

    local exports="ALL,N_LAYER=$L,N_HEAD=$H,N_EMBD=$W,SHARED_TIMESTAMP=$CELL_TS,WANDB_GROUP=$GROUP"
    exports+=",NUM_MODELS=$NUM_MODELS,NUM_EPOCHS=$NUM_EPOCHS,TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
    exports+=",DATA_FRACTION=$DATA_FRACTION,OPTIMIZER=$OPTIMIZER,ENSEMBLE_MODE=$ENSEMBLE_MODE"
    exports+=",VAL_EVERY_N_STEPS=$VAL_EVERY_N_STEPS,COMPILE_MODE=$COMPILE_MODE"
    exports+=",COMPLETEP=$COMPLETEP,NO_VE_PROJS=$NO_VE_PROJS"
    exports+=",WEIGHT_DECAY=$WEIGHT_DECAY"
    exports+=",MUP_BASE_WIDTH=$MUP_BASE_WIDTH,MUP_BASE_DEPTH=$MUP_BASE_DEPTH,MUP_BASE_HEAD_DIM=$MUP_BASE_HEAD_DIM"
    exports+=",CHECKPOINT_EVERY_N_STEPS=0,CHECKPOINT_BASE=$DEST"
    exports+=",PERMANENT_EVERY_N_STEPS=0,PERMANENT_EVERY_N_EPOCHS=5"
    exports+=",ENS_SIZES_STR=$ENS_SIZES_STR,SKIP_INDIV_VAL=1,END_EPOCH=$NUM_EPOCHS"

    # Time budgets — generous for the largest cell.
    local train_time replay_time
    case "$tag" in
        d18_w384)   train_time="04:00:00"; replay_time="03:00:00" ;;
        d6_w1152)   train_time="04:00:00"; replay_time="04:00:00" ;;
        d18_w768)   train_time="04:00:00"; replay_time="04:00:00" ;;
        d12_w1152)  train_time="06:00:00"; replay_time="05:00:00" ;;
        d18_w1152)  train_time="08:00:00"; replay_time="06:00:00" ;;
        d24_w1152)  train_time="10:00:00"; replay_time="07:00:00" ;;
        d18_w1536)  train_time="12:00:00"; replay_time="08:00:00" ;;
        *)          train_time="12:00:00"; replay_time="08:00:00" ;;
    esac

    # train: 5 ind × 2 strats = 10 tasks
    local TRAIN_RANGE="0-$((2*NUM_MODELS - 1))"
    local NUM_SIZES=$(echo $ENS_SIZES_STR | wc -w)
    local REPLAY_RANGE="0-$((2*NUM_SIZES - 1))"
    local CLEANUP_RANGE="0-1"

    local dep=""
    if [ -n "$prev_cleanup" ]; then dep="--dependency=afterok:$prev_cleanup"; fi

    local TJOB
    TJOB=$(submit_one "df02ext_train_${tag}" "$dep" "$train_time" "$TRAIN_RANGE" "$exports" experiments/parallel/train_array.sh)
    echo "  cell $idx [$tag]  train  array=$TRAIN_RANGE  job=$TJOB  $train_time  group=$GROUP" >&2

    local RJOB
    RJOB=$(submit_one "df02ext_replay_${tag}" "--dependency=afterok:$TJOB" "$replay_time" "$REPLAY_RANGE" "$exports" experiments/parallel/replay_array.sh)
    echo "  cell $idx [$tag]  replay array=$REPLAY_RANGE  job=$RJOB  $replay_time  (after $TJOB)" >&2

    local CJOB
    CJOB=$(submit_one "df02ext_cleanup_${tag}" "--dependency=afterok:$RJOB" "00:30:00" "$CLEANUP_RANGE" "$exports" experiments/parallel/cleanup_array.sh)
    echo "  cell $idx [$tag]  clean  array=$CLEANUP_RANGE  job=$CJOB  (after $RJOB)" >&2
    echo "$CJOB"
}

prev_cleanup=""
for i in $(seq 1 ${#CELLS[@]}); do
    if [ "$i" -lt "$START_CELL" ] || [ "$i" -gt "$END_CELL" ]; then continue; fi
    spec="${CELLS[$((i - 1))]}"
    cleanup_jid=$(submit_cell "$i" "$spec" "$prev_cleanup")
    prev_cleanup="$cleanup_jid"
    echo
done

echo "df=0.2 extension submission complete. GRID_TAG=$GRID_TAG"
