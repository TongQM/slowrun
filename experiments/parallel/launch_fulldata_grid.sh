#!/bin/bash
# Full-data grid: 2×2 (depth × width) at df=1.0, wd=0, constant LR.
# Sweep dimensions: N (model size) × E (ensemble size), s free along training curve.
#
# Grid cells:
#   d6/w384   (N = 14.2M non-embed params)
#   d6/w768   (N = 56.6M)
#   d12/w384  (N = 28.3M)
#   d12/w768  (N = 113.2M)
#
# Per cell: 5 individuals × 2 strategies = 10 training tasks.
# Ensemble replay: E ∈ {2,3,4,5} per cell.
# TOKEN_BUDGET = 1B tokens → NUM_EPOCHS = ceil(1B / (df * 99942400)).
#
# Charge: ACCOUNT=cis260009p.  Storage: CHECKPOINT_BASE on cis260161p.
#
# Usage:
#   bash experiments/parallel/launch_fulldata_grid.sh
#   DRY_RUN=1 bash experiments/parallel/launch_fulldata_grid.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

GRID_TAG="${GRID_TAG:-fulldata_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

ACCOUNT=cis260009p
GPU_SPEC=h100-80:1
COMPILE_MODE=inductor
COMPLETEP=1
NO_VE_PROJS=1
NO_WARMDOWN=1
TOTAL_BATCH_SIZE=131072
NUM_MODELS=5
DATA_FRACTION=1.0
TOKEN_BUDGET=3000000000      # 3B tokens → 30 epochs at df=1.0
VAL_EVERY_N_STEPS=152
CHECKPOINT_EVERY_N_STEPS=152
PERMANENT_EVERY_N_STEPS=152    # keep ALL step ckpts (~733 GB total, fits in 2.9 TB free)
PERMANENT_EVERY_N_EPOCHS=1     # keep ALL epoch ckpts
OPTIMIZER=adamw
ENSEMBLE_MODE=logit
WEIGHT_DECAY=0
MUP_BASE_WIDTH=768
MUP_BASE_DEPTH=12
MUP_BASE_HEAD_DIM=64

CIS161=/ocean/projects/cis260161p/ymiao6/scaling/slowrun/checkpoints
DEST="$CIS161"
mkdir -p "$DEST" experiments/logs

# tokens_per_epoch at df=1.0 = 99942400
# NUM_EPOCHS = ceil(1B / 99942400) = 11
TPE=$(python3 -c "print(int(round(${DATA_FRACTION} * 99942400)))")
NUM_EPOCHS=$(python3 -c "import math; print(math.ceil(${TOKEN_BUDGET} / ${TPE}))")

echo "============================================================"
echo "Full-data grid: 2×2 (d × w) at df=${DATA_FRACTION}, wd=${WEIGHT_DECAY}"
echo "  TOKEN_BUDGET=$TOKEN_BUDGET (~$(($TOKEN_BUDGET / 1000000))M tokens)"
echo "  NUM_EPOCHS=$NUM_EPOCHS (tpe=$TPE)"
echo "  ${NUM_MODELS} inds × 2 strats per cell, E ∈ {2,3,4,5} replay"
echo "  ACCOUNT=$ACCOUNT, DEST=$DEST, GPU=$GPU_SPEC, compile=$COMPILE_MODE"
echo "  CompleteP=$COMPLETEP, no_ve_projs=$NO_VE_PROJS, no_warmdown=$NO_WARMDOWN"
echo "  GRID_TAG=$GRID_TAG"
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

# Grid: (depth, n_head, width)
declare -a DEPTHS=(6 6 12 12)
declare -a HEADS=(6 12 12 12)
declare -a WIDTHS=(384 768 384 768)

for idx in "${!DEPTHS[@]}"; do
    L=${DEPTHS[$idx]}
    H=${HEADS[$idx]}
    W=${WIDTHS[$idx]}

    echo
    echo "=== Cell $((idx+1))/4: d${L}/w${W} ==="

    CELL_TS="${GRID_TAG}_d${L}_w${W}"
    GROUP="${GRID_TAG}_d${L}_w${W}"

    mkdir -p "$DEST/parallel_init_ens_${CELL_TS}" "$DEST/parallel_init_shuffle_ens_${CELL_TS}"

    cell_exports="ALL,N_LAYER=$L,N_HEAD=$H,N_EMBD=$W,SHARED_TIMESTAMP=$CELL_TS,WANDB_GROUP=$GROUP"
    cell_exports+=",NUM_MODELS=$NUM_MODELS,NUM_EPOCHS=$NUM_EPOCHS,TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
    cell_exports+=",DATA_FRACTION=$DATA_FRACTION,OPTIMIZER=$OPTIMIZER,ENSEMBLE_MODE=$ENSEMBLE_MODE"
    cell_exports+=",VAL_EVERY_N_STEPS=$VAL_EVERY_N_STEPS,COMPILE_MODE=$COMPILE_MODE"
    cell_exports+=",COMPLETEP=$COMPLETEP,NO_VE_PROJS=$NO_VE_PROJS,NO_WARMDOWN=$NO_WARMDOWN"
    cell_exports+=",WEIGHT_DECAY=$WEIGHT_DECAY"
    cell_exports+=",MUP_BASE_WIDTH=$MUP_BASE_WIDTH,MUP_BASE_DEPTH=$MUP_BASE_DEPTH,MUP_BASE_HEAD_DIM=$MUP_BASE_HEAD_DIM"
    cell_exports+=",CHECKPOINT_EVERY_N_STEPS=$CHECKPOINT_EVERY_N_STEPS,CHECKPOINT_BASE=$DEST"
    cell_exports+=",PERMANENT_EVERY_N_STEPS=$PERMANENT_EVERY_N_STEPS,PERMANENT_EVERY_N_EPOCHS=$PERMANENT_EVERY_N_EPOCHS"
    cell_exports+=",ENS_SIZES_STR=2 3 4 5,SKIP_INDIV_VAL=1,END_EPOCH=$NUM_EPOCHS"

    # Wallclock: ~3B tokens each. d12/w768 ~4h on H100. Pad to 6h.
    timelim="06:00:00"

    # All 4 cells launch in parallel (small enough to coexist on disk)
    # Train: NUM_MODELS × 2 strategies = 10 tasks (array 0-9)
    arr_end=$((NUM_MODELS * 2 - 1))
    TJOB=$(submit_one "fd_train_d${L}_w${W}" "" "$timelim" "0-${arr_end}" "$cell_exports" experiments/parallel/train_array.sh)
    echo "  train  job=$TJOB  timelim=$timelim  group=$GROUP"

    # Replay: 2 tasks (one per strategy), fused over E ∈ {2,3,4,5}
    RJOB=$(submit_one "fd_replay_d${L}_w${W}" "--dependency=afterok:$TJOB" "06:00:00" "0-1" "$cell_exports" experiments/parallel/replay_array_fused.sh)
    echo "  replay job=$RJOB  (after $TJOB)"

    # Cleanup: 2 tasks (one per strategy)
    CJOB=$(submit_one "fd_clean_d${L}_w${W}" "--dependency=afterok:$RJOB" "00:30:00" "0-1" "$cell_exports" experiments/parallel/cleanup_array.sh)
    echo "  clean  job=$CJOB  (after $RJOB)"
done

echo
echo "============================================================"
echo "Full-data grid submission complete."
echo "  GRID_TAG=$GRID_TAG"
echo "  Wandb groups: ${GRID_TAG}_d{6,12}_w{384,768}"
echo "============================================================"
