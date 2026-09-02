#!/bin/bash
# Q1 (v2): Data-size ablation on d12/w768, weight_decay=0, fixed-tokens budget.
# 4 dfs × 2 individuals (1 init_ens + 1 init_shuffle_ens), each run trained to ~1B tokens.
# Goal: clean overfit-onset (in tokens) as function of unique-data fraction.
#
# v2 vs v1: v1 (50 epochs uniform, wd=0.1) failed to show overfit at df ≥ 0.4 because
# per-step weight decay × more steps/epoch suppressed it. v2 sets wd=0 and fixes the
# *token* budget so all dfs do the same amount of training compute.
#
# Constant LR + CompleteP + no-ve-projs + wd=0 throughout.
# Charged to cis260161p, stored on cis260161p.
# No replay (single-individual val curves are sufficient — logged inline at
# every 20M tokens via --val-every-n-steps=152).
# Cleanup keeps per-epoch ckpts at every 5th epoch.
#
# Usage:
#   bash experiments/parallel/launch_q1_data_size_sweep.sh
#   GRID_TAG=<custom> bash experiments/parallel/launch_q1_data_size_sweep.sh
#   DRY_RUN=1 bash ...

set -euo pipefail
cd "$(dirname "$0")/../.."

GRID_TAG="${GRID_TAG:-q1v2_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

ACCOUNT=cis260161p
GPU_SPEC=h100-80:1
COMPILE_MODE=inductor
COMPLETEP=1
NO_VE_PROJS=1
NO_WARMDOWN=1
TOTAL_BATCH_SIZE=131072
NUM_MODELS=1                 # 1 model per strategy, 2 strategies -> 2 train tasks
TOKEN_BUDGET=1000000000      # 1B tokens per run; NUM_EPOCHS computed per-df below
VAL_EVERY_N_STEPS=152        # individual val every ~20M tokens
CHECKPOINT_EVERY_N_STEPS=0   # NO step ckpts (no ensemble replay needed)
PERMANENT_EVERY_N_STEPS=0
PERMANENT_EVERY_N_EPOCHS=5   # cleanup keeps every 5th epoch ckpt
OPTIMIZER=adamw
ENSEMBLE_MODE=logit
WEIGHT_DECAY=0               # v2: no regularization, so overfit shows up cleanly
MUP_BASE_WIDTH=768
MUP_BASE_DEPTH=12
MUP_BASE_HEAD_DIM=64

CIS161=/ocean/projects/cis260161p/ymiao6/scaling/slowrun/checkpoints
DEST="$CIS161"
mkdir -p "$DEST" experiments/logs

declare -a DFS=(0.2 0.3 0.4 0.5)
L=12; H=12; W=768

echo "============================================================"
echo "Q1 v2 data-size sweep: d${L}/w${W}, dfs=${DFS[*]}, wd=$WEIGHT_DECAY"
echo "  TOKEN_BUDGET=$TOKEN_BUDGET (~$(($TOKEN_BUDGET / 1000000))M tokens per run)"
echo "  1 ind per strategy, 2 strats per cell"
echo "  ACCOUNT=$ACCOUNT, DEST=$DEST, GPU=$GPU_SPEC, compile=$COMPILE_MODE"
echo "  CompleteP=$COMPLETEP, no_ve_projs=$NO_VE_PROJS, no_warmdown=$NO_WARMDOWN"
echo "============================================================"

submit_one() {
    # name dep timelim arr exports script
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
    local df=$1
    local CELL_TS="${GRID_TAG}_d${L}_w${W}_df${df}"
    local GROUP="q1_data_size_${GRID_TAG}_d${L}_w${W}_df${df}"

    # Per-df epochs from TOKEN_BUDGET. tokens_per_epoch = df * 100M (approx; exact 99942400).
    # Round up so each run hits at least TOKEN_BUDGET tokens.
    local tpe=$(python3 -c "print(int(round(${df} * 99942400)))")
    local NUM_EPOCHS_DF=$(python3 -c "import math; print(math.ceil(${TOKEN_BUDGET} / ${tpe}))")

    mkdir -p "$DEST/parallel_init_ens_${CELL_TS}" "$DEST/parallel_init_shuffle_ens_${CELL_TS}"

    local exports="ALL,N_LAYER=$L,N_HEAD=$H,N_EMBD=$W,SHARED_TIMESTAMP=$CELL_TS,WANDB_GROUP=$GROUP"
    exports+=",NUM_MODELS=$NUM_MODELS,NUM_EPOCHS=$NUM_EPOCHS_DF,TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
    exports+=",DATA_FRACTION=$df,OPTIMIZER=$OPTIMIZER,ENSEMBLE_MODE=$ENSEMBLE_MODE"
    exports+=",VAL_EVERY_N_STEPS=$VAL_EVERY_N_STEPS,COMPILE_MODE=$COMPILE_MODE"
    exports+=",COMPLETEP=$COMPLETEP,NO_VE_PROJS=$NO_VE_PROJS,NO_WARMDOWN=$NO_WARMDOWN"
    exports+=",WEIGHT_DECAY=$WEIGHT_DECAY"
    exports+=",MUP_BASE_WIDTH=$MUP_BASE_WIDTH,MUP_BASE_DEPTH=$MUP_BASE_DEPTH,MUP_BASE_HEAD_DIM=$MUP_BASE_HEAD_DIM"
    exports+=",CHECKPOINT_EVERY_N_STEPS=$CHECKPOINT_EVERY_N_STEPS,CHECKPOINT_BASE=$DEST"
    exports+=",PERMANENT_EVERY_N_STEPS=$PERMANENT_EVERY_N_STEPS,PERMANENT_EVERY_N_EPOCHS=$PERMANENT_EVERY_N_EPOCHS"
    exports+=",ENS_SIZES_STR=2,SKIP_INDIV_VAL=1,END_EPOCH=$NUM_EPOCHS_DF"

    # Wallclock is roughly the same across dfs (each does ~1B tokens). Pad to 6h.
    local timelim="06:00:00"

    # Train: 2 tasks (init_ens model 0 + init_shuffle_ens model 0)
    local TJOB
    TJOB=$(submit_one "q1v2_train_d12_w768_df${df}" "" "$timelim" "0-1" "$exports" experiments/parallel/train_array.sh)
    echo "  df=$df  epochs=$NUM_EPOCHS_DF  train  job=$TJOB  timelim=$timelim  group=$GROUP" >&2

    # Cleanup: 2 tasks (one per strategy)
    local CJOB
    CJOB=$(submit_one "q1v2_cleanup_d12_w768_df${df}" "--dependency=afterok:$TJOB" "00:30:00" "0-1" "$exports" experiments/parallel/cleanup_array.sh)
    echo "  df=$df  clean  job=$CJOB  (after $TJOB)" >&2
}

for df in "${DFS[@]}"; do
    echo
    echo "=== d12/w768 df=$df ==="
    submit_cell "$df"
done

echo
echo "Q1 submission complete. Wandb groups: q1_data_size_${GRID_TAG}_d12_w768_df{$(IFS=,; echo "${DFS[*]}")}"
