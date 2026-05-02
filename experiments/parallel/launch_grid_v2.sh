#!/bin/bash
# Grid v2 launcher: 9-cell width × depth sweep at df=0.4 with constant LR.
#
# Per-cell pipeline:
#   1) train_array     (5 models × 2 strategies = 10 tasks)
#   2) replay_array    (4 ensemble sizes × 2 strategies = 8 tasks)   [afterok of train]
#   3) cleanup_array   (1 task per strategy = 2 tasks)                [afterok of replay]
#
# Cells run smallest-to-largest. Each cell's chain depends on the previous cell's
# cleanup (--dependency=afterok), so transient ckpts of one cell are deleted before
# the next cell's training starts.
#
# Defaults baked in for this project:
#   --completep --no-ve-projs --no-warmdown          (always)
#   data_fraction=0.4, num_epochs=25, total_batch_size=131072
#   --val-every-n-steps=152          (~ every 20M tokens, individual val)
#   --checkpoint-every-n-steps=152   (transient step ckpts every 20M tokens)
#   PERMANENT_EVERY_N_STEPS=760      (cleanup keeps every 5th ckpt = every 100M tokens)
#   ensemble sizes {2,3,4,5}, ensemble-mode logit, optimizer adamw
#   GPU=h100-80:1, compile=inductor, mup_base=(768, 12, 64)
#
# d24/w1536 special case (transient won't fit any single allocation at 152 cadence):
#   - Save every 305 steps (per epoch / every 40M tokens) → ~372 GB transient
#   - PERMANENT_EVERY_N_STEPS=1525 → keeps every 5th save (every 200M tokens)
#   - Storage destination: cis260009p (more headroom)
#
# Usage:
#   bash experiments/parallel/launch_grid_v2.sh
#
# Optional env overrides:
#   GRID_TAG=<tag>      override timestamp suffix
#   START_CELL=<idx>    skip first N cells (1-9), useful for resume
#   DRY_RUN=1           print sbatch commands without submitting

set -euo pipefail
cd "$(dirname "$0")/../.."

GRID_TAG="${GRID_TAG:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
START_CELL="${START_CELL:-1}"
END_CELL="${END_CELL:-9}"

ACCOUNT="${ACCOUNT:-cis260161p}"
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
CIS009=/ocean/projects/cis260009p/ymiao6/scaling/slowrun/checkpoints
CIS095=/ocean/projects/cis260095p/ymiao6/scaling/slowrun/checkpoints

# Per-cell config: "tag:L:H:W:DEST:CADENCE:PERM_STRIDE"
# - CADENCE=0 disables step ckpts (relies on per-epoch ckpts only); used for the
#   largest cell where step ckpts at cadence=305 would exactly duplicate per-epoch.
# - PERM_STRIDE only matters for step ckpts; per-epoch ckpts are pruned to every 5th
#   epoch by cleanup (PERMANENT_EVERY_N_EPOCHS=5).
# (smallest-to-largest by 2*single_strat_transient_size)
declare -a CELLS=(
    "d6_w384:6:6:384:$CIS161:152:760"
    "d12_w384:12:6:384:$CIS161:152:760"
    "d24_w384:24:6:384:$CIS161:152:760"
    "d6_w768:6:12:768:$CIS161:152:760"
    "d12_w768:12:12:768:$CIS161:152:760"
    "d24_w768:24:12:768:$CIS161:152:760"
    "d6_w1536:6:24:1536:$CIS161:152:760"
    "d12_w1536:12:24:1536:$CIS161:152:760"
    "d24_w1536:24:24:1536:$CIS009:0:0"
)

mkdir -p "$CIS161" "$CIS009" "$CIS095" experiments/logs

echo "============================================================"
echo "Grid v2 launcher  |  GRID_TAG=$GRID_TAG  start_cell=$START_CELL"
echo "  data_fraction=$DATA_FRACTION   num_epochs=$NUM_EPOCHS   constant LR (no warmdown)"
echo "  total_batch_size=$TOTAL_BATCH_SIZE   val_every_n_steps=$VAL_EVERY_N_STEPS"
echo "  ens_sizes=$ENS_SIZES_STR   ensemble_mode=$ENSEMBLE_MODE"
echo "  completep=$COMPLETEP   no_ve_projs=$NO_VE_PROJS   GPU=$GPU_SPEC compile=$COMPILE_MODE"
echo "============================================================"

submit_one() {
    # _submit exports dep_flag time_limit array_range gpu_spec name script
    local exports=$1 dep=$2 timelim=$3 arr=$4 gpu=$5 name=$6 script=$7
    local sbatch_args=(--parsable --account=$ACCOUNT --time="$timelim" --array=$arr
                       --job-name=$name --export="$exports")
    if [ -n "$gpu" ]; then sbatch_args+=(--gpus=$gpu); fi
    if [ -n "$dep" ];  then sbatch_args+=("$dep"); fi
    if [ "$DRY_RUN" = "1" ]; then
        echo "DRY: sbatch ${sbatch_args[*]} $script" >&2
        echo "DRYJOB$RANDOM"
        return
    fi
    sbatch "${sbatch_args[@]}" "$script"
}

submit_cell_chain() {
    # Args: cell_tag L H W dest cadence perm prev_cleanup_jobid
    local cell_tag=$1 L=$2 H=$3 W=$4 dest=$5 cadence=$6 perm=$7 prev=$8
    local CELL_TS="${GRID_TAG}_${cell_tag}"
    local GROUP="grid_${GRID_TAG}_${cell_tag}_df0.4"

    # Pre-create checkpoint dirs at chosen destination
    for STRAT in init_ens init_shuffle_ens; do
        mkdir -p "$dest/parallel_${STRAT}_${CELL_TS}"
    done

    local DEP=""
    if [ -n "$prev" ]; then DEP="--dependency=afterok:$prev"; fi

    local TRAIN_RANGE="0-$((2*NUM_MODELS - 1))"   # 10 train tasks
    local NUM_SIZES=$(echo $ENS_SIZES_STR | wc -w)
    local REPLAY_RANGE="0-$((2*NUM_SIZES - 1))"   # 8 replay tasks
    local CLEANUP_RANGE="0-1"                      # 2 cleanup tasks

    local exports="ALL,SHARED_TIMESTAMP=$CELL_TS,N_LAYER=$L,N_HEAD=$H,N_EMBD=$W"
    exports+=",NUM_MODELS=$NUM_MODELS,NUM_EPOCHS=$NUM_EPOCHS,TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
    exports+=",DATA_FRACTION=$DATA_FRACTION,OPTIMIZER=$OPTIMIZER,ENSEMBLE_MODE=$ENSEMBLE_MODE"
    exports+=",VAL_EVERY_N_STEPS=$VAL_EVERY_N_STEPS,COMPILE_MODE=$COMPILE_MODE"
    exports+=",COMPLETEP=$COMPLETEP,NO_VE_PROJS=$NO_VE_PROJS,NO_WARMDOWN=$NO_WARMDOWN"
    exports+=",MUP_BASE_WIDTH=$MUP_BASE_WIDTH,MUP_BASE_DEPTH=$MUP_BASE_DEPTH,MUP_BASE_HEAD_DIM=$MUP_BASE_HEAD_DIM"
    exports+=",CHECKPOINT_EVERY_N_STEPS=$cadence,CHECKPOINT_BASE=$dest"
    exports+=",WANDB_GROUP=$GROUP,ENS_SIZES_STR=$ENS_SIZES_STR,SKIP_INDIV_VAL=1,END_EPOCH=$NUM_EPOCHS"
    exports+=",PERMANENT_EVERY_N_STEPS=$perm"

    local TIME_TRAIN="${TIME_LIMIT:-12:00:00}"
    local TIME_REPLAY="${REPLAY_TIME_LIMIT:-08:00:00}"
    local TIME_CLEANUP="00:30:00"

    # 1) Training array
    local TRAIN_JOB
    TRAIN_JOB=$(submit_one "$exports" "$DEP" "$TIME_TRAIN" "$TRAIN_RANGE" \
        "$GPU_SPEC" "train_${cell_tag}" experiments/parallel/train_array.sh)
    echo "  train  array=$TRAIN_RANGE  job=$TRAIN_JOB  group=$GROUP" >&2

    # 2) Replay array
    local REPLAY_JOB
    REPLAY_JOB=$(submit_one "$exports" "--dependency=afterok:$TRAIN_JOB" \
        "$TIME_REPLAY" "$REPLAY_RANGE" "$GPU_SPEC" "replay_${cell_tag}" \
        experiments/parallel/replay_array.sh)
    echo "  replay array=$REPLAY_RANGE  job=$REPLAY_JOB  (after $TRAIN_JOB)" >&2

    # 3) Cleanup array (uses H100 too for consistency)
    local CLEANUP_JOB
    CLEANUP_JOB=$(submit_one "$exports" "--dependency=afterok:$REPLAY_JOB" \
        "$TIME_CLEANUP" "$CLEANUP_RANGE" "$GPU_SPEC" "cleanup_${cell_tag}" \
        experiments/parallel/cleanup_array.sh)
    echo "  clean  array=$CLEANUP_RANGE  job=$CLEANUP_JOB  (after $REPLAY_JOB)" >&2

    # Only the final cleanup job ID goes to stdout (captured by caller).
    echo "$CLEANUP_JOB"
}

# ---- Main loop ----
prev_cleanup=""
cell_idx=0
for cell_def in "${CELLS[@]}"; do
    cell_idx=$((cell_idx+1))
    if [ "$cell_idx" -lt "$START_CELL" ]; then continue; fi
    if [ "$cell_idx" -gt "$END_CELL" ]; then break; fi
    IFS=':' read -r tag L H W dest cadence perm <<< "$cell_def"
    proj=$(echo "$dest" | grep -oE 'cis[0-9]+p')
    echo
    echo "=== Cell $cell_idx/9: $tag (d$L w$W)  dest=$proj  cadence=$cadence  perm=$perm ==="
    new_cleanup=$(submit_cell_chain "$tag" "$L" "$H" "$W" "$dest" "$cadence" "$perm" "$prev_cleanup")
    new_cleanup=$(echo "$new_cleanup" | tail -1)
    prev_cleanup="$new_cleanup"
done

echo
echo "Submission complete. Wandb groups: grid_${GRID_TAG}_d{6,12,24}_w{384,768,1536}_df0.4"
echo "Monitor: squeue -u \$USER --format=\"%.12i %.30j %.8T %.10M %R\""
