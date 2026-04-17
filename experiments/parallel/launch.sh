#!/bin/bash
# Orchestrator for Lonestar6 (TACC): submits parallel ensemble training.
# TACC QOS limits to 8 submitted jobs per user, so we submit one strategy
# at a time (8 tasks, each training ceil(20/8)=3 models sequentially).
#
# Usage:
#   # Step 1: launch init ensemble
#   bash experiments/parallel/launch.sh init
#
#   # Step 2: when init finishes, launch init+shuffle ensemble
#   bash experiments/parallel/launch.sh shuffle
#
#   # Step 3: when shuffle finishes, launch replay for both
#   bash experiments/parallel/launch.sh replay
#
#   # Or launch all at once on a cluster without strict QOS limits:
#   bash experiments/parallel/launch.sh all
#
# One-time setup:
#   bash experiments/env/setup_lonestar.sh
#   echo YOUR_WANDB_KEY > ~/.wandb_key && chmod 600 ~/.wandb_key
#   python prepare_data.py

set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root

STAGE="${1:-}"
if [ -z "$STAGE" ]; then
    echo "Usage: bash experiments/parallel/launch.sh {init|shuffle|replay|all}"
    echo
    echo "Run stages sequentially to stay within TACC's 8-job QOS limit."
    echo "Each stage waits for you to confirm the previous one completed."
    exit 1
fi

# Use existing timestamp or generate a new one
if [ -z "${SHARED_TIMESTAMP:-}" ]; then
    SHARED_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
fi
export SHARED_TIMESTAMP

# Save/load timestamp so stages share the same checkpoint dirs
TS_FILE="experiments/logs/.last_timestamp"
mkdir -p experiments/logs

if [ "$STAGE" = "init" ] || [ "$STAGE" = "all" ]; then
    echo "$SHARED_TIMESTAMP" > "$TS_FILE"
    echo "Shared timestamp: $SHARED_TIMESTAMP (saved to $TS_FILE)"
else
    if [ -f "$TS_FILE" ]; then
        SHARED_TIMESTAMP=$(cat "$TS_FILE")
        export SHARED_TIMESTAMP
        echo "Using saved timestamp: $SHARED_TIMESTAMP (from $TS_FILE)"
    else
        echo "ERROR: No saved timestamp. Run 'launch.sh init' first."
        exit 1
    fi
fi

# Pre-create checkpoint dirs
for STRATEGY in init_ens init_shuffle_ens; do
    DIR="checkpoints/parallel_${STRATEGY}_${SHARED_TIMESTAMP}"
    mkdir -p "$DIR"
done

submit_training() {
    local ENS_TYPE="$1"
    local STRAT_NAME="$2"
    local EXTRA_ARGS="${3:-}"

    echo "Submitting $STRAT_NAME (8 tasks × ~3 models each)..."
    JOB=$(sbatch --parsable \
        --array=0-7 \
        $EXTRA_ARGS \
        --export=ALL,SHARED_TIMESTAMP=$SHARED_TIMESTAMP,ENSEMBLE_TYPE=$ENS_TYPE,STRATEGY_NAME=$STRAT_NAME \
        experiments/parallel/train_array.sh 2>/dev/null | tail -1)
    echo "  Job ID: $JOB"
    echo "$JOB"
}

submit_replay() {
    local EXTRA_ARGS="${1:-}"

    echo "Submitting replay (2 tasks)..."
    JOB=$(sbatch --parsable \
        $EXTRA_ARGS \
        --export=ALL,SHARED_TIMESTAMP=$SHARED_TIMESTAMP \
        experiments/parallel/replay_array.sh 2>/dev/null | tail -1)
    echo "  Replay job ID: $JOB"
}

case "$STAGE" in
    init)
        submit_training "init" "init_ens"
        echo
        echo "When all 8 tasks complete, run:"
        echo "  bash experiments/parallel/launch.sh shuffle"
        ;;
    shuffle)
        submit_training "init_shuffle" "init_shuffle_ens"
        echo
        echo "When all 8 tasks complete, run:"
        echo "  bash experiments/parallel/launch.sh replay"
        ;;
    replay)
        submit_replay
        echo
        echo "Replay will compute per-model + ensemble val for both strategies."
        ;;
    all)
        INIT_JOB=$(submit_training "init" "init_ens")
        SHUFFLE_JOB=$(submit_training "init_shuffle" "init_shuffle_ens")
        submit_replay "--dependency=afterok:${SHUFFLE_JOB}"
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Usage: bash experiments/parallel/launch.sh {init|shuffle|replay|all}"
        exit 1
        ;;
esac

echo
echo "Monitor:     squeue -u \$USER"
echo "Checkpoints: checkpoints/parallel_{init,init_shuffle}_ens_${SHARED_TIMESTAMP}/"
echo "Wandb:       project=slowrun_lonestar  group=parallel_d12_w768_df0.2_${SHARED_TIMESTAMP}"
