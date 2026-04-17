#!/bin/bash
# Orchestrator for Lonestar6 (TACC): submits 10 single-GPU training tasks
# (5 models × 2 ensemble strategies) and a dependent 2-task replay array.
# All training tasks of one strategy share a checkpoint dir keyed by a shared
# timestamp.
#
# One-time setup:
#   bash experiments/env/setup_lonestar.sh
#   echo YOUR_WANDB_KEY > ~/.wandb_key && chmod 600 ~/.wandb_key
#   python prepare_data.py    # on a compute node: idev -p gpu-a100-small -A DMS26010
#
# Usage:
#   bash experiments/parallel/launch.sh
#
# Wall time: training ~1-2h per task on A100-40GB, replay ~30m.

set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root

SHARED_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export SHARED_TIMESTAMP

echo "Shared timestamp: $SHARED_TIMESTAMP"

# Pre-create checkpoint dirs (training jobs all write into these)
for STRATEGY in init_ens init_shuffle_ens; do
    DIR="checkpoints/parallel_${STRATEGY}_${SHARED_TIMESTAMP}"
    mkdir -p "$DIR"
    echo "  Created $DIR"
done

mkdir -p experiments/logs

# Submit training array
echo
echo "Submitting training array (10 jobs)..."
TRAIN_JOB=$(sbatch --parsable --export=ALL,SHARED_TIMESTAMP \
    experiments/parallel/train_array.sh)
echo "  Training array job ID: $TRAIN_JOB"

# Submit replay array with dependency
echo
echo "Submitting replay array (2 jobs, depends on $TRAIN_JOB)..."
REPLAY_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB \
    --export=ALL,SHARED_TIMESTAMP \
    experiments/parallel/replay_array.sh)
echo "  Replay array job ID: $REPLAY_JOB"

echo
echo "Submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 30 squeue -u \$USER"
echo
echo "Checkpoint dirs:"
echo "  checkpoints/parallel_init_ens_${SHARED_TIMESTAMP}/"
echo "  checkpoints/parallel_init_shuffle_ens_${SHARED_TIMESTAMP}/"
echo
echo "Wandb group: parallel_d12_w768_df0.2_${SHARED_TIMESTAMP}"
