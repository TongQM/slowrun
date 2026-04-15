#!/bin/bash
# Orchestrator: launches parallel ensemble training (10 jobs) + post-hoc replay (2 jobs).
# Generates a shared timestamp so all training jobs of one strategy share a checkpoint dir.
#
# Usage:
#   bash experiments/parallel/launch.sh
#
# This will:
#   1. Create checkpoint directories
#   2. Submit a 10-task training array (5 models × 2 strategies)
#   3. Submit a 2-task replay array with --dependency=afterok on the training job
#
# Wall time: training jobs run in parallel (~1-2h each on H100), then replay (~30m).

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
