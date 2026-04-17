#!/bin/bash
# Orchestrator for Lonestar6 (TACC): submits parallel ensemble training
# as separate per-strategy arrays (to stay within TACC's 8-job QOS limit),
# then a dependent replay array.
#
# Layout: 2 training arrays of 8 tasks (each task trains ceil(20/8)=3 models
#         sequentially), chained so init runs first then init+shuffle,
#         + 1 replay array (2 tasks, afterok on init+shuffle array)
#
# One-time setup:
#   bash experiments/env/setup_lonestar.sh
#   echo YOUR_WANDB_KEY > ~/.wandb_key && chmod 600 ~/.wandb_key
#   python prepare_data.py    # on a compute node: idev -p gpu-a100-small -A DMS26010
#
# Usage:
#   bash experiments/parallel/launch.sh
#
# Wall time: training ~1-2h per task on A100-40GB, replay ~2-4h.

set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root

SHARED_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export SHARED_TIMESTAMP

echo "Shared timestamp: $SHARED_TIMESTAMP"

# Pre-create checkpoint dirs
for STRATEGY in init_ens init_shuffle_ens; do
    DIR="checkpoints/parallel_${STRATEGY}_${SHARED_TIMESTAMP}"
    mkdir -p "$DIR"
    echo "  Created $DIR"
done

mkdir -p experiments/logs

# Submit init ensemble (20 models, max 8 concurrent)
echo
echo "Submitting init ensemble (8 tasks × ~3 models each)..."
INIT_JOB=$(sbatch --parsable \
    --array=0-7 \
    --export=ALL,SHARED_TIMESTAMP,ENSEMBLE_TYPE=init,STRATEGY_NAME=init_ens \
    experiments/parallel/train_array.sh 2>/dev/null | tail -1)
echo "  Init training job ID: $INIT_JOB"

# Submit init+shuffle ensemble (chained after init to stay within QOS submit limit)
echo
echo "Submitting init+shuffle ensemble (8 tasks × ~3 models each, after init)..."
SHUFFLE_JOB=$(sbatch --parsable \
    --array=0-7 \
    --dependency=afterany:${INIT_JOB} \
    --export=ALL,SHARED_TIMESTAMP,ENSEMBLE_TYPE=init_shuffle,STRATEGY_NAME=init_shuffle_ens \
    experiments/parallel/train_array.sh 2>/dev/null | tail -1)
echo "  Init+shuffle training job ID: $SHUFFLE_JOB"

# Submit replay array (depends on BOTH training arrays completing)
echo
echo "Submitting replay array (2 jobs, depends on $INIT_JOB and $SHUFFLE_JOB)..."
REPLAY_JOB=$(sbatch --parsable \
    --dependency=afterok:${SHUFFLE_JOB} \
    --export=ALL,SHARED_TIMESTAMP \
    experiments/parallel/replay_array.sh 2>/dev/null | tail -1)
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
echo "Wandb project: slowrun_lonestar"
echo "Wandb group:   parallel_d12_w768_df0.2_${SHARED_TIMESTAMP}"
