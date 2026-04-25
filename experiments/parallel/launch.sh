#!/bin/bash
# Orchestrator: launches parallel ensemble training (40 jobs) + post-hoc replay (10 jobs).
# Generates a shared timestamp so all training jobs of one strategy share a checkpoint dir.
#
# Usage:
#   bash experiments/parallel/launch.sh                    # fresh run (default: any GPU, eager)
#   SHARED_TIMESTAMP=<ts> bash experiments/parallel/launch.sh   # reuse existing checkpoints
#
# Optional env overrides:
#   GPU_SPEC=h100-80:1          # SLURM --gpus value (default: "1" = any GPU)
#                               # e.g. "v100-32:1", "l40s-48:1", "h100-80:1"
#   COMPILE_MODE=eager          # torch.compile mode (default: eager; inductor only on H100)
#   TIME_LIMIT=06:00:00         # SLURM --time for training jobs (default 6h, safe for V100 eager)
#   REPLAY_TIME_LIMIT=10:00:00  # SLURM --time for replay jobs (default 10h,
#                               # safe for 20-model ensemble × 30 epochs)
#
# This will:
#   1. Create checkpoint directories (or reuse existing if SHARED_TIMESTAMP is set)
#   2. Submit a 40-task training array (20 models × 2 strategies).
#      Training jobs skip themselves if their final checkpoint already exists.
#   3. Submit a 10-task replay array (2 strategies × 5 ensemble sizes [2,4,8,16,20])
#      with --dependency=afterok on the training job.
#      Size=1 omitted (per-model val already logged during training).

set -euo pipefail

cd "$(dirname "$0")/../.."  # repo root

if [ -z "${SHARED_TIMESTAMP:-}" ]; then
    SHARED_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
fi
export SHARED_TIMESTAMP

GPU_SPEC="${GPU_SPEC:-1}"          # "1" = any GPU SLURM picks; "h100-80:1" for H100 only
COMPILE_MODE="${COMPILE_MODE:-eager}"  # "inductor" only safe on H100 w/ torch 2.8
TIME_LIMIT="${TIME_LIMIT:-06:00:00}"
REPLAY_TIME_LIMIT="${REPLAY_TIME_LIMIT:-10:00:00}"
ACCOUNT="${ACCOUNT:-cis260095p}"   # SLURM allocation to charge

# --- Model + experiment config (pass-through to train/replay arrays) ---
N_LAYER="${N_LAYER:-12}"
N_HEAD="${N_HEAD:-12}"
N_EMBD="${N_EMBD:-768}"
NUM_MODELS="${NUM_MODELS:-20}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
ENS_SIZES_STR="${ENS_SIZES_STR:-2 4 8 16 20}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"

export COMPILE_MODE N_LAYER N_HEAD N_EMBD NUM_MODELS NUM_EPOCHS ENS_SIZES_STR TOTAL_BATCH_SIZE

# Compute array sizes
TRAIN_ARRAY_MAX=$((2 * NUM_MODELS - 1))      # 2 strategies × NUM_MODELS
NUM_SIZES=$(echo $ENS_SIZES_STR | wc -w)
REPLAY_ARRAY_MAX=$((2 * NUM_SIZES - 1))      # 2 strategies × NUM_SIZES

echo "Shared timestamp: $SHARED_TIMESTAMP"
echo "GPU spec:         $GPU_SPEC"
echo "Compile mode:     $COMPILE_MODE"
echo "Time limits:      train=$TIME_LIMIT  replay=$REPLAY_TIME_LIMIT"
echo "Account:          $ACCOUNT"
echo "Model:            d${N_LAYER}_h${N_HEAD}_w${N_EMBD}  ${NUM_MODELS} models × ${NUM_EPOCHS} epochs  (total_batch=${TOTAL_BATCH_SIZE})"
echo "Ensemble sizes:   ${ENS_SIZES_STR}"
echo "Array ranges:     train=0-${TRAIN_ARRAY_MAX}  replay=0-${REPLAY_ARRAY_MAX}"

# Pre-create checkpoint dirs (training jobs all write into these)
for STRATEGY in init_ens init_shuffle_ens; do
    DIR="checkpoints/parallel_${STRATEGY}_${SHARED_TIMESTAMP}"
    mkdir -p "$DIR"
    if [ -n "$(ls -A $DIR 2>/dev/null)" ]; then
        N_EXISTING=$(ls $DIR/model_*_epoch_${NUM_EPOCHS}.pt 2>/dev/null | wc -l)
        echo "  Reusing $DIR ($N_EXISTING models already completed)"
    else
        echo "  Created $DIR"
    fi
done

mkdir -p experiments/logs

# Submit training array
echo
echo "Submitting training array ($((TRAIN_ARRAY_MAX + 1)) jobs; done ones auto-skip)..."
TRAIN_JOB=$(sbatch --parsable \
    --account=$ACCOUNT \
    --gpus=$GPU_SPEC \
    --time=$TIME_LIMIT \
    --array=0-${TRAIN_ARRAY_MAX} \
    --export=ALL,SHARED_TIMESTAMP,COMPILE_MODE,VAL_EVERY_N_STEPS,N_LAYER,N_HEAD,N_EMBD,NUM_MODELS,NUM_EPOCHS,TOTAL_BATCH_SIZE \
    experiments/parallel/train_array.sh)
echo "  Training array job ID: $TRAIN_JOB"

# Submit replay array with dependency
echo
echo "Submitting replay array ($((REPLAY_ARRAY_MAX + 1)) jobs, depends on $TRAIN_JOB)..."
REPLAY_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB \
    --account=$ACCOUNT \
    --gpus=$GPU_SPEC \
    --time=$REPLAY_TIME_LIMIT \
    --array=0-${REPLAY_ARRAY_MAX} \
    --export=ALL,SHARED_TIMESTAMP,SKIP_INDIV_VAL,END_EPOCH,N_LAYER,N_EMBD,NUM_EPOCHS,ENS_SIZES_STR \
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
echo "Wandb group: parallel_d${N_LAYER}_w${N_EMBD}_df0.2_${SHARED_TIMESTAMP}"
