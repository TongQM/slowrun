#!/bin/bash
#SBATCH --job-name=q2_bootstrap
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260161p
#SBATCH --gpus=h100-80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --array=0-19
#SBATCH --output=experiments/logs/%x_%A_%a.out
#SBATCH --error=experiments/logs/%x_%A_%a.err
#
# Bootstrap fused replay for Q2 (model-resampling).
# Array layout: 20 tasks = 10 bootstrap iterations × 2 strategies.
#   array_id 0..9   → init_ens         iter 0..9
#   array_id 10..19 → init_shuffle_ens iter 0..9
#
# Reads the SAME read-only checkpoints as the original Q2 runs. Writes results
# to disk only (no wandb), so original wandb runs are NOT touched.
#
# Output: experiments/figures/02_ensemble_scaling/bootstrap/{strategy}_iter{idx:03d}.npz

set -euo pipefail

module load anaconda3/2024.10-1
conda activate slowrun

cd /ocean/projects/cis260161p/ymiao6/scaling/slowrun

mkdir -p experiments/logs experiments/figures/02_ensemble_scaling/bootstrap
export PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR=/ocean/projects/cis260161p/ymiao6/.tiktoken_cache

B_PER_STRAT=10

if [ "$SLURM_ARRAY_TASK_ID" -lt "$B_PER_STRAT" ]; then
    STRATEGY="init_ens"
    ITER=$SLURM_ARRAY_TASK_ID
else
    STRATEGY="init_shuffle_ens"
    ITER=$((SLURM_ARRAY_TASK_ID - B_PER_STRAT))
fi

# Find the Q2 checkpoint dir for this strategy. Symlink at cis260095p points to
# the actual data on cis260161p (post Q2 ckpt move).
CKPT_DIR=/ocean/projects/cis260095p/ymiao6/scaling/slowrun/checkpoints/parallel_${STRATEGY}_q2_20260502_234110_d12_w768_df0.2

echo "============================================================"
echo "Bootstrap iter $ITER  strategy=$STRATEGY  ckpt_dir=$CKPT_DIR"
echo "============================================================"

python experiments/parallel/replay_bootstrap.py \
    --checkpoint-dir "$CKPT_DIR" \
    --strategy "$STRATEGY" \
    --num-models 20 \
    --ens-sizes 2 5 10 15 20 \
    --num-epochs 40 \
    --bootstrap-iter "$ITER"

echo "Done: $STRATEGY iter $ITER (exit $?)"
