#!/bin/bash
#SBATCH --job-name=wavg
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260161p
#SBATCH --gpus=h100-80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/logs/%x_%A_%a.out
#SBATCH --error=experiments/logs/%x_%A_%a.err
# Array over cells. Env: CKPT_BASE, TAG (checkpoint dir suffix per task via CELLS), OUT_DIR.
set -euo pipefail
module load anaconda3/2024.10-1
conda activate slowrun
cd /ocean/projects/cis260161p/ymiao6/scaling/slowrun
export PYTHONUNBUFFERED=1 TIKTOKEN_CACHE_DIR=/ocean/projects/cis260161p/ymiao6/.tiktoken_cache
read -r -a CELLS_ARR <<< "$CELLS"
CELL=${CELLS_ARR[$SLURM_ARRAY_TASK_ID]}
mkdir -p "$OUT_DIR"
python experiments/parallel/eval_weight_avg.py --checkpoint-dir "$CKPT_BASE/$CELL" --models $MODELS \
    --epochs $EPOCHS --soup-sizes 2 4 --ckpt-windows 2 4 --out "$OUT_DIR/$CELL.json"
echo "Done: $CELL"
