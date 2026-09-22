#!/bin/bash
#SBATCH --job-name=prep_wikitext
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --account=cis260161p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --error=experiments/logs/%x_%j.err
set -euo pipefail
module load anaconda3/2024.10-1
conda activate slowrun
cd /ocean/projects/cis260161p/ymiao6/scaling/slowrun
export TIKTOKEN_CACHE_DIR=/ocean/projects/cis260161p/ymiao6/.tiktoken_cache
python prepare_data.py --dataset wikitext --train_tokens 100000000 --val_tokens 10000000 --local_dir wikitext_data
ls -la wikitext_data
