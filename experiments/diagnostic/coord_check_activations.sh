#!/bin/bash
#SBATCH --job-name=coordact
#SBATCH --partition=GPU-shared
#SBATCH --account=cis260161p
#SBATCH --gpus=h100-80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/logs/coordact_%j.out
#SBATCH --error=experiments/logs/coordact_%j.err
# Activation coordinate check for CompleteP -- see coord_check_activations.py.
# Cheap (minutes): 8 cells x ~9 forward/backward steps, no HP grid.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
module load anaconda3/2024.10-1
conda activate slowrun
WD="${WD:-0.0}"
python experiments/diagnostic/coord_check_activations.py \
    --weight-decay "$WD" \
    --out "experiments/diagnostic/coord_check_activations_wd${WD}.json"
