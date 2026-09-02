#!/bin/bash
# Launch the CompleteP LR-transfer coordinate check (72-task array).
# Iso-aspect-ratio ladder (w/L=64) x lr_multiplier{0.0625..2.0} x 3 seeds,
# 1 epoch / df=0.2 / wd=0 / constant LR. ~10-15 SU. Compute on cis260161p;
# transient 1-epoch ckpts -> cis260095p (deletable; analysis reads .out logs).
#
#   DRY_RUN=1 bash experiments/parallel/launch_coord_check.sh   # preview
#   bash experiments/parallel/launch_coord_check.sh             # submit
set -euo pipefail
cd "$(dirname "$0")/../.."

GRID_TAG="${GRID_TAG:-20260618}"
CKPT_BASE="${CKPT_BASE:-/ocean/projects/cis260095p/ymiao6/scaling/slowrun/checkpoints}"
DRY_RUN="${DRY_RUN:-0}"
LR_LIST="${LR_LIST:-0.0625 0.125 0.25 0.5 1.0 2.0}"
mkdir -p "$CKPT_BASE" experiments/logs

NSEED=3
read -ra LRS <<< "$LR_LIST"
NTASK=$(( 4 * ${#LRS[@]} * NSEED ))   # 4 sizes x #LR x 3 seeds

exports="ALL,GRID_TAG=$GRID_TAG,CKPT_BASE=$CKPT_BASE,LR_LIST=$LR_LIST"
sb=(--parsable --array=0-$((NTASK-1)) --export="$exports" experiments/parallel/coord_check.sh)

echo "Coordinate check: 4 sizes x ${#LRS[@]} LR x 3 seeds = $NTASK tasks"
echo "  ladder: d6/w384, d12/w768, d18/w1152, d24/w1536  (w/L=64, head_dim=64)"
echo "  lr_multiplier: $LR_LIST"
echo "  GRID_TAG=$GRID_TAG  CKPT_BASE=$CKPT_BASE"
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY: sbatch ${sb[*]}"
else
    JOB=$(sbatch "${sb[@]}")
    echo "submitted array job=$JOB  (wandb group coordcheck_${GRID_TAG})"
fi
