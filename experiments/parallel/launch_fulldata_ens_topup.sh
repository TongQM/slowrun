#!/bin/bash
# Extend the df=1.0 base-cell ensemble from E<=5 to E<=10 by training FIVE MORE models.
#
# WHY: the joint scaling law (experiments/analysis/expt_joint_scaling_law.py) fits the
# ensemble term as a strict 1/E on the overfit component, but E is only ever crossed
# with data size P at df=0.2 -- the df sweep is E=1 only. So 1/E is fit at 20M tokens
# and EXTRAPOLATED to 100M, where it over-predicts: at df=1.0 it says E=5 -> 3.479 but
# we measure 3.644, and it says optimal horizon grows as E^0.665 while we measure
# E^0.22. Widening E at df=1.0 gives a second P value with broad E coverage, which is
# exactly what identifies the E x P interaction.
#
# WE DO NOT RETRAIN MODELS 0-4. Their per-epoch checkpoints survived the pruning:
#   .../checkpoints/parallel_init_shuffle_ens_fulldata_20260528_150057_d12_w768
#   (models 0-4, 40 epochs each, 546 MB per ckpt, 107 GB per strategy)
# Seeds are `seeds = [42 + i for i in range(num_models)]` (train.py:2132) -- the seed
# depends ONLY on the model index, not on num_models. And data_seed = seed for
# init_shuffle / 42 for init (train.py:1253). So models 5-9 trained here with
# NUM_MODELS=10 are exactly the models a 10-ensemble would have had, and they compose
# with the existing 0-4. train_array.sh also skips/resumes any model already complete,
# so re-running this is safe.
#
# Config below is copied from the original run's config.json (wd=0, no_warmdown=True,
# completep, no_ve_projs, adamw, lr_mult 0.25, batch 131072, 40 ep, df=1.0).
#
# CHECKPOINT_EVERY_N_STEPS=0 on purpose: the surviving models 0-4 have per-epoch ckpts
# ONLY (their 20M-resolution step ckpts were pruned). Replay needs every model present
# at the same point, so per-epoch is the common grid -- finer saves here would be
# unusable and would cost ~5x the disk.
#
#   STRATEGIES=shuffle  -> tasks 15-19 (init_shuffle only)   [default, ~51 SU, +107 GB]
#   STRATEGIES=both     -> tasks 5-9,15-19                   [~102 SU, +214 GB]
#
#   DRY_RUN=1 bash experiments/parallel/launch_fulldata_ens_topup.sh
#   bash experiments/parallel/launch_fulldata_ens_topup.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

SHARED_TIMESTAMP="${SHARED_TIMESTAMP:-fulldata_20260528_150057}"   # the EXISTING dir
STRATEGIES="${STRATEGIES:-shuffle}"
DRY_RUN="${DRY_RUN:-0}"

ACCOUNT="${ACCOUNT:-cis260161p}"
GPU_SPEC=h100-80:1
TIME_LIMIT="${TIME_LIMIT:-10:00:00}"       # measured 5:10 per model at this cell
COMPILE_MODE=inductor

# --- exactly the original fulldata run (from its config.json) ---
N_LAYER=12; N_HEAD=12; N_EMBD=768
COMPLETEP=1
NO_VE_PROJS=1
NO_WARMDOWN=1                  # wd=0 arm: constant LR, no cooldown
WEIGHT_DECAY=0.0
TOTAL_BATCH_SIZE=131072
NUM_EPOCHS=40
DATA_FRACTION=1.0
OPTIMIZER=adamw
ENSEMBLE_MODE=logit
VAL_EVERY_N_STEPS=152
CHECKPOINT_EVERY_N_STEPS=0     # per-epoch only -- matches surviving models 0-4
MUP_BASE_WIDTH=768
MUP_BASE_DEPTH=12
MUP_BASE_HEAD_DIM=64

NUM_MODELS=10                  # <-- the only structural change: 5 -> 10
CKPT_BASE=/ocean/projects/cis260161p/ymiao6/scaling/slowrun/checkpoints

case "$STRATEGIES" in
    shuffle) ARRAY="15-19" ;;                 # init_shuffle models 5..9
    both)    ARRAY="5-9,15-19" ;;             # + init models 5..9
    *) echo "STRATEGIES must be 'shuffle' or 'both'"; exit 1 ;;
esac

GROUP="fulldata_ens_topup_d${N_LAYER}_w${N_EMBD}_df${DATA_FRACTION}"

exports="ALL,N_LAYER=$N_LAYER,N_HEAD=$N_HEAD,N_EMBD=$N_EMBD"
exports+=",SHARED_TIMESTAMP=$SHARED_TIMESTAMP,WANDB_GROUP=$GROUP"
exports+=",NUM_MODELS=$NUM_MODELS,NUM_EPOCHS=$NUM_EPOCHS,TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
exports+=",DATA_FRACTION=$DATA_FRACTION,OPTIMIZER=$OPTIMIZER,ENSEMBLE_MODE=$ENSEMBLE_MODE"
exports+=",VAL_EVERY_N_STEPS=$VAL_EVERY_N_STEPS,COMPILE_MODE=$COMPILE_MODE"
exports+=",COMPLETEP=$COMPLETEP,NO_VE_PROJS=$NO_VE_PROJS,NO_WARMDOWN=$NO_WARMDOWN"
exports+=",WEIGHT_DECAY=$WEIGHT_DECAY"
exports+=",MUP_BASE_WIDTH=$MUP_BASE_WIDTH,MUP_BASE_DEPTH=$MUP_BASE_DEPTH,MUP_BASE_HEAD_DIM=$MUP_BASE_HEAD_DIM"
exports+=",CHECKPOINT_EVERY_N_STEPS=$CHECKPOINT_EVERY_N_STEPS,CHECKPOINT_BASE=$CKPT_BASE"

sb=(--parsable --account=$ACCOUNT --gpus=$GPU_SPEC --time="$TIME_LIMIT"
    --array="$ARRAY" --job-name="fdtopup_d${N_LAYER}_w${N_EMBD}" --export="$exports")

if [ "$DRY_RUN" = "1" ]; then
    echo "DRY: sbatch ${sb[*]} experiments/parallel/train_array.sh"
    echo "     -> NUM_MODELS=$NUM_MODELS, array=$ARRAY"
    echo "     -> init_shuffle models: $(echo $ARRAY | tr ',' ' ')  (task%10 = model idx)"
    echo "     -> writes into $CKPT_BASE/parallel_*_${SHARED_TIMESTAMP}_d${N_LAYER}_w${N_EMBD}"
else
    JOB=$(sbatch "${sb[@]}" experiments/parallel/train_array.sh)
    echo "submitted top-up: job=$JOB  array=$ARRAY  strategies=$STRATEGIES"
    echo "  -> after it finishes, replay E in {6..10} over models 0-9 (per-epoch grid)"
fi
