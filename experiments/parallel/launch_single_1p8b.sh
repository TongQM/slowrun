#!/bin/bash
# launch_single_1p8b.sh — train ONE ~1.8B looped-transformer model (decoder layers
# 15-24 looped 5x) on 100M FineWeb tokens. CompleteP ON, hybrid Muon+AdamW, ve-projs
# ON, constant LR (--no-warmdown). Single standalone model (num-models=1), single GPU.
#
# Schedule (DUPE_FRACTION):
#   0.0 = q0       : looping active from epoch 1 (the whole run)
#   0.5 = baseline : looping active only for the last 50% of epochs (epoch 6 of 10)
#   1.0            : looping disabled (plain 30-layer model)
# dupe activates at epoch ceil(DUPE_FRACTION*num_epochs)+1.
#
# NOT included (per request): MTP auxiliary loss, attention gating.
# LR schedule: --no-warmdown (constant 0.25x peak LRs); makes 10->32-epoch extend exact.
#
# Cost: ~10k tok/s * looping => ~28-35 h for 10 epochs (fits one 48h alloc, no resume).
#   SU ~55-70 (cis260095p). Ckpts ~7 GB/epoch on cis260161p.
#
# Usage:
#   bash experiments/parallel/launch_single_1p8b.sh                  # q0, 10 epochs
#   DRY_RUN=1 bash experiments/parallel/launch_single_1p8b.sh
#   DUPE_FRACTION=0.5 NUM_EPOCHS=10 bash experiments/parallel/launch_single_1p8b.sh   # baseline
#   NUM_EPOCHS=32 bash experiments/parallel/launch_single_1p8b.sh    # full recipe (will need resume)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

# ---- recipe (overridable) ----
N_LAYER="${N_LAYER:-30}"; N_HEAD="${N_HEAD:-16}"; N_EMBD="${N_EMBD:-2048}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
DF="${DF:-1.0}"
OPTIMIZER="${OPTIMIZER:-hybrid}"
LR_MULTIPLIER="${LR_MULTIPLIER:-0.25}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.3}"
DUPE_START="${DUPE_START:-15}"; DUPE_END="${DUPE_END:-25}"   # decoder layers 15-24, 5 passes
DUPE_FRACTION="${DUPE_FRACTION:-0.0}"                        # 0.0=q0, 0.5=baseline
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-131072}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"
VAL_EVERY_N_STEPS="${VAL_EVERY_N_STEPS:-152}"
COMPILE_MODE="${COMPILE_MODE:-inductor}"
# completep ON (required); ve-projs ON (do NOT pass --no-ve-projs); constant LR
MUP_BASE_WIDTH="${MUP_BASE_WIDTH:-768}"; MUP_BASE_DEPTH="${MUP_BASE_DEPTH:-12}"; MUP_BASE_HEAD_DIM="${MUP_BASE_HEAD_DIM:-64}"
# SLURM / storage
ACCOUNT="${ACCOUNT:-cis260095p}"
GPU_SPEC="${GPU_SPEC:-h100-80:1}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-checkpoints}"
TIME_LIMIT="${TIME_LIMIT:-48:00:00}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

# schedule label
case "$DUPE_FRACTION" in
  0.0|0) SCHED=q0 ;; 1.0|1) SCHED=noloop ;; *) SCHED="loop${DUPE_FRACTION}" ;;
esac
RUN_ID="single_1p8b_${SCHED}_${TAG}"
GROUP="$RUN_ID"
dupe_epoch=$(python3 -c "import math;print(math.ceil($DUPE_FRACTION*$NUM_EPOCHS)+1)" 2>/dev/null || echo "?")

echo "============================================================"
echo "single 1.8B looped model  RUN_ID=$RUN_ID"
echo "  arch d${N_LAYER}/w${N_EMBD}/h${N_HEAD}  loop layers ${DUPE_START}-$((DUPE_END-1)) (5 passes)"
echo "  schedule=$SCHED (dupe_fraction=$DUPE_FRACTION -> activates epoch $dupe_epoch of $NUM_EPOCHS)"
echo "  optimizer=$OPTIMIZER lr_mult=$LR_MULTIPLIER wd=$WEIGHT_DECAY  completep=ON ve_projs=ON warmdown=OFF"
echo "  batch=$TOTAL_BATCH_SIZE (device $DEVICE_BATCH_SIZE)  epochs=$NUM_EPOCHS df=$DF"
echo "  account=$ACCOUNT  ckpt_base=$CHECKPOINT_BASE  time=$TIME_LIMIT"
echo "============================================================"

mkdir -p "$CHECKPOINT_BASE/$RUN_ID" experiments/logs

read -r -d '' EXPORTS <<EOF || true
ALL,RUN_ID=$RUN_ID,WANDB_GROUP=$GROUP,N_LAYER=$N_LAYER,N_HEAD=$N_HEAD,N_EMBD=$N_EMBD,NUM_EPOCHS=$NUM_EPOCHS,DF=$DF,OPTIMIZER=$OPTIMIZER,LR_MULTIPLIER=$LR_MULTIPLIER,WEIGHT_DECAY=$WEIGHT_DECAY,DUPE_START=$DUPE_START,DUPE_END=$DUPE_END,DUPE_FRACTION=$DUPE_FRACTION,TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE,DEVICE_BATCH_SIZE=$DEVICE_BATCH_SIZE,VAL_EVERY_N_STEPS=$VAL_EVERY_N_STEPS,COMPILE_MODE=$COMPILE_MODE,MUP_BASE_WIDTH=$MUP_BASE_WIDTH,MUP_BASE_DEPTH=$MUP_BASE_DEPTH,MUP_BASE_HEAD_DIM=$MUP_BASE_HEAD_DIM,CHECKPOINT_BASE=$CHECKPOINT_BASE
EOF

CMD=(sbatch --parsable --account="$ACCOUNT" --gpus="$GPU_SPEC" --time="$TIME_LIMIT"
     --job-name="$RUN_ID" --export="$EXPORTS" experiments/parallel/single_1p8b.sh)

if [ "$DRY_RUN" = "1" ]; then
    echo "DRY: ${CMD[*]}"
    echo; echo "--- torchrun line that single_1p8b.sh will run ---"
    cat <<EOF
torchrun --standalone --nproc_per_node=1 -- unlimited/train.py \\
  --n_layer=$N_LAYER --n_head=$N_HEAD --n_embd=$N_EMBD --num-models=1 \\
  --num-epochs=$NUM_EPOCHS --data-fraction=$DF \\
  --optimizer=$OPTIMIZER --lr_multiplier=$LR_MULTIPLIER --weight-decay=$WEIGHT_DECAY \\
  --dupe-layers-start=$DUPE_START --dupe-layers-end=$DUPE_END --dupe-fraction=$DUPE_FRACTION \\
  --completep --mup-base-width=$MUP_BASE_WIDTH --mup-base-depth=$MUP_BASE_DEPTH --mup-base-head-dim=$MUP_BASE_HEAD_DIM \\
  --total-batch-size=$TOTAL_BATCH_SIZE --device-batch-size=$DEVICE_BATCH_SIZE \\
  --val-every-n-steps=$VAL_EVERY_N_STEPS --compile-mode=$COMPILE_MODE --no-warmdown \\
  --ensemble-mode=logit --checkpoint-base=$CHECKPOINT_BASE \\
  --resume=$RUN_ID --run=$RUN_ID --wandb_group=$GROUP
EOF
    exit 0
fi

JOB=$("${CMD[@]}")
echo "submitted job=$JOB  ($RUN_ID)"
echo "monitor: tail -f experiments/logs/${RUN_ID}_${JOB}.out  ;  squeue -j $JOB"
