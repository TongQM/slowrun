#!/bin/bash
# P1-FILL: complete the tuned-weight-decay capacity curve at df=1.0, w768.
#
# The wd x size probe so far (cooldown ON, E=1) measured only three depths:
#     L=12 -> lambda_opt=0.1 (L*=3.5695)
#     L=24 -> lambda_opt=0.2 (L*=3.4956)
#     L=48 -> lambda_opt=0.3 (L*=3.4667)
# The unregularized (lambda=0, constant-LR) baseline has six depths
# {6,12,18,24,48,60}, so the tuned curve is missing L in {6, 18, 60}. This fills
# them, which both completes the capacity figure and re-tests the lambda_opt law.
#
# Competing laws for lambda_opt(L), both consistent with the three points above
# only if you squint:
#   (i)  LINEAR  lambda_opt = 0.1 * L/12          -> predicts 0.4 at L=48
#        ALREADY FALSIFIED by launch_wd_size_ext.sh's own criterion: wd=0.4 gave
#        3.5178, worse than wd=0.3's 3.4667 (and wd=0.5 gave 3.6248).
#   (ii) LOG     lambda_opt = 0.1 * log2(L/6)     -> 0.1, 0.2, 0.3 at 12, 24, 48
#        i.e. an EXACT fit to all three measured optima.
#
# Law (ii) predicts   L=6 -> 0.0,  L=18 -> 0.158,  L=60 -> 0.332.
# Law (i)  predicts   L=6 -> 0.05, L=18 -> 0.15,   L=60 -> 0.50.
# The two laws are nearly indistinguishable at L=18 but differ sharply at L=60
# (0.33 vs 0.50), so **L=60 is the discriminating cell** -- its grid brackets both.
#
# Recipe is IDENTICAL to launch_wd_size.sh (cooldown ON, df=1.0, completep,
# no-ve-projs, adamw, batch 131072, 40 epochs, per-epoch ckpts only, E=1
# single model at index 0) so the new points drop straight onto the same curve.
#
# Storage: per-epoch ckpts only, deletable once export-valloss has parsed the
# logs. Routed to cis260161p (~2 TB free); cis260095p is down to ~74 GB, which
# would not hold even one d60 run (72 GB).
#
# Cost (measured single-model runtimes x 2 SU/H100-hr):
#   d6  ~3.5h x 3 =  21 SU     d18 ~7.0h x 3 =  42 SU     d60 ~20.5h x 3 = 123 SU
#   total ~186 SU
#
#   DRY_RUN=1 bash experiments/parallel/launch_wd_size_fill.sh   # preview
#   bash experiments/parallel/launch_wd_size_fill.sh             # submit
set -euo pipefail
cd "$(dirname "$0")/../.."

GRID_TAG="${GRID_TAG:-wdfill_$(date +%Y%m%d)}"
DRY_RUN="${DRY_RUN:-0}"

ACCOUNT="${ACCOUNT:-cis260161p}"
GPU_SPEC=h100-80:1
COMPILE_MODE=inductor
COMPLETEP=1
NO_VE_PROJS=1
NO_WARMDOWN=0                  # COOLDOWN ON -- matches launch_wd_size.sh
TOTAL_BATCH_SIZE=131072
NUM_MODELS=5                   # virtual ensemble size (seed parity); train model 0 only
NUM_EPOCHS=40
DATA_FRACTION=1.0
OPTIMIZER=adamw
ENSEMBLE_MODE=logit
VAL_EVERY_N_STEPS=152
CHECKPOINT_EVERY_N_STEPS=0     # per-epoch ckpts only (resume safety), deletable
MUP_BASE_WIDTH=768
MUP_BASE_DEPTH=12
MUP_BASE_HEAD_DIM=64

# cis260095p is ~full (892 GB used, ~74 GB free); cis260161p has ~2 TB free.
CKPT_BASE="${CKPT_BASE:-/ocean/projects/cis260161p/ymiao6/scaling/slowrun/checkpoints}"
mkdir -p "$CKPT_BASE" experiments/logs

# Per-cell wd grids (unlike launch_wd_size.sh, which used one grid for all cells).
# (n_layer, n_embd, n_head, walltime, wd list)
declare -a LAYERS=(6          18         60)
declare -a EMBDS=( 768        768        768)
declare -a HEADS=( 12         12         12)
declare -a TIMES=("08:00:00" "14:00:00" "30:00:00")
declare -a WDLIST=("0.0 0.05 0.1" "0.1 0.15 0.2" "0.3 0.4 0.5")

echo "============================================================"
echo "P1-FILL  GRID_TAG=$GRID_TAG  account=$ACCOUNT"
echo "  ckpt_base=$CKPT_BASE"
echo "  cooldown ON, df=$DATA_FRACTION, ${NUM_EPOCHS}ep, E=1 (model 0), w768"
echo "============================================================"

for i in "${!LAYERS[@]}"; do
    L=${LAYERS[$i]}; W=${EMBDS[$i]}; H=${HEADS[$i]}; TL=${TIMES[$i]}
    for WD in ${WDLIST[$i]}; do
        CELL_TS="${GRID_TAG}_d${L}_w${W}_wd${WD}"
        GROUP="${GRID_TAG}_d${L}_w${W}_wd${WD}"
        mkdir -p "$CKPT_BASE/parallel_init_ens_${CELL_TS}"

        exports="ALL,N_LAYER=$L,N_HEAD=$H,N_EMBD=$W,SHARED_TIMESTAMP=$CELL_TS,WANDB_GROUP=$GROUP"
        exports+=",NUM_MODELS=$NUM_MODELS,NUM_EPOCHS=$NUM_EPOCHS,TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
        exports+=",DATA_FRACTION=$DATA_FRACTION,OPTIMIZER=$OPTIMIZER,ENSEMBLE_MODE=$ENSEMBLE_MODE"
        exports+=",VAL_EVERY_N_STEPS=$VAL_EVERY_N_STEPS,COMPILE_MODE=$COMPILE_MODE"
        exports+=",COMPLETEP=$COMPLETEP,NO_VE_PROJS=$NO_VE_PROJS,NO_WARMDOWN=$NO_WARMDOWN"
        exports+=",WEIGHT_DECAY=$WD"
        exports+=",MUP_BASE_WIDTH=$MUP_BASE_WIDTH,MUP_BASE_DEPTH=$MUP_BASE_DEPTH,MUP_BASE_HEAD_DIM=$MUP_BASE_HEAD_DIM"
        exports+=",CHECKPOINT_EVERY_N_STEPS=$CHECKPOINT_EVERY_N_STEPS,CHECKPOINT_BASE=$CKPT_BASE"

        sb=(--parsable --account=$ACCOUNT --gpus=$GPU_SPEC --time="$TL"
            --array=0 --job-name="wdsize_d${L}_w${W}_wd${WD}" --export="$exports")
        if [ "$DRY_RUN" = "1" ]; then
            echo "DRY: sbatch ${sb[*]} experiments/parallel/train_array.sh"
        else
            JOB=$(sbatch "${sb[@]}" experiments/parallel/train_array.sh)
            echo "submitted d${L}/w${W} wd=${WD}: job=$JOB  time=$TL  group=$GROUP"
        fi
    done
done

echo "P1-FILL submitted. GRID_TAG=$GRID_TAG  account=$ACCOUNT  ckpt=$CKPT_BASE"
echo "Job names are wdsize_d{L}_w768_wd{WD}_* so the existing analysis picks them up:"
echo "  python experiments/analysis/expt_fig2_capacity_vs_size.py"
