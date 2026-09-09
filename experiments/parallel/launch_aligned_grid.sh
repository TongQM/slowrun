#!/bin/bash
# ALIGNED model-size grid -- one launcher, one recipe per regime, cell lists pinned.
#
# Two regimes, stated once so nothing drifts between launchers again:
#
#   DYNAMICS  lambda=0, constant LR (--no-warmdown), 40 epochs at P=100M.
#             The regime of Figure 1, the stopping-time law, the blow-up rate and
#             every ensemble result. Weight decay >= 0.1 removes the interior
#             minimum within 40 epochs, so this regime cannot use it.
#   CAPACITY  tuned lambda, cooldown ON (last 20% linear to 0), 40 epochs.
#             Best attainable loss only (the L* vs N law); the minimum sits at
#             the final step by construction, so no stopping time is read here.
#
# Common to both (identical to every previous model-size launcher, now pinned):
#   completep ON, no-ve-projs ON, plain_resid OFF (full architecture, residual
#   paths depth-scaled: x0 by 12/L, U-Net skip by (12/L)^0.75 -- the code at the
#   commit this launcher ships in), adamw, logit averaging, batch 131072,
#   lr_multiplier 0.25 (train.py default), warmup 0, muP base w768/d12/h64,
#   n_head = W/64, val every 152 steps (0.2 epoch at P=100M), model 0 = seed 42.
#
# Blocks (BLOCK=<name>; default prints the plan for every block):
#   dyn_rerun   Tier 1  re-run the pre-fix lambda=0 cells off the L=12 row that are
#                       NOT in dyn_ens226: d6/{384,768,1152}, d18/768, d48/768, d60/768
#   dyn_fill    Tier 2  new cells d18/{384,1152,1536}, d24/{384,1152,1536}
#   dyn_ens226  4 individuals (init_shuffle) at the matched-size pair L6/W1536 and
#               L24/W768 (both 226M), per-epoch ckpts, fused replay E in {2,3,4}.
#               Model 0 of each doubles as that cell's E=1 curve.
#   dyn_p20     7-cell lambda=0 constant-LR ladder at P=20M (df=0.2, 50 epochs,
#               val every 38 steps = 0.25 epoch): the second corpus size for the
#               floor-vs-data question.
#   cap_lambda  post-fix lambda-transfer check, cooldown ON, lambda in {0.15,0.3}
#               at d6/768, d48/768, d12/1536
#   cap_grid    tuned-lambda cooldown grid at ONE lambda (WD=..., decide after
#               cap_lambda) over the 12 cells
#
#   DRY_RUN=1 BLOCK=dyn_rerun bash experiments/parallel/launch_aligned_grid.sh
#   BLOCK=dyn_rerun bash experiments/parallel/launch_aligned_grid.sh
#
# Job names carry the block so plotters can glob them:
#   al_dyn_d{L}_w{W}   al_ens_d{L}_w{W}   al_p20_d{L}_w{W}   al_cap_d{L}_w{W}_wd{X}
set -euo pipefail
cd "$(dirname "$0")/../.."

BLOCK="${BLOCK:-plan}"
DRY_RUN="${DRY_RUN:-0}"
GRID_TAG="${GRID_TAG:-aligned_$(date +%Y%m%d)}"
ACCOUNT="${ACCOUNT:-cis260161p}"
GPU_SPEC=h100-80:1
COMPILE_MODE=inductor
COMPLETEP=1
NO_VE_PROJS=1
PLAIN_RESID=0
TOTAL_BATCH_SIZE=131072
NUM_MODELS=5                   # virtual ensemble size (seed parity); most blocks train model 0 only
OPTIMIZER=adamw
ENSEMBLE_MODE=logit
MUP_BASE_WIDTH=768
MUP_BASE_DEPTH=12
MUP_BASE_HEAD_DIM=64
# cis260161p is the only allocation with room (2.8 TB free on 2026-09-09; cis260095p has
# ~70 GB and cis260009p access is revoked). Per-epoch ckpts for the ensemble block are
# ~150 GB per cell; everything else keeps only its resume ckpts.
CKPT_BASE="${CKPT_BASE:-/ocean/projects/cis260161p/ymiao6/scaling/slowrun/checkpoints}"

# Wall-time per cell: measured full-run time on H100 x ~1.3, rounded up.
# Cells not yet run are extrapolated from  t = 0.9h + L * t_W  with
# t_W = {384: 0.20, 768: 0.33, 1152: 0.60, 1536: 0.85} h/layer.
walltime() {  # L W -> HH:MM:SS
    local L=$1 W=$2 h
    case "${L}_${W}" in
        6_384) h=3;;    12_384) h=4;;   18_384) h=6;;   24_384) h=8;;   48_384) h=14;;  60_384) h=17;;
        6_768) h=5;;    12_768) h=7;;   18_768) h=9;;   24_768) h=12;;  48_768) h=22;;  60_768) h=27;;
        6_1152) h=6;;   12_1152) h=11;; 18_1152) h=15;; 24_1152) h=20;;
        6_1536) h=8;;   12_1536) h=15;; 18_1536) h=21;; 24_1536) h=28;;
        *) echo "no walltime for d${L}/w${W}" >&2; exit 1;;
    esac
    printf "%02d:00:00" "$h"
}

# SU estimate at 2 SU/h from the same table, un-padded (measured or extrapolated)
su_est() {
    local L=$1 W=$2
    case "${L}_${W}" in
        6_384) echo 4;;    12_384) echo 5;;   18_384) echo 9;;   24_384) echo 11;;
        6_768) echo 7;;    12_768) echo 10;;  18_768) echo 14;;  24_768) echo 18;;  48_768) echo 33;;  60_768) echo 41;;
        6_1152) echo 9;;   12_1152) echo 16;; 18_1152) echo 23;; 24_1152) echo 31;;
        6_1536) echo 12;;  12_1536) echo 22;; 18_1536) echo 32;; 24_1536) echo 43;;
        *) echo 0;;
    esac
}

common_exports() {  # L W  (n_head = W/64)
    local L=$1 W=$2 H=$(( $2 / 64 ))
    local e="ALL,N_LAYER=$L,N_HEAD=$H,N_EMBD=$W"
    e+=",NUM_MODELS=$NUM_MODELS,TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE,OPTIMIZER=$OPTIMIZER,ENSEMBLE_MODE=$ENSEMBLE_MODE"
    e+=",COMPILE_MODE=$COMPILE_MODE,COMPLETEP=$COMPLETEP,NO_VE_PROJS=$NO_VE_PROJS,PLAIN_RESID=$PLAIN_RESID"
    e+=",MUP_BASE_WIDTH=$MUP_BASE_WIDTH,MUP_BASE_DEPTH=$MUP_BASE_DEPTH,MUP_BASE_HEAD_DIM=$MUP_BASE_HEAD_DIM"
    e+=",CHECKPOINT_BASE=$CKPT_BASE"
    echo "$e"
}

submit() {  # name time array exports [dep] [script]
    local name=$1 tl=$2 arr=$3 exp=$4 dep=${5:-} script=${6:-experiments/parallel/train_array.sh}
    local sb=(--parsable --account=$ACCOUNT --gpus=$GPU_SPEC --time="$tl" --array="$arr" --job-name="$name" --export="$exp")
    [ -n "$dep" ] && sb+=("$dep")
    if [ "$DRY_RUN" = "1" ]; then
        [ "${QUIET:-0}" = "1" ] || echo "    DRY: sbatch ${sb[*]} $script" | sed 's/--export=.* experiments/--export=<...> experiments/' >&2
        echo "DRY$RANDOM"
    else
        sbatch "${sb[@]}" "$script"
    fi
}

TOTAL_SU=0
# ------------------------------------------------------------------ dynamics, E=1
dyn_single() {  # tag  "L:W L:W ..."
    local tag=$1 cells=$2 L W
    for c in $cells; do
        L=${c%%:*}; W=${c##*:}
        local ts="${GRID_TAG}_${tag}_d${L}_w${W}"
        local exp; exp=$(common_exports "$L" "$W")
        exp+=",SHARED_TIMESTAMP=$ts,WANDB_GROUP=$ts,NUM_EPOCHS=40,DATA_FRACTION=1.0"
        exp+=",NO_WARMDOWN=1,WEIGHT_DECAY=0,VAL_EVERY_N_STEPS=152,CHECKPOINT_EVERY_N_STEPS=0"
        [ "$DRY_RUN" = "1" ] || mkdir -p "$CKPT_BASE/parallel_init_ens_${ts}"
        local su; su=$(su_est "$L" "$W"); TOTAL_SU=$((TOTAL_SU + su))
        echo "  d${L}/w${W}  lambda=0 constant-LR 40ep  model 0   ~${su} SU  $(walltime "$L" "$W")"
        JOB=$(submit "al_dyn_d${L}_w${W}" "$(walltime "$L" "$W")" 0 "$exp")
        echo "    job=$JOB"
    done
}

# ------------------------------------------------------------------ dynamics, ensembles
dyn_ensemble() {  # "L:W ..."  -> 4 init_shuffle individuals + fused replay
    local cells=$1 L W
    for c in $cells; do
        L=${c%%:*}; W=${c##*:}
        local ts="${GRID_TAG}_ens_d${L}_w${W}"
        local exp; exp=$(common_exports "$L" "$W")
        exp+=",SHARED_TIMESTAMP=$ts,WANDB_GROUP=$ts,NUM_EPOCHS=40,DATA_FRACTION=1.0"
        exp+=",NO_WARMDOWN=1,WEIGHT_DECAY=0,VAL_EVERY_N_STEPS=152,CHECKPOINT_EVERY_N_STEPS=0"
        exp+=",ENS_SIZES_STR=2 3 4,SKIP_INDIV_VAL=1,END_EPOCH=40,EVAL_MODE=epoch"
        [ "$DRY_RUN" = "1" ] || mkdir -p "$CKPT_BASE/parallel_init_shuffle_ens_${ts}"
        local su; su=$(( $(su_est "$L" "$W") * 4 )); TOTAL_SU=$((TOTAL_SU + su))
        # array tasks NUM_MODELS..NUM_MODELS+3 = init_shuffle models 0..3
        local arr="${NUM_MODELS}-$((NUM_MODELS + 3))"
        echo "  d${L}/w${W}  lambda=0 constant-LR 40ep  init_shuffle models 0-3   ~${su} SU  $(walltime "$L" "$W")"
        TJOB=$(submit "al_ens_d${L}_w${W}" "$(walltime "$L" "$W")" "$arr" "$exp")
        # replay_fused.py drops any epoch missing a ckpt for ANY of num_models, so the
        # replay is told 4 (models 0-3 exist); training keeps NUM_MODELS=5 for seed parity.
        local rexp; rexp=$(echo "$exp" | sed "s/NUM_MODELS=$NUM_MODELS,/NUM_MODELS=4,/")
        RJOB=$(submit "al_ensreplay_d${L}_w${W}" "06:00:00" 1 "$rexp" "--dependency=afterok:$TJOB" experiments/parallel/replay_array_fused.sh)
        echo "    train job=$TJOB   replay job=$RJOB (after train; init_shuffle, E in {2,3,4})"
    done
}

# ------------------------------------------------------------------ dynamics, P=20M ladder
dyn_p20() {
    local cells="6:768 12:768 18:768 24:768 12:384 12:1152 12:1536" L W
    for c in $cells; do
        L=${c%%:*}; W=${c##*:}
        local ts="${GRID_TAG}_p20_d${L}_w${W}"
        local exp; exp=$(common_exports "$L" "$W")
        # df=0.2 -> 19.99M tokens/epoch, 152.5 steps/epoch; 50 epochs = 1B tokens (1/4 of a 100M run)
        exp+=",SHARED_TIMESTAMP=$ts,WANDB_GROUP=$ts,NUM_EPOCHS=50,DATA_FRACTION=0.2"
        exp+=",NO_WARMDOWN=1,WEIGHT_DECAY=0,VAL_EVERY_N_STEPS=38,CHECKPOINT_EVERY_N_STEPS=0"
        [ "$DRY_RUN" = "1" ] || mkdir -p "$CKPT_BASE/parallel_init_ens_${ts}"
        local su; su=$(( ( $(su_est "$L" "$W") + 3 ) / 4 )); TOTAL_SU=$((TOTAL_SU + su))
        local h; h=$(( ( ${walltime_h:-0} ) )); local tl; tl=$(walltime "$L" "$W"); tl=$(printf "%02d:00:00" $(( (10#${tl%%:*} + 3) / 4 + 1 )))
        echo "  d${L}/w${W}  P=20M lambda=0 constant-LR 50ep  model 0   ~${su} SU  $tl"
        JOB=$(submit "al_p20_d${L}_w${W}" "$tl" 0 "$exp")
        echo "    job=$JOB"
    done
}

# ------------------------------------------------------------------ capacity, cooldown ON
cap_cells() {  # tag "L:W ..." "wd wd ..."
    local tag=$1 cells=$2 wds=$3 L W wd
    for c in $cells; do
        L=${c%%:*}; W=${c##*:}
        for wd in $wds; do
            local ts="${GRID_TAG}_${tag}_d${L}_w${W}_wd${wd}"
            local exp; exp=$(common_exports "$L" "$W")
            exp+=",SHARED_TIMESTAMP=$ts,WANDB_GROUP=$ts,NUM_EPOCHS=40,DATA_FRACTION=1.0"
            exp+=",NO_WARMDOWN=0,WEIGHT_DECAY=$wd,VAL_EVERY_N_STEPS=152,CHECKPOINT_EVERY_N_STEPS=0"
            [ "$DRY_RUN" = "1" ] || mkdir -p "$CKPT_BASE/parallel_init_ens_${ts}"
            local su; su=$(su_est "$L" "$W"); TOTAL_SU=$((TOTAL_SU + su))
            echo "  d${L}/w${W}  lambda=${wd} cooldown 40ep  model 0   ~${su} SU  $(walltime "$L" "$W")"
            JOB=$(submit "al_cap_d${L}_w${W}_wd${wd}" "$(walltime "$L" "$W")" 0 "$exp")
            echo "    job=$JOB"
        done
    done
}

ALL12="6:384 12:384 6:768 12:768 6:1152 18:768 6:1536 24:768 12:1152 12:1536 48:768 60:768"

block_su() {  # SU of a block from the tables, no side effects
    local s=0 c L W wd
    case "$1" in
        dyn_rerun)  for c in 6:384 6:768 6:1152 18:768 48:768 60:768; do s=$((s + $(su_est ${c%%:*} ${c##*:}))); done;;
        dyn_fill)   for c in 18:384 18:1152 18:1536 24:384 24:1152 24:1536; do s=$((s + $(su_est ${c%%:*} ${c##*:}))); done;;
        dyn_ens226) for c in 6:1536 24:768; do s=$((s + 4 * $(su_est ${c%%:*} ${c##*:}))); done;;
        dyn_p20)    for c in 6:768 12:768 18:768 24:768 12:384 12:1152 12:1536; do s=$((s + ($(su_est ${c%%:*} ${c##*:}) + 3) / 4)); done;;
        cap_lambda) for c in 6:768 48:768 12:1536; do s=$((s + 2 * $(su_est ${c%%:*} ${c##*:}))); done;;
        cap_grid)   for c in $ALL12; do s=$((s + $(su_est ${c%%:*} ${c##*:}))); done;;
    esac
    echo $s
}

run_block() {
    BLOCK_SU=$(block_su "$1")
    case "$1" in
        dyn_rerun)  echo "== dyn_rerun: pre-fix lambda=0 cells off the L=12 row (ens226 pair excluded) =="
                    dyn_single dyn "6:384 6:768 6:1152 18:768 48:768 60:768";;
        dyn_fill)   echo "== dyn_fill: new lambda=0 cells at L=18, 24 =="
                    dyn_single dyn "18:384 18:1152 18:1536 24:384 24:1152 24:1536";;
        dyn_ens226) echo "== dyn_ens226: 4 init_shuffle individuals at the 226M matched-size pair =="
                    dyn_ensemble "6:1536 24:768";;
        dyn_p20)    echo "== dyn_p20: lambda=0 constant-LR ladder at P=20M =="
                    dyn_p20;;
        cap_lambda) echo "== cap_lambda: post-fix lambda-transfer check, cooldown ON =="
                    cap_cells cap "6:768 48:768 12:1536" "0.15 0.3";;
        cap_grid)   : "${WD:?set WD=<lambda> for cap_grid (decide after cap_lambda)}"
                    echo "== cap_grid: cooldown ON at lambda=$WD over the 12 cells =="
                    cap_cells cap "$ALL12" "$WD";;
        *) echo "unknown BLOCK=$1"; exit 1;;
    esac
}

echo "GRID_TAG=$GRID_TAG  account=$ACCOUNT  ckpt=$CKPT_BASE  DRY_RUN=$DRY_RUN"
if [ "$BLOCK" = "plan" ]; then
    DRY_RUN=1; QUIET=1
    for b in dyn_rerun dyn_fill dyn_ens226 dyn_p20 cap_lambda; do
        run_block "$b" | grep -v "job="
        TOTAL_SU=$((TOTAL_SU + $(block_su "$b")))
        echo "   -> ~$(block_su "$b") SU"
    done
    echo "== cap_grid: 12 cells at one lambda, ~$(block_su cap_grid) SU (not counted below) =="
    echo "TOTAL for the five blocks above: ~${TOTAL_SU} SU   (+ cap_grid ~$(block_su cap_grid) SU)"
else
    run_block "$BLOCK"
    echo "block $BLOCK: ~${BLOCK_SU} SU"
fi
