#!/bin/bash
# 9-cell width × depth ensembling sweep.
#
# For each (n_layer ∈ {6, 12, 24}) × (n_embd ∈ {384, 768, 1536}) cell,
# submits one ensembling experiment via experiments/parallel/launch.sh:
#   - 5 individuals × 2 strategies (init / init_shuffle) = 10 training jobs per cell
#   - 3 ensemble sizes {2, 4, 5} × 2 strategies = 6 replay jobs per cell
# Total: 9 × (10 + 6) = 144 SLURM jobs.
#
# Defaults (locked in for this project):
#   --completep ON, --no-ve-projs ON
#   total_batch_size=131072 (128K), data_fraction=0.2, num_epochs=25
#   GPU=h100-80:1, compile=inductor, ensemble-mode=logit
#   mup_base = (width=768, depth=12, head_dim=64)
#
# Each cell gets its own SHARED_TIMESTAMP and wandb group, so curves stay
# scoped to their own (depth, width) pair while still being comparable
# across the grid via the shared GRID_TAG suffix.
#
# Usage:
#   bash experiments/parallel/launch_grid.sh
#
# Env overrides (rare):
#   GRID_TAG=<tag>   override the shared tag suffix (default: timestamp)
#   CELLS=<csv>      space-separated list of L:H:W triples; default = full 3x3 grid
#   ACCOUNT=<acct>   SLURM allocation
#
# After submission, all wandb groups share the prefix
# `grid_<GRID_TAG>_d<L>_w<W>_df0.2`, and the train/replay arrays are
# tagged with that group so plot.py sweep mode can find them.

set -euo pipefail
cd "$(dirname "$0")/../.."

GRID_TAG="${GRID_TAG:-$(date +%Y%m%d_%H%M%S)}"

# 9 cells: (n_layer, n_head, n_embd). head_dim = 64 throughout.
CELLS_DEFAULT=(
    "6:6:384"   "6:12:768"   "6:24:1536"
    "12:6:384"  "12:12:768"  "12:24:1536"
    "24:6:384"  "24:12:768"  "24:24:1536"
)

if [ -n "${CELLS:-}" ]; then
    read -ra CELLS_ARR <<< "$CELLS"
else
    CELLS_ARR=("${CELLS_DEFAULT[@]}")
fi

echo "============================================================"
echo "9-cell width × depth ensembling sweep"
echo "  Grid tag:       $GRID_TAG"
echo "  Cells (L:H:W):  ${CELLS_ARR[*]}"
echo "  Per-cell:       5 individuals × 2 strategies × 25 epochs"
echo "                  ensemble sizes {2, 4, 5}"
echo "  Defaults:       batch=131072  df=0.2  completep=on  no-ve-projs=on"
echo "                  GPU=h100-80:1  compile=inductor"
echo "============================================================"

for CELL in "${CELLS_ARR[@]}"; do
    IFS=":" read -r L H W <<< "$CELL"

    # Per-cell timestamp keeps checkpoint dirs isolated; GRID_TAG threads
    # them together for cross-cell wandb plotting.
    CELL_TS="${GRID_TAG}_d${L}_w${W}"

    # Wandb group identifies (grid, cell). Same across all 16 jobs of one cell.
    GROUP="grid_${GRID_TAG}_d${L}_w${W}_df0.2"

    echo
    echo "=== Cell d${L}_h${H}_w${W} -> group ${GROUP} ==="
    SHARED_TIMESTAMP="$CELL_TS" \
        N_LAYER=$L N_HEAD=$H N_EMBD=$W \
        NUM_MODELS=5 NUM_EPOCHS=25 \
        ENS_SIZES_STR="2 4 5" \
        TOTAL_BATCH_SIZE=131072 \
        DATA_FRACTION=0.2 \
        OPTIMIZER=adamw \
        ENSEMBLE_MODE=logit \
        COMPLETEP=1 NO_VE_PROJS=1 \
        MUP_BASE_WIDTH=768 MUP_BASE_DEPTH=12 MUP_BASE_HEAD_DIM=64 \
        GPU_SPEC=h100-80:1 COMPILE_MODE=inductor \
        TIME_LIMIT=12:00:00 \
        WANDB_GROUP="$GROUP" \
        bash experiments/parallel/launch.sh
done

echo
echo "All 9 cells submitted under grid tag: $GRID_TAG"
echo "Wandb groups: grid_${GRID_TAG}_d{6,12,24}_w{384,768,1536}_df0.2_*"
