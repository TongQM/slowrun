"""Expt Fig 3.2 — best-val-loss heatmaps across the 4×4 grid, per E and per strategy.

Layout: 2 rows (init / init+shuffle) × 5 cols (E ∈ {1, 2, 3, 4, 5}).
Each panel is a 4×4 heatmap; rows = depth, cols = width. Cell value is the
minimum val loss reached during the 25-epoch run.

E=1 is the mean across the 5 individuals of each individual's per-run min
(i.e., each individual stops at its own ex-post-best epoch). E≥2 is the
fused-replay ensemble's per-run min at that ensemble size.

Color scale: shared across all 10 panels (`magma`: dark = low loss, bright =
high loss → bright cells flag overfitting / under-trained corners).

Reads from data_export/expt3_grid/. Saves to
experiments/figures/03_compute_matched/expt_fig3_2_min_val_heatmap.{pdf,png}.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns

from expt_fig3_loader import (
    DEPTHS, WIDTHS, STRATEGIES, E_SIZES, load_grid, REPO,
)


OUT_DIR = REPO / "experiments" / "figures" / "03_compute_matched"
OUT_NAME = "expt_fig3_2_min_val_heatmap"

STRAT_PRETTY = {"init_ens": "init", "init_shuffle_ens": "init + shuffle"}
ALL_E = (1,) + E_SIZES  # 1,2,3,4,5


def min_val_for_cell(cell, E: int) -> float:
    """Return per-run minimum val loss for ensemble size E in this cell."""
    if E == 1:
        if not cell.individuals:
            return np.nan
        return float(np.mean([vl.min() for _, vl in cell.individuals]))
    if E in cell.ensembles:
        _, vl = cell.ensembles[E]
        return float(vl.min())
    return np.nan


def build_grid_array(grid, strat: str, E: int) -> np.ndarray:
    """Return (n_depths, n_widths) array of min-val-loss values."""
    arr = np.full((len(DEPTHS), len(WIDTHS)), np.nan)
    for di, d in enumerate(DEPTHS):
        for wi, w in enumerate(WIDTHS):
            arr[di, wi] = min_val_for_cell(grid[(d, w, strat)], E)
    return arr


def main():
    sns.set(font_scale=1.2)
    sns.set_style("white")
    plt.rcParams.update({
        "axes.labelsize": 16,
        "axes.linewidth": 1.5,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    grid = load_grid()

    # Build all 10 grid arrays first to set a shared colormap range.
    arrays = {(s, E): build_grid_array(grid, s, E) for s in STRATEGIES for E in ALL_E}
    all_vals = np.concatenate([a.ravel()[~np.isnan(a.ravel())] for a in arrays.values()])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("magma")

    n_rows = len(STRATEGIES)
    n_cols = len(ALL_E)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.6 * n_cols + 1.2, 2.7 * n_rows + 0.6),
        constrained_layout=False,
    )

    for si, strat in enumerate(STRATEGIES):
        for ei, E in enumerate(ALL_E):
            ax = axes[si, ei]
            arr = arrays[(strat, E)]
            im = ax.imshow(arr, cmap=cmap, norm=norm, aspect="equal")

            # Annotate each cell with its numerical value
            for di in range(len(DEPTHS)):
                for wi in range(len(WIDTHS)):
                    val = arr[di, wi]
                    if np.isnan(val):
                        txt = "—"
                        color = "0.4"
                    else:
                        txt = f"{val:.3f}"
                        # White on dark cells, black on light cells
                        rgba = cmap(norm(val))
                        luma = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                        color = "white" if luma < 0.55 else "black"
                    ax.text(wi, di, txt, ha="center", va="center",
                            fontsize=10, color=color, fontweight="bold")

            ax.set_xticks(range(len(WIDTHS))); ax.set_xticklabels([str(w) for w in WIDTHS])
            ax.set_yticks(range(len(DEPTHS))); ax.set_yticklabels([str(d) for d in DEPTHS])
            ax.tick_params(length=0)

            if si == 0:
                ax.set_title(f"$E={E}$", fontsize=15, pad=6)
            if ei == 0:
                ax.set_ylabel(f"{STRAT_PRETTY[strat]}\n\ndepth $L$", fontsize=14)
            if si == n_rows - 1:
                ax.set_xlabel("width $N$", fontsize=14)

    fig.subplots_adjust(left=0.08, right=0.92, top=0.93, bottom=0.10,
                        hspace=0.18, wspace=0.18)

    # Single shared colorbar on the right
    cbar_ax = fig.add_axes([0.94, 0.12, 0.014, 0.78])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cbar_ax)
    cb.set_label("min val loss (over 25 epochs)", fontsize=14)
    cb.ax.tick_params(labelsize=12)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{OUT_NAME}.pdf"
    png = OUT_DIR / f"{OUT_NAME}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    print(f"Saved {pdf}")
    print(f"Saved {png}")


if __name__ == "__main__":
    main()
