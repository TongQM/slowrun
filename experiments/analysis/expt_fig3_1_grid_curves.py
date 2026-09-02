"""Expt Fig 3.1 — 4×4 grid of validation-loss curves at df=0.2.

Stacked layout: top half = init ensembles (4 rows = depth, 4 cols = width),
bottom half = init+shuffle ensembles (same 4×4). All 32 panels share x and y
axes so visual comparison is direct across width, depth, and strategy.

Per panel:
  - 5 individual curves (thin gray, alpha=0.5)
  - 4 ensemble curves: E ∈ {2, 3, 4, 5}, `cool` palette, lw=4
  - faint vertical dashed lines at every epoch boundary (20M tokens each)

Reads from data_export/expt3_grid/. Saves to
experiments/figures/03_compute_matched/expt_fig3_1_grid_both_strats.{pdf,png}.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from expt_fig3_loader import (
    DEPTHS, WIDTHS, STRATEGIES, E_SIZES, TOKENS_PER_EPOCH_DF02,
    load_grid, REPO,
)


OUT_DIR = REPO / "experiments" / "figures" / "03_compute_matched"
OUT_NAME = "expt_fig3_1_grid_both_strats"

STRAT_PRETTY = {"init_ens": "init", "init_shuffle_ens": "init + shuffle"}


def main():
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 18,
        "axes.linewidth": 2.5,
        "legend.fontsize": 16,
        "grid.alpha": 0.18,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    grid = load_grid()

    # Precompute global y-range (use both strategies, ensembles only — robust to
    # individuals' early high values). Then pad slightly.
    all_vals = []
    for d in DEPTHS:
        for w in WIDTHS:
            for s in STRATEGIES:
                cell = grid[(d, w, s)]
                for _, vl in cell.individuals:
                    all_vals.append(vl)
                for _, vl in cell.ensembles.values():
                    all_vals.append(vl)
    all_vals = np.concatenate(all_vals)
    y_lo = float(np.percentile(all_vals, 0.1))
    y_hi = float(np.percentile(all_vals, 99.9))
    # Tighten high end to keep U-curves visible (drop the very-early high losses).
    y_hi = min(y_hi, 6.6)
    y_lo = max(y_lo, 4.0)

    n_rows = 2 * len(DEPTHS)   # 8 rows: top 4 = init, bottom 4 = init+shuffle
    n_cols = len(WIDTHS)        # 4
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 2.4 * n_rows),
        sharex=True, sharey=True,
        constrained_layout=False,
    )

    palette_E = sns.color_palette("cool", len(E_SIZES))
    color_E = {E: palette_E[i] for i, E in enumerate(E_SIZES)}

    # x-axis range: 0 .. 25 epochs * 20M = 500M tokens
    x_max = 25 * TOKENS_PER_EPOCH_DF02

    for strat_idx, strat in enumerate(STRATEGIES):
        for di, d in enumerate(DEPTHS):
            row = strat_idx * len(DEPTHS) + di
            for wi, w in enumerate(WIDTHS):
                ax = axes[row, wi]
                cell = grid[(d, w, strat)]

                # Vertical dashed lines at each epoch
                for k in range(1, 26):
                    ax.axvline(k * TOKENS_PER_EPOCH_DF02, color="gray",
                               lw=0.4, ls="--", alpha=0.25, zorder=0)

                # Individuals (thin gray)
                for tok, vl in cell.individuals:
                    ax.plot(tok, vl, color="0.55", lw=1.0, alpha=0.55, zorder=2)

                # Ensembles (cool palette, thick)
                for E in E_SIZES:
                    if E not in cell.ensembles: continue
                    tok, vl = cell.ensembles[E]
                    ax.plot(tok, vl, color=color_E[E], lw=2.6, zorder=3,
                            label=f"E={E}" if (row == 0 and wi == 0) else None)

                ax.set_xlim(0, x_max)
                ax.set_ylim(y_lo, y_hi)
                # x-tick formatter (millions)
                ax.xaxis.set_major_formatter(
                    plt.FuncFormatter(lambda v, _: f"{int(v/1e6)}M" if v > 0 else "0")
                )
                ax.tick_params(axis="both", which="major", labelsize=11)

                # Title only on the very top row of each strategy block
                if di == 0:
                    ax.set_title(f"$w={w}$", fontsize=14, pad=4)

                # y-axis label only on first column
                if wi == 0:
                    ax.set_ylabel(f"$d={d}$", fontsize=14, rotation=90, labelpad=6)

                # x-axis label only on bottom row
                if row == n_rows - 1:
                    ax.set_xlabel("tokens seen", fontsize=14)

    # Strategy banners on the far left (text outside axes)
    fig.text(0.005, 0.78, STRAT_PRETTY["init_ens"], rotation=90,
             fontsize=22, fontweight="bold", va="center", ha="left")
    fig.text(0.005, 0.34, STRAT_PRETTY["init_shuffle_ens"], rotation=90,
             fontsize=22, fontweight="bold", va="center", ha="left")

    # Horizontal divider between the two strategy blocks
    # (drawn as a figure-level line right between rows 3 and 4)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.95, bottom=0.07,
                        hspace=0.32, wspace=0.10)
    # After subplots_adjust, compute divider y-coordinate
    bbox_top_block = axes[len(DEPTHS) - 1, 0].get_position()
    bbox_bot_block = axes[len(DEPTHS), 0].get_position()
    divider_y = 0.5 * (bbox_top_block.y0 + bbox_bot_block.y1)
    fig.add_artist(plt.Line2D([0.05, 0.99], [divider_y, divider_y],
                              transform=fig.transFigure, color="0.25",
                              lw=1.0, ls="-", alpha=0.5))

    # Bottom-centered legend
    legend_handles = (
        [Line2D([0], [0], color="0.55", lw=1.5, label="individual (5 per cell)")]
        + [Line2D([0], [0], color=color_E[E], lw=2.6, label=f"$E={E}$") for E in E_SIZES]
        + [Line2D([0], [0], color="gray", lw=0.6, ls="--",
                  label="epoch boundary (20M tokens)")]
    )
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.005),
               ncol=len(legend_handles), fontsize=14, frameon=True,
               framealpha=0.92)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{OUT_NAME}.pdf"
    png = OUT_DIR / f"{OUT_NAME}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    print(f"Saved {pdf}")
    print(f"Saved {png}")


if __name__ == "__main__":
    main()
