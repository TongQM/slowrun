"""Expt Fig 4.1 — validation-loss curves across data fraction (P), at the
base cell d=12, w=768, both ensembling strategies.

Layout: 1 row × 2 cols. Cols: init / init+shuffle.
Per panel: 4 curves, one per df ∈ {0.2, 0.3, 0.4, 0.5} in the `cool` palette.
Triangle markers indicate the per-run U-curve nadir.

Reads from data_export/expt4_datasize/wd0_fixed_tokens/. Saves to
experiments/figures/04_TBD/expt_fig4_1_val_curves.{pdf,png}.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from expt_fig4_loader import (
    DFS, STRATEGIES, STRAT_PRETTY, load_all, df_to_P, REPO,
)


OUT_DIR = REPO / "experiments" / "figures" / "04_TBD"
OUT_NAME = "expt_fig4_1_val_curves"


def main():
    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 22, "axes.linewidth": 3.0, "legend.fontsize": 14,
        "grid.alpha": 0.25, "xtick.labelsize": 16, "ytick.labelsize": 16,
    })

    runs = load_all()

    fig, axes = plt.subplots(
        1, len(STRATEGIES),
        figsize=(8.5 * len(STRATEGIES), 5.5),
        sharex=True, sharey=True,
        constrained_layout=False,
    )

    palette = sns.color_palette("cool", len(DFS))

    all_vals = np.concatenate([r.val_loss for r in runs.values()])
    y_lo = max(float(np.percentile(all_vals, 0.5)), 3.8)
    y_hi = min(float(np.percentile(all_vals, 99.5)), 7.8)

    x_max = max(r.tokens.max() for r in runs.values()) / 1e9 * 1.02

    for si, strat in enumerate(STRATEGIES):
        ax = axes[si]
        for di, df in enumerate(DFS):
            run = runs.get((df, strat))
            if run is None:
                continue
            ax.plot(run.tokens / 1e9, run.val_loss,
                    color=palette[di], lw=3.0, alpha=0.95,
                    label=f"df = {df}  (P = {df_to_P(df)/1e6:.0f}M)")
            idx = run.min_val_idx()
            ax.plot(run.tokens[idx] / 1e9, run.val_loss[idx],
                    marker="v", markersize=14, color=palette[di],
                    markeredgecolor="black", markeredgewidth=0.7, zorder=5)

        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(0, x_max)
        ax.set_xlabel("cumulative tokens (B)")
        if si == 0:
            ax.set_ylabel("val loss")
        ax.set_title(STRAT_PRETTY[strat], fontsize=18, pad=8)
        ax.legend(loc="upper right", fontsize=13, frameon=True, framealpha=0.9)

    fig.text(0.5, -0.02,
             r"$\blacktriangledown$ marks the per-run U-curve nadir (min val loss)",
             ha="center", va="top", fontsize=13, style="italic", color="0.25")

    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.13, wspace=0.10)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{OUT_NAME}.pdf"
    png = OUT_DIR / f"{OUT_NAME}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    print(f"Saved {pdf}")
    print(f"Saved {png}")


if __name__ == "__main__":
    main()
