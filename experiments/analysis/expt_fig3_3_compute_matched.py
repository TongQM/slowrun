"""Expt Fig 3.3 — compute-matched ensembling vs scaling, both strategies.

Headline question: at fixed total training compute, is it better to (a) train
one larger model, or (b) train E base-size models and ensemble them?

Layout: 2 rows × 2 cols.
  Row 1: vary WIDTH (d=12, w ∈ {384, 768, 1152, 1536}) vs ensemble at base (d=12, w=768, E ∈ {1..5}).
  Row 2: vary DEPTH (w=768, d ∈ {6, 12, 18, 24})       vs ensemble at base (d=12, w=768, E ∈ {1..5}).
  Col 1: init           strategy.
  Col 2: init+shuffle   strategy.

Compute axis: relative non-embed FLOPs vs the base cell (d=12, w=768, E=1),
where non-embed params ≈ 16·L·N² and total compute ≈ E · params(N, L) (one full
training run per ensemble member, same #epochs across cells).

Y-axis: minimum val loss reached over the 25-epoch run (mean of per-individual
min for E=1; per-run min of the fused-replay ensemble for E≥2).

Reads from data_export/expt3_grid/. Saves to
experiments/figures/03_compute_matched/expt_fig3_3_compute_matched.{pdf,png}.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from expt_fig3_loader import (
    DEPTHS, WIDTHS, STRATEGIES, E_SIZES, load_grid, REPO,
)


OUT_DIR = REPO / "experiments" / "figures" / "03_compute_matched"
OUT_NAME = "expt_fig3_3_compute_matched"

STRAT_PRETTY = {"init_ens": "init", "init_shuffle_ens": "init + shuffle"}
BASE_D, BASE_W = 12, 768
ALL_E = (1,) + E_SIZES   # 1, 2, 3, 4, 5


def non_embed_params(d: int, w: int) -> int:
    """Approx body params of the GPT, excluding embeddings: 16·L·N²."""
    return 16 * d * w * w


def relative_compute(d: int, w: int, E: int) -> float:
    """Compute relative to base cell (d=12, w=768, E=1)."""
    base = non_embed_params(BASE_D, BASE_W)
    return E * non_embed_params(d, w) / base


def min_val(cell, E: int) -> float:
    if E == 1:
        if not cell.individuals:
            return np.nan
        return float(np.mean([vl.min() for _, vl in cell.individuals]))
    if E in cell.ensembles:
        _, vl = cell.ensembles[E]
        return float(vl.min())
    return np.nan


def main():
    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 22,
        "axes.linewidth": 3.0,
        "legend.fontsize": 13,
        "grid.alpha": 0.25,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    grid = load_grid()

    # ---------- Build all four series per strategy ----------
    series = {}
    for strat in STRATEGIES:
        # Width scaling at d=BASE_D, varying w, E=1
        w_x, w_y, w_lab = [], [], []
        for w in WIDTHS:
            cell = grid[(BASE_D, w, strat)]
            w_x.append(relative_compute(BASE_D, w, 1))
            w_y.append(min_val(cell, 1))
            w_lab.append(f"w={w}")

        # Depth scaling at w=BASE_W, varying d, E=1
        d_x, d_y, d_lab = [], [], []
        for d in DEPTHS:
            cell = grid[(d, BASE_W, strat)]
            d_x.append(relative_compute(d, BASE_W, 1))
            d_y.append(min_val(cell, 1))
            d_lab.append(f"d={d}")

        # Ensemble at base, varying E
        e_x, e_y, e_lab = [], [], []
        base_cell = grid[(BASE_D, BASE_W, strat)]
        for E in ALL_E:
            e_x.append(relative_compute(BASE_D, BASE_W, E))
            e_y.append(min_val(base_cell, E))
            e_lab.append(f"E={E}")

        series[strat] = {
            "width":   {"x": np.array(w_x), "y": np.array(w_y), "lab": w_lab},
            "depth":   {"x": np.array(d_x), "y": np.array(d_y), "lab": d_lab},
            "ensemble":{"x": np.array(e_x), "y": np.array(e_y), "lab": e_lab},
        }

    # ---------- Figure ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=True)

    # Distinct colors: scale=blue, ensemble=magenta (extremes of `cool`).
    cool = sns.color_palette("cool", 5)
    color_scale    = cool[0]   # cyan
    color_ensemble = cool[-1]  # magenta

    axis_kinds = ["width", "depth"]
    axis_pretty = {"width": "vary width @ $d{=}12$",
                   "depth": "vary depth @ $w{=}768$"}

    # Pre-compute global y-range so the 4× reference label sits inside every panel
    all_y = np.concatenate([
        np.concatenate([series[s][k]["y"] for k in ("width", "depth", "ensemble")])
        for s in STRATEGIES
    ])
    y_lo, y_hi = float(np.nanmin(all_y)) - 0.03, float(np.nanmax(all_y)) + 0.04

    for ri, axis in enumerate(axis_kinds):
        for ci, strat in enumerate(STRATEGIES):
            ax = axes[ri, ci]
            s = series[strat]
            scale = s[axis]
            ens   = s["ensemble"]

            # Reference vertical at 4× compute (the headline matched-compute point).
            ax.axvline(4.0, color="0.35", lw=1.4, ls=":", alpha=0.7, zorder=0)

            # Scale-model line (cyan, circles, labels BELOW the markers)
            ax.plot(scale["x"], scale["y"], "-o",
                    color=color_scale, lw=3.0, markersize=10,
                    markeredgecolor="black", markeredgewidth=0.6,
                    label=axis_pretty[axis] + ", $E{=}1$",
                    zorder=3)
            for x, y, lab in zip(scale["x"], scale["y"], scale["lab"]):
                ax.annotate(lab, (x, y), textcoords="offset points",
                            xytext=(0, -16), fontsize=10, color=color_scale,
                            fontweight="bold", ha="center", va="top")

            # Ensemble-at-base line (magenta, squares, labels ABOVE the markers)
            ax.plot(ens["x"], ens["y"], "-s",
                    color=color_ensemble, lw=3.0, markersize=10,
                    markeredgecolor="black", markeredgewidth=0.6,
                    label=f"ensemble at base ($d{{=}}{BASE_D}, w{{=}}{BASE_W}$)",
                    zorder=3)
            for x, y, lab in zip(ens["x"], ens["y"], ens["lab"]):
                ax.annotate(lab, (x, y), textcoords="offset points",
                            xytext=(0, 12), fontsize=10, color=color_ensemble,
                            fontweight="bold", ha="center", va="bottom")

            ax.set_xscale("log")
            ax.set_xlim(0.18, 6.5)
            ax.set_ylim(y_lo, y_hi)
            x_ticks = [0.25, 0.5, 1, 2, 4]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f"{t}×" for t in x_ticks])

            # Place "4× compute" reference label just under the top of the axis.
            ax.text(4.05, y_hi - 0.015, "4× compute",
                    fontsize=10, color="0.35", va="top", ha="left",
                    style="italic")

            # Annotate the headline gain at 4× when both lines straddle it
            scale_at_4x = scale["y"][-1] if axis == "width" else None
            ens_at_4x = ens["y"][3]  # E=4
            if scale_at_4x is not None and not np.isnan(scale_at_4x):
                gap = scale_at_4x - ens_at_4x
                ax.annotate("", xy=(4.0, ens_at_4x), xytext=(4.0, scale_at_4x),
                            arrowprops=dict(arrowstyle="<->", color="0.2", lw=1.5),
                            zorder=4)
                ax.text(4.18, 0.5 * (scale_at_4x + ens_at_4x),
                        f"$\\Delta\\!=\\!{gap:.2f}$", fontsize=12,
                        color="0.15", va="center", ha="left", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="0.7", alpha=0.9))

            if ri == 1:
                ax.set_xlabel(r"compute (relative to $d{=}12, w{=}768, E{=}1$)")
            if ci == 0:
                ax.set_ylabel("min val loss")

            ax.set_title(f"{axis_pretty[axis]}  •  {STRAT_PRETTY[strat]}",
                         fontsize=15)

            ax.legend(loc="lower left", fontsize=11, frameon=True, framealpha=0.9)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{OUT_NAME}.pdf"
    png = OUT_DIR / f"{OUT_NAME}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    print(f"Saved {pdf}")
    print(f"Saved {png}")

    # Print headline numbers for the eventual caption
    print("\n=== Headline numbers (at 4× matched compute) ===")
    for strat in STRATEGIES:
        s = series[strat]
        # Width @ 4× = w=1536 (since 16*12*1536^2 = 4 * 16*12*768^2)
        scale_w_4x = s["width"]["y"][-1]
        # Ensemble @ 4× = E=4
        ens_4x = s["ensemble"]["y"][3]   # ALL_E[3] = 4
        print(f"  {STRAT_PRETTY[strat]:<14}  scale w=1536: {scale_w_4x:.3f}   "
              f"ensemble E=4: {ens_4x:.3f}   gain: {scale_w_4x - ens_4x:+.3f}")


if __name__ == "__main__":
    main()
