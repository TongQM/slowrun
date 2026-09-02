"""Expt Fig 2 (combined) — paper-page-budget version.

Two panels (init strategy only) to fit the 5-page paper limit:
  Left   — ensemble val loss vs steps for E ∈ {1, 2, 5, 10, 15, 20} at the
           base cell (d=12, w=768, df=0.2). Dashed verticals at every epoch.
  Right  — compute-matched comparison at d=12 (vary width vs ensemble at base
           cell). Cyan = single-model scaling line (varying w, E=1); magenta
           = ensemble-at-base line (E ∈ {1..5}). Vertical line at 4× compute.

Reads from data_export/expt2_ensemble/ and data_export/expt3_grid/.

Saves:
  experiments/figures/02_ensemble_scaling/expt_fig2_combined.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from expt_fig3_loader import (
    DEPTHS, WIDTHS, STRATEGIES, E_SIZES, load_grid, REPO,
)

# ---------- constants ----------
EXPT2 = REPO / "data_export" / "expt2_ensemble"
OUT_DIR = REPO / "experiments" / "figures" / "02_ensemble_scaling"

STRAT_LEFT = "init_ens"                    # left panel
STRAT_RIGHT = "init_ens"                   # right panel
SIZES = [2, 5, 10, 15, 20]                  # E values for left panel ensembles
BATCH_SIZE = 131072
TOKENS_PER_EPOCH = int(0.2 * 99_942_400)    # ≈ 19,988,480 at df=0.2
STEPS_PER_EPOCH = TOKENS_PER_EPOCH / BATCH_SIZE     # ≈ 152.5

BASE_D, BASE_W = 12, 768
ALL_E = (1,) + E_SIZES   # 1, 2, 3, 4, 5 (compute-matched series)


# ---------- left-panel data ----------

def load_individuals_mean(strat: str):
    """E=1 = mean across the 20 individuals at every snapshot."""
    paths = sorted((EXPT2 / "individuals").glob(f"{strat}_model*.npz"),
                   key=lambda p: int(p.stem.split("model")[-1]))
    Ls, tok_ref = [], None
    for p in paths:
        d = np.load(p, allow_pickle=True)
        tok = d["tokens"].astype(np.float64)
        if tok_ref is None:
            tok_ref = tok
        else:
            assert np.array_equal(tok, tok_ref), f"individual {p} grid mismatch"
        Ls.append(d["val_loss"].astype(np.float64))
    return tok_ref, np.stack(Ls, axis=0).mean(axis=0)


def load_ensemble_curve(strat: str, E: int):
    d = np.load(EXPT2 / "ensembles" / f"{strat}_E{E}.npz", allow_pickle=True)
    return d["tokens"].astype(np.float64), d["val_loss"].astype(np.float64)


# ---------- right-panel data ----------

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
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 24, "axes.linewidth": 4.0, "legend.fontsize": 16,
        "grid.alpha": 0.25, "xtick.labelsize": 20, "ytick.labelsize": 20,
    })

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(20, 7.5))

    # ---------------------- LEFT PANEL ----------------------
    palette_E = sns.color_palette("cool", 6)

    tok1, mean1 = load_individuals_mean(STRAT_LEFT)
    s1 = tok1 / BATCH_SIZE
    axL.plot(s1, mean1, color=palette_E[0], lw=3.0,
             label=r"$E = 1$  (mean of individuals)")
    max_steps = float(s1.max())
    for j, E in enumerate(SIZES):
        tok, vl = load_ensemble_curve(STRAT_LEFT, E)
        s = tok / BATCH_SIZE
        axL.plot(s, vl, color=palette_E[j + 1], lw=3.0, label=f"$E = {E}$")
        max_steps = max(max_steps, float(s.max()))

    # Epoch-boundary vertical dashed lines
    n_epochs = int(np.ceil(max_steps / STEPS_PER_EPOCH))
    for k in range(1, n_epochs + 1):
        axL.axvline(x=k * STEPS_PER_EPOCH, color="gray",
                    linestyle="--", alpha=0.40, linewidth=1.0, zorder=1)

    axL.set_xlabel(rf"steps  $s$  (1 epoch $= T \approx {STEPS_PER_EPOCH:.0f}$ steps)",
                   fontsize=24)
    axL.set_ylabel(r"validation loss  $\mathcal{L}$", fontsize=28)
    axL.set_title("(A)  ensembling reduces loss across training",
                  fontsize=20, loc="left")
    axL.legend(loc="upper right", framealpha=0.95, fontsize=15)

    # ---------------------- RIGHT PANEL ----------------------
    grid = load_grid()

    # Width-scaling line (vary w at d=BASE_D, E=1)
    w_x, w_y, w_lab = [], [], []
    for w in WIDTHS:
        cell = grid[(BASE_D, w, STRAT_RIGHT)]
        w_x.append(relative_compute(BASE_D, w, 1))
        w_y.append(min_val(cell, 1))
        w_lab.append(f"w={w}")
    w_x = np.array(w_x); w_y = np.array(w_y)

    # Ensemble-at-base line
    e_x, e_y, e_lab = [], [], []
    base_cell = grid[(BASE_D, BASE_W, STRAT_RIGHT)]
    for E in ALL_E:
        e_x.append(relative_compute(BASE_D, BASE_W, E))
        e_y.append(min_val(base_cell, E))
        e_lab.append(f"E={E}")
    e_x = np.array(e_x); e_y = np.array(e_y)

    cool = sns.color_palette("cool", 5)
    color_scale, color_ensemble = cool[0], cool[-1]

    # 4× compute reference line
    axR.axvline(4.0, color="0.35", lw=1.4, ls=":", alpha=0.7, zorder=0)

    axR.plot(w_x, w_y, "-o", color=color_scale, lw=3.0, markersize=11,
             markeredgecolor="black", markeredgewidth=0.6,
             label=r"vary width @ $d{=}12$,  $E{=}1$", zorder=3)
    for x, y, lab in zip(w_x, w_y, w_lab):
        axR.annotate(lab, (x, y), textcoords="offset points",
                     xytext=(0, -16), fontsize=12, color=color_scale,
                     fontweight="bold", ha="center", va="top")

    axR.plot(e_x, e_y, "-s", color=color_ensemble, lw=3.0, markersize=11,
             markeredgecolor="black", markeredgewidth=0.6,
             label=fr"ensemble at base  ($d{{=}}{BASE_D}, w{{=}}{BASE_W}$)",
             zorder=3)
    for x, y, lab in zip(e_x, e_y, e_lab):
        axR.annotate(lab, (x, y), textcoords="offset points",
                     xytext=(0, 12), fontsize=12, color=color_ensemble,
                     fontweight="bold", ha="center", va="bottom")

    # Δ-arrow at 4× compute
    scale_4x = w_y[-1]               # w=1536 → 4× compute
    ens_4x   = e_y[3]                # E=4   → 4× compute
    axR.annotate("", xy=(4.0, ens_4x), xytext=(4.0, scale_4x),
                 arrowprops=dict(arrowstyle="<->", color="0.15", lw=2.0),
                 zorder=4)
    axR.text(4.25, 0.5 * (scale_4x + ens_4x),
             rf"$\Delta = {scale_4x - ens_4x:.2f}$",
             fontsize=18, color="0.1", va="center", ha="left", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                       edgecolor="0.6", alpha=0.92))

    axR.set_xscale("log")
    axR.set_xlim(0.18, 6.5)
    x_ticks = [0.25, 0.5, 1, 2, 4]
    axR.set_xticks(x_ticks)
    axR.set_xticklabels([f"{t}×" for t in x_ticks])
    axR.text(4.05, axR.get_ylim()[1] * 0.998, "4× compute",
             fontsize=13, color="0.35", va="top", ha="left", style="italic")

    axR.set_xlabel(r"compute  (relative to $d{=}12, w{=}768, E{=}1$)",
                   fontsize=24)
    axR.set_ylabel(r"min val loss  $\mathcal{L}^\ast$", fontsize=24)
    axR.set_title("(B)  ensembling beats width-scaling at matched compute",
                  fontsize=20, loc="left")
    axR.legend(loc="lower left", framealpha=0.92, fontsize=14)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "expt_fig2_combined.pdf"
    png = OUT_DIR / "expt_fig2_combined.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=300)
    print(f"saved {pdf}")
    print(f"saved {png}")

    # headline numbers for caption
    print(f"\nright panel headline:  scale w=1536 → {scale_4x:.3f}, "
          f"ensemble E=4 → {ens_4x:.3f},  Δ = {scale_4x - ens_4x:+.3f}")


if __name__ == "__main__":
    main()
