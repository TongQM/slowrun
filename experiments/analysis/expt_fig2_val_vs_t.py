"""Ensemble val loss vs steps for different E (precursor to the scaling-law fit).

For each ensemble size E ∈ {2, 5, 10, 15, 20} (post-hoc fused replay) plus
E = 1 (= per-snapshot mean across the 20 individuals), plot validation loss
as a function of optimizer steps s, with dashed vertical lines at every
epoch boundary. At df = 0.2, batch = 131072: T = 20M / 131072 ≈ 152 steps
per epoch, so 40 epochs ≈ 6100 steps.

Reads from data_export/expt2_ensemble/ — no wandb dependency.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data_export" / "expt2_ensemble"
OUT_DIR = REPO / "experiments" / "figures" / "02_ensemble_scaling"

SIZES = [2, 5, 10, 15, 20]
STRATS = [("init_ens", "init"), ("init_shuffle_ens", "init + shuffle")]
BATCH_SIZE = 131072
TOKENS_PER_EPOCH = int(0.2 * 99_942_400)             # ≈ 19,988,480 at df=0.2
STEPS_PER_EPOCH = TOKENS_PER_EPOCH / BATCH_SIZE      # ≈ 152.5


def load_individuals_mean(strat: str):
    """Return (tokens, mean_val_loss) for E=1 across the 20 individuals."""
    paths = sorted((DATA / "individuals").glob(f"{strat}_model*.npz"),
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
    d = np.load(DATA / "ensembles" / f"{strat}_E{E}.npz", allow_pickle=True)
    return d["tokens"].astype(np.float64), d["val_loss"].astype(np.float64)


def main():
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["axes.linewidth"] = 4.0
    plt.rcParams["legend.fontsize"] = 18
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20

    palette = sns.color_palette("cool", 6)

    fig, axes = plt.subplots(1, 2, figsize=(22, 7.5), sharey=True)

    for ax, (strat, label_name) in zip(axes, STRATS):
        tok1, mean1 = load_individuals_mean(strat)
        s1 = tok1 / BATCH_SIZE
        ax.plot(s1, mean1, color=palette[0], lw=3.0,
                label=r"$E = 1$  (mean of individuals)")
        max_steps = float(s1.max())
        for j, E in enumerate(SIZES):
            tok, vl = load_ensemble_curve(strat, E)
            s = tok / BATCH_SIZE
            ax.plot(s, vl, color=palette[j + 1], lw=3.0, label=f"$E = {E}$")
            max_steps = max(max_steps, float(s.max()))

        # Epoch boundaries (dashed vertical lines)
        n_epochs = int(np.ceil(max_steps / STEPS_PER_EPOCH))
        for k in range(1, n_epochs + 1):
            ax.axvline(x=k * STEPS_PER_EPOCH, color="gray",
                       linestyle="--", alpha=0.40, linewidth=1.0, zorder=1)

        ax.set_xlabel(rf"steps  $s$  (1 epoch $= T \approx {STEPS_PER_EPOCH:.0f}$ steps)",
                      fontsize=24)
        # Strategy annotation in the corner
        ax.text(0.02, 0.97, label_name, transform=ax.transAxes,
                fontsize=22, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.9, edgecolor="lightgray"))
        ax.legend(loc="upper right", framealpha=0.95)

    axes[0].set_ylabel(r"validation loss  $\mathcal{L}$", fontsize=28)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "val_vs_t_per_E.pdf"
    out_png = OUT_DIR / "val_vs_t_per_E.png"
    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    print(f"saved {out_pdf}")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
