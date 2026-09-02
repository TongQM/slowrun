"""Expt Fig 3.4 — per-cell ensemble exponent b across the 4×4 grid.

For each (d, w, strat) cell, fit jointly across the 25 epoch snapshots:

    L_{T, E} = a_T · E^{-b} + L_inf_T,    E ∈ {1, 2, 3, 4, 5}

with one shared b per cell, per-snapshot a_T, and per-snapshot L_inf_T.
Total free params per cell: 1 + 25 + 25 = 51, with 5 × 25 = 125 observations
(74 dof). E=1 uses the per-snapshot mean across the 5 individuals.

Headline question: is b ≈ 1 (the "denoising at rate 1/E" prior) across the
grid? Does b trend systematically with width or depth?

Two heatmaps (init / init+shuffle), 4×4 each, color-centered at b = 1
(diverging RdBu_r: red = b > 1 = super-1/E, blue = b < 1 = sub-1/E).
Annotated with `b ± SE`.

Reads from data_export/expt3_grid/. Saves to
experiments/figures/03_compute_matched/expt_fig3_4_b_heatmap.{pdf,png}.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from scipy.optimize import least_squares
from scipy.stats import norm as norm_dist

from expt_fig3_loader import (
    DEPTHS, WIDTHS, STRATEGIES, E_SIZES, load_grid, REPO,
)


OUT_DIR = REPO / "experiments" / "figures" / "03_compute_matched"
OUT_NAME = "expt_fig3_4_b_heatmap"

STRAT_PRETTY = {"init_ens": "init", "init_shuffle_ens": "init + shuffle"}
ALL_E = (1,) + E_SIZES   # 1, 2, 3, 4, 5
E_ARR = np.array(ALL_E, dtype=float).reshape(-1, 1)   # column vec for broadcasting


def build_L_grid(cell) -> tuple[np.ndarray | None, np.ndarray]:
    """Return (n_E, n_T) array stacking E=1 (individual mean) + ensemble curves.
    Returns (L, tokens) or (None, None) if data is missing."""
    if not cell.individuals or len(cell.ensembles) < len(E_SIZES):
        return None, None
    toks0, _ = cell.individuals[0]
    n_T = len(toks0)
    # E=1: per-snapshot mean across individuals
    indiv_stack = np.stack([vl for _, vl in cell.individuals], axis=0)  # (5, n_T)
    indiv_mean = indiv_stack.mean(axis=0)
    rows = [indiv_mean]
    for E in E_SIZES:
        _, vl = cell.ensembles[E]
        if len(vl) != n_T:
            return None, None
        rows.append(vl)
    L = np.stack(rows, axis=0)  # (5, n_T)  E ∈ {1,2,3,4,5}
    return L, toks0


def fit_per_cell(L_grid: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Fit L_{T,E} = a_T · E^{-b} + L_inf_T jointly across T for one cell.
    Returns b, b_se, sigma, a_T, L_inf_T."""
    n_E, n_T = L_grid.shape
    cap = L_grid.min(axis=0)   # per-T upper bound for L_inf

    def residuals(p):
        b = p[0]
        a_T = p[1:1 + n_T]
        Linf = p[1 + n_T:1 + 2 * n_T]
        pred = a_T[None, :] * (E_ARR ** (-b)) + Linf[None, :]
        return (pred - L_grid).ravel()

    a0 = np.full(n_T, 0.5)
    Linf0 = cap - 0.05
    p0 = np.concatenate([[1.0], a0, Linf0])
    lo = np.concatenate([[0.05], np.zeros(n_T), -np.inf * np.ones(n_T)])
    hi = np.concatenate([[5.0], np.inf * np.ones(n_T), cap - 1e-4])
    res = least_squares(residuals, p0, bounds=(lo, hi), max_nfev=20000)

    J = res.jac
    dof = max(J.shape[0] - J.shape[1], 1)
    sigma2 = (res.fun ** 2).sum() / dof
    try:
        cov = sigma2 * np.linalg.inv(J.T @ J)
        b_se = float(np.sqrt(cov[0, 0]))
    except np.linalg.LinAlgError:
        b_se = float("nan")

    return float(res.x[0]), b_se, float(np.sqrt(sigma2)), \
        res.x[1:1 + n_T], res.x[1 + n_T:1 + 2 * n_T]


def fit_full_grid(grid):
    """Return {(d, w, strat): (b, b_se, sigma)} for every cell."""
    out = {}
    for d in DEPTHS:
        for w in WIDTHS:
            for strat in STRATEGIES:
                cell = grid[(d, w, strat)]
                L, _ = build_L_grid(cell)
                if L is None:
                    out[(d, w, strat)] = (np.nan, np.nan, np.nan)
                    continue
                b, b_se, sigma, _, _ = fit_per_cell(L)
                out[(d, w, strat)] = (b, b_se, sigma)
    return out


def main():
    sns.set(font_scale=1.3)
    sns.set_style("white")
    plt.rcParams.update({
        "axes.labelsize": 18,
        "axes.linewidth": 1.5,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    print("Fitting per-cell L = a_T E^-b + L_inf,T (this takes a few seconds)...")
    grid = load_grid()
    fits = fit_full_grid(grid)

    # Build heatmap arrays
    arrays_b   = {s: np.full((len(DEPTHS), len(WIDTHS)), np.nan) for s in STRATEGIES}
    arrays_bse = {s: np.full((len(DEPTHS), len(WIDTHS)), np.nan) for s in STRATEGIES}
    for di, d in enumerate(DEPTHS):
        for wi, w in enumerate(WIDTHS):
            for s in STRATEGIES:
                b, b_se, _ = fits[(d, w, s)]
                arrays_b[s][di, wi] = b
                arrays_bse[s][di, wi] = b_se

    # Color scale: diverging, centered at b=1
    all_b = np.concatenate([a.ravel() for a in arrays_b.values()])
    all_b = all_b[~np.isnan(all_b)]
    half = max(abs(all_b.max() - 1), abs(1 - all_b.min()), 0.3)
    vmin, vmax = 1 - half, 1 + half
    norm = TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    fig, axes = plt.subplots(1, len(STRATEGIES),
                             figsize=(5.6 * len(STRATEGIES) + 1.0, 5.4),
                             constrained_layout=False)

    for si, strat in enumerate(STRATEGIES):
        ax = axes[si]
        b = arrays_b[strat]
        bse = arrays_bse[strat]
        im = ax.imshow(b, cmap=cmap, norm=norm, aspect="equal")

        for di in range(len(DEPTHS)):
            for wi in range(len(WIDTHS)):
                bv = b[di, wi]; sev = bse[di, wi]
                if np.isnan(bv):
                    txt = "—"
                    color = "0.4"
                else:
                    txt = f"${bv:.2f}$\n$\\pm{sev:.2f}$"
                    rgba = cmap(norm(bv))
                    luma = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    color = "white" if luma < 0.45 else "black"
                ax.text(wi, di, txt, ha="center", va="center",
                        fontsize=11, color=color, fontweight="bold")

        ax.set_xticks(range(len(WIDTHS))); ax.set_xticklabels([str(w) for w in WIDTHS])
        ax.set_yticks(range(len(DEPTHS))); ax.set_yticklabels([str(d) for d in DEPTHS])
        ax.tick_params(length=0)
        ax.set_xlabel("width $N$", fontsize=15)
        if si == 0:
            ax.set_ylabel("depth $L$", fontsize=15)
        # Per-strategy summary
        b_clean = b[~np.isnan(b)]
        bse_clean = bse[~np.isnan(bse)]
        # Inverse-variance weighted grand mean
        if bse_clean.size > 0 and (bse_clean > 0).all():
            w_inv = 1.0 / bse_clean ** 2
            b_bar = float((b_clean * w_inv).sum() / w_inv.sum())
            b_bar_se = float(np.sqrt(1.0 / w_inv.sum()))
        else:
            b_bar = float(b_clean.mean()); b_bar_se = float(b_clean.std() / np.sqrt(len(b_clean)))
        ax.set_title(f"{STRAT_PRETTY[strat]}\n"
                     f"grid-mean $\\bar{{b}} = {b_bar:.2f} \\pm {b_bar_se:.2f}$",
                     fontsize=15, pad=8)

    fig.subplots_adjust(left=0.07, right=0.88, top=0.88, bottom=0.10, wspace=0.18)

    # Shared colorbar on right
    cbar_ax = fig.add_axes([0.91, 0.16, 0.018, 0.66])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label("ensemble exponent $b$ (centered at $b{=}1$)", fontsize=14)
    cb.ax.tick_params(labelsize=12)
    # Mark b=1 reference on the bar
    cb.ax.axhline(0.5, color="0.1", lw=1.0)   # axhline normalized 0..1; vcenter=1 maps to 0.5

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{OUT_NAME}.pdf"
    png = OUT_DIR / f"{OUT_NAME}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    print(f"Saved {pdf}")
    print(f"Saved {png}")

    # Print a per-cell table for the eventual caption
    print("\n=== per-cell b ± SE ===")
    for s in STRATEGIES:
        print(f"\n[{STRAT_PRETTY[s]}]")
        print("            " + "  ".join(f"w={w:>4}" for w in WIDTHS))
        for di, d in enumerate(DEPTHS):
            row = "  ".join(f"{arrays_b[s][di, wi]:.2f}±{arrays_bse[s][di, wi]:.2f}"
                            for wi in range(len(WIDTHS)))
            print(f"d={d:>2}    {row}")

    # Wald test against b=1 per strategy on the grid-mean
    print("\n=== Wald tests on grid-mean (inv-var weighted) ===")
    for s in STRATEGIES:
        b_clean = arrays_b[s][~np.isnan(arrays_b[s])]
        bse_clean = arrays_bse[s][~np.isnan(arrays_bse[s])]
        w_inv = 1.0 / bse_clean ** 2
        b_bar = float((b_clean * w_inv).sum() / w_inv.sum())
        b_bar_se = float(np.sqrt(1.0 / w_inv.sum()))
        z = (b_bar - 1.0) / b_bar_se
        p = 2 * (1 - norm_dist.cdf(abs(z)))
        print(f"  {STRAT_PRETTY[s]:<14}  b̄ = {b_bar:.3f} ± {b_bar_se:.3f}   "
              f"H0: b̄=1   z={z:+.2f}  p={p:.2e}")


if __name__ == "__main__":
    main()
