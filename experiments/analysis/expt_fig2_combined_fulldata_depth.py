"""Expt Fig 2 (combined) — DEPTH version, df=1.0 / full 100M-token data, base d12/w768.

Same structure as expt_fig2_combined_fulldata.py (the WIDTH version); Panel B swaps
the width-scaling curve for a DEPTH-scaling curve at fixed width w768:

  (A) ensembling reduces loss across training  (identical to the width figure):
      val loss vs per-model step s, E in {1 (mean indiv), 2, 3, 4, 5}, base d12/w768.
  (B) ensembling beats depth-scaling at matched compute:
      min val loss L* vs compute (rel. d12/w768, E=1), comparing
        - depth scaling : w=768, E=1, depths {6, 12, 18, 24}
                          (compute ratio = L/12, i.e. FLOPs linear in depth)
        - ensembling    : d=12, w=768, E in {1..5}  (compute ratio = E)
      Delta at matched 2x compute: depth d24 vs ensemble E=2.

Depth points: d6/w768, d12/w768 = mean over the 5 init_shuffle individuals; d18/w768,
d24/w768 = single init model from the 2026-07-06 grid-fill. Parsed from .out logs.
"""
import os, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# reuse the width figure's helpers + constants (same parsing conventions)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expt_fig2_combined_fulldata import (
    REPO, LOGS, BATCH, NUM_MODELS, STRAT, STRAT_SIDX, IND_LINE,
    mean_indiv_curve, ensemble_curves,
)

BASE_L = 12
DEPTHS = [6, 12, 18, 24, 48, 60]   # d48=4x, d60=5x complete the E=4/E=5 matched-compute pts


def indiv_curves_for_depth(L):
    """List of (steps, val) per individual for d{L}/w768.
    L in {6,12}: the 5 init_shuffle individuals (fd_train). L in {18,24}: single
    init model from the grid-fill (fd_gridfill, task 0)."""
    curves = []
    if L in (6, 12):
        sidx = STRAT_SIDX[STRAT]
        for m in range(NUM_MODELS):
            task = sidx * NUM_MODELS + m
            d = {}
            for f in sorted(glob.glob(f"{LOGS}/fd_train_d{L}_w768_*_{task}.out")):
                with open(f) as fh:
                    for line in fh:
                        mm = IND_LINE.search(line)
                        if mm:
                            d[int(mm.group(1))] = float(mm.group(2))
            if d:
                s = np.array(sorted(d))
                curves.append((s, np.array([d[k] for k in s])))
    else:  # grid-fill depth cells: single init model (task 0)
        d = {}
        for f in sorted(glob.glob(f"{LOGS}/fd_gridfill_d{L}_w768_*_0.out")):
            with open(f) as fh:
                for line in fh:
                    mm = IND_LINE.search(line)
                    if mm:
                        d[int(mm.group(1))] = float(mm.group(2))
        if d:
            s = np.array(sorted(d))
            curves.append((s, np.array([d[k] for k in s])))
    return curves


def main():
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 24, "axes.linewidth": 4.0, "legend.fontsize": 18,
        "grid.alpha": 0.25, "xtick.labelsize": 20, "ytick.labelsize": 20,
    })

    # E=1 base (mean indiv d12/w768) + ensembles
    base_indiv = indiv_curves_for_depth(BASE_L)
    g1, v1 = mean_indiv_curve(base_indiv)
    ens = ensemble_curves()
    E_present = [1] + sorted(ens)
    palette = sns.color_palette("cool", len(E_present))

    Lstar_ens = {1: float(np.min(v1))}
    for E in sorted(ens):
        Lstar_ens[E] = float(np.min(ens[E][1]))

    # depth sweep: L*(L) at E=1, fixed w768
    Lstar_d, n_ind_d = {}, {}
    for L in DEPTHS:
        gd, vd = mean_indiv_curve(indiv_curves_for_depth(L))
        if vd is not None:
            Lstar_d[L] = float(np.min(vd)); n_ind_d[L] = len(indiv_curves_for_depth(L))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(20, 7.5))

    # ===== Panel A (same as width figure) =====
    axA.plot(g1, v1, lw=3.0, color=palette[0], label="$E = 1$  (mean of individuals)")
    for j, E in enumerate(sorted(ens)):
        s, v = ens[E]
        axA.plot(s, v, lw=3.0, color=palette[j + 1], label=f"$E = {E}$")
    axA.set_xlabel(r"steps $s$  (1 epoch $\approx 763$ steps)")
    axA.set_ylabel(r"validation loss  $\mathcal{L}$")
    axA.set_title("(A)  ensembling reduces loss across training", fontsize=20, loc="left")
    axA.legend(fontsize=15)

    # ===== Panel B: depth-scaling vs ensembling =====
    Ls = sorted(Lstar_d)
    dx = [L / BASE_L for L in Ls]          # compute = L / 12 (FLOPs linear in depth)
    dy = [Lstar_d[L] for L in Ls]
    axB.plot(dx, dy, "o-", color=sns.color_palette("cool", 2)[0], lw=3.0, ms=12,
             markeredgecolor="black", markeredgewidth=0.6,
             label=r"vary depth @ $w=768$, $E=1$")
    for L, x, y in zip(Ls, dx, dy):
        axB.annotate(f"L={L}", (x, y), textcoords="offset points", xytext=(0, -18),
                     fontsize=13, ha="center", color=sns.color_palette("cool", 2)[0])

    Es = sorted(Lstar_ens)
    ex = [float(E) for E in Es]
    ey = [Lstar_ens[E] for E in Es]
    axB.plot(ex, ey, "s-", color=sns.color_palette("magma", 3)[1], lw=3.0, ms=12,
             markeredgecolor="black", markeredgewidth=0.6,
             label=r"ensemble at base ($d=12$, $w=768$)")
    for E, x, y in zip(Es, ex, ey):
        axB.annotate(f"E={E}", (x, y), textcoords="offset points", xytext=(6, 8),
                     fontsize=13, ha="left", color=sns.color_palette("magma", 3)[1])

    # Delta at every matched-compute point where we have BOTH a depth model and an
    # ensemble: 2x (d24 vs E2), 4x (d48 vs E4), 5x (d60 vs E5). Each guarded on presence
    # so the figure renders now (only d24 present) and fills in as d48/d60 land.
    for c, L, E in [(2.0, 24, 2), (4.0, 48, 4), (5.0, 60, 5)]:
        if L in Lstar_d and E in Lstar_ens:
            yd, ye = Lstar_d[L], Lstar_ens[E]
            axB.annotate("", xy=(c, ye), xytext=(c, yd),
                         arrowprops=dict(arrowstyle="<->", lw=2.0, color="black"))
            axB.text(c * 1.04, (yd + ye) / 2, f"$\\Delta = {yd - ye:.2f}$",
                     fontsize=15, va="center",
                     bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="lightgray"))
            axB.axvline(c, color="gray", lw=1.0, ls=":")

    axB.set_xscale("log")
    axB.set_xticks([0.5, 1, 2, 3, 4, 5])
    axB.set_xticklabels([r"0.5$\times$", r"1$\times$", r"2$\times$", r"3$\times$", r"4$\times$", r"5$\times$"])
    axB.set_xlabel(r"compute  (relative to $d=12, w=768, E=1$)")
    axB.set_ylabel(r"min val loss  $\mathcal{L}^*$")
    axB.set_title("(B)  ensembling beats depth-scaling at matched compute", fontsize=20, loc="left")
    axB.legend(fontsize=15, loc="upper right")

    fig.suptitle(f"Full data (df=1.0, 100M tokens), d12/w768 base  —  strategy: {STRAT}  (depth scaling)",
                 fontsize=20, y=1.02)
    fig.tight_layout()

    outdir = os.path.join(REPO, "experiments/figures/02_ensemble_scaling")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "expt_fig2_combined_fulldata_depth.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)

    print(f"strategy = {STRAT}  (depth scaling @ w768)")
    print("Ensemble L* :", {E: round(Lstar_ens[E], 4) for E in Es})
    print("Depth    L* :", {L: (round(Lstar_d[L], 4), f"n={n_ind_d[L]}") for L in Ls})
    if 24 in Lstar_d and 2 in Lstar_ens:
        print(f"Delta @ 2x  : depth(d24)={Lstar_d[24]:.4f}  ens(E=2)={Lstar_ens[2]:.4f}  "
              f"Delta={Lstar_d[24]-Lstar_ens[2]:.4f}")
    print(f"\nSaved {out}\nSaved {out.replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
