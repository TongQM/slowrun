"""Compute-matched comparison: WIDTH | DEPTH side by side in ONE horizontal figure.

Concatenates the two "Panel B" compute-matched comparisons (from the width and
depth combined figures) into a single 1x2 figure:

  (A) ensembling vs WIDTH-scaling  — single Delta arrow at 4x (w1536 vs E=4)
  (B) ensembling vs DEPTH-scaling  — single Delta arrow at 5x (d60  vs E=5)

Both panels plot the SAME ensemble curve (d12/w768, E=1..5, compute = E) against a
model-size-scaling curve at E=1 (width: compute=(w/768)^2; depth: compute=L/12).
Each panel carries exactly ONE Delta arrow, placed at the largest model on its
size-scaling curve (matching the width figure's single-arrow format). The two
panels share a common y-axis so width- and depth-scaling are directly comparable.

L* values (and all data) come straight from the two source scripts' helpers, so
this figure stays in lock-step with them. Parsed from SLURM .out logs.

  -> experiments/figures/02_ensemble_scaling/expt_fig2_compute_matched_width_depth.{pdf,png}
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expt_fig2_combined_fulldata import (
    REPO, BASE_W, WIDTHS, STRAT,
    mean_indiv_curve, ensemble_curves, indiv_curves_for_width,
)
from expt_fig2_combined_fulldata_depth import (
    BASE_L, DEPTHS, indiv_curves_for_depth,
)


def main():
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 24, "axes.linewidth": 4.0, "legend.fontsize": 18,
        "grid.alpha": 0.25, "xtick.labelsize": 20, "ytick.labelsize": 20,
    })

    # ----- shared ensemble curve (d12/w768, E=1..5) -----
    ens = ensemble_curves()
    _, v1 = mean_indiv_curve(indiv_curves_for_width(BASE_W))
    Lstar_ens = {1: float(np.min(v1))}
    for E in sorted(ens):
        Lstar_ens[E] = float(np.min(ens[E][1]))
    Es = sorted(Lstar_ens)
    ex = [float(E) for E in Es]
    ey = [Lstar_ens[E] for E in Es]

    # ----- width sweep L*(w) at E=1 -----
    Lstar_w = {}
    for w in WIDTHS:
        _, vw = mean_indiv_curve(indiv_curves_for_width(w))
        if vw is not None:
            Lstar_w[w] = float(np.min(vw))

    # ----- depth sweep L*(L) at E=1 -----
    Lstar_d = {}
    for L in DEPTHS:
        _, vd = mean_indiv_curve(indiv_curves_for_depth(L))
        if vd is not None:
            Lstar_d[L] = float(np.min(vd))

    cyan = sns.color_palette("cool", 2)[0]      # size-scaling curve
    mag = sns.color_palette("magma", 3)[1]      # ensemble curve

    # common y-limits so width vs depth are on the same scale
    allv = list(Lstar_ens.values()) + list(Lstar_w.values()) + list(Lstar_d.values())
    pad = 0.05 * (max(allv) - min(allv))
    ylim = (min(allv) - pad, max(allv) + pad)

    fig, (axW, axD) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    def add_ensemble(ax):
        ax.plot(ex, ey, "s-", color=mag, lw=3.0, ms=12, markeredgecolor="black",
                markeredgewidth=0.6, label=r"ensemble ($d=12$, $w=768$)")
        for E, x, y in zip(Es, ex, ey):
            ax.annotate(f"E={E}", (x, y), textcoords="offset points", xytext=(6, 8),
                        fontsize=13, ha="left", color=mag)

    def add_delta(ax, xc, y_size, y_ens):
        ax.annotate("", xy=(xc, y_ens), xytext=(xc, y_size),
                    arrowprops=dict(arrowstyle="<->", lw=2.0, color="black"))
        ax.text(xc * 1.05, (y_size + y_ens) / 2, f"$\\Delta = {y_size - y_ens:.2f}$",
                fontsize=17, va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="lightgray"))
        ax.axvline(xc, color="gray", lw=1.0, ls=":")

    # ===== Panel A: ensembling vs WIDTH-scaling =====
    ws = sorted(Lstar_w)
    wx = [(w / BASE_W) ** 2 for w in ws]
    wy = [Lstar_w[w] for w in ws]
    axW.plot(wx, wy, "o-", color=cyan, lw=3.0, ms=12, markeredgecolor="black",
             markeredgewidth=0.6, label=r"vary width @ $d=12$, $E=1$")
    for w, x, y in zip(ws, wx, wy):
        # shift the arrowed point's label left so it clears the vertical Delta arrow
        off, ha = ((-10, -16), "right") if w == 1536 else ((0, -18), "center")
        axW.annotate(f"w={w}", (x, y), textcoords="offset points", xytext=off,
                     fontsize=13, ha=ha, color=cyan)
    add_ensemble(axW)
    if 1536 in Lstar_w and 4 in Lstar_ens:
        add_delta(axW, 4.0, Lstar_w[1536], Lstar_ens[4])
    axW.set_xscale("log")
    axW.set_xticks([0.25, 0.5, 1, 2, 4])
    axW.set_xticklabels([r"0.25$\times$", r"0.5$\times$", r"1$\times$", r"2$\times$", r"4$\times$"])
    axW.set_xlabel(r"compute  (relative to $d=12, w=768, E=1$)")
    axW.set_ylabel(r"min val loss  $\mathcal{L}^*$")
    axW.set_title("(A)  ensembling vs width-scaling", fontsize=20, loc="left")
    axW.legend(fontsize=15, loc="upper right")
    axW.set_ylim(*ylim)

    # ===== Panel B: ensembling vs DEPTH-scaling =====
    Ls = sorted(Lstar_d)
    dx = [L / BASE_L for L in Ls]
    dy = [Lstar_d[L] for L in Ls]
    axD.plot(dx, dy, "o-", color=cyan, lw=3.0, ms=12, markeredgecolor="black",
             markeredgewidth=0.6, label=r"vary depth @ $w=768$, $E=1$")
    for L, x, y in zip(Ls, dx, dy):
        # shift the arrowed point's label left so it clears the vertical Delta arrow
        off, ha = ((-10, -16), "right") if L == 60 else ((0, -18), "center")
        axD.annotate(f"L={L}", (x, y), textcoords="offset points", xytext=off,
                     fontsize=13, ha=ha, color=cyan)
    add_ensemble(axD)
    if 60 in Lstar_d and 5 in Lstar_ens:
        add_delta(axD, 5.0, Lstar_d[60], Lstar_ens[5])
    axD.set_xscale("log")
    axD.set_xticks([0.5, 1, 2, 3, 4, 5])
    axD.set_xticklabels([r"0.5$\times$", r"1$\times$", r"2$\times$", r"3$\times$", r"4$\times$", r"5$\times$"])
    axD.set_xlabel(r"compute  (relative to $d=12, w=768, E=1$)")
    axD.set_title("(B)  ensembling vs depth-scaling", fontsize=20, loc="left")
    axD.legend(fontsize=15, loc="upper right")

    fig.suptitle(
        f"Ensembling beats model-size scaling at matched compute  "
        f"(df=1.0, 100M tokens, d12/w768 base, {STRAT})",
        fontsize=20, y=1.02)
    fig.tight_layout()

    outdir = os.path.join(REPO, "experiments/figures/02_ensemble_scaling")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "expt_fig2_compute_matched_width_depth.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)

    print(f"strategy = {STRAT}")
    print("Ensemble L* :", {E: round(Lstar_ens[E], 4) for E in Es})
    print("Width    L* :", {w: round(Lstar_w[w], 4) for w in ws})
    print("Depth    L* :", {L: round(Lstar_d[L], 4) for L in Ls})
    if 1536 in Lstar_w and 4 in Lstar_ens:
        print(f"Delta width @ 4x : {Lstar_w[1536]-Lstar_ens[4]:.4f}  "
              f"(w1536={Lstar_w[1536]:.4f}, E4={Lstar_ens[4]:.4f})")
    if 60 in Lstar_d and 5 in Lstar_ens:
        print(f"Delta depth @ 5x : {Lstar_d[60]-Lstar_ens[5]:.4f}  "
              f"(d60={Lstar_d[60]:.4f}, E5={Lstar_ens[5]:.4f})")
    print(f"\nSaved {out}\nSaved {out.replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
