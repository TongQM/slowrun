"""Expt Fig 2 (combined), df=1.0 full data — EPOCH-RESOLUTION variant of Panel A.

Identical to expt_fig2_combined_fulldata.py except Panel A samples val loss at
**every epoch boundary** (one point per epoch) instead of every 152 steps. With
constant LR (no-warmdown) + small batch the per-step curve carries a lot of
optimization jitter; reducing to epoch resolution evaluates each curve at integer
epochs (tokens = k * tokens_per_epoch) and gives a far cleaner trend line.

Panel B is unchanged from the per-step version: L* is a minimum over training, so we
read it off the finest-resolution curve, not the epoch-subsampled one.

Parsing is reused verbatim from expt_fig2_combined_fulldata (same .out log sources).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expt_fig2_combined_fulldata import (
    indiv_curves_for_width, ensemble_curves,
    REPO, BATCH, NUM_MODELS, BASE_W, STRAT, WIDTHS,
)

TOK_PER_EPOCH = 99_614_720
STEPS_PER_EPOCH = TOK_PER_EPOCH // BATCH          # = 760
N_EPOCHS = 40                                     # run length; drop any partial tail bin past this


def to_epoch(s, v):
    """Reduce a per-step (steps, vals) curve to per-epoch MEANS.
    Each per-step val measurement is binned into epoch k = ceil(step / STEPS_PER_EPOCH)
    (so steps (k-1)*760+1 .. k*760 -> epoch k) and averaged within the bin. Returns
    (epochs, mean_vals) for every epoch that has at least one measurement."""
    s = np.asarray(s, float); v = np.asarray(v, float)
    k = np.ceil(s / STEPS_PER_EPOCH).astype(int)
    k = np.maximum(k, 1)
    ks = np.array([kk for kk in np.unique(k) if kk <= N_EPOCHS])
    means = np.array([v[k == kk].mean() for kk in ks])
    return ks, means


def mean_indiv_epoch(curves):
    """Per-model epoch-mean, then mean over individuals at each epoch they all reach."""
    permodel = [to_epoch(s, v) for s, v in curves]
    permodel = [(k, val) for k, val in permodel if len(k)]
    if not permodel:
        return np.array([]), np.array([])
    common = sorted(set.intersection(*[set(k.tolist()) for k, _ in permodel]))
    common = np.array(common)
    stacked = np.vstack([
        val[np.isin(k, common)] for k, val in permodel
    ])
    return common, stacked.mean(0)


def main():
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 24, "axes.linewidth": 4.0, "legend.fontsize": 18,
        "grid.alpha": 0.25, "xtick.labelsize": 20, "ytick.labelsize": 20,
    })

    # ---- Panel A data (epoch resolution) ----
    base_indiv = indiv_curves_for_width(BASE_W)
    e1, v1 = mean_indiv_epoch(base_indiv)
    ens = ensemble_curves()                       # E -> (steps, vals) per-step
    ens_epoch = {E: to_epoch(*ens[E]) for E in sorted(ens)}
    E_present = [1] + sorted(ens)
    palette = sns.color_palette("cool", len(E_present))

    # ---- Panel B data (FULL resolution, same as per-step figure) ----
    # L* is a minimum over training, so always read it off the finest per-step curve,
    # never the epoch-binned one. This also guarantees the ensemble E=1 point and the
    # width w768 point are the SAME number at 1x compute (both = the base individual).
    from expt_fig2_combined_fulldata import mean_indiv_curve
    Lstar_w, n_ind_w = {}, {}
    for w in WIDTHS:
        cs = indiv_curves_for_width(w)
        gw, vw = mean_indiv_curve(cs)
        if vw is not None:
            Lstar_w[w] = float(np.min(vw)); n_ind_w[w] = len(cs)
    Lstar_ens = {1: Lstar_w.get(BASE_W, np.nan)}   # E=1 == base individual, by definition
    for E in sorted(ens):
        Lstar_ens[E] = float(np.min(ens[E][1]))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(20, 7.5))

    # ===== Panel A (epoch-wise) =====
    if len(e1):
        axA.plot(e1, v1, "o-", lw=3.0, ms=7, color=palette[0],
                 label="$E = 1$  (mean of individuals)")
    for j, E in enumerate(sorted(ens)):
        ek, ev = ens_epoch[E]
        if len(ek):
            axA.plot(ek, ev, "o-", lw=3.0, ms=7, color=palette[j + 1], label=f"$E = {E}$")
    axA.set_xlabel("epoch")
    axA.set_ylabel(r"validation loss  $\mathcal{L}$")
    axA.set_title("(A)  ensembling reduces loss across training  (per-epoch mean)",
                  fontsize=20, loc="left")
    axA.legend(fontsize=15)

    # ===== Panel B (identical to per-step figure) =====
    ws = sorted(Lstar_w)
    wx = [(w / BASE_W) ** 2 for w in ws]
    wy = [Lstar_w[w] for w in ws]
    axB.plot(wx, wy, "o-", color=sns.color_palette("cool", 2)[0], lw=3.0, ms=12,
             markeredgecolor="black", markeredgewidth=0.6,
             label=r"vary width @ $d=12$, $E=1$")
    for w, x, y in zip(ws, wx, wy):
        axB.annotate(f"w={w}", (x, y), textcoords="offset points", xytext=(0, -18),
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

    if 1536 in Lstar_w and 4 in Lstar_ens:
        x4 = 4.0
        yw, ye = Lstar_w[1536], Lstar_ens[4]
        axB.annotate("", xy=(x4, ye), xytext=(x4, yw),
                     arrowprops=dict(arrowstyle="<->", lw=2.0, color="black"))
        axB.text(x4 * 1.05, (yw + ye) / 2, f"$\\Delta = {yw - ye:.2f}$",
                 fontsize=17, va="center",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="lightgray"))
        axB.axvline(x4, color="gray", lw=1.0, ls=":")

    axB.set_xscale("log")
    axB.set_xticks([0.25, 0.5, 1, 2, 4])
    axB.set_xticklabels([r"0.25$\times$", r"0.5$\times$", r"1$\times$", r"2$\times$", r"4$\times$"])
    axB.set_xlabel(r"compute  (relative to $d=12, w=768, E=1$)")
    axB.set_ylabel(r"min val loss  $\mathcal{L}^*$")
    axB.set_title("(B)  ensembling beats width-scaling at matched compute", fontsize=20, loc="left")
    axB.legend(fontsize=15, loc="upper right")

    fig.suptitle(f"Full data (df=1.0, 100M tokens), d12/w768 base  —  strategy: {STRAT}  (Panel A: per-epoch mean)",
                 fontsize=20, y=1.02)
    fig.tight_layout()

    outdir = os.path.join(REPO, "experiments/figures/02_ensemble_scaling")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "expt_fig2_combined_fulldata_epoch.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)

    print(f"strategy = {STRAT}  (Panel A at epoch resolution)")
    print(f"Panel A E=1 epochs covered: {e1[0] if len(e1) else '-'}..{e1[-1] if len(e1) else '-'}")
    print("Ensemble L* :", {E: round(Lstar_ens[E], 4) for E in Es})
    print("Width    L* :", {w: (round(Lstar_w[w], 4), f"n={n_ind_w[w]}") for w in ws})
    if 1536 in Lstar_w and 4 in Lstar_ens:
        print(f"Delta @ 4x  : width(w1536)={Lstar_w[1536]:.4f}  ens(E=4)={Lstar_ens[4]:.4f}  "
              f"Delta={Lstar_w[1536]-Lstar_ens[4]:.4f}")
    print(f"\nSaved {out}\nSaved {out.replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
