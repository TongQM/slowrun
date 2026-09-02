"""Expt Fig 2 (combined) for df=1.0 / full 100M-token data, base cell d12/w768.

Two panels (matching the df=0.2 combined figure):
  (A) ensembling reduces loss across training:
      val loss vs per-model optimizer step s, for ensemble sizes
      E in {1 (mean of individuals), 2, 3, 4, 5}.  1 epoch ~= 763 steps at df=1.0.
  (B) ensembling beats width-scaling at matched compute:
      min val loss L* vs compute (relative to d12/w768, E=1), comparing
        - width scaling  : d=12, E=1, widths {384, 768, 1152, 1536}
                           (compute ratio = (w/768)^2)
        - ensembling     : d=12, w=768, E in {1..5}  (compute ratio = E)
      annotated with Delta = L*_width(4x) - L*_ens(4x) at matched 4x compute.

All curves parsed from SLURM .out logs (offline wandb is fragmented across the
resume segments, per project convention), NOT wandb:
  individuals : fd_train_d12_w{384,768}_*_{idx}.out          (idx = strategy*5 + model)
                fd_widthext_d12_w{1152,1536}_*_0.out          (E=1 width-sweep extension)
  ensembles   : fd_replay20m_d12_w768_*_{sidx}.out            (sidx: 0=init, 1=init_shuffle)

Width points w1152/w1536 are single-model (init) E=1 estimates from the width
extension; w384/w768 E=1 is the mean over the 5 individuals. If the extension
logs are absent/empty the script still renders with whatever widths are present.
"""
import os, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO = "/ocean/projects/cis260161p/ymiao6/scaling/slowrun"
LOGS = os.path.join(REPO, "experiments/logs")
BATCH = 131072
NUM_MODELS = 5
BASE_W = 768
# strategy for Panel A + the ensemble curve in Panel B. init_shuffle = independent
# per-(model,epoch) shuffle (the more-diverse, stronger-ensembling strategy).
STRAT = "init_shuffle"          # "init" or "init_shuffle"
STRAT_SIDX = {"init": 0, "init_shuffle": 1}
ENS_SIZES = [2, 3, 4, 5]
WIDTHS = [384, 768, 1152, 1536]
ENS_LINE = re.compile(r"\[step \d+ ens=([2-5])\] val_loss=([\d.]+) val_bpb=[\d.]+ tokens=(\d+)")
IND_LINE = re.compile(r"\[model \d+ val @ step (\d+)\] val_loss=([\d.]+)")


# ---------- parsing ----------
def indiv_curves_for_width(w):
    """List of (steps, val) arrays, one per individual model, for d12/w{w}.
    w in {384,768}: the 5 individuals of strategy STRAT (idx group).
    w in {1152,1536}: the single init model from the width extension."""
    curves = []
    if w in (384, 768):
        sidx = STRAT_SIDX[STRAT]
        for m in range(NUM_MODELS):
            task = sidx * NUM_MODELS + m
            d = {}
            for f in sorted(glob.glob(f"{LOGS}/fd_train_d12_w{w}_*_{task}.out")):
                with open(f) as fh:
                    for line in fh:
                        mm = IND_LINE.search(line)
                        if mm:
                            d[int(mm.group(1))] = float(mm.group(2))
            if d:
                s = np.array(sorted(d))
                curves.append((s, np.array([d[k] for k in s])))
    else:  # width-extension cells: single init model (task 0)
        d = {}
        for f in sorted(glob.glob(f"{LOGS}/fd_widthext_d12_w{w}_*_0.out")):
            with open(f) as fh:
                for line in fh:
                    mm = IND_LINE.search(line)
                    if mm:
                        d[int(mm.group(1))] = float(mm.group(2))
        if d:
            s = np.array(sorted(d))
            curves.append((s, np.array([d[k] for k in s])))
    return curves


def mean_indiv_curve(curves):
    """Mean val curve over individuals on the finest common step grid."""
    if not curves:
        return None, None
    lo = max(c[0][0] for c in curves)
    hi = min(c[0][-1] for c in curves)
    grid = curves[0][0]
    grid = grid[(grid >= lo) & (grid <= hi)]
    stacked = np.vstack([np.interp(grid, s, v) for s, v in curves])
    return grid, stacked.mean(0)


def ensemble_curves():
    """dict E -> (steps, val) for the base cell d12/w768, strategy STRAT."""
    sidx = STRAT_SIDX[STRAT]
    perE = {}
    for f in sorted(glob.glob(f"{LOGS}/fd_replay20m_d12_w768_*_{sidx}.out")):
        with open(f) as fh:
            for line in fh:
                m = ENS_LINE.search(line)
                if m:
                    E = int(m.group(1)); tok = int(m.group(3))
                    perE.setdefault(E, {})[tok // BATCH] = float(m.group(2))
    out = {}
    for E, d in perE.items():
        s = np.array(sorted(d))
        out[E] = (s, np.array([d[k] for k in s]))
    return out


# ---------- main ----------
def main():
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 24, "axes.linewidth": 4.0, "legend.fontsize": 18,
        "grid.alpha": 0.25, "xtick.labelsize": 20, "ytick.labelsize": 20,
    })

    # E=1 (mean of individuals at base width) and the ensembles
    base_indiv = indiv_curves_for_width(BASE_W)
    g1, v1 = mean_indiv_curve(base_indiv)
    ens = ensemble_curves()
    E_present = [1] + sorted(ens)
    palette = sns.color_palette("cool", len(E_present))

    # min val loss L* per ensemble size (read off the dip)
    Lstar_ens = {1: float(np.min(v1))}
    for E in sorted(ens):
        Lstar_ens[E] = float(np.min(ens[E][1]))

    # width sweep: L*(w) at E=1
    Lstar_w, n_ind_w = {}, {}
    for w in WIDTHS:
        cs = indiv_curves_for_width(w)
        gw, vw = mean_indiv_curve(cs)
        if vw is not None:
            Lstar_w[w] = float(np.min(vw)); n_ind_w[w] = len(cs)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(20, 7.5))

    # ===== Panel A =====
    axA.plot(g1, v1, lw=3.0, color=palette[0], label="$E = 1$  (mean of individuals)")
    for j, E in enumerate(sorted(ens)):
        s, v = ens[E]
        axA.plot(s, v, lw=3.0, color=palette[j + 1], label=f"$E = {E}$")
    axA.set_xlabel(r"steps $s$  (1 epoch $\approx 763$ steps)")
    axA.set_ylabel(r"validation loss  $\mathcal{L}$")
    axA.set_title("(A)  ensembling reduces loss across training", fontsize=20, loc="left")
    axA.legend(fontsize=15)

    # ===== Panel B =====
    # width-scaling curve (cyan): compute = (w/768)^2
    ws = sorted(Lstar_w)
    wx = [(w / BASE_W) ** 2 for w in ws]
    wy = [Lstar_w[w] for w in ws]
    axB.plot(wx, wy, "o-", color=sns.color_palette("cool", 2)[0], lw=3.0, ms=12,
             markeredgecolor="black", markeredgewidth=0.6,
             label=r"vary width @ $d=12$, $E=1$")
    for w, x, y in zip(ws, wx, wy):
        axB.annotate(f"w={w}", (x, y), textcoords="offset points", xytext=(0, -18),
                     fontsize=13, ha="center", color=sns.color_palette("cool", 2)[0])

    # ensembling curve (magenta): compute = E
    Es = sorted(Lstar_ens)
    ex = [float(E) for E in Es]
    ey = [Lstar_ens[E] for E in Es]
    axB.plot(ex, ey, "s-", color=sns.color_palette("magma", 3)[1], lw=3.0, ms=12,
             markeredgecolor="black", markeredgewidth=0.6,
             label=r"ensemble at base ($d=12$, $w=768$)")
    for E, x, y in zip(Es, ex, ey):
        axB.annotate(f"E={E}", (x, y), textcoords="offset points", xytext=(6, 8),
                     fontsize=13, ha="left", color=sns.color_palette("magma", 3)[1])

    # Delta at matched 4x compute (width w1536 vs ensemble E=4), if both present
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

    fig.suptitle(f"Full data (df=1.0, 100M tokens), d12/w768 base  —  strategy: {STRAT}",
                 fontsize=20, y=1.02)
    fig.tight_layout()

    outdir = os.path.join(REPO, "experiments/figures/02_ensemble_scaling")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "expt_fig2_combined_fulldata.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)

    # ---- console summary ----
    print(f"strategy = {STRAT}")
    print("Panel A: E=1 mean-indiv min = %.4f over %d individuals" % (Lstar_ens[1], len(base_indiv)))
    print("Ensemble L* :", {E: round(Lstar_ens[E], 4) for E in Es})
    print("Width    L* :", {w: (round(Lstar_w[w], 4), f"n={n_ind_w[w]}") for w in ws})
    if 1536 in Lstar_w and 4 in Lstar_ens:
        print(f"Delta @ 4x  : width(w1536)={Lstar_w[1536]:.4f}  ens(E=4)={Lstar_ens[4]:.4f}  "
              f"Delta={Lstar_w[1536]-Lstar_ens[4]:.4f}")
    else:
        miss = [w for w in (1152, 1536) if w not in Lstar_w]
        print(f"NOTE: width extension not yet available for {miss} — Panel B width curve partial.")
    print(f"\nSaved {out}\nSaved {out.replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
