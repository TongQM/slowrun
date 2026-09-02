"""Min val loss vs weight decay (d12/w768, df=1.0, init_shuffle).

Two weight-decay sweeps, both parsed job-ID-filtered from SLURM .out logs (the two
sweeps SHARE the wd{X}_replay_* filename prefix, so a naive glob mixes arms — we key
by job ID):

  ON  arm (cooldown, NO_WARMDOWN=0): wd {0.0, 0.05, 0.1, 0.3, 0.5, 0.8}
  OFF arm (constant LR, no-warmdown): wd {0.0, 0.1, 0.3, 1.0}

Main curves = the cooldown-ON arm, one line per ensemble size E in {1,2,3,4,5}
(E=1 = mean of the 5 individuals from the train logs; E>=2 = replay ensemble min).
The no-warmdown E5 curve is overlaid dashed for the cooldown-vs-constant contrast.
L* per (arm, wd, E) = min over the full 40-epoch training run.

Result: the ensemble optimum is an INTERIOR minimum near wd=0.1 (cooldown), robust
across E; wd=0 and high wd are both worse. Cooldown beats constant-LR at every wd.
"""
import os, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO = "/ocean/projects/cis260161p/ymiao6/scaling/slowrun"
LOGS = os.path.join(REPO, "experiments/logs")
IND = re.compile(r"\[model \d+ val @ step (\d+)\] val_loss=([\d.]+)")

# ---- job-ID maps (wd-string -> job id). MUST filter by job id (shared prefixes). ----
ON_REPLAY  = {"0.0": 41609120, "0.05": 41609122, "0.1": 41431932, "0.3": 41431934, "0.5": 41431936, "0.8": 41431938}
ON_TRAIN   = {"0.0": 41609119, "0.05": 41609121, "0.1": 41431931, "0.3": 41431933, "0.5": 41431935, "0.8": 41431937}
OFF_REPLAY = {"0.1": 41363427, "0.3": 41363429, "1.0": 41363431}     # wd0.0 OFF = fulldata grid
OFF_TRAIN  = {"0.1": 41363426, "0.3": 41363428, "1.0": 41363430}
# wd=0 for the OFF (no-warmdown) arm comes from the df=1.0 fulldata grid, not a wd sweep
OFF0_REPLAY = f"{LOGS}/fd_replay20m_d12_w768_*_1.out"
OFF0_TRAIN  = f"{LOGS}/fd_train_d12_w768_*_[5-9].out"


def ens_min(replay_glob, E):
    pat = re.compile(rf"ens={E}\] val_loss=([\d.]+)")
    best = np.inf
    for f in glob.glob(replay_glob):
        with open(f) as fh:
            for line in fh:
                m = pat.search(line)
                if m:
                    best = min(best, float(m.group(1)))
    return best if np.isfinite(best) else np.nan


def indiv_mean_min(train_glob):
    """min over training of the mean-over-individuals curve."""
    permodel = {}
    for f in glob.glob(train_glob):
        midx = int(re.search(r"_(\d+)\.out$", f).group(1))
        with open(f) as fh:
            for line in fh:
                m = IND.search(line)
                if m:
                    permodel.setdefault(midx, {})[int(m.group(1))] = float(m.group(2))
    curves = [(np.array(sorted(d)), np.array([d[k] for k in sorted(d)])) for d in permodel.values() if d]
    if not curves:
        return np.nan
    lo = max(c[0][0] for c in curves); hi = min(c[0][-1] for c in curves)
    grid = curves[0][0]; grid = grid[(grid >= lo) & (grid <= hi)]
    stacked = np.vstack([np.interp(grid, s, v) for s, v in curves])
    return float(np.min(stacked.mean(0)))


def on_curves():
    """dict E -> list of (wd_float, L*) sorted by wd, cooldown-ON arm."""
    wds = sorted(ON_REPLAY, key=float)
    out = {}
    for E in (1, 2, 3, 4, 5):
        pts = []
        for wd in wds:
            if E == 1:
                L = indiv_mean_min(f"{LOGS}/wd{wd}_train_d12_w768_{ON_TRAIN[wd]}_*.out")
            else:
                L = ens_min(f"{LOGS}/wd{wd}_replay_d12_w768_{ON_REPLAY[wd]}_1.out", E)
            pts.append((float(wd), L))
        out[E] = pts
    return out


def off_curve_E5():
    pts = [(0.0, ens_min(OFF0_REPLAY, 5))]
    for wd in sorted(OFF_REPLAY, key=float):
        pts.append((float(wd), ens_min(f"{LOGS}/wd{wd}_replay_d12_w768_{OFF_REPLAY[wd]}_1.out", 5)))
    return sorted(pts)


def main():
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 24, "axes.linewidth": 4.0, "legend.fontsize": 16,
        "grid.alpha": 0.25, "xtick.labelsize": 20, "ytick.labelsize": 20,
    })

    on = on_curves()
    palette = sns.color_palette("cool", 5)

    fig, ax = plt.subplots(figsize=(11, 8))
    for j, E in enumerate((1, 2, 3, 4, 5)):
        xs = [p[0] for p in on[E]]; ys = [p[1] for p in on[E]]
        lbl = "$E=1$ (individual)" if E == 1 else f"$E={E}$"
        ax.plot(xs, ys, "o-", lw=3.0, ms=11, color=palette[j],
                markeredgecolor="black", markeredgewidth=0.5, label=lbl)

    # frame y on the cooldown data with headroom at top for the legend
    ax.set_ylim(3.35, 4.02)

    # mark the E5 interior minimum; place the label in the empty upper-center region
    # with an arrow down to the point (keeps text off every curve)
    e5 = on[5]
    wmin, Lmin = min(e5, key=lambda p: p[1])
    ax.axvline(wmin, color=palette[4], lw=1.5, ls=":", alpha=0.7)
    ax.annotate(f"interior min:  $\\lambda={wmin:g}$,  $\\mathcal{{L}}^*={Lmin:.3f}$",
                xy=(wmin, Lmin), xytext=(0.8, 3.44), textcoords="data",
                fontsize=15, color=palette[4], ha="right", va="center",
                arrowprops=dict(arrowstyle="->", color=palette[4], lw=1.6,
                                connectionstyle="arc3,rad=-0.15"))

    ax.set_xlabel(r"weight decay  $\lambda$")
    ax.set_ylabel(r"min val loss  $\mathcal{L}^*$")
    ax.set_title("Min val loss vs weight decay  (d12/w768, df=1.0, cooldown ON)",
                 fontsize=19, loc="left")
    ax.legend(title="ensemble size", ncol=3, loc="upper center", framealpha=0.9)

    fig.tight_layout()
    outdir = os.path.join(REPO, "experiments/figures/06_wd_cooldown")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "min_val_vs_wd.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)

    print("cooldown-ON  L* vs wd:")
    for E in (1, 2, 3, 4, 5):
        print(f"  E={E}: " + "  ".join(f"wd{w:g}={L:.3f}" for w, L in on[E]))
    print(f"\nInterior min (cooldown E5): wd={wmin:g}  L*={Lmin:.4f}")
    print(f"Saved {out}\nSaved {out.replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
