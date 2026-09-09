"""Does the model-size law's floor depend on data the way the data law says?

Blake's question: the model-size fits  L* = L_inf + c N^-alpha  are done at one
corpus size. Their intercept L_inf -- the loss an infinitely large model would
reach with optimal stopping -- must depend on P. Do we understand how, and does
it agree with the law obtained by holding the model fixed and varying P?

We have the model-size law at two corpus sizes:
  P=100M   the 12 cells used everywhere else (lambda=0, constant LR, 40 epochs,
           model 0)
  P=20M    the 4x4 grid in data_export/expt3_grid/ (L in {6,12,18,24} x W in
           {384,768,1152,1536}; 5 individuals, init_shuffle; 25 epochs;
           trapezoidal warmdown schedule)

and the fixed-model data law at L=12,W=768 from Figure 1 row 1.

Fits are done on each grid's full cell set AND on the 10 cells the two grids
share (the L=6 and L=12 rows plus L=18/W=768 and L=24/W=768), so the intercept
comparison is not confounded by grid shape.

RESULT, which changes what can be concluded: at P=20M the minimum loss does
not depend on model size at all -- 16 cells spanning 14M to 906M parameters
all land in 4.46-4.56, and the saturating fit returns c<0. So the "floor" at
20M is simply that level, not an extrapolation. At P=100M the same span of
models buys 0.30 nats.

CAVEATS carried into every output: the 20M grid ran at lambda=0.1 with the
trapezoidal warmdown schedule for 25 epochs; the 100M cells ran at lambda=0,
constant LR, 40 epochs. At d12/w768, P=20M, we also have lambda=0.1 under
constant LR (q1 v1) and lambda=0 under constant LR (Figure 1 row 1), which
separates the two confounds: the schedule moves the minimum by ~0.03, the
weight decay by ~0.26.

Outputs:
  experiments/figures/13_floor_vs_data/expt_floor_vs_data.{pdf,png}
  experiments/figures/13_floor_vs_data/expt_floor_vs_data_fits.csv
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from expt_fig1_sister_panels import (  # noqa: E402
    CELLS, CORPUS_TOKENS, find_nadir, load_cell, load_datasize_runs, saturating_fit, setup_style,
)
from expt_fig3_loader import load_grid  # noqa: E402

REPO = HERE.parents[1]
LOGS = REPO / "experiments" / "logs"
OUTDIR = REPO / "experiments" / "figures" / "13_floor_vs_data"
STRAT = "init_shuffle_ens"
VAL_RE = re.compile(r"\[model \d+ val @ step (\d+)\] val_loss=([\d.]+)")


def separable_fit(L, W, y):
    def resid(t):
        return t[0] + t[1] * L ** (-t[2]) + t[3] * W ** (-t[4]) - y
    r = least_squares(resid, [min(y) * 0.98, 1.0, 0.3, 5.0, 0.3], method="trf", max_nfev=200000)
    ss = float(np.sum(r.fun ** 2)); sst = float(np.sum((y - y.mean()) ** 2))
    return [float(v) for v in r.x] + [1 - ss / sst]


def grid_100m():
    out = {}
    for k, pat in CELLS.items():
        c = load_cell(pat)
        if c is not None:
            out[k] = float(np.min(c["val"]))
    return out


def grid_20m():
    """(L, W) -> mean over the 5 individuals of each one's minimum val loss."""
    g = load_grid()
    out = {}
    for (d, w, s), cell in g.items():
        if s != STRAT or not cell.individuals:
            continue
        out[(d, w)] = float(np.mean([vl.min() for _, vl in cell.individuals]))
    return out


def q1_constant_lr_20m():
    """d12/w768 at df=0.2 under CONSTANT LR (q1 sweep) -- bounds the schedule effect."""
    best = []
    for f in glob.glob(str(LOGS / "q1_train_d12_w768_df0.2_*_*.out")):
        pts = [float(m.group(2)) for line in open(f, errors="ignore") for m in [VAL_RE.search(line)] if m]
        if pts:
            best.append(min(pts))
    return float(np.mean(best)) if best else None


def fits_for(cells, label):
    ks = sorted(cells)
    L = np.array([k[0] for k in ks], float); W = np.array([k[1] for k in ks], float)
    y = np.array([cells[k] for k in ks], float); N = 16 * L * W ** 2 / 1e6
    Linf, c, a, r2 = saturating_fit(N, y)
    sLinf, cL, aL, cW, aW, r2s = separable_fit(L, W, y)
    return dict(label=label, n=len(ks), N=N, y=y, Linf=Linf, c=c, alpha=a, R2=r2,
                sep_Linf=sLinf, cL=cL, aL=aL, cW=cW, aW=aW, sep_R2=r2s,
                valid=bool(c > 0), level=float(y.mean()), lo=float(y.min()), hi=float(y.max()))


def main():
    setup_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    g100, g20 = grid_100m(), grid_20m()
    common = sorted(set(g100) & set(g20))
    F = [fits_for(g100, "P=100M, all 12 cells"),
         fits_for(g20, "P=20M, all 16 cells"),
         fits_for({k: g100[k] for k in common}, "P=100M, 10 shared cells"),
         fits_for({k: g20[k] for k in common}, "P=20M, 10 shared cells")]

    # fixed-model data law (row 1 of Figure 1), strategies averaged
    runs = load_datasize_runs()
    Ps = np.array(sorted(runs), float) * CORPUS_TOKENS
    LP = np.array([np.mean([e["val_loss"].min() for e in runs[df]]) for df in sorted(runs)])
    LinfP, cP, aP, r2P = saturating_fit(Ps / 1e6, LP)
    same_model_gap = float(np.interp(20e6, Ps, LP) - np.interp(100e6, Ps, LP))
    row1_20m = float(np.interp(20e6, Ps, LP))          # lambda=0, constant LR, d12/w768
    sched = q1_constant_lr_20m()                        # lambda=0.1, constant LR, d12/w768
    grid_20m_d12 = g20[(12, 768)]                       # lambda=0.1, warmdown, d12/w768
    additive_pred = F[0]["Linf"] + same_model_gap       # floor at 20M if the data term were additive

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(22, 8.8))
    fig.subplots_adjust(wspace=0.30)
    colP = {100: "0.15", 20: sns.color_palette("cool", 3)[0]}

    ax = axes[0]
    for f, P, mk in ((F[0], 100, "o"), (F[1], 20, "^")):
        ax.scatter(f["N"] * 1e6, f["y"], s=170, marker=mk, color=colP[P], edgecolor="black",
                   linewidth=0.9, zorder=4, label=fr"$P$={P}M  ({f['n']} cells)")
        g = np.logspace(np.log10(f["N"].min()), np.log10(f["N"].max()), 200)
        if f["valid"]:
            ax.plot(g * 1e6, f["Linf"] + f["c"] * g ** (-f["alpha"]), "--", color=colP[P], lw=2.8, zorder=2,
                    label=fr"fit: $\mathcal{{L}}_\infty$={f['Linf']:.3f}, $\alpha$={f['alpha']:.2f}")
            ax.axhline(f["Linf"], color=colP[P], ls=":", lw=1.8, alpha=0.8)
        else:
            ax.axhspan(f["lo"], f["hi"], color=colP[P], alpha=0.12, zorder=1)
            ax.axhline(f["level"], color=colP[P], ls="--", lw=2.8, zorder=2,
                       label=fr"no size dependence: level {f['level']:.2f}, range {f['lo']:.2f}-{f['hi']:.2f}")
    ax.annotate(r"$\lambda$=0.1, warmdown, 25 ep", xy=(0.03, 0.80), xycoords="axes fraction",
                fontsize=14, color=colP[20])
    ax.annotate(r"$\lambda$=0, constant LR, 40 ep", xy=(0.03, 0.30), xycoords="axes fraction",
                fontsize=14, color=colP[100])
    ax.set_xscale("log")
    ax.set_xticks([2e7, 5e7, 1e8, 2e8, 5e8])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"non-embedding parameters  $N=16LW^2$", fontsize=24)
    ax.set_ylabel(r"min val loss  $\mathcal{L}^\ast$", fontsize=26)
    ax.set_title("(A)  the model-size law at two corpus sizes", fontsize=20, loc="left")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=14)

    ax = axes[1]
    ax.scatter(Ps, LP, s=170, color="0.55", edgecolor="black", linewidth=0.9, zorder=4,
               label=r"$L$=12,$W$=768 at $\lambda$=0 (Figure 1 row 1)")
    gP = np.logspace(np.log10(Ps.min() / 1e6), np.log10(Ps.max() / 1e6), 200)
    ax.plot(gP * 1e6, LinfP + cP * gP ** (-aP), "--", color="0.45", lw=2.8, zorder=2,
            label=fr"data law at fixed model: $\alpha_P$={aP:.2f}")
    # the 100M floor and the additive-prediction for 20M
    ax.scatter([100e6], [F[0]["Linf"]], s=300, marker="o", color=colP[100], edgecolor="black",
               linewidth=1.2, zorder=6, label=r"$\mathcal{L}_\infty$(100M) from the model-size law")
    ax.scatter([20e6], [additive_pred], s=300, marker="o", facecolors="none", edgecolors=colP[100],
               linewidth=2.5, zorder=6,
               label=fr"$\mathcal{{L}}_\infty$(20M) if the data term were additive: {additive_pred:.2f}")
    ax.plot([20e6, 100e6], [additive_pred, F[0]["Linf"]], ":", color=colP[100], lw=2.2, zorder=3)
    # what we actually measure at 20M, by setting
    ax.scatter([20e6], [F[1]["level"]], s=300, marker="^", color=colP[20], edgecolor="black",
               linewidth=1.2, zorder=6,
               label=fr"20M grid level, any $N$ ($\lambda$=0.1, warmdown): {F[1]['level']:.2f}")
    if sched is not None:
        ax.scatter([20e6], [sched], s=260, marker="x", color=colP[20], linewidth=3.0, zorder=6,
                   label=fr"$L$12/$W$768 at $\lambda$=0.1, constant LR: {sched:.2f}")
    ax.annotate("", xy=(20e6, row1_20m), xytext=(20e6, sched if sched else F[1]["level"]),
                arrowprops=dict(arrowstyle="<->", lw=1.6, color="0.3"))
    ax.text(21.5e6, 0.5 * (row1_20m + (sched if sched else F[1]["level"])),
            fr"$\lambda$: {row1_20m - (sched if sched else F[1]['level']):.2f}", fontsize=14, color="0.3", va="center")
    ax.set_xscale("log")
    ax.set_xticks([1e7, 2e7, 5e7, 1e8])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"unique tokens  $P$", fontsize=24)
    ax.set_ylabel(r"$\mathcal{L}^\ast$  or  $\mathcal{L}_\infty$", fontsize=26)
    ax.set_title("(B)  the floor at 20M sits above the additive prediction", fontsize=20, loc="left")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=12)

    for ext in ("pdf", "png"):
        p = OUTDIR / f"expt_floor_vs_data.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"saved {p}")
    plt.close(fig)

    # ------------------------------------------------------------- outputs
    dLinf_all = F[1]["Linf"] - F[0]["Linf"]
    dLinf_common = F[3]["Linf"] - F[2]["Linf"]
    with open(OUTDIR / "expt_floor_vs_data_fits.csv", "w") as fh:
        fh.write("fit,n_cells,L_inf,c,alpha,R2,sep_L_inf,c_L,alpha_L,c_W,alpha_W,sep_R2,valid,level,min,max\n")
        for f in F:
            fh.write(f"{f['label']},{f['n']},{f['Linf']:.4f},{f['c']:.4f},{f['alpha']:.4f},{f['R2']:.4f},"
                     f"{f['sep_Linf']:.4f},{f['cL']:.4f},{f['aL']:.4f},{f['cW']:.4f},{f['aW']:.4f},{f['sep_R2']:.4f},"
                     f"{'yes' if f['valid'] else 'NO (c<0: no N dependence)'},{f['level']:.4f},{f['lo']:.4f},{f['hi']:.4f}\n")
        fh.write(f"data law (fixed L12/W768),{len(Ps)},{LinfP:.4f},{cP:.4f},{aP:.4f},{r2P:.4f},,,,,,\n")
        fh.write("\ncomparison,value,note\n")
        fh.write(f"floor gap L_inf(20M)-L_inf(100M) all cells,{dLinf_all:.4f},\n")
        fh.write(f"floor gap L_inf(20M)-L_inf(100M) shared 10 cells,{dLinf_common:.4f},\n")
        fh.write(f"same-model gap L*(20M)-L*(100M) at L12/W768,{same_model_gap:.4f},from Figure 1 row 1 (lambda=0)\n")
        fh.write(f"additive prediction for L_inf(20M),{additive_pred:.4f},L_inf(100M) + same-model gap\n")
        fh.write(f"measured 20M level (lambda=0.1 warmdown; any N),{F[1]['level']:.4f},range {F[1]['lo']:.4f}-{F[1]['hi']:.4f}\n")
        fh.write(f"d12/w768 at P=20M lambda=0 constant LR (row 1),{row1_20m:.4f},\n")
        fh.write(f"schedule check d12/w768 P=20M warmdown grid,{g20[(12,768)]:.4f},25 epochs\n")
        if sched is not None:
            fh.write(f"schedule check d12/w768 P=20M constant LR (q1),{sched:.4f},50 epochs; "
                     f"warmdown lower by {sched - g20[(12,768)]:.4f}\n")
    print(f"saved {OUTDIR / 'expt_floor_vs_data_fits.csv'}")

    print(f"\n{'fit':>28} {'n':>3} {'L_inf':>7} {'c':>7} {'alpha':>6} {'R2':>6} | {'sepL_inf':>8} {'a_L':>6} {'a_W':>6} {'R2':>6}")
    for f in F:
        print(f"{f['label']:>28} {f['n']:>3} {f['Linf']:>7.4f} {f['c']:>7.4f} {f['alpha']:>6.3f} {f['R2']:>6.3f} | "
              f"{f['sep_Linf']:>8.4f} {f['aL']:>6.3f} {f['aW']:>6.3f} {f['sep_R2']:>6.3f}")
    print(f"\n20M grid: saturating fit {'VALID' if F[1]['valid'] else 'INVALID (c<0)'}; level {F[1]['level']:.4f}, "
          f"range {F[1]['lo']:.4f}-{F[1]['hi']:.4f} over {F[1]['n']} cells, N from {F[1]['N'].min():.0f}M to {F[1]['N'].max():.0f}M")
    print(f"100M grid: L*(N) falls {F[0]['y'].max()-F[0]['y'].min():.4f} over the same kind of span")
    print(f"\nadditive prediction L_inf(20M) = L_inf(100M) + same-model gap = {F[0]['Linf']:.4f} + {same_model_gap:.4f} = {additive_pred:.4f}")
    print(f"measured level at 20M: {F[1]['level']:.4f} (lambda=0.1) ... and lambda=0 would be higher still (d12/w768: {row1_20m:.4f})")
    print(f"same-model gap at L12/W768 from the data sweep:  {same_model_gap:.4f}")
    print(f"data law: L* = {LinfP:.3f} + {cP:.3f} P_M^-{aP:.3f}  (R2={r2P:.3f})")
    print(f"\nschedule confound at d12/w768, P=20M: warmdown grid {g20[(12,768)]:.4f}"
          + (f"  vs constant-LR q1 {sched:.4f}  (warmdown lower by {sched - g20[(12,768)]:.4f})" if sched else ""))


if __name__ == "__main__":
    main()
