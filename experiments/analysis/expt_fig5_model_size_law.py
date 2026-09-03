"""Model-size scaling law at df=1.0 (100M tokens) — the missing term in Figure 5.

The paper's joint fit L(s,P,E,N) carries a width term c2*N^d_N, but the figure
only ever varies P (data) and E (ensemble size); the caption itself asks "what
about the model size scaling law?". This fits it, over BOTH size axes.

Data: every (depth, width) cell we have at df=1.0, lambda=0, constant LR, E=1,
model 0 -- 12 cells spanning L in {6..60} and N in {384..1536}, a 40x range in
non-embedding parameters.

Two competing forms:

  (a) COLLAPSED   L = L_inf + c * P^-alpha,          P = 16*L*N^2
      depth and width interchangeable -- only the parameter count matters.

  (b) SEPARABLE   L = L_inf + c_L * L^-a_L + c_N * N^-a_N
      depth and width contribute independently, with their own exponents.

If (b) beats (a) materially, "model size" is not a scalar for this model and the
paper's single N term is under-specified.

Caveat carried into the output: these are lambda=0 runs, i.e. the unregularized
regime where we separately showed capacity gains are heavily suppressed (the
whole depth axis moves only 0.155 nats, versus 0.336 from tuning lambda at fixed
size). The exponents here therefore describe the UNREGULARIZED capacity curve.

Outputs:
  experiments/figures/04_scaling_law/expt_fig5_model_size_law.{pdf,png}
  experiments/figures/04_scaling_law/expt_fig5_model_size_law_fits.csv
"""
from __future__ import annotations

import re
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "experiments" / "logs"
OUTDIR = REPO / "experiments" / "figures" / "04_scaling_law"
IND = re.compile(r"\[model \d+ val @ step (\d+)\] val_loss=([\d.]+)")

CELLS = {
    (6, 384): "fd_train_d6_w384_*_0.out",      (6, 768): "fd_train_d6_w768_*_0.out",
    (6, 1152): "fd_gridfill_d6_w1152_*_0.out", (6, 1536): "fd_gridfill_d6_w1536_*_0.out",
    (12, 384): "fd_train_d12_w384_*_0.out",    (12, 768): "fd_train_d12_w768_*_0.out",
    (12, 1152): "fd_widthext_d12_w1152_*_0.out", (12, 1536): "fd_widthext_d12_w1536_*_0.out",
    (18, 768): "fd_gridfill_d18_w768_*_0.out", (24, 768): "fd_gridfill_d24_w768_*_0.out",
    (48, 768): "fd_gridfill_d48_w768_*_0.out", (60, 768): "fd_gridfill_d60_w768_*_0.out",
}


def min_val(pattern):
    pts = {}
    for f in glob.glob(str(LOGS / pattern)):
        with open(f, errors="ignore") as fh:
            for line in fh:
                m = IND.search(line)
                if m:
                    pts[int(m.group(1))] = float(m.group(2))
    return min(pts.values()) if pts else None


def params_m(L, N):
    return 16 * L * N**2 / 1e6


def fit(resid, x0, n_obs, n_par):
    r = least_squares(resid, x0, method="trf", max_nfev=200000)
    ss = float(np.sum(r.fun**2))
    return r.x, ss, ss / max(n_obs - n_par, 1)


def main():
    data = {k: min_val(p) for k, p in CELLS.items()}
    data = {k: v for k, v in data.items() if v is not None}
    Ls = np.array([k[0] for k in data], float)
    Ns = np.array([k[1] for k in data], float)
    ys = np.array([data[k] for k in data], float)
    P = 16 * Ls * Ns**2 / 1e6
    ybar = ys.mean()
    sstot = float(np.sum((ys - ybar) ** 2))

    # (a) collapsed: only the parameter count matters
    def r_col(t):
        Linf, c, a = t
        return Linf + c * P ** (-a) - ys
    pa, ssa, msa = fit(r_col, [3.5, 5.0, 0.3], len(ys), 3)

    # (b) separable: depth and width get their own exponents
    def r_sep(t):
        Linf, cL, aL, cN, aN = t
        return Linf + cL * Ls ** (-aL) + cN * Ns ** (-aN) - ys
    pb, ssb, msb = fit(r_sep, [3.5, 1.0, 0.3, 5.0, 0.3], len(ys), 5)

    print(f"n = {len(ys)} cells,  L in {sorted(set(Ls.astype(int)))},  N in {sorted(set(Ns.astype(int)))}")
    print(f"\n(a) COLLAPSED  L = {pa[0]:.4f} + {pa[1]:.4f} * P^-{pa[2]:.4f}")
    print(f"    SSE={ssa:.5f}  R2={1-ssa/sstot:.4f}  MSE/dof={msa:.6f}  rms={np.sqrt(ssa/len(ys)):.4f}")
    print(f"\n(b) SEPARABLE  L = {pb[0]:.4f} + {pb[1]:.4f} * L^-{pb[2]:.4f} + {pb[3]:.4f} * N^-{pb[4]:.4f}")
    print(f"    SSE={ssb:.5f}  R2={1-ssb/sstot:.4f}  MSE/dof={msb:.6f}  rms={np.sqrt(ssb/len(ys)):.4f}")
    print(f"\n    depth exponent a_L = {pb[2]:.3f}   width exponent a_N = {pb[4]:.3f}")
    print(f"    variance explained by separating the axes: {(ssa-ssb)/ssa*100:.1f}% of the collapsed SSE")

    print("\nper-cell residuals (separable fit):")
    pred = pb[0] + pb[1] * Ls ** (-pb[2]) + pb[3] * Ns ** (-pb[4])
    for (L, N), y, pr in sorted(zip(data.keys(), ys, pred)):
        print(f"  d{int(L):<3}w{int(N):<5} obs={y:.4f} fit={pr:.4f} resid={y-pr:+.4f}")

    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")
    plt.rcParams["axes.linewidth"] = 3.5
    plt.rcParams["grid.alpha"] = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))

    # (1) observed vs params, coloured by depth, with the collapsed fit overlaid
    ax = axes[0]
    depths = sorted(set(Ls.astype(int)))
    cols = sns.color_palette("cool", len(depths))
    for c, d in zip(cols, depths):
        sel = [(p, y) for (L, N), p, y in zip(data.keys(), P, ys) if L == d]
        sel.sort()
        ax.plot([s[0] for s in sel], [s[1] for s in sel], "o-", color=c, lw=2.5, ms=10, label=f"$L$={d}")
    grid = np.logspace(np.log10(P.min()), np.log10(P.max()), 100)
    ax.plot(grid, pa[0] + pa[1] * grid ** (-pa[2]), "k--", lw=2.5, label="collapsed fit")
    ax.set_xscale("log")
    ax.set_xlabel(r"non-embedding params $16LN^2$ (M)")
    ax.set_ylabel(r"min val loss $\mathcal{L}^*$")
    ax.set_title(f"(A) collapsed: $P$ alone\n$R^2$={1-ssa/sstot:.3f}", fontsize=14)
    ax.legend(fontsize=11)

    # (2) predicted vs observed for both forms
    ax = axes[1]
    predA = pa[0] + pa[1] * P ** (-pa[2])
    ax.plot(ys, predA, "o", ms=11, color=cols[-1], label=f"collapsed ($R^2$={1-ssa/sstot:.3f})")
    ax.plot(ys, pred, "s", ms=11, color=cols[0], label=f"separable ($R^2$={1-ssb/sstot:.3f})")
    lo, hi = ys.min() - 0.02, ys.max() + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=2.0)
    ax.set_xlabel(r"observed $\mathcal{L}^*$")
    ax.set_ylabel(r"predicted $\mathcal{L}^*$")
    ax.set_title("(B) fit quality", fontsize=14)
    ax.legend(fontsize=12)

    # (3) the two exponents
    ax = axes[2]
    ax.bar([0, 1], [pb[2], pb[4]], color=[cols[0], cols[-1]], edgecolor="0.2", lw=2.5, width=0.55)
    for x, v in zip([0, 1], [pb[2], pb[4]]):
        ax.text(x, v * 1.03, f"{v:.3f}", ha="center", fontsize=16)
    ax.set_xticks([0, 1], [r"depth  $\alpha_L$", r"width  $\alpha_N$"])
    ax.set_ylabel("fitted exponent")
    ax.set_title("(C) depth and width are not\ninterchangeable", fontsize=14)

    fig.suptitle(r"Model-size scaling at $df=1.0$ (100M tokens), $\lambda=0$, $E=1$:  "
                 r"$\mathcal{L}^* = \mathcal{L}_\infty + c_L L^{-\alpha_L} + c_N N^{-\alpha_N}$",
                 fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUTDIR / f"expt_fig5_model_size_law.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"\nSaved {p}")

    with open(OUTDIR / "expt_fig5_model_size_law_fits.csv", "w") as fh:
        fh.write("form,param,value\n")
        for n, v in zip(("L_inf", "c", "alpha"), pa):
            fh.write(f"collapsed,{n},{v:.6f}\n")
        fh.write(f"collapsed,R2,{1-ssa/sstot:.6f}\n")
        for n, v in zip(("L_inf", "c_L", "alpha_L", "c_N", "alpha_N"), pb):
            fh.write(f"separable,{n},{v:.6f}\n")
        fh.write(f"separable,R2,{1-ssb/sstot:.6f}\n")
        fh.write("data,n_cells,%d\n" % len(ys))
    print(f"Saved {OUTDIR / 'expt_fig5_model_size_law_fits.csv'}")


if __name__ == "__main__":
    main()
