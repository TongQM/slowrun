"""Model-size scaling law at df=1.0 (100M tokens) -- the term Figure 5's caption
asks for, split into three standalone figures rather than one multi-panel plot:

  (1) COLLAPSED relation   L* vs P = 16*L*N^2 alone.
  (2) SEPARABLE relation   L* vs L and L* vs N as two independent partial-
                          residual plots (each axis's fit with the OTHER axis's
                          fitted contribution subtracted out), so the two power
                          laws are visible directly rather than as a bar chart
                          of exponents.
  (3) TRAINING CURVES      the raw val-loss-vs-steps curves every L* point in
                          (1)/(2) is a minimum of, so the reader can see e.g.
                          that d6/w1536 turns back up (overfits) rather than
                          simply plateauing.

Data: every (depth, width) cell we have at df=1.0, lambda=0, constant LR, E=1,
model 0 -- 12 cells spanning L in {6..60} and N in {384..1536}, a 40x range in
non-embedding parameters.

Two fitted forms, both still computed (fit numbers feed all three figures):

  collapsed:  L = L_inf + c * P^-alpha
  separable:  L = L_inf + c_L * L^-alpha_L + c_N * N^-alpha_N

Caveats carried into the captions:
  - lambda=0, i.e. the UNREGULARIZED regime, where capacity gains are heavily
    suppressed relative to tuning lambda at fixed size (see the companion
    capacity figure: 0.155 nats across the whole depth axis here, vs 0.336 nats
    from tuning lambda at fixed L=12).
  - d6/w1536 is non-monotone in the raw curve (min then rises), the largest
    residual in the separable fit.
  - the depth axis was measured under the pre-fix CompleteP residual scaling
    (see unlimited/train.py commit c660582); the depth exponent is provisional.

Outputs:
  experiments/figures/04_scaling_law/expt_fig5_model_size_collapsed.{pdf,png}
  experiments/figures/04_scaling_law/expt_fig5_model_size_separable.{pdf,png}
  experiments/figures/04_scaling_law/expt_fig5_model_size_curves.{pdf,png}
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


def curve(pattern):
    """step -> val_loss, merged across resume segments."""
    pts = {}
    for f in glob.glob(str(LOGS / pattern)):
        with open(f, errors="ignore") as fh:
            for line in fh:
                m = IND.search(line)
                if m:
                    pts[int(m.group(1))] = float(m.group(2))
    return pts


def min_val(pattern):
    pts = curve(pattern)
    return min(pts.values()) if pts else None


def params_m(L, N):
    return 16 * L * N**2 / 1e6


def fit(resid, x0):
    r = least_squares(resid, x0, method="trf", max_nfev=200000)
    return r.x, float(np.sum(r.fun**2))


def style():
    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")
    plt.rcParams["axes.linewidth"] = 3.5
    plt.rcParams["grid.alpha"] = 0.25


def main():
    data = {k: min_val(p) for k, p in CELLS.items()}
    data = {k: v for k, v in data.items() if v is not None}
    Ls = np.array([k[0] for k in data], float)
    Ns = np.array([k[1] for k in data], float)
    ys = np.array([data[k] for k in data], float)
    P = 16 * Ls * Ns**2 / 1e6
    ybar = ys.mean()
    sstot = float(np.sum((ys - ybar) ** 2))

    def r_col(t):
        Linf, c, a = t
        return Linf + c * P ** (-a) - ys
    pa, ssa = fit(r_col, [3.5, 5.0, 0.3])

    def r_sep(t):
        Linf, cL, aL, cN, aN = t
        return Linf + cL * Ls ** (-aL) + cN * Ns ** (-aN) - ys
    pb, ssb = fit(r_sep, [3.5, 1.0, 0.3, 5.0, 0.3])
    Linf, cL, aL, cN, aN = pb

    print(f"n = {len(ys)} cells,  L in {sorted(set(Ls.astype(int)))},  N in {sorted(set(Ns.astype(int)))}")
    print(f"\ncollapsed  L = {pa[0]:.4f} + {pa[1]:.4f} * P^-{pa[2]:.4f}   R2={1-ssa/sstot:.4f}")
    print(f"separable  L = {Linf:.4f} + {cL:.4f} * L^-{aL:.4f} + {cN:.4f} * N^-{aN:.4f}   R2={1-ssb/sstot:.4f}")
    print(f"  depth exponent a_L = {aL:.3f}   width exponent a_N = {aN:.3f}")
    print(f"  variance explained by separating the axes: {(ssa-ssb)/ssa*100:.1f}% of the collapsed SSE")

    style()
    cool_by_L = {d: c for d, c in zip(sorted(set(Ls.astype(int))),
                                      sns.color_palette("cool", len(set(Ls.astype(int)))))}
    marker_by_N = {384: "o", 768: "s", 1152: "^", 1536: "D"}

    # ---------------------------------------------------------------- (1)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for (L, N), p, y in zip(data.keys(), P, ys):
        ax.scatter(p, y, s=140, color=cool_by_L[int(L)], marker=marker_by_N[int(N)],
                   edgecolor="0.2", lw=1.3, zorder=3)
    grid = np.logspace(np.log10(P.min()), np.log10(P.max()), 200)
    ax.plot(grid, pa[0] + pa[1] * grid ** (-pa[2]), "k--", lw=2.5, zorder=2,
           label=fr"fit: $\mathcal{{L}}_\infty$+{pa[1]:.2f}$\,P^{{-{pa[2]:.2f}}}$")
    from matplotlib.lines import Line2D
    leg1 = [Line2D([0], [0], marker=marker_by_N[n], color="0.3", lw=0, markersize=10,
                   label=f"$N$={n}") for n in sorted(marker_by_N)]
    leg2 = [Line2D([0], [0], marker="o", color=cool_by_L[d], lw=0, markersize=10,
                   label=f"$L$={d}") for d in sorted(cool_by_L)]
    ax.set_xscale("log")
    ax.set_xlabel(r"non-embedding parameters  $P=16LN^2$  (M)")
    ax.set_ylabel(r"min val loss  $\mathcal{L}^*$")
    ax.set_title(f"Collapsed relation: $P$ alone   ($R^2$={1-ssa/sstot:.3f})", fontsize=15)
    l1 = ax.legend(handles=leg1, loc="upper right", title="width", fontsize=11, title_fontsize=11)
    ax.add_artist(l1)
    ax.legend(handles=leg2 + [Line2D([0], [0], color="k", ls="--", lw=2.5, label="fit")],
             loc="lower left", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        pth = OUTDIR / f"expt_fig5_model_size_collapsed.{ext}"
        OUTDIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(pth, bbox_inches="tight", dpi=300)
        print(f"Saved {pth}")
    plt.close(fig)

    # ---------------------------------------------------------------- (2)
    # Partial residuals: for the depth panel, remove the FITTED width term from
    # each point so only the L-dependence remains (and symmetrically for width).
    y_minus_width = ys - cN * Ns ** (-aN)
    y_minus_depth = ys - cL * Ls ** (-aL)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    ax = axes[0]
    for (L, N), x, y in zip(data.keys(), Ls, y_minus_width):
        ax.scatter(x, y, s=140, color=cool_by_L[int(L)], marker=marker_by_N[int(N)],
                  edgecolor="0.2", lw=1.3, zorder=3)
    gl = np.logspace(np.log10(Ls.min()), np.log10(Ls.max()), 200)
    ax.plot(gl, Linf + cL * gl ** (-aL), "k--", lw=2.5, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"depth  $L$")
    ax.set_ylabel(r"$\mathcal{L}^* - c_N N^{-\alpha_N}$  (width term removed)")
    ax.set_title(fr"depth relation: $\alpha_L$={aL:.3f}", fontsize=15)

    ax = axes[1]
    for (L, N), x, y in zip(data.keys(), Ns, y_minus_depth):
        ax.scatter(x, y, s=140, color=cool_by_L[int(L)], marker=marker_by_N[int(N)],
                  edgecolor="0.2", lw=1.3, zorder=3)
    gn = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 200)
    ax.plot(gn, Linf + cN * gn ** (-aN), "k--", lw=2.5, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"width  $N$")
    ax.set_ylabel(r"$\mathcal{L}^* - c_L L^{-\alpha_L}$  (depth term removed)")
    ax.set_title(fr"width relation: $\alpha_N$={aN:.3f}", fontsize=15)

    l1 = axes[1].legend(handles=leg1, loc="upper right", title="width", fontsize=10, title_fontsize=10)
    axes[1].add_artist(l1)
    axes[1].legend(handles=leg2, loc="lower left", title="depth", fontsize=10, title_fontsize=10)

    fig.suptitle(r"Separable relation:  $\mathcal{L}^*=\mathcal{L}_\infty+c_L L^{-\alpha_L}+c_N N^{-\alpha_N}$"
                fr"   ($R^2$={1-ssb/sstot:.3f})", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        pth = OUTDIR / f"expt_fig5_model_size_separable.{ext}"
        fig.savefig(pth, bbox_inches="tight", dpi=300)
        print(f"Saved {pth}")
    plt.close(fig)

    # ---------------------------------------------------------------- (3)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    TOTAL_BATCH_SIZE = 131072  # tokens/step, project default (see slowrun/CLAUDE.md)

    ax = axes[0]
    for (L, N) in data:
        if N != 768:
            continue
        c = curve(CELLS[(L, N)])
        st = np.array(sorted(c)); v = np.array([c[k] for k in st])
        tok_b = st * TOTAL_BATCH_SIZE / 1e9
        ax.plot(tok_b, v, "-", color=cool_by_L[int(L)], lw=2.5, label=f"$L$={L}")
    ax.set_xlabel("tokens seen (B, cumulative incl. repeats)")
    ax.set_ylabel(r"val loss $\mathcal{L}$")
    ax.set_title(r"(A) depth ladder, $N$=768 fixed", fontsize=15)
    ax.legend(fontsize=11)
    ax.set_ylim(3.5, 5.5)

    ax = axes[1]
    for (L, N) in data:
        if L not in (6, 12):
            continue
        c = curve(CELLS[(L, N)])
        st = np.array(sorted(c)); v = np.array([c[k] for k in st])
        tok_b = st * TOTAL_BATCH_SIZE / 1e9
        ls = "-" if L == 6 else "--"
        ax.plot(tok_b, v, ls, color=cool_by_L.get(int(L), "0.3"),
                marker=marker_by_N[int(N)], markevery=25, ms=7, lw=2.2,
                label=f"$L$={L}, $N$={N}")
    ax.set_xlabel("tokens seen (B, cumulative incl. repeats)")
    ax.set_ylabel(r"val loss $\mathcal{L}$")
    ax.set_title(r"(B) width ladder, $L\in\{6,12\}$", fontsize=15)
    ax.legend(fontsize=9, ncol=2)
    ax.set_ylim(3.5, 5.5)
    c6_1536 = curve(CELLS[(6, 1536)])
    st6 = np.array(sorted(c6_1536))
    turn_step = st6[np.argmin([c6_1536[k] for k in st6])]
    turn_tok = turn_step * TOTAL_BATCH_SIZE / 1e9
    ax.annotate(r"$d6/w1536$ turns back up",
               xy=(turn_tok, min(c6_1536.values())), xytext=(turn_tok * 0.35, 4.6),
               fontsize=11, color="0.25",
               arrowprops=dict(arrowstyle="->", lw=1.8, color="0.35"))

    fig.suptitle(r"Validation loss vs training, $df=1.0$, $\lambda=0$, constant LR, $E=1$"
                " -- every $\\mathcal{L}^*$ above is the minimum of one such curve", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("pdf", "png"):
        pth = OUTDIR / f"expt_fig5_model_size_curves.{ext}"
        fig.savefig(pth, bbox_inches="tight", dpi=300)
        print(f"Saved {pth}")
    plt.close(fig)

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
