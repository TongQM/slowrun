"""Model-size scaling law at P=100M on the ALIGNED grid -- three standalone figures.

  (1) COLLAPSED   L* vs N = 16 L W^2 alone, floor-free power law  L* = A N^-beta
  (2) SEPARABLE   L* = A L^-b_L W^-b_W as two partial-residual panels (each axis's
                  fit with the OTHER axis's fitted contribution divided out)
  (3) PROFILE     why the fits are floor-free: R^2 of the saturating form
                  L_inf + c N^-alpha as a function of a FIXED L_inf is flat from
                  L_inf = 2 up to ~3.5 and only then falls, so the asymptote (and
                  every exponent conditioned on it) is not identifiable from
                  14M-906M parameters at P=100M. The floor-free law is the limit
                  L_inf -> -inf of that family and is the one well-posed fit.

Data: expt_cells.ALIGNED_CELLS -- 18 cells at df=1.0, lambda=0, constant LR,
E=1, model 0: the 4x4 factorial L in {6,12,18,24} x W in {384,768,1152,1536}
plus d48/768 and d60/768, all under the corrected CompleteP residual scaling
(or on the L=12 row where the correction is the identity). 14M to 906M
non-embedding parameters, a 64x range.

Reading the separable exponents: doubling depth doubles N and buys b_L ln2 in
log-loss; doubling width quadruples N and buys b_W ln2. Per unit of log N that
is b_L for depth against b_W/2 for width, so the ratio b_W / (2 b_L) says
whether parameter count is a sufficient statistic (=1), or width (>1) or depth
(<1) is worth more per parameter.

Outputs:
  experiments/figures/04_scaling_law/expt_fig5_model_size_collapsed.{pdf,png}
  experiments/figures/04_scaling_law/expt_fig5_model_size_separable.{pdf,png}
  experiments/figures/04_scaling_law/expt_fig5_model_size_profile.{pdf,png}
  experiments/figures/04_scaling_law/expt_fig5_model_size_law.{pdf,png}   (1x3: A collapsed, B/C separable -- the manuscript figure)
  experiments/figures/04_scaling_law/expt_fig5_model_size_law_fits.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from expt_cells import ALIGNED_CELLS as CELLS  # noqa: E402
from expt_fig1_sister_panels import load_cell, setup_style  # noqa: E402

REPO = HERE.parents[1]
OUTDIR = REPO / "experiments" / "figures" / "04_scaling_law"


def r2(y, pred):
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def collapsed_fit(N, y):
    """log y = log A - beta log N. Returns A, beta, R2 (in log), R2 (in loss)."""
    a, b = np.polyfit(np.log(N), np.log(y), 1)
    return float(np.exp(b)), float(-a), r2(np.log(y), a * np.log(N) + b), r2(y, np.exp(b) * N ** a)


def separable_fit(L, W, y):
    """log y = log A - b_L log L - b_W log W. Returns A, b_L, b_W, R2, leave-one-out ranges."""
    X = np.column_stack([np.ones_like(L), -np.log(L), -np.log(W)])
    coef, *_ = np.linalg.lstsq(X, np.log(y), rcond=None)
    loo = []
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        c, *_ = np.linalg.lstsq(X[m], np.log(y[m]), rcond=None)
        loo.append((c[1], c[2]))
    loo = np.array(loo)
    return (float(np.exp(coef[0])), float(coef[1]), float(coef[2]), r2(np.log(y), X @ coef),
            (float(loo[:, 0].min()), float(loo[:, 0].max())), (float(loo[:, 1].min()), float(loo[:, 1].max())))


def profile(N, y, floors):
    """R^2 of L_inf + c N^-alpha with L_inf FIXED, for each floor."""
    out = []
    for Linf in floors:
        m = y > Linf
        if m.sum() < 3:
            out.append((Linf, np.nan, np.nan)); continue
        a, b = np.polyfit(np.log(N[m]), np.log(y[m] - Linf), 1)
        out.append((Linf, float(-a), r2(y, Linf + np.exp(b) * N ** a)))
    return out


def main():
    setup_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ks = sorted(CELLS, key=lambda k: 16 * k[0] * k[1] ** 2)
    L = np.array([k[0] for k in ks], float)
    W = np.array([k[1] for k in ks], float)
    y = np.array([float(np.min(load_cell(CELLS[k])["val"])) for k in ks])
    N = 16 * L * W ** 2 / 1e6

    A, beta, r2log, r2lin = collapsed_fit(N, y)
    As, bL, bW, r2s, looL, looW = separable_fit(L, W, y)
    floors = [2.0, 2.5, 3.0, 3.2, 3.4, 3.5, 3.6, 3.7, 3.75]
    prof = profile(N, y, floors)

    print(f"n = {len(y)} cells,  L in {sorted(set(L.astype(int)))},  W in {sorted(set(W.astype(int)))},  N {N.min():.0f}M-{N.max():.0f}M")
    print(f"collapsed  L* = {A:.3f} N_M^-{beta:.4f}   R2(log)={r2log:.3f} R2(loss)={r2lin:.3f}")
    print(f"separable  L* = {As:.3f} L^-{bL:.4f} W^-{bW:.4f}   R2={r2s:.3f}")
    print(f"  leave-one-out: b_L in [{looL[0]:.4f}, {looL[1]:.4f}], b_W in [{looW[0]:.4f}, {looW[1]:.4f}]")
    print(f"  b_W / (2 b_L) = {bW / (2 * bL):.2f}  -> depth worth {2 * bL / bW:.2f}x width per parameter")
    print("profile of the saturating form (fixed L_inf):")
    for Linf, al, rr in prof:
        print(f"  L_inf={Linf:.2f}  alpha={al:.3f}  R2={rr:.3f}")

    cool_by_L = {d: c for d, c in zip(sorted(set(L.astype(int))), sns.color_palette("cool", len(set(L.astype(int)))))}
    marker_by_W = {384: "o", 768: "s", 1152: "^", 1536: "D"}
    leg_w = [Line2D([0], [0], marker=marker_by_W[n], color="0.3", lw=0, markersize=10, label=f"$W$={n}") for n in sorted(marker_by_W)]
    leg_l = [Line2D([0], [0], marker="o", color=cool_by_L[d], lw=0, markersize=10, label=f"$L$={d}") for d in sorted(cool_by_L)]

    # ---------------------------------------------------------------- (1) collapsed
    fig, ax = plt.subplots(figsize=(8, 6.4))
    for k, n, v in zip(ks, N, y):
        ax.scatter(n, v, s=150, color=cool_by_L[k[0]], marker=marker_by_W[k[1]], edgecolor="0.2", lw=1.3, zorder=3)
    g = np.logspace(np.log10(N.min()), np.log10(N.max()), 200)
    ax.plot(g, A * g ** (-beta), "k--", lw=2.5, zorder=2, label=fr"$\mathcal{{L}}^*={A:.2f}\,N^{{-{beta:.3f}}}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_yticks([3.8, 3.9, 4.0, 4.1]); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xticks([20, 50, 100, 200, 500]); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}M"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"non-embedding parameters  $N=16LW^2$")
    ax.set_ylabel(r"min val loss  $\mathcal{L}^*$")
    ax.set_title(fr"Collapsed onto $N$: floor-free power law  ($R^2$={r2lin:.3f})", fontsize=15)
    l1 = ax.legend(handles=leg_w, loc="upper right", title="width", fontsize=11, title_fontsize=11); ax.add_artist(l1)
    ax.legend(handles=leg_l + [Line2D([0], [0], color="k", ls="--", lw=2.5, label="fit")], loc="lower left", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"expt_fig5_model_size_collapsed.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig); print(f"Saved {OUTDIR / 'expt_fig5_model_size_collapsed.pdf'}")

    # ---------------------------------------------------------------- (2) separable
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    ax = axes[0]
    yL = y * W ** bW          # divide out the fitted width factor -> A L^-b_L
    for k, x, v in zip(ks, L, yL):
        ax.scatter(x, v, s=140, color=cool_by_L[k[0]], marker=marker_by_W[k[1]], edgecolor="0.2", lw=1.3, zorder=3)
    gl = np.logspace(np.log10(L.min()), np.log10(L.max()), 200)
    ax.plot(gl, As * gl ** (-bL), "k--", lw=2.5, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([6, 12, 18, 24, 48, 60]); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"depth  $L$"); ax.set_ylabel(r"$\mathcal{L}^*\,W^{b_W}$   (width factor removed)")
    ax.set_title(fr"depth: $b_L$={bL:.4f}  (LOO {looL[0]:.4f}-{looL[1]:.4f})", fontsize=14)
    ax = axes[1]
    yW = y * L ** bL
    for k, x, v in zip(ks, W, yW):
        ax.scatter(x, v, s=140, color=cool_by_L[k[0]], marker=marker_by_W[k[1]], edgecolor="0.2", lw=1.3, zorder=3)
    gw = np.logspace(np.log10(W.min()), np.log10(W.max()), 200)
    ax.plot(gw, As * gw ** (-bW), "k--", lw=2.5, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([384, 768, 1152, 1536]); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"width  $W$"); ax.set_ylabel(r"$\mathcal{L}^*\,L^{b_L}$   (depth factor removed)")
    ax.set_title(fr"width: $b_W$={bW:.4f}  (LOO {looW[0]:.4f}-{looW[1]:.4f})", fontsize=14)
    for a_ in axes:
        a_.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}")); a_.yaxis.set_minor_formatter(plt.NullFormatter())
    l1 = axes[1].legend(handles=leg_w, loc="upper right", title="width", fontsize=10, title_fontsize=10); axes[1].add_artist(l1)
    axes[1].legend(handles=leg_l, loc="lower left", title="depth", fontsize=10, title_fontsize=10)
    fig.suptitle(fr"Separable floor-free law  $\mathcal{{L}}^*=A\,L^{{-b_L}}W^{{-b_W}}$  ($R^2$={r2s:.3f});"
                 fr"   $b_W/(2b_L)$={bW/(2*bL):.2f}: depth is worth {2*bL/bW:.2f}$\times$ width per parameter", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"expt_fig5_model_size_separable.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig); print(f"Saved {OUTDIR / 'expt_fig5_model_size_separable.pdf'}")

    # ---------------------------------------------------------------- combined 1x3 (manuscript)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.2))
    fig.subplots_adjust(wspace=0.32)
    ax = axes[0]
    for k, n, v in zip(ks, N, y):
        ax.scatter(n, v, s=150, color=cool_by_L[k[0]], marker=marker_by_W[k[1]], edgecolor="0.2", lw=1.3, zorder=3)
    ax.plot(g, A * g ** (-beta), "k--", lw=2.5, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_yticks([3.8, 3.9, 4.0, 4.1]); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xticks([20, 50, 100, 200, 500]); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}M"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter()); ax.tick_params(axis="x", labelsize=15)
    ax.set_xlabel(r"parameters  $N=16LW^2$", fontsize=18); ax.set_ylabel(r"min val loss  $\mathcal{L}^*$", fontsize=18)
    ax.set_title(fr"(A)  collapsed:  $\mathcal{{L}}^*\propto N^{{-{beta:.3f}}}$   ($R^2$={r2lin:.2f})", fontsize=15, loc="left")
    l1 = ax.legend(handles=leg_w, loc="upper right", title="width", fontsize=10, title_fontsize=10); ax.add_artist(l1)
    ax.legend(handles=leg_l, loc="lower left", title="depth", fontsize=10, title_fontsize=10)
    ax = axes[1]
    for k, x, v in zip(ks, L, yL):
        ax.scatter(x, v, s=140, color=cool_by_L[k[0]], marker=marker_by_W[k[1]], edgecolor="0.2", lw=1.3, zorder=3)
    ax.plot(gl, As * gl ** (-bL), "k--", lw=2.5, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([6, 12, 18, 24, 48, 60]); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"depth  $L$", fontsize=18); ax.set_ylabel(r"$\mathcal{L}^*\,W^{b_W}$  (width factor removed)", fontsize=16)
    ax.set_title(fr"(B)  separable, depth:  $b_L$={bL:.3f}", fontsize=15, loc="left")
    ax = axes[2]
    for k, x, v in zip(ks, W, yW):
        ax.scatter(x, v, s=140, color=cool_by_L[k[0]], marker=marker_by_W[k[1]], edgecolor="0.2", lw=1.3, zorder=3)
    ax.plot(gw, As * gw ** (-bW), "k--", lw=2.5, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([384, 768, 1152, 1536]); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"width  $W$", fontsize=18); ax.set_ylabel(r"$\mathcal{L}^*\,L^{b_L}$  (depth factor removed)", fontsize=16)
    ax.set_title(fr"(C)  separable, width:  $b_W$={bW:.3f}   ($R^2$={r2s:.2f})", fontsize=15, loc="left")
    for a_ in axes[1:]:
        a_.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}")); a_.yaxis.set_minor_formatter(plt.NullFormatter())
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"expt_fig5_model_size_law.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig); print(f"Saved {OUTDIR / 'expt_fig5_model_size_law.pdf'}")

    # ---------------------------------------------------------------- (3) profile
    fig, ax = plt.subplots(figsize=(8, 5.6))
    pf = np.array([(a, b, c) for a, b, c in prof if np.isfinite(c)])
    ax.plot(pf[:, 0], pf[:, 2], "-o", color="black", lw=2.6, ms=9, label=r"$R^2$ of $\mathcal{L}_\infty + c\,N^{-\alpha}$ at fixed $\mathcal{L}_\infty$")
    ax.axhline(r2lin, color="0.45", ls="--", lw=2.2, label=fr"floor-free power law ($R^2$={r2lin:.3f})")
    ax2 = ax.twinx()
    ax2.plot(pf[:, 0], pf[:, 1], "s:", color="0.5", lw=2.0, ms=8, label=r"fitted $\alpha$ at that $\mathcal{L}_\infty$")
    ax2.set_ylabel(r"$\alpha$  (open axis)", color="0.45"); ax2.grid(False)
    ax.set_xlabel(r"assumed asymptote  $\mathcal{L}_\infty$"); ax.set_ylabel(r"$R^2$")
    ax.set_ylim(0.80, 0.97)
    ax.set_title("The asymptote is not identifiable from 14M-906M at $P$=100M", fontsize=15)
    h1, l1_ = ax.get_legend_handles_labels(); h2, l2_ = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1_ + l2_, loc="lower left", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"expt_fig5_model_size_profile.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig); print(f"Saved {OUTDIR / 'expt_fig5_model_size_profile.pdf'}")

    with open(OUTDIR / "expt_fig5_model_size_law_fits.csv", "w") as fh:
        fh.write("form,param,value,note\n")
        fh.write(f"collapsed,A,{A:.6f},L*=A*N_M^-beta\ncollapsed,beta,{beta:.6f},\ncollapsed,R2_loss,{r2lin:.6f},\n")
        fh.write(f"separable,A,{As:.6f},L*=A*L^-bL*W^-bW\nseparable,b_L,{bL:.6f},LOO {looL[0]:.6f}-{looL[1]:.6f}\n")
        fh.write(f"separable,b_W,{bW:.6f},LOO {looW[0]:.6f}-{looW[1]:.6f}\nseparable,R2_log,{r2s:.6f},\n")
        fh.write(f"separable,bW_over_2bL,{bW/(2*bL):.6f},<1: depth worth more per parameter\n")
        for Linf, al, rr in prof:
            fh.write(f"profile,L_inf={Linf:.2f},{rr:.6f},alpha={al:.4f}\n")
        fh.write(f"data,n_cells,{len(y)},\n")
    print(f"Saved {OUTDIR / 'expt_fig5_model_size_law_fits.csv'}")


if __name__ == "__main__":
    main()
