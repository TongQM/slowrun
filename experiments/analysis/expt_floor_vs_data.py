"""The model-size law at two corpus sizes, same recipe -- and what it says about
the floor, the optimal stopping time, and the interpolation regime.

Blake's question was whether the intercept L_inf(P) of the model-size law tracks
the fixed-model data law. The first pass at this used the old 20M grid, which
was confounded (lambda=0.1, warmdown). The aligned launch added a 7-cell ladder
at P=20M with the recipe of the 100M grid (lambda=0, constant LR, val on the
same 19.9M-token grid), so the two corpus sizes are now directly comparable.

What the comparison shows:
  (A) The attainable loss falls with N at both corpus sizes, but the asymptote
      L_inf is NOT identifiable at either: the R^2 of L_inf + c N^-alpha with a
      fixed L_inf is flat over a wide range of L_inf (panel C). So the honest
      laws are floor-free, L* ~ N^-beta, with beta = 0.017 at 100M and 0.013
      at 20M, and the "floor" question cannot be answered from these ranges.
  (B) The optimal stopping epoch is nearly flat in N at P=20M (10-13 epochs,
      every model stopping near 200-260M tokens seen, N^-0.09) while at P=100M
      it falls from 21 to 6 epochs (N^-0.31). The steep size dependence of the
      optimum is a property of the regime where N crosses P; once every model is
      far above the corpus size it disappears.

Data:
  P=100M  expt_cells.ALIGNED_CELLS (18 cells)
  P=20M   expt_cells.P20_CELLS (7 cells, 152.5 steps/epoch)
  data law at fixed L12/W768: data_export/expt4_datasize/wd0_fixed_tokens/

Outputs:
  experiments/figures/13_floor_vs_data/expt_floor_vs_data.{pdf,png}
  experiments/figures/13_floor_vs_data/expt_floor_vs_data_fits.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from expt_cells import ALIGNED_CELLS, P20_CELLS, P20_STEPS_PER_EPOCH  # noqa: E402
from expt_fig1_sister_panels import (  # noqa: E402
    BATCH_SIZE, CORPUS_TOKENS, STEPS_PER_EPOCH, find_nadir, load_cell, load_datasize_runs,
    powerlaw_fit, setup_style,
)

REPO = HERE.parents[1]
OUTDIR = REPO / "experiments" / "figures" / "13_floor_vs_data"


def r2(y, pred):
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def grid(cells, spe):
    ks = sorted(cells, key=lambda k: 16 * k[0] * k[1] ** 2)
    rows = []
    for k in ks:
        c = load_cell(cells[k])
        i, s_star, l_star = find_nadir(c["val_steps"], c["val"])
        rows.append(dict(L=k[0], W=k[1], N=16 * k[0] * k[1] ** 2 / 1e6, l_star=l_star,
                         ep_star=s_star / spe, tok_star=s_star * BATCH_SIZE / 1e6))
    return rows


def profile(N, y, floors):
    out = []
    for Linf in floors:
        m = y > Linf
        if m.sum() < 3:
            out.append((Linf, np.nan)); continue
        a, b = np.polyfit(np.log(N[m]), np.log(y[m] - Linf), 1)
        out.append((Linf, r2(y, Linf + np.exp(b) * N ** a)))
    return out


def main():
    setup_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    G = {100: grid(ALIGNED_CELLS, STEPS_PER_EPOCH), 20: grid(P20_CELLS, P20_STEPS_PER_EPOCH)}
    F = {}
    for P, rows in G.items():
        N = np.array([r["N"] for r in rows]); y = np.array([r["l_star"] for r in rows])
        E = np.array([r["ep_star"] for r in rows]); T = np.array([r["tok_star"] for r in rows])
        A, bN, r2N = powerlaw_fit(N, y)
        AE, bE, r2E = powerlaw_fit(N, E)
        F[P] = dict(N=N, y=y, E=E, T=T, A=A, bN=bN, r2N=r2N, AE=AE, bE=bE, r2E=r2E,
                    prof=profile(N, y, np.linspace(0.5, y.min() - 0.02, 40)))

    runs = load_datasize_runs()
    Ps = np.array(sorted(runs), float) * CORPUS_TOKENS / 1e6
    LP = np.array([np.mean([e["val_loss"].min() for e in runs[df]]) for df in sorted(runs)])
    AP, bP, r2P = powerlaw_fit(Ps, LP)

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(30, 8.8))
    fig.subplots_adjust(wspace=0.30)
    col = {100: "0.15", 20: sns.color_palette("cool", 3)[0]}
    mk = {100: "o", 20: "^"}

    ax = axes[0]
    for P in (100, 20):
        f = F[P]
        ax.scatter(f["N"], f["y"], s=170, marker=mk[P], color=col[P], edgecolor="black", linewidth=0.9,
                   zorder=4, label=fr"$P$={P}M  ({len(f['N'])} cells)")
        g = np.logspace(np.log10(f["N"].min()), np.log10(f["N"].max()), 200)
        ax.plot(g, f["A"] * g ** (-f["bN"]), "--", color=col[P], lw=2.8, zorder=2,
                label=fr"$\mathcal{{L}}^*\propto N^{{-{f['bN']:.3f}}}$  ($R^2$={f['r2N']:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([20, 50, 100, 200, 500]); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}M"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks([3.8, 4.0, 4.2, 4.4, 4.6, 4.8]); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"non-embedding parameters  $N=16LW^2$", fontsize=24)
    ax.set_ylabel(r"min val loss  $\mathcal{L}^*$", fontsize=26)
    ax.set_title(r"(A)  attainable loss vs $N$ at two corpus sizes  (same recipe)", fontsize=19, loc="left")
    ax.legend(loc="center right", frameon=True, framealpha=0.92, fontsize=14)

    ax = axes[1]
    ax2 = ax.twinx()
    for P in (100, 20):
        f = F[P]
        ax.scatter(f["N"], f["E"], s=190, marker=mk[P], color=col[P], edgecolor="black", linewidth=0.9,
                   zorder=4, label=fr"$P$={P}M:  $\mathcal{{E}}^*\propto N^{{-{f['bE']:.2f}}}$  ($R^2$={f['r2E']:.2f})")
        g = np.logspace(np.log10(f["N"].min()), np.log10(f["N"].max()), 200)
        ax.plot(g, f["AE"] * g ** (-f["bE"]), "--", color=col[P], lw=2.6, zorder=2)
        ax2.scatter(f["N"], f["T"], s=140, marker=mk[P], facecolors="none", edgecolors=col[P], linewidth=1.6, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax2.set_yscale("log")
    ax.set_xticks([20, 50, 100, 200, 500]); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}M"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks([5, 7, 10, 15, 20]); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax2.set_yticks([200, 500, 1000, 2000]); ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:g}B" if v >= 1000 else f"{v:.0f}M"))
    ax2.yaxis.set_minor_formatter(plt.NullFormatter()); ax2.grid(False)
    ax.set_xlabel(r"non-embedding parameters  $N=16LW^2$", fontsize=24)
    ax.set_ylabel(r"optimal stopping epoch  $\mathcal{E}^*$  (filled)", fontsize=22)
    ax2.set_ylabel(r"tokens seen at the optimum  (open)", color="0.35", fontsize=22)
    ax.set_title(r"(B)  the optimum is flat in $N$ at 20M, steep at 100M", fontsize=19, loc="left")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=14)

    ax = axes[2]
    for P in (100, 20):
        f = F[P]; pf = np.array([(a, b) for a, b in f["prof"] if np.isfinite(b)])
        ax.plot(pf[:, 0], pf[:, 1], "-", color=col[P], lw=2.8, label=fr"$P$={P}M: $R^2$ of $\mathcal{{L}}_\infty+c\,N^{{-\alpha}}$ at fixed $\mathcal{{L}}_\infty$")
        ax.axhline(f["r2N"], color=col[P], ls=":", lw=2.0)
    ax.set_xlabel(r"assumed asymptote  $\mathcal{L}_\infty$", fontsize=24)
    ax.set_ylabel(r"$R^2$", fontsize=26)
    ax.set_ylim(0.6, 1.0)
    ax.set_title(r"(C)  no asymptote is identifiable at either $P$  (dotted: floor-free)", fontsize=19, loc="left")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=13)

    for ext in ("pdf", "png"):
        p = OUTDIR / f"expt_floor_vs_data.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"saved {p}")
    plt.close(fig)

    with open(OUTDIR / "expt_floor_vs_data_fits.csv", "w") as fh:
        fh.write("P_M,n_cells,quantity,form,A,exponent,R2\n")
        for P in (100, 20):
            f = F[P]
            fh.write(f"{P},{len(f['N'])},min_val_loss,L*=A*N_M^-b,{f['A']:.6f},{f['bN']:.6f},{f['r2N']:.6f}\n")
            fh.write(f"{P},{len(f['N'])},nadir_epoch,E*=A*N_M^-b,{f['AE']:.6f},{f['bE']:.6f},{f['r2E']:.6f}\n")
        fh.write(f"10-100,{len(Ps)},min_val_loss_fixed_model,L*=A*P_M^-b,{AP:.6f},{bP:.6f},{r2P:.6f}\n")
        fh.write("\nP_M,L,W,N_M,min_val_loss,nadir_epoch,nadir_tokens_M\n")
        for P in (100, 20):
            for r in G[P]:
                fh.write(f"{P},{r['L']},{r['W']},{r['N']:.1f},{r['l_star']:.4f},{r['ep_star']:.1f},{r['tok_star']:.0f}\n")
    print(f"saved {OUTDIR / 'expt_floor_vs_data_fits.csv'}")

    for P in (100, 20):
        f = F[P]
        print(f"P={P}M ({len(f['N'])} cells): L* ~ N^-{f['bN']:.4f} (R2={f['r2N']:.3f});  "
              f"E* ~ N^-{f['bE']:.3f} (R2={f['r2E']:.3f}), epochs {f['E'].min():.0f}-{f['E'].max():.0f}, "
              f"tokens at optimum {f['T'].min():.0f}-{f['T'].max():.0f}M")
    print(f"data law at fixed L12/W768: L* ~ P^-{bP:.4f} (R2={r2P:.3f})")
    print(f"exponent ratio  P : N(100M) = {bP / F[100]['bN']:.1f}")


if __name__ == "__main__":
    main()
