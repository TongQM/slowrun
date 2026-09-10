"""Ensembling versus model size on the same axis — Blake's 4x-ensemble curve.

Question (meeting + red note on Figure 6): take some of the model-size cells,
ensemble them 4x, and plot the ensemble's minimum val loss against
"parameters times ensemble size". Is that a *better* scaling law than the
single-model one, or the *same* law shifted down?

Data (all at P=100M, lambda=0, constant LR):
  E=1   the 12 model-size cells, model 0, from experiments/logs/  (identical to
        Figure 6 / the collapsed fit in expt_fig5_model_size_law.py)
  E>1   data_export/100M_data/val_loss_ensembles.csv -- post-hoc replays of the
        first-E checkpoints at the four cells that were trained with 5
        individuals per strategy: d6/w384 (14M), d12/w384 (28M), d6/w768 (57M),
        d12/w768 (113M). Replays run to step 23560 (~31 epochs); every minimum
        used here lands well inside that.

Effective size:  N_eff = E * N.  At fixed training length this is also the
training-compute ratio, so "4x ensemble of N" and "single model of 4N" sit at
the same x.

Two one-parameter hypotheses for the E=4 points, both anchored to the E=1
collapsed fit  L = L_inf + c N^-alpha:
  H1  lower floor, same excess term:      L = (L_inf - Delta) + c N_eff^-alpha
  H2  same floor, smaller excess term:    L =  L_inf + (rho c) N_eff^-alpha
plus a two-parameter shared-floor fit (c_E, alpha_E) so the exponent itself can
be compared.

Outputs:
  experiments/figures/11_ensemble_vs_size/expt_fig6_ensemble_vs_size.{pdf,png}
  experiments/figures/11_ensemble_vs_size/expt_fig6_ensemble_vs_size_fits.csv
  experiments/figures/11_ensemble_vs_size/ensemble_vs_size_table.csv
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from expt_fig1_sister_panels import (  # noqa: E402
    CELLS, LOGS, STEPS_PER_EPOCH, find_nadir, load_cell, powerlaw_fit, setup_style,
)
from expt_cells import ENSEMBLE_REPLAY_LOGS  # noqa: E402
import glob as _glob
import re as _re
REPLAY_RE = _re.compile(r"\[step (\d+) ens=(\d+)\] val_loss=([\d.]+)")

REPO = HERE.parents[1]
ENS_CSV = REPO / "data_export" / "100M_data" / "val_loss_ensembles.csv"
OUTDIR = REPO / "experiments" / "figures" / "11_ensemble_vs_size"
STRAT = "init_shuffle_ens"          # the strategy the paper now foregrounds
E_SHOW = (2, 3, 4, 5)


def load_ensembles():
    """{(L, W, E): (steps, val)} for the primary strategy."""
    raw = defaultdict(list)
    with open(ENS_CSV) as fh:
        for r in csv.DictReader(fh):
            if r["strategy"] != STRAT:
                continue
            raw[(int(r["depth"]), int(r["width"]), int(r["E"]))].append(
                (int(r["step"]), float(r["val_loss"])))
    # post-fix ensemble cells: parsed from the fused-replay logs (only finished replays)
    for (L, W), pat in ENSEMBLE_REPLAY_LOGS.items():
        for f in _glob.glob(str(LOGS / pat)):
            txt = open(f, errors="ignore").read()
            if "Done:" not in txt:
                continue
            for m in REPLAY_RE.finditer(txt):
                raw[(L, W, int(m.group(2)))].append((int(m.group(1)), float(m.group(3))))
    out = {}
    for k, pts in raw.items():
        pts.sort()
        out[k] = (np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float))
    return out


def main():
    setup_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ---- E=1: the 12 cells, exactly as Figure 6
    single = {}
    for k, pat in CELLS.items():
        c = load_cell(pat)
        if c is None:
            continue
        i, s_star, l_star = find_nadir(c["val_steps"], c["val"])
        single[k] = dict(N=16 * k[0] * k[1] ** 2, l_star=l_star, s_star=s_star)
    N1 = np.array([single[k]["N"] for k in single], float)
    L1 = np.array([single[k]["l_star"] for k in single], float)
    # floor-free E=1 law (the asymptote is not identifiable; see expt_fig5's profile panel)
    c1, a1, r2_1 = powerlaw_fit(N1 / 1e6, L1); Linf = 0.0
    fit1 = lambda n: c1 * (n / 1e6) ** (-a1)  # noqa: E731

    # ---- E>1: replayed ensembles at the four replicated cells
    ens = load_ensembles()
    cells_E = sorted({(k[0], k[1]) for k in ens}, key=lambda k: 16 * k[0] * k[1] ** 2)
    rows = []
    for (L, W) in cells_E:
        N = 16 * L * W ** 2
        for E in E_SHOW:
            if (L, W, E) not in ens:
                continue
            st, v = ens[(L, W, E)]
            i, s_star, l_star = find_nadir(st, v)
            rows.append(dict(L=L, W=W, N=N, E=E, N_eff=E * N, l_star=l_star,
                             s_star=s_star, ep_star=s_star / STEPS_PER_EPOCH,
                             single_l_star=single[(L, W)]["l_star"],
                             single_s_star=single[(L, W)]["s_star"],
                             fit1_at_Neff=fit1(E * N), gap=fit1(E * N) - l_star))

    # ---- hypotheses on the E=4 points
    r4 = [r for r in rows if r["E"] == 4]
    Ne4 = np.array([r["N_eff"] for r in r4], float)
    L4 = np.array([r["l_star"] for r in r4], float)
    # H1: vertical shift Delta (least squares -> mean residual)
    Delta = float(np.mean(fit1(Ne4) - L4))
    sse_H1 = float(np.sum((fit1(Ne4) - Delta - L4) ** 2))
    # H2: prefactor ratio rho on the excess term
    exc_fit = c1 * (Ne4 / 1e6) ** (-a1)
    rho = float(np.sum(exc_fit * (L4 - Linf)) / np.sum(exc_fit ** 2))
    sse_H2 = float(np.sum((Linf + rho * exc_fit - L4) ** 2))
    # shared-floor two-parameter fit
    lx, ly = np.log(Ne4 / 1e6), np.log(L4 - Linf)
    slope, icpt = np.polyfit(lx, ly, 1)
    a4, c4 = -float(slope), float(np.exp(icpt))
    sse_2p = float(np.sum((Linf + c4 * (Ne4 / 1e6) ** (-a4) - L4) ** 2))

    # matched-N comparisons: 4x ensemble of N vs single model at (approximately) 4N
    matched = []
    for r in r4:
        cands = [(k, v) for k, v in single.items() if abs(v["N"] - r["N_eff"]) / r["N_eff"] < 0.02]
        for k, v in cands:
            matched.append(dict(base=f"L{r['L']}/W{r['W']}", N_eff=r["N_eff"], ens4=r["l_star"],
                                single=f"L{k[0]}/W{k[1]}", single_l=v["l_star"],
                                advantage=v["l_star"] - r["l_star"]))

    # ------------------------------------------------------------- figure (single panel)
    # Each replicated cell draws its own ensemble curve E=1..5 at effective size E*N,
    # starting on the single-model law -- the construction of Figure 2B, one curve per
    # base cell -- so the reader sees five short "ensembling branches" leaving the law.
    fig, ax = plt.subplots(figsize=(11, 7.6))
    ax.scatter(N1, L1, s=150, color="0.25", edgecolor="black", linewidth=0.9, zorder=4,
               label=f"single model ($E$=1), {len(N1)} cells")
    gN = np.logspace(np.log10(N1.min()), np.log10(max(N1.max(), Ne4.max() * 1.25)), 300)
    ax.plot(gN, fit1(gN), "k--", lw=2.6, zorder=2, label=fr"single-model law  $\mathcal{{L}}^*\propto N^{{-{a1:.3f}}}$")
    ax.plot(gN, fit1(gN) - Delta, color="0.55", ls=":", lw=2.6, zorder=2,
            label=fr"law shifted down by {Delta:.2f} (mean gap at $E$=4, the largest $E$ all cells share)")
    pal_cells = sns.color_palette("cool", len(cells_E))
    for col, (L, W) in zip(pal_cells, cells_E):
        rr = sorted([r for r in rows if (r["L"], r["W"]) == (L, W)], key=lambda r: r["E"])
        xs = [single[(L, W)]["N"]] + [r["N_eff"] for r in rr]
        ys = [single[(L, W)]["l_star"]] + [r["l_star"] for r in rr]
        ax.plot(xs, ys, "-", color=col, lw=2.4, zorder=3)
        ax.scatter(xs[1:], ys[1:], s=75, color=col, edgecolor="black", linewidth=0.6, zorder=5)
        ax.annotate(fr"$L${L}/$W${W}", xy=(xs[0], ys[0]), xytext=(-6, 6), textcoords="offset points",
                    fontsize=11, color=col, ha="right")
    ax.scatter([], [], s=60, color="0.5", edgecolor="black", linewidth=0.5, label="ensemble of that cell at $E\cdot N$, $E$=2,3,4(,5)")
    ax.set_xscale("log")
    ax.set_xticks([2e7, 5e7, 1e8, 2e8, 5e8, 1e9]); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M" if x < 1e9 else f"{x/1e9:g}B"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"effective size  $E\cdot N$   (parameters $\times$ ensemble size)", fontsize=20)
    ax.set_ylabel(r"min val loss  $\mathcal{L}^*$", fontsize=22)
    ax.set_title("Ensembling against model size: each branch is one cell's ensemble series", fontsize=16, loc="left")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=12)
    for ext in ("pdf", "png"):
        p = OUTDIR / f"expt_fig6_ensemble_vs_size.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"saved {p}")
    plt.close(fig)

    # ------------------------------------------------------------- tables
    with open(OUTDIR / "ensemble_vs_size_table.csv", "w") as fh:
        fh.write("L,W,N,E,N_eff,ens_min_val,ens_nadir_epoch,single_min_val,single_nadir_epoch,"
                 "E1_law_at_Neff,gap_below_law\n")
        for r in rows:
            fh.write(f"{r['L']},{r['W']},{r['N']},{r['E']},{r['N_eff']},{r['l_star']:.4f},"
                     f"{r['ep_star']:.2f},{r['single_l_star']:.4f},"
                     f"{r['single_s_star']/STEPS_PER_EPOCH:.2f},{r['fit1_at_Neff']:.4f},{r['gap']:.4f}\n")
    with open(OUTDIR / "expt_fig6_ensemble_vs_size_fits.csv", "w") as fh:
        fh.write("model,param,value,note\n")
        fh.write(f"E1_collapsed,form,floor-free L*=A*N_M^-beta,{len(N1)} cells\nE1_collapsed,A,{c1:.6f},\n"
                 f"E1_collapsed,beta,{a1:.6f},\nE1_collapsed,R2,{r2_1:.6f},\n")
        fh.write(f"E4_H1_shift,Delta,{Delta:.6f},lower floor same excess; SSE={sse_H1:.6f}\n")
        fh.write(f"E4_H2_prefactor,rho,{rho:.6f},same floor smaller excess; SSE={sse_H2:.6f}\n")
        fh.write(f"E4_shared_floor,c,{c4:.6f},two-parameter; SSE={sse_2p:.6f}\n")
        fh.write(f"E4_shared_floor,alpha,{a4:.6f},compare alpha={a1:.4f} for E=1\n")
        for m in matched:
            fh.write(f"matched_N,{m['base']}_x4_vs_{m['single']},{m['advantage']:.6f},"
                     f"ens4={m['ens4']:.4f} single={m['single_l']:.4f} at N_eff={m['N_eff']/1e6:.0f}M\n")
    print(f"saved {OUTDIR / 'expt_fig6_ensemble_vs_size_fits.csv'}")

    print(f"\nE=1 floor-free law: L* = {c1:.4f} N_M^-{a1:.4f} (R2={r2_1:.3f}), {len(N1)} cells")
    print(f"\n{'cell':>10} {'N':>6} {'E':>2} {'E*N':>6} {'L*_E':>8} {'law(EN)':>8} {'gap':>7} {'ep*':>5}")
    for r in rows:
        print(f"L{r['L']}/W{r['W']:<5} {r['N']/1e6:>5.0f}M {r['E']:>2} {r['N_eff']/1e6:>5.0f}M "
              f"{r['l_star']:>8.4f} {r['fit1_at_Neff']:>8.4f} {r['gap']:>7.4f} {r['ep_star']:>5.1f}")
    print(f"\nE=4 vs the E=1 law (4 points):")
    print(f"  H1 shift down          Delta={Delta:.4f}   SSE={sse_H1:.5f}")
    print(f"  H2 prefactor on excess rho={rho:.4f}     SSE={sse_H2:.5f}")
    print(f"  shared-floor 2-param   alpha_E4={a4:.3f} (E=1: {a1:.3f})  c={c4:.3f} (E=1: {c1:.3f})  SSE={sse_2p:.5f}")
    print("\nmatched N: 4x ensemble of N vs one model of 4N")
    for m in matched:
        print(f"  {m['base']} x4 ({m['N_eff']/1e6:.0f}M): {m['ens4']:.4f}   vs   {m['single']}: "
              f"{m['single_l']:.4f}   ensemble better by {m['advantage']:.4f}")


if __name__ == "__main__":
    main()
