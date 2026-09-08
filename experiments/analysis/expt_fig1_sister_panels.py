"""Figure 1 extended to 2x3 — the overfit U-curve seen along BOTH axes.

Row 1 (A-C): fixed model (d12/w768), unique tokens P varied.
             `data_export/expt4_datasize/wd0_fixed_tokens/` — 10 df x 2 strategies,
             lambda=0, constant LR, ~1B-token budget each. Strategies are averaged
             per df (they are indistinguishable at this cell).

Row 2 (D-F): fixed data (P=100M, df=1.0), model size N=16LW^2 varied.
             12 cells from `experiments/logs/`, lambda=0, constant LR, E=1, model 0
             — the same runs `expt_fig5_model_size_law.py` fits.

Panels:
  A | val + train loss vs steps s, coloured by P
  B | val + train loss vs epoch,   coloured by P
  C | overfit onset vs P     — filled = nadir epoch (left), open = nadir steps (right)
  D | val + train loss vs epoch, DEPTH ladder (W=768), coloured by N
  E | val + train loss vs epoch, WIDTH ladder (L=12),  coloured by N
  F | overfit onset vs N     — nadir epoch (left axis), the same optimum in steps
                               (right axis), and three power-law fits: all 12 cells,
                               panel D's depth ladder, panel E's width ladder

D and E use the epoch axis only: at fixed P the step axis is the same picture
rescaled by 1/763, so a second axis would carry no information (unlike row 1,
where P varies and the epoch axis is what collapses the curves).

The headline contrast the two rows are built to show: the optimal number of
epochs is only weakly sensitive to P (14 -> 9 epochs over a 10x range,
E* ~ P^-0.16) but falls steeply in model size (30 -> 5 epochs over a 40x range,
E* ~ N^-0.43). Per decade of scaling that is a ~1.4x shrink from data vs a
~2.6x shrink from parameters.

Panel F fits each single-axis ladder as well as the joint set, so the two are
reported symmetrically: depth (W=768, 6 cells) gives N^-0.394, which at fixed
width is also the exponent in L; width (L=12, 4 cells) gives N^-0.332, i.e.
W^-0.663 since N ~ W^2.

Caveat, weaker than it first appeared: the depth axis of row 2 was measured
under the pre-fix CompleteP residual scaling (unlimited/train.py commit
c660582), which left the x0-injection and U-Net skip paths unscaled. The L=12
ladder is the one subset the correction cannot touch (12/L = 1 exactly), and it
gives the lowest exponent, 0.332. But the L=6 ladder -- same width-only
structure, and pre-fix-AFFECTED -- gives 0.453, bracketing the joint fit from
the other side. A bug signature would put the immune subset apart from the
affected ones; instead the 0.33-0.45 spread tracks which four points a ladder
happens to contain. So 0.426 is quoted as the exponent, with that spread as its
uncertainty, and the corrected depth runs are still owed.

Outputs:
  experiments/figures/10_sister_panels/expt_fig1_sister_panels.{pdf,png}
  experiments/figures/10_sister_panels/expt_fig1_sister_panels_fits.csv
  experiments/figures/10_sister_panels/optimal_stopping_table.csv
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data_export" / "expt4_datasize" / "wd0_fixed_tokens"
LOGS = REPO / "experiments" / "logs"
OUTDIR = REPO / "experiments" / "figures" / "10_sister_panels"

CORPUS_TOKENS = 100_000_000      # 100M-token FineWeb subset
BATCH_SIZE = 131072              # project default --total-batch-size
STEPS_PER_EPOCH = 763            # CORPUS_TOKENS / BATCH_SIZE at df=1.0

VAL_RE = re.compile(r"\[model \d+ val @ step (\d+)\] val_loss=([\d.]+)")
TRAIN_RE = re.compile(r"\[epoch (\d+)\] \[model \d+\] step (\d+)/(\d+) \| loss: ([\d.]+)")

# (depth, width) -> log glob. Same 12 cells as expt_fig5_model_size_law.py.
CELLS = {
    (6, 384): "fd_train_d6_w384_*_0.out",       (6, 768): "fd_train_d6_w768_*_0.out",
    (6, 1152): "fd_gridfill_d6_w1152_*_0.out",  (6, 1536): "fd_gridfill_d6_w1536_*_0.out",
    (12, 384): "fd_train_d12_w384_*_0.out",     (12, 768): "fd_train_d12_w768_*_0.out",
    (12, 1152): "fd_widthext_d12_w1152_*_0.out", (12, 1536): "fd_widthext_d12_w1536_*_0.out",
    (18, 768): "fd_gridfill_d18_w768_*_0.out",  (24, 768): "fd_gridfill_d24_w768_*_0.out",
    (48, 768): "fd_gridfill_d48_w768_*_0.out",  (60, 768): "fd_gridfill_d60_w768_*_0.out",
}


# ------------------------------------------------------------------ loaders
def load_datasize_runs():
    """runs[df] = list of per-strategy dicts (row 1)."""
    runs: dict[float, list] = {}
    for f in sorted(DATA.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        df = round(float(str(d["df"])), 2)
        entry = dict(strat=str(d["strat"]),
                     tokens=d["tokens"].astype(np.float64),
                     val_loss=d["val_loss"].astype(np.float64))
        if "train_tokens" in d.files and "train_loss" in d.files:
            entry["train_tokens"] = d["train_tokens"].astype(np.float64)
            entry["train_loss"] = d["train_loss"].astype(np.float64)
        runs.setdefault(df, []).append(entry)
    return runs


def load_cell(pattern):
    """Merge val and train curves across resume segments for one model-size cell.

    Val lines carry a global optimizer step. Train lines carry (epoch, step-in-epoch)
    with GLOBAL epoch numbering, so the global step is (epoch-1)*spe + step.
    """
    val, train = {}, {}
    for f in sorted(glob.glob(str(LOGS / pattern))):
        with open(f, errors="ignore") as fh:
            for line in fh:
                m = VAL_RE.search(line)
                if m:
                    val[int(m.group(1))] = float(m.group(2))
                    continue
                m = TRAIN_RE.search(line)
                if m:
                    ep, s, spe, v = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
                    train[(ep - 1) * spe + s] = v
    if not val:
        return None
    vs = np.array(sorted(val))
    ts = np.array(sorted(train))
    return dict(val_steps=vs, val=np.array([val[k] for k in vs]),
                train_steps=ts, train=np.array([train[k] for k in ts]))


def find_nadir(x, y):
    i = int(np.argmin(y))
    return i, float(x[i]), float(y[i])


def powerlaw_fit(x, y):
    """y = A * x^(-a). Fitted in log-log space; returns (A, a, R2)."""
    lx, ly = np.log(x), np.log(y)
    a, b = np.polyfit(lx, ly, 1)
    pred = a * lx + b
    r2 = 1.0 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
    return float(np.exp(b)), float(-a), float(r2)


def saturating_fit(x, y):
    """y = y_inf + c * x^(-a). Returns (y_inf, c, a, R2)."""
    def resid(t):
        return t[0] + t[1] * x ** (-t[2]) - y
    r = least_squares(resid, [min(y) * 0.98, 5.0, 0.3], method="trf", max_nfev=200000)
    ss = float(np.sum(r.fun ** 2))
    sstot = float(np.sum((y - y.mean()) ** 2))
    return (*[float(v) for v in r.x], 1.0 - ss / sstot)


def setup_style():
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["axes.linewidth"] = 4.0
    plt.rcParams["legend.fontsize"] = 18
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20


# ------------------------------------------------- standalone companion figure
def companion_figure(rows, order, Nvals, pal_N, lstar_P, fits):
    """Optimal stopping and attainable loss vs model size — the standalone version
    of panels (D-F) asked for beside the model-size scaling law, plus the
    side-by-side power laws in N and in P."""
    A_all, a_all, r2_all = fits["E_all"]
    A_d, a_d, r2_d = fits["E_depth"]
    A_w, a_w, r2_w = fits["E_width"]
    Linf, cN, aN, r2_LN = fits["L_vs_N"]
    LinfP, cP, aP_, r2_LP = fits["L_vs_P"]

    Ns = np.array([r["N"] for r in rows], float)
    Nd = np.array([r["N"] for r in rows if r["W"] == 768], float)
    Nw = np.array([r["N"] for r in rows if r["L"] == 12], float)
    fig, axes = plt.subplots(1, 2, figsize=(21, 8.5))
    fig.subplots_adjust(wspace=0.42)

    ax = axes[0]
    for k, r in zip(order, rows):
        ax.scatter([r["N"]], [r["ep_star"]], marker="o", s=220, color=pal_N[k],
                   edgecolor="black", linewidth=1.0, zorder=4)
    # steps and epochs differ by the constant 763 steps/epoch at P=100M, so the
    # second scale is a relabelling of the same axis, not a second data series.
    ax2 = ax.twinx()
    ax2.set_yscale("log")
    gN = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 200)
    gD = np.logspace(np.log10(Nd.min()), np.log10(Nd.max()), 200)
    gW = np.logspace(np.log10(Nw.min()), np.log10(Nw.max()), 200)
    ax.plot(gN, A_all * gN ** (-a_all), "k--", lw=3.0, zorder=2,
            label=fr"all 12 cells:  $\mathcal{{E}}^\ast\propto N^{{-{a_all:.2f}}}$  ($R^2$={r2_all:.2f})")
    ax.plot(gD, A_d * gD ** (-a_d), color="0.30", ls="-", lw=2.6, zorder=2,
            label=fr"depth ladder, $W$=768:  $N^{{-{a_d:.2f}}}$  ($R^2$={r2_d:.2f})")
    ax.plot(gW, A_w * gW ** (-a_w), color="0.50", ls=":", lw=3.5, zorder=2,
            label=fr"width ladder, $L$=12:  $N^{{-{a_w:.2f}}}$  ($R^2$={r2_w:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([2e7, 5e7, 1e8, 2e8, 5e8])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks([5, 7, 10, 15, 20, 30])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax2.set_ylim([v * STEPS_PER_EPOCH for v in ax.get_ylim()])
    _sticks = [4000, 6000, 10000, 15000, 20000]
    ax2.yaxis.set_major_locator(FixedLocator(_sticks))
    ax2.yaxis.set_major_formatter(FixedFormatter([f"{t/1000:g}k" for t in _sticks]))
    ax2.yaxis.set_minor_locator(NullLocator())
    ax2.grid(False)
    ax2.tick_params(labelsize=20, colors="0.35")
    ax.set_xlabel(r"non-embedding parameters  $N=16LW^2$", fontsize=26)
    ax.set_ylabel(r"optimal stopping epoch  $\mathcal{E}^\ast$", fontsize=24)
    ax2.set_ylabel(r"optimal stopping step  $s^\ast$", color="0.35", fontsize=24)
    ax.set_title(r"(A)  optimal stopping shortens with model size", fontsize=21, loc="left")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=15)

    ax = axes[1]
    Ps = np.array(sorted(lstar_P), float)
    LP = np.array([np.mean(lstar_P[k]) for k in sorted(lstar_P)], float)
    Ls_ = np.array([r["l_star"] for r in rows], float)
    for k, r in zip(order, rows):
        ax.scatter([r["N"] / 1e6], [r["l_star"]], marker="o", s=220, color=pal_N[k],
                   edgecolor="black", linewidth=1.0, zorder=4)
    ax.scatter(Ps / 1e6, LP, marker="^", s=210, color="0.55", edgecolor="black",
               linewidth=1.0, zorder=4)
    # each fit is drawn over its OWN measured range only; extrapolating the P fit
    # across the N range would show a plunge that the data does not support.
    gN_ = np.logspace(np.log10(Ns.min() / 1e6), np.log10(Ns.max() / 1e6), 200)
    gP_ = np.logspace(np.log10(Ps.min() / 1e6), np.log10(Ps.max() / 1e6), 200)
    ax.plot(gN_, Linf + cN * gN_ ** (-aN), "k--", lw=3.0, zorder=2,
            label=fr"vs $N$:  $\mathcal{{L}}_\infty$={Linf:.2f}, $\alpha_N$={aN:.2f}  ($R^2$={r2_LN:.2f})")
    ax.plot(gP_, LinfP + cP * gP_ ** (-aP_), color="0.45", ls=":", lw=3.5, zorder=2,
            label=fr"vs $P$:  $\mathcal{{L}}_\infty$={LinfP:.2f}, $\alpha_P$={aP_:.2f}  ($R^2$={r2_LP:.2f})")
    ax.set_xscale("log")
    ax.set_xlabel(r"resource count  (millions):  $N$  or  $P$", fontsize=26)
    ax.set_ylabel(r"min val loss  $\mathcal{L}^\ast$", fontsize=26)
    ax.set_title(r"(B)  attainable loss: parameters vs unique tokens", fontsize=21, loc="left")
    ax.grid(True, alpha=0.25, which="both")
    h = [Line2D([], [], marker="o", color="0.3", lw=0, markersize=13,
                markeredgecolor="black", label=r"vary $N$ ($P$=100M)"),
         Line2D([], [], marker="^", color="0.55", lw=0, markersize=13,
                markeredgecolor="black", label=r"vary $P$ ($L$=12,$W$=768)")]
    l1 = ax.legend(handles=h, loc="upper right", frameon=True, framealpha=0.92, fontsize=15)
    ax.add_artist(l1)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=15)

    for ext in ("pdf", "png"):
        pth = OUTDIR / f"expt_optimal_stopping.{ext}"
        fig.savefig(pth, bbox_inches="tight", dpi=300)
        print(f"saved {pth}")
    plt.close(fig)


def write_latex_table(fits, path):
    A_all, a_all, r2_all = fits["E_all"]
    A_w, a_w, r2_w = fits["E_width"]
    Linf, cN, aN, r2_LN = fits["L_vs_N"]
    LinfP, cP, aP_, r2_LP = fits["L_vs_P"]
    A_d, a_d, r2_d = fits["E_depth"]
    A_w6, a_w6, r2_w6 = fits["E_width_L6"]
    A_P, a_P, r2_P = fits["E_vs_P"]
    with open(path, "w") as fh:
        fh.write(r"""% auto-generated by experiments/analysis/expt_fig1_sister_panels.py -- do not hand-edit
\begin{table}[t]
  \centering
  \caption{Fitted exponents for the multi-epoch optimal-stopping and attainable-loss
  relations. Model-size rows use the 12 cells at $P{=}100$M ($\lambda{=}0$, constant LR,
  $E{=}1$); unique-token rows use the 10 data-size runs at $L{=}12,W{=}768$. The two
  single-axis ladders are fitted separately for symmetry. The $L{=}12$ ladder is also the
  only subset untouched by the pre-correction CompleteP residual scaling (the depth factor
  $L_{\mathrm{base}}/L$ is the identity at $L{=}12$), but the $L{=}6$ ladder, which the
  correction does affect, brackets it from above, so the subset spread of $0.33$--$0.45$
  reflects sampling over four-point ladders rather than a parameterization artifact.
  The two attainable-loss fits have very different $\mathcal{L}_\infty$, so their exponents
  describe approach rates to different asymptotes and should not be compared directly.}
  \label{tab:stopping_exponents}
  \begin{tabular}{llcccc}
    \toprule
    Quantity & Fitted form & Exponent & $\mathcal{L}_\infty$ & $R^2$ & Cells \\
    \midrule
""")
        EOL = " \\\\\n"
        fh.write(f"    Optimal stopping epoch & $\\mathcal{{E}}^\\ast \\propto N^{{-\\alpha}}$"
                 f" & ${a_all:.3f}$ & -- & ${r2_all:.3f}$ & 12" + EOL)
        fh.write(f"    \\quad depth ladder ($W{{=}}768$) & $\\mathcal{{E}}^\\ast \\propto N^{{-\\alpha}}$"
                 f" & ${a_d:.3f}$ & -- & ${r2_d:.3f}$ & 6" + EOL)
        fh.write(f"    \\quad width ladder ($L{{=}}12$) & $\\mathcal{{E}}^\\ast \\propto N^{{-\\alpha}}$"
                 f" & ${a_w:.3f}$ & -- & ${r2_w:.3f}$ & 4" + EOL)
        fh.write(f"    \\quad width ladder ($L{{=}}6$) & $\\mathcal{{E}}^\\ast \\propto N^{{-\\alpha}}$"
                 f" & ${a_w6:.3f}$ & -- & ${r2_w6:.3f}$ & 4" + EOL)
        fh.write(f"    Optimal stopping epoch & $\\mathcal{{E}}^\\ast \\propto P^{{-\\alpha}}$"
                 f" & ${a_P:.3f}$ & -- & ${r2_P:.3f}$ & 20" + EOL)
        fh.write("    \\midrule\n")
        fh.write(f"    Min val loss & $\\mathcal{{L}}_\\infty + c\\,N^{{-\\alpha_N}}$"
                 f" & ${aN:.3f}$ & ${Linf:.3f}$ & ${r2_LN:.3f}$ & 12" + EOL)
        fh.write(f"    Min val loss & $\\mathcal{{L}}_\\infty + c\\,P^{{-\\alpha_P}}$"
                 f" & ${aP_:.3f}$ & ${LinfP:.3f}$ & ${r2_LP:.3f}$ & 10" + EOL)
        fh.write(r"""    \bottomrule
  \end{tabular}
\end{table}
""")
    print(f"saved {path}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTDIR / "expt_fig1_sister_panels"))
    args = ap.parse_args()

    setup_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    runs = load_datasize_runs()
    dfs = sorted(runs.keys())
    pal_P = sns.color_palette("cool", len(dfs))

    cells = {k: load_cell(p) for k, p in CELLS.items()}
    cells = {k: v for k, v in cells.items() if v is not None}
    order = sorted(cells, key=lambda k: 16 * k[0] * k[1] ** 2)
    Nvals = {k: 16 * k[0] * k[1] ** 2 for k in order}
    pal_N = {k: c for k, c in zip(order, sns.color_palette("cool", len(order)))}

    fig = plt.figure(figsize=(24, 18.5))
    gs = fig.add_gridspec(2, 3, wspace=0.30, hspace=0.42, top=0.965, bottom=0.085)
    ax_t = fig.add_subplot(gs[0, 0])
    ax_e = fig.add_subplot(gs[0, 1])
    ax_n = fig.add_subplot(gs[0, 2]); ax_n2 = ax_n.twinx()
    ax_d = fig.add_subplot(gs[1, 0])
    ax_w = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2]); ax_f2 = ax_f.twinx()

    STRAT_MARKER = {"init_ens": "o", "init_shuffle_ens": "s"}
    STRAT_PRETTY = {"init_ens": "init", "init_shuffle_ens": "init + shuffle"}
    nadir_per_strat = {s: {"P": [], "ep": [], "tok": [], "color": []} for s in STRAT_MARKER}
    lstar_P: dict[float, list] = {}

    # ------------------------------------------------- row 1: panels A, B, C
    for color, df in zip(pal_P, dfs):
        P = CORPUS_TOKENS * df
        strat_data = runs[df]
        ref_tokens = strat_data[0]["tokens"]

        vls = []
        for entry in strat_data:
            if (len(entry["tokens"]) == len(ref_tokens)
                    and np.allclose(entry["tokens"], ref_tokens)):
                vls.append(entry["val_loss"])
            else:
                vls.append(np.interp(ref_tokens, entry["tokens"], entry["val_loss"]))
            _, tok_s, _ = find_nadir(entry["tokens"], entry["val_loss"])
            nadir_per_strat[entry["strat"]]["P"].append(P)
            nadir_per_strat[entry["strat"]]["ep"].append(tok_s / P)
            nadir_per_strat[entry["strat"]]["tok"].append(tok_s)
            nadir_per_strat[entry["strat"]]["color"].append(color)
            lstar_P.setdefault(P, []).append(float(np.min(entry["val_loss"])))
        vl_mean = np.stack(vls, axis=0).mean(axis=0)

        train_mean = train_tokens_ref = None
        if all("train_tokens" in e for e in strat_data):
            ref_train = strat_data[0]["train_tokens"]
            tls = []
            for entry in strat_data:
                if (len(entry["train_tokens"]) == len(ref_train)
                        and np.allclose(entry["train_tokens"], ref_train)):
                    tls.append(entry["train_loss"])
                else:
                    tls.append(np.interp(ref_train, entry["train_tokens"], entry["train_loss"]))
            train_mean, train_tokens_ref = np.stack(tls, axis=0).mean(axis=0), ref_train

        label = fr"$P = {int(P/1e6)}$M"
        idx_avg, _, _ = find_nadir(ref_tokens, vl_mean)

        ax_t.plot(ref_tokens / BATCH_SIZE, vl_mean, color=color, lw=3.0, label=label)
        ax_e.plot(ref_tokens / P, vl_mean, color=color, lw=3.0, label=label)
        if train_mean is not None:
            ax_t.plot(train_tokens_ref / BATCH_SIZE, train_mean, color=color,
                      lw=1.2, ls="--", alpha=0.40, zorder=1)
            ax_e.plot(train_tokens_ref / P, train_mean, color=color,
                      lw=1.2, ls="--", alpha=0.40, zorder=1)
        for ax, xv in ((ax_t, ref_tokens[idx_avg] / BATCH_SIZE), (ax_e, ref_tokens[idx_avg] / P)):
            ax.scatter([xv], [vl_mean[idx_avg]], marker="v", s=110, color=color,
                       edgecolor="black", linewidth=1.0, zorder=5)

    for strat, d in nadir_per_strat.items():
        ax_n.scatter(np.array(d["P"]), np.array(d["ep"]), marker=STRAT_MARKER[strat],
                     s=180, c=d["color"], edgecolor="black", linewidth=0.8, zorder=4,
                     label=f"{STRAT_PRETTY[strat]}  (epoch)")
        ax_n2.scatter(np.array(d["P"]), np.array(d["tok"]) / BATCH_SIZE,
                      marker=STRAT_MARKER[strat], s=140, facecolors="none",
                      edgecolors="0.35", linewidth=1.4, zorder=3)
    nad_ep = np.concatenate([nadir_per_strat[s]["ep"] for s in STRAT_MARKER])
    nad_steps = np.concatenate([nadir_per_strat[s]["tok"] for s in STRAT_MARKER]) / BATCH_SIZE
    nad_P = np.concatenate([nadir_per_strat[s]["P"] for s in STRAT_MARKER])
    A_P, a_P, r2_P = powerlaw_fit(nad_P, nad_ep)
    gP = np.logspace(np.log10(nad_P.min()), np.log10(nad_P.max()), 200)
    ax_n.plot(gP, A_P * gP ** (-a_P), "k--", lw=2.5, zorder=2,
              label=fr"fit: $\mathcal{{E}}^\ast\propto P^{{-{a_P:.2f}}}$")
    P_handles, P_labels = ax_t.get_legend_handles_labels()

    # ------------------------------------------------- row 2: panels D, E, F
    rows = []
    for k in order:
        c = cells[k]
        ep = c["val_steps"] / STEPS_PER_EPOCH
        i, ep_star, l_star = find_nadir(ep, c["val"])
        rows.append(dict(L=k[0], W=k[1], N=Nvals[k], ep_star=ep_star,
                         step_star=float(c["val_steps"][i]), l_star=l_star))

        targets = []
        if k[1] == 768:
            targets.append((ax_d, f"$L$={k[0]}"))
        if k[0] == 12:                      # d12/w768 is the base cell: it appears in both
            targets.append((ax_w, f"$W$={k[1]}"))
        for ax, lab in targets:
            ax.plot(ep, c["val"], color=pal_N[k], lw=3.0, label=lab)
            if len(c["train_steps"]):
                ax.plot(c["train_steps"] / STEPS_PER_EPOCH, c["train"], color=pal_N[k],
                        lw=1.2, ls="--", alpha=0.40, zorder=1)
            ax.scatter([ep_star], [l_star], marker="v", s=110, color=pal_N[k],
                       edgecolor="black", linewidth=1.0, zorder=5)

    Ns = np.array([r["N"] for r in rows], float)
    eps = np.array([r["ep_star"] for r in rows], float)
    lstars = np.array([r["l_star"] for r in rows], float)
    # The two single-axis ladders are exactly the ones drawn in panels (D) and (E),
    # so panel (F) fits what those panels show plus the joint fit.
    is_depth = np.array([r["W"] == 768 for r in rows])     # panel (D): vary L
    is_width = np.array([r["L"] == 12 for r in rows])      # panel (E): vary W
    is_w6 = np.array([r["L"] == 6 for r in rows])          # second width ladder, for the CSV

    A_all, a_all, r2_all = powerlaw_fit(Ns, eps)
    A_d, a_d, r2_d = powerlaw_fit(Ns[is_depth], eps[is_depth])
    A_w, a_w, r2_w = powerlaw_fit(Ns[is_width], eps[is_width])
    A_w6, a_w6, r2_w6 = powerlaw_fit(Ns[is_w6], eps[is_w6])
    Linf, cN, aN, r2_L = saturating_fit(Ns / 1e6, lstars)
    P_grid = np.array(sorted(lstar_P), float)
    LP_grid = np.array([np.mean(lstar_P[k]) for k in sorted(lstar_P)], float)
    LinfP, cP, aP_, r2_LP = saturating_fit(P_grid / 1e6, LP_grid)

    for k, r in zip(order, rows):
        ax_f.scatter([r["N"]], [r["ep_star"]], marker="o", s=200, color=pal_N[k],
                     edgecolor="black", linewidth=0.9, zorder=4)
    gN = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 200)
    gD = np.logspace(np.log10(Ns[is_depth].min()), np.log10(Ns[is_depth].max()), 200)
    gW = np.logspace(np.log10(Ns[is_width].min()), np.log10(Ns[is_width].max()), 200)
    ax_f.plot(gN, A_all * gN ** (-a_all), "k--", lw=2.5, zorder=2,
              label=fr"all 12 cells:  $N^{{-{a_all:.2f}}}$")
    ax_f.plot(gD, A_d * gD ** (-a_d), color="0.30", ls="-", lw=2.2, zorder=2,
              label=fr"depth ladder (D):  $N^{{-{a_d:.2f}}}$")
    ax_f.plot(gW, A_w * gW ** (-a_w), color="0.50", ls=":", lw=3.2, zorder=2,
              label=fr"width ladder (E):  $N^{{-{a_w:.2f}}}$")

    # ------------------------------------------------------------ cosmetics
    style_handles = [
        Line2D([], [], color="0.4", lw=3.0, ls="-", label="val loss"),
        Line2D([], [], color="0.4", lw=1.2, ls="--", alpha=0.55, label="train loss"),
        Line2D([], [], color="0.5", marker="v", linestyle="", markeredgecolor="black",
               markersize=10, label="val nadir"),
    ]

    ax_t.set_xlabel(r"steps  $s$", fontsize=28)
    ax_t.set_ylabel(r"$\mathcal{L}$", fontsize=28)
    ax_t.set_xlim(0, None); ax_t.set_ylim(0, 8)
    ax_t.set_title(r"(A)  vary $P$:  loss vs steps", fontsize=22, loc="left")
    ax_t.legend(handles=style_handles, loc="upper right", frameon=True,
                framealpha=0.92, fontsize=18)

    ax_e.set_xlabel("epoch", fontsize=28)
    ax_e.set_ylabel(r"$\mathcal{L}$", fontsize=28)
    ax_e.set_xlim(0, 50); ax_e.set_ylim(0, 8)
    ax_e.set_title(r"(B)  vary $P$:  loss vs epoch", fontsize=22, loc="left")
    ax_e.legend(handles=style_handles, loc="upper right", frameon=True,
                framealpha=0.92, fontsize=18)

    ax_n.set_xscale("log")
    ax_n.set_xlabel(r"unique tokens  $P$", fontsize=28)
    ax_n.set_ylabel(r"nadir epoch  $\mathcal{E}^\ast$  (filled)", color="black", fontsize=24)
    ax_n2.set_ylabel(r"nadir steps  $s^\ast$  (open)", color="0.35", fontsize=24)
    ax_n.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1e6)}M"))
    ax_n.set_title(r"(C)  overfit onset vs $P$", fontsize=22, loc="left")
    ax_n.set_ylim(max(0, nad_ep.min() - 1.5), nad_ep.max() + 1.5)
    ax_n2.set_ylim(0, nad_steps.max() * 1.10)
    ax_n.grid(True, alpha=0.25)
    ax_n.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=14)

    for ax, tag, ttl in ((ax_d, "(D)", r"vary $L$ at $W$=768"),
                         (ax_w, "(E)", r"vary $W$ at $L$=12")):
        ax.set_xlabel("epoch", fontsize=28)
        ax.set_ylabel(r"$\mathcal{L}$", fontsize=28)
        ax.set_xlim(0, 40); ax.set_ylim(3.4, 7.0)
        ax.set_title(f"{tag}  {ttl}   ($P$=100M)", fontsize=22, loc="left")
        h, l = ax.get_legend_handles_labels()
        ax.legend(handles=h + style_handles, labels=l + [hh.get_label() for hh in style_handles],
                  loc="upper right", frameon=True, framealpha=0.92, fontsize=15, ncol=2)

    ax_f.set_xscale("log"); ax_f.set_yscale("log")
    ax_f.set_xlabel(r"non-embedding parameters  $N = 16LW^2$", fontsize=28)
    ax_f.set_ylabel(r"nadir epoch  $\mathcal{E}^\ast$", color="black", fontsize=24)
    # right axis mirrors panel (C): the same optimum expressed in optimizer steps.
    ax_f2.set_ylabel(r"nadir steps  $s^\ast$", color="0.35", fontsize=24)
    ax_f2.set_yscale("log")
    ax_f2.set_ylim([v * STEPS_PER_EPOCH for v in ax_f.get_ylim()])
    _fsticks = [4000, 6000, 10000, 15000, 20000]
    ax_f2.yaxis.set_major_locator(FixedLocator(_fsticks))
    ax_f2.yaxis.set_major_formatter(FixedFormatter([f"{t/1000:g}k" for t in _fsticks]))
    ax_f2.yaxis.set_minor_locator(NullLocator())
    ax_f2.tick_params(colors="0.35")
    ax_f2.grid(False)
    ax_f.set_xticks([2e7, 5e7, 1e8, 2e8, 5e8])
    ax_f.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax_f.xaxis.set_minor_formatter(plt.NullFormatter())
    ax_f.set_yticks([5, 7, 10, 15, 20, 30])
    ax_f.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}"))
    ax_f.yaxis.set_minor_formatter(plt.NullFormatter())
    ax_f.set_title(r"(F)  overfit onset vs $N$", fontsize=22, loc="left")
    ax_f.grid(True, alpha=0.25, which="both")
    ax_f.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=14)

    # shared colour legends, one per row, below the figure
    fig.legend(P_handles, P_labels, loc="center", frameon=True, framealpha=0.92,
               fontsize=17, bbox_to_anchor=(0.5, 0.505), ncol=5,
               title=r"row 1: unique tokens $P$   (model fixed at $L$=12, $W$=768)",
               title_fontsize=17)
    N_handles = [Line2D([], [], color=pal_N[k], lw=4.0,
                        label=fr"$L${k[0]}/$W${k[1]}  ({Nvals[k]/1e6:.0f}M)") for k in order]
    fig.legend(handles=N_handles, loc="lower center", frameon=True, framealpha=0.92,
               fontsize=16, bbox_to_anchor=(0.5, -0.055), ncol=6,
               title=r"row 2: model size $N=16LW^2$   (data fixed at $P$=100M)",
               title_fontsize=17)

    out = Path(args.out)
    for ext in ("pdf", "png"):
        p = out.with_suffix("." + ext)
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"saved {p}")
    plt.close(fig)

    # ---------------------------------------------------------------- tables
    tbl = OUTDIR / "optimal_stopping_table.csv"
    with open(tbl, "w") as fh:
        fh.write("depth_L,width_W,N_params,N_params_M,nadir_epoch,nadir_step,min_val_loss\n")
        for r in sorted(rows, key=lambda r: r["N"]):
            fh.write(f"{r['L']},{r['W']},{int(r['N'])},{r['N']/1e6:.1f},"
                     f"{r['ep_star']:.2f},{int(r['step_star'])},{r['l_star']:.4f}\n")
    print(f"saved {tbl}")

    fits_csv = OUTDIR / "expt_fig1_sister_panels_fits.csv"
    with open(fits_csv, "w") as fh:
        fh.write("quantity,subset,n_cells,form,A_or_Linf,c,exponent,R2,note\n")
        fh.write(f"nadir_epoch,all_cells,{len(rows)},E*=A*N^-a,{A_all:.6g},,{a_all:.4f},{r2_all:.4f},"
                 "provisional: depth axis measured pre-CompleteP-residual-fix\n")
        fh.write(f"nadir_epoch,depth_ladder_W768,{int(is_depth.sum())},E*=A*N^-a,{A_d:.6g},,"
                 f"{a_d:.4f},{r2_d:.4f},vary L only; at fixed W the same exponent applies to L\n")
        fh.write(f"nadir_epoch,width_ladder_L12,{int(is_width.sum())},E*=A*N^-a,{A_w:.6g},,"
                 f"{a_w:.4f},{r2_w:.4f},vary W only; N ~ W^2 so the exponent in W is "
                 f"{2*a_w:.4f}. Also the only pre-fix-immune subset (12/L=1 at L=12)\n")
        fh.write(f"nadir_epoch,width_ladder_L6,{int(is_w6.sum())},E*=A*N^-a,{A_w6:.6g},,"
                 f"{a_w6:.4f},{r2_w6:.4f},second width ladder, pre-fix-affected; brackets the "
                 "L=12 ladder from above, so the subset spread is not a bug signature\n")
        fh.write(f"min_val_loss,all_cells,{len(rows)},L*=Linf+c*N_M^-a,{Linf:.6f},{cN:.6f},"
                 f"{aN:.4f},{r2_L:.4f},N in millions\n")
        fh.write(f"nadir_epoch,vs_P_row1,{len(nad_P)},E*=A*P^-a,{A_P:.6g},,{a_P:.4f},{r2_P:.4f},"
                 "row 1: dependence on unique tokens at fixed model size\n")
        fh.write(f"min_val_loss,vs_P_row1,{len(P_grid)},L*=Linf+c*P_M^-a,{LinfP:.6f},{cP:.6f},"
                 f"{aP_:.4f},{r2_LP:.4f},P in millions; L_inf differs from the N fit so the "
                 "two exponents are not directly comparable\n")
    print(f"saved {fits_csv}")

    fits = dict(E_all=(A_all, a_all, r2_all), E_depth=(A_d, a_d, r2_d),
                E_width=(A_w, a_w, r2_w), E_width_L6=(A_w6, a_w6, r2_w6),
                L_vs_N=(Linf, cN, aN, r2_L), L_vs_P=(LinfP, cP, aP_, r2_LP),
                E_vs_P=(A_P, a_P, r2_P))
    companion_figure(rows, order, Nvals, pal_N, lstar_P, fits)
    write_latex_table(fits, OUTDIR / "stopping_exponents_table.tex")

    print("\noptimal stopping vs model size (df=1.0, lambda=0, constant LR, E=1):")
    print(f"{'L':>4} {'W':>6} {'N (M)':>9} {'E*':>7} {'step*':>8} {'L*':>8}")
    for r in sorted(rows, key=lambda r: r["N"]):
        print(f"{r['L']:>4} {r['W']:>6} {r['N']/1e6:>9.1f} {r['ep_star']:>7.2f} "
              f"{int(r['step_star']):>8} {r['l_star']:>8.4f}")
    print(f"\nE* ~ N^-{a_all:.3f}   (all 12 cells,        R2={r2_all:.3f})")
    print(f"E* ~ N^-{a_d:.3f}   (depth ladder W=768,  R2={r2_d:.3f})  -> exponent in L = {a_d:.3f}")
    print(f"E* ~ N^-{a_w:.3f}   (width ladder L=12,   R2={r2_w:.3f})  -> exponent in W = {2*a_w:.3f}  [pre-fix immune]")
    print(f"E* ~ N^-{a_w6:.3f}   (width ladder L=6,    R2={r2_w6:.3f})  -> exponent in W = {2*a_w6:.3f}")
    print(f"L* = {Linf:.4f} + {cN:.4f} * N_M^-{aN:.4f}   (R2={r2_L:.3f})")
    print(f"E* ~ P^-{a_P:.3f}   (row 1, R2={r2_P:.3f})  -- compare: flat in P, steep in N")


if __name__ == "__main__":
    main()
