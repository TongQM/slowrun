"""Expt Fig 2, panel C — model capacity at df=1.0 (full 100M tokens).

Minimum validation loss L* of a SINGLE model (E=1, no ensembling) against
model size along the depth axis at fixed width 768, under two regularization
settings:

  (a) wd = 0, constant LR (no warmdown)  — the unregularized baseline used by
      the rest of the df=1.0 figures. Depths {6, 12, 18, 24, 48, 60}.
  (b) tuned weight decay + LR cooldown   — wd swept per cell, best taken.
      Depths {12, 24, 48}; wd_opt = {0.1, 0.2, 0.3} respectively.

The contrast is the point: at wd = 0 the capacity curve saturates (and turns
up), whereas with per-size-tuned wd + cooldown it keeps descending, so "bigger
does not help in the data-limited regime" is a statement about the
regularization, not about capacity.

Strategy note: every point here is a single member-0 run. At i = 0 the data
seed is 42 under BOTH init and init_shuffle (train.py: `data_seed = seed if
ensemble_type == "init_shuffle" else 42`, with `seeds = [42 + i]`), so these
E=1 points are identical under either ensembling arm. Arm labels only start to
matter from member 1 onward, i.e. for the E>=2 results, not for this figure.

Parsed from SLURM .out logs (offline wandb is fragmented across resume
segments, per project convention).

Outputs:
  experiments/figures/09_val_vs_params/expt_fig2_capacity_vs_size.{pdf,png}
  experiments/figures/09_val_vs_params/expt_fig2_capacity_vs_size.csv
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

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "experiments" / "logs"
OUTDIR = REPO / "experiments" / "figures" / "09_val_vs_params"

WIDTH = 768
IND_LINE = re.compile(r"\[model \d+ val @ step (\d+)\] val_loss=([\d.]+)")

# (a) wd=0, constant LR. d6/d12 come from the fulldata grid (member 0 of the
#     5-model pool); the rest are the single-model grid-fill / depth-deep runs.
WD0_CELLS = {
    6:  "fd_train_d6_w768_*_0.out",
    12: "fd_train_d12_w768_*_0.out",
    18: "fd_gridfill_d18_w768_*_0.out",
    24: "fd_gridfill_d24_w768_*_0.out",
    48: "fd_gridfill_d48_w768_*_0.out",
    60: "fd_gridfill_d60_w768_*_0.out",
}
# (b) tuned wd + cooldown. Every wd probed per depth; we take the best.
WDTUNED_GLOB = "wdsize_d{L}_w768_wd*_*.out"
# the base cell's sweep uses a different job-name prefix
WDBASE_GLOB = "wd*_train_d12_w768_*.out"


def min_val(pattern: str) -> float | None:
    """Min val loss over all evals in every log matching `pattern`.

    Logs are merged by step, so resume segments (and duplicated copies of the
    same log) collapse to one curve.
    """
    pts: dict[int, float] = {}
    for f in glob.glob(str(LOGS / pattern)):
        with open(f, errors="ignore") as fh:
            for line in fh:
                m = IND_LINE.search(line)
                if m:
                    pts[int(m.group(1))] = float(m.group(2))
    return min(pts.values()) if pts else None


def tuned_wd_series() -> dict[int, tuple[float, float]]:
    """depth -> (best L*, wd achieving it) for the cooldown+wd runs."""
    out: dict[int, tuple[float, float]] = {}
    # base cell d12 (job-name prefix wd<VAL>_train_)
    per_wd: dict[float, float] = {}
    for f in glob.glob(str(LOGS / WDBASE_GLOB)):
        m = re.search(r"wd([\d.]+)_train_", Path(f).name)
        if not m:
            continue
        L = min_val(Path(f).name)
        if L is not None:
            wd = float(m.group(1))
            per_wd[wd] = min(per_wd.get(wd, np.inf), L)
    # wd=0 here is the unregularized run, not part of the tuned series
    per_wd.pop(0.0, None)
    if per_wd:
        wd = min(per_wd, key=per_wd.get)
        out[12] = (per_wd[wd], wd)
    # wdsize probe cells
    for depth in (24, 48):
        per_wd = {}
        for f in glob.glob(str(LOGS / WDTUNED_GLOB.format(L=depth))):
            m = re.search(rf"wdsize_d{depth}_w768_wd([\d.]+)_", Path(f).name)
            if not m:
                continue
            L = min_val(Path(f).name)
            if L is not None:
                wd = float(m.group(1))
                per_wd[wd] = min(per_wd.get(wd, np.inf), L)
        if per_wd:
            wd = min(per_wd, key=per_wd.get)
            out[depth] = (per_wd[wd], wd)
    return out


def params_m(depth: int, width: int = WIDTH) -> float:
    """Non-embedding parameter count in millions (the paper's 16 L N^2 proxy)."""
    return 16 * depth * width**2 / 1e6


def main() -> None:
    wd0 = {d: v for d, v in ((d, min_val(p)) for d, p in WD0_CELLS.items()) if v is not None}
    tuned = tuned_wd_series()

    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    sns.set_palette("cool", 2)
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["axes.linewidth"] = 4.0
    plt.rcParams["legend.fontsize"] = 18
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20

    fig, ax = plt.subplots(figsize=(9, 6.5))

    ds = sorted(wd0)
    xs = [params_m(d) for d in ds]
    ys = [wd0[d] for d in ds]
    ax.plot(xs, ys, "o-", lw=3.0, ms=11, label=r"$\lambda = 0$, constant LR")
    for d, x, y in zip(ds, xs, ys):
        ax.annotate(f"$L$={d}", (x, y), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=13)

    dt = sorted(tuned)
    xt = [params_m(d) for d in dt]
    yt = [tuned[d][0] for d in dt]
    ax.plot(xt, yt, "s-", lw=3.0, ms=11,
            label=r"tuned $\lambda$ + cooldown")
    for d, x, y in zip(dt, xt, yt):
        ax.annotate(f"$L$={d},  $\\lambda$={tuned[d][1]:g}", (x, y),
                    textcoords="offset points", xytext=(9, 9),
                    ha="left", fontsize=13)

    # Headline: regularization at fixed size beats 10x capacity without it.
    if 12 in wd0 and 12 in tuned:
        x12 = params_m(12)
        ax.annotate("", xy=(x12, tuned[12][0]), xytext=(x12, wd0[12]),
                    arrowprops=dict(arrowstyle="<->", lw=2.5, color="0.25"))
        ax.annotate(f"$\\Delta$={wd0[12] - tuned[12][0]:.2f}\nat fixed $L$=12",
                    (x12, 0.5 * (wd0[12] + tuned[12][0])),
                    textcoords="offset points", xytext=(-13, 0),
                    ha="right", va="center", fontsize=14, color="0.25")
    if ds and len(ds) > 1:
        span = wd0[ds[0]] - wd0[ds[-1]]
        ax.annotate(f"$\\Delta$={span:.2f} across {params_m(ds[-1]) / params_m(ds[0]):.0f}$\\times$ params",
                    (params_m(ds[-1]), wd0[ds[-1]]), textcoords="offset points",
                    xytext=(-8, -30), ha="right", fontsize=14, color="0.25")

    ax.set_xscale("log")
    ax.set_xticks([60, 100, 200, 400, 600])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(r"non-embedding parameters  $16LN^2$  (millions)")
    ax.set_ylabel(r"min val loss  $\mathcal{L}^*$   ($E=1$)")
    ax.set_title(f"Model capacity at $df=1.0$ (100M tokens), width $N={WIDTH}$",
                 fontsize=17)
    ax.legend(frameon=True, loc="lower left")
    fig.tight_layout()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUTDIR / f"expt_fig2_capacity_vs_size.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"Saved {p}")

    csv = OUTDIR / "expt_fig2_capacity_vs_size.csv"
    with open(csv, "w") as fh:
        fh.write("series,depth,width,params_m,weight_decay,lr_schedule,min_val_loss\n")
        for d in ds:
            fh.write(f"wd0,{d},{WIDTH},{params_m(d):.2f},0,constant,{wd0[d]:.4f}\n")
        for d in dt:
            L, wd = tuned[d]
            fh.write(f"tuned_wd,{d},{WIDTH},{params_m(d):.2f},{wd:g},cooldown,{L:.4f}\n")
    print(f"Saved {csv}")

    print("\nwd=0, constant LR:")
    for d in ds:
        print(f"  d{d:<3} N={params_m(d):8.1f}M  L*={wd0[d]:.4f}")
    print("tuned wd + cooldown:")
    for d in dt:
        print(f"  d{d:<3} N={params_m(d):8.1f}M  L*={tuned[d][0]:.4f}  (wd={tuned[d][1]:g})")


if __name__ == "__main__":
    main()
