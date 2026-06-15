"""
Full-data grid (df=1.0, wd=0, constant LR): per-model individual val-loss
curves overlaid with the ensemble val-loss curves (E in {2,3,4,5}), one panel
per (depth, width) cell.

Data source: SLURM .out logs (the offline-wandb individual curves are
fragmented across NODE_FAIL/TIMEOUT/resume attempts; the logs are complete and
uniform). Individuals come from the training logs at 20M-token resolution
(`[model 1 val @ step S] val_loss=...`); ensembles come from the fused-replay
logs at per-epoch / 100M resolution (`[epoch K ens=S] val_loss=...`).

x-axis = cumulative tokens_seen per model (project convention):
  individual step S -> S * TOTAL_BATCH_SIZE
  ensemble  epoch K -> K * STEPS_PER_EPOCH * TOTAL_BATCH_SIZE

Usage:
  python experiments/analysis/plot_fulldata_indiv_ens.py \
      --grid-tag fulldata_20260528_150057 \
      --strategy init_shuffle_ens \
      --out experiments/figures/01_overfit_demos/fulldata_indiv_ens_shuffle.png
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

TOTAL_BATCH_SIZE = 131072
STEPS_PER_EPOCH = 763
TOKENS_PER_EPOCH = STEPS_PER_EPOCH * TOTAL_BATCH_SIZE  # ~1.0e8

# Cells ordered by non-embedding param count (16*d*w^2): smallest -> largest.
CELLS = [
    (6, 384),    # 14.2M
    (12, 384),   # 28.3M
    (6, 768),    # 56.6M
    (12, 768),   # 113.2M
]
ENS_SIZES = [2, 3, 4, 5]

STRAT_TASK0 = "init_ens"          # array idx // 5 == 0
STRAT_TASK1 = "init_shuffle_ens"  # array idx // 5 == 1

STEP_VAL_RE = re.compile(r"\[model \d+ val @ step (\d+)\] val_loss=([0-9.]+)")
# 100M / per-epoch ensemble replay line (legacy fd_replay_* logs).
ENS_EPOCH_RE = re.compile(r"\[epoch (\d+) ens=(\d+)\] val_loss=([0-9.]+)")
# 20M / per-step ensemble replay line (fd_replay20m_* logs); carries tokens directly.
ENS_STEP_RE = re.compile(r"\[step (\d+) ens=(\d+)\] val_loss=([0-9.]+) val_bpb=[0-9.]+ tokens=(\d+)")


def nonembed_params_m(d, w):
    return 16 * d * w * w / 1e6


def load_individuals(grid_tag, d, w, strat_idx):
    """Return {model_idx: (tokens[np], val_loss[np])} merged across resume logs."""
    by_idx = defaultdict(dict)  # array_idx -> {step: val}
    pat = f"experiments/logs/fd_train_d{d}_w{w}_*.out"
    fre = re.compile(rf"fd_train_d{d}_w{w}_(\d+)_(\d+)\.out")
    for f in glob.glob(pat):
        m = fre.search(os.path.basename(f))
        if not m:
            continue
        arr_idx = int(m.group(2))
        if arr_idx // 5 != strat_idx:
            continue
        with open(f, errors="ignore") as fh:
            for line in fh:
                mm = STEP_VAL_RE.search(line)
                if mm:
                    by_idx[arr_idx][int(mm.group(1))] = float(mm.group(2))
    out = {}
    for arr_idx, d_sv in by_idx.items():
        model = arr_idx % 5
        steps = np.array(sorted(d_sv))
        vals = np.array([d_sv[s] for s in steps])
        out[model] = (steps * TOTAL_BATCH_SIZE, vals)
    return out


def load_ensembles(grid_tag, d, w, strat_name):
    """Return {E: (tokens[np], val_loss[np])} from replay logs.

    Prefers the 20M / per-step replay logs (`fd_replay20m_*`, keyed on
    tokens parsed straight from the line). Falls back to the legacy 100M /
    per-epoch logs (`fd_replay_*`) for any cell that hasn't been re-run yet.
    Replay logs don't encode strategy in the filename (array task 0/1), so
    disambiguate by the `strategy=` run banner inside the file.
    """
    banner = f"strategy={strat_name}"

    def _scan(pattern, line_re, step_mode):
        by_E = defaultdict(dict)  # E -> {tokens: val}
        for f in glob.glob(pattern):
            txt = open(f, errors="ignore").read()
            if banner not in txt:
                continue
            for line in txt.splitlines():
                mm = line_re.search(line)
                if not mm:
                    continue
                if step_mode:
                    E, val, tok = int(mm.group(2)), float(mm.group(3)), int(mm.group(4))
                else:
                    ep, E, val = int(mm.group(1)), int(mm.group(2)), float(mm.group(3))
                    tok = ep * TOKENS_PER_EPOCH
                by_E[E][tok] = val
        return by_E

    by_E = _scan(f"experiments/logs/fd_replay20m_d{d}_w{w}_*.out", ENS_STEP_RE, True)
    if not by_E:  # no 20M run yet for this cell — fall back to per-epoch
        by_E = _scan(f"experiments/logs/fd_replay_d{d}_w{w}_*.out", ENS_EPOCH_RE, False)

    out = {}
    for E, d_tv in by_E.items():
        toks = np.array(sorted(d_tv))
        vals = np.array([d_tv[t] for t in toks])
        out[E] = (toks, vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-tag", default="fulldata_20260528_150057")
    ap.add_argument("--strategy", choices=[STRAT_TASK0, STRAT_TASK1],
                    default=STRAT_TASK1)
    ap.add_argument("--out", default="experiments/figures/01_overfit_demos/fulldata_indiv_ens.png")
    args = ap.parse_args()

    strat_idx = 0 if args.strategy == STRAT_TASK0 else 1

    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["axes.linewidth"] = 4.0
    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18

    ens_colors = sns.color_palette("cool", len(ENS_SIZES))

    # Load every cell's data up front so we can put all panels on a common
    # y-range (lets you compare val loss across model sizes by eye).
    cell_data = []
    vmin, vmax = np.inf, -np.inf
    for (d, w) in CELLS:
        inds = load_individuals(args.grid_tag, d, w, strat_idx)
        enss = load_ensembles(args.grid_tag, d, w, args.strategy)
        cell_data.append((d, w, inds, enss))
        for _, val in inds.values():
            if len(val):
                vmin, vmax = min(vmin, val.min()), max(vmax, val.max())
        for _, val in enss.values():
            if len(val):
                vmin, vmax = min(vmin, val.min()), max(vmax, val.max())
    pad = 0.03 * (vmax - vmin)
    ylim = (vmin - pad, vmax + pad)

    fig, axes = plt.subplots(2, 2, figsize=(16, 13), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (d, w, inds, enss) in zip(axes, cell_data):
        # Individual models: thin, light grey, one legend entry.
        for j, (model, (tok, val)) in enumerate(sorted(inds.items())):
            ax.plot(tok, val, color="0.55", lw=1.3, alpha=0.7, zorder=1,
                    label="individual models" if j == 0 else None)

        # Ensembles: thick, cool palette.
        for c, E in zip(ens_colors, ENS_SIZES):
            if E not in enss:
                continue
            tok, val = enss[E]
            ax.plot(tok, val, color=c, lw=3.0, zorder=3, label=f"E = {E}")

        ax.set_xscale("log")
        ax.set_ylim(*ylim)   # shared across all cells for cross-model comparison
        ax.set_title(f"d{d} / w{w}  ({nonembed_params_m(d, w):.0f}M)",
                     fontsize=20)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    for ax in axes[2:]:
        ax.set_xlabel("tokens seen (cumulative, per model)", fontsize=22)
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("val loss", fontsize=24)

    strat_pretty = ("shared shuffle (init)" if args.strategy == STRAT_TASK0
                    else "independent shuffle (init+shuffle)")
    fig.suptitle(
        f"Individual vs ensemble val loss  -  df=1.0, wd=0, constant LR  -  {strat_pretty}",
        fontsize=22, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
