"""
Compare half-width individuals + ensembles against original-width individuals.

Question: does ensembling N half-width models match a single original-width model?
Param count: original (d12,w768) ≈ 4× half-width (d12,w384), so ens=4 is the
matched-FLOPs comparison.
"""
import argparse
import wandb
import matplotlib.pyplot as plt
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot import (ENTITY_PROJECT, fetch_ens_curve,
                  _all_runs, _merge_individual_val_curves, _latest_run)


def gather_individuals(api, group, strategy, num_models, max_step):
    curves = []
    for i in range(num_models):
        runs = _all_runs(api, group, f"{group}_{strategy}_model{i}")
        if not runs:
            continue
        xs, ys = _merge_individual_val_curves(runs)
        if len(xs) == 0:
            continue
        m = xs <= max_step
        curves.append((xs[m], ys[m]))
    return curves


def gather_ensembles(api, group, strategy, ens_sizes, steps_per_epoch):
    out = {}
    for sz in ens_sizes:
        run = _latest_run(api, group, f"{group}_{strategy}_ens{sz}_replay")
        if run is None:
            continue
        epochs, losses = fetch_ens_curve(api, run.id)
        if len(epochs) == 0:
            continue
        order = np.argsort(epochs)
        out[sz] = (epochs[order] * steps_per_epoch, losses[order])
    return out


def plot_compare(args):
    api = wandb.Api()
    max_step = args.num_epochs * args.steps_per_epoch

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
    strategies = [("init_ens", "init"), ("init_shuffle_ens", "init+shuffle")]

    for ax, (strategy, label) in zip(axes, strategies):
        # Original-width individuals
        orig_curves = gather_individuals(api, args.original_group, strategy,
                                         args.original_num_models, max_step)
        for j, (xs, ys) in enumerate(orig_curves):
            ax.plot(xs, ys, color="tab:blue", alpha=0.25, linewidth=1.0,
                    label=f"original w=768 individuals (N={len(orig_curves)})" if j == 0 else None)

        # Half-width individuals
        hw_curves = gather_individuals(api, args.halfwidth_group, strategy,
                                       args.halfwidth_num_models, max_step)
        for j, (xs, ys) in enumerate(hw_curves):
            ax.plot(xs, ys, color="tab:orange", alpha=0.4, linewidth=1.0,
                    label=f"half-width w=384 individuals (N={len(hw_curves)})" if j == 0 else None)

        # Half-width ensembles
        hw_ens = gather_ensembles(api, args.halfwidth_group, strategy,
                                  args.ensemble_sizes, args.steps_per_epoch)
        cmap = plt.cm.Reds(np.linspace(0.45, 0.95, len(args.ensemble_sizes)))
        for color, sz in zip(cmap, args.ensemble_sizes):
            if sz not in hw_ens:
                continue
            xs, ys = hw_ens[sz]
            mark = "*" if sz == 4 else "o"
            mksize = 11 if sz == 4 else 6
            lw = 2.8 if sz == 4 else 1.8
            ax.plot(xs, ys, color=color, linewidth=lw,
                    marker=mark, markersize=mksize,
                    label=f"half-width ens={sz}" + ("  (≈ matched params)" if sz == 4 else ""))

        # Epoch lines
        for e in range(1, args.num_epochs + 1):
            ax.axvline(x=e * args.steps_per_epoch, color="black",
                       linestyle="--", alpha=0.25, linewidth=0.6)

        ax.set_xlabel("Training step (per model)")
        ax.set_title(f"strategy: {label}")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_ylabel("Val loss")
    fig.suptitle("Does ensembling half-width models match a single original-width model?",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--halfwidth-group", required=True)
    p.add_argument("--original-group", required=True)
    p.add_argument("--halfwidth-num-models", type=int, default=5)
    p.add_argument("--original-num-models", type=int, default=20)
    p.add_argument("--ensemble-sizes", type=int, nargs="+", default=[2, 3, 4, 5])
    p.add_argument("--steps-per-epoch", type=int, default=38)
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--out", default="experiments/analysis/compare_width_vs_ensemble.png")
    args = p.parse_args()
    plot_compare(args)
