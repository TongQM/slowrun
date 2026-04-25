"""
Plot per-step val loss curves (individual models + ensembles of varying sizes)
from a wandb group.

Two modes:

1) Single-run per strategy (old behavior):
    python experiments/analysis/plot.py single \
        --init-run <RUN_ID> \
        --init-shuffle-run <RUN_ID> \
        --steps-per-epoch 38 \
        --num-models 5 \
        --out val_loss.png

2) Ensemble-size sweep (one wandb group containing multiple replay runs):
    python experiments/analysis/plot.py sweep \
        --wandb-group parallel_d12_w768_df0.2_<TIMESTAMP> \
        --ensemble-sizes 1 2 4 8 16 20 \
        --steps-per-epoch 38 \
        --out ensemble_sweep.png

In sweep mode the script auto-discovers replay runs by name pattern
"<group>_<strategy>_ens<N>_replay" for each ensemble size.
"""
import argparse
import wandb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


ENTITY_PROJECT = "xjtumyd-carnegie-mellon-university/slowrun"


def fetch_ens_curve(api, run_id):
    """Return (epochs, val_loss) for one replay run (sparse, at epoch boundaries)."""
    run = api.run(f"{ENTITY_PROJECT}/{run_id}")
    hist = list(run.scan_history(keys=["ens/epoch", "ens/val_loss"]))
    epochs = [r["ens/epoch"] for r in hist if r.get("ens/val_loss") is not None]
    losses = [r["ens/val_loss"] for r in hist if r.get("ens/val_loss") is not None]
    return np.array(epochs), np.array(losses)


def fetch_per_model_curves(api, run_id, num_models):
    """Return {i: (steps, val_loss)} for per-model val eval during training."""
    run = api.run(f"{ENTITY_PROJECT}/{run_id}")
    history = list(run.scan_history())
    per_model = {i: {"step": [], "val_loss": []} for i in range(1, num_models + 1)}
    for row in history:
        for i in range(1, num_models + 1):
            vl = row.get(f"model_{i}/val_loss")
            vbpb = row.get(f"model_{i}/val_bpb")
            step = row.get(f"model_{i}/step")
            if vl is not None and vbpb is not None and step is not None:
                per_model[i]["step"].append(step)
                per_model[i]["val_loss"].append(vl)
    return per_model


def find_run_by_name(api, name_pattern):
    """Find the latest run whose display_name equals name_pattern."""
    runs = list(api.runs(ENTITY_PROJECT, filters={"display_name": name_pattern}))
    if not runs:
        return None
    # Sort by created_at desc, pick latest
    runs.sort(key=lambda r: r.created_at, reverse=True)
    return runs[0]


def sort_xy(xs, ys):
    if not xs:
        return np.array([]), np.array([])
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]


def plot_single(args):
    api = wandb.Api()
    print(f"Fetching init run {args.init_run}...")
    init_pm = fetch_per_model_curves(api, args.init_run, args.num_models)
    init_ens_e, init_ens_l = fetch_ens_curve(api, args.init_run)
    print(f"Fetching init_shuffle run {args.init_shuffle_run}...")
    isf_pm = fetch_per_model_curves(api, args.init_shuffle_run, args.num_models)
    isf_ens_e, isf_ens_l = fetch_ens_curve(api, args.init_shuffle_run)

    fig, ax = plt.subplots(figsize=(14, 7))

    def plot_exp(pm, ens_e, ens_l, pm_color, ens_color, label):
        for i in range(1, args.num_models + 1):
            xs, ys = sort_xy(pm[i]["step"], pm[i]["val_loss"])
            if len(xs) == 0:
                continue
            ax.plot(xs, ys, color=pm_color, alpha=0.3, linewidth=1.0,
                    label=f"{label} per-model val" if i == 1 else None)
        if len(ens_e) > 0:
            order = np.argsort(ens_e)
            x = ens_e[order] * args.steps_per_epoch
            y = ens_l[order]
            ax.plot(x, y, color=ens_color, linewidth=2.5, marker="s", markersize=7,
                    label=f"{label} ensemble val", zorder=10)

    plot_exp(init_pm, init_ens_e, init_ens_l, "tab:blue", "navy", "init")
    plot_exp(isf_pm, isf_ens_e, isf_ens_l, "tab:orange", "darkred", "init+shuffle")

    # Epoch boundary lines
    all_epochs = list(init_ens_e) + list(isf_ens_e)
    max_epoch = int(max(all_epochs)) if all_epochs else 0
    for e in range(1, max_epoch + 1):
        ax.axvline(x=e * args.steps_per_epoch, color="black", linestyle="--",
                   alpha=0.4, linewidth=1.0)

    ax.set_xlabel("Training step (per model)")
    ax.set_ylabel("Val loss")
    ax.set_title(f"Per-step val loss: per-model (transparent) and ensemble (bold)  |  "
                 f"{args.num_models}-model ensembles")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


def plot_sweep(args):
    api = wandb.Api()
    group = args.wandb_group

    # Color by strategy, linestyle/alpha by ensemble size
    strat_styles = {
        "init_ens":          {"color": "tab:blue", "label": "init"},
        "init_shuffle_ens":  {"color": "tab:red",  "label": "init+shuffle"},
    }
    # Map ensemble size to line alpha + width (bigger ensemble = darker)
    sizes = sorted(args.ensemble_sizes)
    size_alphas = np.linspace(0.3, 1.0, len(sizes))
    size_widths = np.linspace(1.0, 2.8, len(sizes))

    fig, ax = plt.subplots(figsize=(14, 7))

    for strategy, style in strat_styles.items():
        for ens_size, alpha, width in zip(sizes, size_alphas, size_widths):
            run_name = f"{group}_{strategy}_ens{ens_size}_replay"
            print(f"Looking for: {run_name}")
            run = find_run_by_name(api, run_name)
            if run is None:
                print(f"  MISSING: no run named {run_name}")
                continue
            epochs, losses = fetch_ens_curve(api, run.id)
            if len(epochs) == 0:
                print(f"  EMPTY: {run_name}")
                continue
            order = np.argsort(epochs)
            x = epochs[order] * args.steps_per_epoch
            y = losses[order]
            ax.plot(x, y, color=style["color"], alpha=alpha, linewidth=width,
                    marker="o", markersize=4,
                    label=f"{style['label']} ens={ens_size}")

    # Epoch boundary lines
    max_epoch = args.num_epochs
    for e in range(1, max_epoch + 1):
        ax.axvline(x=e * args.steps_per_epoch, color="gray", linestyle="--",
                   alpha=0.25, linewidth=0.7)

    ax.set_xlabel("Training step (per model)")
    ax.set_ylabel("Ensemble val loss")
    ax.set_title(f"Ensemble val loss vs training step, across ensemble sizes  |  group: {group}")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


def _latest_run(api, group, name):
    """Return most recent run matching name within group (or None)."""
    runs = list(api.runs(ENTITY_PROJECT, filters={"group": group, "display_name": name}))
    if not runs:
        return None
    runs.sort(key=lambda r: r.created_at, reverse=True)
    return runs[0]


def _all_runs(api, group, name):
    """Return all runs matching display_name within group (oldest first)."""
    runs = list(api.runs(ENTITY_PROJECT, filters={"group": group, "display_name": name}))
    runs.sort(key=lambda r: r.created_at)
    return runs


def _merge_individual_val_curves(runs):
    """For a single-model's training runs (possibly multiple due to resume),
    return (steps, val_loss) merged + sorted by step.
    Later runs' data overrides earlier at duplicate steps (resume writes more precise values)."""
    by_step = {}  # step -> val_loss
    for run in runs:
        s, v = _fetch_individual_val_curve(run)
        for si, vi in zip(s, v):
            by_step[int(si)] = float(vi)
    if not by_step:
        return np.array([]), np.array([])
    steps = np.array(sorted(by_step.keys()))
    losses = np.array([by_step[s] for s in steps])
    return steps, losses


def _fetch_individual_val_curve(run):
    """For a single-model training run, return (steps, val_loss) sorted by step.
    Looks for model_1/val_loss vs model_1/step (since single_model_idx=i virtual but log uses 'model_{i+1}')."""
    hist = list(run.scan_history())
    steps, losses = [], []
    for row in hist:
        # try all model_k/val_loss keys (each individual run only logs one model)
        for k in range(1, 30):
            vl = row.get(f"model_{k}/val_loss")
            st = row.get(f"model_{k}/step")
            if vl is not None and st is not None:
                steps.append(st)
                losses.append(vl)
                break
    if not steps:
        return np.array([]), np.array([])
    order = np.argsort(steps)
    return np.array(steps)[order], np.array(losses)[order]


def plot_combo(args):
    api = wandb.Api()
    group = args.wandb_group

    strat_colors = {
        "init_ens":          {"ind": "tab:blue",   "ens": "navy",    "label": "init"},
        "init_shuffle_ens":  {"ind": "tab:orange", "ens": "darkred", "label": "init+shuffle"},
    }
    sizes = sorted(args.ensemble_sizes)
    size_alphas = np.linspace(0.35, 1.0, len(sizes))
    size_widths = np.linspace(1.3, 2.8, len(sizes))

    fig, ax = plt.subplots(figsize=(14, 7))

    # ---- Individual model curves (transparent) ----
    # Merge val data from ALL training runs for each model (a model may have
    # multiple wandb runs due to resume — e.g. original epochs 1-10 + resumed 11-20).
    max_step = args.num_epochs * args.steps_per_epoch
    for strategy, sty in strat_colors.items():
        first_plotted = False
        for i in range(args.num_models):
            run_name = f"{group}_{strategy}_model{i}"
            runs = _all_runs(api, group, run_name)
            if not runs:
                continue
            xs, ys = _merge_individual_val_curves(runs)
            if len(xs) == 0:
                continue
            mask = xs <= max_step
            xs, ys = xs[mask], ys[mask]
            if len(xs) == 0:
                continue
            ax.plot(xs, ys, color=sty["ind"], alpha=0.15, linewidth=0.8,
                    label=f"{sty['label']} individuals" if not first_plotted else None)
            first_plotted = True

    # ---- Ensemble-size sweep (bold) ----
    for strategy, sty in strat_colors.items():
        for ens_size, alpha, width in zip(sizes, size_alphas, size_widths):
            run_name = f"{group}_{strategy}_ens{ens_size}_replay"
            run = _latest_run(api, group, run_name)
            if run is None:
                print(f"  MISSING: {run_name}")
                continue
            epochs, losses = fetch_ens_curve(api, run.id)
            if len(epochs) == 0:
                continue
            order = np.argsort(epochs)
            x = epochs[order] * args.steps_per_epoch
            y = losses[order]
            ax.plot(x, y, color=sty["ens"], alpha=alpha, linewidth=width,
                    marker="o", markersize=4,
                    label=f"{sty['label']} ens={ens_size}")

    # Epoch boundary lines — dashed, clearly visible
    for e in range(1, args.num_epochs + 1):
        ax.axvline(x=e * args.steps_per_epoch, color="black", linestyle="--",
                   alpha=0.4, linewidth=0.9)
    # Epoch labels along the top
    ylim = ax.get_ylim()
    for e in range(1, args.num_epochs + 1):
        ax.text(e * args.steps_per_epoch, ylim[1],
                f"{e}", fontsize=6, color="gray", ha="center", va="bottom")

    ax.set_xlabel("Training step (per model)")
    ax.set_ylabel("Val loss")
    ax.set_title(f"Individual models (transparent) and ensembles (bold) vs training step  |  {group}")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_single = sub.add_parser("single", help="plot per-model + single ensemble curve per strategy")
    p_single.add_argument("--init-run", required=True)
    p_single.add_argument("--init-shuffle-run", required=True)
    p_single.add_argument("--steps-per-epoch", type=int, required=True)
    p_single.add_argument("--num-models", type=int, default=5)
    p_single.add_argument("--out", default="val_loss.png")
    p_single.set_defaults(fn=plot_single)

    p_sweep = sub.add_parser("sweep", help="plot ensemble val curves across ensemble sizes from wandb group")
    p_sweep.add_argument("--wandb-group", required=True,
                         help="The wandb group all replay runs share")
    p_sweep.add_argument("--ensemble-sizes", type=int, nargs="+",
                         default=[2, 4, 8, 16, 20])
    p_sweep.add_argument("--steps-per-epoch", type=int, required=True)
    p_sweep.add_argument("--num-epochs", type=int, default=30)
    p_sweep.add_argument("--out", default="ensemble_sweep.png")
    p_sweep.set_defaults(fn=plot_sweep)

    p_combo = sub.add_parser("combo", help="individual models (transparent) + ensemble-size sweep (bold) from wandb group")
    p_combo.add_argument("--wandb-group", required=True)
    p_combo.add_argument("--ensemble-sizes", type=int, nargs="+",
                         default=[2, 4, 8, 16, 20])
    p_combo.add_argument("--num-models", type=int, default=20,
                         help="Number of individual models per strategy to plot")
    p_combo.add_argument("--steps-per-epoch", type=int, required=True)
    p_combo.add_argument("--num-epochs", type=int, default=20)
    p_combo.add_argument("--out", default="combo.png")
    p_combo.set_defaults(fn=plot_combo)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
