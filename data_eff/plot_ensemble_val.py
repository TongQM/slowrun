"""
Plot per-step val loss for individual models (transparent lines) and ensemble
val loss at epoch boundaries (bold markers) on a shared step axis.

Run this with the new training flow that logs:
  model_{i}/val_loss per N steps  (model_{i}/step as x)
  ens/val_loss per epoch          (ens/epoch as x, converted to step)

Usage:
    python data_eff/plot_ensemble_val.py \
        --init-run <RUN_ID> \
        --init-shuffle-run <RUN_ID> \
        --steps-per-epoch 38 \
        --num-models 5 \
        --out val_loss.png
"""
import argparse
import wandb
import matplotlib.pyplot as plt
import numpy as np


def fetch_metrics(api, run_id, num_models):
    run = api.run(f"xjtumyd-carnegie-mellon-university/slowrun/{run_id}")
    history = list(run.scan_history())

    per_model = {i: {"step": [], "val_loss": []} for i in range(1, num_models + 1)}
    ens = {"epoch": [], "val_loss": []}

    for row in history:
        for i in range(1, num_models + 1):
            vl = row.get(f"model_{i}/val_loss")
            vbpb = row.get(f"model_{i}/val_bpb")
            step = row.get(f"model_{i}/step")
            if vl is not None and vbpb is not None and step is not None:
                per_model[i]["step"].append(step)
                per_model[i]["val_loss"].append(vl)
        if row.get("ens/val_loss") is not None and row.get("ens/epoch") is not None:
            ens["epoch"].append(row["ens/epoch"])
            ens["val_loss"].append(row["ens/val_loss"])

    return per_model, ens


def sort_xy(xs, ys):
    if not xs:
        return np.array([]), np.array([])
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init-run", required=True)
    p.add_argument("--init-shuffle-run", required=True)
    p.add_argument("--steps-per-epoch", type=int, required=True)
    p.add_argument("--num-models", type=int, default=5)
    p.add_argument("--out", default="val_loss.png")
    args = p.parse_args()

    api = wandb.Api()
    print(f"Fetching init run {args.init_run}...")
    init_pm, init_ens = fetch_metrics(api, args.init_run, args.num_models)
    print(f"Fetching init_shuffle run {args.init_shuffle_run}...")
    isf_pm, isf_ens = fetch_metrics(api, args.init_shuffle_run, args.num_models)

    fig, ax = plt.subplots(figsize=(14, 7))

    def plot_exp(per_model, ens, pm_color, ens_color, label):
        # Per-model per-step val loss (transparent lines)
        for i in range(1, args.num_models + 1):
            xs, ys = sort_xy(per_model[i]["step"], per_model[i]["val_loss"])
            if len(xs) == 0:
                continue
            ax.plot(xs, ys, color=pm_color, alpha=0.3, linewidth=1.0,
                    label=f"{label} per-model val" if i == 1 else None)
        # Ensemble val at epoch boundaries (bold)
        if ens["epoch"]:
            order = np.argsort(ens["epoch"])
            x = np.array(ens["epoch"])[order] * args.steps_per_epoch
            y = np.array(ens["val_loss"])[order]
            ax.plot(x, y, color=ens_color, linewidth=2.5, marker="s", markersize=7,
                    label=f"{label} ensemble val", zorder=10)

    plot_exp(init_pm, init_ens, "tab:blue", "navy", "init")
    plot_exp(isf_pm, isf_ens, "tab:orange", "darkred", "init+shuffle")

    # Determine epoch range
    all_steps = []
    for d in list(init_pm.values()) + list(isf_pm.values()):
        all_steps.extend(d["step"])
    all_epochs = list(init_ens["epoch"]) + list(isf_ens["epoch"])
    max_step_from_models = max(all_steps) if all_steps else 0
    max_step_from_ens = (max(all_epochs) if all_epochs else 0) * args.steps_per_epoch
    max_step = max(max_step_from_models, max_step_from_ens)
    max_epoch = (max_step + args.steps_per_epoch - 1) // args.steps_per_epoch

    # Epoch boundary lines
    for e in range(1, max_epoch + 1):
        ax.axvline(x=e * args.steps_per_epoch, color="black", linestyle="--",
                   alpha=0.4, linewidth=1.0)

    ylim = ax.get_ylim()
    for e in range(1, max_epoch + 1):
        ax.text(e * args.steps_per_epoch, ylim[1],
                f"ep{e}", fontsize=7, color="gray", ha="center", va="bottom")

    ax.set_xlabel("Training step (per model)")
    ax.set_ylabel("Val loss")
    ax.set_title(f"Per-step val loss: per-model (transparent) and ensemble (bold, at epoch boundaries)  |  "
                 f"{args.num_models}-model ensembles")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
