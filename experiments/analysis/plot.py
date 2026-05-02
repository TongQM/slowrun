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


def fetch_ens_tokens_curve(api, run_id):
    """Return (tokens_seen, val_loss) for one replay/sync run, using cumulative tokens
    (per-model tokens at each synchronized epoch boundary)."""
    run = api.run(f"{ENTITY_PROJECT}/{run_id}")
    hist = list(run.scan_history(keys=["ens/tokens_seen", "ens/val_loss"]))
    toks = [r["ens/tokens_seen"] for r in hist if r.get("ens/val_loss") is not None and r.get("ens/tokens_seen") is not None]
    losses = [r["ens/val_loss"] for r in hist if r.get("ens/val_loss") is not None and r.get("ens/tokens_seen") is not None]
    return np.array(toks), np.array(losses)


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


def _fetch_individual_val_tokens_curve(run):
    """Same as _fetch_individual_val_curve but x-axis = cumulative tokens_seen."""
    hist = list(run.scan_history())
    toks, losses = [], []
    for row in hist:
        for k in range(1, 30):
            vl = row.get(f"model_{k}/val_loss")
            ts = row.get(f"model_{k}/tokens_seen")
            if vl is not None and ts is not None:
                toks.append(ts)
                losses.append(vl)
                break
    if not toks:
        return np.array([]), np.array([])
    order = np.argsort(toks)
    return np.array(toks)[order], np.array(losses)[order]


def _merge_individual_val_tokens_curves(runs):
    """Merge tokens-seen-vs-val_loss across resume-runs of one model."""
    by_tok = {}
    for run in runs:
        t, v = _fetch_individual_val_tokens_curve(run)
        for ti, vi in zip(t, v):
            by_tok[int(ti)] = float(vi)
    if not by_tok:
        return np.array([]), np.array([])
    toks = np.array(sorted(by_tok.keys()))
    losses = np.array([by_tok[t] for t in toks])
    return toks, losses


def plot_grid(args):
    """3x3 figure: rows = n_layer, cols = n_embd. Each cell overlays individual model
    val curves (transparent) and ensemble val curves (size 2/4/5) for both strategies.
    X-axis = cumulative tokens_seen."""
    api = wandb.Api()
    depths = [int(d) for d in args.depths]
    widths = [int(w) for w in args.widths]
    sizes = sorted(int(s) for s in args.ensemble_sizes)

    strat_colors = {
        "init_ens":         {"ind": "tab:blue",   "ens": "navy",    "label": "init"},
        "init_shuffle_ens": {"ind": "tab:orange", "ens": "darkred", "label": "init+shuffle"},
    }
    size_alphas = np.linspace(0.45, 1.0, len(sizes))
    size_widths = np.linspace(1.4, 2.6, len(sizes))

    fig, axes = plt.subplots(len(depths), len(widths),
                             figsize=(5.5 * len(widths), 4.2 * len(depths)),
                             sharex=False, sharey=False)
    if len(depths) == 1 and len(widths) == 1:
        axes = np.array([[axes]])
    elif len(depths) == 1 or len(widths) == 1:
        axes = np.array(axes).reshape(len(depths), len(widths))

    legend_handles = []
    legend_labels = []

    for i, L in enumerate(depths):
        for j, W in enumerate(widths):
            ax = axes[i, j]
            group = f"grid_{args.grid_tag}_d{L}_w{W}_df{args.data_fraction}"
            print(f"\n[d{L}_w{W}] group: {group}")

            cell_has_data = False
            # Individual model curves (transparent)
            for strategy, sty in strat_colors.items():
                first_for_strat = True
                for m in range(args.num_models):
                    run_name = f"{group}_{strategy}_model{m}"
                    runs = _all_runs(api, group, run_name)
                    if not runs:
                        continue
                    xs, ys = _merge_individual_val_tokens_curves(runs)
                    if len(xs) == 0:
                        continue
                    line, = ax.plot(xs, ys, color=sty["ind"], alpha=0.20, linewidth=0.9)
                    cell_has_data = True
                    if first_for_strat and i == 0 and j == 0:
                        legend_handles.append(line)
                        legend_labels.append(f"{sty['label']} individual")
                        first_for_strat = False

            # Ensemble curves (bold)
            for strategy, sty in strat_colors.items():
                for ens_size, alpha, lw in zip(sizes, size_alphas, size_widths):
                    run_name = f"{group}_{strategy}_ens{ens_size}_replay"
                    run = _latest_run(api, group, run_name)
                    if run is None:
                        print(f"  MISSING: {run_name}")
                        continue
                    toks, losses = fetch_ens_tokens_curve(api, run.id)
                    if len(toks) == 0:
                        # fallback: use ens/epoch * inferred tokens_per_epoch
                        epochs, losses = fetch_ens_curve(api, run.id)
                        if len(epochs) == 0:
                            continue
                        # Infer tokens_per_epoch from first individual model run
                        ind_runs = _all_runs(api, group, f"{group}_{strategy}_model0")
                        tpe = None
                        if ind_runs:
                            t, _ = _fetch_individual_val_tokens_curve(ind_runs[-1])
                            if len(t) > 0:
                                tpe = t[0]  # tokens_seen at end of epoch 1
                        if tpe is None:
                            print(f"  cannot infer tokens_per_epoch for {run_name}")
                            continue
                        order = np.argsort(epochs)
                        toks = epochs[order] * tpe
                        losses = losses[order]
                    else:
                        order = np.argsort(toks)
                        toks = toks[order]
                        losses = losses[order]

                    line, = ax.plot(toks, losses, color=sty["ens"], alpha=alpha,
                                    linewidth=lw, marker="o", markersize=3.5)
                    cell_has_data = True
                    if i == 0 and j == 0:
                        legend_handles.append(line)
                        legend_labels.append(f"{sty['label']} ens={ens_size}")

            # Cell labels
            ax.set_title(f"d{L} × w{W}", fontsize=11)
            if i == len(depths) - 1:
                ax.set_xlabel("cumulative tokens seen")
            if j == 0:
                ax.set_ylabel("val loss")
            ax.grid(True, alpha=0.25)
            if not cell_has_data:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="gray", fontsize=14)

            # x-axis tick formatting (tokens in millions)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))

    fig.suptitle(f"Grid val loss: rows = depth, cols = width  |  tag {args.grid_tag}, df={args.data_fraction}",
                 fontsize=13, y=0.995)
    fig.legend(legend_handles, legend_labels, loc="lower center",
               ncol=min(len(legend_labels), 5), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {args.out}")


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


def _group_name(grid_tag, depth, width, df):
    return f"grid_{grid_tag}_d{depth}_w{width}_df{df}"


def fetch_cell_curve(api, grid_tag, depth, width, df, strategy, ens_size, num_models=5):
    """Return (tokens_seen, val_loss) for one (depth, width, df, strategy, ens_size).

    ens_size == 1 -> mean across the `num_models` individual model val curves.
    ens_size > 1  -> the post-hoc replay run for that size.
    """
    group = _group_name(grid_tag, depth, width, df)
    if ens_size == 1:
        # mean over individual models (each model run logs only its own val)
        per_model = []
        for m in range(num_models):
            runs = _all_runs(api, group, f"{group}_{strategy}_model{m}")
            if not runs:
                continue
            t, v = _merge_individual_val_tokens_curves(runs)
            if len(t) == 0:
                continue
            per_model.append(dict(zip(t.tolist(), v.tolist())))
        if not per_model:
            return np.array([]), np.array([])
        # restrict to tokens points present in ALL models (so the mean is well-defined)
        common = set(per_model[0].keys())
        for d in per_model[1:]:
            common &= set(d.keys())
        if not common:
            return np.array([]), np.array([])
        toks = np.array(sorted(common))
        losses = np.array([np.mean([d[t] for d in per_model]) for t in toks])
        return toks, losses
    else:
        run_name = f"{group}_{strategy}_ens{ens_size}_replay"
        run = _latest_run(api, group, run_name)
        if run is None:
            return np.array([]), np.array([])
        toks, losses = fetch_ens_tokens_curve(api, run.id)
        if len(toks) == 0:
            # fallback: ens/epoch * tokens_per_epoch (inferred from a model-0 individual run)
            ep, ls = fetch_ens_curve(api, run.id)
            if len(ep) == 0:
                return np.array([]), np.array([])
            ind_runs = _all_runs(api, group, f"{group}_{strategy}_model0")
            tpe = None
            if ind_runs:
                t, _ = _fetch_individual_val_tokens_curve(ind_runs[-1])
                if len(t) > 0:
                    tpe = t[0]
            if tpe is None:
                return np.array([]), np.array([])
            order = np.argsort(ep)
            return ep[order] * tpe, ls[order]
        order = np.argsort(toks)
        return toks[order], losses[order]


def plot_slice(args):
    """Plot one panel with one line per (depth, width, ens_size, strategy, df) combo.

    Pass any axis as a list (--depths 6 12 24, --widths 768, --ens-sizes 1 2 4 5,
    --strategies init_shuffle_ens, --dfs 0.2). Lines are colored by the axes that vary.
    """
    api = wandb.Api()
    depths = [int(d) for d in args.depths]
    widths = [int(w) for w in args.widths]
    ens_sizes = [int(s) for s in args.ens_sizes]
    strategies = list(args.strategies)
    dfs = list(args.dfs)

    # axis lengths to detect "varying" vs "fixed"
    axes_info = [
        ("depth",    "d",     depths),
        ("width",    "w",     widths),
        ("ens_size", "ens",   ens_sizes),
        ("strategy", "strat", strategies),
        ("df",       "df",    dfs),
    ]
    varying = [(name, abbr, vals) for (name, abbr, vals) in axes_info if len(vals) > 1]
    fixed   = [(name, abbr, vals[0]) for (name, abbr, vals) in axes_info if len(vals) == 1]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by the *primary* varying axis (first in axes_info order). If multiple
    # axes vary, secondary axes get linestyle / marker variation.
    primary = varying[0] if varying else None
    secondary = varying[1] if len(varying) >= 2 else None

    primary_vals = primary[2] if primary else [None]
    secondary_vals = secondary[2] if secondary else [None]
    cmap = cm.get_cmap("viridis", max(2, len(primary_vals)))
    linestyles = ["-", "--", ":", "-."]
    markers = ["o", "s", "D", "^", "v", "x"]

    # Build the cartesian iteration order
    for d in depths:
        for w in widths:
            for s in ens_sizes:
                for strat in strategies:
                    for df in dfs:
                        t, l = fetch_cell_curve(api, args.grid_tag, d, w, df, strat, s,
                                                num_models=args.num_models)
                        if len(t) == 0:
                            continue
                        # determine color/linestyle from primary/secondary axis values
                        slot = {"depth": d, "width": w, "ens_size": s, "strategy": strat, "df": df}
                        color = "tab:blue"
                        if primary is not None:
                            pi = primary[2].index(slot[primary[0]])
                            color = cmap(pi / max(1, len(primary[2]) - 1))
                        ls = "-"
                        marker = "o"
                        if secondary is not None:
                            si = secondary[2].index(slot[secondary[0]])
                            ls = linestyles[si % len(linestyles)]
                            marker = markers[si % len(markers)]

                        # Build label from varying axes only
                        label_parts = []
                        for (name, abbr, _) in varying:
                            label_parts.append(f"{abbr}={slot[name]}")
                        label = ", ".join(label_parts) if label_parts else f"d{d}_w{w}_ens{s}_{strat}_df{df}"

                        ax.plot(t, l, color=color, linestyle=ls, marker=marker,
                                markersize=3.5, linewidth=2.0, alpha=0.9, label=label)

    # Title with fixed dims
    fixed_str = ", ".join(f"{abbr}={val}" for (_, abbr, val) in fixed)
    vary_str  = ", ".join(name for (name, _, _) in varying) or "(no variation)"
    ax.set_title(f"val loss slice — varying: {vary_str}   |   fixed: {fixed_str}", fontsize=11)
    ax.set_xlabel("cumulative tokens seen")
    ax.set_ylabel("val loss")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=max(1, len(ax.get_lines()) // 8 + 1))
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")


def _compute_arch_params(L, d, vocab=50257, ffn_mult=4):
    """Approximate total params for our slowrun arch (no ve_projs, SwiGLU h=4d)."""
    h = ffn_mult * d
    per_layer = 4 * d * d + 3 * d * h + 2 * d
    embed_lm = 2 * vocab * d
    return L * per_layer + embed_lm + d


def plot_param_match(args):
    """Test 'wider ≈ ensembling' hypothesis at matched total params.

    Width-doubling does NOT exactly quadruple params for our arch (embedding + LM head
    are linear-in-d, not d²). True ratios: w384→w768 = 2.85x, w768→w1536 = 3.19x.
    So neither ens=2 nor ens=4 is a perfect match; we BRACKET with both:
      - ens=2 narrow: ensemble has FEWER params than wider single (under-matched)
      - ens=4 narrow: ensemble has MORE params than wider single (over-matched)
    If the ensemble wins in BOTH directions, 'wider ≈ ensembling' is unambiguously rejected.
    """
    api = wandb.Api()
    depth = int(args.depth)
    df = args.data_fraction
    strategy = args.strategy
    pairs = []
    for p in args.pairs:
        a, b = p.split(":")
        pairs.append((int(a), int(b)))

    fig, axes = plt.subplots(1, len(pairs), figsize=(8 * len(pairs), 6),
                             sharey=False, squeeze=False)
    axes = axes[0]

    for ax, (sm_w, bg_w) in zip(axes, pairs):
        print(f"\n--- pair: w={sm_w} vs w={bg_w} (depth={depth}, strat={strategy}) ---")
        sm_p = _compute_arch_params(depth, sm_w)
        bg_p = _compute_arch_params(depth, bg_w)
        ratio = bg_p / sm_p
        print(f"   single-model params: w{sm_w}={sm_p/1e6:.1f}M, w{bg_w}={bg_p/1e6:.1f}M, ratio={ratio:.2f}x")

        sm_ens1_t, sm_ens1_l = fetch_cell_curve(api, args.grid_tag, depth, sm_w, df, strategy, 1, args.num_models)
        sm_ens2_t, sm_ens2_l = fetch_cell_curve(api, args.grid_tag, depth, sm_w, df, strategy, 2, args.num_models)
        sm_ens3_t, sm_ens3_l = fetch_cell_curve(api, args.grid_tag, depth, sm_w, df, strategy, 3, args.num_models)
        sm_ens4_t, sm_ens4_l = fetch_cell_curve(api, args.grid_tag, depth, sm_w, df, strategy, 4, args.num_models)
        bg_ens1_t, bg_ens1_l = fetch_cell_curve(api, args.grid_tag, depth, bg_w, df, strategy, 1, args.num_models)
        bg_ens5_t, bg_ens5_l = fetch_cell_curve(api, args.grid_tag, depth, bg_w, df, strategy, 5, args.num_models)

        # Ensemble param totals (per ensemble member) for label clarity
        ens2_p = 2 * sm_p
        ens4_p = 4 * sm_p

        if len(sm_ens1_t):
            ax.plot(sm_ens1_t, sm_ens1_l, color="tab:gray", linewidth=1.4, marker=".",
                    markersize=4, alpha=0.6,
                    label=f"w{sm_w} ens=1  ({sm_p/1e6:.0f}M  baseline)")
        if len(sm_ens2_t):
            ax.plot(sm_ens2_t, sm_ens2_l, color="tab:green", linewidth=1.6, marker="s",
                    markersize=4, linestyle="--", alpha=0.65,
                    label=f"w{sm_w} ens=2  ({ens2_p/1e6:.0f}M  UNDER, {ens2_p/bg_p:.2f}x)")
        if len(sm_ens3_t):
            ax.plot(sm_ens3_t, sm_ens3_l, color="tab:green", linewidth=2.8, marker="s",
                    markersize=5.5,
                    label=f"w{sm_w} ens=3  ({3*sm_p/1e6:.0f}M  ← TRUE MATCH, {3*sm_p/bg_p:.2f}x)")
        if len(sm_ens4_t):
            ax.plot(sm_ens4_t, sm_ens4_l, color="tab:green", linewidth=1.6, marker="s",
                    markersize=4, linestyle=":", alpha=0.65,
                    label=f"w{sm_w} ens=4  ({ens4_p/1e6:.0f}M  OVER, {ens4_p/bg_p:.2f}x)")
        if len(bg_ens1_t):
            ax.plot(bg_ens1_t, bg_ens1_l, color="tab:red", linewidth=2.6, marker="o",
                    markersize=5,
                    label=f"w{bg_w} ens=1  ({bg_p/1e6:.0f}M  ← WIDER SINGLE, ref)")
        if len(bg_ens5_t):
            ax.plot(bg_ens5_t, bg_ens5_l, color="tab:purple", linewidth=1.4, marker="^",
                    markersize=4, alpha=0.4,
                    label=f"w{bg_w} ens=5  ({5*bg_p/1e6:.0f}M  wider's own ens, upper-ref)")

        ax.set_title(f"d={depth}: w{sm_w} (ens=2,4) vs w{bg_w} (ens=1)   "
                     f"[wider/narrow params = {ratio:.2f}x]",
                     fontsize=10)
        ax.set_xlabel("cumulative tokens seen (per model)")
        ax.set_ylabel("val loss")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

        # Annotation: bracket Δs (under/over) + true match (ens=3)
        if len(sm_ens2_l) and len(sm_ens4_l) and len(bg_ens1_l):
            sm2_min = float(np.min(sm_ens2_l))
            sm3_min = float(np.min(sm_ens3_l)) if len(sm_ens3_l) else None
            sm4_min = float(np.min(sm_ens4_l))
            bg_min  = float(np.min(bg_ens1_l))
            gap2 = sm2_min - bg_min
            gap4 = sm4_min - bg_min
            lines = [
                f"min(w{sm_w} ens=2) = {sm2_min:.4f}   Δ = {gap2:+.4f}",
            ]
            if sm3_min is not None:
                gap3 = sm3_min - bg_min
                lines.append(f"min(w{sm_w} ens=3) = {sm3_min:.4f}   Δ = {gap3:+.4f}  ← TRUE MATCH")
            lines += [
                f"min(w{sm_w} ens=4) = {sm4_min:.4f}   Δ = {gap4:+.4f}",
                f"min(w{bg_w} ens=1) = {bg_min:.4f}   (wider single, reference)",
            ]
            ax.text(0.02, 0.02, "\n".join(lines),
                    transform=ax.transAxes, fontsize=8, family="monospace",
                    verticalalignment="bottom",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85,
                              edgecolor="lightgray"))

    fig.suptitle(f"Width vs ensembling at matched params (bracketed)  |  d={depth}, strat={strategy}, df={df}",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {args.out}")


def plot_heatmap(args):
    """Heatmap: one panel per (strategy, ens_size). Cell color = best ensemble val loss."""
    api = wandb.Api()
    depths = [int(d) for d in args.depths]
    widths = [int(w) for w in args.widths]
    strategies = list(args.strategies)
    ens_sizes = [int(s) for s in args.ens_sizes]
    df = args.data_fraction

    n_rows = len(strategies)
    n_cols = len(ens_sizes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.8 * n_rows),
                             squeeze=False)

    # First pass: collect all best val losses to compute a shared color range
    grid = np.full((n_rows, n_cols, len(depths), len(widths)), np.nan)
    for r, strat in enumerate(strategies):
        for c, s in enumerate(ens_sizes):
            for i, d in enumerate(depths):
                for j, w in enumerate(widths):
                    t, l = fetch_cell_curve(api, args.grid_tag, d, w, df, strat, s,
                                            num_models=args.num_models)
                    if len(l):
                        grid[r, c, i, j] = float(np.min(l))

    vmin = np.nanmin(grid)
    vmax = np.nanmax(grid)

    for r, strat in enumerate(strategies):
        for c, s in enumerate(ens_sizes):
            ax = axes[r][c]
            arr = grid[r, c]
            im = ax.imshow(arr, cmap="viridis_r", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(widths)))
            ax.set_xticklabels([f"w{w}" for w in widths])
            ax.set_yticks(range(len(depths)))
            ax.set_yticklabels([f"d{d}" for d in depths])
            short_strat = strat.replace("_ens", "")
            ax.set_title(f"{short_strat}, ens={s}", fontsize=10)
            for i in range(len(depths)):
                for j in range(len(widths)):
                    v = arr[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                                color="white" if v > (vmin + vmax) / 2 else "black",
                                fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="best val loss",
                 fraction=0.025, pad=0.02)
    fig.suptitle(f"Best ensemble val loss across (depth, width)  |  tag {args.grid_tag}, df={df}",
                 fontsize=12)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
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

    p_grid = sub.add_parser("grid", help="3x3 (depth × width) grid of val loss curves with ensembling overlays")
    p_grid.add_argument("--grid-tag", required=True,
                        help="Shared GRID_TAG used in wandb group names (e.g. 20260430_152533)")
    p_grid.add_argument("--depths", nargs="+", default=[6, 12, 24],
                        help="Depth values (n_layer); rows of the grid")
    p_grid.add_argument("--widths", nargs="+", default=[384, 768, 1536],
                        help="Width values (n_embd); cols of the grid")
    p_grid.add_argument("--ensemble-sizes", type=int, nargs="+", default=[2, 4, 5])
    p_grid.add_argument("--num-models", type=int, default=5,
                        help="Number of individual models per strategy to plot")
    p_grid.add_argument("--data-fraction", default="0.2",
                        help="Used to construct wandb group name")
    p_grid.add_argument("--out", default="experiments/analysis/grid_combo.png")
    p_grid.set_defaults(fn=plot_grid)

    # ---- slice: vary one (or more) axes, hold the rest fixed ----
    p_slice = sub.add_parser("slice", help="One panel; vary one (or more) of {depth, width, ens_size, strategy, df}")
    p_slice.add_argument("--grid-tag", required=True)
    p_slice.add_argument("--depths", type=int, nargs="+", default=[12])
    p_slice.add_argument("--widths", type=int, nargs="+", default=[768])
    p_slice.add_argument("--ens-sizes", type=int, nargs="+", default=[5],
                         help="1 = mean of individuals; 2/4/5 = post-hoc replay")
    p_slice.add_argument("--strategies", nargs="+", default=["init_shuffle_ens"],
                         choices=["init_ens", "init_shuffle_ens"])
    p_slice.add_argument("--dfs", nargs="+", default=["0.2"],
                         help="One or more data_fraction values (matches wandb group name)")
    p_slice.add_argument("--num-models", type=int, default=5)
    p_slice.add_argument("--out", required=True)
    p_slice.set_defaults(fn=plot_slice)

    # ---- param-match: width vs ensembling at equal params ----
    p_pm = sub.add_parser("param-match",
                          help="Compare ensemble of N narrow models vs single √N-wider model at matched params")
    p_pm.add_argument("--grid-tag", required=True)
    p_pm.add_argument("--depth", type=int, default=12)
    p_pm.add_argument("--pairs", nargs="+", default=["384:768", "768:1536"],
                      help="Width pairs (small:big). Each pair is matched at ~4× params (ens=4 vs ens=1)")
    p_pm.add_argument("--strategy", default="init_shuffle_ens",
                      choices=["init_ens", "init_shuffle_ens"])
    p_pm.add_argument("--data-fraction", default="0.2")
    p_pm.add_argument("--num-models", type=int, default=5)
    p_pm.add_argument("--ens-size", type=int, default=5,
                      help="Ensemble size for the narrow-width ensemble line (default 5 = best we have)")
    p_pm.add_argument("--out", required=True)
    p_pm.set_defaults(fn=plot_param_match)

    # ---- heatmap: best val loss across (depth, width) ----
    p_heat = sub.add_parser("heatmap", help="Best val loss heatmap over (depth × width), one panel per (strategy, ens_size)")
    p_heat.add_argument("--grid-tag", required=True)
    p_heat.add_argument("--depths", type=int, nargs="+", default=[6, 12, 24])
    p_heat.add_argument("--widths", type=int, nargs="+", default=[384, 768, 1536])
    p_heat.add_argument("--ens-sizes", type=int, nargs="+", default=[1, 5])
    p_heat.add_argument("--strategies", nargs="+", default=["init_ens", "init_shuffle_ens"])
    p_heat.add_argument("--data-fraction", default="0.2")
    p_heat.add_argument("--num-models", type=int, default=5)
    p_heat.add_argument("--out", required=True)
    p_heat.set_defaults(fn=plot_heatmap)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
