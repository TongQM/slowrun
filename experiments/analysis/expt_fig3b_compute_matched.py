"""Expt Fig 3b/c: compute-matched ensembling vs width/depth scaling.

For each axis (width @ d=12; depth @ w=768), plot best val loss vs effective
training compute, where:
  - "Scale model" point: single model at scaled size, compute ∝ non-embed params
  - "Ensemble at base" point: E base-size models, compute = E × params(base)

Headline claim: at matched compute, ensembling base beats scaling.
"""
import os, re
import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

ENT = "xjtumyd-carnegie-mellon-university/slowrun"
TAG_BASE = "20260430_152533"
TAG_EXT = "df02ext_20260504_224131"
EPOCH_VAL_RE = re.compile(r"\[model \d+\]\s+epoch\s+(\d+)\s+val_loss=([0-9.]+)")


def group_for(d, w):
    if d == 18 or w == 1152:
        return f"grid_{TAG_EXT}_d{d}_w{w}_df0.2"
    return f"grid_{TAG_BASE}_d{d}_w{w}_df0.2"


def non_embed_params(d, w):
    """16 * L * N^2 — transformer body params."""
    return 16 * d * w ** 2


def fetch_min_val_indiv(api, group, n_indiv=5, strat="init_shuffle_ens"):
    """Mean min val loss across n_indiv individual training runs."""
    mins = []
    for m in range(n_indiv):
        name = f"{group}_{strat}_model{m}"
        runs = list(api.runs(ENT, filters={"group": group, "display_name": name}))
        if not runs:
            continue
        # Try several model_k namespaces (different launchers use different idx).
        all_vls = []
        for r in runs:
            for k in range(1, 25):
                vls = [h.get(f"model_{k}/val_loss") for h in r.scan_history(keys=[f"model_{k}/val_loss"])
                       if h.get(f"model_{k}/val_loss") is not None]
                if vls:
                    all_vls.extend(vls)
                    break
        if all_vls:
            mins.append(min(all_vls))
    if not mins:
        return np.nan, np.nan
    return float(np.mean(mins)), float(np.std(mins))


def fetch_min_val_ens(api, group, E, strat="init_shuffle_ens"):
    """Min val loss for ensemble of size E at this cell."""
    name = f"{group}_{strat}_ens{E}_replay"
    runs = list(api.runs(ENT, filters={"group": group, "display_name": name}))
    runs.sort(key=lambda r: r.created_at, reverse=True)
    if not runs: return np.nan
    r = runs[0]
    vls = []
    for h in r.scan_history(keys=["ens/val_loss"]):
        vl = h.get("ens/val_loss")
        if vl is not None: vls.append(vl)
    return min(vls) if vls else np.nan


def main():
    sns.set(font_scale=1.5)
    sns.set_style('whitegrid')
    sns.set_palette('cool', 6)
    plt.rcParams['axes.labelsize']  = 24
    plt.rcParams['axes.linewidth']  = 4.0
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['grid.alpha']      = 0.25
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    api = wandb.Api()
    palette = sns.color_palette('cool', 2)

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # ===== 3b: width axis at d=12 =====
    ax = axes[0]
    base_d, base_w = 12, 768
    base_params = non_embed_params(base_d, base_w)
    base_group = group_for(base_d, base_w)

    # (i) Scale model width
    width_xs, width_ys = [], []
    for w in [384, 768, 1152, 1536]:
        g = group_for(base_d, w)
        mean_min, _ = fetch_min_val_indiv(api, g)
        params = non_embed_params(base_d, w)
        width_xs.append(params / base_params)
        width_ys.append(mean_min)
        print(f"width-scale d={base_d} w={w}: P={params/1e6:.1f}M ({params/base_params:.2f}×), min_val={mean_min:.4f}")
    ax.plot(width_xs, width_ys, color=palette[0], lw=3.0, marker="o", markersize=14,
            markeredgecolor="black", markeredgewidth=0.7,
            label="scale width (E=1)")

    # (ii) Ensemble at base width
    ens_xs, ens_ys = [], []
    e1_mean, _ = fetch_min_val_indiv(api, base_group)
    ens_xs.append(1.0); ens_ys.append(e1_mean)
    for E in [2, 3, 4, 5]:
        L = fetch_min_val_ens(api, base_group, E)
        ens_xs.append(float(E)); ens_ys.append(L)
        print(f"ensemble  d={base_d} w={base_w} E={E}: compute={E}×, min_val={L:.4f}")
    ax.plot(ens_xs, ens_ys, color=palette[1], lw=3.0, marker="s", markersize=14,
            markeredgecolor="black", markeredgewidth=0.7,
            label="ensemble at $w$=768 (varying $E$)")

    ax.set_xscale('log')
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.25×", "0.5×", "1×", "2×", "4×"])
    ax.set_xlabel(r"compute (relative to $d=12, w=768, E=1$)")
    ax.set_ylabel("min val loss")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.text(0.02, 0.97, "vary width vs $E$ at $d$=12",
            transform=ax.transAxes, fontsize=20, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.85, edgecolor="lightgray"))

    # ===== 3c: depth axis at w=768 =====
    ax = axes[1]
    base_d, base_w = 12, 768
    base_params = non_embed_params(base_d, base_w)
    base_group = group_for(base_d, base_w)

    depth_xs, depth_ys = [], []
    for d in [6, 12, 18, 24]:
        g = group_for(d, base_w)
        mean_min, _ = fetch_min_val_indiv(api, g)
        params = non_embed_params(d, base_w)
        depth_xs.append(params / base_params)
        depth_ys.append(mean_min)
        print(f"depth-scale d={d} w={base_w}: P={params/1e6:.1f}M ({params/base_params:.2f}×), min_val={mean_min:.4f}")
    ax.plot(depth_xs, depth_ys, color=palette[0], lw=3.0, marker="o", markersize=14,
            markeredgecolor="black", markeredgewidth=0.7,
            label="scale depth (E=1)")

    # Same ensemble line at base
    ax.plot(ens_xs, ens_ys, color=palette[1], lw=3.0, marker="s", markersize=14,
            markeredgecolor="black", markeredgewidth=0.7,
            label="ensemble at $d$=12 (varying $E$)")

    ax.set_xscale('log')
    ax.set_xticks([0.5, 1, 1.5, 2, 5])
    ax.set_xticklabels(["0.5×", "1×", "1.5×", "2×", "5×"])
    ax.set_xlabel(r"compute (relative to $d=12, w=768, E=1$)")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.text(0.02, 0.97, "vary depth vs $E$ at $w$=768",
            transform=ax.transAxes, fontsize=20, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.85, edgecolor="lightgray"))

    out = "experiments/figures/03_compute_matched/expt_fig3bc.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches='tight', dpi=300)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
