"""Expt Fig 3d: heatmap of best ensemble val loss across (depth, width) at E=5.

For each of the 16 (d, w) cells in the 4x4 grid, compute the minimum val loss
over training epochs at E=5 ensembling (init_shuffle), and display as a 4x4 heatmap.
"""
import os
import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

ENT = "xjtumyd-carnegie-mellon-university/slowrun"
TAG_BASE = "20260430_152533"
TAG_EXT = "df02ext_20260504_224131"


def group_for(d, w):
    if d == 18 or w == 1152:
        return f"grid_{TAG_EXT}_d{d}_w{w}_df0.2"
    return f"grid_{TAG_BASE}_d{d}_w{w}_df0.2"


def fetch_min_val_ens(api, group, E, strat="init_shuffle_ens"):
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
    sns.set_style('white')
    plt.rcParams['axes.labelsize']  = 24
    plt.rcParams['axes.linewidth']  = 4.0
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    api = wandb.Api()
    DEPTHS = [6, 12, 18, 24]
    WIDTHS = [384, 768, 1152, 1536]
    E = 5

    grid = np.full((len(DEPTHS), len(WIDTHS)), np.nan)
    for i, d in enumerate(DEPTHS):
        for j, w in enumerate(WIDTHS):
            g = group_for(d, w)
            v = fetch_min_val_ens(api, g, E)
            grid[i, j] = v
            print(f"d{d}/w{w}: min_val_E{E} = {v:.4f}")

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(grid, cmap="cool", aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("min val loss (E=5)", fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    ax.set_xticks(range(len(WIDTHS)))
    ax.set_xticklabels([f"{w}" for w in WIDTHS])
    ax.set_yticks(range(len(DEPTHS)))
    ax.set_yticklabels([f"{d}" for d in DEPTHS])
    ax.set_xlabel(r"width $N$ (n_embd)")
    ax.set_ylabel(r"depth $L$ (n_layer)")

    # Annotate each cell
    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    for i in range(len(DEPTHS)):
        for j in range(len(WIDTHS)):
            v = grid[i, j]
            if np.isnan(v): continue
            color = "white" if v > 0.5 * (vmin + vmax) else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color=color, fontsize=20, fontweight="bold")

    out = "experiments/figures/03_compute_matched/expt_fig3d_heatmap.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches='tight', dpi=300)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
