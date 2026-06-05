"""Expt Fig 3a: 4×4 grid of val curves at df=0.2 across depth × width.

Cells from two wandb tags (combined into one figure):
  - grid_20260430_152533: depths {6,12,24} × widths {384,768,1536}  (9 cells)
  - df02ext_20260504_224131: d18 row + w1152 column                 (7 new cells)

Per cell: 5 individuals (transparent) + ensembles E ∈ {2,3,4,5} (bold).
"""
import os
import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

ENT = "xjtumyd-carnegie-mellon-university/slowrun"
TAG_BASE = "20260430_152533"
TAG_EXT  = "df02ext_20260504_224131"

DEPTHS = [6, 12, 18, 24]
WIDTHS = [384, 768, 1152, 1536]
E_SIZES = [2, 3, 4, 5]
N_INDIV = 5


def group_for(d, w):
    """Pick the right wandb group for this cell."""
    if d == 18 or w == 1152:
        return f"grid_{TAG_EXT}_d{d}_w{w}_df0.2"
    return f"grid_{TAG_BASE}_d{d}_w{w}_df0.2"


def fetch_individual_curve(api, group, name):
    runs = list(api.runs(ENT, filters={"group": group, "display_name": name}))
    runs.sort(key=lambda r: r.created_at)
    by_tok = {}
    for r in runs:
        for h in r.scan_history(keys=["model_1/val_loss"]):
            ts = None
            vl = h.get("model_1/val_loss")
            if vl is None: continue
        # need to fetch tokens_seen too
        for h in r.scan_history(keys=["model_1/val_loss", "model_1/tokens_seen"]):
            vl = h.get("model_1/val_loss"); ts = h.get("model_1/tokens_seen")
            if vl is not None and ts is not None: by_tok[int(ts)] = float(vl)
    if not by_tok: return np.array([]), np.array([])
    toks = np.array(sorted(by_tok))
    return toks, np.array([by_tok[t] for t in toks])


def fetch_ensemble_curve(api, group, name):
    runs = list(api.runs(ENT, filters={"group": group, "display_name": name}))
    runs.sort(key=lambda r: r.created_at, reverse=True)
    if not runs: return np.array([]), np.array([])
    r = runs[0]
    toks, vls = [], []
    for h in r.scan_history(keys=["ens/tokens_seen", "ens/val_loss"]):
        ts = h.get("ens/tokens_seen"); vl = h.get("ens/val_loss")
        if ts is not None and vl is not None:
            toks.append(int(ts)); vls.append(float(vl))
    if not toks: return np.array([]), np.array([])
    order = np.argsort(toks)
    return np.array(toks)[order], np.array(vls)[order]


def main():
    sns.set(font_scale=1.2)
    sns.set_style('whitegrid')
    plt.rcParams['axes.labelsize']  = 18
    plt.rcParams['axes.linewidth']  = 2.5
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['grid.alpha']      = 0.25
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    api = wandb.Api()
    fig, axes = plt.subplots(len(DEPTHS), len(WIDTHS),
                             figsize=(5.5 * len(WIDTHS), 4.0 * len(DEPTHS)),
                             squeeze=False)

    palette_E = sns.color_palette('cool', len(E_SIZES) + 1)  # +1 because we want a darker color
    palette_E = palette_E[1:]  # skip the lightest, start from darker
    ind_color = "lightgray"

    legend_handles = []
    legend_labels = []

    for i, d in enumerate(DEPTHS):
        for j, w in enumerate(WIDTHS):
            ax = axes[i][j]
            group = group_for(d, w)
            print(f"d{d}/w{w}: group={group}")

            for strat in ["init_shuffle_ens"]:  # use one strategy per cell to avoid clutter
                # Individuals (transparent)
                first_ind_plotted = False
                for m in range(N_INDIV):
                    name = f"{group}_{strat}_model{m}"
                    toks, vls = fetch_individual_curve(api, group, name)
                    if len(toks) == 0: continue
                    line, = ax.plot(toks, vls, color=ind_color, alpha=0.7, lw=1.2, zorder=2)
                    if i == 0 and j == 0 and not first_ind_plotted:
                        legend_handles.append(line)
                        legend_labels.append(f"individual ({N_INDIV} per cell)")
                        first_ind_plotted = True

                # Ensembles
                for k, E in enumerate(E_SIZES):
                    name = f"{group}_{strat}_ens{E}_replay"
                    toks, vls = fetch_ensemble_curve(api, group, name)
                    if len(toks) == 0:
                        print(f"  MISSING: {name}")
                        continue
                    line, = ax.plot(toks, vls, color=palette_E[k], lw=2.6,
                                    marker="o", markersize=4, zorder=3 + k)
                    if i == 0 and j == 0:
                        legend_handles.append(line)
                        legend_labels.append(f"E = {E}")

            ax.set_title(f"d={d}, w={w}", fontsize=13)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
            if i == len(DEPTHS) - 1:
                ax.set_xlabel("tokens seen")
            if j == 0:
                ax.set_ylabel("val loss")

    fig.legend(legend_handles, legend_labels, loc="lower center",
               ncol=len(legend_labels), fontsize=13, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    out = "experiments/figures/03_compute_matched/grid_4x4.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches='tight', dpi=300)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
