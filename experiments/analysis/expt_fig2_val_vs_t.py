"""Ensemble val loss vs tokens for different E (precursor to the scaling-law fit).

For each ensemble size E in {2, 5, 10, 15, 20} (Q2 ext fused replay) plus
E=1 (= mean across 20 individuals from the Q2 ext train logs), plot the
validation loss as a function of cumulative tokens-seen, with vertical
dashed lines at every epoch boundary (= 20M tokens at df=0.2).
"""
import os, re
import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

ENT = "xjtumyd-carnegie-mellon-university/slowrun"
GROUP = "q2_ensemble_size_q2_20260502_234110_d12_w768_df0.2"
SIZES = [2, 5, 10, 15, 20]
TOKENS_PER_EPOCH = int(0.2 * 99942400)  # ≈19,988,480 at df=0.2


def fetch_ens_curve(api, name):
    runs = list(api.runs(ENT, filters={"group": GROUP, "display_name": name}))
    runs.sort(key=lambda r: r.created_at, reverse=True)
    r = runs[0]
    toks, vls = [], []
    for h in r.scan_history(keys=["ens/val_loss"]):
        if h.get("ens/val_loss") is not None: vls.append(h["ens/val_loss"])
    for h in r.scan_history(keys=["ens/tokens_seen"]):
        if h.get("ens/tokens_seen") is not None: toks.append(h["ens/tokens_seen"])
    order = np.argsort(toks)
    return np.array(toks)[order], np.array(vls)[order]


# Parse train logs to get E=1 mean (mean of 20 individuals' val curves).
EPOCH_VAL_RE = re.compile(r"\[model \d+\]\s+epoch\s+(\d+)\s+val_loss=([0-9.]+)")
def parse_log(path):
    d = {}
    if not os.path.isfile(path): return d
    with open(path) as f:
        for line in f:
            m = EPOCH_VAL_RE.search(line)
            if m: d[int(m.group(1))] = float(m.group(2))
    return d


def e1_mean_for_strategy(strategy_idx, num_models=20):
    rows = []
    for m in range(num_models):
        task = strategy_idx * num_models + m
        d = {**parse_log(f"experiments/logs/q2_train_d12_w768_df0.2_40532724_{task}.out"),
             **parse_log(f"experiments/logs/q2ext_train_40583426_{task}.out")}
        if d: rows.append(d)
    eps = sorted(set().union(*[set(d) for d in rows]))
    arr = np.array([[d.get(e, np.nan) for e in eps] for d in rows])
    return np.array(eps) * TOKENS_PER_EPOCH, np.nanmean(arr, axis=0)


def main():
    sns.set(font_scale=1.5)
    sns.set_style('whitegrid')

    plt.rcParams['axes.labelsize']  = 24
    plt.rcParams['axes.linewidth']  = 4.0
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['grid.alpha']      = 0.25
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    api = wandb.Api()
    palette = sns.color_palette('cool', 6)

    strats = [("init_ens", "init", 0), ("init_shuffle_ens", "init + shuffle", 1)]
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)

    for ax, (strat, label_name, sidx) in zip(axes, strats):
        toks1, mean1 = e1_mean_for_strategy(strategy_idx=sidx)
        ax.plot(toks1, mean1, color=palette[0], lw=3.0,
                label="E = 1  (mean of individuals)")
        for j, E in enumerate(SIZES):
            name = f"{GROUP}_{strat}_ens{E}_replay"
            toks, vls = fetch_ens_curve(api, name)
            ax.plot(toks, vls, color=palette[j + 1], lw=3.0, label=f"E = {E}")

        # Epoch boundaries
        n_epochs = max(40, int(toks1.max() / TOKENS_PER_EPOCH))
        for k in range(1, n_epochs + 1):
            ax.axvline(x=k * TOKENS_PER_EPOCH, color="gray",
                       linestyle="--", alpha=0.45, linewidth=1.0, zorder=1)

        ax.set_xlabel(f"tokens seen (1 epoch = {TOKENS_PER_EPOCH/1e6:.0f}M tokens)")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
        # Annotate strategy in upper-left corner of the panel
        ax.text(0.02, 0.97, label_name, transform=ax.transAxes,
                fontsize=22, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="lightgray"))
        ax.legend(loc="upper right", framealpha=0.95)

    axes[0].set_ylabel("validation loss")

    out = "experiments/figures/02_ensemble_scaling/val_vs_t_per_E.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches='tight', dpi=300)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
