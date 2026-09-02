"""Parse Q2 train logs (original 25-ep + extension 26-40-ep) and plot 20 individuals'
val curves at d12/w768, df=0.2. Bypasses the wandb sync issue (which silently
overwrote part of the original cloud data when the offline-mode runs were uploaded).

Sources of truth:
  experiments/logs/q2_train_d12_w768_df0.2_40532724_<task>.out  — original 25 ep
  experiments/logs/q2ext_train_40583426_<task>.out              — extension 26-40 ep

Task → (strategy, model) map:
  task 0..19    → init_ens         model 0..19
  task 20..39   → init_shuffle_ens model 0..19
"""
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

LOGS = "experiments/logs"
ORIG_PREFIX = f"{LOGS}/q2_train_d12_w768_df0.2_40532724"
EXT_PREFIX = f"{LOGS}/q2ext_train_40583426"
TOKENS_PER_EPOCH = int(0.2 * 99942400)  # ≈ 19,988,480

# Pattern: "  [model 1] epoch <N> val_loss=<F> val_bpb=<F>"
EPOCH_VAL_RE = re.compile(r"\[model \d+\]\s+epoch\s+(\d+)\s+val_loss=([0-9.]+)")


def parse_log(path):
    """Return dict {epoch: val_loss}."""
    d = {}
    if not os.path.isfile(path):
        return d
    with open(path) as f:
        for line in f:
            m = EPOCH_VAL_RE.search(line)
            if m:
                ep = int(m.group(1))
                vl = float(m.group(2))
                d[ep] = vl
    return d


def collect(strategy_idx, model_idx, num_models=20):
    """Merge val curves from original + extension logs for one (strategy, model)."""
    task = strategy_idx * num_models + model_idx
    orig = parse_log(f"{ORIG_PREFIX}_{task}.out")
    ext = parse_log(f"{EXT_PREFIX}_{task}.out")
    merged = {**orig, **ext}  # extension wins on overlaps (none expected)
    return merged


def main():
    # Project plot style (per CLAUDE.md)
    sns.set(font_scale=1.5)
    sns.set_style('whitegrid')
    sns.set_palette('cool', 1)

    plt.rcParams['axes.labelsize']  = 24
    plt.rcParams['axes.linewidth']  = 4.0
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['grid.alpha']      = 0.25
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    n_individuals = 20

    # Collect per-(epoch, model) val loss; assemble a (n_individuals, n_epochs) matrix
    per_model = []
    for m in range(n_individuals):
        d = collect(strategy_idx=0, model_idx=m, num_models=n_individuals)
        if d:
            per_model.append(d)
    if not per_model:
        print("no data found")
        return

    # Build dense matrix indexed by epoch (assume all 1..40 present)
    all_epochs = sorted(set().union(*[set(d) for d in per_model]))
    losses = np.array([[d.get(e, np.nan) for e in all_epochs] for d in per_model])
    eps = np.array(all_epochs)
    toks = eps * TOKENS_PER_EPOCH

    mean = np.nanmean(losses, axis=0)
    std = np.nanstd(losses, axis=0)
    print(f"{losses.shape[0]} individuals across {len(eps)} epochs")
    print(f"  mean range: {mean.min():.4f} (ep {int(eps[mean.argmin()])}) → {mean[-1]:.4f}")
    print(f"  std at min-mean epoch: {std[mean.argmin()]:.4f}")

    fig, ax = plt.subplots(figsize=(13, 7))

    # Epoch boundaries — vertical dashed lines. Each epoch = 20M tokens at df=0.2,
    # so the spacing visually conveys the unique-dataset size being repeated.
    # Uniform style across all epochs so the 20M spacing reads clearly.
    n_epochs_max = int(eps.max())
    for k in range(1, n_epochs_max + 1):
        ax.axvline(x=k * TOKENS_PER_EPOCH, color="gray",
                   linestyle="--", alpha=0.45, linewidth=1.0, zorder=1)

    color = sns.color_palette('cool', 1)[0]
    ax.plot(toks, mean, color=color, lw=3.0,
            label=f"mean ± std (E={n_individuals} individuals)", zorder=4)
    ax.fill_between(toks, mean - std, mean + std, color=color, alpha=0.25, zorder=3)

    # Mark mean's minimum
    i_min = int(np.argmin(mean))
    ax.plot([toks[i_min]], [mean[i_min]], marker="v", markersize=12,
            color=color, markeredgecolor="black", markeredgewidth=0.7, zorder=5)

    ax.set_xlabel(f"tokens seen (1 epoch = {TOKENS_PER_EPOCH/1e6:.0f}M tokens)")
    ax.set_ylabel("validation loss")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.legend(loc="upper left", framealpha=0.95)

    out = "experiments/figures/01_overfit_demos/q2ext_individuals_overfit.pdf"
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches='tight', dpi=300)
    print(f"\nSaved {out}")
    print(f"Saved {out.replace('.pdf', '.png')} (preview)")


if __name__ == "__main__":
    main()
