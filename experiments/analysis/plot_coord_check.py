"""CompleteP LR-transfer coordinate check (iso-aspect-ratio ladder, w/L=64).

For each model size on the ladder {d6/w384, d12/w768, d18/w1152, d24/w1536} and each
lr_multiplier {0.0625..2.0}, we trained 3 seeds for 1 epoch (df=0.2, wd=0, constant LR).
This plots val loss vs lr_multiplier, one curve per size, to check whether the MINIMUM
aligns across sizes (= CompleteP LR transfer holding). Absolute loss falls with size and
is deliberately NOT the point, so the right panel normalizes each curve to its own minimum.

Parsed from experiments/logs/coordcheck_*.out (header line `COORDCHECK cell=... lr_multiplier=... seed=...`
plus `[model N val @ step S] val_loss=L`). Per run we take the end-of-epoch val (last val@step).
"""
import os, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO = "/ocean/projects/cis260161p/ymiao6/scaling/slowrun"
LOGS = os.path.join(REPO, "experiments/logs")
OUT = os.path.join(REPO, "experiments/figures/08_coord_check")
os.makedirs(OUT, exist_ok=True)

HDR = re.compile(r"COORDCHECK cell=(d\d+_w\d+) n_head=\d+ lr_multiplier=([\d.]+) seed=(\d+)")
VAL = re.compile(r"val @ step (\d+)\] val_loss=([\d.]+)")
# ladder order (small -> large) for consistent color/legend
CELL_ORDER = ["d6_w384", "d12_w768", "d18_w1152", "d24_w1536"]
CELL_SIZE = {"d6_w384": "0.5x", "d12_w768": "1x (base)", "d18_w1152": "1.5x", "d24_w1536": "2x"}


def collect():
    """nested dict cell -> lr -> list of (final_val, min_val) over seeds."""
    data = {}
    for f in sorted(glob.glob(f"{LOGS}/coordcheck_*.out")):
        txt = open(f).read()
        h = HDR.search(txt)
        if not h:
            continue
        cell, lr = h.group(1), float(h.group(2))
        vals = [(int(m.group(1)), float(m.group(2))) for m in VAL.finditer(txt)]
        if not vals:
            continue
        vals.sort()
        final_val = vals[-1][1]
        min_val = min(v for _, v in vals)
        data.setdefault(cell, {}).setdefault(lr, []).append((final_val, min_val))
    return data


def main():
    data = collect()
    cells = [c for c in CELL_ORDER if c in data]
    if not cells:
        print("No coordcheck logs with val data yet.")
        return

    sns.set(font_scale=1.5); sns.set_style("whitegrid")
    plt.rcParams.update({"axes.labelsize": 24, "axes.linewidth": 4.0, "legend.fontsize": 16,
                         "grid.alpha": 0.25, "xtick.labelsize": 18, "ytick.labelsize": 18})
    colors = sns.color_palette("cool", len(cells))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(20, 7.5))
    print(f"{'cell':>11} {'argmin lr':>9}  curve (lr: mean_final_val +/- std)")
    argmins = {}
    for i, cell in enumerate(cells):
        lrs = sorted(data[cell])
        mean = np.array([np.mean([v[0] for v in data[cell][lr]]) for lr in lrs])
        std = np.array([np.std([v[0] for v in data[cell][lr]]) for lr in lrs])
        lrs = np.array(lrs)
        argmin_lr = lrs[int(np.argmin(mean))]
        argmins[cell] = argmin_lr

        # A: absolute
        axA.plot(lrs, mean, "o-", lw=3.0, ms=9, color=colors[i],
                 label=f"{cell}  ({CELL_SIZE[cell]})")
        axA.fill_between(lrs, mean - std, mean + std, color=colors[i], alpha=0.12)
        # B: normalized to own min
        axB.plot(lrs, mean - mean.min(), "o-", lw=3.0, ms=9, color=colors[i],
                 label=f"{cell}: argmin={argmin_lr:g}")
        axB.fill_between(lrs, mean - mean.min() - std, mean - mean.min() + std,
                         color=colors[i], alpha=0.12)
        axB.axvline(argmin_lr, color=colors[i], lw=1.5, ls=":", alpha=0.6)
        print(f"{cell:>11} {argmin_lr:>9g}  " +
              " ".join(f"{lr:g}:{m:.3f}±{s:.3f}" for lr, m, s in zip(lrs, mean, std)))

    for ax in (axA, axB):
        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"lr\_multiplier")
        ax.set_xticks([0.0625, 0.125, 0.25, 0.5, 1.0, 2.0])
        ax.set_xticklabels(["1/16", "1/8", "1/4", "1/2", "1", "2"])
    axA.set_ylabel("val loss (end of 1 epoch)")
    axA.set_title("(A)  absolute  —  larger model = lower loss (expected)", fontsize=18, loc="left")
    axA.legend(fontsize=14)
    axB.set_ylabel(r"val loss $-$ per-size min")
    axB.set_title("(B)  normalized  —  do the minima align? (CompleteP transfer)", fontsize=18, loc="left")
    axB.legend(fontsize=14, title="dotted = argmin")

    aligned = len(set(argmins.values())) == 1
    verdict = (f"LR optimum CONSTANT across sizes at lr_multiplier={list(argmins.values())[0]:g}"
               if aligned else
               f"LR optimum varies: {{{', '.join(f'{c}:{argmins[c]:g}' for c in cells)}}}")
    fig.suptitle("CompleteP LR-transfer coordinate check  (w/L=64 ladder, 1 epoch df=0.2, wd=0)  —  "
                 + verdict, fontsize=17, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/coord_check_lr_transfer.{ext}", bbox_inches="tight", dpi=300)
    print(f"\n{verdict}\nSaved {OUT}/coord_check_lr_transfer.pdf/.png")


if __name__ == "__main__":
    main()
