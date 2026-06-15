#!/usr/bin/env python
"""Val-loss curve for the single looped 1.8B model (d30 w2048, dupe 15-25, hybrid, 10 epochs)."""
import re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO = "/ocean/projects/cis260161p/ymiao6/scaling/slowrun"
LOG = os.path.join(REPO, "experiments/logs/single_1p8b_q0_20260612_021030_41363766.out")
OUT = os.path.join(REPO, "experiments/figures/07_single_1p8b")
os.makedirs(OUT, exist_ok=True)
BATCH = 131072
TOK_PER_EPOCH = 99_950_592
txt = open(LOG).read()

# per-step val: tokens = step * batch
step_t, step_v = [], []
for m in re.finditer(r"\[model 1 val @ step (\d+)\] val_loss=([\d.]+)", txt):
    step_t.append(int(m.group(1)) * BATCH); step_v.append(float(m.group(2)))
step_t, step_v = np.array(step_t), np.array(step_v)

# per-epoch val + bpb
ep_k, ep_v, ep_b = [], [], []
for m in re.finditer(r"\[model 1\] epoch (\d+) val_loss=([\d.]+) val_bpb=([\d.]+)", txt):
    ep_k.append(int(m.group(1))); ep_v.append(float(m.group(2))); ep_b.append(float(m.group(3)))
ep_t = np.array(ep_k) * TOK_PER_EPOCH

sns.set(font_scale=1.5); sns.set_style("whitegrid")
plt.rcParams.update({"axes.labelsize": 24, "axes.linewidth": 4.0, "legend.fontsize": 16,
                     "grid.alpha": 0.25, "xtick.labelsize": 18, "ytick.labelsize": 18})
col = sns.color_palette("cool", 4)

fig, ax = plt.subplots(figsize=(11, 7.5))
ax.plot(step_t, step_v, lw=2.0, color=col[0], alpha=0.55, label="per-step val")
ax.plot(ep_t, ep_v, "o-", lw=3.0, ms=9, color=col[2], label="per-epoch val")
imin = int(np.argmin(ep_v))
ax.axhline(ep_v[imin], color="black", lw=2.0, ls=":")
ax.annotate(f"min epoch {ep_k[imin]}: {ep_v[imin]:.3f}  (bpb {ep_b[imin]:.4f})",
            (ep_t[imin], ep_v[imin]), textcoords="offset points", xytext=(10, 12), fontsize=15)
ax.set_xscale("log")
ax.set_xlabel("tokens seen (cumulative)")
ax.set_ylabel("val loss")
ax.set_title("Single looped 1.8B  (d30 w2048, dupe 15-25, hybrid, no-warmdown)", fontsize=18)
ax.legend()
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/single_1p8b_valloss.{ext}", bbox_inches="tight", dpi=300)
print("wrote single_1p8b_valloss.pdf/.png")
print(f"\nepochs: {ep_k}")
print(f"val_loss: {[round(v,4) for v in ep_v]}")
print(f"val_bpb : {[round(b,4) for b in ep_b]}")
print(f"best: epoch {ep_k[imin]}  val_loss={ep_v[imin]:.4f}  bpb={ep_b[imin]:.4f}")
print(f"final epoch {ep_k[-1]}: val_loss={ep_v[-1]:.4f}  bpb={ep_b[-1]:.4f}")
