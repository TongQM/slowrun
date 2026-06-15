#!/usr/bin/env python
"""Plot the weight-decay x multi-epoch-overfit sweep (d12_w768, init_shuffle, 40 epochs).

Parses individuals from wd{WD}_train_d12_w768_*.out (model idx = arr%5 + 1) and
ensembles from wd{WD}_replay_d12_w768_*.out. x-axis = cumulative tokens_seen
(per-model, counts multi-epoch repeats) per project convention.
"""
import re, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO = "/ocean/projects/cis260161p/ymiao6/scaling/slowrun"
LOGS = os.path.join(REPO, "experiments/logs")
OUT = os.path.join(REPO, "experiments/figures/06_wd_cooldown")
os.makedirs(OUT, exist_ok=True)
BATCH = 131072
NUM_MODELS = 5
WDS = [0.1, 0.3, 1.0]

# ---- parse ----
def parse_individuals(wd):
    """returns dict model_idx -> (tokens[], val_loss[]) merged over array tasks."""
    per_model = {}
    for f in sorted(glob.glob(f"{LOGS}/wd{wd}_train_d12_w768_*_*.out")):
        arr = int(re.search(r"_(\d+)\.out$", f).group(1))
        midx = arr % NUM_MODELS  # 0-based
        d = per_model.setdefault(midx, {})
        for m in re.finditer(r"\[model \d+ val @ step (\d+)\] val_loss=([\d.]+)", open(f).read()):
            step = int(m.group(1)); d[step * BATCH] = float(m.group(2))
    out = {}
    for midx, d in per_model.items():
        toks = np.array(sorted(d)); out[midx] = (toks, np.array([d[t] for t in toks]))
    return out

def parse_ensembles(wd):
    """returns dict E -> (tokens[], val_loss[], val_bpb[])."""
    per_E = {}
    for f in sorted(glob.glob(f"{LOGS}/wd{wd}_replay_d12_w768_*_*.out")):
        for m in re.finditer(r"\[step \d+ ens=(\d+)\] val_loss=([\d.]+) val_bpb=([\d.]+) tokens=(\d+)", open(f).read()):
            E = int(m.group(1)); per_E.setdefault(E, {})[int(m.group(4))] = (float(m.group(2)), float(m.group(3)))
    out = {}
    for E, d in per_E.items():
        toks = np.array(sorted(d))
        out[E] = (toks, np.array([d[t][0] for t in toks]), np.array([d[t][1] for t in toks]))
    return out

ind = {wd: parse_individuals(wd) for wd in WDS}
ens = {wd: parse_ensembles(wd) for wd in WDS}

# individual mean/std on a common token grid (use union of model grids per wd)
def ind_mean_std(wd):
    models = ind[wd]
    grid = models[0][0]  # all models share the step grid
    stacked = np.vstack([np.interp(grid, t, v) for t, v in models.values()])
    return grid, stacked.mean(0), stacked.std(0)

# ---- style (project convention) ----
sns.set(font_scale=1.5); sns.set_style("whitegrid")
plt.rcParams.update({
    "axes.labelsize": 24, "axes.linewidth": 4.0, "legend.fontsize": 16,
    "grid.alpha": 0.25, "xtick.labelsize": 18, "ytick.labelsize": 18,
})
wd_colors = sns.color_palette("cool", len(WDS))

# ===== Figure 1: 3-panel combo =====
fig, axes = plt.subplots(1, 3, figsize=(27, 7.5))

# Panel A: individual mean +/- band per WD
axA = axes[0]
for i, wd in enumerate(WDS):
    g, mu, sd = ind_mean_std(wd)
    axA.plot(g, mu, lw=3.0, color=wd_colors[i], label=f"wd = {wd}")
    axA.fill_between(g, mu - sd, mu + sd, color=wd_colors[i], alpha=0.12)
axA.set_xscale("log")
axA.set_xlabel("tokens seen (cumulative)")
axA.set_ylabel("val loss")
axA.set_title("Individual models (mean $\\pm$ std, N=5)", fontsize=20)
axA.legend(title="weight decay")

# Panel B: ensemble E=5 (solid) vs individual mean (dashed) per WD
axB = axes[1]
for i, wd in enumerate(WDS):
    t5, v5, _ = ens[wd][5]
    axB.plot(t5, v5, lw=3.0, color=wd_colors[i], label=f"wd = {wd}")
    g, mu, _ = ind_mean_std(wd)
    axB.plot(g, mu, "--", lw=2.0, color=wd_colors[i], alpha=0.6)
axB.set_xscale("log")
axB.set_xlabel("tokens seen (cumulative)")
axB.set_ylabel("val loss")
axB.set_title("Ensemble E=5 (solid) vs indiv. mean (dashed)", fontsize=20)
axB.legend(title="weight decay")

# Panel C: ensemble-size sweep at wd=1.0 (strongest regularization)
axC = axes[2]
wd_focus = 1.0
Es = sorted(ens[wd_focus])
e_colors = sns.color_palette("cool", len(Es) + 1)
g, mu, _ = ind_mean_std(wd_focus)
axC.plot(g, mu, "--", lw=2.5, color="black", label="indiv. mean")
for j, E in enumerate(Es):
    t, v, _ = ens[wd_focus][E]
    axC.plot(t, v, lw=3.0, color=e_colors[j], label=f"E = {E}")
axC.set_xscale("log")
axC.set_xlabel("tokens seen (cumulative)")
axC.set_ylabel("val loss")
axC.set_title(f"Ensemble size sweep (wd = {wd_focus})", fontsize=20)
axC.legend(ncol=2)

fig.suptitle("Weight decay $\\times$ multi-epoch overfit  —  d12 w768, init_shuffle, 40 epochs (100M tokens/epoch)",
             fontsize=22, y=1.02)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/wd_sweep_combo.{ext}", bbox_inches="tight", dpi=300)
print("wrote wd_sweep_combo.pdf/.png")

# ===== Figure 2: ensemble gain (E=5 minus indiv mean) per WD =====
fig2, ax = plt.subplots(figsize=(10, 7.5))
for i, wd in enumerate(WDS):
    t5, v5, _ = ens[wd][5]
    g, mu, _ = ind_mean_std(wd)
    gain = np.interp(t5, g, mu) - v5  # positive = ensemble better
    ax.plot(t5, gain, lw=3.0, color=wd_colors[i], label=f"wd = {wd}")
ax.axhline(0, color="black", lw=2.0, ls=":")
ax.set_xscale("log")
ax.set_xlabel("tokens seen (cumulative)")
ax.set_ylabel("indiv mean $-$ ensemble  (val loss)")
ax.set_title("Ensembling gain vs weight decay (E=5)", fontsize=20)
ax.legend(title="weight decay")
fig2.tight_layout()
for ext in ("pdf", "png"):
    fig2.savefig(f"{OUT}/wd_ensemble_gain.{ext}", bbox_inches="tight", dpi=300)
print("wrote wd_ensemble_gain.pdf/.png")

# ---- print summary table ----
print("\n=== best (min) val loss over training ===")
print(f"{'wd':>5} {'indiv-min(mean)':>16} {'@Btok':>7} {'ens5-min':>9} {'@Btok':>7} {'final-indiv':>12} {'final-ens5':>11}")
for wd in WDS:
    g, mu, _ = ind_mean_std(wd)
    t5, v5, _ = ens[wd][5]
    print(f"{wd:>5} {mu.min():>16.4f} {g[mu.argmin()]/1e9:>7.2f} {v5.min():>9.4f} "
          f"{t5[v5.argmin()]/1e9:>7.2f} {mu[-1]:>12.4f} {v5[-1]:>11.4f}")
