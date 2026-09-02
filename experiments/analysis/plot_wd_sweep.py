#!/usr/bin/env python
"""Weight-decay x multi-epoch-overfit sweeps (d12/w768, init_shuffle, 40 epochs).

TWO arms, distinguished by SLURM job-id (filenames share the wd{wd}_train_... prefix
so job-id filtering is mandatory):
  OFF = constant LR / no-warmdown  (wdsweep_20260612, jobs 41363*),  wd {0.1,0.3,1.0}
  ON  = cooldown                   (wdcooldown_20260615, jobs 41431*), wd {0.1,0.3,0.5,0.8}
       cooldown = constant LR for the first 32 epochs, cosine to 0 over the final ~8.

Outputs to experiments/figures/06_wd_cooldown/:
  wd_sweep_combo[_on].{pdf,png}     3-panel combo per arm
  wd_ensemble_gain[_on].{pdf,png}   ensembling gain (indiv mean - E=5) vs wd per arm
  wd_compare_on_vs_off.{pdf,png}    ON vs OFF: matched-wd curves + min-loss wd response

x-axis = cumulative tokens_seen (per-model, counts multi-epoch repeats) per project
convention. Parsed from .out logs (offline wandb fragmented), not wandb.
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
TOK_PER_EPOCH = 99_614_720
COOLDOWN_EPOCH = 32                      # cooldown onset (last 20% of 40 epochs)

# arm -> {wd: (train_jobid, replay_jobid)}
ARMS = {
    "off": {"label": "no cooldown", "ls": "--", "marker": "o",
            "wds": [0.1, 0.3, 1.0],
            "jobs": {0.1: (41363426, 41363427), 0.3: (41363428, 41363429),
                     1.0: (41363430, 41363431)}},
    "on":  {"label": "cooldown", "ls": "-", "marker": "s",
            "wds": [0.1, 0.3, 0.5, 0.8],
            "jobs": {0.1: (41431931, 41431932), 0.3: (41431933, 41431934),
                     0.5: (41431935, 41431936), 0.8: (41431937, 41431938)}},
}

IND_RE = re.compile(r"\[model \d+ val @ step (\d+)\] val_loss=([\d.]+)")
ENS_RE = re.compile(r"\[step \d+ ens=(\d+)\] val_loss=([\d.]+) val_bpb=([\d.]+) tokens=(\d+)")


# ---------- parse ----------
def parse_individuals(arm, wd):
    """dict model_idx -> (tokens[], val[]), filtered to this arm's train job-id."""
    jid = ARMS[arm]["jobs"][wd][0]
    per_model = {}
    for f in sorted(glob.glob(f"{LOGS}/wd{wd}_train_d12_w768_{jid}_*.out")):
        arr = int(re.search(r"_(\d+)\.out$", f).group(1))
        d = per_model.setdefault(arr % NUM_MODELS, {})
        with open(f) as fh:
            for line in fh:
                m = IND_RE.search(line)
                if m:
                    d[int(m.group(1)) * BATCH] = float(m.group(2))
    return {mi: (np.array(sorted(d)), np.array([d[t] for t in sorted(d)]))
            for mi, d in per_model.items() if d}


def parse_ensembles(arm, wd):
    """dict E -> (tokens[], val[], bpb[]), filtered to this arm's replay job-id."""
    jid = ARMS[arm]["jobs"][wd][1]
    per_E = {}
    for f in sorted(glob.glob(f"{LOGS}/wd{wd}_replay_d12_w768_{jid}_*.out")):
        with open(f) as fh:
            for line in fh:
                m = ENS_RE.search(line)
                if m:
                    per_E.setdefault(int(m.group(1)), {})[int(m.group(4))] = \
                        (float(m.group(2)), float(m.group(3)))
    out = {}
    for E, d in per_E.items():
        t = np.array(sorted(d))
        out[E] = (t, np.array([d[x][0] for x in t]), np.array([d[x][1] for x in t]))
    return out


def ind_mean_std(ind):
    grid = ind[0][0]
    stacked = np.vstack([np.interp(grid, t, v) for t, v in ind.values()])
    return grid, stacked.mean(0), stacked.std(0)


# load everything once
DATA = {arm: {wd: {"ind": parse_individuals(arm, wd), "ens": parse_ensembles(arm, wd)}
              for wd in ARMS[arm]["wds"]} for arm in ARMS}


# ---------- style ----------
sns.set(font_scale=1.5); sns.set_style("whitegrid")
plt.rcParams.update({"axes.labelsize": 24, "axes.linewidth": 4.0, "legend.fontsize": 16,
                     "grid.alpha": 0.25, "xtick.labelsize": 18, "ytick.labelsize": 18})


# ===== per-arm 3-panel combo =====
def make_combo(arm):
    wds = ARMS[arm]["wds"]
    colors = sns.color_palette("cool", len(wds))
    suf = "_on" if arm == "on" else ""
    fig, axes = plt.subplots(1, 3, figsize=(27, 7.5))

    # A: individual mean +/- band per wd
    for i, wd in enumerate(wds):
        g, mu, sd = ind_mean_std(DATA[arm][wd]["ind"])
        axes[0].plot(g, mu, lw=3.0, color=colors[i], label=f"wd = {wd}")
        axes[0].fill_between(g, mu - sd, mu + sd, color=colors[i], alpha=0.12)
    axes[0].set_title("Individual models (mean $\\pm$ std, N=5)", fontsize=20)

    # B: ensemble E=5 (solid) vs indiv mean (dashed) per wd
    for i, wd in enumerate(wds):
        t5, v5, _ = DATA[arm][wd]["ens"][5]
        axes[1].plot(t5, v5, lw=3.0, color=colors[i], label=f"wd = {wd}")
        g, mu, _ = ind_mean_std(DATA[arm][wd]["ind"])
        axes[1].plot(g, mu, "--", lw=2.0, color=colors[i], alpha=0.6)
    axes[1].set_title("Ensemble E=5 (solid) vs indiv. mean (dashed)", fontsize=20)

    # C: ensemble-size sweep at the strongest wd in this arm
    wd_focus = max(wds)
    Es = sorted(DATA[arm][wd_focus]["ens"])
    ecol = sns.color_palette("cool", len(Es) + 1)
    g, mu, _ = ind_mean_std(DATA[arm][wd_focus]["ind"])
    axes[2].plot(g, mu, "--", lw=2.5, color="black", label="indiv. mean")
    for j, E in enumerate(Es):
        t, v, _ = DATA[arm][wd_focus]["ens"][E]
        axes[2].plot(t, v, lw=3.0, color=ecol[j], label=f"E = {E}")
    axes[2].set_title(f"Ensemble size sweep (wd = {wd_focus})", fontsize=20)

    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("tokens seen (cumulative)")
        ax.set_ylabel("val loss")
        if arm == "on":
            ax.axvline(COOLDOWN_EPOCH * TOK_PER_EPOCH, color="gray", lw=1.5, ls=":")
    axes[0].legend(title="weight decay"); axes[1].legend(title="weight decay"); axes[2].legend(ncol=2)

    title = ("Weight decay $\\times$ multi-epoch overfit  —  d12 w768, init_shuffle, 40 epochs"
             f"  [{ARMS[arm]['label']}]")
    fig.suptitle(title, fontsize=22, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/wd_sweep_combo{suf}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote wd_sweep_combo{suf}.pdf/.png")

    # ensemble gain
    fig2, ax = plt.subplots(figsize=(10, 7.5))
    for i, wd in enumerate(wds):
        t5, v5, _ = DATA[arm][wd]["ens"][5]
        g, mu, _ = ind_mean_std(DATA[arm][wd]["ind"])
        ax.plot(t5, np.interp(t5, g, mu) - v5, lw=3.0, color=colors[i], label=f"wd = {wd}")
    ax.axhline(0, color="black", lw=2.0, ls=":")
    if arm == "on":
        ax.axvline(COOLDOWN_EPOCH * TOK_PER_EPOCH, color="gray", lw=1.5, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("tokens seen (cumulative)")
    ax.set_ylabel("indiv mean $-$ ensemble  (val loss)")
    ax.set_title(f"Ensembling gain vs weight decay (E=5) [{ARMS[arm]['label']}]", fontsize=18)
    ax.legend(title="weight decay")
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(f"{OUT}/wd_ensemble_gain{suf}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig2)
    print(f"wrote wd_ensemble_gain{suf}.pdf/.png")


# ===== ON vs OFF comparison =====
def make_comparison():
    matched = [0.1, 0.3]               # wd present in BOTH arms
    mcol = sns.color_palette("cool", len(matched))
    fig, axes = plt.subplots(1, 3, figsize=(27, 7.5))

    # Panel 1: individual mean, ON (solid) vs OFF (dashed) at matched wd
    for i, wd in enumerate(matched):
        for arm in ("off", "on"):
            g, mu, _ = ind_mean_std(DATA[arm][wd]["ind"])
            axes[0].plot(g, mu, ARMS[arm]["ls"], lw=3.0, color=mcol[i],
                         alpha=0.9 if arm == "on" else 0.55,
                         label=f"wd={wd} {ARMS[arm]['label']}")
    axes[0].set_title("Individual mean: ON (solid) vs OFF (dashed)", fontsize=20)

    # Panel 2: ensemble E=5, ON vs OFF at matched wd
    for i, wd in enumerate(matched):
        for arm in ("off", "on"):
            t5, v5, _ = DATA[arm][wd]["ens"][5]
            axes[1].plot(t5, v5, ARMS[arm]["ls"], lw=3.0, color=mcol[i],
                         alpha=0.9 if arm == "on" else 0.55,
                         label=f"wd={wd} {ARMS[arm]['label']}")
    axes[1].set_title("Ensemble E=5: ON (solid) vs OFF (dashed)", fontsize=20)

    for ax in axes[:2]:
        ax.axvline(COOLDOWN_EPOCH * TOK_PER_EPOCH, color="gray", lw=1.8, ls=":")
        ax.set_xscale("log"); ax.set_xlabel("tokens seen (cumulative)"); ax.set_ylabel("val loss")
        ax.legend(fontsize=14)
        ax.text(COOLDOWN_EPOCH * TOK_PER_EPOCH, ax.get_ylim()[1], " cooldown\n onset",
                fontsize=12, va="top", color="gray")

    # Panel 3: min val loss vs wd (the wd response), both arms, indiv + ensemble
    ax = axes[2]
    acol = {"off": sns.color_palette("rocket", 3)[1], "on": sns.color_palette("mako", 3)[1]}
    for arm in ("off", "on"):
        wds = ARMS[arm]["wds"]
        ind_min = [ind_mean_std(DATA[arm][wd]["ind"])[1].min() for wd in wds]
        ens_min = [DATA[arm][wd]["ens"][5][1].min() for wd in wds]
        ax.plot(wds, ind_min, ARMS[arm]["marker"] + "--", lw=2.0, ms=11, color=acol[arm], alpha=0.6,
                markeredgecolor="black", markeredgewidth=0.5,
                label=f"{ARMS[arm]['label']}: indiv (E=1)")
        ax.plot(wds, ens_min, ARMS[arm]["marker"] + "-", lw=3.0, ms=12, color=acol[arm],
                markeredgecolor="black", markeredgewidth=0.6,
                label=f"{ARMS[arm]['label']}: ensemble E=5")
    ax.set_xlabel("weight decay"); ax.set_ylabel("min val loss")
    ax.set_title("Best val loss vs weight decay", fontsize=20)
    ax.legend(fontsize=13)

    fig.suptitle("Cooldown vs no-cooldown weight-decay sweep  —  d12 w768, init_shuffle, 40 epochs",
                 fontsize=22, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/wd_compare_on_vs_off.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote wd_compare_on_vs_off.pdf/.png")


if __name__ == "__main__":
    make_combo("off")
    make_combo("on")
    make_comparison()

    print("\n=== min val loss (indiv mean / ensemble E=5) ===")
    print(f"{'arm':>12} {'wd':>5} {'indiv-min':>10} {'ens5-min':>10}")
    for arm in ("off", "on"):
        for wd in ARMS[arm]["wds"]:
            im = ind_mean_std(DATA[arm][wd]["ind"])[1].min()
            em = DATA[arm][wd]["ens"][5][1].min()
            print(f"{ARMS[arm]['label']:>12} {wd:>5} {im:>10.4f} {em:>10.4f}")
