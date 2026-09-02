"""Val loss vs parameter count (df=1.0 full data), E=1 individual models.

For every full-data (depth x width) cell we have individual training for, plot the
minimum val loss L* (best point over 40 epochs, E=1) against the model's parameter
count. Points are grouped into depth series (d6, d12) and connected so the width-scaling
trend is visible and matched-parameter depth comparisons read off directly.

Cells (df=1.0, d12/w768 base, completep, no-ve-projs, no-warmdown, wd=0):
  d6 : w384, w768                       (fulldata grid)
  d12: w384, w768                       (fulldata grid)
  d12: w1152, w1536                     (width extension, single init model)

L* per cell = min over training of the mean-over-individuals val curve.
  base cells      : mean over ALL individuals present (both strategies, up to 10)
  width-extension : the single init model (E=1)
Parsed from SLURM .out logs; total_params read from each cell's config.json.
"""
import os, re, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO = "/ocean/projects/cis260161p/ymiao6/scaling/slowrun"
LOGS = os.path.join(REPO, "experiments/logs")
C161 = os.path.join(REPO, "checkpoints")
C095 = "/ocean/projects/cis260095p/ymiao6/scaling/slowrun/checkpoints"
IND_LINE = re.compile(r"\[model \d+ val @ step (\d+)\] val_loss=([\d.]+)")

# cell -> (depth, width, log-glob pattern, config.json path)
CELLS = {
    "d6_w384":   (6,  384,  f"{LOGS}/fd_train_d6_w384_*_[0-9].out",
                  f"{C161}/parallel_init_ens_fulldata_20260528_150057_d6_w384/config.json"),
    "d6_w768":   (6,  768,  f"{LOGS}/fd_train_d6_w768_*_[0-9].out",
                  f"{C161}/parallel_init_ens_fulldata_20260528_150057_d6_w768/config.json"),
    "d12_w384":  (12, 384,  f"{LOGS}/fd_train_d12_w384_*_[0-9].out",
                  f"{C161}/parallel_init_ens_fulldata_20260528_150057_d12_w384/config.json"),
    "d12_w768":  (12, 768,  f"{LOGS}/fd_train_d12_w768_*_[0-9].out",
                  f"{C161}/parallel_init_ens_fulldata_20260528_150057_d12_w768/config.json"),
    "d12_w1152": (12, 1152, f"{LOGS}/fd_widthext_d12_w1152_*_0.out",
                  f"{C095}/parallel_init_ens_fulldata_widthext_20260618_d12_w1152/config.json"),
    "d12_w1536": (12, 1536, f"{LOGS}/fd_widthext_d12_w1536_*_0.out",
                  f"{C095}/parallel_init_ens_fulldata_widthext_20260618_d12_w1536/config.json"),
    # 2026-07-06 grid-fill (single init model each)
    "d6_w1152":  (6,  1152, f"{LOGS}/fd_gridfill_d6_w1152_*_0.out",
                  f"{C095}/parallel_init_ens_fulldata_gridfill_20260706_d6_w1152/config.json"),
    "d6_w1536":  (6,  1536, f"{LOGS}/fd_gridfill_d6_w1536_*_0.out",
                  f"{C095}/parallel_init_ens_fulldata_gridfill_20260706_d6_w1536/config.json"),
    "d18_w768":  (18, 768,  f"{LOGS}/fd_gridfill_d18_w768_*_0.out",
                  f"{C095}/parallel_init_ens_fulldata_gridfill_20260706_d18_w768/config.json"),
    "d24_w768":  (24, 768,  f"{LOGS}/fd_gridfill_d24_w768_*_0.out",
                  f"{C095}/parallel_init_ens_fulldata_gridfill_20260706_d24_w768/config.json"),
}


def per_model_curves(pattern):
    """Group val measurements by model index (across resume segments), one curve each."""
    permodel = {}   # model_idx -> {step: val}
    for f in sorted(glob.glob(pattern)):
        m = re.search(r"_(\d+)\.out$", f)
        midx = int(m.group(1))
        with open(f) as fh:
            for line in fh:
                mm = IND_LINE.search(line)
                if mm:
                    permodel.setdefault(midx, {})[int(mm.group(1))] = float(mm.group(2))
    curves = []
    for d in permodel.values():
        if d:
            s = np.array(sorted(d))
            curves.append((s, np.array([d[k] for k in s])))
    return curves


def lstar(curves):
    """min over training of the mean-over-individuals curve, on their common step range."""
    if not curves:
        return np.nan, 0
    lo = max(c[0][0] for c in curves)
    hi = min(c[0][-1] for c in curves)
    grid = curves[0][0]
    grid = grid[(grid >= lo) & (grid <= hi)]
    stacked = np.vstack([np.interp(grid, s, v) for s, v in curves])
    return float(np.min(stacked.mean(0))), len(curves)


def main():
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 24, "axes.linewidth": 4.0, "legend.fontsize": 18,
        "grid.alpha": 0.25, "xtick.labelsize": 20, "ytick.labelsize": 20,
    })

    rows = []   # (cell, depth, width, params, Lstar, n_indiv)
    for cell, (d, w, pat, cfg) in CELLS.items():
        params = json.load(open(cfg))["model"]["total_params"]
        L, n = lstar(per_model_curves(pat))
        rows.append((cell, d, w, params, L, n))
    rows.sort(key=lambda r: r[3])

    depths = sorted(set(r[1] for r in rows))
    palette = sns.color_palette("cool", len(depths))
    dcolor = {d: palette[i] for i, d in enumerate(depths)}

    fig, ax = plt.subplots(figsize=(11, 8))
    for d in depths:
        pts = sorted([r for r in rows if r[1] == d], key=lambda r: r[3])
        xs = [r[3] / 1e6 for r in pts]
        ys = [r[4] for r in pts]
        ax.plot(xs, ys, "o-", lw=3.0, ms=13, color=dcolor[d],
                markeredgecolor="black", markeredgewidth=0.6,
                label=fr"depth $L={d}$")
        for r, x, y in zip(pts, xs, ys):
            ax.annotate(f"w={r[2]}", (x, y), textcoords="offset points",
                        xytext=(0, 12), fontsize=13, ha="center", color=dcolor[d])

    ax.set_xscale("log")
    ax.set_xlabel(r"parameters  $N$  (millions)")
    ax.set_ylabel(r"min val loss  $\mathcal{L}^*$  (E=1)")
    ax.set_title("Val loss vs model size  (df=1.0, 100M tokens, wd=0)",
                 fontsize=20, loc="left")
    ax.legend(title="fixed depth, vary width")

    fig.tight_layout()
    outdir = os.path.join(REPO, "experiments/figures/09_val_vs_params")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "val_vs_params.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)

    print(f"{'cell':>11} {'depth':>5} {'width':>5} {'params(M)':>10} {'L*':>8} {'n_indiv':>7}")
    for cell, d, w, p, L, n in rows:
        print(f"{cell:>11} {d:>5} {w:>5} {p/1e6:>10.1f} {L:>8.4f} {n:>7}")
    print(f"\nSaved {out}\nSaved {out.replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
