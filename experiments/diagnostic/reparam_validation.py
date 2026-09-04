"""Comprehensive validation of the x0/U-Net depth fix: does it deliver transfer,
and does it cost training performance?

Reads the CSVs produced by the local sweeps (see the commit that added this file
for how they were generated) and renders the summary figure. Three variants, all
with IDENTICAL learning rates / weight decay / betas -- only the functional form
of the two extra residual paths differs:

  broken   no depth correction on x0 or the U-Net skips (pre-fix behaviour)
  constant effective = (L_base/L)^expo * raw_learnable   [what is committed]
  logexp   effective = c0 * exp(raw_learnable), c0 depth-scaled  [rejected]

FINDINGS
--------
1. Transfer (coordinate check, the standard muP diagnostic). Across a 32x depth
   ladder at production lr, 200 steps, 3 seeds -- residual RMS max/min:
       broken   17.9 -> 39.2   catastrophic AND growing
       constant  1.14 -> 1.31  plateaus
       logexp    1.14 -> 1.62  better before ~step 100, worse after, still rising
   The fix delivers a ~30x improvement in depth invariance over broken. It does
   NOT reach 1.00: ~1.3 is the residual, and the plain architecture with both
   paths removed reaches ~1.02, so ~0.3 is the standing price of keeping x0 and
   the U-Net skips at all.

2. Performance. 3 variants x 4 depths x 11 lrs x 4 seeds = 528 training runs,
   best-vs-best (each arm at its OWN optimal lr): the largest gap between any
   two variants at any depth is 0.0036 nats. The fix does not impair training.

CAVEAT, stated plainly: the loss-based half of this is weak evidence, because
every task small enough to run locally saturates. On the bigram task all depths
and all variants converge to ~1.78 regardless, so "no performance difference"
partly reflects "nothing could differ here". A loss-vs-lr HP-transfer test was
attempted four times and is inconclusive at this scale for the same reason --
the lr optimum is set by the stability edge, not by the parameterization. The
coordinate check is the load-bearing evidence; it has real power because its
effect (17.9 vs 1.14) dwarfs the ~0.02 seed noise floor.
"""
from __future__ import annotations

import sys
import glob
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "experiments" / "figures" / "08_coord_check"
SC = sys.argv[1] if len(sys.argv) > 1 else \
    "/private/tmp/claude-501/-Users-miaoyidi-Desktop-pretraining/124e307c-a54e-495e-9e13-8fa59adb3ccf/scratchpad"

NAMES = {"broken": "broken", "linear": "constant", "logexp": "log/exp"}
ORDER = ["broken", "linear", "logexp"]

# ---- coordinate check (production lr) ----
coord = defaultdict(list)
for f in glob.glob(f"{SC}/coord_prod/*.csv"):
    for r in csv.DictReader(open(f)):
        sz = int(r["depth"]) if r["ladder"] == "depth" else int(r["width"])
        coord[(r["variant"], r["ladder"], sz, int(r["step"]))].append(float(r["rms"]))
steps = sorted({k[3] for k in coord})
DEPTHS = (2, 4, 8, 16, 32, 64)
WIDTHS = (32, 64, 128, 256, 512)


def spread(variant, ladder, sizes, step):
    vals = [np.mean(coord[(variant, ladder, s, step)]) for s in sizes]
    return max(vals) / min(vals)


# ---- performance ----
perf = defaultdict(list)
for sub in ("shards", "ext", "ext2"):
    for f in glob.glob(f"{SC}/{sub}/*.csv"):
        for r in csv.DictReader(open(f)):
            perf[(r["variant"], int(r["depth"]), float(r["lr_mult"]))].append(float(r["loss"]))
lrs = sorted({k[2] for k in perf})
PDEPTHS = (4, 8, 16, 32)

sns.set(font_scale=1.25)
sns.set_style("whitegrid")
plt.rcParams["axes.linewidth"] = 3.0
plt.rcParams["grid.alpha"] = 0.25
cols = dict(zip(ORDER, sns.color_palette("cool", 3)))
fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.8))

ax = axes[0]
for v in ORDER:
    ax.plot(steps, [spread(v, "depth", DEPTHS, s) for s in steps], "-o",
            color=cols[v], lw=3.0, ms=7, label=NAMES[v])
ax.axhline(1.0, color="0.3", ls="--", lw=2.0)
ax.set_yscale("log")
ax.set_xlabel("training step")
ax.set_ylabel("residual RMS max/min across depth")
ax.set_title("(A) depth transfer, $L\\in\\{2..64\\}$ (32$\\times$)\nproduction lr, 3 seeds", fontsize=14)
ax.legend(frameon=True)

ax = axes[1]
for v in ORDER:
    ax.plot(steps, [spread(v, "width", WIDTHS, s) for s in steps], "-o",
            color=cols[v], lw=3.0, ms=7, label=NAMES[v])
ax.axhline(1.0, color="0.3", ls="--", lw=2.0)
ax.set_xlabel("training step")
ax.set_ylabel("residual RMS max/min across width")
ax.set_title("(B) width transfer, $N\\in\\{32..512\\}$ (16$\\times$)\n"
             "broken $\\equiv$ constant here by construction", fontsize=14)
ax.legend(frameon=True)

ax = axes[2]
xs = np.arange(len(PDEPTHS))
w = 0.8 / len(ORDER)
for k, v in enumerate(ORDER):
    best, errs = [], []
    for d in PDEPTHS:
        cand = [(np.mean(perf[(v, d, lr)]),
                 np.std(perf[(v, d, lr)], ddof=1) / np.sqrt(len(perf[(v, d, lr)])))
                for lr in lrs if perf.get((v, d, lr))]
        m, e = min(cand, key=lambda t: t[0])
        best.append(m); errs.append(e)
    ax.bar(xs + (k - 1) * w, best, width=w, yerr=errs, capsize=4,
           color=cols[v], edgecolor="0.25", lw=1.8, label=NAMES[v])
ax.set_xticks(xs); ax.set_xticklabels([f"d{d}" for d in PDEPTHS])
ax.set_ylabel("best loss (own optimal lr)")
ax.set_ylim(1.770, 1.790)
ax.set_title("(C) performance: best-vs-best\nall variants within 0.0036 nats", fontsize=14)
ax.legend(frameon=True, fontsize=11)

fig.suptitle("Validating the x0 / U-Net depth fix: transfer improves ~30$\\times$; "
             "training performance is unaffected", fontsize=16)
fig.tight_layout(rect=(0, 0, 1, 0.91))
OUTDIR.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    p = OUTDIR / f"reparam_validation.{ext}"
    fig.savefig(p, bbox_inches="tight", dpi=300)
    print(f"Saved {p}")

with open(OUTDIR / "reparam_validation.csv", "w", newline="") as fh:
    w_ = csv.writer(fh)
    w_.writerow(["section", "variant", "ladder", "size", "step", "value"])
    for (v, lad, sz, st), vals in sorted(coord.items()):
        w_.writerow(["coord", v, lad, sz, st, f"{np.mean(vals):.6f}"])
    for (v, d, lr), vals in sorted(perf.items()):
        w_.writerow(["perf", v, "lr", d, lr, f"{np.mean(vals):.6f}"])
print(f"Saved {OUTDIR/'reparam_validation.csv'}")

print("\ndepth-ladder spread (max/min), production lr:")
print("  variant    " + "".join(f"{'st'+str(s):>8}" for s in steps))
for v in ORDER:
    print(f"  {NAMES[v]:<10} " + "".join(f"{spread(v,'depth',DEPTHS,s):>8.2f}" for s in steps))
