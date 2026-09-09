"""How does validation loss rise after its minimum -- power law or exponential?

Blake's question in the meeting. For every curve in both rows of Figure 1 we
take the post-nadir excess  e(s) = L(s) - L*  as a function of the distance
past the optimum  ds = s - s*,  and fit two straight lines in log e:

    power law:     log e = gamma * log ds + b        (straight on log-log)
    exponential:   log e = kappa * ds     + b        (straight on semi-log)

The better R^2 says which form the rise follows; gamma says how fast.

Fitting window: points with ds > 0 and e >= EXCESS_MIN (0.03 nats), so the
per-epoch sawtooth near the minimum does not dominate the log. Curves with
fewer than MIN_PTS such points (the rise barely started before the run ended)
are reported but not fitted. Val curves are used raw, exactly as in Figure 1.

Data:
  row 1  data_export/expt4_datasize/wd0_fixed_tokens/  (fixed L=12,W=768; P varies;
         strategies averaged as in Figure 1)
  row 2  the 12 model-size cells from experiments/logs/  (fixed P=100M; N varies)

Outputs:
  experiments/figures/12_blowup_rate/expt_blowup_rate.{pdf,png}
  experiments/figures/12_blowup_rate/expt_blowup_rate_fits.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from expt_fig1_sister_panels import (  # noqa: E402
    BATCH_SIZE, CELLS, CORPUS_TOKENS, find_nadir, load_cell, load_datasize_runs, setup_style,
)

REPO = HERE.parents[1]
OUTDIR = REPO / "experiments" / "figures" / "12_blowup_rate"
EXCESS_MIN = 0.03
MIN_PTS = 10
R2_MIN = 0.5      # a fit that explains under half the variance is not reported as a verdict


def fit_forms(ds, e):
    """Return dict with gamma, R2_pow, kappa, R2_exp on the masked window."""
    m = (ds > 0) & (e >= EXCESS_MIN)
    if m.sum() < MIN_PTS:
        return dict(n=int(m.sum()))
    x_pow, x_exp, y = np.log(ds[m]), ds[m], np.log(e[m])

    def r2(x):
        a, b = np.polyfit(x, y, 1)
        pred = a * x + b
        return float(a), float(b), float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    g, bp, r2p = r2(x_pow)
    k, be, r2e = r2(x_exp)
    out = dict(n=int(m.sum()), gamma=g, b_pow=bp, R2_pow=r2p, kappa=k, b_exp=be, R2_exp=r2e,
               ds_lo=float(ds[m].min()), ds_hi=float(ds[m].max()))
    out["reliable"] = max(r2p, r2e) >= R2_MIN
    return out


def row1_curves():
    runs = load_datasize_runs()
    out = []
    for df in sorted(runs):
        P = CORPUS_TOKENS * df
        ref = runs[df][0]["tokens"]
        vls = [np.interp(ref, e["tokens"], e["val_loss"]) if not (len(e["tokens"]) == len(ref)
               and np.allclose(e["tokens"], ref)) else e["val_loss"] for e in runs[df]]
        v = np.stack(vls).mean(0)
        out.append(dict(label=fr"$P$={int(P/1e6)}M", resource=P, steps=ref / BATCH_SIZE, val=v))
    return out


def row2_curves():
    out = []
    for k, pat in sorted(CELLS.items(), key=lambda kv: 16 * kv[0][0] * kv[0][1] ** 2):
        c = load_cell(pat)
        if c is None:
            continue
        out.append(dict(label=fr"$L${k[0]}/$W${k[1]}", resource=16 * k[0] * k[1] ** 2,
                        steps=c["val_steps"].astype(float), val=c["val"]))
    return out


def analyse(curves):
    res = []
    for c in curves:
        i, s_star, l_star = find_nadir(c["steps"], c["val"])
        ds = c["steps"][i:] - s_star
        e = c["val"][i:] - l_star
        f = fit_forms(ds, e)
        res.append(dict(**c, s_star=s_star, l_star=l_star, ds=ds, e=e, **f))
    return res


def main():
    setup_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    R1 = analyse(row1_curves())
    R2 = analyse(row2_curves())

    fig, axes = plt.subplots(2, 3, figsize=(30, 17))
    fig.subplots_adjust(wspace=0.28, hspace=0.34)
    for row, (R, rname, xlab) in enumerate(((R1, r"vary $P$ ($L$=12, $W$=768)", r"unique tokens $P$"),
                                            (R2, r"vary $N$ ($P$=100M)", r"parameters $N=16LW^2$"))):
        pal = sns.color_palette("cool", len(R))
        fitted = [r for r in R if r.get("reliable")]
        # (left) log-log
        ax = axes[row, 0]
        for col, r in zip(pal, R):
            m = (r["ds"] > 0) & (r["e"] > 0)
            ax.plot(r["ds"][m], r["e"][m], "-", color=col, lw=2.2, alpha=0.9, label=r["label"])
            if r.get("reliable"):
                g = np.logspace(np.log10(r["ds_lo"]), np.log10(r["ds_hi"]), 50)
                ax.plot(g, np.exp(r["b_pow"]) * g ** r["gamma"], ":", color="black", lw=1.6, alpha=0.7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"steps past the optimum  $s-s^\ast$", fontsize=24)
        ax.set_ylabel(r"excess loss  $\mathcal{L}-\mathcal{L}^\ast$", fontsize=24)
        ax.set_title(f"({'AD'[row]})  {rname}: log-log  (dotted = power-law fit)", fontsize=19, loc="left")
        ax.legend(fontsize=11, ncol=2, loc="lower right", frameon=True, framealpha=0.9)
        # (middle) semi-log
        ax = axes[row, 1]
        for col, r in zip(pal, R):
            m = (r["ds"] > 0) & (r["e"] > 0)
            ax.plot(r["ds"][m], r["e"][m], "-", color=col, lw=2.2, alpha=0.9)
            if r.get("reliable"):
                g = np.linspace(r["ds_lo"], r["ds_hi"], 50)
                ax.plot(g, np.exp(r["b_exp"] + r["kappa"] * g), ":", color="black", lw=1.6, alpha=0.7)
        ax.set_yscale("log")
        ax.set_xlabel(r"steps past the optimum  $s-s^\ast$", fontsize=24)
        ax.set_ylabel(r"excess loss  $\mathcal{L}-\mathcal{L}^\ast$", fontsize=24)
        ax.set_title(f"({'BE'[row]})  same curves, semi-log  (dotted = exponential fit)", fontsize=19, loc="left")
        # (right) exponent and fit quality vs resource
        ax = axes[row, 2]
        xs = np.array([r["resource"] for r in fitted], float)
        ax.plot(xs, [r["gamma"] for r in fitted], "-o", color="black", lw=2.6, ms=11,
                markeredgecolor="black", label=r"power-law exponent $\gamma$")
        ax.axhline(1.0, color="0.5", ls=":", lw=1.8)
        ax.set_xscale("log")
        ax.set_xlabel(xlab, fontsize=24)
        ax.set_ylabel(r"$\gamma$  in  $\mathcal{L}-\mathcal{L}^\ast\propto(s-s^\ast)^\gamma$", fontsize=22)
        ax2 = ax.twinx()
        ax2.plot(xs, [r["R2_pow"] for r in fitted], "s--", color="0.35", lw=2.0, ms=9, label=r"$R^2$ power law")
        ax2.plot(xs, [r["R2_exp"] for r in fitted], "^--", color="0.65", lw=2.0, ms=9, label=r"$R^2$ exponential")
        ax2.set_ylabel(r"fit quality $R^2$  (open axis)", color="0.35", fontsize=22)
        ax2.set_ylim(0.5, 1.02)
        ax2.grid(False)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower left", frameon=True, framealpha=0.92, fontsize=14)
        fmt = (lambda x, _: f"{x/1e6:.0f}M")
        ax.set_xticks([1e7, 2e7, 5e7, 1e8] if row == 0 else [2e7, 5e7, 1e8, 2e8, 5e8])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_title(f"({'CF'[row]})  exponent and which form fits  (reliable fits only)", fontsize=19, loc="left")

    for ext in ("pdf", "png"):
        p = OUTDIR / f"expt_blowup_rate.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"saved {p}")
    plt.close(fig)

    with open(OUTDIR / "expt_blowup_rate_fits.csv", "w") as fh:
        fh.write("row,curve,resource,nadir_step,min_val,n_fit_points,gamma,R2_powerlaw,kappa_per_step,R2_exponential,verdict\n")
        for rname, R in (("vary_P", R1), ("vary_N", R2)):
            for r in R:
                if "gamma" in r:
                    verdict = ("power" if r["R2_pow"] > r["R2_exp"] else "exponential") if r["reliable"] else "unreliable"
                    fh.write(f"{rname},{r['label'].replace('$','')},{r['resource']:.0f},{r['s_star']:.0f},"
                             f"{r['l_star']:.4f},{r['n']},{r['gamma']:.4f},{r['R2_pow']:.4f},"
                             f"{r['kappa']:.3e},{r['R2_exp']:.4f},{verdict}\n")
                else:
                    fh.write(f"{rname},{r['label'].replace('$','')},{r['resource']:.0f},{r['s_star']:.0f},"
                             f"{r['l_star']:.4f},{r['n']},,,,,too few points\n")
    print(f"saved {OUTDIR / 'expt_blowup_rate_fits.csv'}")

    for rname, R in (("row 1: vary P", R1), ("row 2: vary N", R2)):
        print(f"\n{rname}")
        print(f"{'curve':>12} {'n':>3} {'gamma':>7} {'R2 pow':>7} {'R2 exp':>7}  verdict")
        for r in R:
            lab = r["label"].replace("$", "")
            if "gamma" in r:
                v = ("power" if r["R2_pow"] > r["R2_exp"] else "EXPONENTIAL") if r["reliable"] else "unreliable (R2<0.5)"
                print(f"{lab:>12} {r['n']:>3} {r['gamma']:>7.3f} {r['R2_pow']:>7.3f} {r['R2_exp']:>7.3f}  {v}")
            else:
                print(f"{lab:>12} {r['n']:>3}   (too few post-nadir points to fit)")


if __name__ == "__main__":
    main()
