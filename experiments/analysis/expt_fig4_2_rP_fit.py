"""Expt Fig 4.2 — fit (or rather, attempt to fit) the data-limited exponent
r_P at the base cell, and characterize the identification problem honestly.

Premise: at the U-curve nadir, variance and bias-dynamics are at their minima,
so min_L over a run should approximately equal

    min_L(P) ≈ c_P · P^{-r_P} + L_0_eff

We have 4 P values (df ∈ {0.2, 0.3, 0.4, 0.5}) per strategy. Three free
parameters from 4 observations leaves only 1 dof per strategy — and the
optimum runs into a (c, r_P, L_0) degeneracy: small L_0 gives small r_P,
large L_0 (close to min_L) gives large r_P. Neither is identifiable from
the data alone.

So we report what *can* be said honestly:
  (A) overlay several r_P / L_0 pairs that all fit min_L well, to visualize
      the degeneracy band;
  (B) plot r_P as a function of *assumed* L_0 (the key sensitivity);
  (C) the onset position vs P, which delivers a clean answer to the second
      Expt 4 question ("is overfit-onset just total epochs?").

Saves to experiments/figures/04_TBD/expt_fig4_2_rP_fit.{pdf,png} and
expt_fig4_2_rP_fit_coefs.csv (which gives r_P at a grid of L_0 anchors).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.optimize import least_squares

from expt_fig4_loader import (
    DFS, STRATEGIES, STRAT_PRETTY, load_all, df_to_P, REPO,
)


OUT_DIR = REPO / "experiments" / "figures" / "04_TBD"
OUT_NAME = "expt_fig4_2_rP_fit"


def collect(runs, strat: str):
    P, L, ep, tok, df = [], [], [], [], []
    for d_ in DFS:
        r = runs.get((d_, strat))
        if r is None: continue
        idx = r.min_val_idx()
        P.append(df_to_P(d_)); L.append(r.min_val_loss())
        ep.append(int(r.epoch[idx])); tok.append(int(r.tokens[idx])); df.append(d_)
    return {"P": np.array(P, dtype=float), "L": np.array(L),
            "ep": np.array(ep), "tok": np.array(tok), "df": np.array(df)}


def fit_rP_given_L0(P: np.ndarray, L: np.ndarray, L0: float):
    """Fit min_L − L_0 = c · P^{-r_P} via OLS in log space at fixed L_0.
    Returns (c, r_P, r_P_se, residual_sigma)."""
    gap = L - L0
    if (gap <= 0).any():
        return np.nan, np.nan, np.nan, np.nan
    lx, ly = np.log(P), np.log(gap)
    n = len(lx)
    X = np.column_stack([lx, np.ones_like(lx)])
    beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
    slope, intercept = beta[0], beta[1]
    resid = X @ beta - ly
    dof = max(n - 2, 1)
    sigma2 = (resid ** 2).sum() / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    slope_se = float(np.sqrt(cov[0, 0]))
    return float(np.exp(intercept)), float(-slope), slope_se, float(np.sqrt(sigma2))


def fit_rP_unconstrained(P: np.ndarray, L: np.ndarray):
    """3-param NLLS: min_L = c · P^{-r_P} + L_0. Returns dict."""
    def residuals(p):
        c, r_P, L0 = p
        return c * P ** (-r_P) + L0 - L
    p0 = np.array([1e3, 0.3, float(L.min()) - 0.1])
    lo = np.array([1e-3, 0.01, 0.0])
    hi = np.array([1e15, 3.0,  L.min()])
    res = least_squares(residuals, p0, bounds=(lo, hi), max_nfev=20000)
    J = res.jac
    dof = max(J.shape[0] - J.shape[1], 1)
    sigma2 = (res.fun ** 2).sum() / dof
    try:
        cov = sigma2 * np.linalg.inv(J.T @ J); se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(3, np.nan)
    c, r_P, L0 = res.x
    return {"c": c, "c_se": se[0], "r_P": r_P, "r_P_se": se[1],
            "L0": L0, "L0_se": se[2], "sigma": float(np.sqrt(sigma2))}


def main():
    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 20, "axes.linewidth": 2.5, "legend.fontsize": 13,
        "grid.alpha": 0.25, "xtick.labelsize": 15, "ytick.labelsize": 15,
    })

    runs = load_all()
    data = {s: collect(runs, s) for s in STRATEGIES}

    # Choose a grid of L_0 anchors that span "no irreducible loss" up to "L_0
    # nearly at the smallest observed min_L". 2.5 and 3.0 are physically
    # plausible Bayes-error proxies for FineWeb val on a 1.8B-param GPT.
    L0_anchors = [0.0, 2.0, 2.5, 3.0, 3.5, 4.0]
    L0_continuous = np.linspace(0.0, 4.15, 80)

    print("=== r_P fit at fixed L_0 anchors (per strategy) ===\n")
    print(f"  {'L_0':>5}  {'init_rP':>10}  {'init_se':>9}  {'σ':>7}   "
          f"{'shuf_rP':>10}  {'shuf_se':>9}  {'σ':>7}")
    coef_rows = ["strategy,L0,c,r_P,r_P_se,sigma"]
    for L0 in L0_anchors:
        line = f"  {L0:>5.2f}  "
        for s in STRATEGIES:
            c, rP, rP_se, sig = fit_rP_given_L0(data[s]["P"], data[s]["L"], L0)
            line += f"{rP:>10.4f}  {rP_se:>9.4f}  {sig:>7.4f}   "
            coef_rows.append(f"{s},{L0},{c:.6e},{rP:.6f},{rP_se:.6f},{sig:.6f}")
        print(line)

    print("\n=== Unconstrained 3-param fit (c, r_P, L_0) ===\n")
    for s in STRATEGIES:
        f = fit_rP_unconstrained(data[s]["P"], data[s]["L"])
        print(f"  [{STRAT_PRETTY[s]:<14}]  σ = {f['sigma']:.4f}")
        print(f"    r_P = {f['r_P']:.3f} ± {f['r_P_se']:.3f}")
        print(f"    L_0 = {f['L0']:.3f} ± {f['L0_se']:.3f}")
        print(f"    c   = {f['c']:.3e} ± {f['c_se']:.3e}\n")

    csv = OUT_DIR / f"{OUT_NAME}_coefs.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    csv.write_text("\n".join(coef_rows) + "\n")
    print(f"Saved {csv}")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.0))
    palette_df = sns.color_palette("cool", len(DFS))
    palette_L0 = sns.color_palette("viridis_r", len(L0_anchors))
    strat_marker = {"init_ens": "o", "init_shuffle_ens": "s"}

    # ---- Panel A: data + several candidate fits at different assumed L_0 ----
    axA = axes[0]
    axA.set_xscale("log")
    P_grid = np.geomspace(15e6, 60e6, 80)
    # Use init+shuffle as the "headline" strategy for the overlay
    s = "init_shuffle_ens"
    d = data[s]
    axA.scatter(d["P"], d["L"],
                color=[palette_df[i] for i in range(len(DFS))],
                marker=strat_marker[s], s=180, edgecolor="black", lw=0.8,
                zorder=5)
    for j, L0 in enumerate(L0_anchors):
        c, rP, rP_se, sig = fit_rP_given_L0(d["P"], d["L"], L0)
        if np.isnan(rP): continue
        axA.plot(P_grid, c * P_grid ** (-rP) + L0,
                 color=palette_L0[j], lw=2.0, alpha=0.85,
                 label=f"$L_0 = {L0:.1f}$  $\\Rightarrow$  "
                       f"$r_P = {rP:.2f} \\pm {rP_se:.2f}$")
    axA.set_xlabel(r"unique tokens $P$")
    axA.set_ylabel(r"min val loss")
    axA.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1e6)}M"))
    axA.set_title(r"(A) min val loss vs $P$ — fits at various assumed $L_0$"
                  "\n(init + shuffle shown; init is nearly identical)",
                  fontsize=14)
    axA.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.92,
               title="anchor")

    # ---- Panel B: r_P as a function of assumed L_0 (the identification curve) ----
    axB = axes[1]
    for s in STRATEGIES:
        d = data[s]
        rP_curve = []; rP_se_curve = []
        for L0 in L0_continuous:
            c, rP, rP_se, _ = fit_rP_given_L0(d["P"], d["L"], L0)
            rP_curve.append(rP); rP_se_curve.append(rP_se)
        rP_curve = np.array(rP_curve); rP_se_curve = np.array(rP_se_curve)
        valid = ~np.isnan(rP_curve)
        axB.plot(L0_continuous[valid], rP_curve[valid], lw=3.0,
                 label=STRAT_PRETTY[s])
        axB.fill_between(L0_continuous[valid],
                         rP_curve[valid] - rP_se_curve[valid],
                         rP_curve[valid] + rP_se_curve[valid],
                         alpha=0.18)
    # Vertical guide lines at the anchor L_0 values
    for L0, col in zip(L0_anchors, palette_L0):
        axB.axvline(L0, color=col, lw=0.8, ls=":", alpha=0.5)
    axB.set_xlabel(r"assumed $L_0$ (Bayes / data-limited floor)")
    axB.set_ylabel(r"fitted $r_P$  (with $\pm 1\sigma$ band)")
    axB.set_title(r"(B) $r_P$ identification curve: $r_P$ depends on $L_0$",
                  fontsize=14)
    axB.legend(loc="upper left", fontsize=12, frameon=True, framealpha=0.92)
    axB.set_ylim(0, 3.5)

    # ---- Panel C: nadir epoch and nadir tokens vs P ----
    axC = axes[2]
    ax2 = axC.twinx()
    for s in STRATEGIES:
        d = data[s]
        colors = [palette_df[i] for i in range(len(DFS))]
        axC.scatter(d["P"], d["ep"], color=colors,
                    marker=strat_marker[s], s=180, edgecolor="black", lw=0.8,
                    zorder=4, label=f"{STRAT_PRETTY[s]}  (epoch)")
        ax2.scatter(d["P"], d["tok"] / 1e6, color=colors,
                    marker=strat_marker[s], s=140,
                    facecolor="none", edgecolor="0.35", lw=1.4,
                    zorder=3)
    axC.set_xscale("log")
    axC.set_xlabel(r"unique tokens $P$")
    axC.set_ylabel("nadir epoch  (filled)", color="black")
    ax2.set_ylabel("nadir cumulative tokens (M)  (open)", color="0.35")
    axC.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1e6)}M"))
    axC.set_title(r"(C) overfit-onset position vs $P$"
                  "\nepoch is constant; tokens scale linearly with $P$",
                  fontsize=14)
    axC.set_ylim(8, 14)
    axC.grid(True, alpha=0.25)
    axC.legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.9)

    fig.tight_layout()

    pdf = OUT_DIR / f"{OUT_NAME}.pdf"
    png = OUT_DIR / f"{OUT_NAME}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    print(f"Saved {pdf}")
    print(f"Saved {png}")

    # Onset diagnostic table
    print("\n=== Onset diagnostics (init + shuffle) ===")
    d = data["init_shuffle_ens"]
    print(f"  {'df':>4}  {'P (M)':>6}  {'epoch_nadir':>12}  {'tokens_nadir (M)':>18}")
    for i in range(len(d["P"])):
        print(f"  {d['df'][i]:>4}  {d['P'][i]/1e6:>5.0f}M  "
              f"{d['ep'][i]:>12}  {d['tok'][i]/1e6:>16.0f}M")


if __name__ == "__main__":
    main()
