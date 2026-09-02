"""Expt Fig 3.5 — joint scaling-law fit on the df=0.2 4×4 grid.

Model (with df=0.2 fixed, so P^{-r_P} is absorbed into L_0):

    L(s, N, L, E) = (s/s0)^{-r_t}          # bias dynamics
                  + (N^2 L / k0)^{-r_param} # parameter-driven bias floor
                  + E^{-δ_E} · N^{δ_N} · L^{δ_L} · (s/T)^δ   # variance buildup
                  + L_0                     # data-limited + Bayes term

Variables per observation:
    s    = cumulative tokens at snapshot
    s/T  = epoch number (1, 2, ..., 25)
    N    = width
    L    = depth
    E    = ensemble size in {1, 2, 3, 4, 5}; E=1 = per-snapshot mean of 5 individuals

Two fits per strategy:
  (A) all 7 params free
  (B) δ_E pinned at the Expt 2 value (init: 0.88, shuffle: 1.09)

s and N²L are normalized by reference values (base cell, 100M tokens) so
the scaling constants don't get absorbed into L_0 numerically.

Figure: 3-panel diagnostic
  - left:  predicted vs observed scatter (all obs, color by depth, marker by strategy)
  - mid:   residual vs predicted, banded by E
  - right: per-cell U-curve overlay at E ∈ {1, 5} for one example cell — verifies
           the model captures both the U-shape and the ensemble effect.

Reads from data_export/expt3_grid/. Saves to
experiments/figures/03_compute_matched/expt_fig3_5_scaling_law.{pdf,png}
and a coefficient CSV.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.optimize import least_squares
from scipy.stats import norm

from expt_fig3_loader import (
    DEPTHS, WIDTHS, STRATEGIES, E_SIZES, TOKENS_PER_EPOCH_DF02,
    load_grid, REPO,
)


OUT_DIR = REPO / "experiments" / "figures" / "03_compute_matched"
OUT_NAME = "expt_fig3_5_scaling_law"

STRAT_PRETTY = {"init_ens": "init", "init_shuffle_ens": "init + shuffle"}
ALL_E = (1, 2, 3, 4, 5)

# Pinned δ_E from canonical Expt 2 fit (with E=1 anchor, 6 sizes, 480 obs)
DE_FROM_EXPT2 = {"init_ens": 0.878, "init_shuffle_ens": 1.092}

# Reference scales for numerical conditioning
S0_TOKENS = 100_000_000.0          # 100M
NL0 = 12 * 768 * 768                # base cell N²L


# ---------------------------- data assembly ----------------------------

def build_dataset_for_strat(grid, strat: str):
    """Return long-format arrays for one strategy.

    Returns dict with keys:
      s, sT, N, L, E, L_obs   (all 1D arrays of length n_obs)
    where n_obs = 16 cells × 5 E × 25 snapshots = 2000.
    """
    s_list, sT_list, N_list, L_list, E_list, L_list_obs = [], [], [], [], [], []
    for d in DEPTHS:
        for w in WIDTHS:
            cell = grid[(d, w, strat)]
            if not cell.individuals or len(cell.ensembles) < len(E_SIZES):
                continue
            tok = cell.individuals[0][0]   # (n_T,)
            n_T = len(tok)
            sT = np.arange(1, n_T + 1, dtype=float)   # epoch numbers
            # E = 1: per-snapshot mean of individuals
            indiv = np.stack([vl for _, vl in cell.individuals], axis=0)  # (5, n_T)
            e1 = indiv.mean(axis=0)
            for E_val, vl_curve in [(1, e1)] + [(int(E), cell.ensembles[E][1]) for E in E_SIZES]:
                s_list.append(tok)
                sT_list.append(sT)
                N_list.append(np.full(n_T, w, dtype=float))
                L_list.append(np.full(n_T, d, dtype=float))
                E_list.append(np.full(n_T, float(E_val)))
                L_list_obs.append(vl_curve)
    return {
        "s":     np.concatenate(s_list),
        "sT":    np.concatenate(sT_list),
        "N":     np.concatenate(N_list),
        "L":     np.concatenate(L_list),
        "E":     np.concatenate(E_list),
        "L_obs": np.concatenate(L_list_obs),
    }


# ---------------------------- model + fit ----------------------------

def predict(params, ds, dE_fixed=None):
    """Evaluate the additive scaling law with explicit amplitudes.

    Generalized form (10 params total when δ_E free, 9 when pinned):

        L = a · (s/s_0)^{-r_t}                    [bias dynamics]
          + b · (N²L/k_0)^{-r_param}              [parameter-driven bias floor]
          + c · E^{-δ_E} (N/N_0)^{δ_N} (L/L_0)^{δ_L} (s/T)^δ   [variance buildup]
          + L_0                                   [data-limited + Bayes]

    Amplitudes a, b, c absorb scaling constants and decouple from the exponents
    in the optimization. (N/N_0), (L/L_0) are normalized so δ_N, δ_L are
    interpretable as "loss-gap doubles when width / depth doubles past base."
    """
    if dE_fixed is None:
        a, b, c, r_t, r_param, dE, dN, dL, d, L0 = params
    else:
        a, b, c, r_t, r_param, dN, dL, d, L0 = params
        dE = dE_fixed
    s_norm   = ds["s"] / S0_TOKENS
    NL_norm  = (ds["N"] ** 2 * ds["L"]) / NL0
    N_norm   = ds["N"] / 768.0
    Ld_norm  = ds["L"] / 12.0
    bias_dyn   = a * s_norm ** (-r_t)
    bias_param = b * NL_norm ** (-r_param)
    var_term   = c * (ds["E"] ** (-dE)) * (N_norm ** dN) * (Ld_norm ** dL) * (ds["sT"] ** d)
    return bias_dyn + bias_param + var_term + L0


def fit_one_strat(ds, dE_fixed=None, verbose=True, n_restarts=8, seed=0):
    """Multi-start log-residual NLLS fit.

    Sign constraints:
      a, b, c > 0   (positive amplitudes)
      r_t, r_param, δ_E > 0
      δ_N, δ_L ≥ 0  (variance grows with N, L)
      δ        > 0  (variance grows with epochs)
    """
    # δ is bounded to [0.1, 2.0] — the proposed law conjectures δ = 0.5; we
    # allow up to 2 for super-quadratic variance growth in epochs but disallow
    # the runaway high-δ corner that trades off with tiny c (degeneracy with
    # the bias-dynamics term s^{-r_t}).
    if dE_fixed is None:
        p0_base = np.array([1.0, 0.5, 0.3, 0.5, 0.15, 1.00, 0.30, 0.30, 0.50, 3.00])
        lo = np.array([1e-4, 1e-5, 1e-6, 0.05, 0.01, 0.30, 0.0, 0.0, 0.10, 2.5])
        hi = np.array([20.0, 10.0, 10.0, 2.0,  2.0,  3.0,  3.0, 3.0, 2.0,  4.5])
        names = ["a", "b", "c", "r_t", "r_param", "δ_E", "δ_N", "δ_L", "δ", "L_0"]
    else:
        p0_base = np.array([1.0, 0.5, 0.3, 0.5, 0.15, 0.30, 0.30, 0.50, 3.00])
        lo = np.array([1e-4, 1e-5, 1e-6, 0.05, 0.01, 0.0, 0.0, 0.10, 2.5])
        hi = np.array([20.0, 10.0, 10.0, 2.0,  2.0,  3.0, 3.0, 2.0,  4.5])
        names = ["a", "b", "c", "r_t", "r_param", "δ_N", "δ_L", "δ", "L_0"]

    obs = ds["L_obs"]

    def residuals(p):
        pred = predict(p, ds, dE_fixed)
        # Log-residuals: equal relative weight, prevents early high-loss
        # observations from dominating and lets the variance term emerge.
        return np.log(np.maximum(pred, 1e-3)) - np.log(np.maximum(obs, 1e-3))

    rng = np.random.default_rng(seed)
    best = None
    for restart in range(n_restarts):
        if restart == 0:
            p0 = p0_base.copy()
        else:
            # Perturb in log-space for amplitude/exponent params
            jitter = np.exp(rng.normal(0, 0.5, size=p0_base.shape))
            p0 = np.clip(p0_base * jitter, lo + 1e-6, hi - 1e-6)
        try:
            res = least_squares(residuals, p0, bounds=(lo, hi),
                                method="trf", max_nfev=20000, xtol=1e-12)
        except Exception:
            continue
        cost = float((res.fun ** 2).sum())
        if best is None or cost < best[0]:
            best = (cost, res)
    res = best[1]
    J = res.jac
    n_obs, n_params = J.shape
    dof = max(n_obs - n_params, 1)
    sigma2 = (res.fun ** 2).sum() / dof
    try:
        cov = sigma2 * np.linalg.inv(J.T @ J)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(n_params, np.nan)

    pred = predict(res.x, ds, dE_fixed)
    obs = ds["L_obs"]
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    if verbose:
        print(f"  σ = {np.sqrt(sigma2):.4f}, R² = {r2:.4f}, n_obs = {n_obs}, n_params = {n_params}")
        for i, name in enumerate(names):
            print(f"    {name:<10} = {res.x[i]:+.4f} ± {se[i]:.4f}")

    return {
        "x": res.x, "se": se, "names": names,
        "sigma": float(np.sqrt(sigma2)), "r2": float(r2),
        "n_obs": n_obs, "n_params": n_params,
        "dE_fixed": dE_fixed,
        "pred": pred,
    }


# ---------------------------- figure ----------------------------

def make_figure(datasets, fits_free, fits_pinned):
    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 20, "axes.linewidth": 2.5, "legend.fontsize": 12,
        "grid.alpha": 0.25, "xtick.labelsize": 14, "ytick.labelsize": 14,
    })

    fig = plt.figure(figsize=(20, 6.5))
    gs = fig.add_gridspec(1, 3, wspace=0.30)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    palette_d = sns.color_palette("cool", len(DEPTHS))
    color_d = {d: palette_d[i] for i, d in enumerate(DEPTHS)}
    strat_marker = {"init_ens": "o", "init_shuffle_ens": "s"}

    # ---- Panel A: predicted vs observed (using pinned-δ_E fit) ----
    for strat in STRATEGIES:
        ds = datasets[strat]
        pred = fits_pinned[strat]["pred"]
        for d in DEPTHS:
            mask = ds["L"] == d
            axA.scatter(ds["L_obs"][mask], pred[mask],
                        s=18, alpha=0.45, color=color_d[d],
                        marker=strat_marker[strat],
                        edgecolors="none")
    lo = min(np.concatenate([d["L_obs"] for d in datasets.values()]).min(),
             np.concatenate([fits_pinned[s]["pred"] for s in STRATEGIES]).min())
    hi = max(np.concatenate([d["L_obs"] for d in datasets.values()]).max(),
             np.concatenate([fits_pinned[s]["pred"] for s in STRATEGIES]).max())
    axA.plot([lo, hi], [lo, hi], "k-", lw=1.2, alpha=0.7)
    axA.set_xlabel("observed val loss")
    axA.set_ylabel("predicted val loss")
    r2_lines = "\n".join(
        f"{STRAT_PRETTY[s]}: $R^2 = {fits_pinned[s]['r2']:.3f}$, "
        f"$\\sigma = {fits_pinned[s]['sigma']:.3f}$"
        for s in STRATEGIES
    )
    axA.text(0.04, 0.96, r2_lines, transform=axA.transAxes,
             fontsize=12, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       alpha=0.85, edgecolor="lightgray"))
    axA.set_title(r"(A) predicted vs observed ($\delta_E$ pinned to Expt 2)", fontsize=14)
    # Depth + strategy legend
    handles = (
        [Line2D([0], [0], marker="o", lw=0, color=color_d[d],
                markersize=8, label=f"d = {d}") for d in DEPTHS]
        + [Line2D([0], [0], marker="o", lw=0, color="0.4",
                  markersize=8, label="init"),
           Line2D([0], [0], marker="s", lw=0, color="0.4",
                  markersize=8, label="init+shuffle")]
    )
    axA.legend(handles=handles, loc="lower right", fontsize=10, ncol=2,
               frameon=True, framealpha=0.9)

    # ---- Panel B: residual vs predicted, colored by E ----
    palette_E = sns.color_palette("cool", len(ALL_E))
    color_E = {E: palette_E[i] for i, E in enumerate(ALL_E)}
    for strat in STRATEGIES:
        ds = datasets[strat]
        pred = fits_pinned[strat]["pred"]
        resid = ds["L_obs"] - pred
        for E in ALL_E:
            mask = ds["E"] == E
            axB.scatter(pred[mask], resid[mask],
                        s=14, alpha=0.45, color=color_E[E],
                        marker=strat_marker[strat], edgecolors="none")
    axB.axhline(0, color="0.2", lw=1.0, ls="-", alpha=0.7)
    axB.set_xlabel("predicted val loss")
    axB.set_ylabel("residual (obs − pred)")
    axB.set_title("(B) residual vs predicted (colored by $E$)", fontsize=14)
    handles_B = [Line2D([0], [0], marker="o", lw=0, color=color_E[E],
                        markersize=8, label=f"E = {E}") for E in ALL_E]
    axB.legend(handles=handles_B, loc="upper right", fontsize=10, ncol=1,
               frameon=True, framealpha=0.9)

    # ---- Panel C: example cell U-curves (overlaid model fit) ----
    # Pick a "stress test" cell: d=24, w=1536 (largest, strongest U-curve).
    show_cell = (24, 1536)
    show_strat = "init_shuffle_ens"
    ds = datasets[show_strat]
    fit = fits_pinned[show_strat]
    cell_mask = (ds["N"] == show_cell[1]) & (ds["L"] == show_cell[0])
    for E in ALL_E:
        m = cell_mask & (ds["E"] == E)
        s = ds["s"][m]
        obs = ds["L_obs"][m]
        pred = fit["pred"][m]
        order = np.argsort(s)
        axC.plot(s[order] / 1e6, obs[order], "o", markersize=6,
                 color=color_E[E], alpha=0.7, label=f"obs E={E}",
                 markeredgecolor="black", markeredgewidth=0.4)
        axC.plot(s[order] / 1e6, pred[order], "-", lw=2.0,
                 color=color_E[E], alpha=0.9)
    axC.set_xlabel("tokens seen (M)")
    axC.set_ylabel("val loss")
    axC.set_title(f"(C) example cell d={show_cell[0]}, w={show_cell[1]}\n"
                  f"({STRAT_PRETTY[show_strat]}) — observed vs model",
                  fontsize=13)
    axC.legend(loc="upper right", fontsize=10, ncol=1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{OUT_NAME}.pdf"
    png = OUT_DIR / f"{OUT_NAME}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    print(f"\nSaved {pdf}")
    print(f"Saved {png}")


def write_csv(fits_free, fits_pinned):
    rows = ["strategy,fit,param,value,se"]
    for s in STRATEGIES:
        for tag, fit in [("free_dE", fits_free[s]), ("pinned_dE", fits_pinned[s])]:
            for nm, v, sev in zip(fit["names"], fit["x"], fit["se"]):
                rows.append(f"{s},{tag},{nm},{v:.6f},{sev:.6f}")
            rows.append(f"{s},{tag},sigma,{fit['sigma']:.6f},")
            rows.append(f"{s},{tag},r2,{fit['r2']:.6f},")
            rows.append(f"{s},{tag},n_obs,{fit['n_obs']},")
            rows.append(f"{s},{tag},n_params,{fit['n_params']},")
    csv_path = OUT_DIR / f"{OUT_NAME}_coefs.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(rows) + "\n")
    print(f"Saved {csv_path}")


def main():
    grid = load_grid()
    datasets = {s: build_dataset_for_strat(grid, s) for s in STRATEGIES}
    for s in STRATEGIES:
        ds = datasets[s]
        print(f"  {s}: n_obs = {len(ds['L_obs'])}")

    print("\n=== Fit (A): all 7 params free, per strategy ===")
    fits_free = {}
    for s in STRATEGIES:
        print(f"\n[{STRAT_PRETTY[s]}]")
        fits_free[s] = fit_one_strat(datasets[s], dE_fixed=None)

    print(f"\n=== Fit (B): δ_E pinned to Expt 2 (init={DE_FROM_EXPT2['init_ens']}, "
          f"shuffle={DE_FROM_EXPT2['init_shuffle_ens']}) ===")
    fits_pinned = {}
    for s in STRATEGIES:
        print(f"\n[{STRAT_PRETTY[s]}]")
        fits_pinned[s] = fit_one_strat(datasets[s], dE_fixed=DE_FROM_EXPT2[s])

    make_figure(datasets, fits_free, fits_pinned)
    write_csv(fits_free, fits_pinned)


if __name__ == "__main__":
    main()
