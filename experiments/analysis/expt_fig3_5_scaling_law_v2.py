"""Expt Fig 3.5 v2 — joint scaling-law fit with grouped (N²L) capacity term
and an early-loss filter.

Per the advisor's two suggestions:
  (1) Group N (width) and L (depth) into the single parameter-count proxy
      N²L for the variance term, so it shares the same capacity proxy as
      the bias floor.
  (2) Drop the early-training high-loss observations (loss > 5.5) which
      sit in the warm-up phase and don't follow the powerlaw.

Model:

    L(s, N, L, E) = a (s/s_0)^{-r_t}
                  + b (N^2 L / k_0)^{-r_param}
                  + c E^{-δ_E} (N^2 L / k_0)^{-δ_param} (s/T)^δ
                  + L_0

Compared to v1 (`expt_fig3_5_scaling_law.py`) the variance term replaces
the (N/N_0)^{δ_N}(L/L_0)^{δ_L} pair with a single (N^2 L/k_0)^{-δ_param},
trimming one free parameter. δ_param is allowed to be either sign — a
positive δ_param means the per-epoch variance term *decreases* with
parameter count, a negative δ_param means it grows (which is what v1
recovered via positive δ_N, δ_L).

Two fits per strategy:
  (A) all params free
  (B) δ_E pinned to the Expt 2 value (init: 0.88, shuffle: 1.09)

Outputs:
  experiments/figures/03_compute_matched/v2/expt_fig3_5_scaling_law_v2.{pdf,png}
  experiments/figures/03_compute_matched/v2/expt_fig3_5_scaling_law_v2_coefs.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.optimize import least_squares

from expt_fig3_loader import (
    DEPTHS, WIDTHS, STRATEGIES, E_SIZES, load_grid, REPO,
)


OUT_DIR = REPO / "experiments" / "figures" / "03_compute_matched" / "v2"
OUT_NAME = "expt_fig3_5_scaling_law_v2"

STRAT_PRETTY = {"init_ens": "init", "init_shuffle_ens": "init + shuffle"}
ALL_E = (1, 2, 3, 4, 5)
DE_PIN = {"init_ens": 1.0, "init_shuffle_ens": 1.0}   # δ_E pinned to 1

BATCH_SIZE = 131072                       # project default --total-batch-size
S0_STEPS = 100_000_000.0 / BATCH_SIZE     # 100M tokens / batch ≈ 763 steps
NL0 = 12 * 768 * 768                       # base-cell N²L
LOSS_FILTER = 5.5                         # leading-prefix L_obs > 5.5 dropped


# ---------------------------- data ----------------------------

def build_dataset_for_strat(grid, strat: str):
    s_list, sT_list, N_list, L_list, E_list, L_obs_list, cid_list = (
        [], [], [], [], [], [], [])
    cid = 0
    for d in DEPTHS:
        for w in WIDTHS:
            cell = grid[(d, w, strat)]
            if not cell.individuals or len(cell.ensembles) < len(E_SIZES):
                continue
            tok = cell.individuals[0][0]
            steps = tok / BATCH_SIZE                 # x-axis: optimizer steps
            n_T = len(tok)
            sT = np.arange(1, n_T + 1, dtype=float)  # sT = s/T = epoch index
            indiv = np.stack([vl for _, vl in cell.individuals], axis=0)
            e1 = indiv.mean(axis=0)
            for E_val, vl_curve in [(1, e1)] + [(int(E), cell.ensembles[E][1]) for E in E_SIZES]:
                s_list.append(steps)
                sT_list.append(sT)
                N_list.append(np.full(n_T, w, dtype=float))
                L_list.append(np.full(n_T, d, dtype=float))
                E_list.append(np.full(n_T, float(E_val)))
                L_obs_list.append(vl_curve)
                cid_list.append(np.full(n_T, cid, dtype=int))
                cid += 1
    return {
        "s":        np.concatenate(s_list),
        "sT":       np.concatenate(sT_list),
        "N":        np.concatenate(N_list),
        "L":        np.concatenate(L_list),
        "E":        np.concatenate(E_list),
        "L_obs":    np.concatenate(L_obs_list),
        "curve_id": np.concatenate(cid_list),
    }


def filter_ds(ds: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in ds.items()}


def leading_high_loss_mask(ds: dict, threshold: float) -> np.ndarray:
    """Return a `keep` bool mask. Per curve (curve_id), drop only the LEADING
    contiguous block of points whose L_obs > threshold (the warm-up phase).
    Once a curve dips to L_obs ≤ threshold, all subsequent points are kept,
    even if they later cross back above (the late-overfit U-tail).
    Curves that never dip below the threshold are dropped entirely.
    """
    keep = np.ones(len(ds["L_obs"]), dtype=bool)
    for cid in np.unique(ds["curve_id"]):
        idxs = np.where(ds["curve_id"] == cid)[0]
        # idxs are already in time order because build_dataset appends per-curve
        L = ds["L_obs"][idxs]
        below = L <= threshold
        if not below.any():
            keep[idxs] = False
            continue
        first_below = int(below.argmax())   # first index where L ≤ threshold
        keep[idxs[:first_below]] = False
    return keep


# ---------------------------- model ----------------------------

def predict(params, ds, dE_fixed=None):
    """L = a(s/s_0)^{-r_t} + b(NL/k_0)^{-r_param}
            + c E^{-δ_E}(NL/k_0)^{-δ_param}(s/T)^δ + L_0
    where NL := N²L (total params proxy).
    """
    if dE_fixed is None:
        a, b, c, r_t, r_param, dE, dParam, d, L0 = params
    else:
        a, b, c, r_t, r_param, dParam, d, L0 = params
        dE = dE_fixed
    s_norm  = ds["s"] / S0_STEPS
    NL_norm = (ds["N"] ** 2 * ds["L"]) / NL0
    bias_dyn   = a * s_norm  ** (-r_t)
    bias_param = b * NL_norm ** (-r_param)
    var_term   = c * (ds["E"] ** (-dE)) * (NL_norm ** (-dParam)) * (ds["sT"] ** d)
    return bias_dyn + bias_param + var_term + L0


def fit_one_strat(ds_fit, dE_fixed=None, verbose=True, n_restarts=12, seed=0):
    if dE_fixed is None:
        # [a, b, c, r_t, r_param, δ_E, δ_param, δ, L_0]
        p0_base = np.array([1.0, 0.5, 0.3, 0.5, 0.15, 1.00, -0.30, 0.50, 3.00])
        lo = np.array([1e-4, 1e-5, 1e-9, 0.05, 0.01, 0.05, -3.0, 0.05, 2.5])
        hi = np.array([20.0, 10.0, 10.0, 3.0,  3.0,  4.0,   3.0, 5.0,  4.5])
        names = ["a", "b", "c", "r_t", "r_param", "δ_E", "δ_param", "δ", "L_0"]
    else:
        p0_base = np.array([1.0, 0.5, 0.3, 0.5, 0.15, -0.30, 0.50, 3.00])
        lo = np.array([1e-4, 1e-5, 1e-9, 0.05, 0.01, -3.0, 0.05, 2.5])
        hi = np.array([20.0, 10.0, 10.0, 3.0,  3.0,   3.0, 5.0,  4.5])
        names = ["a", "b", "c", "r_t", "r_param", "δ_param", "δ", "L_0"]

    obs = ds_fit["L_obs"]

    def residuals(p):
        pred = predict(p, ds_fit, dE_fixed)
        return np.log(np.maximum(pred, 1e-3)) - np.log(np.maximum(obs, 1e-3))

    rng = np.random.default_rng(seed)
    best = None
    for restart in range(n_restarts):
        if restart == 0:
            p0 = p0_base.copy()
        else:
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

    pred = predict(res.x, ds_fit, dE_fixed)
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot

    if verbose:
        print(f"  σ = {np.sqrt(sigma2):.4f}, R² = {r2:.4f}, "
              f"n_obs = {n_obs}, n_params = {n_params}")
        for i, name in enumerate(names):
            print(f"    {name:<10} = {res.x[i]:+.4f} ± {se[i]:.4f}")

    return {
        "x": res.x, "se": se, "names": names,
        "sigma": float(np.sqrt(sigma2)), "r2": float(r2),
        "n_obs": n_obs, "n_params": n_params,
        "dE_fixed": dE_fixed,
    }


# ---------------------------- figure ----------------------------

def make_figure(datasets_full, keep_masks, fits_pinned, fits_free,
                show_cell=(24, 1536), show_strat="init_shuffle_ens"):
    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 20, "axes.linewidth": 2.5, "legend.fontsize": 12,
        "grid.alpha": 0.25, "xtick.labelsize": 14, "ytick.labelsize": 14,
    })

    fig, axC = plt.subplots(figsize=(12, 6.5))
    palette_E = sns.color_palette("cool", len(ALL_E))
    color_E = {E: palette_E[i] for i, E in enumerate(ALL_E)}

    # example-cell U-curves: data vs pinned-δ_E model
    ds = datasets_full[show_strat]
    keep = keep_masks[show_strat]
    x = fits_pinned[show_strat]["x"]
    dE_fixed = fits_pinned[show_strat]["dE_fixed"]
    pred_full = predict(x, ds, dE_fixed)
    cell_mask = (ds["N"] == show_cell[1]) & (ds["L"] == show_cell[0])
    for E in ALL_E:
        m = cell_mask & (ds["E"] == E)
        s = ds["s"][m]
        obs = ds["L_obs"][m]
        pp = pred_full[m]
        used = keep[m]
        order = np.argsort(s)
        s_o = s[order]; obs_o = obs[order]; pp_o = pp[order]; used_o = used[order]
        # used (filled)
        axC.plot(s_o[used_o], obs_o[used_o], "o", markersize=7,
                 color=color_E[E], alpha=0.85, label=f"obs E={E}",
                 markeredgecolor="black", markeredgewidth=0.5)
        # dropped (open) — only the leading high-loss prefix per curve
        if (~used_o).any():
            axC.plot(s_o[~used_o], obs_o[~used_o], "o", markersize=7,
                     markerfacecolor="none", markeredgecolor=color_E[E],
                     markeredgewidth=1.5, alpha=0.85)
        axC.plot(s_o, pp_o, "-", lw=2.0,
                 color=color_E[E], alpha=0.9)
    # cutoff line + label
    axC.axhline(LOSS_FILTER, color="0.4", lw=1.2, ls=":", alpha=0.8)
    axC.text(axC.get_xlim()[1] * 0.97, LOSS_FILTER + 0.05,
             rf"warm-up cutoff:  $L_{{obs}} \leq {LOSS_FILTER}$  (leading-prefix only)",
             color="0.35", fontsize=11, ha="right", va="bottom", alpha=0.9)
    axC.set_xlabel("optimizer steps $s$")
    axC.set_ylabel("val loss")
    axC.set_title(f"example cell d={show_cell[0]}, w={show_cell[1]}\n"
                  rf"({STRAT_PRETTY[show_strat]}) — observed vs model "
                  rf"($\delta_E = 1$ pinned)",
                  fontsize=14)
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
    datasets_full = {s: build_dataset_for_strat(grid, s) for s in STRATEGIES}
    keep_masks = {s: leading_high_loss_mask(datasets_full[s], LOSS_FILTER)
                  for s in STRATEGIES}
    datasets_fit = {s: filter_ds(datasets_full[s], keep_masks[s]) for s in STRATEGIES}
    for s in STRATEGIES:
        n_full = len(datasets_full[s]["L_obs"])
        n_kept = len(datasets_fit[s]["L_obs"])
        print(f"  {s}: n_full = {n_full}, n_kept (drop leading L>{LOSS_FILTER}) = "
              f"{n_kept}  (dropped {n_full - n_kept})")

    print("\n=== Fit (A): all params free, per strategy ===")
    fits_free = {}
    for s in STRATEGIES:
        print(f"\n[{STRAT_PRETTY[s]}]")
        fits_free[s] = fit_one_strat(datasets_fit[s], dE_fixed=None)

    print("\n=== Fit (B): δ_E pinned to 1.0 (naive 1/E variance reduction) ===")
    fits_pinned = {}
    for s in STRATEGIES:
        print(f"\n[{STRAT_PRETTY[s]}]")
        fits_pinned[s] = fit_one_strat(datasets_fit[s], dE_fixed=DE_PIN[s])

    make_figure(datasets_full, keep_masks, fits_pinned, fits_free)
    write_csv(fits_free, fits_pinned)


if __name__ == "__main__":
    main()
