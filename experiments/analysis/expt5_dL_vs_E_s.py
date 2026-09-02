"""Expt 5 — Fit δL = L_E − L_20 ≈ C · E^(−δ_E) · s^(δ_s).

Data: data_export/expt2_ensemble/  (d=12, w=768, df=0.2, 40 epochs).
  ensembles/   E ∈ {2, 5, 10, 15, 20}     point estimates
  individuals/ 20 inds                    point estimates  (E=1 anchor for plot)
  bootstrap/   10 iters × E ∈ {2,5,10,15,20}
               L_20 is fixed (full 20-model ensemble); L_E<20 is bootstrap-
               resampled by drawing E indices from the 20 individuals. So
               δL = L_E − L_20 inherits sampling variance from L_E only.

Fit per strategy (init_ens / init_shuffle_ens) in log-log space:
  log δL = log C − δ_E · log E + δ_s · log s

Uncertainty: per-iteration OLS fit on each of the 10 bootstrap replicates;
report mean ± SD of (C, δ_E, δ_s) across iters as the bootstrap SE.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data_export" / "expt2_ensemble"
TOTAL_BATCH = 131072
STRATS = ("init_ens", "init_shuffle_ens")
E_ENS = (2, 5, 10, 15)        # E used in the fit (E=20 is baseline, E=1 is plot-only)
E_BASELINE = 20


def load_bootstrap(strat: str):
    """Return (s, L20, L_iters) where:
        s       (n_snap,)
        L20     (n_snap,)               fixed across iters (full 20-model ensemble)
        L_iters dict E -> (n_iter, n_snap) bootstrap-resampled L_E
    """
    files = sorted((DATA / "bootstrap").glob(f"{strat}_iter*.npz"))
    assert files, f"no bootstrap files for {strat}"
    Ls, sizes_ref, tokens_ref = [], None, None
    L20_ref = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        if sizes_ref is None:
            sizes_ref = d["sizes"]
            tokens_ref = d["tokens"]
        else:
            assert np.array_equal(d["sizes"], sizes_ref)
            assert np.array_equal(d["tokens"], tokens_ref)
        L = d["L"].astype(np.float64)        # (n_E, n_snap)
        # find E=20 row
        i20 = int(np.where(sizes_ref == E_BASELINE)[0][0])
        if L20_ref is None:
            L20_ref = L[i20].copy()
        # all iters should share the same L_20 — verify
        assert np.allclose(L[i20], L20_ref), \
            f"L_20 differs across iters for {strat}: file={f.name}"
        Ls.append(L)
    Ls = np.stack(Ls, axis=0)                # (n_iter, n_E, n_snap)
    s = tokens_ref.astype(np.float64) / TOTAL_BATCH
    L_iters = {}
    for i, E in enumerate(sizes_ref):
        if int(E) == E_BASELINE:
            continue
        L_iters[int(E)] = Ls[:, i, :]        # (n_iter, n_snap)
    return s, L20_ref, L_iters


def fit_one_iter(L_iter_dict_at_k: dict, s: np.ndarray, L20: np.ndarray):
    """Fit advisor's form on one bootstrap iter. L_iter_dict_at_k: E -> (n_snap,)."""
    rows_E, rows_s, rows_dL = [], [], []
    for E, L in L_iter_dict_at_k.items():
        dL = L - L20
        rows_E.append(np.full_like(s, E, dtype=float))
        rows_s.append(s)
        rows_dL.append(dL)
    E_arr = np.concatenate(rows_E)
    s_arr = np.concatenate(rows_s)
    dL_arr = np.concatenate(rows_dL)
    pos = dL_arr > 0
    return fit_loglog(E_arr[pos], s_arr[pos], dL_arr[pos])


def fit_bootstrap(strat: str):
    """Per-iter fits; return point estimate (pooled) + bootstrap mean/SE/CI."""
    s, L20, L_iters = load_bootstrap(strat)
    n_iter = next(iter(L_iters.values())).shape[0]

    # per-iteration fits
    per_iter = []
    for k in range(n_iter):
        snap = {E: L[k] for E, L in L_iters.items()}
        per_iter.append(fit_one_iter(snap, s, L20))
    deltas_E = np.array([f["delta_E"] for f in per_iter])
    deltas_s = np.array([f["delta_s"] for f in per_iter])
    Cs       = np.array([f["C"]       for f in per_iter])

    # pooled fit (point estimate using all iters together)
    rows_E, rows_s, rows_dL = [], [], []
    for k in range(n_iter):
        for E, L in L_iters.items():
            dL = L[k] - L20
            rows_E.append(np.full_like(s, E, dtype=float))
            rows_s.append(s)
            rows_dL.append(dL)
    E_arr = np.concatenate(rows_E)
    s_arr = np.concatenate(rows_s)
    dL_arr = np.concatenate(rows_dL)
    pos = dL_arr > 0
    pooled = fit_loglog(E_arr[pos], s_arr[pos], dL_arr[pos])

    return dict(
        s=s, L20=L20, L_iters=L_iters,
        pooled=pooled,
        per_iter=dict(C=Cs, delta_E=deltas_E, delta_s=deltas_s),
        boot_mean_delta_E=float(deltas_E.mean()),
        boot_se_delta_E  =float(deltas_E.std(ddof=1)),
        boot_ci_delta_E  =tuple(np.quantile(deltas_E, [0.025, 0.975])),
        boot_mean_delta_s=float(deltas_s.mean()),
        boot_se_delta_s  =float(deltas_s.std(ddof=1)),
        boot_ci_delta_s  =tuple(np.quantile(deltas_s, [0.025, 0.975])),
        boot_mean_C      =float(Cs.mean()),
        boot_se_C        =float(Cs.std(ddof=1)),
        n_iter=n_iter,
    )


def load_strategy(strat: str):
    """Return (s, L20, L_E_dict, L_indiv_array) for one strategy."""
    base = np.load(DATA / "ensembles" / f"{strat}_E{E_BASELINE}.npz", allow_pickle=True)
    tokens = base["tokens"].astype(np.int64)
    s = tokens / TOTAL_BATCH
    L20 = base["val_loss"].astype(np.float64)

    L_E = {}
    for E in E_ENS:
        d = np.load(DATA / "ensembles" / f"{strat}_E{E}.npz", allow_pickle=True)
        assert np.array_equal(d["tokens"], tokens), f"token grid mismatch at E={E}"
        L_E[E] = d["val_loss"].astype(np.float64)

    indiv_paths = sorted((DATA / "individuals").glob(f"{strat}_model*.npz"),
                         key=lambda p: int(p.stem.split("model")[-1]))
    L_indiv = []
    for p in indiv_paths:
        d = np.load(p, allow_pickle=True)
        assert np.array_equal(d["tokens"], tokens), f"token grid mismatch at {p.name}"
        L_indiv.append(d["val_loss"].astype(np.float64))
    L_indiv = np.stack(L_indiv, axis=0)  # (n_indiv, n_snap)

    return s, L20, L_E, L_indiv


def build_fit_table(s: np.ndarray, L20: np.ndarray, L_E: dict, L_indiv: np.ndarray):
    """Return (E_arr, s_arr, dL_arr, weight_arr) of all positive-δL points.

    Each individual contributes an (E=1, s) point per snapshot; each ensemble
    file contributes one (E, s) point per snapshot. We give every row weight=1
    so individuals' 20 replicates naturally up-weight the E=1 anchor.
    """
    rows_E, rows_s, rows_dL = [], [], []
    # E = 1: 20 individuals × n_snap rows
    for i in range(L_indiv.shape[0]):
        dL = L_indiv[i] - L20
        rows_E.append(np.ones_like(s))
        rows_s.append(s)
        rows_dL.append(dL)
    # E ∈ {2,5,10,15}: 1 row per snap
    for E, L in L_E.items():
        dL = L - L20
        rows_E.append(np.full_like(s, E, dtype=float))
        rows_s.append(s)
        rows_dL.append(dL)

    E_arr = np.concatenate(rows_E)
    s_arr = np.concatenate(rows_s)
    dL_arr = np.concatenate(rows_dL)
    pos = dL_arr > 0
    return E_arr[pos], s_arr[pos], dL_arr[pos]


def fit_loglog(E: np.ndarray, s: np.ndarray, dL: np.ndarray):
    """OLS:  log dL = a0 + a_E · log E + a_s · log s.
    Returns dict with C, delta_E, delta_s, R2, residual std, n.
    """
    y = np.log(dL)
    X = np.column_stack([np.ones_like(y), np.log(E), np.log(s)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a0, a_E, a_s = beta
    yhat = X @ beta
    resid = y - yhat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return dict(
        C=float(np.exp(a0)),
        delta_E=float(-a_E),
        delta_s=float(a_s),
        R2=1.0 - ss_res / ss_tot,
        rmse_log=float(np.sqrt(ss_res / len(y))),
        n=int(len(y)),
    )




def predict(fit: dict, E: np.ndarray, s: np.ndarray) -> np.ndarray:
    return fit["C"] * np.power(E, -fit["delta_E"]) * np.power(s, fit["delta_s"])


# ---------- plotting ----------

def setup_style():
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["axes.linewidth"] = 4.0
    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18


def plot_dL_vs_E(ax, s, L20, L_iters, L_indiv, fit, s_picks):
    """δL vs E at a few s slices. Uses bootstrap iters for E∈{2,5,10,15} spread."""
    palette = sns.color_palette("cool", len(s_picks))
    snap_idx = [int(np.argmin(np.abs(s - sp))) for sp in s_picks]

    E_axis = np.array([1, 2, 5, 10, 15], dtype=float)
    E_grid = np.geomspace(1, 16, 64)
    for c, k in zip(palette, snap_idx):
        s_k = s[k]
        # E=1 cluster from individuals
        dL_indiv = L_indiv[:, k] - L20[k]
        ax.scatter(np.ones(20), dL_indiv, s=18, color=c, alpha=0.30,
                   edgecolor="none", zorder=2)
        ax.scatter([1.0], [dL_indiv.mean()], s=80, color=c,
                   edgecolor="black", linewidth=1.0, zorder=4)
        # E∈{2,5,10,15}: scatter all bootstrap iters + mean marker
        for E, L_kE in L_iters.items():
            dL_iters = L_kE[:, k] - L20[k]
            ax.scatter(np.full_like(dL_iters, E), dL_iters, s=18, color=c,
                       alpha=0.30, edgecolor="none", zorder=2)
            ax.scatter([E], [dL_iters.mean()], s=80, color=c,
                       edgecolor="black", linewidth=1.0, zorder=4)
        ax.loglog(E_grid, predict(fit, E_grid, np.full_like(E_grid, s_k)),
                  "--", color=c, lw=2.5,
                  label=f"$s = {int(s_k):,}$")

    ax.set_xlabel(r"$E$ (ensemble size)")
    ax.set_ylabel(r"$\delta L = L_E - L_{20}$")
    ax.set_xticks(E_axis)
    ax.set_xticklabels([str(int(e)) for e in E_axis])
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, title=None)


def plot_dL_vs_s(ax, s, L20, L_iters, L_indiv, fit, E_picks):
    palette = sns.color_palette("cool", len(E_picks))
    s_grid = np.geomspace(s.min(), s.max(), 128)
    all_pos = []
    for c, E in zip(palette, E_picks):
        if E == 1:
            stack = L_indiv - L20[None, :]
        else:
            stack = L_iters[E] - L20[None, :]        # (n_iter, n_snap)
        # log-space mean & std so band stays positive and symmetric on log axis
        log_stack = np.log(np.where(stack > 0, stack, np.nan))
        log_mean = np.nanmean(log_stack, axis=0)
        log_std = np.nanstd(log_stack, axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros_like(log_mean)
        mean_dL = np.exp(log_mean)
        lo = np.exp(log_mean - log_std)
        hi = np.exp(log_mean + log_std)
        mask = np.isfinite(log_mean)
        ax.loglog(s[mask], mean_dL[mask], "o", color=c, ms=6,
                  markeredgecolor="black", markeredgewidth=0.6, zorder=3,
                  label=f"$E = {E}$")
        ax.fill_between(s[mask], lo[mask], hi[mask], color=c, alpha=0.18)
        ax.loglog(s_grid, predict(fit, np.full_like(s_grid, E), s_grid),
                  "--", color=c, lw=2.5)
        all_pos.append(mean_dL[mask])

    ax.set_xlabel(r"$s$ (optimizer steps)")
    ax.set_ylabel(r"$\delta L = L_E - L_{20}$")
    # tight y-limits driven by mean δL only (ignore band tails)
    ymin = max(1e-3, 0.5 * float(np.min([a.min() for a in all_pos])))
    ymax = 2.0 * float(np.max([a.max() for a in all_pos]))
    ax.set_ylim(ymin, ymax)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, title=None,
              ncol=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "experiments/analysis/expt5_dL_fit"))
    args = ap.parse_args()

    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    fits = {}
    for row, strat in enumerate(STRATS):
        # individuals (E=1) for plot context only — not in the fit
        _, _, _, L_indiv = load_strategy(strat)
        boot = fit_bootstrap(strat)
        s, L20, L_iters = boot["s"], boot["L20"], boot["L_iters"]
        pooled = boot["pooled"]
        fits[strat] = boot

        print(f"[{strat}]  n_iter={boot['n_iter']}  pooled n={pooled['n']}  "
              f"C={pooled['C']:.4g}  "
              f"δ_E={pooled['delta_E']:.3f}  δ_s={pooled['delta_s']:.3f}  "
              f"R²={pooled['R2']:.3f}")
        print(f"           bootstrap (per-iter, mean ± SD across {boot['n_iter']} iters):")
        print(f"             δ_E = {boot['boot_mean_delta_E']:.3f} ± {boot['boot_se_delta_E']:.3f}  "
              f"95% CI = [{boot['boot_ci_delta_E'][0]:.3f}, {boot['boot_ci_delta_E'][1]:.3f}]")
        print(f"             δ_s = {boot['boot_mean_delta_s']:.3f} ± {boot['boot_se_delta_s']:.3f}  "
              f"95% CI = [{boot['boot_ci_delta_s'][0]:.3f}, {boot['boot_ci_delta_s'][1]:.3f}]")
        print(f"             C   = {boot['boot_mean_C']:.4g} ± {boot['boot_se_C']:.4g}")

        s_picks = np.quantile(s, [0.20, 0.50, 0.80, 1.00])
        plot_dL_vs_E(axes[row, 0], s, L20, L_iters, L_indiv, pooled, s_picks)
        plot_dL_vs_s(axes[row, 1], s, L20, L_iters, L_indiv, pooled,
                     E_picks=(1, 2, 5, 10, 15))

        eq = (rf"  $\delta_E = {pooled['delta_E']:.2f}\pm{boot['boot_se_delta_E']:.2f}$,  "
              rf"$\delta_s = {pooled['delta_s']:.2f}\pm{boot['boot_se_delta_s']:.2f}$,  "
              rf"$R^2 = {pooled['R2']:.3f}$")
        axes[row, 0].set_title(strat, fontsize=18, loc="left", fontweight="bold")
        axes[row, 1].set_title(eq, fontsize=16, loc="left")

    fig.suptitle(r"Expt 5: $\delta L = L_E - L_{20} \approx C\,E^{-\delta_E}\,s^{\delta_s}$"
                 f"   (d=12, w=768, df=0.2, 40 epochs; bootstrap n={fits[STRATS[0]]['n_iter']})",
                 fontsize=22, y=0.995)
    fig.tight_layout()

    out_pdf = Path(args.out).with_suffix(".pdf")
    out_png = Path(args.out).with_suffix(".png")
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"saved {out_pdf}")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
