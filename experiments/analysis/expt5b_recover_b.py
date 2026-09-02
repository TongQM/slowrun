"""Expt 5b — Refit with explicit floor:  L_E(s) = a(s) · E^(-b) + L_∞(s).

The advisor's δL = L_E − L_20 ≈ C·E^(-δ_E)·s^(δ_s) form yields δ_E ≈ 1.5–1.6,
but that exponent is biased high because L_20 is not the true asymptote — the
subtraction implicit in `L_E − L_20 = a(E^(-b) − 20^(-b))` curves the log-log
plot downward and inflates the apparent slope.

This script refits without the bias by leaving the floor free:
  - per-snapshot fit:  L_E(s_j) = a_j · E^(-b_j) + L_∞,j   (3 params per snapshot)
  - joint fit:         shared b across all snapshots, per-snap (a_j, L_∞,j)

Data leverage at each snapshot:
  E ∈ {1}   from 20 individuals    (one observation per individual)
  E ∈ {2, 5, 10, 15, 20}   from each of 10 bootstrap iters
  → 20 + 5×10 = 70 observations per snapshot.
L_20 is the same across iters by construction (full 20-model ensemble).

Bootstrap SE on the recovered b: refit the joint fit on each of the 10 iters
separately and take SD across iters.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data_export" / "expt2_ensemble"
TOTAL_BATCH = 131072
STRATS = ("init_ens", "init_shuffle_ens")
E_BASELINE = 20
E_BOOT = (2, 5, 10, 15)


# ---------- data loading (mirrors expt5_dL_vs_E_s.py) ----------

def load_individuals(strat: str):
    paths = sorted((DATA / "individuals").glob(f"{strat}_model*.npz"),
                   key=lambda p: int(p.stem.split("model")[-1]))
    Ls = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        Ls.append(d["val_loss"].astype(np.float64))
    return np.stack(Ls, axis=0)  # (20, n_snap)


def load_bootstrap(strat: str):
    files = sorted((DATA / "bootstrap").glob(f"{strat}_iter*.npz"))
    assert files, f"no bootstrap files for {strat}"
    Ls, sizes_ref, tokens_ref, L20_ref = [], None, None, None
    for f in files:
        d = np.load(f, allow_pickle=True)
        if sizes_ref is None:
            sizes_ref = d["sizes"]
            tokens_ref = d["tokens"]
        L = d["L"].astype(np.float64)
        i20 = int(np.where(sizes_ref == E_BASELINE)[0][0])
        if L20_ref is None:
            L20_ref = L[i20].copy()
        Ls.append(L)
    Ls = np.stack(Ls, axis=0)             # (n_iter, n_E, n_snap)
    s = tokens_ref.astype(np.float64) / TOTAL_BATCH
    L_iters = {int(E): Ls[:, i, :] for i, E in enumerate(sizes_ref)
               if int(E) != E_BASELINE}
    return s, L20_ref, L_iters


# ---------- fits ----------

def per_snapshot_obs(j: int, L20, L_iters, L_indiv):
    """Return (E_arr, L_arr) of all observations at snapshot j."""
    E_pts, L_pts = [], []
    # E = 1: 20 individuals
    for i in range(L_indiv.shape[0]):
        E_pts.append(1.0)
        L_pts.append(L_indiv[i, j])
    # E ∈ {2, 5, 10, 15}: bootstrap iters
    for E, L_arr in L_iters.items():
        for k in range(L_arr.shape[0]):
            E_pts.append(float(E))
            L_pts.append(L_arr[k, j])
    # E = 20: deterministic
    E_pts.append(float(E_BASELINE))
    L_pts.append(L20[j])
    return np.array(E_pts), np.array(L_pts)


def fit_one_snapshot(E_pts, L_pts, b_init=1.0):
    """Fit  L = a · E^(-b) + c.  Returns (a, b, c, success)."""
    L_min = L_pts.min()
    a0 = L_pts.max() - L_min
    c0 = L_min
    def resid(params):
        a, b, c = params
        return a * E_pts ** (-b) + c - L_pts
    try:
        r = least_squares(resid, [a0, b_init, c0],
                          bounds=([0.0, 0.05, 0.0], [np.inf, 5.0, np.inf]),
                          max_nfev=5000)
        return float(r.x[0]), float(r.x[1]), float(r.x[2]), bool(r.success)
    except Exception:
        return np.nan, np.nan, np.nan, False


def fit_joint(s, L20, L_iters, L_indiv, b_init=1.0):
    """Joint fit: one shared b, per-snapshot (a_j, c_j)."""
    n_snap = len(s)

    # flatten all observations
    rows_E, rows_L, rows_j = [], [], []
    for j in range(n_snap):
        Ep, Lp = per_snapshot_obs(j, L20, L_iters, L_indiv)
        rows_E.append(Ep)
        rows_L.append(Lp)
        rows_j.append(np.full_like(Ep, j, dtype=int))
    E_arr = np.concatenate(rows_E)
    L_arr = np.concatenate(rows_L)
    j_arr = np.concatenate(rows_j)

    # initial per-snap (a_j, c_j) from the per-snapshot fits at b_init
    a_init = np.zeros(n_snap)
    c_init = np.zeros(n_snap)
    for j in range(n_snap):
        Ep, Lp = per_snapshot_obs(j, L20, L_iters, L_indiv)
        a_init[j], _, c_init[j], _ = fit_one_snapshot(Ep, Lp, b_init=b_init)
    params0 = np.concatenate([[b_init], a_init, c_init])

    def residuals(params):
        b = params[0]
        a = params[1:1+n_snap]
        c = params[1+n_snap:]
        return a[j_arr] * E_arr ** (-b) + c[j_arr] - L_arr

    lo = np.concatenate([[0.05], np.zeros(n_snap), np.zeros(n_snap)])
    hi = np.concatenate([[5.0],  np.full(n_snap, np.inf), np.full(n_snap, np.inf)])
    r = least_squares(residuals, params0, bounds=(lo, hi), max_nfev=20000)
    b = float(r.x[0])
    a = r.x[1:1+n_snap].copy()
    c = r.x[1+n_snap:].copy()
    # residual stats
    yhat = a[j_arr] * E_arr ** (-b) + c[j_arr]
    rss = float(np.sum((L_arr - yhat) ** 2))
    return dict(b=b, a=a, Linf=c, rss=rss, success=bool(r.success))


def fit_joint_one_iter(s, L20, L_iters, L_indiv, iter_k):
    """Same joint fit but using only iter_k of the bootstrap (E∈{2,5,10,15})."""
    L_iters_k = {E: L[iter_k:iter_k+1] for E, L in L_iters.items()}
    return fit_joint(s, L20, L_iters_k, L_indiv)


# ---------- plotting ----------

def setup_style():
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams["axes.labelsize"] = 22
    plt.rcParams["axes.linewidth"] = 4.0
    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "experiments/analysis/expt5b_b_recovery"))
    args = ap.parse_args()

    setup_style()

    results = {}
    for strat in STRATS:
        s, L20, L_iters = load_bootstrap(strat)
        L_indiv = load_individuals(strat)

        # per-snapshot fits (point-estimate using all 70 obs)
        n_snap = len(s)
        b_per = np.zeros(n_snap)
        a_per = np.zeros(n_snap)
        c_per = np.zeros(n_snap)
        ok_per = np.zeros(n_snap, dtype=bool)
        for j in range(n_snap):
            Ep, Lp = per_snapshot_obs(j, L20, L_iters, L_indiv)
            a_per[j], b_per[j], c_per[j], ok_per[j] = fit_one_snapshot(Ep, Lp)

        # joint shared-b fit
        joint = fit_joint(s, L20, L_iters, L_indiv)

        # bootstrap SE on shared b: refit using each iter alone (E∈{2,5,10,15})
        b_iter = []
        for k in range(next(iter(L_iters.values())).shape[0]):
            j_k = fit_joint_one_iter(s, L20, L_iters, L_indiv, k)
            b_iter.append(j_k["b"])
        b_iter = np.array(b_iter)
        b_boot_mean = float(b_iter.mean())
        b_boot_se = float(b_iter.std(ddof=1))
        b_boot_ci = tuple(np.quantile(b_iter, [0.025, 0.975]))

        results[strat] = dict(
            s=s, L20=L20, L_iters=L_iters, L_indiv=L_indiv,
            b_per=b_per, a_per=a_per, c_per=c_per, ok_per=ok_per,
            joint=joint,
            b_iter=b_iter, b_boot_mean=b_boot_mean, b_boot_se=b_boot_se,
            b_boot_ci=b_boot_ci,
        )

        print(f"[{strat}]")
        print(f"  per-snapshot b:  median={np.nanmedian(b_per):.3f}  "
              f"mean={np.nanmean(b_per):.3f}  std={np.nanstd(b_per):.3f}  "
              f"min={np.nanmin(b_per):.3f}  max={np.nanmax(b_per):.3f}")
        print(f"  joint b (shared across snapshots): {joint['b']:.3f}")
        print(f"  bootstrap (refit per iter, n=10): "
              f"b = {b_boot_mean:.3f} ± {b_boot_se:.3f}   "
              f"95% CI = [{b_boot_ci[0]:.3f}, {b_boot_ci[1]:.3f}]")

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # Top row: data + fit + b=1 overlay at the LAST snapshot, one panel per strat
    for col, strat in enumerate(STRATS):
        ax = fig.add_subplot(gs[0, col])
        r = results[strat]
        s, L20, L_iters, L_indiv = r["s"], r["L20"], r["L_iters"], r["L_indiv"]
        j = len(s) - 1                           # last snapshot

        # observations (mean across iters at each E)
        Es_obs = np.array([1] + list(E_BOOT) + [E_BASELINE], dtype=float)
        L_obs_mean = np.array(
            [L_indiv[:, j].mean()]
            + [L_iters[E][:, j].mean() for E in E_BOOT]
            + [L20[j]]
        )
        L_obs_std = np.array(
            [L_indiv[:, j].std(ddof=1)]
            + [L_iters[E][:, j].std(ddof=1) for E in E_BOOT]
            + [0.0]
        )
        ax.errorbar(Es_obs, L_obs_mean, yerr=L_obs_std, fmt="o", ms=8,
                    color="C0", ecolor="C0", capsize=3, zorder=4,
                    markeredgecolor="black", markeredgewidth=0.8,
                    label="data (mean ± SD)")

        # recovered fit at this snapshot (per-snap)
        E_grid = np.geomspace(1, 25, 200)
        a_j, b_j, c_j = r["a_per"][j], r["b_per"][j], r["c_per"][j]
        ax.plot(E_grid, a_j * E_grid ** (-b_j) + c_j, "--", lw=3.0,
                color="C3", label=fr"fit: $b = {b_j:.2f}$,  $L_\infty = {c_j:.3f}$")

        # b=1 overlay using a',c' refit at fixed b=1
        Ep, Lp = per_snapshot_obs(j, L20, L_iters, L_indiv)
        def resid_b1(p):
            a, c = p
            return a * Ep ** (-1.0) + c - Lp
        rb1 = least_squares(resid_b1, [a_j, c_j],
                            bounds=([0.0, 0.0], [np.inf, np.inf]))
        a_b1, c_b1 = rb1.x
        ax.plot(E_grid, a_b1 * E_grid ** (-1.0) + c_b1, ":", lw=3.0,
                color="C2", label=fr"$b = 1$ overlay,  $L_\infty = {c_b1:.3f}$")

        # joint shared-b at this snapshot
        a_jt, c_jt = r["joint"]["a"][j], r["joint"]["Linf"][j]
        b_jt = r["joint"]["b"]
        ax.plot(E_grid, a_jt * E_grid ** (-b_jt) + c_jt, "-.", lw=2.5,
                color="C1",
                label=fr"joint fit: $b = {b_jt:.2f}$ (shared)")

        ax.set_xscale("log")
        ax.set_xlabel(r"$E$ (ensemble size)")
        ax.set_ylabel(r"$L_E$")
        ax.set_title(strat + f"   (s = {int(s[j]):,})", fontsize=18, loc="left")
        ax.legend(loc="upper right", frameon=True, framealpha=0.92)

    # Bottom: b vs s, both strategies on one panel
    ax = fig.add_subplot(gs[1, :])
    palette = sns.color_palette("cool", 2)
    for c, strat in zip(palette, STRATS):
        r = results[strat]
        s = r["s"]
        ok = r["ok_per"]
        ax.plot(s[ok], r["b_per"][ok], "o-", ms=5, lw=2.0, color=c, alpha=0.85,
                label=f"{strat}   per-snap  (median = {np.nanmedian(r['b_per'][ok]):.2f})")
        # joint b and bootstrap CI band
        b_jt = r["joint"]["b"]
        ax.axhline(b_jt, color=c, lw=2.5, ls="--", alpha=0.9,
                   label=fr"{strat}   joint $b = {b_jt:.2f} \pm {r['b_boot_se']:.2f}$ "
                         fr"(boot 95% CI [{r['b_boot_ci'][0]:.2f}, {r['b_boot_ci'][1]:.2f}])")
        ax.axhspan(r["b_boot_ci"][0], r["b_boot_ci"][1], color=c, alpha=0.10)
    ax.axhline(1.0, color="black", lw=2.0, ls=":", alpha=0.7, label=r"$b = 1$ reference")
    ax.set_xscale("log")
    ax.set_xlabel(r"$s$ (optimizer steps)")
    ax.set_ylabel(r"recovered $b$")
    ax.set_ylim(0, max(2.5, ax.get_ylim()[1]))
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, ncol=1, fontsize=12)
    ax.set_title(r"Recovered $b$ from $L_E = a\,E^{-b} + L_\infty$ vs. $b=1$ reference",
                 fontsize=18, loc="left")

    fig.suptitle(r"Expt 5b: recover $b$ by fitting the floor explicitly  "
                 r"(d=12, w=768, df=0.2, 40 epochs)", fontsize=22, y=0.995)

    out_pdf = Path(args.out).with_suffix(".pdf")
    out_png = Path(args.out).with_suffix(".png")
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"saved {out_pdf}")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
