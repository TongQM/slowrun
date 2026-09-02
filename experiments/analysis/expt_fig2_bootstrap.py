"""Expt Fig 2 — bootstrap version.

Same model as `expt_fig2_ensemble_scaling.py`:

    L_{T,E} = a_T · E^{-b} + L_{∞,T}

with shared L_∞,T across strategies and per-strategy (b, a_T). The
**only** change is uncertainty quantification: instead of the Jacobian
asymptotic SEs, we refit per bootstrap iteration to get a sampling
distribution over (b, a_T, L_∞,T) under resampling of the 20-model pool.

Bootstrap structure (per iter k):
  boot_indices_k   ← shared between strategies (verified)
  E = 1: mean over the bootstrap-sampled individuals at every snapshot
  E ∈ {2, 5, 10, 15, 20}: read from iter k's stored ensemble L_E

So each iter gives a coherent (init, shuffle) pair on the same pool.
We refit the joint NLLS on each iter and report:
  b           = mean ± SD over iters  (95% CI from quantiles)
  a_T(snap)   = mean ± SD per snapshot
  L_∞,T(snap) = mean ± SD per snapshot

Outputs:
  experiments/figures/02_ensemble_scaling/bootstrap/expt_fig2_bootstrap.{pdf,png}
  experiments/figures/02_ensemble_scaling/bootstrap/expt_fig2_bootstrap_fits.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data_export" / "expt2_ensemble"
OUT_DIR = REPO / "experiments" / "figures" / "02_ensemble_scaling" / "bootstrap"

STRATS = ["init_ens", "init_shuffle_ens"]
STRAT_LABEL = {"init_ens": "init", "init_shuffle_ens": "init + shuffle"}
ENS_E = np.array([2, 5, 10, 15, 20], dtype=float)
ALL_E = np.array([1, 2, 5, 10, 15, 20], dtype=float)
E_BASE = 20
BATCH_SIZE = 131072                                   # tokens → steps conversion
S_FIT_MIN = 1000                                       # ignore s < S_FIT_MIN in α_s power-law fit
                                                       # (pre-power-law transient)


# ---------------------------- data ----------------------------

def load_individuals(strat: str):
    paths = sorted((DATA / "individuals").glob(f"{strat}_model*.npz"),
                   key=lambda p: int(p.stem.split("model")[-1]))
    Ls, tok_ref = [], None
    for p in paths:
        d = np.load(p, allow_pickle=True)
        tok = d["tokens"].astype(np.int64)
        if tok_ref is None:
            tok_ref = tok
        else:
            assert np.array_equal(tok, tok_ref), f"individual {p} grid mismatch"
        Ls.append(d["val_loss"].astype(np.float64))
    return np.stack(Ls, axis=0), tok_ref      # (20, n_T), (n_T,)


def load_bootstrap_iter(strat: str, k: int):
    f = DATA / "bootstrap" / f"{strat}_iter{k:03d}.npz"
    d = np.load(f, allow_pickle=True)
    return dict(L=d["L"].astype(np.float64),                    # (5, n_T)
                tokens=d["tokens"].astype(np.int64),
                sizes=d["sizes"].astype(int),
                boot_indices=d["boot_indices"].astype(int))


def build_grid_for_iter(L_indiv_strat: np.ndarray,
                        boot_indices: np.ndarray,
                        L_iter: np.ndarray,
                        sizes_iter: np.ndarray):
    """Return L_grid (6, n_T) for E ∈ ALL_E, this iter."""
    n_T = L_indiv_strat.shape[1]
    rows = []
    # E=1: bootstrap-sampled individuals' mean
    e1 = L_indiv_strat[boot_indices].mean(axis=0)              # (n_T,)
    rows.append(e1)
    for E in (2, 5, 10, 15, 20):
        i = int(np.where(sizes_iter == E)[0][0])
        rows.append(L_iter[i])
    return np.stack(rows, axis=0)                               # (6, n_T)


# ---------------------------- joint NLLS ----------------------------

def joint_fit(L_init: np.ndarray, L_shuf: np.ndarray, E_sizes=ALL_E,
              b_fixed: float | None = None):
    """Returns dict with b_init/b_shuffle/a_T_*/Linf_T.
    If `b_fixed` is given, both strategies use that β and only (a_T, L_∞,T) are fit.
    """
    n_E, n_T = L_init.shape
    E = E_sizes.reshape(-1, 1)
    a0 = np.full(n_T, 1.0)
    Linf0 = np.minimum(L_init.min(0), L_shuf.min(0)) - 0.05
    Linf0 = np.clip(Linf0, 0.0, None)

    if b_fixed is None:
        def residuals(p):
            b_i, b_s = p[0], p[1]
            a_i = p[2:2 + n_T]
            a_s = p[2 + n_T:2 + 2 * n_T]
            Linf = p[2 + 2 * n_T:2 + 3 * n_T]
            pred_i = a_i[None, :] * (E ** (-b_i)) + Linf[None, :]
            pred_s = a_s[None, :] * (E ** (-b_s)) + Linf[None, :]
            return np.concatenate([(pred_i - L_init).ravel(),
                                   (pred_s - L_shuf).ravel()])
        p0 = np.concatenate([[1.0, 1.0], a0, a0, Linf0])
        lo = np.concatenate([[0.05, 0.05], np.zeros(2 * n_T), np.zeros(n_T)])
        hi = np.concatenate([[5.0, 5.0], np.full(2 * n_T, np.inf), np.full(n_T, np.inf)])
        res = least_squares(residuals, p0, bounds=(lo, hi), max_nfev=20000)
        return dict(
            b_init=float(res.x[0]),
            b_shuffle=float(res.x[1]),
            a_T_init=res.x[2:2 + n_T].copy(),
            a_T_shuffle=res.x[2 + n_T:2 + 2 * n_T].copy(),
            Linf_T=res.x[2 + 2 * n_T:2 + 3 * n_T].copy(),
        )
    else:
        b = float(b_fixed)
        def residuals(p):
            a_i = p[0:n_T]
            a_s = p[n_T:2 * n_T]
            Linf = p[2 * n_T:3 * n_T]
            pred_i = a_i[None, :] * (E ** (-b)) + Linf[None, :]
            pred_s = a_s[None, :] * (E ** (-b)) + Linf[None, :]
            return np.concatenate([(pred_i - L_init).ravel(),
                                   (pred_s - L_shuf).ravel()])
        p0 = np.concatenate([a0, a0, Linf0])
        lo = np.concatenate([np.zeros(2 * n_T), np.zeros(n_T)])
        hi = np.concatenate([np.full(2 * n_T, np.inf), np.full(n_T, np.inf)])
        res = least_squares(residuals, p0, bounds=(lo, hi), max_nfev=20000)
        return dict(
            b_init=b, b_shuffle=b,
            a_T_init=res.x[0:n_T].copy(),
            a_T_shuffle=res.x[n_T:2 * n_T].copy(),
            Linf_T=res.x[2 * n_T:3 * n_T].copy(),
        )


# ---------------------------- bootstrap driver ----------------------------

def bootstrap_fits(b_fixed: float | None = None):
    inds = {s: load_individuals(s) for s in STRATS}
    L_indiv = {s: inds[s][0] for s in STRATS}
    tok_ref = inds["init_ens"][1]
    n_T = len(tok_ref)

    fits = []
    for k in range(10):
        ki = load_bootstrap_iter("init_ens", k)
        ks = load_bootstrap_iter("init_shuffle_ens", k)
        assert np.array_equal(ki["boot_indices"], ks["boot_indices"]), \
            f"iter {k} boot_indices differ across strategies"
        boot = ki["boot_indices"]
        L_init = build_grid_for_iter(L_indiv["init_ens"], boot,
                                     ki["L"], ki["sizes"])
        L_shuf = build_grid_for_iter(L_indiv["init_shuffle_ens"], boot,
                                     ks["L"], ks["sizes"])
        f = joint_fit(L_init, L_shuf, b_fixed=b_fixed)
        f["L_init_grid"] = L_init
        f["L_shuf_grid"] = L_shuf
        fits.append(f)

    # aggregate
    b_init = np.array([f["b_init"] for f in fits])
    b_shuf = np.array([f["b_shuffle"] for f in fits])
    a_init = np.stack([f["a_T_init"] for f in fits], 0)         # (10, n_T)
    a_shuf = np.stack([f["a_T_shuffle"] for f in fits], 0)
    Linf   = np.stack([f["Linf_T"]    for f in fits], 0)

    summary = dict(
        b_init_mean=float(b_init.mean()),
        b_init_sd=float(b_init.std(ddof=1)),
        b_init_ci=tuple(np.quantile(b_init, [0.025, 0.975])),
        b_shuf_mean=float(b_shuf.mean()),
        b_shuf_sd=float(b_shuf.std(ddof=1)),
        b_shuf_ci=tuple(np.quantile(b_shuf, [0.025, 0.975])),
        b_init_iters=b_init,
        b_shuf_iters=b_shuf,
        a_init_mean=a_init.mean(0),
        a_init_sd=a_init.std(0, ddof=1),
        a_shuf_mean=a_shuf.mean(0),
        a_shuf_sd=a_shuf.std(0, ddof=1),
        Linf_mean=Linf.mean(0),
        Linf_sd=Linf.std(0, ddof=1),
        L_init_iters=np.stack([f["L_init_grid"] for f in fits], 0),  # (10, 6, n_T)
        L_shuf_iters=np.stack([f["L_shuf_grid"] for f in fits], 0),
        tok_ref=tok_ref,
        n_iter=len(fits),
    )
    return summary


# ---------------------------- power-law a_T(T) helper ----------------------------

def power_law_fit(x, y, y_se=None):
    mask = (y > 0) & np.isfinite(x) & np.isfinite(y)
    if y_se is not None:
        mask &= (y_se > 0) & np.isfinite(y_se)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    lx, ly = np.log(x[mask]), np.log(y[mask])
    if y_se is not None:
        w = (y[mask] / y_se[mask]) ** 2
        Lx = np.column_stack([lx, np.ones_like(lx)])
        XtWX = Lx.T @ (w[:, None] * Lx); XtWy = Lx.T @ (w * ly)
        try:
            beta = np.linalg.solve(XtWX, XtWy)
            cov = np.linalg.inv(XtWX) * (np.sum(w * (ly - Lx @ beta) ** 2)
                                          / max(len(lx) - 2, 1))
            p, log_c = beta; p_se = np.sqrt(cov[0, 0])
        except np.linalg.LinAlgError:
            p, log_c = np.polyfit(lx, ly, 1); p_se = np.nan
    else:
        p, log_c = np.polyfit(lx, ly, 1); p_se = np.nan
    return float(np.exp(log_c)), float(p), float(p_se)


# ---------------------------- figure ----------------------------

def make_figure(summary, snap_epochs=(5, 10, 15, 25, 40)):
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 24, "axes.linewidth": 4.0, "legend.fontsize": 18,
        "grid.alpha": 0.25, "xtick.labelsize": 20, "ytick.labelsize": 20,
    })

    fig = plt.figure(figsize=(24, 7))
    gs = fig.add_gridspec(1, 3, wspace=0.32)
    ax_L = fig.add_subplot(gs[0, 0])
    ax_M = fig.add_subplot(gs[0, 1], sharey=ax_L)
    ax_R = fig.add_subplot(gs[0, 2])

    palette_T = sns.color_palette("cool", len(snap_epochs))
    palette_S = sns.color_palette("cool", 2)
    strat_marker = {"init_ens": "o", "init_shuffle_ens": "s"}

    L_iters_by_strat = {
        "init_ens":         summary["L_init_iters"],
        "init_shuffle_ens": summary["L_shuf_iters"],
    }
    a_iters_by_strat = {
        "init_ens":         (summary["a_init_mean"], summary["a_init_sd"]),
        "init_shuffle_ens": (summary["a_shuf_mean"], summary["a_shuf_sd"]),
    }
    b_by_strat = {
        "init_ens":         (summary["b_init_mean"], summary["b_init_sd"], summary["b_init_ci"]),
        "init_shuffle_ens": (summary["b_shuf_mean"], summary["b_shuf_sd"], summary["b_shuf_ci"]),
    }

    for ax_idx, strat in enumerate(STRATS):
        ax = ax_L if ax_idx == 0 else ax_M
        L_iters = L_iters_by_strat[strat]                # (10, 6, n_T)
        steps = summary["tok_ref"].astype(float) / BATCH_SIZE
        Linf_mean = summary["Linf_mean"]
        a_mean, a_sd = a_iters_by_strat[strat]
        b_mean, b_sd, _ = b_by_strat[strat]

        for j, snap_ep in enumerate(snap_epochs):
            idx = int(snap_ep) - 1
            if idx >= L_iters.shape[2]:
                continue
            gap_iters = L_iters[:, :, idx] - Linf_mean[idx]
            log_gap = np.log(np.where(gap_iters > 0, gap_iters, np.nan))
            log_mean = np.nanmean(log_gap, axis=0)
            log_sd   = np.nanstd(log_gap,   axis=0, ddof=1)
            yh = np.exp(log_mean); ylo = np.exp(log_mean - log_sd); yhi = np.exp(log_mean + log_sd)

            ax.errorbar(ALL_E, yh,
                        yerr=[yh - ylo, yhi - yh],
                        fmt=strat_marker[strat], lw=0,
                        color=palette_T[j], markersize=11,
                        markeredgecolor="black", markeredgewidth=0.6,
                        elinewidth=1.0, capsize=3, capthick=1.0,
                        label=fr"$s = {int(steps[idx])}$  (epoch {snap_ep})")
            E_grid = np.geomspace(0.85, ALL_E.max() * 1.1, 100)
            ax.loglog(E_grid, a_mean[idx] * E_grid ** (-b_mean),
                      color=palette_T[j], lw=2.0, ls="-", alpha=0.7)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"ensemble size $E$", fontsize=24)
        if ax_idx == 0:
            ax.set_ylabel(r"$\mathcal{L}_{s,E} - \mathcal{L}_{s,\infty}$",
                          fontsize=28)
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.set_xticklabels(["1", "2", "5", "10", "20"])
        ax.text(0.02, 0.05,
                f"{STRAT_LABEL[strat]}\n"
                fr"$\beta = {b_mean:.2f} \pm {b_sd:.2f}$",
                transform=ax.transAxes, fontsize=18, fontweight="bold",
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.92, edgecolor="lightgray"))
        ax.legend(loc="center left", fontsize=14)

    # Panel C: α_s vs s with bootstrap errorbars + power-law fit
    ax = ax_R
    pT_summary = {}
    s_axis = summary["tok_ref"].astype(float) / BATCH_SIZE
    for j, strat in enumerate(STRATS):
        a_mean, a_sd = a_iters_by_strat[strat]
        c, p, p_se = power_law_fit(s_axis, a_mean, a_sd)
        pT_summary[strat] = dict(c=c, p=p, p_se=p_se)
        ax.errorbar(s_axis, a_mean, yerr=a_sd, fmt=strat_marker[strat], markersize=10,
                    color=palette_S[j], markeredgecolor="black", markeredgewidth=0.5,
                    elinewidth=1.2, capsize=3, capthick=1.2,
                    label=fr"{STRAT_LABEL[strat]}: $\alpha_s \propto s^{{{p:.2f}\pm{p_se:.2f}}}$")
        s_grid = np.geomspace(s_axis.min(), s_axis.max(), 100)
        ax.loglog(s_grid, c * s_grid ** p, lw=2.5, color=palette_S[j],
                  ls="-", alpha=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(s_axis.min() * 0.7, s_axis.max() * 1.4)
    ax.set_xlabel(r"steps $s$", fontsize=28)
    ax.set_ylabel(r"loss-gap prefactor $\alpha_s$", fontsize=24)
    ax.legend(loc="upper left", fontsize=16)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "expt_fig2_bootstrap.pdf"
    png = OUT_DIR / "expt_fig2_bootstrap.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    print(f"\nsaved {pdf}")
    print(f"saved {png}")
    return pT_summary


def main():
    print("running 10 per-iter joint fits ...")
    summary = bootstrap_fits()

    print("\n=== Bootstrap summary (per-iter joint NLLS, n_iter = "
          f"{summary['n_iter']}) ===")
    print(f"  b_init    = {summary['b_init_mean']:.3f} ± {summary['b_init_sd']:.3f}   "
          f"95% CI [{summary['b_init_ci'][0]:.3f}, {summary['b_init_ci'][1]:.3f}]")
    print(f"  b_shuffle = {summary['b_shuf_mean']:.3f} ± {summary['b_shuf_sd']:.3f}   "
          f"95% CI [{summary['b_shuf_ci'][0]:.3f}, {summary['b_shuf_ci'][1]:.3f}]")
    db_iters = summary["b_shuf_iters"] - summary["b_init_iters"]
    print(f"  Δb (shuffle - init)  = {db_iters.mean():+.3f} ± {db_iters.std(ddof=1):.3f}   "
          f"95% CI [{np.quantile(db_iters, 0.025):+.3f}, {np.quantile(db_iters, 0.975):+.3f}]")
    print(f"  per-iter b_init values:    {np.round(summary['b_init_iters'], 3).tolist()}")
    print(f"  per-iter b_shuffle values: {np.round(summary['b_shuf_iters'], 3).tolist()}")

    pT = make_figure(summary)
    print("\n=== Power-law a_T = c · T^p (weighted by bootstrap SE) ===")
    for s in STRATS:
        x = pT[s]
        print(f"  {STRAT_LABEL[s]:<14}  c = {x['c']:.3e},  "
              f"p = {x['p']:.3f} ± {x['p_se']:.3f}")

    csv = OUT_DIR / "expt_fig2_bootstrap_fits.csv"
    rows = ["T_tokens,a_init_mean,a_init_sd,a_shuffle_mean,a_shuffle_sd,Linf_mean,Linf_sd"]
    T = summary["tok_ref"]
    for i in range(len(T)):
        rows.append(",".join([
            str(int(T[i])),
            f"{summary['a_init_mean'][i]:.6f}", f"{summary['a_init_sd'][i]:.6f}",
            f"{summary['a_shuf_mean'][i]:.6f}", f"{summary['a_shuf_sd'][i]:.6f}",
            f"{summary['Linf_mean'][i]:.6f}",  f"{summary['Linf_sd'][i]:.6f}",
        ]))
    csv.write_text("\n".join(rows) + "\n")
    print(f"saved {csv}")


if __name__ == "__main__":
    main()
