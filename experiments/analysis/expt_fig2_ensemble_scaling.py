"""Expt Fig 2 — ensemble scaling law L_{T,E} = a_T · E^{-b} + L_{inf,T}.

Treats E=1 as a special case of ensembling (per-snapshot mean across the 20
individuals) and includes it in the joint fit alongside the post-hoc replays
at E ∈ {2, 5, 10, 15, 20}. So the fit uses six ensemble sizes per snapshot
and 40 epoch snapshots per strategy.

Reads from data_export/expt2_ensemble/ (no wandb dependency).

Outputs (canonical):
  experiments/figures/02_ensemble_scaling/expt_fig2.{pdf,png}
  experiments/figures/02_ensemble_scaling/expt_fig2_fits.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import least_squares
from scipy.stats import norm


REPO = Path(__file__).resolve().parents[2]
ENS_DIR = REPO / "data_export" / "expt2_ensemble" / "ensembles"
IND_DIR = REPO / "data_export" / "expt2_ensemble" / "individuals"
OUT_DIR = REPO / "experiments" / "figures" / "02_ensemble_scaling"

STRATS = ["init_ens", "init_shuffle_ens"]
STRAT_LABEL = {"init_ens": "init", "init_shuffle_ens": "init + shuffle"}
ENS_E_SIZES = np.array([2, 5, 10, 15, 20], dtype=float)
ALL_E_SIZES = np.array([1, 2, 5, 10, 15, 20], dtype=float)


# ---------------------------- data ----------------------------

def load_grid_with_e1():
    """Return {strat: {"L": (6, n_T), "tok": (n_T,), "epoch": (n_T,)}}.

    E=1 is the per-snapshot mean across the 20 individuals.
    E ∈ {2, 5, 10, 15, 20} are the post-hoc fused-replay ensembles.
    """
    out = {}
    for strat in STRATS:
        ind_files = sorted(IND_DIR.glob(f"{strat}_model*.npz"),
                           key=lambda p: int(p.stem.split("model")[-1]))
        ind_curves, tok_ref, ep_ref = [], None, None
        for p in ind_files:
            d = np.load(p, allow_pickle=True)
            tok, vl, ep = d["tokens"], d["val_loss"], d["epoch"]
            order = np.argsort(tok)
            tok, vl, ep = tok[order], vl[order], ep[order]
            if tok_ref is None:
                tok_ref, ep_ref = tok, ep
            else:
                assert np.array_equal(tok, tok_ref), f"individual {p} token grid differs"
            ind_curves.append(vl)
        if not ind_curves:
            raise RuntimeError(f"no individuals for {strat}")
        ind_stack = np.stack(ind_curves, axis=0)        # (20, n_T)
        e1_mean = ind_stack.mean(axis=0)                 # (n_T,)

        rows = [e1_mean]
        for E in ENS_E_SIZES:
            d = np.load(ENS_DIR / f"{strat}_E{int(E)}.npz", allow_pickle=True)
            tok, vl = d["tokens"], d["val_loss"]
            order = np.argsort(tok); tok, vl = tok[order], vl[order]
            assert np.array_equal(tok, tok_ref), f"ens E={E} grid mismatch"
            rows.append(vl)
        out[strat] = {
            "L":     np.stack(rows, axis=0),   # (6, n_T)
            "tok":   tok_ref,
            "epoch": ep_ref,
            "n_indiv": len(ind_curves),
        }
        print(f"  {strat}: E=1 from {len(ind_curves)} indivs; "
              f"L grid {out[strat]['L'].shape}; "
              f"T range {tok_ref[0]/1e6:.0f}M..{tok_ref[-1]/1e6:.0f}M")
    return out


# ---------------------------- fits ----------------------------

def joint_fit_shared_Linf(L_init, L_shuf, E_sizes=ALL_E_SIZES):
    """Joint linear-space NLLS across both strategies with shared L_inf,T.

    Model:
      init:    L_{T,E} = a_T^{init}    · E^{-b_init}    + L_inf,T
      shuffle: L_{T,E} = a_T^{shuffle} · E^{-b_shuffle} + L_inf,T

    Each L grid is (n_E, n_T). Strategies share L_inf,T per snapshot but have
    their own b and per-T prefactors a_T.
    """
    assert L_init.shape == L_shuf.shape
    n_E, n_T = L_init.shape
    E = E_sizes.reshape(-1, 1)

    def residuals(p):
        b_i, b_s = p[0], p[1]
        a_i = p[2:2 + n_T]
        a_s = p[2 + n_T:2 + 2 * n_T]
        Linf = p[2 + 2 * n_T:2 + 3 * n_T]
        pred_i = a_i[None, :] * (E ** (-b_i)) + Linf[None, :]
        pred_s = a_s[None, :] * (E ** (-b_s)) + Linf[None, :]
        return np.concatenate([(pred_i - L_init).ravel(),
                               (pred_s - L_shuf).ravel()])

    a0 = np.full(n_T, 1.0)
    Linf0 = np.minimum(L_init.min(0), L_shuf.min(0)) - 0.05
    p0 = np.concatenate([[1.0, 1.0], a0, a0, Linf0])
    lo = np.concatenate([[0.05, 0.05], np.zeros(2 * n_T), -np.inf * np.ones(n_T)])
    hi = np.concatenate([[5.0, 5.0], np.inf * np.ones(2 * n_T), np.inf * np.ones(n_T)])
    res = least_squares(residuals, p0, bounds=(lo, hi), max_nfev=20000)

    J = res.jac
    n_obs, n_params = J.shape
    dof = max(n_obs - n_params, 1)
    sigma2 = (res.fun ** 2).sum() / dof
    try:
        cov = sigma2 * np.linalg.inv(J.T @ J)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(n_params, np.nan)

    return {
        "b_init":      res.x[0], "b_init_se":     se[0],
        "b_shuffle":   res.x[1], "b_shuffle_se":  se[1],
        "a_T_init":    res.x[2:2 + n_T],
        "a_T_init_se": se[2:2 + n_T],
        "a_T_shuffle":    res.x[2 + n_T:2 + 2 * n_T],
        "a_T_shuffle_se": se[2 + n_T:2 + 2 * n_T],
        "Linf_T":      res.x[2 + 2 * n_T:2 + 3 * n_T],
        "Linf_T_se":   se[2 + 2 * n_T:2 + 3 * n_T],
        "sigma":       float(np.sqrt(sigma2)),
        "n_obs":       n_obs,
        "n_params":    n_params,
    }


def power_law_fit(x, y, y_se=None):
    """y = c · x^p via log-linear regression. Optional weights via delta method."""
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
            cov = np.linalg.inv(XtWX) * (np.sum(w * (ly - Lx @ beta) ** 2) / max(len(lx) - 2, 1))
            p, log_c = beta; p_se = np.sqrt(cov[0, 0])
        except np.linalg.LinAlgError:
            p, log_c = np.polyfit(lx, ly, 1); p_se = np.nan
    else:
        p, log_c = np.polyfit(lx, ly, 1); p_se = np.nan
    return float(np.exp(log_c)), float(p), float(p_se)


# ---------------------------- reporting ----------------------------

def print_summary(fit):
    print("\n=== Joint fit (linear-space, E ∈ {1, 2, 5, 10, 15, 20}) ===")
    print(f"  n_obs = {fit['n_obs']}, n_params = {fit['n_params']}, σ = {fit['sigma']:.4g}")
    print(f"  b_init     = {fit['b_init']:.3f} ± {fit['b_init_se']:.3f}")
    print(f"  b_shuffle  = {fit['b_shuffle']:.3f} ± {fit['b_shuffle_se']:.3f}")
    for tag, val, se in [("init", fit['b_init'], fit['b_init_se']),
                         ("shuffle", fit['b_shuffle'], fit['b_shuffle_se'])]:
        z = (val - 1.0) / max(se, 1e-12)
        p = 2 * (1 - norm.cdf(abs(z)))
        print(f"  H0: b_{tag}=1     z={z:+.2f}  p={p:.2e}")
    diff = fit['b_shuffle'] - fit['b_init']
    diff_se = np.sqrt(fit['b_init_se'] ** 2 + fit['b_shuffle_se'] ** 2)
    z_d = diff / diff_se
    p_d = 2 * (1 - norm.cdf(abs(z_d)))
    print(f"  Δb = {diff:+.3f} ± {diff_se:.3f}  z={z_d:+.2f}  p={p_d:.2e}")


# ---------------------------- figure ----------------------------

def make_figure(data, fit, snap_epochs=(5, 10, 15, 25, 40)):
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
    axes = [ax_L, ax_M, ax_R]

    palette_T = sns.color_palette("cool", len(snap_epochs))
    palette_S = sns.color_palette("cool", 2)
    strat_marker = {"init_ens": "o", "init_shuffle_ens": "s"}
    strat_ls = {"init_ens": "-", "init_shuffle_ens": "--"}

    fits = {
        "init_ens": {
            "b": fit["b_init"], "b_se": fit["b_init_se"],
            "a_T": fit["a_T_init"], "a_T_se": fit["a_T_init_se"],
        },
        "init_shuffle_ens": {
            "b": fit["b_shuffle"], "b_se": fit["b_shuffle_se"],
            "a_T": fit["a_T_shuffle"], "a_T_se": fit["a_T_shuffle_se"],
        },
    }

    # Panels 0/1 — log-log gap vs E per strategy
    for ax_idx, strat in enumerate(STRATS):
        ax = axes[ax_idx]
        L_grid = data[strat]["L"]
        T = data[strat]["tok"]
        b = fits[strat]["b"]; a_T = fits[strat]["a_T"]
        Linf_T = fit["Linf_T"]
        for j, snap_ep in enumerate(snap_epochs):
            idx = int(snap_ep) - 1
            if idx >= L_grid.shape[1]: continue
            gap = np.maximum(L_grid[:, idx] - Linf_T[idx], 1e-6)
            ax.loglog(ALL_E_SIZES, gap, marker=strat_marker[strat], lw=0,
                      color=palette_T[j], markersize=11,
                      markeredgecolor="black", markeredgewidth=0.6,
                      label=f"T = {snap_ep} ep ({T[idx]/1e6:.0f}M)")
            E_grid = np.geomspace(0.85, ALL_E_SIZES.max() * 1.1, 100)
            ax.loglog(E_grid, a_T[idx] * E_grid ** (-b),
                      color=palette_T[j], lw=2.0, ls=strat_ls[strat], alpha=0.7)
        ax.set_xlabel(r"ensemble size $E$")
        if ax_idx == 0:
            ax.set_ylabel(r"$L_{T,E} - L_{\infty,T}$")
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.set_xticklabels(["1", "2", "5", "10", "20"])
        ax.text(0.02, 0.05,
                f"{STRAT_LABEL[strat]}\n$b = {b:.2f} \\pm {fits[strat]['b_se']:.2f}$",
                transform=ax.transAxes, fontsize=20, fontweight="bold",
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="lightgray"))
        ax.legend(loc="upper right", fontsize=14)

    # Panel 2 — a_T vs T (log-log) for both strategies, plus c · T^p fits
    ax = axes[2]
    pT_summary = {}
    for j, strat in enumerate(STRATS):
        T = data[strat]["tok"].astype(float)
        a_T = fits[strat]["a_T"]
        a_T_se = fits[strat]["a_T_se"]
        c, p, p_se = power_law_fit(T, a_T, a_T_se)
        pT_summary[strat] = {"c": c, "p": p, "p_se": p_se}
        ax.errorbar(T, a_T, yerr=a_T_se, fmt=strat_marker[strat], markersize=10,
                    color=palette_S[j], markeredgecolor="black", markeredgewidth=0.5,
                    elinewidth=1.2, capsize=3, capthick=1.2,
                    label=f"{STRAT_LABEL[strat]}: $a_T \\propto T^{{{p:.2f}\\pm{p_se:.2f}}}$")
        T_grid = np.geomspace(T.min(), T.max(), 100)
        ax.loglog(T_grid, c * T_grid ** p, lw=2.5, color=palette_S[j],
                  ls=strat_ls[strat], alpha=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    T_all = np.concatenate([data[s]["tok"].astype(float) for s in STRATS])
    ax.set_xlim(T_all.min() * 0.7, T_all.max() * 1.4)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.set_xlabel(r"cumulative tokens $T$")
    ax.set_ylabel(r"loss-gap prefactor $a_T$")
    ax.legend(loc="upper left", fontsize=15)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "expt_fig2.pdf"
    png = OUT_DIR / "expt_fig2.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=300)
    print(f"\nSaved {pdf}")
    print(f"Saved {png}")

    # Per-snapshot fit table
    csv = OUT_DIR / "expt_fig2_fits.csv"
    rows = ["T_tokens,init_a_T,init_a_T_se,init_Linf_T,shuffle_a_T,shuffle_a_T_se,shuffle_Linf_T"]
    T_all = data["init_ens"]["tok"]
    for i in range(len(T_all)):
        rows.append(",".join([
            str(int(T_all[i])),
            f"{fit['a_T_init'][i]:.6f}", f"{fit['a_T_init_se'][i]:.6f}",
            f"{fit['Linf_T'][i]:.6f}",
            f"{fit['a_T_shuffle'][i]:.6f}", f"{fit['a_T_shuffle_se'][i]:.6f}",
            f"{fit['Linf_T'][i]:.6f}",
        ]))
    csv.write_text("\n".join(rows) + "\n")
    print(f"Saved {csv}")
    return pT_summary


def main():
    print("Loading expt2 data with E=1 (mean of 20 individuals)...")
    data = load_grid_with_e1()

    L_init = data["init_ens"]["L"]
    L_shuf = data["init_shuffle_ens"]["L"]
    fit = joint_fit_shared_Linf(L_init, L_shuf, ALL_E_SIZES)
    print_summary(fit)
    pT = make_figure(data, fit)
    print("\n=== Power-law a_T = c · T^p ===")
    for s in STRATS:
        x = pT[s]
        print(f"  {STRAT_LABEL[s]:<14}  c = {x['c']:.3e}, "
              f"p = {x['p']:.3f} ± {x['p_se']:.3f}")


if __name__ == "__main__":
    main()
