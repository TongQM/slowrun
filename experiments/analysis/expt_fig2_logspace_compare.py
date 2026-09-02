"""Expt Fig 2 — log-space vs linear-space regression comparison.

Original `expt_fig2_ensemble_scaling.py` fits

    L_{T,E} = a_T * E^{-b} + L_inf,T

by minimizing the **linear-space** residuals (L_obs - pred)^2, with shared L_inf,T
across strategies and per-strategy (b, a_T). That fit gives b_init ≈ 0.75,
b_shuffle ≈ 1.15.

This script asks: does b survive a regression on the log of the loss-gap, where
the residuals being minimized are

    log(L_obs - L_inf,T) - [log(a_T) - b * log(E)]

?  In log space, the small-gap (large-E) observations carry equal weight as the
large-gap ones, so the slope b is much more sensitive to high-E behavior. The
linear-space fit weights the absolute residual, which over-emphasizes the
small-E (large-gap) anchor.

Three fits run side-by-side:
  1. **linear**  — reproduce the original joint fit (b, a_T, L_inf shared across strats).
  2. **log_fix** — log-space fit with L_inf held fixed at the linear-space estimate.
  3. **log_full**— log-space fit with all params (b, a_T, L_inf) fitted jointly in log space.

Outputs (no overwrite of original):
  experiments/figures/02_ensemble_scaling/logspace/expt_fig2_logspace.{pdf,png}
  experiments/figures/02_ensemble_scaling/logspace/fits_compare.csv

Reads local NPZ files from data_export/expt2_ensemble/ensembles/ — no wandb.
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
DATA_DIR = REPO / "data_export" / "expt2_ensemble" / "ensembles"
OUT_DIR = REPO / "experiments" / "figures" / "02_ensemble_scaling" / "logspace"

SIZES = np.array([2, 5, 10, 15, 20], dtype=float)
STRATS = ["init_ens", "init_shuffle_ens"]
STRAT_LABEL = {"init_ens": "init", "init_shuffle_ens": "init + shuffle"}


def load_ensemble_grid() -> dict:
    """Return {strat: {"L": (n_E, n_T), "tok": (n_T,), "epoch": (n_T,)}}."""
    out = {}
    for strat in STRATS:
        L_rows, tok_ref, ep_ref = [], None, None
        for E in SIZES:
            f = DATA_DIR / f"{strat}_E{int(E)}.npz"
            d = np.load(f, allow_pickle=True)
            tok, vl, ep = d["tokens"], d["val_loss"], d["epoch"]
            order = np.argsort(tok)
            tok, vl, ep = tok[order], vl[order], ep[order]
            if tok_ref is None:
                tok_ref, ep_ref = tok, ep
            else:
                assert np.array_equal(tok, tok_ref), f"token grids disagree for {strat} E={E}"
            L_rows.append(vl)
        out[strat] = {"L": np.array(L_rows), "tok": tok_ref, "epoch": ep_ref}
        print(f"  loaded {strat}: L shape {out[strat]['L'].shape}, "
              f"T range {tok_ref[0]/1e6:.0f}M..{tok_ref[-1]/1e6:.0f}M")
    return out


def joint_fit_linear(L_init: np.ndarray, L_shuf: np.ndarray) -> dict:
    """Original: minimize sum (L_obs - pred)^2 with shared L_inf_T."""
    n_E, n_T = L_init.shape
    E = SIZES.reshape(-1, 1)

    def residuals(p):
        b_i, b_s = p[0], p[1]
        a_i = p[2:2 + n_T]
        a_s = p[2 + n_T:2 + 2 * n_T]
        Linf = p[2 + 2 * n_T:2 + 3 * n_T]
        pred_i = a_i[None, :] * (E ** (-b_i)) + Linf[None, :]
        pred_s = a_s[None, :] * (E ** (-b_s)) + Linf[None, :]
        return np.concatenate([(pred_i - L_init).ravel(), (pred_s - L_shuf).ravel()])

    a0 = np.full(n_T, 1.0)
    Linf0 = np.minimum(L_init.min(0), L_shuf.min(0)) - 0.05
    p0 = np.concatenate([[1.0, 1.0], a0, a0, Linf0])
    lo = np.concatenate([[0.1, 0.1], np.zeros(2 * n_T), -np.inf * np.ones(n_T)])
    hi = np.concatenate([[5.0, 5.0], np.inf * np.ones(2 * n_T), np.inf * np.ones(n_T)])
    res = least_squares(residuals, p0, bounds=(lo, hi), max_nfev=20000)
    J = res.jac
    dof = max(J.shape[0] - J.shape[1], 1)
    sigma2 = (res.fun ** 2).sum() / dof
    try:
        cov = sigma2 * np.linalg.inv(J.T @ J)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(J.shape[1], np.nan)

    return _pack(res.x, se, n_T, sigma2)


def joint_fit_logspace_full(L_init: np.ndarray, L_shuf: np.ndarray,
                            Linf_anchor: np.ndarray | None = None,
                            Linf_window: float = 0.05) -> dict:
    """Joint log-space fit, all params free except L_inf is constrained to a
    physically sensible window around the linear-space estimate.

    Why the window? With unbounded L_inf, the log-space residual is degenerate:
    pushing L_inf → min_E L_obs(E,T) makes the smallest-gap residual blow up
    in log-space, and the optimizer responds by killing the E-dependence
    (b → lower bound). The linear-space fit gives a well-identified L_inf
    (residuals are O(1) near the asymptote), so we anchor around it.

    Residual:  log(L_obs - L_inf_T) - [log(a_T) - b * log(E)]
    """
    n_E, n_T = L_init.shape
    log_E = np.log(SIZES).reshape(-1, 1)

    if Linf_anchor is None:
        Linf_anchor = joint_fit_linear(L_init, L_shuf)["Linf_T"]

    cap = np.minimum(L_init.min(0), L_shuf.min(0))
    Linf_lo = Linf_anchor - Linf_window
    Linf_hi = np.minimum(Linf_anchor + Linf_window, cap - 1e-4)

    def residuals(p):
        b_i, b_s = p[0], p[1]
        log_a_i = p[2:2 + n_T]
        log_a_s = p[2 + n_T:2 + 2 * n_T]
        Linf = p[2 + 2 * n_T:2 + 3 * n_T]
        gap_i = np.maximum(L_init - Linf[None, :], 1e-10)
        gap_s = np.maximum(L_shuf - Linf[None, :], 1e-10)
        pred_log_i = log_a_i[None, :] - b_i * log_E
        pred_log_s = log_a_s[None, :] - b_s * log_E
        return np.concatenate([(np.log(gap_i) - pred_log_i).ravel(),
                               (np.log(gap_s) - pred_log_s).ravel()])

    lin = joint_fit_linear(L_init, L_shuf)
    log_a_i0 = np.log(np.maximum(lin["a_T_init"], 1e-6))
    log_a_s0 = np.log(np.maximum(lin["a_T_shuffle"], 1e-6))
    Linf0 = np.clip(lin["Linf_T"], Linf_lo + 1e-5, Linf_hi - 1e-5)
    p0 = np.concatenate([[lin["b_init"], lin["b_shuffle"]], log_a_i0, log_a_s0, Linf0])

    lo = np.concatenate([[0.1, 0.1], -20 * np.ones(2 * n_T), Linf_lo])
    hi = np.concatenate([[5.0, 5.0], 20 * np.ones(2 * n_T), Linf_hi])
    p0 = np.clip(p0, lo + 1e-6, hi - 1e-6)
    res = least_squares(residuals, p0, bounds=(lo, hi), max_nfev=40000)
    J = res.jac
    dof = max(J.shape[0] - J.shape[1], 1)
    sigma2 = (res.fun ** 2).sum() / dof
    try:
        cov = sigma2 * np.linalg.inv(J.T @ J)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(J.shape[1], np.nan)

    # Convert log_a to a, with delta-method SE.
    out = _pack_logspace(res.x, se, n_T, sigma2)
    return out


def fit_logspace_fixed_Linf(L_init: np.ndarray, L_shuf: np.ndarray, Linf_T: np.ndarray) -> dict:
    """Per-strategy log-linear fit holding L_inf_T fixed (closed form OLS)."""
    n_E, n_T = L_init.shape
    log_E = np.log(SIZES)

    def fit_one(L_grid):
        gap = np.maximum(L_grid - Linf_T[None, :], 1e-10)
        log_gap = np.log(gap)
        n_obs, n_par = n_E * n_T, 1 + n_T
        X = np.zeros((n_obs, n_par)); y = np.zeros(n_obs)
        r = 0
        for t in range(n_T):
            for i in range(n_E):
                X[r, 0] = -log_E[i]
                X[r, 1 + t] = 1.0
                y[r] = log_gap[i, t]
                r += 1
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = X @ beta - y
        dof = max(n_obs - n_par, 1)
        sigma2 = (resid ** 2).sum() / dof
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        a_T = np.exp(beta[1:])
        a_T_se = a_T * se[1:]
        return beta[0], se[0], a_T, a_T_se, np.sqrt(sigma2)

    b_i, bi_se, a_i, ai_se, sig_i = fit_one(L_init)
    b_s, bs_se, a_s, as_se, sig_s = fit_one(L_shuf)
    return {
        "b_init": b_i, "b_init_se": bi_se,
        "b_shuffle": b_s, "b_shuffle_se": bs_se,
        "a_T_init": a_i, "a_T_init_se": ai_se,
        "a_T_shuffle": a_s, "a_T_shuffle_se": as_se,
        "Linf_T": Linf_T, "Linf_T_se": np.zeros_like(Linf_T),
        "sigma_init": sig_i, "sigma_shuffle": sig_s,
        "sigma": np.sqrt((sig_i ** 2 + sig_s ** 2) / 2),
    }


def _pack(x, se, n_T, sigma2):
    return {
        "b_init": x[0], "b_init_se": se[0],
        "b_shuffle": x[1], "b_shuffle_se": se[1],
        "a_T_init": x[2:2 + n_T], "a_T_init_se": se[2:2 + n_T],
        "a_T_shuffle": x[2 + n_T:2 + 2 * n_T], "a_T_shuffle_se": se[2 + n_T:2 + 2 * n_T],
        "Linf_T": x[2 + 2 * n_T:2 + 3 * n_T], "Linf_T_se": se[2 + 2 * n_T:2 + 3 * n_T],
        "sigma": np.sqrt(sigma2),
    }


def _pack_logspace(x, se, n_T, sigma2):
    log_a_i = x[2:2 + n_T]
    log_a_s = x[2 + n_T:2 + 2 * n_T]
    a_i = np.exp(log_a_i)
    a_s = np.exp(log_a_s)
    return {
        "b_init": x[0], "b_init_se": se[0],
        "b_shuffle": x[1], "b_shuffle_se": se[1],
        "a_T_init": a_i, "a_T_init_se": a_i * se[2:2 + n_T],
        "a_T_shuffle": a_s, "a_T_shuffle_se": a_s * se[2 + n_T:2 + 2 * n_T],
        "Linf_T": x[2 + 2 * n_T:2 + 3 * n_T],
        "Linf_T_se": se[2 + 2 * n_T:2 + 3 * n_T],
        "sigma": np.sqrt(sigma2),
    }


def power_law_fit(x, y, y_se=None):
    """y = c x^p in log space. Optional weights via delta method."""
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


def _wald(b, b_se, target=1.0):
    z = (b - target) / max(b_se, 1e-12)
    pval = 2 * (1 - norm.cdf(abs(z)))
    return z, pval


def print_summary(name: str, fit: dict):
    print(f"\n=== {name} ===")
    print(f"  b_init     = {fit['b_init']:.3f} ± {fit['b_init_se']:.3f}")
    print(f"  b_shuffle  = {fit['b_shuffle']:.3f} ± {fit['b_shuffle_se']:.3f}")
    z_i, p_i = _wald(fit['b_init'], fit['b_init_se'], 1.0)
    z_s, p_s = _wald(fit['b_shuffle'], fit['b_shuffle_se'], 1.0)
    print(f"  H0: b_init=1     z={z_i:+.2f}  p={p_i:.2e}")
    print(f"  H0: b_shuffle=1  z={z_s:+.2f}  p={p_s:.2e}")
    diff = fit['b_shuffle'] - fit['b_init']
    diff_se = np.sqrt(fit['b_init_se'] ** 2 + fit['b_shuffle_se'] ** 2)
    z_d = diff / max(diff_se, 1e-12)
    p_d = 2 * (1 - norm.cdf(abs(z_d)))
    print(f"  Δb = {diff:+.3f} ± {diff_se:.3f}  z={z_d:+.2f}  p={p_d:.2e}")
    print(f"  residual sigma = {fit['sigma']:.3g}")


def make_figure(data, fits: dict, out_path: Path):
    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 22, "axes.linewidth": 3.0, "legend.fontsize": 14,
        "grid.alpha": 0.25, "xtick.labelsize": 16, "ytick.labelsize": 16,
    })

    snap_epochs = [5, 10, 15, 25, 40]
    palette_T = sns.color_palette("cool", len(snap_epochs))
    strat_marker = {"init_ens": "o", "init_shuffle_ens": "s"}
    fit_styles = {
        "linear":   {"ls": "-",  "label": "linear-space fit"},
        "log_fix":  {"ls": "--", "label": "log-space fit (L∞ fixed)"},
        "log_full": {"ls": ":",  "label": "log-space fit (L∞ free)"},
    }
    fit_colors = {"linear": "tab:blue", "log_fix": "tab:orange", "log_full": "tab:red"}

    fig = plt.figure(figsize=(20, 6.5))
    gs = fig.add_gridspec(1, 3, wspace=0.32)
    axL = fig.add_subplot(gs[0, 0])
    axM = fig.add_subplot(gs[0, 1], sharey=axL)
    axR = fig.add_subplot(gs[0, 2])

    for ax_idx, strat in enumerate(STRATS):
        ax = axL if ax_idx == 0 else axM
        L_grid = data[strat]["L"]
        T = data[strat]["tok"]
        Linf_lin = fits["linear"]["Linf_T"]
        for j, snap_ep in enumerate(snap_epochs):
            idx = int(snap_ep) - 1
            if idx >= L_grid.shape[1]: continue
            gap_obs = np.maximum(L_grid[:, idx] - Linf_lin[idx], 1e-6)
            ax.loglog(SIZES, gap_obs, marker=strat_marker[strat], lw=0,
                      color=palette_T[j], markersize=10,
                      markeredgecolor="black", markeredgewidth=0.5,
                      label=f"T = {snap_ep} ep ({T[idx]/1e6:.0f}M)")
            E_grid = np.geomspace(SIZES.min() * 0.9, SIZES.max() * 1.1, 100)
            for fname, style in fit_styles.items():
                f = fits[fname]
                b = f["b_init"] if strat == "init_ens" else f["b_shuffle"]
                a_T = f["a_T_init"] if strat == "init_ens" else f["a_T_shuffle"]
                # Plot the *predicted* gap-from-linear-Linf so curves overlay obs.
                pred_gap = a_T[idx] * E_grid ** (-b) + (f["Linf_T"][idx] - Linf_lin[idx])
                pred_gap = np.maximum(pred_gap, 1e-8)
                ax.loglog(E_grid, pred_gap, color=fit_colors[fname], lw=1.6,
                          ls=style["ls"], alpha=0.85)
        ax.set_xlabel(r"ensemble size $E$")
        if ax_idx == 0:
            ax.set_ylabel(r"$L_{T,E} - \widehat{L}_{\infty,T}^{\text{(linear)}}$")
        ax.set_xticks([2, 5, 10, 20]); ax.set_xticklabels(["2", "5", "10", "20"])
        b_lin  = fits["linear"]["b_init" if strat == "init_ens" else "b_shuffle"]
        b_lin_se = fits["linear"]["b_init_se" if strat == "init_ens" else "b_shuffle_se"]
        b_lf   = fits["log_fix"]["b_init" if strat == "init_ens" else "b_shuffle"]
        b_lf_se= fits["log_fix"]["b_init_se" if strat == "init_ens" else "b_shuffle_se"]
        b_lfu  = fits["log_full"]["b_init" if strat == "init_ens" else "b_shuffle"]
        b_lfu_se=fits["log_full"]["b_init_se" if strat == "init_ens" else "b_shuffle_se"]
        ax.text(0.02, 0.05,
                f"{STRAT_LABEL[strat]}\n"
                f"linear:  b = {b_lin:.2f} ± {b_lin_se:.2f}\n"
                f"log-fix: b = {b_lf:.2f} ± {b_lf_se:.2f}\n"
                f"log-full:b = {b_lfu:.2f} ± {b_lfu_se:.2f}",
                transform=ax.transAxes, fontsize=12, va="bottom", ha="left",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.9, edgecolor="lightgray"))
        if ax_idx == 0:
            # Shared legend for T-snapshots + fit linestyles
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], color=fit_colors[k], lw=2, ls=v["ls"], label=v["label"])
                       for k, v in fit_styles.items()]
            ax.legend(handles=handles, loc="upper right", fontsize=11)
        else:
            ax.legend(loc="upper right", fontsize=10, ncol=1)

    # Right panel: a_T vs T for each fit
    axR.set_xscale("log"); axR.set_yscale("log")
    for fname, style in fit_styles.items():
        f = fits[fname]
        for s_idx, strat in enumerate(STRATS):
            T = data[strat]["tok"].astype(float)
            a_T = f["a_T_init"] if strat == "init_ens" else f["a_T_shuffle"]
            a_T_se = f["a_T_init_se"] if strat == "init_ens" else f["a_T_shuffle_se"]
            c, p, p_se = power_law_fit(T, a_T, a_T_se)
            mfc = fit_colors[fname] if strat == "init_ens" else "white"
            axR.errorbar(T, a_T, yerr=a_T_se, fmt=strat_marker[strat],
                         markersize=7, color=fit_colors[fname], mfc=mfc,
                         markeredgecolor=fit_colors[fname], elinewidth=0.8,
                         capsize=2, alpha=0.7,
                         label=f"{style['label']} / {STRAT_LABEL[strat]} : "
                               f"$a_T \\propto T^{{{p:.2f}\\pm{p_se:.2f}}}$")
            T_grid = np.geomspace(T.min(), T.max(), 80)
            axR.plot(T_grid, c * T_grid ** p, lw=1.4, color=fit_colors[fname],
                     ls=style["ls"], alpha=0.55)
    axR.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    axR.set_xlabel(r"cumulative tokens $T$")
    axR.set_ylabel(r"prefactor $a_T$")
    axR.legend(loc="lower right", fontsize=8.5, framealpha=0.85)

    fig.suptitle("Expt 2: linear vs log-space regression for $L_{T,E} = a_T E^{-b} + L_{\\infty,T}$",
                 fontsize=18, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    plt.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"\nSaved {out_path.with_suffix('.pdf')}")
    print(f"Saved {out_path.with_suffix('.png')}")


def write_fits_csv(fits: dict, T: np.ndarray, out_csv: Path):
    rows = ["fit,strategy,b,b_se,p_aT,p_aT_se,sigma"]
    for fname in ("linear", "log_fix", "log_full"):
        f = fits[fname]
        for strat in STRATS:
            b = f["b_init" if strat == "init_ens" else "b_shuffle"]
            bse = f["b_init_se" if strat == "init_ens" else "b_shuffle_se"]
            a_T = f["a_T_init" if strat == "init_ens" else "a_T_shuffle"]
            a_T_se = f["a_T_init_se" if strat == "init_ens" else "a_T_shuffle_se"]
            _, p, p_se = power_law_fit(T.astype(float), a_T, a_T_se)
            rows.append(f"{fname},{strat},{b:.5f},{bse:.5f},{p:.5f},{p_se:.5f},{f.get('sigma', np.nan):.5g}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(rows) + "\n")
    print(f"Saved fit table {out_csv}")


def main():
    print("Loading expt2 ensemble grid from data_export ...")
    data = load_ensemble_grid()
    L_init = data["init_ens"]["L"]
    L_shuf = data["init_shuffle_ens"]["L"]
    T_ref = data["init_ens"]["tok"]

    fit_lin = joint_fit_linear(L_init, L_shuf)
    print_summary("LINEAR-SPACE (original)", fit_lin)

    fit_logfix = fit_logspace_fixed_Linf(L_init, L_shuf, fit_lin["Linf_T"])
    print_summary("LOG-SPACE (L_inf fixed at linear estimate)", fit_logfix)

    fit_logfull = joint_fit_logspace_full(L_init, L_shuf)
    print_summary("LOG-SPACE (all params jointly fit, L_inf bounded ±0.05 of linear)", fit_logfull)
    # Diagnostic: how often did L_inf hit the bound? (Indicates corner solution.)
    cap = np.minimum(L_init.min(0), L_shuf.min(0))
    Linf_lo = fit_lin["Linf_T"] - 0.05
    Linf_hi = np.minimum(fit_lin["Linf_T"] + 0.05, cap - 1e-4)
    at_lo = (fit_logfull["Linf_T"] - Linf_lo) < 1e-3
    at_hi = (Linf_hi - fit_logfull["Linf_T"]) < 1e-3
    print(f"  diagnostic: Linf at lower bound at {at_lo.sum()}/{len(at_lo)} snapshots, "
          f"upper bound at {at_hi.sum()}/{len(at_hi)} snapshots")
    print(f"  → if many at upper bound, the fit is bound-driven; trust log_fix more.")

    fits = {"linear": fit_lin, "log_fix": fit_logfix, "log_full": fit_logfull}

    out_fig = OUT_DIR / "expt_fig2_logspace"
    make_figure(data, fits, out_fig)

    write_fits_csv(fits, T_ref, OUT_DIR / "fits_compare.csv")


if __name__ == "__main__":
    main()
