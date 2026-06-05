"""Expt Fig 2: ensemble scaling law L_{T,E} = a_T * E^{-b} + L_inf,T at base size d12/w768.

For each time-snapshot T (epoch boundary in the Q2 ext fused replay), fit a power-law
in ensemble size E to the data {(E, L) : E in {2,5,10,15,20}}. Fit jointly with a single
shared exponent b across all T per strategy (cleaner than per-T fits with 3 params and
only 5 data points; user's prior is b ≈ 1).

Plot:
  Left  panel: L - L_inf,T vs E on log-log axes, for ~5 representative T values
               with fitted lines. Compare init vs init+shuffle.
  Right panel: a_T (prefactor) vs T (cumulative tokens) on log-log axes for both
               strategies, with a power-law fit a_T = c T^p.

Source: q2_ensemble_size_q2_20260502_234110_d12_w768_df0.2 wandb group.
"""
import os
import numpy as np
import wandb
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt
import seaborn as sns


ENT = "xjtumyd-carnegie-mellon-university/slowrun"
GROUP = "q2_ensemble_size_q2_20260502_234110_d12_w768_df0.2"
SIZES = np.array([2, 5, 10, 15, 20], dtype=float)
STRATS = ["init_ens", "init_shuffle_ens"]
STRAT_LABEL = {"init_ens": "init", "init_shuffle_ens": "init + shuffle"}


def fetch_ensemble_grid(api):
    """Return per-strategy dict with:
       'L'    : (n_E, n_T) array of ens val loss
       'tok'  : (n_T,) cumulative tokens-seen (per model)
       'epoch': (n_T,) epoch number
    """
    out = {}
    for strat in STRATS:
        rows_L = []
        toks_ref = None
        eps_ref = None
        for E in SIZES:
            name = f"{GROUP}_{strat}_ens{int(E)}_replay"
            runs = list(api.runs(ENT, filters={"group": GROUP, "display_name": name}))
            runs.sort(key=lambda r: r.created_at, reverse=True)
            r = runs[0]
            eps, toks, vls = [], [], []
            for h in r.scan_history(keys=["ens/val_loss"]):
                vl = h.get("ens/val_loss")
                if vl is not None:
                    vls.append(vl)
            for h in r.scan_history(keys=["ens/tokens_seen"]):
                ts = h.get("ens/tokens_seen")
                if ts is not None:
                    toks.append(ts)
            for h in r.scan_history(keys=["ens/epoch"]):
                ep = h.get("ens/epoch")
                if ep is not None:
                    eps.append(ep)
            order = np.argsort(toks)
            toks = np.array(toks)[order]
            vls = np.array(vls)[order]
            if eps:
                eps = np.array(eps)[order]
            if toks_ref is None:
                toks_ref, eps_ref = toks, eps
            rows_L.append(vls)
        L = np.array(rows_L)
        out[strat] = {"L": L, "tok": toks_ref, "epoch": eps_ref}
        print(f"  {strat}: L grid shape {L.shape}, T range {toks_ref[0]/1e6:.0f}M..{toks_ref[-1]/1e6:.0f}M")
    return out


def logspace_fit_shared_Linf(L_grid_init, L_grid_shuffle, Linf_T):
    """Log-space fit holding L_inf_T fixed (taken from the linear-space fit).
    For each strategy, fits jointly across T:
        log(L_obs - L_inf_T) = log(a_T) - b * log(E)
    with shared b across all T per strategy and per-T log(a_T).
    Returns same dict shape as joint_fit_shared_Linf for symmetry, plus
    'Linf_T' / 'Linf_T_se' inherited (SE = 0 since fixed).
    """
    n_E, n_T = L_grid_init.shape
    log_E = np.log(SIZES)              # (n_E,)
    eps = 1e-8

    def fit_one_strategy(L_grid):
        # gap_TE: shape (n_E, n_T)
        gap = np.maximum(L_grid - Linf_T[None, :], eps)
        log_gap = np.log(gap)          # (n_E, n_T)
        # Model: log_gap[i,t] = log_a[t] - b * log_E[i]
        # Stack as a linear system in params (b, log_a_0, ..., log_a_{n_T-1}).
        n_obs = n_E * n_T
        n_params = 1 + n_T
        X = np.zeros((n_obs, n_params))
        y = np.zeros(n_obs)
        row = 0
        for t in range(n_T):
            for i in range(n_E):
                X[row, 0] = -log_E[i]    # coefficient on b
                X[row, 1 + t] = 1.0      # coefficient on log_a[t]
                y[row] = log_gap[i, t]
                row += 1
        # OLS via lstsq (closed form for log-linear regression)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        b = beta[0]
        log_a_T = beta[1:]
        # Residual variance + covariance for SEs
        resid = X @ beta - y
        dof = max(n_obs - n_params, 1)
        sigma2 = (resid ** 2).sum() / dof
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        b_se = se[0]
        a_T = np.exp(log_a_T)
        a_T_se = a_T * se[1:]   # delta method
        return b, b_se, a_T, a_T_se, np.sqrt(sigma2)

    b_i, b_i_se, a_i, a_i_se, sig_i = fit_one_strategy(L_grid_init)
    b_s, b_s_se, a_s, a_s_se, sig_s = fit_one_strategy(L_grid_shuffle)
    return {
        "b_init":     b_i, "b_init_se":    b_i_se,
        "b_shuffle":  b_s, "b_shuffle_se": b_s_se,
        "a_T_init":    a_i,  "a_T_init_se":    a_i_se,
        "a_T_shuffle": a_s,  "a_T_shuffle_se": a_s_se,
        "Linf_T":      Linf_T,            "Linf_T_se":   np.zeros_like(Linf_T),
        "sigma_init":  sig_i, "sigma_shuffle": sig_s,
    }


def joint_fit_shared_Linf(L_grid_init, L_grid_shuffle, b_init=1.0):
    """Joint fit across BOTH strategies with shared L_inf_T per snapshot.

    Model:
      init:    L_{T,E} = a_T^{init}    * E^{-b_init}    + L_inf_T
      shuffle: L_{T,E} = a_T^{shuffle} * E^{-b_shuffle} + L_inf_T

    Each L_grid is (n_E, n_T). Strategies share L_inf_T but have their own
    b and a_T arrays. Total params: 2 b's + 2*n_T a_T's + n_T L_inf_T's.

    Returns dict {b_init, b_shuffle, a_T_init, a_T_shuffle, Linf_T}.
    """
    assert L_grid_init.shape == L_grid_shuffle.shape
    n_E, n_T = L_grid_init.shape
    E = SIZES.reshape(-1, 1)

    def residuals(params):
        b_i = params[0]
        b_s = params[1]
        a_i = params[2:2 + n_T]
        a_s = params[2 + n_T:2 + 2 * n_T]
        Linf = params[2 + 2 * n_T:2 + 3 * n_T]
        pred_i = a_i[None, :] * (E ** (-b_i)) + Linf[None, :]
        pred_s = a_s[None, :] * (E ** (-b_s)) + Linf[None, :]
        return np.concatenate([(pred_i - L_grid_init).ravel(),
                               (pred_s - L_grid_shuffle).ravel()])

    a0 = np.full(n_T, 1.0)
    Linf0 = np.minimum(L_grid_init.min(axis=0), L_grid_shuffle.min(axis=0)) - 0.05
    p0 = np.concatenate([[b_init, b_init], a0, a0, Linf0])
    bounds_lo = np.concatenate([[0.1, 0.1], np.zeros(2 * n_T), -np.inf * np.ones(n_T)])
    bounds_hi = np.concatenate([[5.0, 5.0], np.inf * np.ones(2 * n_T), np.inf * np.ones(n_T)])
    res = least_squares(residuals, p0, bounds=(bounds_lo, bounds_hi), max_nfev=20000)

    # Asymptotic SEs from Jacobian (Gauss-Newton):
    #   cov(theta_hat) ≈ sigma^2 * (J^T J)^{-1},  sigma^2 = SSR / (n_obs - n_params)
    J = res.jac
    n_obs = J.shape[0]
    n_params = J.shape[1]
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
        "sigma":       np.sqrt(sigma2),
    }


def logspace_fit_shared_Linf(L_grid_init, L_grid_shuffle, Linf_T_fixed):
    """Log-space fit with L_inf_T held fixed (typically at the linear-fit value).

    Per strategy, model is linear in log space:
        log(L_{T,E} - L_inf,T) = log(a_T) - b * log(E)
    so we solve a single OLS regression per strategy with (n_T + 1) free
    params: log(a_T) for each T, plus a single shared b. Returns same
    keys as joint_fit_shared_Linf so the rest of main() can use it.
    """
    n_E, n_T = L_grid_init.shape
    log_E = np.log(SIZES)
    out = {"Linf_T": Linf_T_fixed.copy(),
           "Linf_T_se": np.zeros_like(Linf_T_fixed)}

    for tag, L_grid in [("init", L_grid_init), ("shuffle", L_grid_shuffle)]:
        # Build design matrix: rows are (T, E) observations, cols are
        # [log(a_T1), log(a_T2), ..., log(a_TN), b].
        rows, ys = [], []
        for t_idx in range(n_T):
            gap = L_grid[:, t_idx] - Linf_T_fixed[t_idx]
            mask = gap > 1e-6
            for k in np.where(mask)[0]:
                row = np.zeros(n_T + 1)
                row[t_idx] = 1.0       # coefficient on log(a_T)
                row[n_T] = -log_E[k]   # coefficient on b
                rows.append(row)
                ys.append(np.log(gap[k]))
        X = np.array(rows); y = np.array(ys)
        XtX = X.T @ X; Xty = X.T @ y
        try:
            theta = np.linalg.solve(XtX, Xty)
            resid = y - X @ theta
            sigma2 = (resid ** 2).sum() / max(len(y) - len(theta), 1)
            cov = sigma2 * np.linalg.inv(XtX)
            se = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            theta = np.full(n_T + 1, np.nan)
            se = np.full_like(theta, np.nan)
        log_a = theta[:n_T]
        a_T = np.exp(log_a)
        a_T_se = a_T * se[:n_T]   # delta method: SE(a) = a * SE(log a)
        b = theta[n_T]; b_se = se[n_T]
        out[f"b_{tag}"] = b
        out[f"b_{tag}_se"] = b_se
        out[f"a_T_{tag}"] = a_T
        out[f"a_T_{tag}_se"] = a_T_se
    return out


def power_law_fit(x, y, y_se=None):
    """Fit y = c * x^p via log-linear regression in log space.
    Optionally weight by 1 / (y_se / y)^2 (delta method on log y).
    Returns (c, p, p_se)."""
    mask = (y > 0) & np.isfinite(x) & np.isfinite(y)
    if y_se is not None:
        mask = mask & (y_se > 0) & np.isfinite(y_se)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    if y_se is not None:
        # Var(log y) ≈ (sigma_y / y)^2  (delta method)
        w = (y[mask] / y_se[mask]) ** 2
        # Weighted least squares on (lx, ly)
        Lx = np.column_stack([lx, np.ones_like(lx)])
        W = np.diag(w)
        XtWX = Lx.T @ W @ Lx
        XtWy = Lx.T @ W @ ly
        try:
            beta = np.linalg.solve(XtWX, XtWy)
            cov = np.linalg.inv(XtWX) * (np.sum(w * (ly - Lx @ beta) ** 2) / max(len(lx) - 2, 1))
            p, log_c = beta
            p_se = np.sqrt(cov[0, 0])
        except np.linalg.LinAlgError:
            p, log_c = np.polyfit(lx, ly, 1)
            p_se = np.nan
    else:
        p, log_c = np.polyfit(lx, ly, 1)
        p_se = np.nan
    return np.exp(log_c), p, p_se


def main():
    sns.set(font_scale=1.5)
    sns.set_style('whitegrid')

    plt.rcParams['axes.labelsize']  = 24
    plt.rcParams['axes.linewidth']  = 4.0
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['grid.alpha']      = 0.25
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    api = wandb.Api()
    print("Fetching Q2 ext fused-replay data...")
    data = fetch_ensemble_grid(api)

    # ---- Joint fit across BOTH strategies with shared L_inf_T ----
    L_init = data["init_ens"]["L"]
    L_shuf = data["init_shuffle_ens"]["L"]
    T_ref  = data["init_ens"]["tok"]
    res = joint_fit_shared_Linf(L_init, L_shuf)

    # ---- Alternative log-space fit (L_inf fixed at linear-fit value) ----
    res_log = logspace_fit_shared_Linf(L_init, L_shuf, res["Linf_T"])
    print(f"\n=== Linear-space (raw residuals) vs log-space (L_inf fixed) fit ===")
    print(f"             linear-space         log-space (L_inf fixed)")
    print(f"  b_init     {res['b_init']:.3f} ± {res['b_init_se']:.3f}        "
          f"{res_log['b_init']:.3f} ± {res_log['b_init_se']:.3f}")
    print(f"  b_shuffle  {res['b_shuffle']:.3f} ± {res['b_shuffle_se']:.3f}        "
          f"{res_log['b_shuffle']:.3f} ± {res_log['b_shuffle_se']:.3f}")
    for ep_show in [5, 15, 25, 40]:
        idx = ep_show - 1
        print(f"  a_T@ep{ep_show:>2} (init)    "
              f"{res['a_T_init'][idx]:.3f}±{res['a_T_init_se'][idx]:.3f}        "
              f"{res_log['a_T_init'][idx]:.3f}±{res_log['a_T_init_se'][idx]:.3f}")
        print(f"  a_T@ep{ep_show:>2} (shuf)    "
              f"{res['a_T_shuffle'][idx]:.3f}±{res['a_T_shuffle_se'][idx]:.3f}        "
              f"{res_log['a_T_shuffle'][idx]:.3f}±{res_log['a_T_shuffle_se'][idx]:.3f}")

    fits = {
        "init_ens": {
            "b": res["b_init"], "b_se": res["b_init_se"],
            "a_T": res["a_T_init"], "a_T_se": res["a_T_init_se"],
            "Linf_T": res["Linf_T"], "Linf_T_se": res["Linf_T_se"],
            "T": T_ref,
        },
        "init_shuffle_ens": {
            "b": res["b_shuffle"], "b_se": res["b_shuffle_se"],
            "a_T": res["a_T_shuffle"], "a_T_se": res["a_T_shuffle_se"],
            "Linf_T": res["Linf_T"], "Linf_T_se": res["Linf_T_se"],
            "T": T_ref,
        },
    }
    for strat in STRATS:
        T = fits[strat]["T"].astype(float)
        c, p, p_se = power_law_fit(T, fits[strat]["a_T"], fits[strat]["a_T_se"])
        fits[strat]["c"] = c
        fits[strat]["p"] = p
        fits[strat]["p_se"] = p_se

    print(f"\nShared-L_inf joint fit (asymptotic SE from Jacobian):")
    print(f"  b (init)         = {res['b_init']:.3f}  ± {res['b_init_se']:.3f}")
    print(f"  b (init+shuffle) = {res['b_shuffle']:.3f}  ± {res['b_shuffle_se']:.3f}")
    # Hypothesis test: H0: b = 1 (one-sample z-test, asymptotic)
    for tag, val, se in [("init", res['b_init'], res['b_init_se']),
                         ("init+shuffle", res['b_shuffle'], res['b_shuffle_se'])]:
        z = (val - 1.0) / max(se, 1e-12)
        from scipy.stats import norm
        pval_2sided = 2 * (1 - norm.cdf(abs(z)))
        print(f"  H0: b_{tag} = 1   →   z = {z:+.2f},  p = {pval_2sided:.3g}")
    # H0: b_init = b_shuffle (z-test on difference)
    diff = res['b_shuffle'] - res['b_init']
    diff_se = np.sqrt(res['b_init_se']**2 + res['b_shuffle_se']**2)
    z_diff = diff / max(diff_se, 1e-12)
    from scipy.stats import norm
    pval_diff = 2 * (1 - norm.cdf(abs(z_diff)))
    print(f"  H0: b_init = b_shuffle   →   Δb = {diff:+.3f} ± {diff_se:.3f},  z = {z_diff:+.2f},  p = {pval_diff:.3g}")

    for strat in STRATS:
        print(f"\n[{strat}]:")
        a_T = fits[strat]["a_T"]
        a_T_se = fits[strat]["a_T_se"]
        c, p, p_se = fits[strat]["c"], fits[strat]["p"], fits[strat]["p_se"]
        print(f"  a_T  = c * T^p   →   c = {c:.3e}, p = {p:.3f} ± {p_se:.3f}")
        print(f"  a_T (epoch 5/15/25/40)  = "
              f"{a_T[4]:.3f}±{a_T_se[4]:.3f} / {a_T[14]:.3f}±{a_T_se[14]:.3f} / "
              f"{a_T[24]:.3f}±{a_T_se[24]:.3f} / {a_T[39]:.3f}±{a_T_se[39]:.3f}")
    print(f"\n  Linf_T (shared, epoch 5/15/25/40) = "
          f"{res['Linf_T'][4]:.3f}±{res['Linf_T_se'][4]:.3f} / "
          f"{res['Linf_T'][14]:.3f}±{res['Linf_T_se'][14]:.3f} / "
          f"{res['Linf_T'][24]:.3f}±{res['Linf_T_se'][24]:.3f} / "
          f"{res['Linf_T'][39]:.3f}±{res['Linf_T_se'][39]:.3f}")

    # ---- Figure: 3 panels — (init, init+shuffle, prefactor comparison) ----
    # Left/middle share y so b slopes are visually comparable; right is its own.
    fig = plt.figure(figsize=(24, 7))
    # Need extra wspace between middle and right because the middle panel
    # shows y-tick labels and the right panel's y-label sits on its left edge.
    gs = fig.add_gridspec(1, 3, wspace=0.32)
    ax_L = fig.add_subplot(gs[0, 0])
    ax_M = fig.add_subplot(gs[0, 1], sharey=ax_L)
    ax_R = fig.add_subplot(gs[0, 2])
    axes = [ax_L, ax_M, ax_R]
    # Keep y-tick labels visible on both left and middle panels even though
    # they share the axis (per-user request — easier to read absolute values).

    snap_epochs = [5, 10, 15, 25, 40]
    palette_T = sns.color_palette('cool', len(snap_epochs))
    palette_S = sns.color_palette('cool', 2)
    strat_marker = {"init_ens": "o", "init_shuffle_ens": "s"}
    strat_ls = {"init_ens": "-", "init_shuffle_ens": "--"}

    # Panels 0/1: per-strategy L - L_inf vs E (log-log)
    for ax_idx, strat in enumerate(STRATS):
        ax = axes[ax_idx]
        L_grid = data[strat]["L"]
        b = fits[strat]["b"]
        a_T = fits[strat]["a_T"]
        Linf_T = fits[strat]["Linf_T"]
        T = data[strat]["tok"]
        for j, snap_ep in enumerate(snap_epochs):
            idx = int(snap_ep) - 1
            if idx >= L_grid.shape[1]: continue
            gap = L_grid[:, idx] - Linf_T[idx]
            label = f"T = {snap_ep} ep ({T[idx]/1e6:.0f}M)"
            ax.loglog(SIZES, gap, marker=strat_marker[strat], lw=0,
                      color=palette_T[j], markersize=11, label=label,
                      markeredgecolor="black", markeredgewidth=0.6)
            E_grid = np.geomspace(SIZES.min() * 0.9, SIZES.max() * 1.1, 100)
            ax.loglog(E_grid, a_T[idx] * E_grid ** (-b),
                      color=palette_T[j], lw=2.0, ls=strat_ls[strat], alpha=0.7)
        ax.set_xlabel(r"ensemble size $E$")
        if ax_idx == 0: ax.set_ylabel(r"$L_{T,E} - L_{\infty,T}$")
        ax.set_xticks([2, 5, 10, 20])
        ax.set_xticklabels(["2", "5", "10", "20"])
        b_se = fits[strat]["b_se"]
        ax.text(0.02, 0.05,
                f"{STRAT_LABEL[strat]}\n$b = {b:.2f} \\pm {b_se:.2f}$",
                transform=ax.transAxes, fontsize=20, fontweight="bold",
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="lightgray"))
        ax.legend(loc="upper right", fontsize=14)

    # Panel 2: a_T vs T on log-log axes for both strategies, with error bars
    ax = axes[2]
    for j, strat in enumerate(STRATS):
        T = fits[strat]["T"].astype(float)
        a_T = fits[strat]["a_T"]
        a_T_se = fits[strat]["a_T_se"]
        c, p, p_se = fits[strat]["c"], fits[strat]["p"], fits[strat]["p_se"]
        ax.errorbar(T, a_T, yerr=a_T_se, fmt=strat_marker[strat], markersize=10,
                    color=palette_S[j], markeredgecolor="black", markeredgewidth=0.5,
                    elinewidth=1.2, capsize=3, capthick=1.2,
                    label=f"{STRAT_LABEL[strat]}: $a_T \\propto T^{{{p:.2f} \\pm {p_se:.2f}}}$")
        T_grid = np.geomspace(T.min(), T.max(), 100)
        ax.loglog(T_grid, c * T_grid ** p, lw=2.5, color=palette_S[j], alpha=0.7,
                  ls=strat_ls[strat])
    ax.set_xscale('log'); ax.set_yscale('log')
    # Pin x-axis range to actual T span (errorbar autoscale picks weird bounds)
    T_all = np.concatenate([fits[s]["T"].astype(float) for s in STRATS])
    ax.set_xlim(T_all.min() * 0.7, T_all.max() * 1.4)
    # Use M (millions) tick formatter so x-axis reads as physical token count
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.set_xlabel(r"cumulative tokens $T$")
    ax.set_ylabel(r"loss-gap prefactor $a_T$")
    ax.legend(loc="upper left", fontsize=15)

    out = "experiments/figures/02_ensemble_scaling/expt_fig2.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches='tight', dpi=300)
    print(f"\nSaved {out}")
    print(f"Saved {out.replace('.pdf', '.png')} (preview)")

    # Also save fit parameters as a small CSV-ish table for the paper text
    txt = "T_tokens,init_a_T,init_Linf_T,shuffle_a_T,shuffle_Linf_T\n"
    T_init = fits["init_ens"]["T"]
    for i in range(len(T_init)):
        txt += f"{int(T_init[i])},{fits['init_ens']['a_T'][i]:.6f},{fits['init_ens']['Linf_T'][i]:.6f},"
        txt += f"{fits['init_shuffle_ens']['a_T'][i]:.6f},{fits['init_shuffle_ens']['Linf_T'][i]:.6f}\n"
    csv_out = out.replace(".pdf", "_fits.csv")
    with open(csv_out, "w") as f:
        f.write(txt)
    print(f"Saved fit table {csv_out}")


if __name__ == "__main__":
    main()
