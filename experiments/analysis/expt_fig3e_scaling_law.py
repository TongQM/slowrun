"""Expt Fig 3e: general scaling-law fit across the 4×4 grid + ensembles.

Model:
  L(s, N, L_d, E) = s^{-r_t}
                  + (N^2 * L_d)^{-r_param}
                  + a * E^{-d_E} * N^{d_N} * L_d^{d_L} * (s/T)^{d_t}
                  + L_0

where:
  s    = cumulative tokens (per model)
  N    = width (n_embd)
  L_d  = depth (n_layer)
  E    = ensemble size
  T    = scale parameter for variance term

Data: 16 cells × 4 ensemble sizes × ~25 epoch snapshots ≈ 1600 (s, N, L_d, E, L) points,
loaded from data_export/expt3_grid/.

Fit: scipy.optimize.least_squares in linear loss space, with bounded exponents.
Conjecture from theory: d_E ≈ 1 (matches Expt 2's b ≈ 1) and d_t ≈ 0.5.

Plot:
  Panel A: predicted vs measured L (identity line)
  Panel B: residuals vs s (split by N or E)
  Panel C: slice plots — pick (N, L_d, E) settings, overlay prediction on data
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import least_squares
from scipy.stats import norm

DATA = "data_export/expt3_grid"
DEPTHS = [6, 12, 18, 24]
WIDTHS = [384, 768, 1152, 1536]
E_SIZES = [2, 3, 4, 5]
STRAT = "init_shuffle_ens"  # use one strategy for the fit (cleaner)


def load_grid_data():
    """Load all (s, N, L_d, E, L_obs) points from data_export.
    Use ensemble curves only (E ∈ {2,3,4,5}); add E=1 from individuals as the mean."""
    rows = []
    for d in DEPTHS:
        for w in WIDTHS:
            # Ensembles
            for E in E_SIZES:
                p = os.path.join(DATA, "ensembles", f"d{d}_w{w}_{STRAT}_E{E}.npz")
                if not os.path.isfile(p): continue
                z = np.load(p, allow_pickle=True)
                toks = z["tokens"]
                vls = z["val_loss"]
                for t, l in zip(toks, vls):
                    rows.append((int(t), w, d, int(E), float(l)))
            # E=1: mean of 5 individuals
            inds = []
            for m in range(5):
                p = os.path.join(DATA, "individuals", f"d{d}_w{w}_{STRAT}_model{m}.npz")
                if not os.path.isfile(p): continue
                z = np.load(p, allow_pickle=True)
                inds.append((z["tokens"], z["val_loss"]))
            if inds:
                # Align on common tokens (assume same eval points across inds)
                # For ensembles E=2..5 above, the replay's token grid was the same; pick that.
                base_p = os.path.join(DATA, "ensembles", f"d{d}_w{w}_{STRAT}_E5.npz")
                if not os.path.isfile(base_p): continue
                z = np.load(base_p, allow_pickle=True)
                target_toks = set(int(t) for t in z["tokens"])
                # Mean across inds at each shared token
                shared = {}
                for t, v in inds:
                    for ti, vi in zip(t, v):
                        ti = int(ti)
                        if ti in target_toks:
                            shared.setdefault(ti, []).append(float(vi))
                for ti, vs in shared.items():
                    if len(vs) >= 1:
                        rows.append((ti, w, d, 1, float(np.mean(vs))))
    return np.array(rows, dtype=[
        ("s", np.int64), ("N", np.int32), ("L", np.int32),
        ("E", np.int32), ("Lval", np.float64)])


def model(theta, s, N, L_d, E):
    """Scaling-law prediction with all terms normalized to O(1) at typical values.

    Reference scales:  N0=768, L0=12, s0=300M tokens (typical overfit-onset).
    Reduced variables:  N_r = N/N0, L_r = L/L0, s_r = s/s0.

    Model:
      L = c_t * s_r^{-r_t}
        + c_p * (N_r^2 L_r)^{-r_param}
        + c_v * E^{-d_E} * N_r^{d_N} * L_r^{d_L} * s_r^{d_t}
        + L_0

    All `c_*` are non-negative scale parameters with linear effect (so the
    Jacobian doesn't have the a-T degeneracy of the original form).
    theta = [r_t, r_param, d_E, d_N, d_L, d_t, c_t, c_p, c_v, L_0]
    """
    r_t, r_param, d_E, d_N, d_L, d_t, c_t, c_p, c_v, L_0 = theta
    s0 = 3e8
    N0 = 768.0
    L0 = 12.0
    s_r = s / s0
    N_r = N / N0
    L_r = L_d / L0
    bias = c_t * s_r ** (-r_t)
    param_term = c_p * (N_r ** 2 * L_r) ** (-r_param)
    var = c_v * E ** (-d_E) * N_r ** d_N * L_r ** d_L * s_r ** d_t
    return bias + param_term + var + L_0


def fit(data):
    s = data["s"].astype(float)
    N = data["N"].astype(float)
    Ld = data["L"].astype(float)
    E = data["E"].astype(float)
    L_obs = data["Lval"].astype(float)

    def residuals(theta):
        return model(theta, s, N, Ld, E) - L_obs

    # Initial guesses anchored at theory conjectures: d_E=1, d_t=0.5.
    # Loss is around 4-7. L_0 ≈ 4 (Bayes floor). Other terms contribute the residual ~0-3.
    # c_t (bias-dynamics scale): typical descent 6.5 → 4.5 over 1-2 orders of magnitude in s_r,
    #   so c_t ≈ 1.0 with r_t ≈ 0.3.
    # c_p (param-term scale): bigger model → smaller, so this is small at typical N, L.
    # c_v (variance-term scale): late-epoch overfit climb is ~0.2 nats from min to end, so
    #   variance term at s_r=2 (= 600M, 2× past minimum) ~ 0.2, → c_v ≈ 0.1.
    p0 = [0.30, 0.30, 1.0, 0.5, 0.5, 0.5, 0.5, 0.2, 0.1, 4.0]
    bounds_lo = [0.01, 0.01, 0.1, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0,  0.0]
    bounds_hi = [3.0,  3.0,  5.0, 5.0, 5.0, 5.0,  20.0, 20.0, 20.0, 8.0]
    res = least_squares(residuals, p0, bounds=(bounds_lo, bounds_hi), max_nfev=50000)

    # Asymptotic SE
    J = res.jac
    n_obs, n_params = J.shape
    dof = max(n_obs - n_params, 1)
    sigma2 = (res.fun ** 2).sum() / dof
    try:
        cov = sigma2 * np.linalg.inv(J.T @ J)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(n_params, np.nan)

    names = ["r_t", "r_param", "d_E", "d_N", "d_L", "d_t", "c_t", "c_p", "c_v", "L_0"]
    return {
        "theta": res.x, "se": se, "names": names,
        "n_obs": n_obs, "n_params": n_params, "dof": dof,
        "sigma": np.sqrt(sigma2),
        "ssr": (res.fun ** 2).sum(),
        "residuals": res.fun, "predictions": L_obs - res.fun,
        "data": data,
    }


def make_figure(fit_result):
    sns.set(font_scale=1.5)
    sns.set_style('whitegrid')
    plt.rcParams['axes.labelsize']  = 22
    plt.rcParams['axes.linewidth']  = 4.0
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['grid.alpha']      = 0.25
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18

    data = fit_result["data"]
    L_pred = fit_result["predictions"]
    L_obs = data["Lval"]
    resid = fit_result["residuals"]

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # Panel A: predicted vs measured
    ax = axes[0]
    palette_E = sns.color_palette('cool', 5)
    Es = np.unique(data["E"])
    for j, E in enumerate(Es):
        sel = data["E"] == E
        ax.scatter(L_obs[sel], L_pred[sel], s=18, alpha=0.6,
                   color=palette_E[j], label=f"E = {int(E)}",
                   edgecolor="black", linewidth=0.3)
    lo, hi = float(L_obs.min()), float(L_obs.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=2.0, alpha=0.6, label="identity")
    ax.set_xlabel("measured val loss")
    ax.set_ylabel("predicted val loss")
    ax.legend(loc="upper left", fontsize=14)

    # Panel B: residuals vs s, colored by E
    ax = axes[1]
    for j, E in enumerate(Es):
        sel = data["E"] == E
        ax.scatter(data["s"][sel], resid[sel], s=18, alpha=0.6,
                   color=palette_E[j], label=f"E = {int(E)}",
                   edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="black", lw=1.5, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"cumulative tokens $s$")
    ax.set_ylabel("residual (predicted − measured)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.legend(loc="upper right", fontsize=14)

    # Panel C: slice — d12, w768, vs s, all E
    ax = axes[2]
    palette_E_full = sns.color_palette('cool', len(Es))
    for j, E in enumerate(Es):
        sel = (data["E"] == E) & (data["N"] == 768) & (data["L"] == 12)
        order = np.argsort(data["s"][sel])
        s = data["s"][sel][order]
        Lv = L_obs[sel][order]
        Lp = L_pred[sel][order]
        ax.plot(s, Lv, marker="o", lw=0, markersize=8, color=palette_E_full[j],
                label=f"E={int(E)}", markeredgecolor="black", markeredgewidth=0.3)
        ax.plot(s, Lp, "--", lw=2.0, color=palette_E_full[j], alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel(r"cumulative tokens $s$")
    ax.set_ylabel("val loss (slice: d=12, w=768)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.legend(loc="upper right", fontsize=14)

    out = "experiments/figures/03_compute_matched/expt_fig3e_scaling_law.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches='tight', dpi=300)
    print(f"Saved {out}")


def main():
    if not os.path.isdir(DATA):
        print(f"Missing {DATA}. Run experiments/analysis/assemble_data.py first.")
        return
    print("Loading grid data...")
    data = load_grid_data()
    print(f"  {len(data)} (s, N, L, E, L_obs) points")
    print(f"  unique cells: {len(np.unique(data[['N','L']]))}")
    print(f"  unique E: {sorted(np.unique(data['E']).tolist())}")
    print(f"  s range: {data['s'].min()/1e6:.0f}M..{data['s'].max()/1e6:.0f}M")

    print("\nFitting...")
    fit_result = fit(data)
    print(f"  n_obs={fit_result['n_obs']}, n_params={fit_result['n_params']}, dof={fit_result['dof']}")
    print(f"  sigma={fit_result['sigma']:.4f}, SSR={fit_result['ssr']:.3f}")
    print(f"\nFitted parameters (mean ± SE):")
    for n, v, s in zip(fit_result["names"], fit_result["theta"], fit_result["se"]):
        print(f"  {n:<10} = {v:.4f} ± {s:.4f}")

    # Wald tests on conjectures
    idx_dE = fit_result["names"].index("d_E")
    idx_dt = fit_result["names"].index("d_t")
    z_dE = (fit_result["theta"][idx_dE] - 1.0) / max(fit_result["se"][idx_dE], 1e-12)
    z_dt = (fit_result["theta"][idx_dt] - 0.5) / max(fit_result["se"][idx_dt], 1e-12)
    p_dE = 2 * (1 - norm.cdf(abs(z_dE)))
    p_dt = 2 * (1 - norm.cdf(abs(z_dt)))
    print(f"\nHypothesis tests on theory conjectures:")
    print(f"  H0: d_E = 1   →  z = {z_dE:+.2f}, p = {p_dE:.3g}")
    print(f"  H0: d_t = 0.5 →  z = {z_dt:+.2f}, p = {p_dt:.3g}")

    print("\nMaking figure...")
    make_figure(fit_result)


if __name__ == "__main__":
    main()
