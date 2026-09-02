"""Profile-likelihood scan over δ_P1.

Fix δ_P1 at a grid of values, refit the other 7 parameters of

    L(s, P, E) = c1·s^{δ_s1} + c3·P^{δ_P1} + (c4/E)·P^{δ_P2}·s^{δ_s2} + σ²

and track the residual σ. The minimum identifies δ_P1; a flat curve
indicates the parameter is unidentifiable (degenerate with σ²).

Other constraints during the scan:
  • δ_s1 ≤ 0 (bias-dynamics decays with s)
  • δ_P2 free
  • δ_s2 free

Output:
  experiments/figures/05_global_fit/v3/profile_dp1.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[2]
EXPT4 = REPO / "data_export" / "expt4_datasize" / "wd0_fixed_tokens"
OUT_DIR = REPO / "experiments" / "figures" / "05_global_fit" / "v3"

BATCH_SIZE = 131072
LOSS_FILTER = 5.5
S0 = 1e3
P0 = 1e7


def load_pooled():
    out = dict(s=[], P=[], E=[], L_obs=[], curve_id=[])
    cid = 0
    for f in sorted(EXPT4.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        df = float(str(d["df"]))
        P = 1e8 * df
        s = d["tokens"].astype(float) / BATCH_SIZE
        L = d["val_loss"].astype(float)
        n = len(s)
        out["s"].append(s)
        out["P"].append(np.full(n, float(P)))
        out["E"].append(np.full(n, 1.0))
        out["L_obs"].append(L)
        out["curve_id"].append(np.full(n, cid, dtype=int))
        cid += 1
    return {k: np.concatenate(v) for k, v in out.items()}


def filter_leading_prefix(ds, threshold=LOSS_FILTER):
    keep = np.ones(len(ds["L_obs"]), dtype=bool)
    for cid in np.unique(ds["curve_id"]):
        idxs = np.where(ds["curve_id"] == cid)[0]
        L = ds["L_obs"][idxs]
        below = L <= threshold
        if not below.any():
            keep[idxs] = False; continue
        first_below = int(below.argmax())
        keep[idxs[:first_below]] = False
    return {k: v[keep] for k, v in ds.items()}


def predict(p, s, P, E, dP1):
    c1, c3, c4, dS1, dP2, dS2, sig2 = p
    sn = s / S0; Pn = P / P0
    return (c1 * sn ** dS1
            + c3 * Pn ** dP1
            + (c4 / E) * Pn ** dP2 * sn ** dS2
            + sig2)


def fit_with_dp1_fixed(ds, dP1, n_restarts=20, seed=0):
    s, P, E, L = ds["s"], ds["P"], ds["E"], ds["L_obs"]

    p0_base = np.array([3.0, 0.5, 0.5, -1.0, -1.3, 0.7, 2.5])
    lo      = np.array([1e-4, 1e-6, 1e-4, -3.0, -3.0, -3.0, 0.0])
    hi      = np.array([50.0, 50.0, 50.0,  0.0,  3.0,  3.0, 5.0])

    def residuals(par):
        pred = predict(par, s, P, E, dP1)
        return np.log(np.maximum(pred, 1e-3)) - np.log(np.maximum(L, 1e-3))

    rng = np.random.default_rng(seed)
    best = None
    for r in range(n_restarts):
        p0 = p0_base.copy() if r == 0 else np.clip(
            p0_base * np.exp(rng.normal(0, 0.4, size=len(p0_base))),
            lo + 1e-6, hi - 1e-6)
        try:
            res = least_squares(residuals, p0, bounds=(lo, hi),
                                method="trf", max_nfev=20000, xtol=1e-12)
            cost = float((res.fun ** 2).sum())
            if best is None or cost < best[0]:
                best = (cost, res)
        except Exception:
            continue

    res = best[1]
    n_obs, n_p = res.jac.shape
    sigma_resid = float(np.sqrt((res.fun ** 2).sum() / max(n_obs - n_p - 1, 1)))
    pred = predict(res.x, s, P, E, dP1)
    ss_res = float(np.sum((L - pred) ** 2))
    ss_tot = float(np.sum((L - L.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    return dict(x=res.x, sigma_resid=sigma_resid, r2=r2, n_obs=n_obs)


def main():
    sns.set(font_scale=1.5); sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 24, "axes.linewidth": 4.0, "legend.fontsize": 14,
        "grid.alpha": 0.25, "xtick.labelsize": 20, "ytick.labelsize": 20,
    })

    ds = filter_leading_prefix(load_pooled())
    print(f"n_fit = {len(ds['L_obs'])}")

    dp1_grid = np.concatenate([np.array([-2.0, -1.5]), np.arange(-1.2, 0.01, 0.05)])
    sigmas, r2s, fits = [], [], {}
    for dp1 in dp1_grid:
        f = fit_with_dp1_fixed(ds, float(dp1))
        sigmas.append(f["sigma_resid"]); r2s.append(f["r2"])
        fits[float(dp1)] = f
        print(f"  δ_P1 = {dp1:+.3f}   σ_resid = {f['sigma_resid']:.4f}   "
              f"R² = {f['r2']:.4f}   "
              f"c3 = {f['x'][1]:+.4g}   σ² = {f['x'][6]:+.4g}")

    sigmas = np.array(sigmas); r2s = np.array(r2s)
    i_best = int(np.argmin(sigmas))
    dp1_best = float(dp1_grid[i_best])
    print(f"\nbest:  δ_P1 = {dp1_best:+.3f}   σ_resid = {sigmas[i_best]:.4f}")

    # 95%-ish CI from 1+ unit drop in chi2 (n_p increases by 1 → Δχ²=4 ≈ 95%)
    sigma_min = sigmas.min()
    chi2_min = sigma_min ** 2 * len(ds["L_obs"])
    chi2_thresh = chi2_min + 4.0   # rough Δχ² for 95% on 1 dof
    sigma_thresh = float(np.sqrt(chi2_thresh / len(ds["L_obs"])))
    in_ci = sigmas <= sigma_thresh

    # ---- figure ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(20, 7.5))

    axA.plot(dp1_grid, sigmas, "o-", color="C0", lw=3, ms=8,
             markeredgecolor="black", markeredgewidth=0.5)
    axA.axhline(sigma_thresh, color="0.5", lw=1.5, ls=":",
                label=fr"95% CI threshold ($\Delta\chi^2 = 4$)")
    axA.axvline(dp1_best, color="C3", lw=2, ls="--",
                label=fr"best:  $\delta_{{P_1}} = {dp1_best:.2f}$")
    axA.set_xlabel(r"fixed  $\delta_{P_1}$", fontsize=28)
    axA.set_ylabel(r"$\sigma_{\rm resid}$  (log-residual)", fontsize=24)
    axA.set_title(r"(A)  profile-likelihood scan over $\delta_{P_1}$",
                  fontsize=22, loc="left")
    axA.legend(loc="upper left", fontsize=15)

    # also plot R²
    axB.plot(dp1_grid, r2s, "o-", color="C2", lw=3, ms=8,
             markeredgecolor="black", markeredgewidth=0.5)
    axB.axvline(dp1_best, color="C3", lw=2, ls="--",
                label=fr"best:  $\delta_{{P_1}} = {dp1_best:.2f}$")
    axB.set_xlabel(r"fixed  $\delta_{P_1}$", fontsize=28)
    axB.set_ylabel(r"$R^2$", fontsize=28)
    axB.set_title(r"(B)  $R^2$ vs $\delta_{P_1}$",
                  fontsize=22, loc="left")
    axB.legend(loc="lower left", fontsize=15)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "profile_dp1.pdf"
    png = OUT_DIR / "profile_dp1.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=300)
    print(f"\nsaved {pdf}")
    print(f"saved {png}")

    # also report best fit's params at δ_P1 = δ_P1_best
    print(f"\nfit params at δ_P1 = {dp1_best}:")
    f = fits[dp1_best]
    names = ["c1", "c3", "c4", "delta_s1", "delta_P2", "delta_s2", "sigma2"]
    for nm, v in zip(names, f["x"]):
        print(f"  {nm:<10} = {v:+.4f}")


if __name__ == "__main__":
    main()
