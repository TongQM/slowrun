"""Expt Fig 5 — global loss fit.

Form (per strategy):

    L(s, P, E) = α s^{-(a-1)/b}
                + β P^{-(a-1)}
                + (γ/E) P^{-(a-1+b/2)} s^{δ_s}
                + σ²

with 7 free parameters (α, β, γ, a, b, δ_s, σ²). Pinned: δ_E = 1.
δ_s is freed (the original advisor form pinned it at 1/2 — see Expt 5b
discussion: empirical α_s ∝ s^{0.78}, so the s^{1/2} pin caused β to
collapse to zero in the previous fit).

Data: combined sources, per strategy
  • expt4_datasize/wd0_fixed_tokens/   E = 1, ten P ∈ {10..100}M, wd = 0
  • expt2_ensemble/individuals/        E = 1, P = 20M, wd = 0.1
  • expt2_ensemble/ensembles/          E ∈ {2,5,10,15,20}, P = 20M, wd = 0.1

The two expt2 contributions pin γ via 1/E reduction; expt4 pins (α, β, a,
b, σ²) via the (s, P) plane.

Filter: leading high-loss prefix dropped per curve (L_obs > 5.5).
Uncertainty: refit per bootstrap iter (10 iters from expt2_ensemble/bootstrap/),
report mean ± SD across iters.

Outputs:
  experiments/figures/05_global_fit/expt_fig5_global_fit.{pdf,png}
  experiments/figures/05_global_fit/expt_fig5_global_fit_coefs.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[2]
EXPT2 = REPO / "data_export" / "expt2_ensemble"
EXPT4 = REPO / "data_export" / "expt4_datasize" / "wd0_fixed_tokens"
OUT_DIR = REPO / "experiments" / "figures" / "05_global_fit"
OUT_NAME = "expt_fig5_global_fit"

STRATS = ["init_ens", "init_shuffle_ens"]
STRAT_PRETTY = {"init_ens": "init", "init_shuffle_ens": "init + shuffle"}
BATCH_SIZE = 131072

S0 = 1e3                                  # step normalization (median run length)
P0 = 1e7                                  # data-size normalization (10M tokens)
LOSS_FILTER = 5.5                         # leading-prefix L_obs > 5.5 dropped


# ---------------------------- data assembly ----------------------------

def _add_curve(out, s, P, E, L_obs, source):
    cid = out["_cid"]
    out["s"].append(s)
    out["P"].append(np.full_like(s, P, dtype=float))
    out["E"].append(np.full_like(s, E, dtype=float))
    out["L_obs"].append(L_obs)
    out["curve_id"].append(np.full_like(s, cid, dtype=int))
    out["source"].append([source] * len(s))
    out["_cid"] += 1


def load_strategy(strat: str, boot_iter: int | None = None):
    """Return long-format arrays for one strategy, optionally for one bootstrap iter."""
    out = dict(s=[], P=[], E=[], L_obs=[], curve_id=[], source=[], _cid=0)

    # --- expt4: E=1, varying P ---
    for f in sorted(EXPT4.glob(f"*_{strat}.npz")):
        d = np.load(f, allow_pickle=True)
        df = float(str(d["df"]))
        P = 1e8 * df
        s = d["tokens"].astype(float) / BATCH_SIZE
        _add_curve(out, s, P, 1.0, d["val_loss"].astype(float), "expt4")

    # --- expt2: P=20M, varying E ---
    P2 = 20_000_000.0   # df=0.2 × 100M
    if boot_iter is None:
        # E=1: per-snapshot mean over the 20 individuals
        ind_files = sorted((EXPT2 / "individuals").glob(f"{strat}_model*.npz"),
                           key=lambda p: int(p.stem.split("model")[-1]))
        Ls = []
        for p in ind_files:
            d = np.load(p, allow_pickle=True)
            tok_ref = d["tokens"]
            Ls.append(d["val_loss"].astype(float))
        Ls = np.stack(Ls, axis=0)
        s2 = tok_ref.astype(float) / BATCH_SIZE
        _add_curve(out, s2, P2, 1.0, Ls.mean(axis=0), "expt2")
        # E ∈ {2, 5, 10, 15, 20} from ensembles
        for E in (2, 5, 10, 15, 20):
            d = np.load(EXPT2 / "ensembles" / f"{strat}_E{E}.npz", allow_pickle=True)
            s2 = d["tokens"].astype(float) / BATCH_SIZE
            _add_curve(out, s2, P2, float(E), d["val_loss"].astype(float), "expt2")
    else:
        # use bootstrap iter k for expt2
        b = np.load(EXPT2 / "bootstrap" / f"{strat}_iter{boot_iter:03d}.npz",
                    allow_pickle=True)
        s2 = b["tokens"].astype(float) / BATCH_SIZE
        sizes = b["sizes"].astype(int)
        boot_idx = b["boot_indices"].astype(int)
        # E=1: mean over the resampled 20 individuals
        ind_files = sorted((EXPT2 / "individuals").glob(f"{strat}_model*.npz"),
                           key=lambda p: int(p.stem.split("model")[-1]))
        Ls_all = np.stack([np.load(p, allow_pickle=True)["val_loss"].astype(float)
                           for p in ind_files], axis=0)
        e1 = Ls_all[boot_idx].mean(axis=0)
        _add_curve(out, s2, P2, 1.0, e1, "expt2")
        # E ∈ {2, 5, 10, 15, 20} from this iter
        for i, E in enumerate(sizes):
            if int(E) == 1: continue
            _add_curve(out, s2, P2, float(E), b["L"][i].astype(float), "expt2")

    out.pop("_cid")
    return {k: (np.concatenate(v) if k != "source"
                else np.array([x for sub in v for x in sub]))
            for k, v in out.items()}


def filter_leading_prefix(ds, threshold=LOSS_FILTER):
    keep = np.ones(len(ds["L_obs"]), dtype=bool)
    for cid in np.unique(ds["curve_id"]):
        idxs = np.where(ds["curve_id"] == cid)[0]
        L = ds["L_obs"][idxs]
        below = L <= threshold
        if not below.any():
            keep[idxs] = False
            continue
        first_below = int(below.argmax())
        keep[idxs[:first_below]] = False
    return {k: v[keep] for k, v in ds.items()}, keep


# ---------------------------- model + fit ----------------------------

PARAM_NAMES = ["alpha", "beta", "gamma", "a", "b", "delta_s", "sigma2"]


def predict(p, s, P, E):
    alpha, beta, gamma, a, b, delta_s, sigma2 = p
    sn = s / S0
    Pn = P / P0
    rt   = (a - 1) / b                          # bias-dynamics exponent
    rP   =  a - 1                                # bias-floor P-exponent
    dvP  =  a - 1 + b / 2                        # variance-term P-exponent
    return (alpha * sn ** (-rt)
            + beta  * Pn ** (-rP)
            + (gamma / E) * Pn ** (-dvP) * sn ** delta_s
            + sigma2)


def fit_global(ds, n_restarts=12, seed=0):
    s, P, E, L_obs = ds["s"], ds["P"], ds["E"], ds["L_obs"]

    # [α,    β,    γ,    a,    b,    δ_s,  σ²]
    p0_base = np.array([3.0,  0.5,  0.1,  1.6,  1.0,  0.7,  2.5])
    lo      = np.array([1e-3, 1e-5, 1e-6, 1.01, 0.05, 0.05, 0.0])
    hi      = np.array([50.0, 50.0, 50.0, 5.0,  5.0,  3.0,  5.0])

    def residuals(p):
        pred = predict(p, s, P, E)
        return np.log(np.maximum(pred, 1e-3)) - np.log(np.maximum(L_obs, 1e-3))

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
    pred = predict(res.x, s, P, E)
    ss_res = float(np.sum((L_obs - pred) ** 2))
    ss_tot = float(np.sum((L_obs - L_obs.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot

    # Jacobian-based asymptotic SE
    J = res.jac
    n_obs, n_p = J.shape
    dof = max(n_obs - n_p, 1)
    sigma_resid_2 = (res.fun ** 2).sum() / dof
    try:
        cov = sigma_resid_2 * np.linalg.inv(J.T @ J)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(n_p, np.nan)

    return dict(x=res.x, se=se, r2=r2, sigma_resid=float(np.sqrt(sigma_resid_2)),
                n_obs=n_obs)


# ---------------------------- driver ----------------------------

def main():
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 24, "axes.linewidth": 4.0, "legend.fontsize": 14,
        "grid.alpha": 0.25, "xtick.labelsize": 20, "ytick.labelsize": 20,
    })

    fits, fits_boot = {}, {}
    for strat in STRATS:
        ds_full = load_strategy(strat, boot_iter=None)
        ds_fit, _ = filter_leading_prefix(ds_full)
        n_full, n_fit = len(ds_full["L_obs"]), len(ds_fit["L_obs"])
        print(f"\n[{STRAT_PRETTY[strat]}]  n_full = {n_full}, "
              f"n_fit (L≤{LOSS_FILTER}) = {n_fit}")

        f = fit_global(ds_fit)
        fits[strat] = (f, ds_full, ds_fit)
        print(f"  σ_resid = {f['sigma_resid']:.4f}, R² = {f['r2']:.4f}")
        for nm, v, sev in zip(PARAM_NAMES, f["x"], f["se"]):
            print(f"    {nm:<8} = {v:+.4f} ± {sev:.4f}")

        # bootstrap (10 iters from expt2_ensemble/bootstrap/)
        boot_xs = []
        for k in range(10):
            ds_b = load_strategy(strat, boot_iter=k)
            ds_bf, _ = filter_leading_prefix(ds_b)
            fb = fit_global(ds_bf, n_restarts=4, seed=k)
            boot_xs.append(fb["x"])
        boot_xs = np.stack(boot_xs, axis=0)
        boot_mean = boot_xs.mean(axis=0)
        boot_sd = boot_xs.std(axis=0, ddof=1)
        fits_boot[strat] = dict(mean=boot_mean, sd=boot_sd, samples=boot_xs)
        print("  bootstrap (n=10, mean ± SD across iters):")
        for nm, m, sd in zip(PARAM_NAMES, boot_mean, boot_sd):
            print(f"    {nm:<8} = {m:+.4f} ± {sd:.4f}")

    # --- figure: 3 panels ---
    fig = plt.figure(figsize=(22, 7))
    gs = fig.add_gridspec(1, 3, wspace=0.32)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    palette_P = sns.color_palette("cool", 10)
    palette_E = sns.color_palette("cool", 6)
    strat_marker = {"init_ens": "o", "init_shuffle_ens": "s"}

    # Panel A: predicted vs observed (init+shuffle, all sources)
    s_strat = "init_shuffle_ens"
    f, ds_full, ds_fit = fits[s_strat]
    pred_fit = predict(f["x"], ds_fit["s"], ds_fit["P"], ds_fit["E"])
    is_e2 = ds_fit["source"] == "expt2"
    axA.scatter(ds_fit["L_obs"][~is_e2], pred_fit[~is_e2],
                s=18, color="C0", alpha=0.5, label="expt4  (E=1, varies P)")
    axA.scatter(ds_fit["L_obs"][is_e2], pred_fit[is_e2],
                s=18, color="C3", alpha=0.6, label="expt2  (varies E, P=20M)")
    lo_v = float(min(ds_fit["L_obs"].min(), pred_fit.min()))
    hi_v = float(max(ds_fit["L_obs"].max(), pred_fit.max()))
    pad = 0.05 * (hi_v - lo_v)
    axA.plot([lo_v, hi_v], [lo_v, hi_v], "k-", lw=1.0, alpha=0.6)
    axA.set_xlim(lo_v - pad, hi_v + pad)
    axA.set_ylim(lo_v - pad, hi_v + pad)
    axA.set_xlabel(r"observed  $\mathcal{L}$", fontsize=24)
    axA.set_ylabel(r"predicted  $\mathcal{L}$", fontsize=24)
    axA.set_title(r"(A)  joint fit  ($\delta_E = 1$, $\delta_s = 1/2$ pinned;"
                  r" init + shuffle)", fontsize=14)
    axA.text(0.04, 0.96,
             f"$R^2 = {f['r2']:.3f}$\n$\\sigma = {f['sigma_resid']:.3f}$\n"
             f"$n = {f['n_obs']}$",
             transform=axA.transAxes, fontsize=14, va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       alpha=0.92, edgecolor="lightgray"))
    axA.legend(loc="lower right", fontsize=12)

    # Panel B: example expt4 P-curves with model overlay (init+shuffle)
    P_picks = [10e6, 20e6, 50e6, 100e6]
    for j, Pp in enumerate(P_picks):
        m = (ds_full["P"] == Pp) & (ds_full["E"] == 1.0) & (ds_full["source"] == "expt4")
        if m.sum() == 0:
            continue
        used = ds_full["L_obs"][m] <= LOSS_FILTER
        s_p = ds_full["s"][m]
        L_p = ds_full["L_obs"][m]
        order = np.argsort(s_p)
        idx_pal = int(round(((Pp / 1e7 - 1) / 9) * (len(palette_P) - 1)))
        col = palette_P[idx_pal]
        axB.plot(s_p[order], L_p[order], "o", ms=5, color=col, alpha=0.6,
                 markeredgecolor="black", markeredgewidth=0.4,
                 label=f"$P = {int(Pp/1e6)}$M  (data)")
        # only plot fit overlay over the data range, not extrapolating
        s_grid = np.geomspace(s_p[used].min() if used.any() else s_p.min(),
                              s_p.max(), 200)
        L_grid = predict(f["x"], s_grid, np.full_like(s_grid, Pp),
                         np.ones_like(s_grid))
        axB.plot(s_grid, L_grid, "-", lw=2.0, color=col, alpha=0.85)
    axB.set_xscale("log")
    axB.set_xlabel(r"steps  $s$", fontsize=24)
    axB.set_ylabel(r"$\mathcal{L}$", fontsize=24)
    axB.set_title("(B)  expt4 U-curves (E=1) — fit overlay", fontsize=14)
    axB.set_ylim(3.5, 8.0)
    axB.legend(loc="upper right", fontsize=11)

    # Panel C: expt2 E-curves with model overlay at P=20M
    P2 = 20e6
    for j, E in enumerate([1, 2, 5, 10, 15, 20]):
        m = (ds_full["P"] == P2) & (ds_full["E"] == E) & (ds_full["source"] == "expt2")
        if m.sum() == 0:
            continue
        s_p = ds_full["s"][m]
        L_p = ds_full["L_obs"][m]
        order = np.argsort(s_p)
        col = palette_E[j]
        axC.plot(s_p[order], L_p[order], "o", ms=5, color=col, alpha=0.7,
                 markeredgecolor="black", markeredgewidth=0.4,
                 label=f"$E = {E}$  (data)")
        used = ds_full["L_obs"][m] <= LOSS_FILTER
        s_grid = np.geomspace(s_p[used].min() if used.any() else s_p.min(),
                              s_p.max(), 200)
        L_grid = predict(f["x"], s_grid, np.full_like(s_grid, P2),
                         np.full_like(s_grid, E))
        axC.plot(s_grid, L_grid, "-", lw=2.0, color=col, alpha=0.85)
    axC.set_xscale("log")
    axC.set_xlabel(r"steps  $s$", fontsize=24)
    axC.set_ylabel(r"$\mathcal{L}$", fontsize=24)
    axC.set_title(r"(C)  expt2 ensemble curves at $P = 20$M — fit overlay",
                  fontsize=14)
    axC.set_ylim(3.8, 6.7)
    axC.legend(loc="upper right", fontsize=11, ncol=2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / f"{OUT_NAME}.pdf"
    out_png = OUT_DIR / f"{OUT_NAME}.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    print(f"\nsaved {out_pdf}")
    print(f"saved {out_png}")

    # CSV
    rows = ["strategy,fit,param,value,se"]
    for s in STRATS:
        f_, _, _ = fits[s]
        for nm, v, sev in zip(PARAM_NAMES, f_["x"], f_["se"]):
            rows.append(f"{s},point,{nm},{v:.6f},{sev:.6f}")
        for nm, m, sd in zip(PARAM_NAMES, fits_boot[s]["mean"], fits_boot[s]["sd"]):
            rows.append(f"{s},bootstrap,{nm},{m:.6f},{sd:.6f}")
        rows.append(f"{s},point,r2,{f_['r2']:.6f},")
        rows.append(f"{s},point,sigma,{f_['sigma_resid']:.6f},")
        rows.append(f"{s},point,n_obs,{f_['n_obs']},")
    csv = OUT_DIR / f"{OUT_NAME}_coefs.csv"
    csv.write_text("\n".join(rows) + "\n")
    print(f"saved {csv}")


if __name__ == "__main__":
    main()
