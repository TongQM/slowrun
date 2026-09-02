"""Expt Fig 5 v3 — clean global fit on expt 4 only.

Form (N term dropped — expt 4 has fixed N at the base cell, and we don't
trust expt 2 / expt 3 for this fit because they used wd=0.1 and (expt 3)
trapezoidal LR with warmdown, both incompatible with the theoretical setup):

    L(s, P, E) = c1 · s^{δ_s1}
               + c3 · P^{δ_P1}
               + (c4 / E) · P^{δ_P2} · s^{δ_s2}
               + σ²

8 free parameters: c1, c3, c4, δ_s1, δ_P1, δ_P2, δ_s2, σ². δ_E pinned at 1.
Note: expt 4 has E ≡ 1, so c4 still identifiable but the 1/E reduction
itself isn't tested (would require an E sweep at the same wd=0 / constant
LR settings — see notes in the v2 docstring).

Data: expt4_datasize/wd0_fixed_tokens/  — pooled across strategies,
leading-prefix L_obs > 5.5 dropped.

Outputs:
  experiments/figures/05_global_fit/v3/expt_fig5_global_fit_v3.{pdf,png,csv}
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
OUT_NAME = "expt_fig5_global_fit_v3"

BATCH_SIZE = 131072
LOSS_FILTER = 5.5

S0 = 1e3                # 1000 steps
P0 = 1e7                # 10M tokens

PARAM_NAMES = ["c1", "c3", "c4",
               "delta_s1", "delta_P1", "delta_P2", "delta_s2",
               "sigma2"]


# ---------------------------- data ----------------------------

def load_expt4_pooled():
    """Load expt 4 (wd=0, constant LR), pool both strategies into one dataset."""
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
            keep[idxs] = False
            continue
        first_below = int(below.argmax())
        keep[idxs[:first_below]] = False
    return {k: v[keep] for k, v in ds.items()}, keep


# ---------------------------- model + fit ----------------------------

def predict(p, s, P, E):
    c1, c3, c4, dS1, dP1, dP2, dS2, sig2 = p
    sn = s / S0
    Pn = P / P0
    return (c1 * sn ** dS1
            + c3 * Pn ** dP1
            + (c4 / E) * Pn ** dP2 * sn ** dS2
            + sig2)


def fit_global(ds, n_restarts=24, seed=0):
    s, P, E, L_obs = ds["s"], ds["P"], ds["E"], ds["L_obs"]

    # [c1, c3, c4, δ_s1, δ_P1, δ_P2, δ_s2, σ²]
    # δ_P1 constrained ≤ 0 (physical bias-floor: more data ⇒ lower floor).
    # δ_s1 ≤ 0 (bias-dynamics decays with s).  Other δ's free.
    p0_base = np.array([3.0, 0.5, 0.5, -1.0, -1.0, -1.3,  0.7, 2.5])
    lo      = np.array([1e-4, 1e-4, 1e-4, -3.0, -3.0, -3.0, -3.0, 0.0])
    hi      = np.array([50.0, 50.0, 50.0,  0.0,  0.0,  3.0,  3.0, 5.0])

    def residuals(par):
        pred = predict(par, s, P, E)
        return np.log(np.maximum(pred, 1e-3)) - np.log(np.maximum(L_obs, 1e-3))

    rng = np.random.default_rng(seed)
    best = None
    for r in range(n_restarts):
        if r == 0:
            p0 = p0_base.copy()
        else:
            p0 = p0_base.copy()
            for i in range(len(p0)):
                if PARAM_NAMES[i].startswith("c") or PARAM_NAMES[i] == "sigma2":
                    p0[i] *= float(np.exp(rng.normal(0, 0.5)))
                else:
                    p0[i] += float(rng.normal(0, 0.4))
            p0 = np.clip(p0, lo + 1e-6, hi - 1e-6)
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

    J = res.jac
    n_obs, n_p = J.shape
    dof = max(n_obs - n_p, 1)
    sig_resid_2 = (res.fun ** 2).sum() / dof
    try:
        cov = sig_resid_2 * np.linalg.inv(J.T @ J)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(n_p, np.nan)

    return dict(x=res.x, se=se, r2=r2,
                sigma_resid=float(np.sqrt(sig_resid_2)),
                n_obs=n_obs)


# ---------------------------- driver ----------------------------

def main():
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 24, "axes.linewidth": 4.0, "legend.fontsize": 14,
        "grid.alpha": 0.25, "xtick.labelsize": 20, "ytick.labelsize": 20,
    })

    print("loading expt4 (wd=0, constant LR), both strategies pooled …")
    ds_full = load_expt4_pooled()
    ds_fit, _ = filter_leading_prefix(ds_full)
    n_full, n_fit = len(ds_full["L_obs"]), len(ds_fit["L_obs"])
    print(f"  n_full = {n_full}, n_fit (leading L > {LOSS_FILTER} dropped) = {n_fit}")

    f = fit_global(ds_fit, n_restarts=24)
    print(f"\nfit:  σ_resid = {f['sigma_resid']:.4f},  R² = {f['r2']:.4f},  "
          f"n_obs = {f['n_obs']}")
    for nm, v, sev in zip(PARAM_NAMES, f["x"], f["se"]):
        print(f"  {nm:<10} = {v:+.4f} ± {sev:.4f}")

    # ---------------- figure (2 panels: vary P, residuals vs s by P) ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(20, 7.5))

    palette_P = sns.color_palette("cool", 10)
    P_picks = sorted(np.unique(ds_full["P"]))     # all 10 P values
    pair = list(zip(P_picks, palette_P))

    # Panel A: data + model overlay vs s
    for Pp, col in pair:
        m = (ds_full["P"] == Pp) & (ds_full["E"] == 1.0)
        s_p = ds_full["s"][m]; L_p = ds_full["L_obs"][m]
        order = np.argsort(s_p)
        used = ds_full["L_obs"][m][order] <= LOSS_FILTER
        # data: open if dropped, filled if kept
        axA.plot(s_p[order][used], L_p[order][used], "o", ms=5, color=col, alpha=0.6,
                 markeredgecolor="black", markeredgewidth=0.3,
                 label=f"$P = {int(Pp/1e6)}$M")
        if (~used).any():
            axA.plot(s_p[order][~used], L_p[order][~used], "o", ms=5,
                     markerfacecolor="none", markeredgecolor=col,
                     markeredgewidth=1.0, alpha=0.5)
        s_grid = np.geomspace(s_p.min(), s_p.max(), 200)
        L_grid = predict(f["x"], s_grid, np.full_like(s_grid, Pp),
                         np.ones_like(s_grid))
        axA.plot(s_grid, L_grid, "-", lw=2.5, color=col, alpha=0.9)
    axA.set_xscale("log")
    axA.set_ylim(3.5, 8.0)
    axA.set_xlabel(r"steps  $s$", fontsize=28)
    axA.set_ylabel(r"$\mathcal{L}$", fontsize=28)
    axA.set_title(r"(A)  vary $P$  ($E{=}1$) — fit overlay",
                  fontsize=22, loc="left")
    axA.legend(loc="upper right", fontsize=12, ncol=2)

    # Panel B: residual = obs - pred, per-curve, vs s, colored by P
    pred_full = predict(f["x"], ds_full["s"], ds_full["P"], ds_full["E"])
    for Pp, col in pair:
        m = (ds_full["P"] == Pp) & (ds_full["E"] == 1.0)
        s_p = ds_full["s"][m]
        resid = ds_full["L_obs"][m] - pred_full[m]
        order = np.argsort(s_p)
        used = ds_full["L_obs"][m][order] <= LOSS_FILTER
        axB.plot(s_p[order][used], resid[order][used], "o", ms=5,
                 color=col, alpha=0.6,
                 markeredgecolor="black", markeredgewidth=0.3,
                 label=f"$P = {int(Pp/1e6)}$M")
    axB.axhline(0, color="0.2", lw=1.0, ls="-", alpha=0.7)
    axB.set_xscale("log")
    axB.set_xlabel(r"steps  $s$", fontsize=28)
    axB.set_ylabel(r"residual  $\mathcal{L}_{\rm obs} - \mathcal{L}_{\rm pred}$",
                   fontsize=22)
    axB.set_title("(B)  fit residuals vs $s$",
                  fontsize=22, loc="left")
    axB.legend(loc="upper right", fontsize=12, ncol=2)

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{OUT_NAME}.pdf"
    png = OUT_DIR / f"{OUT_NAME}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=300)
    print(f"\nsaved {pdf}")
    print(f"saved {png}")

    csv = OUT_DIR / f"{OUT_NAME}_coefs.csv"
    rows = ["param,value,se"]
    for nm, v, sev in zip(PARAM_NAMES, f["x"], f["se"]):
        rows.append(f"{nm},{v:.6f},{sev:.6f}")
    rows.append(f"r2,{f['r2']:.6f},")
    rows.append(f"sigma,{f['sigma_resid']:.6f},")
    rows.append(f"n_obs,{f['n_obs']},")
    csv.write_text("\n".join(rows) + "\n")
    print(f"saved {csv}")


if __name__ == "__main__":
    main()
