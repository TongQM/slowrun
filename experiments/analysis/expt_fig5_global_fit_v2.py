"""Expt Fig 5 v2 — global loss fit with free exponents.

Form (advisor's relaxed version):

    L(s, P, E, N) = c1 · s^{δ_s1}
                  + c2 · N^{δ_N}
                  + c3 · P^{δ_P1}
                  + (c4 / E) · P^{δ_P2} · s^{δ_s2}
                  + σ²

10 free parameters: c1..c4, δ_s1, δ_N, δ_P1, δ_P2, δ_s2, σ².  δ_E pinned at 1.

Data sources (init and init+shuffle pooled — no per-strategy params):
  • expt2_ensemble        E ∈ {1..20}, P = 20M, N = base cell
  • expt3_grid            E ∈ {1..5},  P = 20M, varies N (4×4 d×w grid)
  • expt4_datasize/wd0    E = 1,        varies P,    N = base cell

Filter: leading high-loss prefix per curve (L_obs > 5.5).

Outputs:
  experiments/figures/05_global_fit/v2/expt_fig5_global_fit_v2.{pdf,png,csv}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.optimize import least_squares

from expt_fig3_loader import (
    DEPTHS, WIDTHS, STRATEGIES, E_SIZES, load_grid, REPO,
)


EXPT2 = REPO / "data_export" / "expt2_ensemble"
EXPT4 = REPO / "data_export" / "expt4_datasize" / "wd0_fixed_tokens"
OUT_DIR = REPO / "experiments" / "figures" / "05_global_fit" / "v2"
OUT_NAME = "expt_fig5_global_fit_v2"

BATCH_SIZE = 131072
LOSS_FILTER = 5.5

S0 = 1e3                                      # 1000 steps
N0 = 16 * 12 * 768 ** 2                       # base-cell non-embed params
P0 = 1e7                                      # 10M tokens

PARAM_NAMES = ["c1", "c2", "c3", "c4",
               "delta_s1", "delta_N",
               "delta_P1", "delta_P2", "delta_s2",
               "sigma2"]


# ---------------------------- data ----------------------------

def _add_curve(out, s, P, E, N, L_obs, source):
    cid = out["_cid"]
    n = len(s)
    out["s"].append(s.astype(float))
    out["P"].append(np.full(n, float(P)))
    out["E"].append(np.full(n, float(E)))
    out["N"].append(np.full(n, float(N)))
    out["L_obs"].append(L_obs.astype(float))
    out["curve_id"].append(np.full(n, cid, dtype=int))
    out["source"].append([source] * n)
    out["_cid"] += 1


def load_all_pooled():
    """Combined dataset, init + init+shuffle pooled into one set."""
    out = dict(s=[], P=[], E=[], N=[], L_obs=[], curve_id=[], source=[], _cid=0)

    # ------ expt 4: E=1 individuals at varying P ------
    for f in sorted(EXPT4.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        df = float(str(d["df"]))
        P = 1e8 * df
        s = d["tokens"].astype(float) / BATCH_SIZE
        N = N0  # base cell
        _add_curve(out, s, P, 1.0, N, d["val_loss"].astype(float), "expt4")

    # ------ expt 2: at base cell, P=20M ------
    P2 = 20_000_000.0
    for strat in STRATEGIES:
        # E=1: per-snapshot mean over 20 individuals
        ind_files = sorted((EXPT2 / "individuals").glob(f"{strat}_model*.npz"),
                           key=lambda p: int(p.stem.split("model")[-1]))
        Ls = []
        tok_ref = None
        for p in ind_files:
            d = np.load(p, allow_pickle=True)
            if tok_ref is None: tok_ref = d["tokens"]
            Ls.append(d["val_loss"].astype(float))
        s = tok_ref.astype(float) / BATCH_SIZE
        e1 = np.stack(Ls, axis=0).mean(axis=0)
        _add_curve(out, s, P2, 1.0, N0, e1, "expt2")
        # E ∈ {2,5,10,15,20}
        for E in (2, 5, 10, 15, 20):
            d = np.load(EXPT2 / "ensembles" / f"{strat}_E{E}.npz", allow_pickle=True)
            s = d["tokens"].astype(float) / BATCH_SIZE
            _add_curve(out, s, P2, float(E), N0,
                       d["val_loss"].astype(float), "expt2")

    # ------ expt 3: 4×4 (d, w) grid at P=20M ------
    grid = load_grid()
    for d_ in DEPTHS:
        for w_ in WIDTHS:
            N_cell = 16 * d_ * w_ * w_
            for strat in STRATEGIES:
                cell = grid[(d_, w_, strat)]
                if not cell.individuals or len(cell.ensembles) < len(E_SIZES):
                    continue
                tok = cell.individuals[0][0]
                s = tok.astype(float) / BATCH_SIZE
                # E=1: per-snapshot mean of the 5 individuals
                indiv = np.stack([vl for _, vl in cell.individuals], axis=0)
                _add_curve(out, s, P2, 1.0, N_cell, indiv.mean(axis=0), "expt3")
                # E ∈ {2,3,4,5}
                for E_ in E_SIZES:
                    if E_ in cell.ensembles:
                        _, vl = cell.ensembles[E_]
                        _add_curve(out, s, P2, float(E_), N_cell,
                                   vl.astype(float), "expt3")

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

def predict(p, s, P, E, N):
    c1, c2, c3, c4, dS1, dN, dP1, dP2, dS2, sig2 = p
    sn = s / S0
    Pn = P / P0
    Nn = N / N0
    return (c1 * sn ** dS1
            + c2 * Nn ** dN
            + c3 * Pn ** dP1
            + (c4 / E) * Pn ** dP2 * sn ** dS2
            + sig2)


def fit_global(ds, n_restarts=20, seed=0):
    s, P, E, N, L_obs = ds["s"], ds["P"], ds["E"], ds["N"], ds["L_obs"]

    # [c1, c2, c3, c4, δ_s1, δ_N, δ_P1, δ_P2, δ_s2, σ²]
    # δ_P1 capped at [-2, 0] to enforce smooth power-law (else fit pins at -∞).
    p0_base = np.array([3.0, 0.3, 0.3, 0.5, -1.0, -0.3, -1.0, -1.3, 0.7, 2.5])
    lo      = np.array([1e-4, 1e-4, 1e-4, 1e-4, -3.0, -3.0, -2.0, -3.0, -3.0, 0.0])
    hi      = np.array([50.0, 50.0, 50.0, 50.0,  3.0,  3.0,  0.0,  3.0,  3.0, 5.0])

    def residuals(par):
        pred = predict(par, s, P, E, N)
        return np.log(np.maximum(pred, 1e-3)) - np.log(np.maximum(L_obs, 1e-3))

    rng = np.random.default_rng(seed)
    best = None
    for r in range(n_restarts):
        if r == 0:
            p0 = p0_base.copy()
        else:
            p0 = p0_base.copy()
            # jitter: log-space for amplitudes/σ², linear for exponents
            for i in range(len(p0)):
                if PARAM_NAMES[i].startswith("c") or PARAM_NAMES[i] == "sigma2":
                    p0[i] *= float(np.exp(rng.normal(0, 0.5)))
                else:
                    p0[i] += float(rng.normal(0, 0.4))
            p0 = np.clip(p0, lo + 1e-6, hi - 1e-6)
        try:
            res = least_squares(residuals, p0, bounds=(lo, hi),
                                method="trf", max_nfev=30000, xtol=1e-12)
            cost = float((res.fun ** 2).sum())
            if best is None or cost < best[0]:
                best = (cost, res)
        except Exception:
            continue

    res = best[1]
    pred = predict(res.x, s, P, E, N)
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

    print("loading combined dataset (expt2 + expt3 + expt4, both strategies pooled) …")
    ds_full = load_all_pooled()
    ds_fit, keep = filter_leading_prefix(ds_full)
    print(f"  n_full = {len(ds_full['L_obs'])}, "
          f"n_fit (leading L>{LOSS_FILTER} dropped) = {len(ds_fit['L_obs'])} "
          f"(dropped {len(ds_full['L_obs']) - len(ds_fit['L_obs'])})")
    for src in np.unique(ds_full["source"]):
        n_full = int((ds_full["source"] == src).sum())
        n_kept = int((ds_fit["source"] == src).sum())
        print(f"    {src:<8}  n_full = {n_full:>5}, n_kept = {n_kept:>5}")

    f = fit_global(ds_fit, n_restarts=24)
    print(f"\nfit:  σ_resid = {f['sigma_resid']:.4f},  R² = {f['r2']:.4f},  "
          f"n_obs = {f['n_obs']}")
    for nm, v, sev in zip(PARAM_NAMES, f["x"], f["se"]):
        print(f"  {nm:<10} = {v:+.4f} ± {sev:.4f}")

    # ---------------- figure ----------------
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(20, 7.5))

    # Panel A: vary P (E=1) with model overlay
    palette_P = sns.color_palette("cool", 10)
    P_picks = [10e6, 30e6, 50e6, 70e6, 100e6]
    for j, Pp in enumerate(P_picks):
        m = (ds_full["P"] == Pp) & (ds_full["E"] == 1.0) & (ds_full["source"] == "expt4")
        if m.sum() == 0: continue
        s_p = ds_full["s"][m]; L_p = ds_full["L_obs"][m]
        order = np.argsort(s_p)
        idx_pal = int(round((Pp / 1e7 - 1) / 9 * (len(palette_P) - 1)))
        col = palette_P[idx_pal]
        axA.plot(s_p[order], L_p[order], "o", ms=5, color=col, alpha=0.6,
                 markeredgecolor="black", markeredgewidth=0.3,
                 label=f"$P = {int(Pp/1e6)}$M")
        s_grid = np.geomspace(s_p.min(), s_p.max(), 200)
        L_grid = predict(f["x"], s_grid, np.full_like(s_grid, Pp),
                         np.ones_like(s_grid), np.full_like(s_grid, N0))
        axA.plot(s_grid, L_grid, "-", lw=2.5, color=col, alpha=0.9)
    axA.set_xscale("log")
    axA.set_ylim(3.5, 8.0)
    axA.set_xlabel(r"steps  $s$", fontsize=28)
    axA.set_ylabel(r"$\mathcal{L}$", fontsize=28)
    axA.set_title(r"(A)  vary $P$  ($E{=}1$) — fit overlay",
                  fontsize=22, loc="left")
    axA.legend(loc="upper right", fontsize=15)

    # Panel B: vary E at P=20M with model overlay
    palette_E = sns.color_palette("cool", 6)
    P2 = 20e6
    for j, E in enumerate([1, 2, 5, 10, 15, 20]):
        m = (ds_full["P"] == P2) & (ds_full["E"] == E) & (ds_full["source"] == "expt2")
        if m.sum() == 0: continue
        s_p = ds_full["s"][m]; L_p = ds_full["L_obs"][m]
        order = np.argsort(s_p)
        col = palette_E[j]
        axB.plot(s_p[order], L_p[order], "o", ms=5, color=col, alpha=0.6,
                 markeredgecolor="black", markeredgewidth=0.3,
                 label=f"$E = {E}$")
        s_grid = np.geomspace(s_p.min(), s_p.max(), 200)
        L_grid = predict(f["x"], s_grid, np.full_like(s_grid, P2),
                         np.full_like(s_grid, E), np.full_like(s_grid, N0))
        axB.plot(s_grid, L_grid, "-", lw=2.5, color=col, alpha=0.9)
    axB.set_xscale("log")
    axB.set_ylim(3.8, 6.7)
    axB.set_xlabel(r"steps  $s$", fontsize=28)
    axB.set_ylabel(r"$\mathcal{L}$", fontsize=28)
    axB.set_title(r"(B)  vary $E$ at $P{=}20$M — fit overlay",
                  fontsize=22, loc="left")
    axB.legend(loc="upper right", fontsize=14, ncol=2)

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
