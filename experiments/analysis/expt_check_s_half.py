"""Quick check: does L(s) - L_min ∝ s^{1/2} in the post-nadir climb?

Theory predicts the variance term grows as s^{1/2}. The post-nadir overfit
climb should be dominated by that term, so

    L(s) - L_min  ≈  C · s^{1/2}     for s > s_nadir

Test on:
  expt2/init_ens (P=20M, varies E)
  expt4/init_ens (E=1, varies P)

For each curve, we fit log(L - L_min) ≈ log(C) + p · log(s) on the post-nadir
points and report p. If theory is right, p ≈ 0.5; we previously got ~0.78.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]
EXPT2 = REPO / "data_export" / "expt2_ensemble"
EXPT4 = REPO / "data_export" / "expt4_datasize" / "wd0_fixed_tokens"
OUT_DIR = REPO / "experiments" / "figures" / "05_global_fit"

BATCH_SIZE = 131072
STRAT = "init_ens"


def load_expt2_curves():
    """Return list of (label, s, L) for E ∈ {1,2,5,10,15,20} at P=20M."""
    curves = []
    # E=1: per-snapshot mean across 20 individuals
    paths = sorted((EXPT2 / "individuals").glob(f"{STRAT}_model*.npz"),
                   key=lambda p: int(p.stem.split("model")[-1]))
    Ls, tok_ref = [], None
    for p in paths:
        d = np.load(p, allow_pickle=True)
        if tok_ref is None: tok_ref = d["tokens"]
        Ls.append(d["val_loss"].astype(float))
    s = tok_ref.astype(float) / BATCH_SIZE
    curves.append(("E=1, P=20M", s, np.stack(Ls, axis=0).mean(axis=0)))
    for E in (2, 5, 10, 15, 20):
        d = np.load(EXPT2 / "ensembles" / f"{STRAT}_E{E}.npz", allow_pickle=True)
        s = d["tokens"].astype(float) / BATCH_SIZE
        curves.append((f"E={E}, P=20M", s, d["val_loss"].astype(float)))
    return curves


def load_expt4_curves():
    """Return list of (label, s, L) for E=1 at P ∈ {10,...,100}M."""
    curves = []
    for f in sorted(EXPT4.glob(f"*_{STRAT}.npz")):
        d = np.load(f, allow_pickle=True)
        df = float(str(d["df"]))
        P = int(1e8 * df)
        s = d["tokens"].astype(float) / BATCH_SIZE
        curves.append((f"E=1, P={P//1_000_000}M", s, d["val_loss"].astype(float)))
    return curves


def fit_late_slope(s, L):
    """Fit  L(s) = L_floor + C · s^p  on post-nadir points (3-param NLLS).
    Returns (p, p_se, n_used, s_nadir, L_min).
    """
    from scipy.optimize import least_squares
    L_min = float(L.min())
    i_min = int(L.argmin())
    s_nadir = float(s[i_min])
    s_post = s[i_min:]      # include nadir itself for the floor anchor
    L_post = L[i_min:]
    n = len(s_post)
    if n < 5:
        return None, None, n, s_nadir, L_min

    def residuals(params):
        L_floor, log_C, p = params
        return (L_floor + np.exp(log_C) * s_post ** p) - L_post

    p0 = np.array([L_min - 0.05, np.log(0.001), 0.5])
    lo = np.array([0.0, -20.0, 0.05])
    hi = np.array([L_min + 0.01, 20.0, 5.0])
    try:
        res = least_squares(residuals, p0, bounds=(lo, hi), max_nfev=10000)
    except Exception:
        return None, None, n, s_nadir, L_min
    p_hat = float(res.x[2])
    # Jacobian SE on p (3rd param)
    J = res.jac
    sigma2 = (res.fun ** 2).sum() / max(n - 3, 1)
    try:
        cov = sigma2 * np.linalg.inv(J.T @ J)
        p_se = float(np.sqrt(cov[2, 2]))
    except np.linalg.LinAlgError:
        p_se = float("nan")
    return p_hat, p_se, n, s_nadir, L_min


def main():
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 24, "axes.linewidth": 4.0, "legend.fontsize": 14,
        "grid.alpha": 0.25, "xtick.labelsize": 20, "ytick.labelsize": 20,
    })

    expt2_curves = load_expt2_curves()
    expt4_curves = load_expt4_curves()

    print("=== expt 2 (P=20M, varies E)  —  fit late-s slope p ===")
    print(f"  theory: p = 0.5")
    print(f"  {'curve':<14} {'p':>6} {'p_se':>6} {'n':>4} {'s_nadir':>9} {'L_min':>7}")
    e2_results = []
    for label, s, L in expt2_curves:
        p, p_se, n, sn, Lm = fit_late_slope(s, L)
        e2_results.append((label, s, L, p, p_se, n, sn, Lm))
        if p is not None:
            print(f"  {label:<14} {p:>6.3f} {p_se:>6.3f} {n:>4} {sn:>9.0f} {Lm:>7.3f}")
        else:
            print(f"  {label:<14} (too few post-nadir points: n={n})")

    print()
    print("=== expt 4 (E=1, varies P)  —  fit late-s slope p ===")
    print(f"  {'curve':<14} {'p':>6} {'p_se':>6} {'n':>4} {'s_nadir':>9} {'L_min':>7}")
    e4_results = []
    for label, s, L in expt4_curves:
        p, p_se, n, sn, Lm = fit_late_slope(s, L)
        e4_results.append((label, s, L, p, p_se, n, sn, Lm))
        if p is not None:
            print(f"  {label:<14} {p:>6.3f} {p_se:>6.3f} {n:>4} {sn:>9.0f} {Lm:>7.3f}")
        else:
            print(f"  {label:<14} (too few post-nadir points: n={n})")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 2, figsize=(20, 7.5))
    palette_E = sns.color_palette("cool", 6)
    palette_P = sns.color_palette("cool", 10)

    # Left panel: expt 2
    axL = axes[0]
    for j, (label, s, L, p, p_se, n, sn, Lm) in enumerate(e2_results):
        post_s = s[s > sn]
        post_excess = (L[s > sn] - Lm)
        m = post_excess > 1e-4
        axL.loglog(post_s[m], post_excess[m], "o-", color=palette_E[j], lw=2.5,
                   ms=6, label=fr"{label}  ($p = {p:.2f}$)" if p is not None else label)
    # s^{1/2} reference, anchored at the median (s, excess) point
    all_s = np.concatenate([s[s > sn] for _, s, L, _, _, _, sn, _ in e2_results])
    all_e = np.concatenate([(L[s > sn] - Lm) for _, s, L, _, _, _, sn, Lm in e2_results
                            if (L[s > sn] - Lm > 1e-4).any()])
    keep = all_e > 1e-4
    log_anchor = np.median(np.log(all_e[keep]) - 0.5 * np.log(all_s[keep]))
    s_grid = np.geomspace(all_s.min(), all_s.max(), 100)
    axL.loglog(s_grid, np.exp(log_anchor) * s_grid ** 0.5, "k--", lw=2.5,
               alpha=0.85, label=r"theory:  $s^{1/2}$")
    axL.set_xlabel(r"steps  $s$", fontsize=24)
    axL.set_ylabel(r"$\mathcal{L}(s) - \mathcal{L}_{\min}$  (post-nadir climb)",
                   fontsize=22)
    axL.set_title(r"(A)  expt 2: $P = 20$M, varies $E$",
                  fontsize=20, loc="left")
    axL.legend(loc="upper left", fontsize=12, ncol=1)

    # Right panel: expt 4
    axR = axes[1]
    for j, (label, s, L, p, p_se, n, sn, Lm) in enumerate(e4_results):
        post_s = s[s > sn]
        post_excess = (L[s > sn] - Lm)
        m = post_excess > 1e-4
        axR.loglog(post_s[m], post_excess[m], "o-", color=palette_P[j], lw=2.5,
                   ms=6, label=fr"{label}  ($p = {p:.2f}$)" if p is not None else label)
    all_s4 = np.concatenate([s[s > sn] for _, s, L, _, _, _, sn, _ in e4_results])
    all_e4 = np.concatenate([(L[s > sn] - Lm) for _, s, L, _, _, _, sn, Lm in e4_results
                             if (L[s > sn] - Lm > 1e-4).any()])
    keep4 = all_e4 > 1e-4
    log_anchor4 = np.median(np.log(all_e4[keep4]) - 0.5 * np.log(all_s4[keep4]))
    s_grid4 = np.geomspace(all_s4.min(), all_s4.max(), 100)
    axR.loglog(s_grid4, np.exp(log_anchor4) * s_grid4 ** 0.5, "k--", lw=2.5,
               alpha=0.85, label=r"theory:  $s^{1/2}$")
    axR.set_xlabel(r"steps  $s$", fontsize=24)
    axR.set_ylabel(r"$\mathcal{L}(s) - \mathcal{L}_{\min}$", fontsize=22)
    axR.set_title("(B)  expt 4: $E = 1$, varies $P$",
                  fontsize=20, loc="left")
    axR.legend(loc="upper left", fontsize=11, ncol=2)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "expt_check_s_half.pdf"
    png = OUT_DIR / "expt_check_s_half.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=300)
    print(f"\nsaved {pdf}")
    print(f"saved {png}")

    # mean / median slope summary
    p_e2 = [r[3] for r in e2_results if r[3] is not None]
    p_e4 = [r[3] for r in e4_results if r[3] is not None]
    print(f"\nexpt 2 slopes:  mean = {np.mean(p_e2):.3f}, median = {np.median(p_e2):.3f}, "
          f"n = {len(p_e2)}")
    print(f"expt 4 slopes:  mean = {np.mean(p_e4):.3f}, median = {np.median(p_e4):.3f}, "
          f"n = {len(p_e4)}")


if __name__ == "__main__":
    main()
