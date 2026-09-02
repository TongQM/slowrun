"""Expt 2 sensitivity check: does including E=1 (per-snapshot individual mean)
in the joint fit change b?

The original Expt 2 fit uses only E ∈ {2, 5, 10, 15, 20} (post-hoc replays).
Expt 3 uses E ∈ {1, 2, 3, 4, 5}, where E=1 is the per-snapshot mean across
individuals. To make the two experiments comparable — and to test whether
including the E=1 anchor moves the headline b — we refit Expt 2 with E ∈
{1, 2, 5, 10, 15, 20} where E=1 is the mean across the 20 individuals at each
snapshot.

Reads from data_export/expt2_ensemble/. Saves a comparison table and a
side-by-side figure to
experiments/figures/02_ensemble_scaling/with_e1/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import least_squares
from scipy.stats import norm


REPO = Path(__file__).resolve().parents[2]
ENS_DIR = REPO / "data_export" / "expt2_ensemble" / "ensembles"
IND_DIR = REPO / "data_export" / "expt2_ensemble" / "individuals"
OUT_DIR = REPO / "experiments" / "figures" / "02_ensemble_scaling" / "with_e1"

STRATS = ["init_ens", "init_shuffle_ens"]
STRAT_LABEL = {"init_ens": "init", "init_shuffle_ens": "init + shuffle"}
ENS_E_SIZES = np.array([2, 5, 10, 15, 20], dtype=float)
ALL_E_SIZES = np.array([1, 2, 5, 10, 15, 20], dtype=float)


def load_grid_with_e1():
    """Return {strat: {"L_with_e1": (6, n_T), "L_orig": (5, n_T), "tok": (n_T,)}}."""
    out = {}
    for strat in STRATS:
        # Load 20 individuals → per-snapshot mean
        ind_files = sorted(IND_DIR.glob(f"{strat}_model*.npz"),
                           key=lambda p: int(p.stem.split("model")[-1]))
        ind_curves = []
        tok_ref = None
        for p in ind_files:
            d = np.load(p, allow_pickle=True)
            tok, vl = d["tokens"], d["val_loss"]
            order = np.argsort(tok)
            tok, vl = tok[order], vl[order]
            if tok_ref is None:
                tok_ref = tok
            else:
                # All 40-epoch individuals share the same token grid
                assert np.array_equal(tok, tok_ref), \
                    f"individual {p} token grid differs"
            ind_curves.append(vl)
        if not ind_curves:
            raise RuntimeError(f"No individuals for {strat}")
        ind_stack = np.stack(ind_curves, axis=0)        # (20, n_T)
        e1_mean = ind_stack.mean(axis=0)                 # (n_T,)

        # Load 5 ensembles
        rows = []
        for E in ENS_E_SIZES:
            d = np.load(ENS_DIR / f"{strat}_E{int(E)}.npz", allow_pickle=True)
            tok, vl = d["tokens"], d["val_loss"]
            order = np.argsort(tok)
            tok, vl = tok[order], vl[order]
            assert np.array_equal(tok, tok_ref), f"ens E={E} grid differs"
            rows.append(vl)
        L_orig = np.stack(rows, axis=0)                  # (5, n_T)
        L_with_e1 = np.vstack([e1_mean[None, :], L_orig])  # (6, n_T)
        out[strat] = {"L_with_e1": L_with_e1, "L_orig": L_orig, "tok": tok_ref}
        print(f"  {strat}: E=1 from {len(ind_curves)} individuals; "
              f"L_with_e1 shape {L_with_e1.shape}")
    return out


def joint_fit(L_init, L_shuf, E_sizes):
    """Linear-space joint fit with shared L_inf,T, per-strategy (b, a_T).

    L_init, L_shuf: (n_E, n_T)  (must use the same E_sizes order).
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
    dof = max(J.shape[0] - J.shape[1], 1)
    sigma2 = (res.fun ** 2).sum() / dof
    try:
        cov = sigma2 * np.linalg.inv(J.T @ J)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(J.shape[1], np.nan)

    return {
        "b_init":      res.x[0], "b_init_se":     se[0],
        "b_shuffle":   res.x[1], "b_shuffle_se":  se[1],
        "a_T_init":    res.x[2:2 + n_T],
        "a_T_init_se": se[2:2 + n_T],
        "a_T_shuffle":    res.x[2 + n_T:2 + 2 * n_T],
        "a_T_shuffle_se": se[2 + n_T:2 + 2 * n_T],
        "Linf_T":      res.x[2 + 2 * n_T:2 + 3 * n_T],
        "sigma":       float(np.sqrt(sigma2)),
    }


def wald(b, b_se, target=1.0):
    z = (b - target) / max(b_se, 1e-12)
    return z, 2 * (1 - norm.cdf(abs(z)))


def print_summary(name, fit):
    print(f"\n=== {name} ===")
    print(f"  b_init     = {fit['b_init']:.3f} ± {fit['b_init_se']:.3f}")
    print(f"  b_shuffle  = {fit['b_shuffle']:.3f} ± {fit['b_shuffle_se']:.3f}")
    z_i, p_i = wald(fit['b_init'], fit['b_init_se'])
    z_s, p_s = wald(fit['b_shuffle'], fit['b_shuffle_se'])
    print(f"  H0: b_init=1     z={z_i:+.2f}  p={p_i:.2e}")
    print(f"  H0: b_shuffle=1  z={z_s:+.2f}  p={p_s:.2e}")
    diff = fit['b_shuffle'] - fit['b_init']
    diff_se = np.sqrt(fit['b_init_se'] ** 2 + fit['b_shuffle_se'] ** 2)
    z_d = diff / diff_se
    p_d = 2 * (1 - norm.cdf(abs(z_d)))
    print(f"  Δb = {diff:+.3f} ± {diff_se:.3f}  z={z_d:+.2f}  p={p_d:.2e}")
    print(f"  residual sigma = {fit['sigma']:.4g}")


def make_figure(data, fit_orig, fit_e1):
    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 22, "axes.linewidth": 3.0, "legend.fontsize": 14,
        "grid.alpha": 0.25, "xtick.labelsize": 16, "ytick.labelsize": 16,
    })

    snap_epochs = [5, 10, 15, 25, 40]
    palette_T = sns.color_palette("cool", len(snap_epochs))
    strat_marker = {"init_ens": "o", "init_shuffle_ens": "s"}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)

    for ax_idx, strat in enumerate(STRATS):
        ax = axes[ax_idx]
        L_grid = data[strat]["L_with_e1"]
        T = data[strat]["tok"]
        Linf_e1 = fit_e1["Linf_T"]
        b_orig = fit_orig["b_init"] if strat == "init_ens" else fit_orig["b_shuffle"]
        a_T_orig = fit_orig["a_T_init"] if strat == "init_ens" else fit_orig["a_T_shuffle"]
        Linf_orig = fit_orig["Linf_T"]
        b_e1 = fit_e1["b_init"] if strat == "init_ens" else fit_e1["b_shuffle"]
        a_T_e1 = fit_e1["a_T_init"] if strat == "init_ens" else fit_e1["a_T_shuffle"]

        for j, snap_ep in enumerate(snap_epochs):
            idx = int(snap_ep) - 1
            if idx >= L_grid.shape[1]: continue
            gap_obs = np.maximum(L_grid[:, idx] - Linf_e1[idx], 1e-6)
            ax.loglog(ALL_E_SIZES, gap_obs, marker=strat_marker[strat], lw=0,
                      color=palette_T[j], markersize=10,
                      markeredgecolor="black", markeredgewidth=0.5,
                      label=f"T = {snap_ep} ep ({T[idx]/1e6:.0f}M)")
            E_grid = np.geomspace(0.85, 22, 100)
            # Original (E∈{2..20}) prediction
            pred_orig = a_T_orig[idx] * E_grid ** (-b_orig) + (Linf_orig[idx] - Linf_e1[idx])
            ax.loglog(E_grid, np.maximum(pred_orig, 1e-8),
                      color=palette_T[j], lw=1.6, ls=":", alpha=0.75)
            # New (E∈{1..20}) prediction
            pred_e1 = a_T_e1[idx] * E_grid ** (-b_e1)
            ax.loglog(E_grid, np.maximum(pred_e1, 1e-8),
                      color=palette_T[j], lw=2.0, ls="-", alpha=0.85)

        ax.set_xlabel(r"ensemble size $E$ (with $E{=}1$ point)")
        if ax_idx == 0:
            ax.set_ylabel(r"$L_{T,E} - \widehat{L}_{\infty,T}^{\text{(with } E=1\text{)}}$")
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.set_xticklabels(["1", "2", "5", "10", "20"])
        ax.text(0.02, 0.05,
                f"{STRAT_LABEL[strat]}\n"
                f"orig E∈{{2..20}}: b = {b_orig:.2f} ± "
                f"{(fit_orig['b_init_se' if strat=='init_ens' else 'b_shuffle_se']):.2f}\n"
                f"with E=1:        b = {b_e1:.2f} ± "
                f"{(fit_e1['b_init_se' if strat=='init_ens' else 'b_shuffle_se']):.2f}",
                transform=ax.transAxes, fontsize=12, va="bottom", ha="left",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.9, edgecolor="lightgray"))
        ax.legend(loc="upper right", fontsize=11)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "expt_fig2_with_e1.pdf"
    png = OUT_DIR / "expt_fig2_with_e1.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=300)
    fig.savefig(png, bbox_inches="tight", dpi=200)
    print(f"\nSaved {pdf}")
    print(f"Saved {png}")


def main():
    print("Loading expt2 data with E=1 (mean of 20 individuals) ...")
    data = load_grid_with_e1()

    L_orig_init = data["init_ens"]["L_orig"]
    L_orig_shuf = data["init_shuffle_ens"]["L_orig"]
    L_e1_init = data["init_ens"]["L_with_e1"]
    L_e1_shuf = data["init_shuffle_ens"]["L_with_e1"]

    fit_orig = joint_fit(L_orig_init, L_orig_shuf, ENS_E_SIZES)
    print_summary("ORIGINAL  E ∈ {2, 5, 10, 15, 20}", fit_orig)

    fit_e1 = joint_fit(L_e1_init, L_e1_shuf, ALL_E_SIZES)
    print_summary("WITH E=1  E ∈ {1, 2, 5, 10, 15, 20}", fit_e1)

    # Side-by-side delta
    print("\n=== Δ = with-E1 minus original ===")
    for tag in ("init", "shuffle"):
        b_o, se_o = fit_orig[f"b_{tag}"], fit_orig[f"b_{tag}_se"]
        b_n, se_n = fit_e1[f"b_{tag}"], fit_e1[f"b_{tag}_se"]
        print(f"  b_{tag:<8}  orig {b_o:.3f} ± {se_o:.3f}  →  "
              f"with-E1 {b_n:.3f} ± {se_n:.3f}   (Δ {b_n - b_o:+.3f})")

    make_figure(data, fit_orig, fit_e1)


if __name__ == "__main__":
    main()
