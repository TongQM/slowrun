"""Slice plots: data + fitted-model curves on the s axis at fixed (N, P, E).

Three panels, each showing val_loss vs cumulative tokens s (log x) with markers
for the data and dashed lines for the fitted scaling law:

  Panel A — vary P at fixed N (= base d12/w768), fixed E (= 1).
            Source: Q1 sweep (10 dfs from 0.1 to 1.0).

  Panel B — vary E at fixed N (= base), fixed P (= df=0.2).
            Source: Q2 ext (E ∈ {1, 2, 5, 10, 15, 20}).

  Panel C — vary N at fixed P (= df=0.2), fixed E (= 5).
            Source: Q3 grid (16 cells × E=5).

Uses the LOG-space fit from expt_joint_scaling_law.py (less dominated by
high-loss q1 points).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import least_squares

import sys
sys.path.insert(0, "experiments/analysis")
from expt_joint_scaling_law import (
    load_q1_sweep, load_q2, load_q3, model, S0, N0, P0, non_embed_params
)


def fit_logspace():
    rows = load_q1_sweep() + load_q2() + load_q3()
    arr = np.array([(r[0], r[1], r[2], r[3], r[4]) for r in rows], dtype=np.float64)
    s, N, P, E, L_obs = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]

    def residuals_log(theta):
        pred = model(theta, s, N, P, E)
        pred = np.maximum(pred, 0.1)
        return np.log(pred) - np.log(L_obs)

    p0 = [0.5, -0.3, 0.05, 0.5, 0.5, -1.0, 0.2, -0.3, 0.5, 3.5]
    bounds_lo = [0.0, -3.0, 0.0, -3.0, 0.0, -5.0, 0.0, -3.0, 0.05, 0.0]
    bounds_hi = [10.0, 0.0, 10.0, 3.0, 10.0,  0.0, 10.0, 1.0, 5.0, 6.0]
    res = least_squares(residuals_log, p0, bounds=(bounds_lo, bounds_hi), max_nfev=50000)
    return res.x


def main():
    sns.set(font_scale=1.5)
    sns.set_style('whitegrid')
    plt.rcParams['axes.labelsize']  = 22
    plt.rcParams['axes.linewidth']  = 4.0
    plt.rcParams['legend.fontsize'] = 13
    plt.rcParams['grid.alpha']      = 0.25
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18

    print("Fitting (log-space)...")
    theta = fit_logspace()
    names = ["c_1", "d_s1", "c_2", "d_N", "c_3", "d_P1", "c_4", "d_P2", "d_s2", "sigma2"]
    print("Fitted parameters:")
    for n, v in zip(names, theta):
        print(f"  {n} = {v:+.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # ----- Panel A: vary P -----
    ax = axes[0]
    q1 = load_q1_sweep()
    dfs = sorted(set(float(str(r[2] / 99942400)) for r in q1))
    palette_P = sns.color_palette('cool', len(dfs))
    base_N = float(non_embed_params(12, 768))
    for i, df in enumerate(dfs):
        sel = [(r[0], r[4]) for r in q1 if abs(r[2] / 99942400 - df) < 1e-6]
        sel.sort()
        s_vals = np.array([x[0] for x in sel])
        L_vals = np.array([x[1] for x in sel])
        ax.plot(s_vals, L_vals, marker="o", lw=0, markersize=4,
                color=palette_P[i], alpha=0.5)
        # Model line
        s_grid = np.geomspace(s_vals.min(), s_vals.max(), 80)
        L_pred = model(theta, s_grid,
                       np.full_like(s_grid, base_N),
                       np.full_like(s_grid, df * 99942400),
                       np.ones_like(s_grid))
        ax.plot(s_grid, L_pred, "--", lw=2.2, color=palette_P[i],
                label=f"df = {df:.1f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"cumulative tokens $s$")
    ax.set_ylabel("val loss")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.legend(loc="upper right", ncol=2)
    ax.text(0.02, 0.97, "vary $P$  (E=1, N=d12/w768)",
            transform=ax.transAxes, fontsize=18, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.85, edgecolor="lightgray"))

    # ----- Panel B: vary E -----
    ax = axes[1]
    q2 = load_q2()
    Es = [1, 2, 5, 10, 15, 20]
    palette_E = sns.color_palette('cool', len(Es))
    for i, E in enumerate(Es):
        sel = [(r[0], r[4]) for r in q2 if int(r[3]) == E]
        sel.sort()
        if not sel: continue
        s_vals = np.array([x[0] for x in sel])
        L_vals = np.array([x[1] for x in sel])
        ax.plot(s_vals, L_vals, marker="o", lw=0, markersize=4,
                color=palette_E[i], alpha=0.5)
        s_grid = np.geomspace(s_vals.min(), s_vals.max(), 80)
        L_pred = model(theta, s_grid,
                       np.full_like(s_grid, base_N),
                       np.full_like(s_grid, 0.2 * 99942400),
                       np.full_like(s_grid, float(E)))
        ax.plot(s_grid, L_pred, "--", lw=2.2, color=palette_E[i],
                label=f"E = {E}")
    ax.set_xscale("log")
    ax.set_xlabel(r"cumulative tokens $s$")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.legend(loc="upper right")
    ax.text(0.02, 0.97, "vary $E$  (P=df0.2, N=d12/w768)",
            transform=ax.transAxes, fontsize=18, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.85, edgecolor="lightgray"))

    # ----- Panel C: vary N -----
    ax = axes[2]
    q3 = load_q3()
    cells = sorted(set((int(non_embed_params_inverse_d(r[1])), int(non_embed_params_inverse_w(r[1])))
                       for r in q3))  # placeholder; we'll get unique N values directly
    # Instead just use unique N values directly
    Ns = sorted(set(float(r[1]) for r in q3))
    palette_N = sns.color_palette('cool', len(Ns))
    P_q3 = 0.2 * 99942400
    E_target = 5.0
    for i, N_val in enumerate(Ns):
        sel = [(r[0], r[4]) for r in q3 if abs(r[1] - N_val) < 1e-3 and abs(r[3] - E_target) < 1e-6]
        sel.sort()
        if not sel: continue
        s_vals = np.array([x[0] for x in sel])
        L_vals = np.array([x[1] for x in sel])
        ax.plot(s_vals, L_vals, marker="o", lw=0, markersize=4,
                color=palette_N[i], alpha=0.5)
        s_grid = np.geomspace(s_vals.min(), s_vals.max(), 80)
        L_pred = model(theta, s_grid,
                       np.full_like(s_grid, N_val),
                       np.full_like(s_grid, P_q3),
                       np.full_like(s_grid, E_target))
        ax.plot(s_grid, L_pred, "--", lw=1.8, color=palette_N[i],
                label=f"N = {N_val/1e6:.0f}M")
    ax.set_xscale("log")
    ax.set_xlabel(r"cumulative tokens $s$")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.legend(loc="upper right", ncol=2, fontsize=10)
    ax.text(0.02, 0.97, "vary $N$  (E=5, P=df0.2)",
            transform=ax.transAxes, fontsize=18, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.85, edgecolor="lightgray"))

    out = "experiments/figures/04_scaling_law/expt_joint_slices.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches='tight', dpi=300)
    print(f"\nSaved {out}")


# Helper stubs (we use Ns directly from data instead)
def non_embed_params_inverse_d(N): return 12  # unused
def non_embed_params_inverse_w(N): return 768  # unused


if __name__ == "__main__":
    main()
