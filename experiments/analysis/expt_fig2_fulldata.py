"""Expt Fig 2 (df=1.0 / full 100M-token data): ensemble scaling law
    L_{T,E} = a_T * E^{-b} + L_inf,T   at base size d12/w768.

Same structure as expt_fig2_ensemble_scaling.py (the df=0.2 version) but:
  - data source is the full-data (df=1.0) run `fulldata_20260528_150057`, d12/w768,
    parsed from the SLURM replay .out logs (offline wandb is fragmented across the
    resume segments, per project convention), NOT wandb.
  - available ensemble sizes are E in {2,3,4,5} (5 individuals trained), so the
    E lever-arm is shorter than the df=0.2 figure's {2,5,10,15,20}. The power-law
    fit in E still applies; b / L_inf,T are just estimated from 4 points per T.

Two replay segments per strategy are merged by token count:
    fd_replay20m_d12_w768_41095198_{0,1}.out   (0   -> 3.09B tokens)
    fd_replay20m_d12_w768_41151096_{0,1}.out   (3.1 -> 3.98B tokens)
array index 0 = init, index 1 = init+shuffle.

Panels (matching the df=0.2 figure):
  Left   : init   L_{T,E}-L_inf,T vs E (log-log) at several epoch snapshots + fits.
  Middle : init+shuffle, same (shared y-axis).
  Right  : prefactor a_T vs cumulative tokens T (log-log) for both strategies + fit.
"""
import os, re, glob
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO = "/ocean/projects/cis260161p/ymiao6/scaling/slowrun"
LOGS = os.path.join(REPO, "experiments/logs")
CELL = "d12_w768"
SIZES = np.array([2, 3, 4, 5], dtype=float)
# array-index -> wandb-style strategy key + pretty label
STRATS = ["init_ens", "init_shuffle_ens"]
STRAT_IDX = {"init_ens": 0, "init_shuffle_ens": 1}
STRAT_LABEL = {"init_ens": "init", "init_shuffle_ens": "init + shuffle"}
TOK_PER_EPOCH = 99_614_720  # 760 optim steps * 131072 batch (df=1.0, 100M tokens)
ENS_LINE = re.compile(r"\[step \d+ ens=(\d+)\] val_loss=([\d.]+) val_bpb=([\d.]+) tokens=(\d+)")


def parse_strategy(idx):
    """Merge all replay segments for one strategy index -> dict E -> {tokens: val_loss}."""
    per_E = {}
    files = sorted(glob.glob(f"{LOGS}/fd_replay20m_{CELL}_*_{idx}.out"))
    for f in files:
        with open(f) as fh:
            for line in fh:
                m = ENS_LINE.search(line)
                if not m:
                    continue
                E = int(m.group(1))
                if E not in (2, 3, 4, 5):
                    continue
                tok = int(m.group(4))
                per_E.setdefault(E, {})[tok] = float(m.group(2))
    return per_E, files


def build_grid():
    """Return L (n_E, n_T), tok (n_T,), epoch (n_T,) per strategy on a shared token grid."""
    raw = {}
    for strat in STRATS:
        per_E, files = parse_strategy(STRAT_IDX[strat])
        raw[strat] = per_E
        print(f"  {strat}: merged {len(files)} seg(s); "
              f"E={sorted(per_E)} ; n_pts(E=2)={len(per_E.get(2, {}))}")
    # shared token grid = tokens present for ALL E in BOTH strategies
    tok_sets = []
    for strat in STRATS:
        for E in SIZES:
            tok_sets.append(set(raw[strat][int(E)].keys()))
    common = sorted(set.intersection(*tok_sets))
    tok = np.array(common, dtype=float)
    epoch = np.rint(tok / TOK_PER_EPOCH).astype(int)
    out = {}
    for strat in STRATS:
        L = np.array([[raw[strat][int(E)][int(t)] for t in common] for E in SIZES])
        out[strat] = {"L": L, "tok": tok, "epoch": epoch}
    print(f"  shared grid: {len(common)} snapshots, "
          f"T = {tok[0]/1e6:.0f}M .. {tok[-1]/1e6:.0f}M (epoch {epoch[0]}..{epoch[-1]})")
    return out


def joint_fit_shared_Linf(L_init, L_shuf, b0=1.0):
    """L_{T,E} = a_T * E^{-b} + L_inf,T ; shared L_inf,T across strategies,
    per-strategy b and a_T. Returns dict of params + asymptotic SEs."""
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
    p0 = np.concatenate([[b0, b0], a0, a0, Linf0])
    lo = np.concatenate([[0.1, 0.1], np.zeros(2 * n_T), -np.inf * np.ones(n_T)])
    hi = np.concatenate([[5.0, 5.0], np.inf * np.ones(2 * n_T), np.inf * np.ones(n_T)])
    res = least_squares(residuals, p0, bounds=(lo, hi), max_nfev=40000)

    J = res.jac
    dof = max(J.shape[0] - J.shape[1], 1)
    sigma2 = (res.fun ** 2).sum() / dof
    try:
        se = np.sqrt(np.diag(sigma2 * np.linalg.inv(J.T @ J)))
    except np.linalg.LinAlgError:
        se = np.full(J.shape[1], np.nan)
    return {
        "b_init": res.x[0], "b_init_se": se[0],
        "b_shuffle": res.x[1], "b_shuffle_se": se[1],
        "a_T_init": res.x[2:2 + n_T], "a_T_init_se": se[2:2 + n_T],
        "a_T_shuffle": res.x[2 + n_T:2 + 2 * n_T], "a_T_shuffle_se": se[2 + n_T:2 + 2 * n_T],
        "Linf_T": res.x[2 + 2 * n_T:2 + 3 * n_T], "Linf_T_se": se[2 + 2 * n_T:2 + 3 * n_T],
    }


def power_law_fit(x, y, y_se=None):
    """y = c * x^p via (weighted) log-linear regression. Returns (c, p, p_se)."""
    mask = (y > 0) & np.isfinite(x) & np.isfinite(y)
    if y_se is not None:
        mask = mask & (y_se > 0) & np.isfinite(y_se)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    lx, ly = np.log(x[mask]), np.log(y[mask])
    if y_se is not None:
        w = (y[mask] / y_se[mask]) ** 2
        X = np.column_stack([lx, np.ones_like(lx)])
        W = np.diag(w)
        beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ ly)
        cov = np.linalg.inv(X.T @ W @ X) * (np.sum(w * (ly - X @ beta) ** 2) / max(len(lx) - 2, 1))
        p, log_c = beta
        return np.exp(log_c), p, np.sqrt(cov[0, 0])
    p, log_c = np.polyfit(lx, ly, 1)
    return np.exp(log_c), p, np.nan


def main():
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "axes.labelsize": 24, "axes.linewidth": 4.0, "legend.fontsize": 18,
        "grid.alpha": 0.25, "xtick.labelsize": 20, "ytick.labelsize": 20,
    })

    print("Parsing full-data (df=1.0) replay logs...")
    data = build_grid()
    L_init = data["init_ens"]["L"]
    L_shuf = data["init_shuffle_ens"]["L"]
    T_ref = data["init_ens"]["tok"]
    epoch = data["init_ens"]["epoch"]

    res = joint_fit_shared_Linf(L_init, L_shuf)
    fits = {
        "init_ens": {"b": res["b_init"], "b_se": res["b_init_se"],
                     "a_T": res["a_T_init"], "a_T_se": res["a_T_init_se"]},
        "init_shuffle_ens": {"b": res["b_shuffle"], "b_se": res["b_shuffle_se"],
                             "a_T": res["a_T_shuffle"], "a_T_se": res["a_T_shuffle_se"]},
    }
    for strat in STRATS:
        c, p, p_se = power_law_fit(T_ref.astype(float), fits[strat]["a_T"], fits[strat]["a_T_se"])
        fits[strat].update(c=c, p=p, p_se=p_se)

    print(f"\nShared-L_inf joint fit (df=1.0, d12/w768, E in {{2,3,4,5}}):")
    for tag, b, bse in [("init", res["b_init"], res["b_init_se"]),
                        ("init+shuffle", res["b_shuffle"], res["b_shuffle_se"])]:
        z = (b - 1.0) / max(bse, 1e-12)
        print(f"  b_{tag:12s} = {b:.3f} ± {bse:.3f}   (H0 b=1: z={z:+.2f}, p={2*(1-norm.cdf(abs(z))):.3g})")
    for strat in STRATS:
        print(f"  a_T_{STRAT_LABEL[strat]:12s} = c*T^p : c={fits[strat]['c']:.3e}, "
              f"p={fits[strat]['p']:.3f} ± {fits[strat]['p_se']:.3f}")

    # ---- 3-panel figure ----
    fig = plt.figure(figsize=(24, 7))
    gs = fig.add_gridspec(1, 3, wspace=0.32)
    ax_L = fig.add_subplot(gs[0, 0])
    ax_M = fig.add_subplot(gs[0, 1], sharey=ax_L)
    ax_R = fig.add_subplot(gs[0, 2])
    axes = [ax_L, ax_M, ax_R]

    snap_epochs = [5, 10, 20, 30, 40]
    palette_T = sns.color_palette("cool", len(snap_epochs))
    palette_S = sns.color_palette("cool", 2)
    strat_marker = {"init_ens": "o", "init_shuffle_ens": "s"}
    strat_ls = {"init_ens": "-", "init_shuffle_ens": "--"}

    def nearest_idx(ep):
        return int(np.argmin(np.abs(epoch - ep)))

    for ax_idx, strat in enumerate(STRATS):
        ax = axes[ax_idx]
        L_grid = data[strat]["L"]
        b, a_T, Linf_T = fits[strat]["b"], fits[strat]["a_T"], res["Linf_T"]
        for j, ep in enumerate(snap_epochs):
            idx = nearest_idx(ep)
            gap = L_grid[:, idx] - Linf_T[idx]
            ax.loglog(SIZES, gap, marker=strat_marker[strat], lw=0,
                      color=palette_T[j], markersize=11,
                      markeredgecolor="black", markeredgewidth=0.6,
                      label=f"T = {epoch[idx]} ep ({T_ref[idx]/1e6:.0f}M)")
            E_grid = np.geomspace(SIZES.min() * 0.9, SIZES.max() * 1.1, 100)
            ax.loglog(E_grid, a_T[idx] * E_grid ** (-b),
                      color=palette_T[j], lw=2.0, ls=strat_ls[strat], alpha=0.7)
        ax.set_xlabel(r"ensemble size $E$")
        if ax_idx == 0:
            ax.set_ylabel(r"$L_{T,E} - L_{\infty,T}$")
        ax.set_xticks([2, 3, 4, 5])
        ax.set_xticklabels(["2", "3", "4", "5"])
        ax.text(0.02, 0.05, f"{STRAT_LABEL[strat]}\n$b = {b:.2f} \\pm {fits[strat]['b_se']:.2f}$",
                transform=ax.transAxes, fontsize=20, fontweight="bold", va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="lightgray"))
        ax.legend(loc="upper right", fontsize=14)

    ax = axes[2]
    for j, strat in enumerate(STRATS):
        a_T, a_T_se = fits[strat]["a_T"], fits[strat]["a_T_se"]
        c, p, p_se = fits[strat]["c"], fits[strat]["p"], fits[strat]["p_se"]
        ax.errorbar(T_ref, a_T, yerr=a_T_se, fmt=strat_marker[strat], markersize=7,
                    color=palette_S[j], markeredgecolor="black", markeredgewidth=0.4,
                    elinewidth=1.0, capsize=2, alpha=0.7,
                    label=f"{STRAT_LABEL[strat]}: $a_T \\propto T^{{{p:.2f} \\pm {p_se:.2f}}}$")
        T_grid = np.geomspace(T_ref.min(), T_ref.max(), 100)
        ax.loglog(T_grid, c * T_grid ** p, lw=2.5, color=palette_S[j], alpha=0.8, ls=strat_ls[strat])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(T_ref.min() * 0.7, T_ref.max() * 1.4)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.set_xlabel(r"cumulative tokens $T$")
    ax.set_ylabel(r"loss-gap prefactor $a_T$")
    ax.legend(loc="upper left", fontsize=15)

    fig.suptitle("Ensemble scaling law $L_{T,E}=a_T E^{-b}+L_{\\infty,T}$  —  "
                 "full data (df=1.0, 100M tokens), d12/w768",
                 fontsize=22, y=1.03)

    outdir = os.path.join(REPO, "experiments/figures/02_ensemble_scaling")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "expt_fig2_fulldata.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"\nSaved {out}\nSaved {out.replace('.pdf', '.png')}")

    # fit table
    csv = out.replace(".pdf", "_fits.csv")
    with open(csv, "w") as fh:
        fh.write("T_tokens,epoch,init_a_T,init_a_T_se,shuffle_a_T,shuffle_a_T_se,Linf_T,Linf_T_se\n")
        for i in range(len(T_ref)):
            fh.write(f"{int(T_ref[i])},{epoch[i]},"
                     f"{fits['init_ens']['a_T'][i]:.6f},{fits['init_ens']['a_T_se'][i]:.6f},"
                     f"{fits['init_shuffle_ens']['a_T'][i]:.6f},{fits['init_shuffle_ens']['a_T_se'][i]:.6f},"
                     f"{res['Linf_T'][i]:.6f},{res['Linf_T_se'][i]:.6f}\n")
    print(f"Saved fit table {csv}")


if __name__ == "__main__":
    main()
