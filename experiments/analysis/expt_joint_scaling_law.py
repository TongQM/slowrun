"""Joint fit of the 4-axis scaling law to all available data.

Model:
  L(s, N, P, E) = c_1 s_r^{d_s1}
                + c_2 N_r^{d_N}
                + c_3 P_r^{d_P1}
                + (c_4 / E) * P_r^{d_P2} * s_r^{d_s2}
                + sigma2

Reduced variables: s_r = s / s_0, N_r = N / N_0, P_r = P / P_0
with anchor scales s_0 = 3e8 tokens, N_0 = 16 * 12 * 768^2 (non-embed params at base
cell d12/w768), P_0 = 2e7 tokens (df=0.2).

Data axes:
  s — every training-curve snapshot (tokens-seen per model)
  N — non-embed params (16 * d * w^2)
  P — data size in tokens (= df * 99942400)
  E — ensemble size (1 for individuals, 2..20 from replays)

Sources (all under `data_export/`):
  expt4_datasize/wd0_fixed_tokens/  : Q1 P-sweep (10 dfs at d12/w768, E=1, init+shuffle)
  expt2_ensemble/individuals/       : 20 inds at d12/w768/df0.2 (mean → E=1)
  expt2_ensemble/ensembles/         : E ∈ {2,5,10,15,20} at d12/w768/df0.2
  expt3_grid/individuals/           : 16 cells × 5 inds (mean per cell → E=1)
  expt3_grid/ensembles/             : 16 cells × {E=2,3,4,5}

We use the init+shuffle strategy for consistency with the rest of the paper.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import least_squares
from scipy.stats import norm

DATA = "data_export"

# Anchor scales (reduced variables s_r=s/s0, etc.)
S0 = 3e8
N0 = 16 * 12 * 768 ** 2  # base non-embed body params
P0 = 0.2 * 99942400      # df=0.2 in tokens

STRAT = "init_shuffle_ens"


def non_embed_params(d, w):
    return 16 * d * w ** 2


def load_q1_sweep():
    """Q1 P-sweep: d12/w768, E=1, init_shuffle_ens, 10 different P values."""
    rows = []
    base_N = non_embed_params(12, 768)
    for f in sorted(glob.glob(os.path.join(DATA, "expt4_datasize/wd0_fixed_tokens/df*_init_shuffle_ens.npz"))):
        z = np.load(f, allow_pickle=True)
        df = float(str(z["df"]))
        P = df * 99942400
        toks = z["tokens"]
        L = z["val_loss"]
        for s, l in zip(toks, L):
            rows.append((float(s), float(base_N), float(P), 1.0, float(l), "q1"))
    return rows


def load_q2():
    """Q2 ext: d12/w768, df=0.2, E ∈ {1,2,5,10,15,20}."""
    rows = []
    base_N = non_embed_params(12, 768)
    P = 0.2 * 99942400

    # E=1: mean of 20 individuals at each tokens point
    indiv_files = sorted(glob.glob(os.path.join(DATA, "expt2_ensemble/individuals/init_shuffle_ens_model*.npz")))
    by_tok = {}
    for f in indiv_files:
        z = np.load(f, allow_pickle=True)
        for s, l in zip(z["tokens"], z["val_loss"]):
            by_tok.setdefault(int(s), []).append(float(l))
    for s in sorted(by_tok):
        rows.append((float(s), float(base_N), float(P), 1.0, float(np.mean(by_tok[s])), "q2"))

    # E ∈ {2,5,10,15,20}
    for E in [2, 5, 10, 15, 20]:
        f = os.path.join(DATA, f"expt2_ensemble/ensembles/init_shuffle_ens_E{E}.npz")
        if not os.path.isfile(f): continue
        z = np.load(f, allow_pickle=True)
        for s, l in zip(z["tokens"], z["val_loss"]):
            rows.append((float(s), float(base_N), float(P), float(E), float(l), "q2"))
    return rows


def load_q3():
    """Q3 grid: 16 cells, df=0.2, E ∈ {1, 2, 3, 4, 5}."""
    rows = []
    P = 0.2 * 99942400
    DEPTHS = [6, 12, 18, 24]
    WIDTHS = [384, 768, 1152, 1536]
    for d in DEPTHS:
        for w in WIDTHS:
            N = non_embed_params(d, w)
            # E=1: mean of 5 individuals
            inds = []
            for m in range(5):
                f = os.path.join(DATA, f"expt3_grid/individuals/d{d}_w{w}_init_shuffle_ens_model{m}.npz")
                if not os.path.isfile(f): continue
                z = np.load(f, allow_pickle=True)
                inds.append((z["tokens"], z["val_loss"]))
            if inds:
                # Align to ensemble's tokens grid (since replay's grid is the canonical eval set)
                f_ref = os.path.join(DATA, f"expt3_grid/ensembles/d{d}_w{w}_init_shuffle_ens_E5.npz")
                if os.path.isfile(f_ref):
                    target = set(int(t) for t in np.load(f_ref, allow_pickle=True)["tokens"])
                    by_tok = {}
                    for t, v in inds:
                        for ti, vi in zip(t, v):
                            ti = int(ti)
                            if ti in target: by_tok.setdefault(ti, []).append(float(vi))
                    for ti, vs in by_tok.items():
                        rows.append((float(ti), float(N), float(P), 1.0, float(np.mean(vs)), "q3"))
            # Ensembles
            for E in [2, 3, 4, 5]:
                f = os.path.join(DATA, f"expt3_grid/ensembles/d{d}_w{w}_init_shuffle_ens_E{E}.npz")
                if not os.path.isfile(f): continue
                z = np.load(f, allow_pickle=True)
                for s, l in zip(z["tokens"], z["val_loss"]):
                    rows.append((float(s), float(N), float(P), float(E), float(l), "q3"))
    return rows


def model(theta, s, N, P, E):
    """Returns predicted L."""
    c1, d_s1, c2, d_N, c3, d_P1, c4, d_P2, d_s2, sigma2 = theta
    s_r = s / S0
    N_r = N / N0
    P_r = P / P0
    bias = c1 * s_r ** d_s1
    param = c2 * N_r ** d_N
    data_lim = c3 * P_r ** d_P1
    var = (c4 / E) * P_r ** d_P2 * s_r ** d_s2
    return bias + param + data_lim + var + sigma2


def fit(rows):
    arr = np.array([(r[0], r[1], r[2], r[3], r[4]) for r in rows], dtype=np.float64)
    s, N, P, E, L_obs = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    sources = np.array([r[5] for r in rows])

    def residuals(theta):
        return model(theta, s, N, P, E) - L_obs

    # theta = [c1, d_s1, c2, d_N, c3, d_P1, c4, d_P2, d_s2, sigma2]
    # Allow d_N to be positive (in the data-limited regime, bigger model can hurt).
    # Allow d_P1 down to -5 (first attempt hit -3 lower bound).
    p0 = [0.5, -0.3, 0.05, 0.0, 0.5, -1.0, 0.2, -0.3, 0.5, 3.5]
    bounds_lo = [0.0, -3.0, 0.0, -3.0, 0.0, -5.0, 0.0, -3.0, 0.05, 0.0]
    bounds_hi = [10.0, 0.0, 10.0, 3.0, 10.0,  0.0, 10.0, 1.0, 5.0, 6.0]
    res = least_squares(residuals, p0, bounds=(bounds_lo, bounds_hi), max_nfev=50000)

    J = res.jac
    n_obs, n_params = J.shape
    dof = max(n_obs - n_params, 1)
    sigma_resid = np.sqrt((res.fun ** 2).sum() / dof)
    try:
        cov = (sigma_resid ** 2) * np.linalg.inv(J.T @ J)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(n_params, np.nan)

    return {
        "theta": res.x, "se": se,
        "names": ["c_1", "d_s1", "c_2", "d_N", "c_3", "d_P1", "c_4", "d_P2", "d_s2", "sigma2"],
        "n_obs": n_obs, "n_params": n_params, "dof": dof,
        "sigma_resid": sigma_resid,
        "ssr": (res.fun ** 2).sum(),
        "predictions": L_obs - res.fun,
        "L_obs": L_obs, "s": s, "N": N, "P": P, "E": E, "sources": sources,
    }


def make_figure(fit_result):
    sns.set(font_scale=1.3)
    sns.set_style('whitegrid')
    plt.rcParams['axes.labelsize']  = 18
    plt.rcParams['axes.linewidth']  = 3.0
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['grid.alpha']      = 0.25
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    L_obs = fit_result["L_obs"]
    L_pred = fit_result["predictions"]
    sources = fit_result["sources"]
    resid = L_pred - L_obs

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    src_palette = {"q1": "tab:red", "q2": "tab:blue", "q3": "tab:green"}

    # Panel A: predicted vs measured
    ax = axes[0]
    for src, color in src_palette.items():
        sel = sources == src
        ax.scatter(L_obs[sel], L_pred[sel], s=12, alpha=0.6, color=color,
                   label=f"{src} ({sel.sum()} pts)", edgecolor="none")
    lo, hi = float(L_obs.min()), float(L_obs.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=2, alpha=0.6, label="identity")
    ax.set_xlabel("measured val loss")
    ax.set_ylabel("predicted val loss")
    ax.legend(loc="upper left")

    # Panel B: residuals vs s
    ax = axes[1]
    for src, color in src_palette.items():
        sel = sources == src
        ax.scatter(fit_result["s"][sel], resid[sel], s=12, alpha=0.6, color=color, label=src)
    ax.axhline(0, color="black", lw=1.5, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"cumulative tokens $s$")
    ax.set_ylabel("residual")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.legend(loc="upper right")

    # Panel C: residuals vs P (Q1) and N (Q3) — diagnostic for axis-specific misfit
    ax = axes[2]
    sel_q1 = sources == "q1"
    sel_q3 = sources == "q3"
    ax.scatter(fit_result["P"][sel_q1] / 1e6, resid[sel_q1], s=14, alpha=0.6,
               color="tab:red", label="q1 vs P")
    # For q3 we plot by N (model size), repurposing the same axis with a secondary
    secax = ax.twiny()
    secax.scatter(fit_result["N"][sel_q3] / 1e6, resid[sel_q3], s=14, alpha=0.6,
                  color="tab:green", marker="^", label="q3 vs N")
    ax.axhline(0, color="black", lw=1.5, alpha=0.5)
    ax.set_xscale("log"); secax.set_xscale("log")
    ax.set_xlabel(r"$P$ (M tokens)  for q1")
    secax.set_xlabel(r"$N$ (M params)  for q3")
    ax.set_ylabel("residual")
    ax.legend(loc="upper left")
    secax.legend(loc="upper right")

    out = "experiments/figures/04_scaling_law/expt_joint_scaling_law.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches='tight', dpi=300)
    print(f"Saved {out}")


def main():
    print("Loading data...")
    rows = []
    rows.extend(load_q1_sweep())
    rows.extend(load_q2())
    rows.extend(load_q3())
    print(f"  q1: {sum(1 for r in rows if r[5] == 'q1')} pts")
    print(f"  q2: {sum(1 for r in rows if r[5] == 'q2')} pts")
    print(f"  q3: {sum(1 for r in rows if r[5] == 'q3')} pts")
    print(f"  total: {len(rows)} pts")

    print("\nFitting...")
    fit_result = fit(rows)
    print(f"  n_obs={fit_result['n_obs']}, n_params={fit_result['n_params']}, dof={fit_result['dof']}")
    print(f"  residual std: {fit_result['sigma_resid']:.4f} nats   (SSR = {fit_result['ssr']:.2f})")
    print(f"\nFitted parameters (mean ± SE):")
    for n, v, s in zip(fit_result["names"], fit_result["theta"], fit_result["se"]):
        ratio = ""
        if v != 0:
            z = v / max(s, 1e-12)
            p = 2 * (1 - norm.cdf(abs(z)))
            ratio = f"   z = {z:+.2f},  p = {p:.3g}"
        print(f"  {n:<8} = {v:+.4f} ± {s:.4f}{ratio}")

    print("\nMaking figure...")
    make_figure(fit_result)


if __name__ == "__main__":
    main()
