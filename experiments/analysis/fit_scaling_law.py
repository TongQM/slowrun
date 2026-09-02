"""Joint scaling-law fit at fixed data D=100M (df=1.0): L*(w, L, E, wd).

Model (additive, separable — tested against the df=0.2 E x size grid):

    L*(w, L, E, wd) = Linf
                      + A * ( (w/w0)^2 * (L/L0)^kappa )^(-alpha)   # capacity
                      + B * E^(-beta)                              # ensemble
                      + delta(wd)                                  # recipe (cooldown+wd)

  w0=768, L0=12 (base cell). kappa>1 means depth buys more loss-reduction per
  parameter than width (matches "depth beats width at matched params"). The
  ensemble term is the variance-like component removed by averaging E models;
  as E->inf it vanishes and the model sits at its capacity floor Linf+A*cap.

  Capacity + ensemble shape (Linf,A,alpha,kappa,B,beta) is fit on the CLEAN grid
  (no-warmdown, wd=0). delta(wd) is fit separately from the base-cell cooldown wd
  sweep as the offset relative to the clean base E-curve; delta(no-warmdown,wd=0)=0.

Compute (relative to base d12/w768/E1):  C = (w/w0)^2 * (L/L0) * E   (params x E).

All L* parsed from SLURM .out logs (reusing the figure scripts' helpers).
"""
import os, re, glob, sys
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expt_fig2_combined_fulldata import (
    LOGS, BATCH, NUM_MODELS, mean_indiv_curve, ensemble_curves,
    indiv_curves_for_width,
)
from expt_fig2_combined_fulldata_depth import indiv_curves_for_depth

W0, L0 = 768, 12

# ---- clean grid (no-warmdown, wd=0) cells: (w, L) ----
WIDTH_CELLS = [(384, 12), (768, 12), (1152, 12), (1536, 12)]
DEPTH_CELLS = [(768, 6), (768, 12), (768, 18), (768, 24), (768, 48), (768, 60)]

# ---- base-cell cooldown wd sweep job IDs (init_shuffle, sidx=1) ----
ON_TRAIN = {"0.0": 41609119, "0.05": 41609121, "0.1": 41431931,
            "0.3": 41431933, "0.5": 41431935, "0.8": 41431937}
ON_REPLAY = {"0.0": 41609120, "0.05": 41609122, "0.1": 41431932,
             "0.3": 41431934, "0.5": 41431936, "0.8": 41431938}
ENS_LINE = re.compile(r"\[step \d+ ens=([2-9])\] val_loss=([\d.]+)")
IND_LINE = re.compile(r"\[model \d+ val @ step (\d+)\] val_loss=([\d.]+)")


def lstar_size(w, L):
    curves = indiv_curves_for_width(w) if L == 12 else indiv_curves_for_depth(L)
    if L != 12 and w != 768:
        return None
    g, v = mean_indiv_curve(curves)
    return None if v is None else float(np.min(v))


def wd_sweep():
    """base-cell cooldown: dict wd -> {E: L*}, E in {1..5}."""
    out = {}
    for wd, jid in ON_TRAIN.items():
        curves = []
        for m in range(NUM_MODELS):
            task = 5 + m  # init_shuffle sidx=1
            d = {}
            for f in glob.glob(f"{LOGS}/wd{wd}_train_d12_w768_{jid}_{task}.out"):
                for line in open(f):
                    mm = IND_LINE.search(line)
                    if mm:
                        d[int(mm.group(1))] = float(mm.group(2))
            if d:
                s = np.array(sorted(d)); curves.append((s, np.array([d[k] for k in s])))
        g, v = mean_indiv_curve(curves)
        out.setdefault(wd, {})
        if v is not None:
            out[wd][1] = float(np.min(v))
    for wd, jid in ON_REPLAY.items():
        perE = {}
        for f in glob.glob(f"{LOGS}/wd{wd}_replay_d12_w768_{jid}_1.out"):
            for line in open(f):
                mm = ENS_LINE.search(line)
                if mm:
                    E = int(mm.group(1)); perE.setdefault(E, []).append(float(mm.group(2)))
        for E, vals in perE.items():
            out.setdefault(wd, {})[E] = float(np.min(vals))
    return out


def assemble_clean():
    """rows (w, L, E, Lstar) for the clean wd=0 grid."""
    rows = []
    seen = set()
    for (w, L) in WIDTH_CELLS + DEPTH_CELLS:
        if (w, L) in seen:
            continue
        seen.add((w, L))
        ls = lstar_size(w, L)
        if ls is not None:
            rows.append((w, L, 1, ls))
    ens = ensemble_curves()
    for E in sorted(ens):
        rows.append((W0, L0, E, float(np.min(ens[E][1]))))
    return rows


# ---- df=0.2 (20M) E x size grid: pins exponents + tests separability ----
DF02_CELLS = [(1152, 6), (1152, 12), (384, 18), (768, 18),
              (1152, 18), (1536, 18), (1152, 24)]


def df02_points():
    """rows (w, L, E, Lstar) for the df=0.2 extension grid, init_shuffle."""
    rows = []
    for (w, L) in DF02_CELLS:
        # E=1: mean over the 5 init_shuffle individuals (tasks 5..9)
        curves = []
        for task in range(5, 10):
            d = {}
            for f in glob.glob(f"{LOGS}/df02ext_train_d{L}_w{w}_*_{task}.out"):
                for line in open(f):
                    mm = IND_LINE.search(line)
                    if mm:
                        d[int(mm.group(1))] = float(mm.group(2))
            if d:
                s = np.array(sorted(d)); curves.append((s, np.array([d[k] for k in s])))
        g, v = mean_indiv_curve(curves)
        if v is not None:
            rows.append((w, L, 1, float(np.min(v))))
        # E=2..5: min ens_val_loss from init_shuffle replay tasks 4..7
        for E, task in zip((2, 3, 4, 5), (4, 5, 6, 7)):
            vals = []
            for f in glob.glob(f"{LOGS}/df02ext_replay_d{L}_w{w}_*_{task}.out"):
                for line in open(f):
                    m = re.search(r"ens_val_loss=([\d.]+)", line)
                    if m:
                        vals.append(float(m.group(1)))
            if vals:
                rows.append((w, L, E, float(np.min(vals))))
    return rows


def fit_grid(rows, p0, tag):
    W = np.array([r[0] for r in rows], float)
    Ld = np.array([r[1] for r in rows], float)
    Ee = np.array([r[2] for r in rows], float)
    Y = np.array([r[3] for r in rows], float)
    lb = [1.5, 0.0, 0.05, 0.5, 0.0, 0.1]
    ub = [6.0, 6.0, 3.0, 4.0, 3.0, 3.0]
    sol = least_squares(lambda p: model(p, W, Ld, Ee) - Y, p0,
                        bounds=(lb, ub), method="trf", max_nfev=40000)
    pred = model(sol.x, W, Ld, Ee)
    r2 = 1 - np.sum((pred - Y) ** 2) / np.sum((Y - Y.mean()) ** 2)
    rmse = float(np.sqrt(np.mean((pred - Y) ** 2)))
    Linf, A, alpha, kappa, B, beta = sol.x
    print(f"\n=== FIT [{tag}]  ({len(rows)} pts) ===")
    print(f"  Linf={Linf:.4f}  A={A:.4f}  alpha={alpha:.4f}  kappa={kappa:.4f}  "
          f"B={B:.4f}  beta={beta:.4f}")
    print(f"  R^2={r2:.4f}  RMSE={rmse:.4f}  max|resid|={np.max(np.abs(pred-Y)):.4f}")
    return sol.x, r2


def cap(w, L, kappa):
    return ((w / W0) ** 2 * (L / L0) ** kappa)


def model(p, w, L, E):
    Linf, A, alpha, kappa, B, beta = p
    return Linf + A * cap(w, L, kappa) ** (-alpha) + B * E ** (-beta)


def main():
    rows = assemble_clean()
    print("=== CLEAN grid points (wd=0, no-warmdown) ===")
    print(f"{'w':>5}{'L':>4}{'E':>3}{'compute':>9}{'L*':>9}")
    for w, L, E, ls in rows:
        C = (w / W0) ** 2 * (L / L0) * E
        print(f"{w:>5}{L:>4}{E:>3}{C:>9.2f}{ls:>9.4f}")

    W = np.array([r[0] for r in rows], float)
    Ld = np.array([r[1] for r in rows], float)
    Ee = np.array([r[2] for r in rows], float)
    Y = np.array([r[3] for r in rows], float)

    def resid(p):
        return model(p, W, Ld, Ee) - Y

    p0 = [3.3, 0.25, 0.5, 1.5, 0.35, 0.7]
    lb = [2.0, 0.0, 0.05, 0.5, 0.0, 0.1]
    ub = [3.9, 3.0, 3.0, 4.0, 3.0, 3.0]
    sol = least_squares(resid, p0, bounds=(lb, ub), method="trf", max_nfev=20000)
    Linf, A, alpha, kappa, B, beta = sol.x
    pred = model(sol.x, W, Ld, Ee)
    ss_res = float(np.sum((pred - Y) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean((pred - Y) ** 2)))

    print("\n=== FIT (clean grid) ===")
    print(f"  Linf  = {Linf:.4f}   (irreducible loss @ 100M, inf size, inf ensemble)")
    print(f"  A     = {A:.4f}     capacity amplitude")
    print(f"  alpha = {alpha:.4f}     capacity exponent")
    print(f"  kappa = {kappa:.4f}     depth/width asymmetry (>1 = depth wins per-param)")
    print(f"  B     = {B:.4f}     ensemble amplitude")
    print(f"  beta  = {beta:.4f}     ensemble exponent")
    print(f"  R^2   = {r2:.4f}   RMSE = {rmse:.4f}   max|resid| = {np.max(np.abs(pred-Y)):.4f}")
    print(f"  capacity floor (base, E->inf) Linf+A = {Linf+A:.4f}")

    # ---- df=0.2 grid: pin exponents + separability test ----
    r02 = df02_points()
    print("\n=== df=0.2 (20M) E x size grid points ===")
    for w, L, E, ls in sorted(r02):
        print(f"  w{w:>5} L{L:>3} E{E} : {ls:.4f}")
    p02, r2_02 = fit_grid(r02, [3.5, 1.0, 0.3, 1.5, 0.5, 0.9], "df=0.2")
    print(f"  NOTE: at 20M, A={p02[1]:.3f} (capacity benefit ~0) -> more size HURTS "
          f"(overfit). Cannot transfer capacity exponents across data budget.")
    # ensemble-drop per cell = the separability evidence (size-independent E gain?)
    from collections import defaultdict
    bycell = defaultdict(dict)
    for w, L, E, ls in r02:
        bycell[(w, L)][E] = ls
    print("  ensemble drop E1->E5 per df=0.2 cell (tests separability of E):")
    drops = []
    for (w, L), dd in sorted(bycell.items()):
        if 1 in dd and 5 in dd:
            dr = dd[1] - dd[5]; drops.append(dr)
            print(f"    w{w:>5} L{L:>3}: {dd[1]:.3f} -> {dd[5]:.3f}  drop={dr:.3f}")
    print(f"    => drop mean={np.mean(drops):.3f} std={np.std(drops):.3f}  "
          f"({'roughly size-independent -> ensembling SEPARABLE' if np.std(drops)<0.03 else 'size-dependent'})")

    # prediction uses the df=1.0 IN-SAMPLE fit (weak but positive alpha); flag caveat
    xstar = sol.x
    print("\n  [prediction uses df=1.0 in-sample fit; alpha weakly identified -> "
          "trust the CONSTRAINED optimum below, not sub-base widths]")

    # ---- wd recipe offset ----
    wd = wd_sweep()
    clean_base = {E: model(sol.x, W0, L0, E) for E in range(1, 6)}
    print("\n=== COOLDOWN wd sweep @ base (L* and delta vs clean base same-E) ===")
    print(f"{'wd':>6} " + "".join(f"E{E:>7}" for E in range(1, 6)) + "   mean_delta")
    deltas = {}
    for w_ in sorted(wd, key=float):
        row = wd[w_]
        ds = []
        cells = []
        for E in range(1, 6):
            if E in row:
                d = row[E] - clean_base[E]
                ds.append(d); cells.append(f"{row[E]:.3f}")
            else:
                cells.append("  -  ")
        md = np.mean(ds) if ds else np.nan
        deltas[float(w_)] = md
        print(f"{w_:>6} " + "".join(f"{c:>8}" for c in cells) + f"   {md:+.4f}")
    best_wd = min(deltas, key=lambda k: deltas[k])
    print(f"\n  best recipe: wd={best_wd}  delta={deltas[best_wd]:+.4f}  "
          f"(=> subtract ~{-deltas[best_wd]:.3f} from any clean prediction)")

    # ---- compute-optimal frontier (transferred-exponent law + best wd) ----
    print("\n=== COMPUTE-OPTIMAL config vs budget (transferred exponents + wd=0.1) ===")
    ws = [384, 512, 768, 1024, 1152, 1536, 2048]
    Ls = [6, 8, 12, 16, 18, 24, 32, 48, 60]
    Es = list(range(1, 17))
    dwd = deltas[best_wd]
    print(f"{'budget':>7}  {'w':>5}{'L':>4}{'E':>3}  {'pred L*':>8}  note")
    for budget in [1, 2, 4, 8, 16, 32]:
        best = None
        for w in ws:
            for L in Ls:
                for E in Es:
                    C = (w / W0) ** 2 * (L / L0) * E
                    if C > budget * 1.03:
                        continue
                    pl = model(xstar, w, L, E) + dwd
                    if best is None or pl < best[0]:
                        best = (pl, w, L, E, C)
        pl, w, L, E, C = best
        print(f"{budget:>6}x  {w:>5}{L:>4}{E:>3}  {pl:>8.4f}  (uses {C:.1f}x)")

    # sensitivity: best config if we FORBID shrinking below base width/depth
    print("\n=== COMPUTE-OPTIMAL, constrained w>=768 & L>=12 (no sub-base models) ===")
    for budget in [1, 2, 4, 8, 16, 32]:
        best = None
        for w in [768, 1024, 1152, 1536, 2048]:
            for L in [12, 16, 18, 24, 32, 48, 60]:
                for E in Es:
                    C = (w / W0) ** 2 * (L / L0) * E
                    if C > budget * 1.03:
                        continue
                    pl = model(xstar, w, L, E) + dwd
                    if best is None or pl < best[0]:
                        best = (pl, w, L, E, C)
        pl, w, L, E, C = best
        print(f"{budget:>6}x  {w:>5}{L:>4}{E:>3}  {pl:>8.4f}  (uses {C:.1f}x)")

    return xstar, deltas[best_wd]


if __name__ == "__main__":
    main()
