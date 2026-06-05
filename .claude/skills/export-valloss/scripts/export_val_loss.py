"""
Export per-model (individual) and ensemble validation-loss curves from SLURM
.out logs into tidy, durable data files (CSV + npz).

WHY logs and not wandb: training ran WANDB_MODE=offline, and every
NODE_FAIL/TIMEOUT/resume spawned a *separate* offline-run dir with the same
display name, so each model's full curve is fragmented across truncated offline
runs. The `.out` logs, by contrast, are complete and uniform — so individuals
are parsed from the training logs, ensembles from the replay logs.

Data sources (per cell d{depth}/w{width}, per strategy):
  - Individuals: training logs `fd_train_d{d}_w{w}_{job}_{arr}.out`
      line `[model 1 val @ step S] val_loss=L`   (tokens = S * total_batch_size)
      arr//num_models = strategy index, arr%num_models = model index.
      (Single-model training always logs under namespace `model_1/`; the model
       identity comes from the array index, not the namespace.)
  - Ensembles: replay logs, preferring the 20M/per-step logs
      `fd_replay20m_d{d}_w{w}_*.out`  line
      `[step S ens=E] val_loss=L val_bpb=B tokens=T`  (tokens read directly),
      falling back to per-epoch logs `fd_replay_d{d}_w{w}_*.out` line
      `[epoch K ens=E] val_loss=L`  (tokens = K * tokens_per_epoch).
      Strategy is disambiguated by the `strategy=` banner inside each file.
      Logs are MERGED by step/token, so multiple replay runs (e.g. a base run
      + an extension run covering later steps) combine into one curve.

Outputs (under --out-dir):
  - val_loss_individuals.csv  strategy, depth, width, params_m, model, step, tokens_seen, val_loss
  - val_loss_ensembles.csv    strategy, depth, width, params_m, E, step, tokens_seen, val_loss, val_bpb
  - val_loss_long.csv         unified long format (kind, series) for plotting/fitting
  - val_loss_grid.npz         same data as nested numpy arrays for fast reload

Usage:
  python export_val_loss.py                       # defaults: the df=1.0 2x2 grid
  python export_val_loss.py --cells 6x384,12x768 --strategies init_ens,init_shuffle_ens
  python export_val_loss.py --logs-dir /path/to/logs --out-dir /path/to/data
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np

# Repo root = four levels up from this script (.claude/skills/export-valloss/scripts/).
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))

STEP_VAL_RE = re.compile(r"\[model \d+ val @ step (\d+)\] val_loss=([0-9.]+)")
ENS_STEP_RE = re.compile(r"\[step (\d+) ens=(\d+)\] val_loss=([0-9.]+) val_bpb=([0-9.]+) tokens=(\d+)")
ENS_EPOCH_RE = re.compile(r"\[epoch (\d+) ens=(\d+)\] val_loss=([0-9.]+)(?: val_bpb=([0-9.]+))?")


def nonembed_params_m(d, w):
    return 16 * d * w * w / 1e6


def parse_cells(s):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        d, w = tok.lower().split("x")
        out.append((int(d), int(w)))
    return out


def load_individuals(logs, d, w, strat_idx, num_models, total_batch_size):
    """{model_idx: {step: val_loss}} merged across resume logs."""
    by_idx = defaultdict(dict)
    fre = re.compile(rf"fd_train_d{d}_w{w}_(\d+)_(\d+)\.out")
    for f in glob.glob(os.path.join(logs, f"fd_train_d{d}_w{w}_*.out")):
        m = fre.search(os.path.basename(f))
        if not m or int(m.group(2)) // num_models != strat_idx:
            continue
        model = int(m.group(2)) % num_models
        with open(f, errors="ignore") as fh:
            for line in fh:
                mm = STEP_VAL_RE.search(line)
                if mm:
                    by_idx[model][int(mm.group(1))] = float(mm.group(2))
    return by_idx


def load_ensembles(logs, d, w, strat_name, tokens_per_epoch):
    """{E: {step_or_epoch: (tokens, val_loss, val_bpb)}}. Prefer 20M step logs."""
    banner = f"strategy={strat_name}"

    def scan(pattern, line_re, step_mode):
        by_E = defaultdict(dict)
        for f in glob.glob(os.path.join(logs, pattern)):
            txt = open(f, errors="ignore").read()
            if banner not in txt:
                continue
            for line in txt.splitlines():
                mm = line_re.search(line)
                if not mm:
                    continue
                if step_mode:
                    S, E = int(mm.group(1)), int(mm.group(2))
                    val, bpb, tok = float(mm.group(3)), float(mm.group(4)), int(mm.group(5))
                    by_E[E][S] = (tok, val, bpb)
                else:
                    ep, E, val = int(mm.group(1)), int(mm.group(2)), float(mm.group(3))
                    bpb = float(mm.group(4)) if mm.group(4) else float("nan")
                    by_E[E][ep] = (ep * tokens_per_epoch, val, bpb)
        return by_E

    by_E = scan(f"fd_replay20m_d{d}_w{w}_*.out", ENS_STEP_RE, True)
    if not by_E:
        by_E = scan(f"fd_replay_d{d}_w{w}_*.out", ENS_EPOCH_RE, False)
    return by_E


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", default=os.path.join(REPO, "experiments", "logs"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "experiments", "analysis", "data"))
    ap.add_argument("--cells", default="6x384,12x384,6x768,12x768",
                    help="comma-separated depthXwidth, e.g. 6x384,12x768")
    ap.add_argument("--strategies", default="init_ens,init_shuffle_ens",
                    help="comma-separated; order defines strategy index (arr//num_models)")
    ap.add_argument("--num-models", type=int, default=5)
    ap.add_argument("--total-batch-size", type=int, default=131072)
    ap.add_argument("--steps-per-epoch", type=int, default=763,
                    help="optimizer steps per epoch (for per-epoch ensemble fallback)")
    args = ap.parse_args()

    cells = parse_cells(args.cells)
    strats = [s.strip() for s in args.strategies.split(",") if s.strip()]
    tokens_per_epoch = args.steps_per_epoch * args.total_batch_size
    os.makedirs(args.out_dir, exist_ok=True)

    ind_rows, ens_rows, npz = [], [], {}

    for strat_idx, strat in enumerate(strats):
        for (d, w) in cells:
            pm = round(nonembed_params_m(d, w), 2)

            inds = load_individuals(args.logs_dir, d, w, strat_idx,
                                    args.num_models, args.total_batch_size)
            for model in sorted(inds):
                steps = np.array(sorted(inds[model]))
                for step in steps:
                    ind_rows.append([strat, d, w, pm, model, int(step),
                                     int(step) * args.total_batch_size, inds[model][step]])
                npz[f"ind/{strat}/d{d}_w{w}/model{model}/tokens"] = steps * args.total_batch_size
                npz[f"ind/{strat}/d{d}_w{w}/model{model}/val_loss"] = np.array(
                    [inds[model][s] for s in steps])

            enss = load_ensembles(args.logs_dir, d, w, strat, tokens_per_epoch)
            for E in sorted(enss):
                keys = sorted(enss[E])
                for k in keys:
                    tok, val, bpb = enss[E][k]
                    ens_rows.append([strat, d, w, pm, E, k, tok, val, bpb])
                arr = np.array([enss[E][k] for k in keys])  # (n,3): tok,val,bpb
                npz[f"ens/{strat}/d{d}_w{w}/E{E}/tokens"] = arr[:, 0]
                npz[f"ens/{strat}/d{d}_w{w}/E{E}/val_loss"] = arr[:, 1]
                npz[f"ens/{strat}/d{d}_w{w}/E{E}/val_bpb"] = arr[:, 2]

    ind_path = os.path.join(args.out_dir, "val_loss_individuals.csv")
    with open(ind_path, "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["strategy", "depth", "width", "params_m", "model",
                     "step", "tokens_seen", "val_loss"])
        w_.writerows(ind_rows)

    ens_path = os.path.join(args.out_dir, "val_loss_ensembles.csv")
    with open(ens_path, "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["strategy", "depth", "width", "params_m", "E",
                     "step", "tokens_seen", "val_loss", "val_bpb"])
        w_.writerows(ens_rows)

    long_path = os.path.join(args.out_dir, "val_loss_long.csv")
    with open(long_path, "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["strategy", "depth", "width", "params_m", "kind",
                     "series", "step", "tokens_seen", "val_loss", "val_bpb"])
        for r in ind_rows:   # strat,d,w,pm,model,step,tok,val
            w_.writerow([r[0], r[1], r[2], r[3], "individual", r[4], r[5], r[6], r[7], ""])
        for r in ens_rows:   # strat,d,w,pm,E,step,tok,val,bpb
            w_.writerow([r[0], r[1], r[2], r[3], "ensemble", r[4], r[5], r[6], r[7], r[8]])

    npz_path = os.path.join(args.out_dir, "val_loss_grid.npz")
    np.savez_compressed(npz_path, **npz)

    print(f"wrote {ind_path}  ({len(ind_rows)} rows)")
    print(f"wrote {ens_path}  ({len(ens_rows)} rows)")
    print(f"wrote {long_path}  ({len(ind_rows) + len(ens_rows)} rows)")
    print(f"wrote {npz_path}  ({len(npz)} arrays)")
    if not ind_rows and not ens_rows:
        print("WARNING: no rows parsed — check --logs-dir, --cells, and --strategies.")


if __name__ == "__main__":
    main()
