"""Assemble data from Expt 1, Expt 2, Expt 3, and the data-size sweeps (Q1 v1 + Q1 v2)
into a single ``data_export/`` directory for offline analysis.

Schema:
  data_export/
    manifest.csv              -- one row per saved file
    schema.md                 -- column docs
    expt1_overfit/            -- U-curve / overfit-demo data
      q1_v2/                  -- 4 dfs × 2 strats, wd=0, ~1B tokens (fixed-tokens budget)
      q1_v1/                  -- 4 dfs × 2 strats, wd=0.1, 50 epochs (fixed-epochs)
      q2_individuals/         -- 20 inds × 2 strats at d12/w768/df0.2 (40 epochs, from train logs)
      df02_grid_picks/        -- d12/w1536 and d24/w1536 individuals (used in fig1)
    expt2_ensemble/
      individuals/            -- same as expt1/q2_individuals (symlink? just re-save)
      ensembles/              -- 5 sizes × 2 strats, fused replay 40 ep
      bootstrap/              -- placeholder for permutation-bootstrap (filled later when job finishes)
    expt3_grid/
      individuals/            -- 16 cells × 2 strats × 5 inds, 25 ep each
      ensembles/              -- 16 cells × 2 strats × 4 sizes (E ∈ {2,3,4,5}), 25 ep each
    train_base_dataframe/     -- "train base under varying datasize" = Q1 v1 + Q1 v2 (alias of expt1)

Each NPZ file contains arrays:
  tokens   : int64,  cumulative tokens per model
  val_loss : float64
  epoch    : int32  (when available)
  meta     : dict of run metadata (cell, strategy, source, etc.)

The manifest.csv has columns:
  path, experiment, group, cell_d, cell_w, df, strategy, kind (indiv|ens), index_or_E, n_points, source
"""
import os
import re
import csv
import shutil
import numpy as np
import wandb

ENT = "xjtumyd-carnegie-mellon-university/slowrun"
OUTDIR = "data_export"
EPOCH_VAL_RE = re.compile(r"\[model \d+\]\s+epoch\s+(\d+)\s+val_loss=([0-9.]+)")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)


def parse_log(path):
    d = {}
    if not os.path.isfile(path):
        return d
    with open(path) as f:
        for line in f:
            m = EPOCH_VAL_RE.search(line)
            if m:
                d[int(m.group(1))] = float(m.group(2))
    return d


def fetch_indiv_curve(api, group, name):
    """Return tokens, val_loss, epoch arrays for an individual training run.
    Merges across resume-runs by tokens_seen."""
    runs = list(api.runs(ENT, filters={"group": group, "display_name": name}))
    runs.sort(key=lambda r: r.created_at)
    by_tok = {}
    for r in runs:
        # Try each model_k namespace in turn (different launchers index differently).
        used_k = None
        for k in range(1, 30):
            n = sum(1 for h in r.scan_history(keys=[f"model_{k}/val_loss"])
                    if h.get(f"model_{k}/val_loss") is not None)
            if n > 0:
                used_k = k
                break
        if used_k is None:
            continue
        for h in r.scan_history(keys=[f"model_{used_k}/val_loss",
                                       f"model_{used_k}/tokens_seen",
                                       f"model_{used_k}/epoch"]):
            vl = h.get(f"model_{used_k}/val_loss")
            ts = h.get(f"model_{used_k}/tokens_seen")
            ep = h.get(f"model_{used_k}/epoch")
            if vl is not None and ts is not None:
                by_tok[int(ts)] = (float(vl), int(ep) if ep is not None else -1)
    if not by_tok:
        return np.array([]), np.array([]), np.array([])
    toks = np.array(sorted(by_tok), dtype=np.int64)
    vls = np.array([by_tok[t][0] for t in toks], dtype=np.float64)
    eps = np.array([by_tok[t][1] for t in toks], dtype=np.int32)
    return toks, vls, eps


def fetch_ens_curve(api, group, name):
    runs = list(api.runs(ENT, filters={"group": group, "display_name": name}))
    runs.sort(key=lambda r: r.created_at, reverse=True)
    if not runs:
        return np.array([]), np.array([]), np.array([])
    r = runs[0]
    by_tok = {}
    for h in r.scan_history(keys=["ens/tokens_seen", "ens/val_loss", "ens/epoch"]):
        vl = h.get("ens/val_loss")
        ts = h.get("ens/tokens_seen")
        ep = h.get("ens/epoch")
        if vl is not None and ts is not None:
            by_tok[int(ts)] = (float(vl), int(ep) if ep is not None else -1)
    if not by_tok:
        return np.array([]), np.array([]), np.array([])
    toks = np.array(sorted(by_tok), dtype=np.int64)
    vls = np.array([by_tok[t][0] for t in toks], dtype=np.float64)
    eps = np.array([by_tok[t][1] for t in toks], dtype=np.int32)
    return toks, vls, eps


def save_npz(path, tokens, val_loss, epoch, **meta):
    """Save with a tokens/val_loss/epoch payload + arbitrary metadata fields."""
    np.savez(path, tokens=tokens, val_loss=val_loss, epoch=epoch, **meta)


# ------------------------------------------------------------------
# Manifest writer
# ------------------------------------------------------------------

class Manifest:
    def __init__(self, csvpath):
        self.csvpath = csvpath
        self.rows = []
        self.header = ["path", "experiment", "group", "cell_d", "cell_w", "df",
                       "strategy", "kind", "index_or_E", "n_points", "source"]

    def add(self, path, **kw):
        rel = os.path.relpath(path, OUTDIR)
        row = {"path": rel}
        row.update({k: kw.get(k, "") for k in self.header[1:]})
        self.rows.append(row)

    def flush(self):
        with open(self.csvpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.header)
            w.writeheader()
            w.writerows(self.rows)


# ------------------------------------------------------------------
# Per-experiment exporters
# ------------------------------------------------------------------

def export_q1_v2(api, manifest):
    """Q1 v2: 4 dfs × 2 strats × 1 ind, wd=0, ~1B tokens budget."""
    out = os.path.join(OUTDIR, "expt1_overfit", "q1_v2")
    safe_mkdir(out)
    tag = "q1v2_20260503_115259"
    for df in ["0.2", "0.3", "0.4", "0.5"]:
        group = f"q1_data_size_{tag}_d12_w768_df{df}"
        for strat in ["init_ens", "init_shuffle_ens"]:
            name = f"{group}_{strat}_model0"
            toks, vls, eps = fetch_indiv_curve(api, group, name)
            if len(toks) == 0:
                print(f"  MISSING: {name}"); continue
            p = os.path.join(out, f"df{df}_{strat}.npz")
            save_npz(p, toks, vls, eps,
                     group=group, df=df, strat=strat, kind="individual",
                     wd=0.0, lr_schedule="constant", model_idx=0)
            manifest.add(p, experiment="expt1_q1v2", group=group,
                         cell_d=12, cell_w=768, df=df, strategy=strat,
                         kind="indiv", index_or_E=0, n_points=len(toks),
                         source="wandb")
            print(f"  saved {p}: {len(toks)} pts")


def export_q1_v1(api, manifest):
    """Q1 v1: 4 dfs × 2 strats × 1 ind, wd=0.1, 50 epochs."""
    out = os.path.join(OUTDIR, "expt1_overfit", "q1_v1")
    safe_mkdir(out)
    tag = "q1_20260502_234110"
    for df in ["0.2", "0.4", "0.6", "0.8"]:
        group = f"q1_data_size_{tag}_d12_w768_df{df}"
        for strat in ["init_ens", "init_shuffle_ens"]:
            name = f"{group}_{strat}_model0"
            toks, vls, eps = fetch_indiv_curve(api, group, name)
            if len(toks) == 0:
                print(f"  MISSING: {name}"); continue
            p = os.path.join(out, f"df{df}_{strat}.npz")
            save_npz(p, toks, vls, eps,
                     group=group, df=df, strat=strat, kind="individual",
                     wd=0.1, lr_schedule="constant", model_idx=0)
            manifest.add(p, experiment="expt1_q1v1", group=group,
                         cell_d=12, cell_w=768, df=df, strategy=strat,
                         kind="indiv", index_or_E=0, n_points=len(toks),
                         source="wandb")
            print(f"  saved {p}: {len(toks)} pts")


def export_q2_individuals(manifest):
    """Q2 ext individuals (20 × 2 strats), parsed from TRAIN LOGS (the wandb
    cloud version has epochs 1-15 missing due to the offline-sync overwrite issue)."""
    out_e1 = os.path.join(OUTDIR, "expt1_overfit", "q2_individuals")
    out_e2 = os.path.join(OUTDIR, "expt2_ensemble", "individuals")
    safe_mkdir(out_e1)
    safe_mkdir(out_e2)
    TOKENS_PER_EPOCH = int(0.2 * 99942400)
    for strat_idx, strat in enumerate(["init_ens", "init_shuffle_ens"]):
        for m in range(20):
            task = strat_idx * 20 + m
            d = {**parse_log(f"experiments/logs/q2_train_d12_w768_df0.2_40532724_{task}.out"),
                 **parse_log(f"experiments/logs/q2ext_train_40583426_{task}.out")}
            if not d:
                print(f"  {strat} model_{m}: NO DATA"); continue
            eps = np.array(sorted(d), dtype=np.int32)
            toks = eps.astype(np.int64) * TOKENS_PER_EPOCH
            vls = np.array([d[int(e)] for e in eps], dtype=np.float64)
            for out in (out_e1, out_e2):
                p = os.path.join(out, f"{strat}_model{m}.npz")
                save_npz(p, toks, vls, eps,
                         strat=strat, model_idx=m, kind="individual",
                         d=12, w=768, df=0.2, wd=0.1, source="train_logs")
            manifest.add(os.path.join(out_e1, f"{strat}_model{m}.npz"),
                         experiment="expt1_q2_individuals",
                         group="q2_ensemble_size_q2_20260502_234110_d12_w768_df0.2",
                         cell_d=12, cell_w=768, df=0.2, strategy=strat,
                         kind="indiv", index_or_E=m, n_points=len(toks),
                         source="train_logs")
            manifest.add(os.path.join(out_e2, f"{strat}_model{m}.npz"),
                         experiment="expt2_individuals",
                         group="q2_ensemble_size_q2_20260502_234110_d12_w768_df0.2",
                         cell_d=12, cell_w=768, df=0.2, strategy=strat,
                         kind="indiv", index_or_E=m, n_points=len(toks),
                         source="train_logs")
            print(f"  saved {strat}_model{m}.npz: {len(eps)} eps")


def export_df02_grid_picks(api, manifest):
    """For Expt 1 fig1: d12/w1536 and d24/w1536 init_shuffle model 0 from original df=0.2 grid."""
    out = os.path.join(OUTDIR, "expt1_overfit", "df02_grid_picks")
    safe_mkdir(out)
    for d, w in [(12, 1536), (24, 1536)]:
        group = f"grid_20260430_152533_d{d}_w{w}_df0.2"
        for m in [0]:
            name = f"{group}_init_shuffle_ens_model{m}"
            toks, vls, eps = fetch_indiv_curve(api, group, name)
            if len(toks) == 0:
                print(f"  MISSING: {name}"); continue
            p = os.path.join(out, f"d{d}_w{w}_init_shuffle_model{m}.npz")
            save_npz(p, toks, vls, eps,
                     group=group, d=d, w=w, df=0.2, wd=0.1,
                     strat="init_shuffle_ens", model_idx=m, kind="individual",
                     lr_schedule="trapezoidal_warmdown")
            manifest.add(p, experiment="expt1_df02_grid_picks", group=group,
                         cell_d=d, cell_w=w, df=0.2, strategy="init_shuffle_ens",
                         kind="indiv", index_or_E=m, n_points=len(toks), source="wandb")
            print(f"  saved {p}: {len(toks)} pts")


def export_q2_ensembles(api, manifest):
    """Q2 ext fused replay: 5 sizes × 2 strats × 40 ep at d12/w768/df0.2."""
    out = os.path.join(OUTDIR, "expt2_ensemble", "ensembles")
    safe_mkdir(out)
    group = "q2_ensemble_size_q2_20260502_234110_d12_w768_df0.2"
    for strat in ["init_ens", "init_shuffle_ens"]:
        for E in [2, 5, 10, 15, 20]:
            name = f"{group}_{strat}_ens{E}_replay"
            toks, vls, eps = fetch_ens_curve(api, group, name)
            if len(toks) == 0:
                print(f"  MISSING: {name}"); continue
            p = os.path.join(out, f"{strat}_E{E}.npz")
            save_npz(p, toks, vls, eps,
                     group=group, strat=strat, E=E,
                     d=12, w=768, df=0.2, wd=0.1, kind="ensemble")
            manifest.add(p, experiment="expt2_ensembles", group=group,
                         cell_d=12, cell_w=768, df=0.2, strategy=strat,
                         kind="ens", index_or_E=E, n_points=len(toks), source="wandb")
            print(f"  saved {p}: {len(toks)} pts")


def export_grid_4x4(api, manifest):
    """16 cells × 2 strats × 5 inds + 4 ens sizes at df=0.2."""
    out_indiv = os.path.join(OUTDIR, "expt3_grid", "individuals")
    out_ens   = os.path.join(OUTDIR, "expt3_grid", "ensembles")
    safe_mkdir(out_indiv); safe_mkdir(out_ens)

    DEPTHS = [6, 12, 18, 24]
    WIDTHS = [384, 768, 1152, 1536]
    TAG_BASE = "20260430_152533"
    TAG_EXT = "df02ext_20260504_224131"

    def group_for(d, w):
        if d == 18 or w == 1152:
            return f"grid_{TAG_EXT}_d{d}_w{w}_df0.2"
        return f"grid_{TAG_BASE}_d{d}_w{w}_df0.2"

    for d in DEPTHS:
        for w in WIDTHS:
            group = group_for(d, w)
            for strat in ["init_ens", "init_shuffle_ens"]:
                for m in range(5):
                    name = f"{group}_{strat}_model{m}"
                    toks, vls, eps = fetch_indiv_curve(api, group, name)
                    if len(toks) == 0:
                        print(f"  MISSING: {name}"); continue
                    p = os.path.join(out_indiv, f"d{d}_w{w}_{strat}_model{m}.npz")
                    save_npz(p, toks, vls, eps,
                             group=group, d=d, w=w, df=0.2, wd=0.1,
                             strat=strat, model_idx=m, kind="individual",
                             lr_schedule=("trapezoidal_warmdown" if d != 18 and w != 1152
                                          else "trapezoidal_warmdown"))
                    manifest.add(p, experiment="expt3_grid_indiv", group=group,
                                 cell_d=d, cell_w=w, df=0.2, strategy=strat,
                                 kind="indiv", index_or_E=m, n_points=len(toks), source="wandb")
                for E in [2, 3, 4, 5]:
                    name = f"{group}_{strat}_ens{E}_replay"
                    toks, vls, eps = fetch_ens_curve(api, group, name)
                    if len(toks) == 0:
                        print(f"  MISSING: {name}"); continue
                    p = os.path.join(out_ens, f"d{d}_w{w}_{strat}_E{E}.npz")
                    save_npz(p, toks, vls, eps,
                             group=group, d=d, w=w, df=0.2, wd=0.1,
                             strat=strat, E=E, kind="ensemble")
                    manifest.add(p, experiment="expt3_grid_ens", group=group,
                                 cell_d=d, cell_w=w, df=0.2, strategy=strat,
                                 kind="ens", index_or_E=E, n_points=len(toks), source="wandb")
            print(f"  d{d}/w{w}: done")


def write_schema(outpath):
    schema = """\
# data_export schema

Each `*.npz` file contains:
- `tokens`   : int64 array, cumulative tokens-seen per model (per-model x-axis)
- `val_loss` : float64 array
- `epoch`    : int32 array (per-epoch index; -1 if not logged)
- arbitrary metadata fields per file (group, d, w, df, wd, strat, kind, ...)

To read a file in Python:

```python
import numpy as np
data = np.load("expt2_ensemble/ensembles/init_ens_E5.npz", allow_pickle=True)
tokens, val_loss = data["tokens"], data["val_loss"]
print({k: data[k] for k in data.files if k not in ("tokens", "val_loss", "epoch")})
```

## Directories

- `expt1_overfit/` — data for L(t) U-curve figures (Expt Fig 1)
  - `q1_v2/`             dfs {0.2, 0.3, 0.4, 0.5}, wd=0, fixed-tokens budget (~1B tokens), 1 ind/strategy
  - `q1_v1/`             dfs {0.2, 0.4, 0.6, 0.8}, wd=0.1, fixed-epochs (50), 1 ind/strategy
  - `q2_individuals/`    20 inds × 2 strats at d12/w768/df0.2, 40 epochs (from train logs)
  - `df02_grid_picks/`   d12/w1536 and d24/w1536 init_shuffle model 0 (used in fig1 overlay)

- `expt2_ensemble/` — data for ensemble scaling figures (Expt Fig 2)
  - `individuals/`  same 20 individuals as expt1/q2_individuals (re-saved for convenience)
  - `ensembles/`    fused-replay ensemble val curves; 5 sizes × 2 strats at d12/w768/df0.2
  - `bootstrap/`    permutation-bootstrap L grids (job 40614133, fills in when complete)

- `expt3_grid/` — 4×4 (depth × width) grid at df=0.2 (Expt Fig 3)
  - `individuals/`  16 cells × 2 strats × 5 inds (= 160 files)
  - `ensembles/`    16 cells × 2 strats × 4 sizes (E ∈ {2,3,4,5}) (= 128 files)

## manifest.csv

Top-level index of every NPZ file with searchable metadata columns.

## Sources

- `wandb`        : pulled from W&B cloud
- `train_logs`   : parsed from local SLURM `.out` files (used when wandb data was
                   incomplete due to the offline-sync overwrite incident on 2026-05-04)
"""
    with open(outpath, "w") as f:
        f.write(schema)


def main():
    safe_mkdir(OUTDIR)
    api = wandb.Api()
    manifest = Manifest(os.path.join(OUTDIR, "manifest.csv"))

    print("=== Expt 1 overfit (Q1 v2) ===")
    export_q1_v2(api, manifest)
    print("\n=== Expt 1 overfit (Q1 v1) ===")
    export_q1_v1(api, manifest)
    print("\n=== Q2 individuals (from train logs) ===")
    export_q2_individuals(manifest)
    print("\n=== Expt 1 grid picks (df=0.2 grid, d12/w1536 + d24/w1536) ===")
    export_df02_grid_picks(api, manifest)
    print("\n=== Expt 2 ensembles (Q2 ext fused replay) ===")
    export_q2_ensembles(api, manifest)
    print("\n=== Expt 3 grid (4×4) ===")
    export_grid_4x4(api, manifest)

    manifest.flush()
    write_schema(os.path.join(OUTDIR, "schema.md"))

    # Summary
    print(f"\n=== DONE ===")
    print(f"manifest: {os.path.join(OUTDIR, 'manifest.csv')}")
    print(f"total files: {len(manifest.rows)}")


if __name__ == "__main__":
    main()
