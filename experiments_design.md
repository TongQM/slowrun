# Experiments Design

This document is the index of every grid sweep we run in this project — what's done, what's in flight, what's planned, and the dimensions each grid covers. For implementation details see [CLAUDE.md](CLAUDE.md) and [experiments/README.md](experiments/README.md).

## Research question

Multi-epoch training dynamics in the **data-limited regime**: with a fixed FineWeb subset and unlimited compute, when does ensembling beat scaling, and which ensembling axis matters most?

Two ensembling axes:
- **`init`** — N models with different initializations, shared per-epoch data permutation π_k
- **`init_shuffle`** — N models with different initializations **and** independent per-epoch permutations π_{i,k}

## Sweep dimensions

Every grid runs over the cartesian product of the dimensions below.

| Dimension | Values | Notes |
|---|---|---|
| **n_layer (depth)** | {6, 12, 24} | head_dim fixed at 64; n_head = n_embd / 64 |
| **n_embd (width)** | {384, 768, 1536} | yields 9 (depth × width) cells per grid |
| **ensemble strategy** | {`init_ens`, `init_shuffle_ens`} | 2 strategies per cell |
| **individuals per cell** | 5 | seeded by `seed * 10000 + epoch` for independence |
| **ensemble size** | {2, 3, 4, 5} | post-hoc replay over saved checkpoints |
| **ensemble averaging** | `logit` (default) | could vary `prob` later |
| **data fraction** | {0.2, 0.4, …} | 20M / 40M / … tokens per epoch from a 100M FineWeb subset |
| **num_epochs** | 25 | extendable via resume |
| **LR schedule** | constant or trapezoidal-warmdown | see per-grid notes |
| **total batch size** | 131072 (128K tokens/step) | "small batch" — finer overfit dynamics resolution |

### Fixed across all grids
- Optimizer: **AdamW** (CompleteP HP-transfer claims target this)
- **CompleteP** ON (`--completep`): width-aware init + forward multipliers + L_base/L depth scaling
- **No value-embedding projections** (`--no-ve-projs`)
- mup_base = (width 768, depth 12, head_dim 64) — d12_w768 is the "1×" reference cell
- GPU: **H100-80**, compile mode: **inductor**, torch 2.9.1+cu128
- Model architecture: GPT with U-Net skips, RoPE, sliding-window attention, SiLU-gated MLP

## Grids

### Grid 1: `df=0.2` (DONE) — `grid_20260430_152533`

- 20M tokens/epoch × 25 epochs = **500M cumulative tokens / model**
- LR schedule: WARMUP=0 + flat (80%) + linear warmdown to 0 (last 20%)
- Val cadence: once per epoch
- Wandb groups: `grid_20260430_152533_d{6,12,24}_w{384,768,1536}_df0.2`
- Compute: ~600 SU on cis260161p
- 90 training tasks + 54 replay tasks (sizes {2, 4, 5}) + ens=3 retrofit (18 replay tasks)

**Findings** (analysis figures in [experiments/analysis/](experiments/analysis/)):
- Overfit visible only on largest cells (d24/*, d12/w1536) at 25 epochs.
- **Width ≠ ensembling at equal compute**: at matched-params (ens=3 of small_w vs ens=1 of √4× wider), ensemble wins by ~0.27–0.30 nats.
- Init vs init_shuffle: tiny gap, init_shuffle slightly favored at large ensemble sizes.
- Ensembling gains: ~0.25 nats from ens=1→5 across all cells; saturating quickly past ens=3.

Outputs: `grid_combo_20260430_152533.png`, `slice_*.png`, `heatmap_best_val_loss.png`, `width_vs_ensemble_d12_with_ens3.png`.

### Grid 2: `df=0.4` (PARTIAL — CANCELLED) — `grid_20260502_013702`

- 40M tokens/epoch × 25 epochs = **1B cumulative tokens / model**
- LR schedule: **constant** (`--no-warmdown`)
- Val cadence: every 20M cumulative tokens (152 optimizer steps)
- Wandb groups: `grid_20260502_013702_d{6,12,24}_w{384,768,1536}_df0.4`

**Status (2026-05-04):**
- **Cells 1–3 (w384 column): DONE** via `launch_grid_v2.sh`, charged to cis260161p (~135 SU). Per-epoch ckpts at every 5th epoch on cis260161p.
- **Cells 4–9 (w768 + w1536 rows): CANCELLED** on 2026-05-04. Why: jobs 40532076–40532099 (24 PD tasks on cis260009p) sat PD-Priority for 36+ hours due to fairshare suppression (we'd drawn 366K SU-equiv from cis260009p, fairshare = 0.245 vs cluster ~0.5). With 5-day project deadline approaching, we redirected the remaining ~1230 SU budget to expanding the df=0.2 grid (which fits cis260161p and runs faster).
- **What survives**: w384-column data is comparable to the equivalent w384 cells in the `df=0.2` grid — gives a 3-cell × 2-df slice for "does doubling unique data shift overfit dynamics at the smallest width?" That figure is still possible.
- **What's lost**: full 9-cell df=0.4 sweep; in particular no df=0.4 readings for w768/w1536 cells. Future work could revisit if SUs become available.

**Hypothesis pre-cancellation:** with wd=0.1 + 25 epochs at higher df, smaller cells (d6/w384, d12/w384) likely don't overfit (Q1 v2 confirmed this in spirit at d12/w768 — see Q1 v1/v2 lessons on the wd-per-epoch confound).

### Grid 1 extension: `df=0.2` interpolation (IN FLIGHT) — `df02ext_20260504_224131`

- Goal: enrich the existing 9-cell df=0.2 grid with **7 new cells** to fit a scaling law over (depth, width). Combined with `grid_20260430_152533`, gives 16 (d, w) points.
- New cells: **d18 row + w1152 column** (factor √2 interpolation between existing axes).
  - d18 row: d18/{w384, w768, w1152, w1536}
  - w1152 col: {d6, d12, d18, d24}/w1152  (d18/w1152 shared with d18 row → 7 unique cells)
- Setup matches `grid_20260430_152533` exactly for direct comparability:
  - Trapezoidal LR (warmup 0 + flat 80% + linear warmdown), wd=0.1, optimizer=adamw
  - CompleteP ON, no-ve-projs ON, mup_base=(768, 12, 64)
  - 5 individuals × 2 strategies × 25 epochs at df=0.2, batch 131K
  - Replay sizes {2, 3, 4, 5}; per-epoch ckpts only; cleanup keeps every-5-epoch
- Compute: ~630 SU on cis260161p. Storage on cis260161p (~2 TB free).
- Wallclock acceleration via **wave parallelism** (vs 3.7 days serial → ~1.5 days):
  - **Wave 1**: 5 cells parallel (sum transient ~900 GB): d18/w384, d6/w1152, d18/w768, d12/w1152, d18/w1152. Max wallclock ~14.5h.
  - **Wave 2**: 2 cells parallel (sum transient ~1120 GB) after wave 1 cleanup: d24/w1152, d18/w1536. Max wallclock ~20.5h.
- Wandb groups: `grid_df02ext_20260504_224131_<cell>_df0.2`
- Launcher: [experiments/parallel/launch_df02_extension.sh](experiments/parallel/launch_df02_extension.sh) (sequential mode); inline-submitted for the 2-wave variant.
- SLURM: 40602416–40602436 (21 jobs total).

### Targeted experiment Q1: data-size ablation at d12/w768

**Q1 v1** (DONE) — `q1_20260502_234110`. Dfs {0.2, 0.4, 0.6, 0.8}, 50 epochs uniform, wd=0.1.
- Result: only df=0.2 overfit (val Δ +0.45 nats min→end). Df=0.4–0.8 stayed flat. Diagnosed as a **weight-decay confound**: `wd=0.1` is applied per-step, and higher df has 2–4× more steps per epoch, so cumulative wd-per-epoch scales with df. Train loss *also* rose (+0.5 to +0.9 nats) — the wd signature, not overfit.
- Cost: ~85 SU on cis260161p.
- Outputs: `q1_data_size.png`, `q1_data_size_zoom.png`.

**Q1 v2** (IN FLIGHT) — `q1v2_20260503_115259`. Dfs {0.2, 0.3, 0.4, 0.5}, **wd=0**, fixed-tokens budget.
- Goal: locate the **overfit-onset (in tokens-seen) as a function of unique-data fraction**, with the wd confound removed.
- Design: 4 dfs × 2 individuals (one `init_ens` + one `init_shuffle_ens`), each trained to ~1B tokens.
  - Per-df epochs computed from `ceil(1B / (df × 100M))`: `{0.2: 51, 0.3: 34, 0.4: 26, 0.5: 21}`.
  - Why "fixed tokens" not "fixed epochs": equalizes training compute across dfs and matches the Chinchilla framing for repeated-data.
  - Why dfs {0.2..0.5} (not {0.2..0.8}): with wd=0, even df=0.5 should overfit before 1B tokens; smaller cap saves SUs.
- Constant LR + CompleteP + no-ve-projs + **wd=0** throughout.
- Storage: cis260161p. Compute: cis260161p. Cost: ~64 SU.
- Wandb groups: `q1_data_size_q1v2_<TAG>_d12_w768_df{0.2,0.3,0.4,0.5}`.
- Launcher: [experiments/parallel/launch_q1_data_size_sweep.sh](experiments/parallel/launch_q1_data_size_sweep.sh). Now pipes `WEIGHT_DECAY` env var through to `train_array.sh → train.py --weight-decay`.
- SLURM: 40542399–40542406 (4 train + 4 cleanup tasks).

### Targeted experiment Q2: ensemble-size sweep up to N=20 at d12/w768, df=0.2 (IN FLIGHT)

- Goal: locate **ensemble saturation** — does val loss continue improving past N=5 ensemble members, and where does it plateau?
- Design: 1 cell × 2 strategies × **20 individuals** × 25 epochs at d12/w768, df=0.2.
- Replay at sizes {2, 5, 10, 15, 20} per strategy.
- Constant LR + CompleteP + no-ve-projs (matches the rest of the new grid; **the original df=0.2 grid `grid_20260430_152533` had trapezoidal LR and is *not* directly comparable**).
- Storage: cis260095p (~500 GB transient peak — won't fit cis260161p alongside the in-flight df=0.4 grid). Compute: cis260161p. Cost: ~100 SU (15 SU saved by fused replay).
- **Fused replay**: uses `replay_fused.py` instead of `replay.py`. One task per strategy (2 total, vs the usual 10 = 5 sizes × 2 strats); each task forwards each of the N=20 models once per evaluation point and computes ensemble val for all sizes via cumulative logit-sum. ~60% fewer forward passes vs the per-size replay.
- Wandb group: `q2_ensemble_size_q2_<TAG>_d12_w768_df0.2`. One run per (strategy, ens_size) — same schema as the per-size replay, so analysis tools (plot.py) need no changes.
- Launcher: [experiments/parallel/launch_q2_ensemble_size_sweep.sh](experiments/parallel/launch_q2_ensemble_size_sweep.sh) (submits train + per-size replay) + manual swap to [experiments/parallel/replay_array_fused.sh](experiments/parallel/replay_array_fused.sh) for the actual replay step.
- SLURM: 40532724 train + 40533112 fused-replay + 40533113 cleanup.

### Grid 3: TBD (post-Q1/Q2)

Possible follow-ups (ordered by likely value):
1. **Resume df=0.4 grid to 50 epochs** for the cells that haven't overfit. Constant-LR makes this seamless: load `model_*_epoch_25.pt`, continue another 25 epochs.
2. **df=0.1 grid** (10M tokens/epoch × 25 epochs = 250M total): faster overfit + sanity-check the data-size dependence on the full 9-cell grid (Q1 only does d12/w768).
3. **Optimizer ablation**: rerun a row at `--optimizer hybrid` or `--optimizer muon` vs AdamW.
4. **Warmdown ablation**: rerun a cell at `df=0.4` with the original trapezoidal schedule to verify constant LR doesn't change conclusions.

## What's measured

For every (cell, strategy, individual model):
- `model_{i}/train_loss` (EMA), `model_{i}/train_loss_raw` (per step)
- `model_{i}/val_loss`, `model_{i}/val_bpb` (at every 20M-token boundary in df=0.4 grid; at epoch boundary in df=0.2 grid)
- `model_{i}/epoch`, `model_{i}/step`, **`model_{i}/tokens_seen`** (canonical x-axis)

For every (cell, strategy, ensemble size):
- `ens/val_loss`, `ens/val_bpb`, `ens/num_models`
- **`ens/tokens_seen`** (canonical x-axis), `ens/epoch` (when applicable)

## Cross-grid analysis

Tools in [experiments/analysis/plot.py](experiments/analysis/plot.py):
- `slice` subcommand — vary any axis (depth, width, ens_size, strategy, df) at fixed others. **Multi-`--dfs` support already built in for cross-grid comparison.**
- `heatmap` subcommand — best-val-loss heatmap across (depth, width)
- `grid` subcommand — 3×3 grid of val curves for a single grid_tag
- `param-match` subcommand — width-vs-ensembling at matched params (the "is wider just like ensembling?" test)

When new grids land, regenerate the figures with the new grid tags and update this doc's findings section.

## Compute & storage accounting

| Grid / experiment | SU charged | Allocation(s) | Storage |
|---|---|---|---|
| df=0.2 grid | ~600 SU | cis260161p | active grid moved off-PSC to GDrive (in progress); permanents on PSC, every-5-epoch ckpts only |
| df=0.4 cells 1–3 | ~135 SU | cis260161p | cis260161p |
| df=0.4 cells 4–9 (waves A–D) | ~1230 SU | cis260009p | cis260161p (≤1 TB) + cis260095p (≤600 GB) |
| Q1 data-size ablation | ~85 SU | cis260161p | cis260161p |
| Q2 ensemble-size sweep | ~115 SU | cis260161p | cis260095p |

## Engineering notes

- All grids use the **paths (b)/(c)** parallel + post-hoc replay model from CLAUDE.md.
- `launch_grid_v2.sh` is the default for full 9-cell sweeps.
- `launch_grid_v2_waves.sh` is for parallelizing cells across waves with multi-allocation routing — used for df=0.4 cells 4–9.
- Per-cell pipeline always: train_array → replay_array → cleanup_array, gated by `afterok`. Cleanup deletes transient ckpts immediately so storage drops between waves/cells.
- Backups off PSC: `rclone` to personal Google Drive via `data.bridges2.psc.edu`. Login throttled (~1.5 MB/s); plan for ~3 days for a 400 GB pruned-grid backup.
