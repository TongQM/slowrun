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

### Grid 2: `df=0.4` (IN FLIGHT) — `grid_20260502_013702`

- 40M tokens/epoch × 25 epochs = **1B cumulative tokens / model**
- LR schedule: **constant** (`--no-warmdown`) — no schedule artifact in late epochs
- Val cadence: **every 20M cumulative tokens** (152 optimizer steps at batch 131K) for both individuals and ensembles
- Permanent checkpoints: **every 100M cumulative tokens** (10 step-ckpts/model) plus per-epoch ckpts at every 5 epochs
- Special: d24/w1536 uses per-epoch ckpts only (cadence=305) — transient at every-20M (~743 GB/strategy) won't fit any single allocation
- Wandb groups: `grid_20260502_013702_d{6,12,24}_w{384,768,1536}_df0.4`
- Compute: ~1370 SU total. Cells 1–3 charged to cis260161p (~135 SU), cells 4–9 charged to **cis260009p** (~1230 SU).
- Storage routed to keep both budgets healthy: cis260161p (≤1 TB transient peak), cis260095p (≤600 GB transient peak).

**Pipeline (current):**
- Cells 1–3 (d{6,12,24}/w384): submitted serially via `launch_grid_v2.sh`, all done.
- Cells 4–9: re-submitted in **4 waves** via `launch_grid_v2_waves.sh` (jobs 40532076–40532099).
  - Wave A: cells 4, 5, 6 (w768 row) parallel on cis260161p
  - Wave B: cell 7 (d6/w1536, cis260161p) + cell 8 init_ens (d12/w1536, cis260095p) parallel
  - Wave C: cell 8 init_shuffle (cis260161p) + cell 9 init_ens (d24/w1536, cis260095p) parallel
  - Wave D: cell 9 init_shuffle (cis260161p) alone

**Hypothesis to verify:**
- Will overfit appear within 25 epochs at 2× more unique data? Smaller cells (d6/w384, d12/w384) likely *will not* — the d6/w384 trajectory through epoch 25 is still descending. Larger cells will. Decision on whether to extend to 50 epochs deferred until full grid lands.

### Targeted experiment Q1: data-size ablation at d12/w768 (IN FLIGHT)

- Goal: locate the **overfit-onset epoch as a function of data fraction** at the base model size.
- Design: 4 dfs × 2 individuals (one `init_ens` + one `init_shuffle_ens`) × **50 epochs** at d12/w768.
- Dfs: {0.2, 0.4, 0.6, 0.8} — twice as much unique data each time.
- Constant LR (`--no-warmdown`) + CompleteP + no-ve-projs throughout, fully comparable across df values.
- No ensembling step (1 individual per strategy is sufficient to read off overfit-onset).
- Storage: cis260161p. Compute: cis260161p. Cost: ~85 SU.
- Wandb groups: `q1_data_size_q1_<TAG>_d12_w768_df{0.2,0.4,0.6,0.8}`.
- Launcher: [experiments/parallel/launch_q1_data_size_sweep.sh](experiments/parallel/launch_q1_data_size_sweep.sh).
- SLURM: 40532716–40532723 (8 train + 8 cleanup tasks).

### Targeted experiment Q2: ensemble-size sweep up to N=20 at d12/w768, df=0.2 (IN FLIGHT)

- Goal: locate **ensemble saturation** — does val loss continue improving past N=5 ensemble members, and where does it plateau?
- Design: 1 cell × 2 strategies × **20 individuals** × 25 epochs at d12/w768, df=0.2.
- Replay at sizes {2, 5, 10, 15, 20} per strategy.
- Constant LR + CompleteP + no-ve-projs (matches the rest of the new grid; **the original df=0.2 grid `grid_20260430_152533` had trapezoidal LR and is *not* directly comparable**).
- Storage: cis260095p (~500 GB transient peak — won't fit cis260161p alongside the in-flight df=0.4 grid). Compute: cis260161p. Cost: ~115 SU.
- Wandb group: `q2_ensemble_size_q2_<TAG>_d12_w768_df0.2`.
- Launcher: [experiments/parallel/launch_q2_ensemble_size_sweep.sh](experiments/parallel/launch_q2_ensemble_size_sweep.sh).
- SLURM: 40532724 train + 40532725 replay + 40532726 cleanup.

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
