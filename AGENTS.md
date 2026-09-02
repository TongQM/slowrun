# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Research Goal

We study **multi-epoch training dynamics** in the data-limited regime: what happens when you train on fixed data (100M tokens) with unlimited compute? The paper (Appendix B.1) investigates:
- Whether ensembling over model inits and data order improves performance at fixed FLOPs
- Optimal model size vs dataset size in multi-epoch training
- CompleteP (muP + L_base/L depth scaling) for width/depth transfer
- Comparing Adam vs Muon optimizers under these conditions

All experiments use `unlimited/train.py` as the training backbone, modified to support these experimental axes.

## Project defaults (always-on)

These are baked into every launcher in this repo and apply to every new run unless explicitly overridden:

| Setting | Value | Why |
|---|---|---|
| `--completep` | **ON** | Width/depth HP-transfer; required for cross-cell scaling comparisons |
| `--no-ve-projs` | **ON** | Disable value-embedding projections (advisor-specified ablation) |
| `--no-warmdown` | **ON** | Constant LR throughout training (no LR cooldown). Makes resume-extension trivial; eliminates LR/overfit confounds in late-epoch val curves. |
| `--optimizer adamw` | default | CompleteP HP-transfer claims target AdamW |
| `--ensemble-mode logit` | default | Averaging raw logits before softmax (cleaner than `prob`) |
| `GPU_SPEC=h100-80:1` | always | Compute-cheaper per training token than V100/L40S despite 2× hourly rate |
| `COMPILE_MODE=inductor` | always | ~3× speedup on H100; requires torch 2.8+ (we have 2.9.1) |
| `--total-batch-size=131072` | 128K | "Small batch" — gives finer LR/overfit dynamics resolution than 512K default |
| `--mup-base-width 768`, `--mup-base-depth 12`, `--mup-base-head-dim 64` | always | Reference base for CompleteP scaling. d12_w768 is the "1×" cell. |

**Wandb x-axis convention:** all curves use cumulative `tokens_seen` (per-model, including multi-epoch repeats), set as `step_metric` for both `model_{i}/*` and `ens/*`. Not optimizer step. This makes plots directly readable as "how does val loss evolve as each model re-reads the fixed dataset" — the central question of the project.

**Plot styling convention.** Standard setup at the top of any plotting function (executive decision; project-wide default):

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale=1.5)
sns.set_style('whitegrid')
sns.set_palette('cool', len(num_lines))   # default — pick palette size up-front

plt.rcParams['axes.labelsize']  = 24
plt.rcParams['axes.linewidth']  = 4.0      # thick bounding box
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['grid.alpha']      = 0.25
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
```

Plot defaults:
- **Palette: `cool`** (sequential, perceptually uniform). Alt: `rocket` for warm-sequential, `magma` for high-contrast. Don't mix palettes within one figure.
- **Linewidth: thick (`lw=3.0` for primary lines; `lw=2.5` for reference dashed lines).** Helps multi-panel readability.
- **Save format: PDF for paper, save with `bbox_inches='tight'` and `dpi=300`** (high DPI matters when journal rasterizes).
- **Axis labels: `fontsize=28` for math notation; default `axes.labelsize=24` otherwise.**
- **Bands for std/CI**: `plt.fill_between(..., color=f'C{i}', alpha=0.1)` matching the line color.

Example (per-line band + dashed reference fit):
```python
for i, line in enumerate(lines):
    plt.loglog(xs, line, label=f'N = {N[i]}', lw=3.0)
    plt.fill_between(xs, line - std[i], line + std[i], color=f'C{i}', alpha=0.1)
    plt.loglog(xs, fit[i], '--', color='black', lw=2.5)
plt.savefig('fig.pdf', bbox_inches='tight', dpi=300)
```

## Experiment Setup

### Primary Script
`unlimited/train.py` — trains N GPT models in **synchronized epoch-by-epoch fashion** (all models in memory simultaneously), evaluates each model and the ensemble at every epoch boundary. This gives on-the-fly ensemble val loss during training.

Distillation is removed in our fork — models are fully independent. Checkpoints saved per-epoch for each model.

### Training Parallelism — two paths

**(a) Synchronized sequential (in-process)** — `train_ensemble_sync` in `unlimited/train.py`. All N models live in GPU memory; within each epoch model 0 trains 1 epoch → model 1 → ... → per-model + ensemble val eval → next epoch. Gives on-the-fly ensemble metrics, ~Nx slower per GPU than true parallel. Launched by `experiments/sync/run.sh` (2-task array: init / init+shuffle) and by `experiments/sync/sweep.py`.

**(b) Parallel SLURM + post-hoc replay** — `experiments/parallel/launch.sh`. Submits a 40-task training array via `train_array.sh` (20 models × 2 ensemble strategies, each task trains ONE model via `--single-model-idx`) into a shared checkpoint dir keyed by `SHARED_TIMESTAMP`. Training jobs skip themselves if their final checkpoint already exists, and now also **resume from the last per-epoch checkpoint** when partial progress is on disk — safe to resubmit at any time. A 10-task replay array (`replay_array.sh` → `replay.py`) runs with `--dependency=afterok`, looping over ensemble sizes `{2, 4, 8, 16, 20}` per strategy — each replay loads the first-N checkpoints at each epoch and logs **only ensemble** val loss (`SKIP_INDIV_VAL=1` by default — per-model val is already on wandb from training).

**(c) Grid v2: cell-by-cell scan with train → replay → cleanup chain** — `experiments/parallel/launch_grid_v2.sh`. The current production pipeline. Iterates over its 9 hardcoded (depth × width) cells smallest-to-largest, and for each cell submits three SLURM arrays gated by `--dependency=afterok`:
1. **train_array** (5 models × 2 strategies = 10 tasks)
2. **replay_array** (4 ensemble sizes × 2 strategies = 8 tasks) — depends on train completion
3. **cleanup_array** (2 tasks: 1 per strategy) — depends on replay; deletes transient mid-epoch checkpoints, keeps only "permanent" ones (every Nth save)

The next cell's training waits for the previous cell's cleanup, so transient checkpoints from one cell are reclaimed before the next cell's transients pile up. Both strategies of one cell run in parallel by default; large cells (e.g. d24/w1536) override to sequential strategies because their per-strategy transient already exceeds any single allocation.

**(c′) df=0.2 grid extension** — `experiments/parallel/launch_df02_extension.sh`. Same three-array chain as (c), but for the 7 cells that turn the 3×3 core into the **4×4 headline grid**: the d18 row (`d18/{w384, w768, w1152, w1536}`) and the w1152 column (`{d6, d12, d18, d24}/w1152`, sharing d18/w1152). Deliberately aligned to the *core's* schedule, i.e. WARMUP=0 + flat 80% + linear warmdown — **not** the `--no-warmdown` project default — so the new cells are directly comparable to `grid_20260430_152533`. Per-epoch ckpts only (no step ckpts), replay sizes `{2,3,4,5}`, cleanup keeps every 5th epoch.

**The shipped df=0.2 grid is therefore 4×4**: depths `{6, 12, 18, 24}` × widths `{384, 768, 1152, 1536}`, both strategies, 5 individuals per cell. That is the shape `experiments/analysis/expt_fig3_loader.py` and `data_export/expt3_grid/` expect (16 cells × 2 strategies × 5 individuals = 160 individual files; × 4 ensemble sizes = 128 ensemble files). The 3×3 cell table inside `launch_grid_v2.sh` describes only the core pass.

**Picking a path:** use (a) for quick iteration on a single small cell; (b) for a single-cell parallel sweep with one set of replay sizes; (c) for a fresh grid; (c′) to extend an existing grid with new rows/columns on the original grid's schedule.

**Tunables for paths (b) and (c)** — all configurable via env vars:
- `GPU_SPEC` (default `h100-80:1`): always H100 in this project
- `COMPILE_MODE` (default `inductor`): always inductor (H100 + torch 2.9)
- `TIME_LIMIT` (default `12:00:00`), `REPLAY_TIME_LIMIT` (default `08:00:00`)
- `ACCOUNT` (default `cis260161p`): SLURM allocation; jobs may write checkpoints into `cis260009p` / `cis260095p` for storage balancing (see "Storage management" below)
- `VAL_EVERY_N_STEPS` (default `152`): per-N-step individual val eval (~ every 20M tokens at batch 131072)
- `CHECKPOINT_EVERY_N_STEPS` (default `152` for most cells, `305` for d24/w1536): step-based checkpoint saves for fine-grained ensemble replay
- `PERMANENT_EVERY_N_STEPS` (default `760` for most cells, `1525` for d24/w1536): cleanup keeps only `model_*_step_{S}.pt` where `S % PERMANENT_EVERY_N_STEPS == 0`
- `SKIP_INDIV_VAL` (default `1`): replay skips per-model val (training logs it inline)
- `CHECKPOINT_BASE` (default `checkpoints`): top-level write location; per-cell override lets the launcher route big cells to less-full allocations
- `START_CELL`, `END_CELL` (1–9): subset the cell loop, useful for resume / smoke-test
- `DRY_RUN=1`: print sbatch commands without submitting
- `SHARED_TIMESTAMP`: reuse an existing checkpoint dir (resume an interrupted cell)

### Key Experimental Flags (added by us)
| Flag | Purpose |
|------|---------|
| `--ensemble-type {init,init_shuffle}` | `init` → π_k (shared cross-epoch shuffle across models); `init_shuffle` → π_{i,k} (independent per model+epoch). Both reshuffle data per epoch; multiplicative seeding so all (model, epoch) pairs are independent. |
| `--ensemble-mode {prob,logit}` | Probability vs logit averaging at eval time (default `logit`) |
| `--optimizer {hybrid,muon,adamw}` | Optimizer selection for comparison |
| `--completep` | Enable fuller CompleteP: width-aware init std, FFN/attention forward multipliers, depth scaling, attention scale, output multiplier |
| `--mup-base-width d_base` | Reference `n_embd` for CompleteP (default 768) |
| `--mup-base-depth L_base` | Reference `n_layer` for depth scaling `L_base/L` (default 12) |
| `--mup-base-head-dim d_head_base` | Reference attention head dimension for CompleteP softmax scaling (default 64) |
| `--num-models N` | Ensemble size (virtual size used for seed derivation in single-model mode) |
| `--single-model-idx i` | Train only model `i` of the N-ensemble (used by path b parallel jobs) |
| `--val-every-n-steps N` | Run individual val eval every N optimizer steps (default 0 → only at epoch boundaries) |
| `--data-fraction f` | Use only fraction `f ∈ (0,1]` of the training data (smaller + more epochs = multi-epoch dynamics study) |
| `--compile-mode {eager,aot_eager,inductor}` | Torch compile backend. `eager` works everywhere; `inductor` only safe on H100 w/ torch 2.8+ |
| `--no-warmdown` | Disable LR warmdown — constant LR (multiplier 1.0) throughout training. Eliminates LR/overfit confounds and makes resume-extension trivial. **Project default.** |
| `--checkpoint-every-n-steps N` | Save `model_{i}_step_{S}.pt` every N optimizer steps in single-model mode. Used for fine-grained ensemble replay (every 20M tokens via N=152 at batch 131072). Saved **in addition** to per-epoch ckpts. |
| `--no-ve-projs` | Disable value-embedding projections entirely. **Project default.** |

### Running Experiments

**Data prep** (once):
```bash
python prepare_data.py
```

**Path (a) — synchronized sync-ensemble** (2-task array: init / init+shuffle):
```bash
sbatch experiments/sync/run.sh
```

**Path (b) — single-cell parallel training + post-hoc replay**:
```bash
# Defaults baked in: H100 + inductor + completep + no-ve-projs + no-warmdown
bash experiments/parallel/launch.sh

# Resume an existing run (reuse checkpoint dir from a previous SHARED_TIMESTAMP)
SHARED_TIMESTAMP=<TS> bash experiments/parallel/launch.sh
```

**Path (c) — grid sweep (current production)**:
```bash
# Submit all 9 core cells smallest-to-largest, each as a train→replay→cleanup chain
bash experiments/parallel/launch_grid_v2.sh

# Smoke-test just the smallest cell first
END_CELL=1 bash experiments/parallel/launch_grid_v2.sh

# Resume from a partial run (skip already-done cells)
GRID_TAG=<existing_tag> START_CELL=4 bash experiments/parallel/launch_grid_v2.sh

# Dry run — print sbatch commands without submitting
DRY_RUN=1 bash experiments/parallel/launch_grid_v2.sh
```
- Wandb groups: `grid_<GRID_TAG>_d{6,12,24}_w{384,768,1536}_df<f>` — one group per cell, all share `GRID_TAG`.
- Per-cell config table is hardcoded in `launch_grid_v2.sh` (size, destination, cadence, permanent stride). Edit it there for new grids (e.g. df=0.2 vs df=0.4).

**Path (c′) — extend the df=0.2 grid to 4×4** (d18 row + w1152 column, 7 cells, ~630 SU):
```bash
bash experiments/parallel/launch_df02_extension.sh

# subset the cell loop (1–7) / resume / dry run — same env-var interface as (c)
START_CELL=4 bash experiments/parallel/launch_df02_extension.sh
DRY_RUN=1    bash experiments/parallel/launch_df02_extension.sh
```

**Single-question sweeps** (each hardcodes its own cell + schedule; read the header comment first):
```bash
# Q1: data-size ablation at d12/w768, wd=0, fixed ~1B-token budget across dfs
bash experiments/parallel/launch_q1_data_size_sweep.sh

# Q2: ensemble-size sweep to E=20 at d12/w768 df=0.2, constant LR, 20 individuals × 2 strategies
bash experiments/parallel/launch_q2_ensemble_size_sweep.sh
```
Q2's replays use the **fused** path — `replay_fused.py` (`replay_array_fused.sh`) does one forward pass per (model, eval-point) and scores every ensemble size from it, instead of `replay.py`'s one pass per (model, size, eval-point). `replay_bootstrap.py` (`replay_bootstrap.sh`) is the membership-variance variant: resamples the 20-model pool with replacement per iteration and recomputes ensemble L, giving the error bands in Expt 2.

**Orchestrator** (multi-size sweeps, drives path (a)):
```bash
python experiments/sync/sweep.py \
    --model-sizes 12:12:768,20:10:1280,26:14:1792 \
    --num-models 5 --num-epochs 12 \
    --launch-prefix "torchrun --standalone --nproc_per_node=2"
```

**Paper figures** — regenerate offline from the exports (no cluster, no wandb; needs
numpy/matplotlib/seaborn/scipy only). Each script writes its own `experiments/figures/NN_topic/`:
```bash
python experiments/analysis/expt_fig2_ensemble_scaling.py
python experiments/analysis/expt_fig3_1_grid_curves.py

# rebuild data_export/ + manifest.csv from wandb and train logs (needs wandb access)
python experiments/analysis/assemble_data.py
```

**Live wandb plots** from a finished run (legacy path, predates `data_export/`):
```bash
# Per-model + single ensemble curve per strategy
python experiments/analysis/plot.py single \
    --init-run <RUN_ID> --init-shuffle-run <RUN_ID> \
    --steps-per-epoch 38 --num-models 5 --out val.png

# Ensemble-size sweep (auto-discovers replay runs in a wandb group)
python experiments/analysis/plot.py sweep \
    --wandb-group parallel_d12_w768_df0.2_<TS> \
    --ensemble-sizes 2 4 8 16 20 \
    --steps-per-epoch 38 --num-epochs 20 --out sweep.png
```

### Tracking

All metrics logged to **wandb** project `slowrun` (entity: `xjtumyd-carnegie-mellon-university`).
Metric namespaces are isolated via `wandb.define_metric(step_metric=...)` so each `model_{i}/*` chart uses `model_{i}/tokens_seen` as x-axis and each `ens/*` chart uses `ens/tokens_seen`. **Tokens_seen is cumulative per-model and counts repeats across multi-epoch passes** (i.e. by epoch K each model has seen `K × tokens_per_epoch` tokens).

**Per-step (during training, for each model i):**
- `model_{i}/train_loss_raw`, `model_{i}/train_loss` (EMA smoothed, resets each epoch)
- `model_{i}/epoch`, `model_{i}/epoch_step`, `model_{i}/step` (cumulative per-model), **`model_{i}/tokens_seen`** (canonical x-axis)

**Per-epoch (at epoch boundary, plus every `--val-every-n-steps` if set):**
- `model_{i}/val_loss`, `model_{i}/val_bpb` — individual model metrics
- `model_{i}/epoch_mean_train_loss` — arithmetic mean per epoch

**Ensemble (only at epoch boundary in path (a); post-hoc in path (b)/(c) replay):**
- `ens/val_loss`, `ens/val_bpb`, `ens/num_models`, **`ens/tokens_seen`**, `ens/epoch` (only when the ckpt is per-epoch; absent for step-based ckpts)

Each run also saves a full `config.json` to its checkpoint dir with model architecture, optimizer param groups, batch sizes, tokens-per-epoch, total-training-tokens, ensemble config, and data paths. wandb key stored at `/ocean/projects/cis260161p/ymiao6/.wandb_key`.

### Storage management for large grids

A full 9-cell grid at `df=0.4` × 5 individuals × 2 strategies × 25 epochs × ckpt-every-20M would be ~3.4 TB raw at fp32. We can't fit that in any single allocation. The pipeline handles this with three mechanisms:

**1. Transient + permanent checkpoints.** Train saves both per-epoch ckpts (`model_{i}_epoch_{k}.pt`) and step-based ckpts (`model_{i}_step_{S}.pt`) every `CHECKPOINT_EVERY_N_STEPS` steps. After replay completes, `cleanup_array.sh` deletes step ckpts that aren't on the permanent stride, keeping only `S % PERMANENT_EVERY_N_STEPS == 0` (e.g. every 100M tokens = every 5th save). The bulk of disk usage is therefore transient and recovered before the next cell starts.

**2. Multi-allocation routing.** Three project allocations are available — `cis260161p` (2 TB), `cis260009p` (6 TB), `cis260095p` (1 TB). The launcher's per-cell config sets `CHECKPOINT_BASE` to whichever allocation has room for the cell's transient peak. Default is `cis260161p`; the d24/w1536 cell (transient peak ~743 GB at 152-cadence) is routed to `cis260009p`. The `mv` between allocations on the same Lustre filesystem is metadata-only for cold files (instant) but real I/O for actively-written files (~270 MB/s).

**3. Symlink scheme for in-flight cells.** When checkpoints have to be moved cross-allocation while replays are still reading them, leave a symlink at the original path: `mv src dst && ln -s dst src`. Replays continue to find their data via the symlink. Window between mv-unlink and ln -s creation is microseconds — risk of catching a replay open() in that gap is negligible (verified across many cell migrations on the df=0.2 sweep).

**4. Coarser cadence for the largest cell.** d24/w1536 alone has ~3 GB per ckpt × 5 models × 50 saves = ~750 GB transient per strategy, exceeding any single allocation. Saving every 305 steps (per-epoch, every 40M tokens) instead of 152 (every 20M) halves this to ~370 GB and fits. Permanent stride correspondingly bumped to 1525.

**5. Backup off PSC.** rclone to personal Google Drive via `data.bridges2.psc.edu` for permanent off-cluster archive. Login nodes throttle aggressively (~1.5 MB/s); compute nodes have no outbound internet; DTN nodes (`br010`) accept only file-transfer protocols. So in practice rclone runs on a login node and 404 GB takes ~3–4 days at the throttled rate. **Recommendation: prune to every-5-epoch checkpoints first** (see `experiments/parallel/cleanup_array.sh` for the pattern), then rclone the trimmed set.

### Cluster (PSC Bridges-2)
- H100-80GB nodes: `w001-w010` (8 GPUs each); ~15 min/model at d12/df=0.2/30ep with `--compile-mode=inductor`. 2 SU/hr — but ~3× faster compute net-saves SUs vs V100.
- V100-32GB: 33 nodes × 8 GPUs (easy to queue; use `--compile-mode=eager`; ~50–100 min/model). 1 SU/hr.
- L40S-48GB: 3 nodes × 8 GPUs (~2× V100; `eager` only, no FA3). 1 SU/hr.
- GPU-shared partition has cpus-per-gpu cap of 5/GPU on V100 (use `--cpus-per-task=4`).
- CUDA driver on all nodes: 12.6. Torch 2.8/2.9 + cu128 pip binaries work; torch 2.10/2.11 (cu130) don't (driver too old).
- Conda env: `conda activate slowrun`. The actual env lives at `/ocean/projects/cis260161p/ymiao6/scaling/slowrun/.conda_env` (torch 2.9.1+cu128). It is exposed to `conda activate slowrun` via a symlink at `/jet/home/ymiao6/.conda/envs/slowrun`. Entry-point scripts (e.g. `torchrun`) inside the env have hardcoded shebangs to the original build path `/ocean/projects/cis260095p/ymiao6/scaling/slowrun/.conda_env/bin/python3.11` — that path must also be a valid symlink to the real env, otherwise compute-node jobs fail with "bad interpreter". A symlink at `/ocean/projects/cis260095p/ymiao6/scaling/slowrun/.conda_env → /ocean/projects/cis260161p/.../slowrun/.conda_env` keeps both paths working.
- Legacy uv venv at `/ocean/projects/cis260161p/ymiao6/scaling/slowrun/.venv` (torch 2.6) — not used.
- Known torch 2.8 inductor bug workaround: we set `torch._inductor.config.shape_padding = False` at import time.
- SU pricing: V100 = 1 SU/GPU-hour, L40S = 1 SU/GPU-hour, H100 = 2 SU/GPU-hour. For our d12/df=0.2 workload, H100+inductor (~15 min/model × 2) ≈ 0.5 SU/model; V100+eager (~100 min/model × 1) ≈ 1.7 SU/model. **H100 is ~3× cheaper per model in SU despite 2× hourly rate.**

### CompleteP status
Our `--completep` matches the spec given by our advisor — width-aware init combined with output-projection forward multipliers, so the effective output variance is width-independent at every layer:
- **Init** (truncated normal, ±3σ, `init_std = 0.02`):
  - Embedding: constant `init_std` (no width scaling).
  - Q, K, V (fan-in = `d`): `init_std × sqrt(d / d_base)`.
  - Attn `c_proj` (fan-in = `d`): `init_std × sqrt(d / d_base)`.
  - FFN `c_gate`, `c_fc` (fan-in = `d`): `init_std × sqrt(d / d_base)`.
  - FFN `c_proj` (fan-in = `hidden`): `init_std × sqrt(hidden / hidden_base)`.
  - LM head: constant `init_std` (no width scaling).
  - At `d = d_base` every `sqrt` factor is 1 → all weights init at `init_std`.
- **Forward multipliers** (every dense layer except the embedding/LM-head readout-style layers gets a `d_base/d` factor; spec leaves Q/K/V at 1.0, our extension applies it there too):
  - Attn Q, K, V outputs × `d_base / d` (off-spec extension; on Q,K it's a no-op because of the subsequent RMSNorm; on V it tightens the variance balance — `Var(V) ∝ d_base`).
  - Attn `c_proj` output × `d_base / d`.
  - Attn softmax scale: `sqrt(d_head_base) / d_head` (collapses to standard `1/sqrt(d_head)` at constant head_dim).
  - FFN `c_gate`, `c_fc` outputs × `d_base / d`.
  - FFN `c_proj` output × `hidden_base / hidden`.
  - `ve_projs` output × `d_base / d` (`ve_multiplier`).
  - LM head logit × `d_base / d` (equivalent to multiplying input `h` by `d_base / d`).
  - Residual branch × `L_base / L` (default `12 / L`).
- **Single LR across widths** — width handling lives entirely in init + forward multipliers; the AdamW matrix LR is width-invariant.

## Data Efficiency Metric

For a method's val loss, find the equivalent nanochat baseline token count via piecewise interpolation. Data efficiency = equivalent tokens / actual tokens. Baseline table in `experiments/README.md`.

## Codebase Structure

### Repo layout (top level)
```
unlimited/train.py     # ensemble training backbone (~2000 lines)
experiments/           # our orchestration, replay, and analysis (the focus)
├── sync/              # path (a): synchronized in-process ensemble
│   ├── run.sh         # 2-task SLURM array (init / init+shuffle)
│   └── sweep.py       # multi-size, multi-ensemble orchestrator
├── parallel/          # paths (b), (c), (c′): parallel single-model training + replay
│   ├── launch.sh                  # path (b): submits training + replay arrays for ONE cell
│   ├── launch_grid_v2.sh          # path (c): 9-cell core grid, train→replay→cleanup chain per cell
│   ├── launch_df02_extension.sh   # path (c′): d18 row + w1152 column → 4×4 df=0.2 grid (warmdown LR)
│   ├── launch_grid.sh             # superseded v1 grid launcher (no cleanup stage); kept for reference
│   ├── launch_grid_v2_waves.sh    # wave-parallel variant of (c) for the cancelled df=0.4 cells 4–9
│   ├── launch_q1_data_size_sweep.sh    # Q1: data-fraction ablation, wd=0, fixed ~1B-token budget
│   ├── launch_q2_ensemble_size_sweep.sh # Q2: 20 individuals × 2 strategies for E up to 20
│   ├── train_array.sh    # training array (5 models × 2 strategies = 10 tasks/cell); skips done, resumes partial
│   ├── replay_array.sh   # replay array (4 ensemble sizes × 2 strategies = 8 tasks/cell)
│   ├── replay.py         # post-hoc ensemble eval; enumerates BOTH per-epoch and per-step ckpts, computes ensemble val at every distinct token-count point
│   ├── replay_fused.py / replay_array_fused.sh   # one forward pass per (model, eval-point), scores ALL sizes from it
│   ├── replay_bootstrap.py / replay_bootstrap.sh # resamples the model pool with replacement → membership-variance bands
│   └── cleanup_array.sh  # per-(cell, strategy) cleanup: deletes step ckpts not on the permanent stride; keeps every-100M-token ckpts
├── analysis/          # offline figure scripts — read data_export/, no wandb dependency
│   ├── expt_fig<N>_*.py       # one script per paper figure; see "Figure pipeline" below
│   ├── expt_fig<N>_loader.py  # shared per-figure-family loader (fig3 = the 4×4 grid, fig4 = data-size)
│   ├── assemble_data.py       # builds data_export/ (+ manifest.csv) from wandb and train logs
│   └── plot.py                # legacy live wandb plotting (single-run / ensemble-size sweep)
├── diagnostic/        # CompleteP correctness check: 8 single-model jobs spanning width & depth from d12_w768
├── figures/           # canonical figure outputs, numbered by topic (NN_slug/); 00_archive/ holds superseded ones
├── logs/              # SLURM stdout/stderr (gitignored)
└── README.md          # authoritative spec of our diffs to unlimited/train.py
data_export/           # tidy NPZ curves for offline analysis (untracked); README.md + schema.md + manifest.csv
paper/expt_fig<N>.tex  # manuscript figure blocks; each opens with a comment naming its source figure path
prepare_data.py        # FineWeb tokenization (shared)
legacy/                # upstream benchmark tracks — not the research focus
├── limited_train.py   # 1-hour single-model track
├── tiny/              # 15-min track
└── dev/               # upstream experimental attention variants
```

### Figure pipeline

```
data_export/<exptN>/**.npz → experiments/analysis/expt_fig<N>_<slug>.py
  → experiments/figures/NN_topic/*.{pdf,png}[, *_fits.csv] → paper/expt_fig<N>.tex
```

- One script per figure; anything shared across a figure family goes in `expt_fig<N>_loader.py`.
- Emit **both** `.pdf` and `.png`, plus a `*_fits.csv` whenever the figure reports fitted parameters.
- Superseded figures move to `experiments/figures/00_archive/` — they are not deleted.
- Keep the source-path comment at the top of each `paper/expt_fig<N>.tex` accurate; it is the only
  link back from the manuscript to the code that produced the figure.
- Analysis scripts resolve paths via `REPO = Path(__file__).resolve().parents[2]`, so they run from
  anywhere and always write into this repo. They need only numpy/matplotlib/seaborn/scipy — no GPU,
  no wandb — so every paper figure is reproducible offline from a laptop checkout.

### Architecture (GPT in `unlimited/train.py`)
- Default config: d12, n_embd=768, n_head=12 (~166M params). Larger configs supported via CLI.
- n_layer/2 encoder + n_layer/2 decoder layers with U-Net skip connections
- RoPE, sliding window attention (SSSL pattern), Flash Attention 2/3 with SDPA fallback
- SiLU-gated MLP, value embedding (ResFormer on alternating layers), logit soft-capping
- Pure AdamW (default) with ZeRO-2 sharding; hybrid MuonAdamW and pure Muon also available via `--optimizer`
- Light regularization by default: `dropout=0.0`, `weight_decay=0.1` (slowrun competition defaults were `0.1`/`1.3`; we relaxed to observe multi-epoch overfit)

### No Tests or CI
Validation is empirical via training runs comparing val loss/BPB.

## Key Modifications to `unlimited/train.py`

See `experiments/README.md` for full details. Summary:
- **Synchronized ensemble training** (`train_ensemble_sync`): all N models in GPU memory; epoch-by-epoch sync; supports single-model mode via `--single-model-idx` (paths b/c) with **per-epoch + per-step resume** (auto-loads latest `model_{i}_epoch_{k}.pt` or `model_{i}_step_{S}.pt` and fast-forwards loader epoch counter)
- **Step-based checkpoints** (`--checkpoint-every-n-steps N`): saves `model_{i}_step_{S}.pt` every N optimizer steps in addition to per-epoch ckpts. Used for fine-grained ensemble replay (e.g. every 20M tokens).
- **Per-step wandb logging** with per-model `tokens_seen` x-axis (isolated namespaces via `wandb.define_metric(step_metric=...)`); ensemble curves use `ens/tokens_seen`. Cumulative tokens count multi-epoch repeats.
- **Per-epoch checkpoints** saved for every active model (`model_{i}_epoch_{k}.pt`); feeds the replay step
- **Cross-epoch shuffle semantics** per advisor: `init` → shared π_k, `init_shuffle` → independent π_{i,k}; multiplicative seed `seed*10000+epoch` so pairs are independent
- **Fuller CompleteP** (`--completep`, project default): width-aware init std, attention + FFN per-weight forward multipliers, `L_base/L` residual scaling, output multiplier
- **No-warmdown LR** (`--no-warmdown`, project default): constant LR (multiplier 1.0) throughout training; eliminates LR/overfit confounds and makes resume-extension trivial
- **No value-embedding projections** (`--no-ve-projs`, project default): disables ResFormer-style value injection (advisor-specified ablation)
- **Logit vs probability** ensemble averaging (`--ensemble-mode`, default `logit`)
- **Three-way optimizer**: `adamw` (default, pure AdamW), `hybrid` (Muon matrices + AdamW others), `muon` (Muon for all 2D)
- **Attention backends**: FA3 → FA2 → PyTorch SDPA fallback
- **Compile modes**: `eager`, `aot_eager`, `inductor` (project default, H100 only)
- **Periodic val eval** (`--val-every-n-steps`): finer-grained val loss curves during training
- **Data subsetting** (`--data-fraction`): study multi-epoch dynamics with less data + more epochs
- **Torch 2.8/2.9 inductor bug workaround**: `shape_padding=False` set at module import
- Chain distillation removed from main flow; `config.json` dumped per run for full reproducibility
