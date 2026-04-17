# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Goal

We study **multi-epoch training dynamics** in the data-limited regime: what happens when you train on fixed data (100M tokens) with unlimited compute? The paper (Appendix B.1) investigates:
- Whether ensembling over model inits and data order improves performance at fixed FLOPs
- Optimal model size vs dataset size in multi-epoch training
- CompleteP (muP + 1/L depth scaling) for width/depth transfer
- Comparing Adam vs Muon optimizers under these conditions

All experiments use `unlimited/train.py` as the training backbone, modified to support these experimental axes.

## Experiment Setup

### Primary Script
`unlimited/train.py` — trains N GPT models in **synchronized epoch-by-epoch fashion** (all models in memory simultaneously), evaluates each model and the ensemble at every epoch boundary. This gives on-the-fly ensemble val loss during training.

Distillation is removed in our fork — models are fully independent. Checkpoints saved per-epoch for each model.

### Training Parallelism — two paths

**(a) Synchronized sequential (in-process)** — `train_ensemble_sync` in `unlimited/train.py`. All N models live in GPU memory; within each epoch model 0 trains 1 epoch → model 1 → ... → per-model + ensemble val eval → next epoch. Gives on-the-fly ensemble metrics, ~Nx slower per GPU than true parallel. Launched by `experiments/sync/run.sh` (2-task array: init / init+shuffle) and by `experiments/sync/sweep.py`.

**(b) Parallel SLURM + post-hoc replay** — `experiments/parallel/launch.sh`. Submits a 10-task training array via `train_array.sh` (5 models × 2 ensemble strategies, each task trains ONE model with `--num-models 1`) into a shared checkpoint dir keyed by `SHARED_TIMESTAMP`. A 2-task replay array (`replay_array.sh` → `replay.py`) runs with `--dependency=afterok`, loading per-epoch checkpoints and computing ensemble val loss offline. Supports resume.

**Picking a path:** use (a) for quick iteration and small models; switch to (b) when single-model epoch time × N exceeds a SLURM slot, or when you want to rerun ensemble eval (different `--ensemble-mode`, subset of models) without retraining.

### Key Experimental Flags (added by us)
| Flag | Purpose |
|------|---------|
| `--ensemble-type {init,init_shuffle}` | `init` = same data order, different inits; `init_shuffle` = both differ |
| `--ensemble-mode {prob,logit}` | Probability vs logit averaging at eval time |
| `--optimizer {hybrid,muon,adamw}` | Optimizer selection for comparison |
| `--completep` | Enable muP width scaling + 1/L depth scaling |
| `--mup-base-width N` | Base width for muP LR scaling (default 256) |
| `--num-models N` | Number of ensemble members |

### Running Experiments

**Data prep** (once):
```bash
python prepare_data.py
```

**Path (a) — synchronized sync-ensemble** (2-task array: init / init+shuffle):
```bash
sbatch experiments/sync/run.sh
```

**Path (b) — parallel single-model training + post-hoc replay**:
```bash
bash experiments/parallel/launch.sh   # submits training array + replay array with afterok
```

**Orchestrator** (multi-size sweeps, drives path (a)):
```bash
python experiments/sync/sweep.py \
    --model-sizes 12:12:768,20:10:1280,26:14:1792 \
    --num-models 5 --num-epochs 12 \
    --launch-prefix "torchrun --standalone --nproc_per_node=2"
```

**Plot** ensemble val curves from a finished run:
```bash
python experiments/analysis/plot.py   # reads wandb exports
```

### Tracking

All metrics logged to **wandb** project `slowrun` (entity: `xjtumyd-carnegie-mellon-university`).

**Per-step (during training, for each model i):**
- `model_{i}/train_loss_raw`, `model_{i}/train_loss` (EMA smoothed)
- `model_{i}/epoch`, `model_{i}/epoch_step`, `model_{i}/tokens_seen`
- `step_global` — shared x-axis across all models

**Per-epoch (at epoch boundary):**
- `model_{i}/val_loss`, `model_{i}/val_bpb` — individual model metrics
- `model_{i}/epoch_mean_train_loss` — arithmetic mean per epoch
- `ensemble/val_loss`, `ensemble/val_bpb` — ensemble metrics (on-the-fly)
- `ensemble/num_models` — ensemble size
- `epoch` — current epoch

wandb key stored at `~/.wandb_key` (chmod 600).

### Cluster (TACC Lonestar6)
- Repo root: `/work/11426/yzfx0416/ls6/slowrun`
- Partitions: `gpu-a100-small` (1× A100-40GB slice, what we use), `gpu-a100` (3× A100/node), `gpu-h100` (2× H100-80/node, 4 nodes total)
- Allocation: `dms26007` (billed); `dms26010` and `default` also available
- CUDA module: `cuda/12.8`; Python: `python/3.12.11` via `module load`
- Venv: `/work/11426/yzfx0416/ls6/slowrun/.venv` (created by `experiments/env/setup_lonestar.sh`)
- A100 requires `flash-attn` (FA2) installed explicitly — otherwise SDPA fallback drops sliding-window attention
- FA3 is Hopper-only and auto-skipped on A100 via `torch.cuda.get_device_capability()` major check

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
├── parallel/          # path (b): parallel single-model + replay
│   ├── launch.sh      # entrypoint: submits training + replay arrays
│   ├── train_array.sh # 10-task training array (5 models × 2 strategies)
│   ├── replay_array.sh# 2-task replay array (depends on training)
│   └── replay.py      # post-hoc ensemble eval from saved checkpoints
├── analysis/          # plots and notebooks
│   └── plot.py
├── logs/              # SLURM stdout/stderr (gitignored)
└── README.md          # authoritative spec of our diffs to unlimited/train.py
prepare_data.py        # FineWeb tokenization (shared)
legacy/                # upstream benchmark tracks — not the research focus
├── limited_train.py   # 1-hour single-model track
├── tiny/              # 15-min track
└── dev/               # upstream experimental attention variants
```

### Architecture (GPT in `unlimited/train.py`)
- ~1.8B params at default config (d30, n_embd=2048, n_head=16)
- 15 encoder + 15 decoder layers with U-Net skip connections
- RoPE, sliding window attention (SSSL pattern), Flash Attention 2/3
- SiLU-gated MLP, value embedding (ResFormer), logit soft-capping
- Hybrid MuonAdamW optimizer with ZeRO-2 sharding

### No Tests or CI
Validation is empirical via training runs comparing val loss/BPB.

## Key Modifications to `unlimited/train.py`

See `experiments/README.md` for full details. Summary:
- **Synchronized ensemble training** (`train_ensemble_sync`): all N models live in GPU memory; train epoch-by-epoch; per-epoch ensemble val eval on the fly
- Per-step wandb logging for each model + per-epoch ensemble metrics (`ens/val_bpb`, `ens/val_loss`)
- Per-epoch checkpoints saved for every model (`model_{i}_epoch_{k}.pt`)
- Ensemble type (init vs init+shuffle) via seed decoupling
- Logit vs probability ensemble averaging
- CompleteP (1/L depth scaling + muP output multiplier + width-dependent LR)
- Three-way optimizer selection (hybrid/muon/adamw)
- Flash Attention 2 fallback / SDPA fallback when FA3 unavailable
- `--no-compile` flag for environments without Python dev headers
- Chain distillation removed from main flow
