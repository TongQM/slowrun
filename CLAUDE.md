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

### Training Parallelism (current: sequential)

Within one ensemble strategy (e.g. `init_ens`), the N models train **sequentially on one GPU**: model 0 trains for 1 epoch → model 1 trains for 1 epoch → ... → ensemble eval → next epoch. This is ~Nx slower than true parallel training but keeps the code simple.

**Parallel SLURM allocation** *does* apply across ensemble strategies: the 3 jobs (no-ensemble, init, init+shuffle) in the job array run on separate GPUs simultaneously.

**When to refactor for parallel training:** when model size grows (e.g. d30, ~1.8B params) and sequential training becomes the bottleneck. The refactor would assign each rank (one per GPU) a different model from the seeds list, disable DDP gradient sync, and gather logits cross-rank for ensemble eval. Non-trivial rewrite — defer until needed.

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

**SLURM job array** (3 parallel runs: no ensemble, init ensemble, init+shuffle ensemble):
```bash
sbatch data_eff/run_baseline.sh
```

**Orchestrator** (multi-size sweeps):
```bash
python data_eff/completep.py \
    --model-sizes 12:12:768,20:10:1280,26:14:1792 \
    --num-models 5 --num-epochs 12 \
    --launch-prefix "torchrun --standalone --nproc_per_node=2"
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

wandb key stored at `/ocean/projects/cis260095p/ymiao6/.wandb_key`.

### Cluster (PSC Bridges-2)
- H100-80GB nodes: `w001-w010` (8 GPUs each, need `--constraint=h100`)
- V100-32GB: 33 nodes, L40S-48GB: 3 nodes (need FA2 fallback, no FA3)
- CUDA driver: 12.6 — requires `torch<2.7` (torch 2.6.0 with CUDA 12.4)
- Venv: `/ocean/projects/cis260095p/ymiao6/scaling/slowrun/.venv`
- UV dirs: `UV_CACHE_DIR=/ocean/projects/cis260095p/ymiao6/.uv/cache`, `UV_PYTHON_INSTALL_DIR=/ocean/projects/cis260095p/ymiao6/.uv/python`

## Data Efficiency Metric

For a method's val loss, find the equivalent nanochat baseline token count via piecewise interpolation. Data efficiency = equivalent tokens / actual tokens. Baseline table in `data_eff/README.md`.

## Codebase Structure

### Our Additions (`data_eff/`)
- `data_eff/completep.py` — orchestrator: launches multi-size, multi-ensemble experiments
- `data_eff/run_baseline.sh` — SLURM job array for baseline experiments
- `data_eff/README.md` — documents all modifications to upstream slowrun

### Upstream (unmodified)
- `train.py` — limited track (1 hour, single model)
- `tiny/train.py` — tiny track (15 min)
- `prepare_data.py` — FineWeb tokenization

### Architecture (GPT in `unlimited/train.py`)
- ~1.8B params at default config (d30, n_embd=2048, n_head=16)
- 15 encoder + 15 decoder layers with U-Net skip connections
- RoPE, sliding window attention (SSSL pattern), Flash Attention 2/3
- SiLU-gated MLP, value embedding (ResFormer), logit soft-capping
- Hybrid MuonAdamW optimizer with ZeRO-2 sharding

### No Tests or CI
Validation is empirical via training runs comparing val loss/BPB.

## Key Modifications to `unlimited/train.py`

See `data_eff/README.md` for full details. Summary:
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
