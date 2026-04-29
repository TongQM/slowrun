# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Goal

We study **multi-epoch training dynamics** in the data-limited regime: what happens when you train on fixed data (100M tokens) with unlimited compute? The paper (Appendix B.1) investigates:
- Whether ensembling over model inits and data order improves performance at fixed FLOPs
- Optimal model size vs dataset size in multi-epoch training
- CompleteP (muP + L_base/L depth scaling) for width/depth transfer
- Comparing Adam vs Muon optimizers under these conditions

All experiments use `unlimited/train.py` as the training backbone, modified to support these experimental axes.

## Experiment Setup

### Primary Script
`unlimited/train.py` — trains N GPT models in **synchronized epoch-by-epoch fashion** (all models in memory simultaneously), evaluates each model and the ensemble at every epoch boundary. This gives on-the-fly ensemble val loss during training.

Distillation is removed in our fork — models are fully independent. Checkpoints saved per-epoch for each model.

### Training Parallelism — two paths

**(a) Synchronized sequential (in-process)** — `train_ensemble_sync` in `unlimited/train.py`. All N models live in GPU memory; within each epoch model 0 trains 1 epoch → model 1 → ... → per-model + ensemble val eval → next epoch. Gives on-the-fly ensemble metrics, ~Nx slower per GPU than true parallel. Launched by `experiments/sync/run.sh` (2-task array: init / init+shuffle) and by `experiments/sync/sweep.py`.

**(b) Parallel SLURM + post-hoc replay** — `experiments/parallel/launch.sh`. Submits a 40-task training array via `train_array.sh` (20 models × 2 ensemble strategies, each task trains ONE model via `--single-model-idx`) into a shared checkpoint dir keyed by `SHARED_TIMESTAMP`. Training jobs skip themselves if their final checkpoint already exists, and now also **resume from the last per-epoch checkpoint** when partial progress is on disk — safe to resubmit at any time. A 10-task replay array (`replay_array.sh` → `replay.py`) runs with `--dependency=afterok`, looping over ensemble sizes `{2, 4, 8, 16, 20}` per strategy — each replay loads the first-N checkpoints at each epoch and logs **only ensemble** val loss (`SKIP_INDIV_VAL=1` by default — per-model val is already on wandb from training).

**Picking a path:** use (a) for quick iteration and small models; switch to (b) when single-model epoch time × N exceeds a SLURM slot, or to do an ensemble-size sweep over pre-trained checkpoints.

**Tunables for path (b)** — all configurable via env vars passed to `launch.sh`:
- `GPU_SPEC` (default `1` = any GPU): SLURM gres spec, e.g. `v100-32:1`, `h100-80:1`, `l40s-48:1`
- `COMPILE_MODE` (default `eager`): set to `inductor` only on H100 (~3x speedup)
- `TIME_LIMIT` (default `06:00:00`), `REPLAY_TIME_LIMIT` (default `10:00:00`)
- `ACCOUNT` (default `cis260095p`): SLURM allocation
- `VAL_EVERY_N_STEPS` (default `0`): >0 enables per-N-step val during training (slow; usually leave at 0)
- `SKIP_INDIV_VAL` (default `1`): replay skips per-model val (training already logs it)
- `END_EPOCH` (default = `NUM_EPOCHS` in script): cap replay at this epoch
- `SHARED_TIMESTAMP`: reuse an existing checkpoint dir (e.g. resume / expand a run)

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
# Default (any GPU, eager compile)
bash experiments/parallel/launch.sh

# Reuse existing checkpoints + target V100 (easier to schedule)
SHARED_TIMESTAMP=<TS> GPU_SPEC=v100-32:1 bash experiments/parallel/launch.sh

# H100 with full inductor compile
GPU_SPEC=h100-80:1 COMPILE_MODE=inductor bash experiments/parallel/launch.sh
```
Training array (40 jobs) auto-skips fully-completed models AND resumes partial ones from their last per-epoch checkpoint. Replay array (10 jobs) sweeps ensemble sizes `{2,4,8,16,20}` per strategy (size=1 omitted; per-model val is in the training runs).

**Orchestrator** (multi-size sweeps, drives path (a)):
```bash
python experiments/sync/sweep.py \
    --model-sizes 12:12:768,20:10:1280,26:14:1792 \
    --num-models 5 --num-epochs 12 \
    --launch-prefix "torchrun --standalone --nproc_per_node=2"
```

**Plot** ensemble val curves from a finished run:
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
Metric namespaces are isolated via `wandb.define_metric(step_metric=...)` so each `model_{i}/*` chart uses `model_{i}/step` as x-axis and each `ens/*` chart uses `ens/epoch`.

**Per-step (during training, for each model i):**
- `model_{i}/train_loss_raw`, `model_{i}/train_loss` (EMA smoothed, resets each epoch)
- `model_{i}/epoch`, `model_{i}/epoch_step`, `model_{i}/step` (cumulative per-model), `model_{i}/tokens_seen`

**Per-epoch (at epoch boundary, plus every `--val-every-n-steps` if set):**
- `model_{i}/val_loss`, `model_{i}/val_bpb` — individual model metrics
- `model_{i}/epoch_mean_train_loss` — arithmetic mean per epoch

**Ensemble (only at epoch boundary in path (a); post-hoc in path (b) replay):**
- `ens/val_loss`, `ens/val_bpb`, `ens/num_models`, `ens/epoch`

Each run also saves a full `config.json` to its checkpoint dir with model architecture, optimizer param groups, batch sizes, tokens-per-epoch, total-training-tokens, ensemble config, and data paths. wandb key stored at `/ocean/projects/cis260095p/ymiao6/.wandb_key`.

### Cluster (PSC Bridges-2)
- H100-80GB nodes: `w001-w010` (8 GPUs each); ~15 min/model at d12/df=0.2/30ep with `--compile-mode=inductor`. 2 SU/hr — but ~3× faster compute net-saves SUs vs V100.
- V100-32GB: 33 nodes × 8 GPUs (easy to queue; use `--compile-mode=eager`; ~50–100 min/model). 1 SU/hr.
- L40S-48GB: 3 nodes × 8 GPUs (~2× V100; `eager` only, no FA3). 1 SU/hr.
- GPU-shared partition has cpus-per-gpu cap of 5/GPU on V100 (use `--cpus-per-task=4`).
- CUDA driver on all nodes: 12.6. Torch 2.8 + cu128 pip binaries work; torch 2.10/2.11 (cu130) don't (driver too old).
- Conda env (current): `conda activate slowrun` (mapped to `/jet/home/ymiao6/.conda/envs/slowrun`, torch 2.8.0+cu128)
- Legacy uv venv still at `/ocean/projects/cis260095p/ymiao6/scaling/slowrun/.venv` (torch 2.6) — not used
- Known torch 2.8 inductor bug workaround: we set `torch._inductor.config.shape_padding = False` at import time
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
├── parallel/          # path (b): parallel single-model + ensemble-size-sweep replay
│   ├── launch.sh      # entrypoint: submits training + replay arrays with dependencies
│   ├── train_array.sh # 40-task training array (20 models × 2 strategies); skips done, resumes partial
│   ├── replay_array.sh# 10-task replay array: 2 strategies × 5 ensemble sizes {2,4,8,16,20}
│   └── replay.py      # post-hoc ensemble eval (--skip-individual-val by default)
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
- **Synchronized ensemble training** (`train_ensemble_sync`): all N models in GPU memory; epoch-by-epoch sync; supports single-model mode via `--single-model-idx` (path b) with **per-epoch resume** (auto-loads latest `model_{i}_epoch_{k}.pt` and fast-forwards loader epoch + LR step counter)
- **Per-step wandb logging** for each model with per-model `step` x-axis (isolated namespaces via `wandb.define_metric(step_metric=...)`)
- **Per-epoch checkpoints** saved for every active model (`model_{i}_epoch_{k}.pt`); feeds the replay step
- **Cross-epoch shuffle semantics** per advisor: `init` → shared π_k, `init_shuffle` → independent π_{i,k}; multiplicative seed `seed*10000+epoch` so pairs are independent
- **Fuller CompleteP**: width-aware init std, attention + FFN per-weight forward multipliers, `L_base/L` residual scaling, output multiplier, optional AdamW LR scaling (ablatable)
- **Logit vs probability** ensemble averaging (`--ensemble-mode`)
- **Three-way optimizer**: `adamw` (default, pure AdamW), `hybrid` (Muon matrices + AdamW others), `muon` (Muon for all 2D)
- **Attention backends**: FA3 → FA2 → PyTorch SDPA fallback
- **Compile modes**: `eager`, `aot_eager`, `inductor` (only H100/torch2.8)
- **Periodic val eval**: `--val-every-n-steps` for finer-grained val loss curves during training
- **Data subsetting**: `--data-fraction` (study multi-epoch dynamics with less data + more epochs)
- **Torch 2.8 inductor bug workaround**: `shape_padding=False` set at module import
- Chain distillation removed from main flow; `config.json` dumped per run for full reproducibility
