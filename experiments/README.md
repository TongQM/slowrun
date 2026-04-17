## Running on Lonestar6 (TACC)

One-time setup, inside an idev session (flash-attn build is too heavy for the login node):

```bash
idev -p gpu-a100-small -A DMS26010 -t 1:00:00
bash experiments/env/setup_lonestar.sh         # creates .venv + installs torch + flash-attn (FA2)
echo YOUR_WANDB_KEY > ~/.wandb_key && chmod 600 ~/.wandb_key
python prepare_data.py                          # tokenize 100M FineWeb tokens
```

Launch the parallel ensembling experiment (path (b)):

```bash
bash experiments/parallel/launch.sh
```

This submits two SBATCH arrays on partition `gpu-a100-small`, account `DMS26010`:
- **train_array** (10 tasks): 5 models × {init, init_shuffle} trained independently into shared checkpoint dirs
- **replay_array** (2 tasks, `--dependency=afterok`): per-epoch ensemble eval + per-model val loss, logged to wandb

See [parallel/train_array.sh](parallel/train_array.sh) for model size, epoch count, and ensemble-mode knobs. Edit there before launching if the defaults (d12, w768, 30 epochs, data_fraction=0.2, logit averaging) aren't what you want; make sure [parallel/replay_array.sh](parallel/replay_array.sh) stays in sync.

### A100 note

FA3 is Hopper-only and is auto-skipped on A100. With `flash-attn` installed, FA2 is used (sliding-window attention preserved). Without it, the code falls back to SDPA and disables sliding-window — [setup_lonestar.sh](env/setup_lonestar.sh) installs it so you don't have to think about this.

---

## Data efficiency calculation

For a given method's validation loss, we use piecewise linear interpolation between the baseline validation loss to find the equivalent dataset size that achieves the same loss. Data efficiency is then the ratio of this equivalent size to the base token budget. For example if our method trained on 100M tokens matches the test loss of 1B tokens using a baseline, the data efficiency is 10x. 

We use nanochat as our baseline to compute data efficiency. For each token count, we train nanochat across multiple model sizes (d12, d20, d26), and take the best performance at each token count for the interpolation calculation. The validation loss for d12, d20, and d26 model sizes trained with nanochat defaults at various token counts are given here: 

|Data|d12|d20|d26|
|---|---|---|---|
|200M|3.703|3.794|3.883|
|400M|3.460|3.416|3.370|
|600M|3.356|3.270|3.200|
|800M|3.302|3.184|3.123|
|1B|3.251|3.124|3.046|
|2B|3.144|2.973|2.892|

## Modifications to `unlimited/train.py` (vs upstream slowrun)

### Synchronized Ensemble Training (primary change)

Replaces the upstream flow (train model 1 fully, then model 2, etc.) with **synchronized epoch-by-epoch training** of all N models:

1. Build N models in GPU memory at once (different init/data seeds per ensemble-type)
2. For each epoch: train each model for 1 epoch, then evaluate each model individually, then evaluate the ensemble
3. Save per-epoch checkpoints: `model_{i}_epoch_{k}.pt` for all i, k

Benefit: per-epoch ensemble validation metrics are logged to wandb **on the fly** during training.

Implemented in `train_ensemble_sync()` + `evaluate_ensemble_in_memory()`. Chain distillation removed from the main flow.

### Per-Step Wandb Tracking
- `model_{i}/train_loss_raw`: unsmoothed per-step loss
- `model_{i}/train_loss`: EMA smoothed, resets at epoch boundaries
- `model_{i}/epoch`, `model_{i}/epoch_step`, `model_{i}/tokens_seen`
- `step_global`: shared x-axis across all models

### Per-Epoch Ensemble Tracking (wandb)
- `model_{i}/val_bpb`, `model_{i}/val_loss` — each model's metrics at each epoch
- `model_{i}/epoch_mean_train_loss` — arithmetic mean train loss per epoch
- `ens/val_bpb`, `ens/val_loss`, `ens/num_models` — ensemble metrics
- `epoch` — standalone epoch axis for grouping

### Ensemble Averaging Mode (`--ensemble-mode {prob,logit}`)
- `prob` (default): averages `softmax(logits)` across models, loss = `-log(avg_prob)`
- `logit`: averages raw logits before softmax, loss = `cross_entropy(avg_logits, target)`

### Ensemble Type (`--ensemble-type {init,init_shuffle}`)
Both modes use cross-epoch shuffling (data order changes per epoch). The difference is whether it's shared across models:
- `init` → **π_k** (shared schedule): all models use `data_seed=42`, so at epoch k every model sees the same permutation π_k. Different epochs have different permutations. Only model initialization differs across ensemble members.
- `init_shuffle` → **π_{i,k}** (independent schedules): each model uses `data_seed=42+i` so every (model, epoch) pair has its own unique permutation. Init AND data order diverge across models.

### CompleteP (`--completep`)
Enables muP width scaling + 1/L depth scaling:
- **1/L depth scaling**: each Block's attn and MLP outputs are multiplied by `1/n_layer` before the residual add. This is separate from `resid_lambdas` (which scale the residual stream before each block).
- **Output multiplier**: logits are scaled by `mup_base_width / n_embd` before soft-capping. Default `mup_base_width=256` (`--mup-base-width`).
- **muP LR scaling** (AdamW only): when `--optimizer adamw --completep`, matrix param LR is scaled by `base_width / n_embd`. Muon doesn't need this (orthogonalization is width-invariant).

### Optimizer Selection (`--optimizer {hybrid,muon,adamw}`)
- `hybrid` (default): Muon for matrix params + AdamW for embeddings, scalars, LM head
- `muon`: Muon for all trainable 2D+ params (matrices, embeddings, LM head); AdamW for 1D scalars only
- `adamw`: pure AdamW for all parameters

### Orchestrator: `experiments/sync/sweep.py`
For each model size, launches up to 3 `unlimited/train.py` runs:
1. No ensemble (`--num-models 1`)
2. Init ensemble (`--num-models N --ensemble-type init`)
3. Init+shuffle ensemble (`--num-models N --ensemble-type init_shuffle`)

Model sizes are specified as comma-separated `layer:head:embd` triples.

```
# Single model size
python experiments/sync/sweep.py \
    --model-sizes 12:12:768 \
    --num-models 5 --num-epochs 12 \
    --optimizer hybrid --nproc 8

# Multi-size sweep (muP width/depth transfer)
python experiments/sync/sweep.py \
    --model-sizes 12:12:768,20:10:1280,26:14:1792 \
    --num-models 5 --num-epochs 12 \
    --optimizer hybrid --nproc 8 \
    --wandb-group completep_sweep
```
