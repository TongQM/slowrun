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
Width-aware init + per-output-projection forward multipliers. Spec from advisor with one off-spec extension: Q/K/V also get the `d_base/d` forward multiplier (no-op on Q,K under RMSNorm; tightens V-side variance balance to `Var(V) ∝ d_base`).
- **Init** (truncated normal, ±3σ, `init_std = 0.02`): embedding and LM head use constant `init_std` (no width scaling). Q/K/V, attn `c_proj`, FFN `c_gate`/`c_fc`, and `ve_projs` use `init_std × sqrt(d / mup_base_width)` (fan-in = `d`). FFN `c_proj` uses `init_std × sqrt(hidden / hidden_base)` (fan-in = `hidden`). At `d == mup_base_width` every `sqrt` factor is 1.
- **Forward multipliers**: every dense matrix except the readout-style embedding/LM-head layers gets `d_base/d` (or `hidden_base/hidden` for FFN `c_proj`). Specifically: attn Q/K/V (off-spec) and `c_proj`; FFN `c_gate, c_fc, c_proj`; `ve_projs` (`ve_multiplier`); LM head logits.
- **Depth scaling**: each Block's attn and MLP outputs are multiplied by `mup_base_depth / n_layer` before the residual add. Default base depth 12.
- **Attention scale**: `sqrt(mup_base_head_dim) / head_dim` instead of `1 / sqrt(head_dim)`. At constant head_dim (our setup) this collapses to the default.
- **No matrix-LR rescaling**: width handling lives entirely in init + forward multipliers; AdamW matrix LR is width-invariant. Train all widths with the same LR.

### Optimizer Selection (`--optimizer {hybrid,muon,adamw}`)
- `adamw` (default): pure AdamW for all parameters. CompleteP HP-transfer claims target this optimizer, so it's the right baseline for width/depth experiments.
- `hybrid`: Muon for matrix params + AdamW for embeddings, scalars, LM head
- `muon`: Muon for all trainable 2D+ params (matrices, embeddings, LM head); AdamW for 1D scalars only

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
    --optimizer adamw --nproc 8

# Multi-size sweep (muP width/depth transfer)
python experiments/sync/sweep.py \
    --model-sizes 12:12:768,20:10:1280,26:14:1792 \
    --num-models 5 --num-epochs 12 \
    --optimizer adamw --nproc 8 \
    --wandb-group completep_sweep
```
