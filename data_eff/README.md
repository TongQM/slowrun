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

### Per-Epoch Tracking
- `train_loss_raw`: unsmoothed per-step loss logged to wandb
- `epoch`, `epoch_step`, `tokens_seen`: logged per step for intra-epoch analysis
- `epoch_mean_train_loss`: arithmetic mean of all step losses in each epoch
- EMA smoothed loss resets at epoch boundaries (debiasing uses `epoch_step`)

### Ensemble Averaging Mode (`--ensemble-mode {prob,logit}`)
- `prob` (default): averages `softmax(logits)` across models, loss = `-log(avg_prob)`
- `logit`: averages raw logits before softmax, loss = `cross_entropy(avg_logits, target)`
- Applied in `evaluate_ensemble_bpb()` at all 3 call sites

### Ensemble Type (`--ensemble-type {init,init_shuffle}`)
- `init_shuffle` (default): each model gets a different seed for both init and data shuffling (original behavior)
- `init`: models get different init seeds but the same data shuffle seed (42), isolating the effect of init diversity from data order diversity

### CompleteP (`--completep`)
Enables muP width scaling + 1/L depth scaling:
- **1/L depth scaling**: each Block's attn and MLP outputs are multiplied by `1/n_layer` before the residual add. This is separate from `resid_lambdas` (which scale the residual stream before each block).
- **Output multiplier**: logits are scaled by `mup_base_width / n_embd` before soft-capping. Default `mup_base_width=256` (`--mup-base-width`).
- **muP LR scaling** (AdamW only): when `--optimizer adamw --completep`, matrix param LR is scaled by `base_width / n_embd`. Muon doesn't need this (orthogonalization is width-invariant).

### Optimizer Selection (`--optimizer {hybrid,muon,adamw}`)
- `hybrid` (default): Muon for matrix params + AdamW for embeddings, scalars, LM head
- `muon`: Muon for all trainable 2D+ params (matrices, embeddings, LM head); AdamW for 1D scalars only
- `adamw`: pure AdamW for all parameters

### Orchestrator: `data_eff/completep.py`
For each model size, launches up to 3 `unlimited/train.py` runs:
1. No ensemble (`--num-models 1`)
2. Init ensemble (`--num-models N --ensemble-type init`)
3. Init+shuffle ensemble (`--num-models N --ensemble-type init_shuffle`)

Model sizes are specified as comma-separated `layer:head:embd` triples.

```
# Single model size
python data_eff/completep.py \
    --model-sizes 12:12:768 \
    --num-models 5 --num-epochs 12 \
    --optimizer hybrid --nproc 8

# Multi-size sweep (muP width/depth transfer)
python data_eff/completep.py \
    --model-sizes 12:12:768,20:10:1280,26:14:1792 \
    --num-models 5 --num-epochs 12 \
    --optimizer hybrid --nproc 8 \
    --wandb-group completep_sweep
```
