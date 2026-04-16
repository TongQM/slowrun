# Multi-Epoch Ensembling under Limited Training Data

We study **multi-epoch training dynamics** in the data-limited regime: 100M tokens of FineWeb, fixed, with unlimited compute. Under this regime, does ensembling over model **inits** and **data orders** improve val loss at fixed FLOPs — and when it does, which axis matters more?

We isolate two ensembling axes and compare them head-to-head:
- `init` ensemble — N models with different initializations, same per-epoch data permutation.
- `init_shuffle` ensemble — N models with different inits **and** independent per-epoch permutations.

The training backbone is [unlimited/train.py](unlimited/train.py) (a modified fork of NanoGPT Slowrun's unlimited track). We run every experiment along two **implementation paths** that produce identical ensemble metrics but trade off engineering cost:

| Path | Where | How it works | Use when |
|---|---|---|---|
| **(a) Synchronized in-process** | [experiments/sync/](experiments/sync/) | All N models live in GPU memory; train epoch-by-epoch; per-model + ensemble val eval each epoch boundary, logged on-the-fly. | Small models, quick iteration. |
| **(b) Parallel + post-hoc replay** | [experiments/parallel/](experiments/parallel/) | One model per SLURM task into a shared checkpoint dir; a dependent replay job loads `model_{i}_epoch_{k}.pt` and computes ensemble metrics offline. | Larger models, longer epochs, or when you want to rerun ensemble eval (e.g. `prob` vs `logit` averaging) without retraining. |

## Quick start

```bash
# one-time: tokenize 100M FineWeb tokens -> fineweb_data/
python prepare_data.py

# path (a): 2-task SLURM array (init / init+shuffle)
sbatch experiments/sync/run.sh

# path (b): submits 10-task training array + dependent 2-task replay array
bash experiments/parallel/launch.sh

# multi-size sweep (CompleteP: muP width + 1/L depth scaling)
python experiments/sync/sweep.py \
    --model-sizes 12:12:768,20:10:1280,26:14:1792 \
    --num-models 5 --num-epochs 12
```

The experimental flags we added (`--ensemble-type`, `--ensemble-mode`, `--optimizer`, `--completep`, `--num-models`, ...) and their defaults are documented in [experiments/README.md](experiments/README.md). Architecture, cluster notes, and the data-efficiency metric live in [CLAUDE.md](CLAUDE.md).

## What we measure

Every run logs to wandb project `slowrun` (entity: `xjtumyd-carnegie-mellon-university`):

- **Per-step, per model:** `model_{i}/train_loss_raw`, `model_{i}/train_loss` (EMA), `model_{i}/tokens_seen`.
- **Per epoch, per model:** `model_{i}/val_loss`, `model_{i}/val_bpb`.
- **Per epoch, ensemble:** `ens/val_loss`, `ens/val_bpb`, `ens/num_models` — under both `prob` (avg softmax) and `logit` (avg pre-softmax) averaging modes.
- **Data-efficiency multiplier:** for a given val loss, we find the equivalent nanochat single-epoch token count by piecewise interpolation against a baseline table of `(d12, d20, d26) × (200M, 400M, ..., 2B)` val losses. Ratio = equivalent-tokens / actual-tokens. Baseline table in [experiments/README.md](experiments/README.md).

## Why limited data, unlimited compute?

The bitter lesson tells us to prefer algorithms that scale with compute alone. We can't improve models at the rate compute scales as long as performance is bottlenecked by data. Most published pre-training contests (modded-nanogpt, etc.) set wall-clock time as the binding constraint and implicitly reward algorithms that read data faster. That filters out an entire class of data-efficient methods — ensembling, heavy regularization, expensive optimizers, evolutionary search — which is exactly the class we want to study.

We choose 100M tokens because it is small enough to afford dozens of ablations, while large enough that the winning techniques may transfer to larger scale (an open empirical question).

## Baseline

Our base model follows the recipe from Kim et al. (2025)[^1] and the upstream unlimited-track leaderboard: a heavily-regularized 1.2–2.7B transformer (U-Net skip connections, SwiGLU MLP, RoPE, sliding-window attention, Muon + AdamW hybrid optimizer), dropout 0.1, weight decay 1.3–1.6. Heavy regularization is load-bearing: without it, a 1.4B model beats a 2.7B one. With it, scale becomes monotonic.

![Overparametrization](overparametrization.png)
*From Andrew Gordon Wilson, ["Deep Learning is Not So Mysterious or Different."](https://arxiv.org/abs/2503.02113)*

## Lineage

This fork descends from [NanoGPT Slowrun](https://github.com/qlabs-eng/slowrun) — a pre-training benchmark in the fixed-data / unlimited-compute regime. The unlimited-track leaderboard there already demonstrated strong results from ensembling (3.024 val loss from a 20-model probability-averaged ensemble, 210 H100-hours). Our contribution is to disentangle **which** ensembling axis drives those gains and whether the two implementation paths yield identical curves. The upstream benchmark and its limited/tiny tracks are preserved under [legacy/](legacy/).

[^1]: Konwoo Kim, Suhas Kotha, Percy Liang, Tatsunori Hashimoto. ["Pre-training under infinite compute."](https://arxiv.org/abs/2509.14786) arXiv:2509.14786, 2025.
