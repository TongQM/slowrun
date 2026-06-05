---
name: export-valloss
description: Snapshot individual-model and ensemble validation-loss curves from SLURM training/replay .out logs into tidy, durable, git-tracked data files (CSV + npz). Use after a slowrun training/replay sweep finishes, before pruning checkpoints or logs, or to produce a stable input for scaling-law fits. Parses individuals from fd_train_*.out and ensembles from fd_replay*_*.out (NOT offline wandb, which is fragmented across resume runs).
---

# export-valloss

Turn the slowrun grid's scattered SLURM logs into a small set of durable, tidy
data files so the validation curves survive even if the (gitignored) logs or the
checkpoints are later deleted.

## When to use

- A training + replay sweep just finished and you want the numbers in a safe place.
- You're about to prune checkpoints/logs and need to capture the curves first.
- A downstream task (scaling-law fitting, plotting) needs a stable, parsed input
  rather than re-parsing logs each time.

## Why logs, not wandb (the key gotcha)

Training runs `WANDB_MODE=offline`, and every NODE_FAIL/TIMEOUT/resume spawns a
**separate** offline-run dir with the same display name — so each model's full
0→end curve is fragmented across truncated offline runs (most reach only a
fraction of training). The `.out` logs are complete and uniform. Therefore:

- **Individuals** ← training logs `fd_train_d{d}_w{w}_{job}_{arr}.out`,
  line `[model 1 val @ step S] val_loss=L`. Array index decodes identity:
  `arr // num_models` = strategy index, `arr % num_models` = model index.
  (Single-model training always logs under namespace `model_1/`; trust the array
  index, not the namespace.) Tokens = `step * total_batch_size`.
- **Ensembles** ← replay logs. Prefers 20M/per-step `fd_replay20m_d{d}_w{w}_*.out`
  (`[step S ens=E] val_loss=L val_bpb=B tokens=T`, tokens read directly); falls
  back to per-epoch `fd_replay_d{d}_w{w}_*.out` (`[epoch K ens=E] val_loss=L`,
  tokens = `K * tokens_per_epoch`). Strategy disambiguated by the `strategy=`
  banner **inside** the file (replay log filenames don't encode strategy).

Logs are **merged by step/token**, so a base run plus an extension run that
covers later steps combine into one continuous curve automatically. Re-running a
replay over an overlapping range is harmless (same token key → same value).

## Usage

```bash
conda activate slowrun   # needs numpy; system python3 is too old
python .claude/skills/export-valloss/scripts/export_val_loss.py
```

Defaults target the df=1.0 2×2 grid (cells `6x384,12x384,6x768,12x768`,
strategies `init_ens,init_shuffle_ens`, batch 131072, 763 steps/epoch). Override
for other sweeps:

```bash
python .claude/skills/export-valloss/scripts/export_val_loss.py \
    --cells 6x768,12x768 --strategies init_ens,init_shuffle_ens \
    --num-models 5 --total-batch-size 131072 --steps-per-epoch 763 \
    --logs-dir experiments/logs --out-dir experiments/analysis/data
```

The script computes the repo root from its own location, so the default
`--logs-dir`/`--out-dir` work regardless of CWD.

## Outputs (in --out-dir, default `experiments/analysis/data/`)

| File | Columns |
|---|---|
| `val_loss_individuals.csv` | strategy, depth, width, params_m, model, step, tokens_seen, val_loss |
| `val_loss_ensembles.csv` | strategy, depth, width, params_m, E, step, tokens_seen, val_loss, **val_bpb** |
| `val_loss_long.csv` | strategy, depth, width, params_m, **kind** (individual\|ensemble), **series** (model idx or E), step, tokens_seen, val_loss, val_bpb |
| `val_loss_grid.npz` | nested arrays `ind/{strat}/d{d}_w{w}/model{m}/{tokens,val_loss}` and `ens/{strat}/d{d}_w{w}/E{E}/{tokens,val_loss,val_bpb}` |

Notes:
- Individuals carry `val_loss` only (training logs don't emit per-model bpb);
  ensembles carry both `val_loss` and `val_bpb`.
- `params_m` = non-embedding params = `16 * depth * width^2 / 1e6`.
- The `out-dir` is git-trackable (unlike `experiments/logs/`, which is
  gitignored) — **commit it** to make the curves truly durable.

## Sanity check

Row counts should equal `n_strategies × n_cells × n_series × n_points`:
- individuals: `× num_models × (steps logged)`
- ensembles: `× n_ens_sizes × (replay points)`

A `WARNING: no rows parsed` line means `--logs-dir`/`--cells`/`--strategies`
didn't match any files — check the cell list and that the logs exist.
