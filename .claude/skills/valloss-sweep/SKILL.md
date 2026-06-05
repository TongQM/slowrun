---
name: valloss-sweep
description: Launch or extend the slowrun multi-epoch training+replay pipeline to produce individual-model and ensemble validation-loss curves over a (depth x width) grid at a chosen token resolution. Use to (a) run a fresh grid sweep on fixed data, or (b) extend an existing grid to more epochs to see late-epoch overfit. Handles resume-to-more-epochs, replay resolution (20M per-step vs 100M per-epoch), START_STEP to recompute only the new region, SLURM account/storage routing, and monitoring. After it finishes, use the export-valloss skill to snapshot the curves.
---

# valloss-sweep

Produce individual + ensemble validation-loss curves for the slowrun multi-epoch
study by submitting, per (depth×width) cell, a `train → replay` SLURM chain. The
backbone scripts stay in `experiments/parallel/` (`train_array.sh`,
`replay_array_fused.sh`, `replay_fused.py`, `unlimited/train.py`); this skill's
`scripts/run_sweep.sh` is the parameterized launcher and `scripts/monitor_sweep.sh`
is a progress snapshot.

## What it produces

- **Individuals**: each cell trains `num_models × 2 strategies` single-model jobs
  (`--single-model-idx`), logging per-step val loss to `fd_train_*.out`.
- **Ensembles**: a fused replay loads the first-N checkpoints at each eval point
  and logs ensemble val loss for `E ∈ {2,3,4,5}` to `fd_replay20m_*.out`.
- The curves are then parsed into durable files by the **export-valloss** skill.

Strategies: `init_ens` (shared per-epoch shuffle π_k) and `init_shuffle_ens`
(independent per-model shuffle π_{i,k}). Array index → identity:
`arr // num_models` = strategy, `arr % num_models` = model.

## Resolution (the knob that matters)

`CHECKPOINT_EVERY_N_STEPS` sets the per-step checkpoint cadence; at batch 131072,
**152 steps ≈ 20M tokens**. Then:
- `EVAL_MODE=step` → ensemble val at every per-step ckpt (**~20M resolution**).
- `EVAL_MODE=epoch` → ensemble val at per-epoch ckpts only (**~100M resolution**, cheaper, ~5× fewer points).

Individuals are always at the per-step (`VAL_EVERY_N_STEPS`) cadence.

## Fresh run

```bash
# df=1.0 2x2 grid (d{6,12} x w{384,768}), 31 epochs, 20M resolution, charge cis260009p
bash .claude/skills/valloss-sweep/scripts/run_sweep.sh

# preview without submitting
DRY_RUN=1 bash .claude/skills/valloss-sweep/scripts/run_sweep.sh

# custom cells / epochs / resolution
CELLS="6:6:384 12:12:768" NUM_EPOCHS=25 CHECKPOINT_EVERY_N_STEPS=305 \
  bash .claude/skills/valloss-sweep/scripts/run_sweep.sh
```
`GRID_TAG` is auto-generated (`sweep_<timestamp>`) and printed; note it for
monitoring/extension. Cells are `depth:head:width`, smallest first.

## Extend an existing run to more epochs

Because of constant LR (`--no-warmdown`), extending is exact: training resumes
from the latest per-epoch checkpoint and continues; already-done epochs are
skipped; the step counter continues. Replays then recompute **only the new
region** (via `START_STEP`) and merge with the existing points by token.

```bash
# extend the df=1.0 grid from 31 -> 40 epochs
MODE=extend GRID_TAG=fulldata_20260528_150057 NUM_EPOCHS=40 PREV_EPOCHS=31 \
  bash .claude/skills/valloss-sweep/scripts/run_sweep.sh
```
`START_STEP` is auto-derived as `PREV_EPOCHS * STEPS_PER_EPOCH + 1` (e.g.
`31*763+1 = 23654`, so the replay evaluates only steps ≥ 23712). Override
`START_STEP` directly if you computed it yourself.

## Parameters (env vars)

| Var | Default | Meaning |
|---|---|---|
| `MODE` | `fresh` | `fresh` or `extend` |
| `GRID_TAG` | auto (fresh) | run id; **required** for extend (the existing run's tag) |
| `CELLS` | `6:6:384 6:12:768 12:12:384 12:12:768` | `depth:head:width`, space-separated |
| `DF` | `1.0` | data fraction (df=1.0 = full ~100M tokens) |
| `NUM_EPOCHS` | `31` | target epochs (extend: the new, larger total) |
| `NUM_MODELS` | `5` | individuals per strategy |
| `CHECKPOINT_EVERY_N_STEPS` | `152` | ckpt cadence (152 ≈ 20M tokens) |
| `EVAL_MODE` | `step` | ensemble resolution: `step` (20M) / `epoch` (100M) / `both` |
| `PREV_EPOCHS` | — | extend+step: prior epoch count, to derive `START_STEP` |
| `START_STEP` | `0` (fresh) | replay evaluates step ckpts with `S ≥ START_STEP` |
| `WEIGHT_DECAY` | `0` | AdamW decoupled wd |
| `ACCOUNT` | `cis260009p` | SLURM allocation charged for **compute** |
| `CHECKPOINT_BASE` | `checkpoints` | where ckpts are written (under repo) |
| `TRAIN_TIME` / `REPLAY_TIME` | `03:00:00` / `06:00:00` | per-array time limits |
| `DRY_RUN` | `0` | print sbatch lines only |

Always-on project defaults (overridable): `COMPLETEP=1`, `NO_VE_PROJS=1`,
`NO_WARMDOWN=1`, `COMPILE_MODE=inductor`, `TOTAL_BATCH_SIZE=131072`,
`OPTIMIZER=adamw`, `ENSEMBLE_MODE=logit`, mup bases 768/12/64.

## Monitor

```bash
bash .claude/skills/valloss-sweep/scripts/monitor_sweep.sh <GRID_TAG>
```
Shows the queue, per-replay completion (`done=`) and points so far, and training
resume evidence. Run under `/loop` to watch continuously. **Verify resume on
extend**: the training log should show `Resumed weights from …epoch_<PREV>.pt
(epoch <PREV>)` — if a task trains `Epoch 1/…` fresh instead, the resume failed.

## Gotchas / hard-won lessons

- **Resume reads the per-epoch ckpt**, not the step ckpt — so extension needs the
  `model_*_epoch_<PREV>.pt` files present (don't delete those before extending).
- **`START_STEP` must actually filter** in `replay_fused.py::_step_points`
  (`common = [S for S in common if S >= args.start_step]`). A missing filter
  silently recomputes the whole 1→N range and the replay times out — size
  `REPLAY_TIME` for the *new region only* when extending.
- **Replay resolution ≠ checkpoint cadence retained.** Step ckpts are needed only
  during replay; once exported you can prune them (see a future prune skill).
- **Storage**: ckpts land under `CHECKPOINT_BASE` (cis260161p). `ACCOUNT` only
  charges compute — its disk being full is irrelevant. For very large grids route
  big cells to a roomier allocation via `CHECKPOINT_BASE` + the symlink scheme.
- **cis260009p compute, cis260161p storage** is the standing convention here.

## After it finishes

Run the **export-valloss** skill to snapshot the curves into tidy CSV/npz, then
plot (`experiments/analysis/plot_fulldata_indiv_ens.py`) or fit.
