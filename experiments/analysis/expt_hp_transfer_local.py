"""Do hyperparameters actually TRANSFER after the CompleteP residual-path fix?

The coord check (expt_coord_check_grid.py) measures activation invariance, which
is the *precondition* for HP transfer, not transfer itself. This measures the
thing we actually care about: does the optimal learning rate land in the same
place at every model size?

Method — the muP LR-transfer check:
    for each cell (L, N) and each lr_multiplier, train briefly and record loss;
    plot loss vs lr_multiplier, one curve per cell, and ask whether the minima
    align. Aligned minima = transfer. Drifting minima = no transfer.

Two variants are compared: the corrected model, and the pre-2026-09 behaviour
(reproduced by forcing `resid_path_scale = 1.0`, i.e. the depth factor reaching
only the attn/mlp branches).

Data: a fixed random BIGRAM source, not uniform tokens. Uniform tokens have no
structure to learn -- loss just floors at log(V) and the LR optimum is
meaningless. A bigram chain gives a genuine, learnable signal while staying
entirely local and reproducible.

Runs on MPS/CPU with tiny models: a parameterization property is scale-free, so
the ladder reproduces it without a GPU.

Outputs:
  experiments/figures/08_coord_check/hp_transfer_local.{pdf,png}
  experiments/figures/08_coord_check/hp_transfer_local.csv
"""
from __future__ import annotations

import sys
import types
import argparse
from pathlib import Path

sys.argv, _ARGS = [sys.argv[0]], sys.argv[1:]
for _m in ("wandb", "tiktoken"):
    try:
        __import__(_m)
    except ImportError:
        sys.modules[_m] = types.ModuleType(_m)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "unlimited"))
import train as T  # noqa: E402

OUTDIR = REPO / "experiments" / "figures" / "08_coord_check"
VOCAB, SEQ, BATCH = 256, 64, 8
HEAD_DIM, BASE_W, BASE_D = 16, 64, 4
# sqrt(2)-spaced and centred on the bracketed optimum (~2-4). An earlier grid of
# {1/64 .. 1} pinned every cell at its RIGHT edge -- loss was still falling 0.25-0.30
# per doubling there -- which looks like perfect agreement but is just a boundary
# artifact. Bracket first, then sweep: resolution finer than 2x is needed to see a
# 2x drift in the argmin.
LRMULTS = (1.0, 1.4142, 2.0, 2.8284, 4.0, 5.6569, 8.0)
DEPTH_LADDER = [(4, 64), (8, 64), (16, 64), (32, 64)]
WIDTH_LADDER = [(4, 64), (4, 128), (4, 256), (4, 512)]


def bigram_source(seed=1234):
    """A fixed, peaked random bigram chain — genuinely learnable structure."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(VOCAB, VOCAB, generator=g) * 3.0
    P = torch.softmax(logits, dim=-1)

    def sample(n_seq, gen):
        idx = torch.randint(0, VOCAB, (n_seq, 1), generator=gen)
        out = [idx]
        for _ in range(SEQ):
            probs = P[out[-1].squeeze(-1)]
            nxt = torch.multinomial(probs, 1, generator=gen)
            out.append(nxt)
        return torch.cat(out, dim=1)
    return sample


def train_cell(depth, width, lrmult, fixed, sample, steps, device, seed=0):
    cfg = T.GPTConfig(sequence_len=SEQ, vocab_size=VOCAB, n_layer=depth, n_embd=width,
                      n_head=width // HEAD_DIM, n_kv_head=width // HEAD_DIM,
                      completep=True, no_ve_projs=True, mup_base_width=BASE_W,
                      mup_base_depth=BASE_D, mup_base_head_dim=HEAD_DIM, dropout=0.0)
    torch.manual_seed(seed)
    model = T.GPT(cfg).to(device)
    model.init_weights(convert_embed=False)
    if not fixed:
        model.resid_path_scale = 1.0          # pre-2026-09 behaviour
    model.to(device)

    nd = dict(model.named_parameters())
    mats = [p for n, p in nd.items()
            if p.ndim >= 2 and "wte" not in n and "lm_head" not in n]
    M = lrmult
    opt = torch.optim.AdamW(
        [dict(params=mats, lr=0.04 * M, betas=(0.9, 0.95)),
         dict(params=[nd["transformer.wte.weight"]], lr=0.15 * M, betas=(0.9, 0.95)),
         dict(params=[nd["lm_head.weight"]], lr=0.002 * M, betas=(0.9, 0.95)),
         dict(params=[model.resid_lambdas], lr=0.1 * M * 0.01, betas=(0.9, 0.95)),
         dict(params=[model.x0_lambdas], lr=0.1 * M, betas=(0.96, 0.95)),
         dict(params=[model.skip_weights], lr=0.1 * M * 0.01, betas=(0.9, 0.95))],
        eps=1e-10, weight_decay=0.0)

    g = torch.Generator().manual_seed(seed + 7)
    losses = []
    for step in range(steps):
        ids = sample(BATCH, g).to(device)
        loss = model(ids[:, :-1].contiguous(), ids[:, 1:].contiguous())
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
    del model, opt
    tail = losses[-max(10, steps // 10):]
    v = float(np.mean(tail))
    return v if np.isfinite(v) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--steps", type=int, default=150)
    a = ap.parse_args(_ARGS)
    dev = a.device
    if dev == "mps" and not torch.backends.mps.is_available():
        dev = "cpu"
    print(f"device: {dev}  steps: {a.steps}")

    sample = bigram_source()
    rows = []
    for lname, cells in (("depth", DEPTH_LADDER), ("width", WIDTH_LADDER)):
        for fixed in (False, True):
            for (L, N) in cells:
                for lm in LRMULTS:
                    v = train_cell(L, N, lm, fixed, sample, a.steps, dev)
                    rows.append(dict(ladder=lname, fixed=fixed, depth=L, width=N,
                                     lrmult=lm, loss=v))
                print(f"  [{lname}] {'fixed' if fixed else 'as-is'} d{L}_w{N} done", flush=True)

    csv = OUTDIR / "hp_transfer_local.csv"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(csv, "w") as fh:
        fh.write("ladder,variant,depth,width,lr_multiplier,loss\n")
        for r in rows:
            fh.write(f"{r['ladder']},{'fixed' if r['fixed'] else 'as-is'},{r['depth']},"
                     f"{r['width']},{r['lrmult']},{r['loss']:.6f}\n")
    print(f"Saved {csv}")

    sns.set(font_scale=1.3)
    sns.set_style("whitegrid")
    plt.rcParams["axes.linewidth"] = 3.0
    plt.rcParams["grid.alpha"] = 0.25
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    for row, (lname, cells) in enumerate((("depth", DEPTH_LADDER), ("width", WIDTH_LADDER))):
        colors = sns.color_palette("cool", len(cells))
        for col, fixed in enumerate((False, True)):
            ax = axes[row, col]
            argmins = []
            for c, (L, N) in zip(colors, cells):
                sel = [r for r in rows if r["ladder"] == lname and r["fixed"] == fixed
                       and r["depth"] == L and r["width"] == N]
                sel.sort(key=lambda r: r["lrmult"])
                xs = [r["lrmult"] for r in sel]
                ys = [r["loss"] for r in sel]
                lab = f"d{L}" if lname == "depth" else f"w{N}"
                ax.plot(xs, ys, "-o", color=c, lw=3.0, ms=8, label=lab)
                if np.all(np.isnan(ys)):
                    continue
                am = xs[int(np.nanargmin(ys))]
                argmins.append(am)
                ax.axvline(am, color=c, ls=":", lw=2.0, alpha=0.8)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("lr_multiplier")
            ax.set_ylabel("loss (mean of last 10%)")
            spread = (max(argmins) / min(argmins)) if argmins else float("nan")
            tag = "with fix" if fixed else "as-is"
            ax.set_title(f"{lname} ladder — {tag}\nargmin spread = {spread:.0f}x"
                         + ("  ✓ aligned" if spread <= 1.01 else ""), fontsize=14)
            ax.legend(frameon=True, fontsize=12)
    fig.suptitle("LR transfer: do the minima line up across model sizes?\n"
                 "dotted verticals = each cell's best lr_multiplier", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        p = OUTDIR / f"hp_transfer_local.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"Saved {p}")

    print("\nbest lr_multiplier per cell:")
    for lname, cells in (("depth", DEPTH_LADDER), ("width", WIDTH_LADDER)):
        for fixed in (False, True):
            out = []
            for (L, N) in cells:
                sel = sorted([r for r in rows if r["ladder"] == lname and r["fixed"] == fixed
                              and r["depth"] == L and r["width"] == N], key=lambda r: r["lrmult"])
                ys = [r["loss"] for r in sel]
                out.append("nan" if np.all(np.isnan(ys))
                           else f"{sel[int(np.nanargmin(ys))]['lrmult']:g}")
            print(f"  {lname:5s} {'fixed' if fixed else 'as-is':6s}: " + "  ".join(
                f"{('d' + str(L)) if lname == 'depth' else ('w' + str(N))}={o}"
                for (L, N), o in zip(cells, out)))


if __name__ == "__main__":
    main()
