"""Why CompleteP depth transfer fails in this architecture — the picture.

Measures the residual-stream RMS leaving every layer, at initialization, for a
depth ladder, and contrasts it with the same models after the missing depth
factor is applied.

  (A) as-is             the residual stream grows with depth. In the encoder it
                        climbs a flat +0.1 per layer -- literally `x0_lambdas`
                        re-injecting the same unit-RMS embedding x0 at every
                        layer, so the additions accumulate coherently (prop. L,
                        not sqrt(L)). At the encoder/decoder boundary the U-Net
                        skips fire and each decoder layer dumps in a whole
                        encoder output at weight 1.0.
  (B) fixed             multiply x0_lambdas and skip_weights by L_base/L -- the
                        same factor Block.forward already applies to the attn
                        and mlp branches -- and the curves collapse.
  (C) summary           max/min across the depth ladder, per activation site.
                        attn and mlp are already flat: `depth_scale` works where
                        it is applied. Only the residual stream, fed by the two
                        unscaled paths, is not.

Runs on CPU in seconds -- a parameterization bug is scale-free, so tiny models
reproduce it exactly. wandb/tiktoken are stubbed since only the model classes
are needed.

Outputs:
  experiments/figures/08_coord_check/coord_check_depth_mechanism.{pdf,png}
  experiments/figures/08_coord_check/coord_check_depth_mechanism.csv
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

sys.argv = [sys.argv[0]]
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
DEPTHS = (4, 8, 16, 32)
WIDTH, HEAD_DIM, BASE_D, VOCAB, SEQ = 64, 16, 4, 512, 64


def measure(depth: int, fix: bool):
    """Per-layer RMS at (resid, attn, mlp), at init. `fix` applies L_base/L to
    the two paths that currently receive no depth scaling."""
    cfg = T.GPTConfig(sequence_len=SEQ, vocab_size=VOCAB, n_layer=depth,
                      n_embd=WIDTH, n_head=WIDTH // HEAD_DIM,
                      n_kv_head=WIDTH // HEAD_DIM, completep=True,
                      no_ve_projs=True, mup_base_width=WIDTH,
                      mup_base_depth=BASE_D, mup_base_head_dim=HEAD_DIM,
                      dropout=0.0)
    model = T.GPT(cfg)
    model.init_weights(convert_embed=False)
    if fix:
        with torch.no_grad():
            s = BASE_D / depth
            model.x0_lambdas.mul_(s)
            model.skip_weights.mul_(s)

    got = {"resid": {}, "attn": {}, "mlp": {}}

    def mk(kind, i):
        def hook(_m, _i, out):
            o = out[0] if isinstance(out, tuple) else out
            got[kind][i] = o.detach().float().pow(2).mean().sqrt().item()
        return hook

    hs = []
    for i, blk in enumerate(model.transformer.h):
        hs.append(blk.register_forward_hook(mk("resid", i)))
        hs.append(blk.attn.register_forward_hook(mk("attn", i)))
        hs.append(blk.mlp.register_forward_hook(mk("mlp", i)))

    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, VOCAB, (2, SEQ + 1), generator=g)
    with torch.no_grad():
        model(ids[:, :-1].contiguous(), ids[:, 1:].contiguous())
    for h in hs:
        h.remove()
    return {k: [v[i] for i in sorted(v)] for k, v in got.items()}


def main():
    data = {(d, fix): measure(d, fix) for d in DEPTHS for fix in (False, True)}

    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams["axes.labelsize"] = 22
    plt.rcParams["axes.linewidth"] = 4.0
    plt.rcParams["legend.fontsize"] = 15
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["xtick.labelsize"] = 17
    plt.rcParams["ytick.labelsize"] = 17
    colors = sns.color_palette("cool", len(DEPTHS))

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))

    for ax, fix, title in ((axes[0], False, "(A) as-is — residual stream grows with depth"),
                           (axes[1], True, r"(B) fixed — $\times\,L_{base}/L$ on both paths")):
        for c, d in zip(colors, DEPTHS):
            y = data[(d, fix)]["resid"]
            ax.plot(np.arange(len(y)) / max(len(y) - 1, 1), y, "-o", color=c,
                    lw=3.0, ms=5, label=f"$L$={d}")
        ax.axvline(0.5, color="0.45", ls=":", lw=2.5)
        ax.set_xlabel("relative depth  (layer / $L$)")
        ax.set_title(title, fontsize=15)
        ax.set_ylim(0, 36)
    axes[0].set_ylabel("residual stream RMS")
    axes[0].legend(frameon=True, loc="upper left")
    axes[0].annotate("encoder: $+0.1$ per layer\n($x_0$ re-injected, same vector)",
                     xy=(0.22, 2.0), xytext=(0.04, 15.5), fontsize=13, color="0.25",
                     arrowprops=dict(arrowstyle="->", lw=2, color="0.35"))
    axes[0].annotate("decoder: U-Net skips\nadd a whole encoder output",
                     xy=(0.75, 22), xytext=(0.30, 28), fontsize=13, color="0.25",
                     arrowprops=dict(arrowstyle="->", lw=2, color="0.35"))
    axes[1].annotate("curves collapse", xy=(0.5, 3.2), xytext=(0.18, 14),
                     fontsize=15, color="0.25",
                     arrowprops=dict(arrowstyle="->", lw=2, color="0.35"))

    ax = axes[2]
    kinds = ["attn", "mlp", "resid"]
    labels = ["attn branch", "mlp branch", "residual stream"]
    xs = np.arange(len(kinds))
    for off, fix, lab, col in ((-0.19, False, "as-is", colors[-1]),
                               (0.19, True, r"$\times L_{base}/L$", colors[0])):
        ratios = []
        for k in kinds:
            v = np.array([np.mean(data[(d, fix)][k]) for d in DEPTHS])
            ratios.append(v.max() / v.min())
        ax.bar(xs + off, ratios, width=0.36, color=col, label=lab,
               edgecolor="0.25", lw=2)
        for x, r in zip(xs + off, ratios):
            ax.text(x, r * 1.04, f"{r:.2f}", ha="center", fontsize=13)
    ax.axhline(1.0, color="0.3", ls="--", lw=2.5)
    ax.text(-0.42, 1.03, "perfect transfer", fontsize=12, color="0.3", ha="left", va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_yscale("log")
    ax.set_ylabel("max/min across $L$")
    ax.set_title("(C) which sites transfer across depth?", fontsize=15)
    ax.legend(frameon=True, loc="upper left")

    fig.suptitle(r"CompleteP depth transfer: $depth\_scale=L_{base}/L$ reaches the attn/mlp branches, "
                 r"but not the $x_0$ injection or the U-Net skips", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUTDIR / f"coord_check_depth_mechanism.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"Saved {p}")

    csv = OUTDIR / "coord_check_depth_mechanism.csv"
    with open(csv, "w") as fh:
        fh.write("variant,depth,site,layer,rms\n")
        for (d, fix), rec in data.items():
            for k, vals in rec.items():
                for i, v in enumerate(vals):
                    fh.write(f"{'fixed' if fix else 'as-is'},{d},{k},{i},{v:.6f}\n")
    print(f"Saved {csv}")

    for fix in (False, True):
        tag = "fixed" if fix else "as-is"
        for k in kinds:
            v = np.array([np.mean(data[(d, fix)][k]) for d in DEPTHS])
            print(f"  {tag:6s} {k:6s} max/min = {v.max() / v.min():6.2f}")


if __name__ == "__main__":
    main()
