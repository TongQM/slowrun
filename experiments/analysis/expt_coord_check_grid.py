"""Does CompleteP transfer across a full depth x width grid — before and after
extending the depth scaling to every residual path (`--completep-resid-paths`)?

Runs locally on Apple MPS (or CPU/CUDA) with small models. A parameterization
defect is scale-free, so a 4x4 grid of toy models reproduces it exactly and in
seconds, which is what makes a real grid affordable at all -- the equivalent on
the cluster was a 72-task array.

Measured, at initialization and after a few AdamW steps:
    residual stream RMS, averaged over layers, for every (L, N) cell.

Under a correct parameterization this is invariant across the WHOLE grid, so the
diagnostic is simply: how far does max/min depart from 1?

Outputs:
  experiments/figures/08_coord_check/coord_check_grid.{pdf,png}
  experiments/figures/08_coord_check/coord_check_grid.csv
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
DEPTHS = (4, 8, 16, 32)
WIDTHS = (64, 128, 256, 512)
HEAD_DIM, BASE_D, BASE_W, VOCAB, SEQ = 16, 4, 64, 512, 64


def measure(depth, width, fix, steps, device, seed=0):
    cfg = T.GPTConfig(sequence_len=SEQ, vocab_size=VOCAB, n_layer=depth, n_embd=width,
                      n_head=width // HEAD_DIM, n_kv_head=width // HEAD_DIM,
                      completep=True, no_ve_projs=True,
                      mup_base_width=BASE_W, mup_base_depth=BASE_D,
                      mup_base_head_dim=HEAD_DIM, dropout=0.0)
    torch.manual_seed(seed)
    model = T.GPT(cfg).to(device)
    model.init_weights(convert_embed=False)
    if not fix:
        model.resid_path_scale = 1.0   # reproduce the pre-2026-09 behaviour
    model.to(device)

    # Mirror train.py's per-group LRs exactly (lr_multiplier 0.25 applied):
    #   matrix 0.04, embed 0.15, unembed 0.002, scalar 0.1; resid/skip get
    #   SCALAR_LR*0.01, x0 gets the full SCALAR_LR with betas (0.96, 0.95).
    M = 0.25
    # LRs are NOT size-scaled -- Table 1 gives bias/scalar LRs m_L^(alpha-1) = 1 at
    # alpha=1, and that is what train.py already does. The fix lives entirely in the
    # forward pass as a constant factor on the extra residual paths.
    named = dict(model.named_parameters())
    mats = [p for n, p in named.items()
            if p.ndim >= 2 and "wte" not in n and "lm_head" not in n]
    opt = torch.optim.AdamW(
        [dict(params=mats, lr=0.04 * M, betas=(0.9, 0.95)),
         dict(params=[named["transformer.wte.weight"]], lr=0.15 * M, betas=(0.9, 0.95)),
         dict(params=[named["lm_head.weight"]], lr=0.002 * M, betas=(0.9, 0.95)),
         dict(params=[model.resid_lambdas], lr=0.1 * M * 0.01, betas=(0.9, 0.95)),
         dict(params=[model.x0_lambdas], lr=0.1 * M, betas=(0.96, 0.95)),
         dict(params=[model.skip_weights], lr=0.1 * M * 0.01, betas=(0.9, 0.95))],
        eps=1e-10, weight_decay=0.0)

    got: dict = {}

    def mk(i):
        def hook(_m, _i, out):
            o = out[0] if isinstance(out, tuple) else out
            got[i] = o.detach().float().pow(2).mean().sqrt().item()
        return hook

    hs = [b.register_forward_hook(mk(i)) for i, b in enumerate(model.transformer.h)]
    g = torch.Generator().manual_seed(seed)
    rows = []
    for step in range(steps + 1):
        ids = torch.randint(0, VOCAB, (2, SEQ + 1), generator=g).to(device)
        got.clear()
        loss = model(ids[:, :-1].contiguous(), ids[:, 1:].contiguous())
        rows.append(dict(depth=depth, width=width, step=step,
                         rms=float(np.mean([got[i] for i in sorted(got)]))))
        if step < steps:
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
    for h in hs:
        h.remove()
    del model, opt
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--steps", type=int, default=5)
    a = ap.parse_args(_ARGS)
    dev = a.device
    if dev == "mps" and not torch.backends.mps.is_available():
        dev = "cpu"
    print(f"device: {dev}")

    data = []
    for fix in (False, True):
        for L in DEPTHS:
            for N in WIDTHS:
                for r in measure(L, N, fix, a.steps, dev):
                    r["fix"] = fix
                    data.append(r)
            print(f"  {'fixed ' if fix else 'as-is '} L={L} done", flush=True)

    def grid(fix, step):
        M = np.full((len(DEPTHS), len(WIDTHS)), np.nan)
        for r in data:
            if r["fix"] == fix and r["step"] == step:
                M[DEPTHS.index(r["depth"]), WIDTHS.index(r["width"])] = r["rms"]
        return M

    sns.set(font_scale=1.3)
    sns.set_style("white")
    plt.rcParams["axes.linewidth"] = 3.0
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    for row, step in enumerate((0, a.steps)):
        vmin = min(np.nanmin(grid(f, step)) for f in (False, True))
        vmax = max(np.nanmax(grid(f, step)) for f in (False, True))
        for col, fix in enumerate((False, True)):
            ax = axes[row, col]
            M = grid(fix, step)
            im = ax.imshow(M, cmap="cool", vmin=vmin, vmax=vmax, aspect="auto")
            for i in range(len(DEPTHS)):
                for j in range(len(WIDTHS)):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=13, color="0.15")
            ax.set_xticks(range(len(WIDTHS)), [f"w{w}" for w in WIDTHS])
            ax.set_yticks(range(len(DEPTHS)), [f"d{d}" for d in DEPTHS])
            ratio = np.nanmax(M) / np.nanmin(M)
            tag = "with fix" if fix else "as-is"
            ax.set_title(f"{tag} — step {step}\nmax/min = {ratio:.2f}"
                         + ("   ✓ flat" if ratio < 1.3 else ""), fontsize=14)
            if col == 0:
                ax.set_ylabel("depth")
            ax.set_xlabel("width")
            fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Residual-stream RMS across a depth x width grid\n"
                 "flat = hyperparameters transfer; the fix extends $L_{base}/L$ to every residual path",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUTDIR / f"coord_check_grid.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"Saved {p}")

    csv = OUTDIR / "coord_check_grid.csv"
    with open(csv, "w") as fh:
        fh.write("variant,depth,width,step,resid_rms\n")
        for r in data:
            fh.write(f"{'fixed' if r['fix'] else 'as-is'},{r['depth']},{r['width']},"
                     f"{r['step']},{r['rms']:.6f}\n")
    print(f"Saved {csv}")

    print("\nmax/min of residual RMS over the whole 4x4 grid:")
    for step in (0, a.steps):
        for fix in (False, True):
            M = grid(fix, step)
            tag = "fixed" if fix else "as-is"
            print(f"  step {step:<2} {tag:<6} max/min = {np.nanmax(M)/np.nanmin(M):6.2f}")


if __name__ == "__main__":
    main()
