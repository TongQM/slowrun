"""Search for a depth-transfer fix for the x0-injection and U-Net skip paths
that is STRICTLY a reparametrization: it changes only the functional form of
how each path's coefficient enters the residual stream, and never touches any
learning rate, weight decay, or beta. (Companion to expt_coord_check_depth.py,
which established that the currently-committed fix -- resid_path_scale =
L_base/L, a constant on top of the raw learnable scalars -- repairs
INITIALIZATION but drifts under training: step-5 residual RMS max/min across
depth 5.45 -> 1.51-1.63, not the ~1.0 a correct parameterization should give.)

Why the current fix drifts under training
------------------------------------------
AdamW's update is near scale-invariant to gradient magnitude once its second
moment estimate warms up: multiplying a parameter's gradient by a constant
barely changes how far AdamW moves the RAW parameter per step. So a constant
forward multiplier (`resid_path_scale`) fixes the value at init, but can't
prevent the raw scalar itself from drifting by a similar ABSOLUTE amount at
every depth during training -- and an absolute drift, applied on top of a
depth-scaled base value, is a much larger RELATIVE (and hence RMS-visible)
perturbation at large depth than at small depth.

The fix: log/exp reparametrization
------------------------------------
Instead of learning the coefficient c directly, learn an unconstrained scalar
u with c = c0 * exp(u), u initialized at 0 (so c starts at exactly c0, same as
today). AdamW's near-constant ABSOLUTE step on u becomes a near-constant
RELATIVE (percentage) step on c once exponentiated -- which is depth-invariant
by construction, since a percentage drift doesn't care what the (depth-scaled)
base value c0 happens to be. Implemented via
`torch.nn.utils.parametrize.register_parametrization`, so `model.x0_lambdas`
and `model.skip_weights` remain drop-in-compatible with the rest of the
model's forward code (which still just reads them as before) -- only how the
optimizer's updates translate into their value changes.

For skip_weights specifically, exponent 1.0 (matching x0's exponent, and
`depth_scale`) turned out NOT optimal even at init when isolated -- an exponent
sweep (part A below) found beta=0.75 is close to flat across the whole 40-step
trajectory (mean spread 1.086 at BOTH step 0 and step 40), while beta=1.0
drifts 1.14 -> 1.20. beta<0.75 undercorrects, beta>0.85 overcorrects and drifts
back up. This is plausible: a skip addition is a full copy of another layer's
already-large activation, not a small residual branch output, so it needn't
obey the exact same depth law as the attn/mlp branches.

Result (full model, both paths active together, mean over seeds {7,17,27}):

  variant                                    step 0    step 40
  broken (no depth correction at all)         5.45     3.9-4.1   (erratic)
  linear (currently committed fix)            1.17     1.33-1.37 (steady drift)
  combined (log x0 @ exp 1.0 + log skip @ exp 0.75)
                                               1.07     1.06-1.10 (flat)

All three use IDENTICAL learning rates, weight decay, and betas throughout --
matrix 0.04*M, embed 0.15*M, unembed 0.002*M, resid_lambdas 0.1*M*0.01,
x0_lambdas 0.1*M (betas 0.96/0.95), skip_weights 0.1*M*0.01 -- the same values
train.py already uses with M = lr_multiplier = 0.25.

This is a FINDING, not yet a change to unlimited/train.py: applying it to the
real training pipeline needs `setup_optimizer()`'s param-group construction
updated to reference the parametrization's raw `.original` tensor rather than
`self.x0_lambdas`/`self.skip_weights` directly (those become non-leaf computed
tensors once parametrized, which a leaf-tensor optimizer group cannot accept),
across all three optimizer branches (adamw/hybrid/muon) and interacting with
the custom ZeRO-sharded fused-AdamW kernel -- untested here, deliberately kept
out of the production forward/optimizer path pending a decision on that.

Runs on MPS/CPU with tiny models; a parameterization property is scale-free.

Outputs:
  experiments/figures/08_coord_check/reparam_search_depth.{pdf,png}
  experiments/figures/08_coord_check/reparam_search_depth.csv
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
import torch.nn as nn
import torch.nn.utils.parametrize as P
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "unlimited"))
import train as T  # noqa: E402

OUTDIR = REPO / "experiments" / "figures" / "08_coord_check"
BD, BW, HD, VOCAB, SEQ = 4, 64, 16, 256, 64
DEPTHS = (4, 8, 16, 32)
SEEDS = (7, 17, 27)
STEPS = 40
CHECK = (0, 5, 10, 20, 30, 40)
BEST_SKIP_BETA = 0.75   # from part (A) below


class ExpReparam(nn.Module):
    """c = c0 * exp(raw). Registration is a no-op on the value when the
    parameter already equals c0 (right_inverse(c0) = log(1) = 0)."""
    def __init__(self, c0: float):
        super().__init__()
        self.c0 = c0

    def forward(self, raw):
        return self.c0 * torch.exp(raw)

    def right_inverse(self, value):
        return torch.log(value / self.c0)


def bigram_source(seed=1234):
    """Fixed random bigram chain -- genuine learnable structure, unlike uniform
    random tokens (which floor the loss at log(V) and drive parameter drift by
    pure gradient noise rather than anything training-like)."""
    g = torch.Generator().manual_seed(seed)
    Pm = torch.softmax(torch.randn(VOCAB, VOCAB, generator=g) * 3.0, dim=-1)
    def sample(n, gen):
        idx = torch.randint(0, VOCAB, (n, 1), generator=gen)
        out = [idx]
        for _ in range(SEQ):
            out.append(torch.multinomial(Pm[out[-1].squeeze(-1)], 1, generator=gen))
        return torch.cat(out, dim=1)
    return sample


def base_model(L, device, model_seed=0):
    cfg = T.GPTConfig(sequence_len=SEQ, vocab_size=VOCAB, n_layer=L, n_embd=BW,
                      n_head=BW // HD, n_kv_head=BW // HD, completep=True,
                      no_ve_projs=True, mup_base_width=BW, mup_base_depth=BD,
                      mup_base_head_dim=HD, dropout=0.0)
    torch.manual_seed(model_seed)
    m = T.GPT(cfg).to(device)
    m.init_weights(convert_embed=False)
    m.to(device)
    return m


def prod_groups(m, extra, M=0.25):
    nd = dict(m.named_parameters())
    mats = [p for n, p in nd.items()
            if p.ndim >= 2 and "wte" not in n and "lm_head" not in n and "parametrizations" not in n]
    return [dict(params=mats, lr=0.04 * M, betas=(0.9, 0.95)),
            dict(params=[nd["transformer.wte.weight"]], lr=0.15 * M, betas=(0.9, 0.95)),
            dict(params=[nd["lm_head.weight"]], lr=0.002 * M, betas=(0.9, 0.95)),
            dict(params=[m.resid_lambdas], lr=0.1 * M * 0.01, betas=(0.9, 0.95))] + extra


def _run(m, opt, steps, sample_fn, data_seed, device):
    got = {}
    hs = [b.register_forward_hook(
        lambda _m, _i, o, i=i: got.__setitem__(
            i, (o[0] if isinstance(o, tuple) else o).detach().float().pow(2).mean().sqrt().item()))
        for i, b in enumerate(m.transformer.h)]
    g = torch.Generator().manual_seed(data_seed)
    rms = {}
    for st in range(steps + 1):
        ids = sample_fn(4, g).to(device)
        got.clear()
        loss = m(ids[:, :-1].contiguous(), ids[:, 1:].contiguous())
        rms[st] = float(np.mean([got[i] for i in sorted(got)]))
        if st < steps:
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
    for h in hs:
        h.remove()
    return rms


def variant_skip_beta(L, beta, steps, sample_fn, data_seed, device, model_seed=0):
    """Isolated (x0 zeroed) constant-exponent test, for the beta sweep."""
    m = base_model(L, device, model_seed)
    with torch.no_grad():
        m.x0_lambdas.zero_()
    m.x0_lambdas.requires_grad_(False)
    m.resid_path_scale = (BD / L) ** beta
    grps = prod_groups(m, [dict(params=[m.skip_weights], lr=0.1 * 0.25 * 0.01, betas=(0.9, 0.95))])
    opt = torch.optim.AdamW(grps, eps=1e-10, weight_decay=0.0)
    return _run(m, opt, steps, sample_fn, data_seed, device)


def variant_broken(L, steps, sample_fn, data_seed, device, model_seed=0):
    m = base_model(L, device, model_seed)
    m.resid_path_scale = 1.0
    grps = prod_groups(m, [
        dict(params=[m.x0_lambdas], lr=0.1 * 0.25, betas=(0.96, 0.95)),
        dict(params=[m.skip_weights], lr=0.1 * 0.25 * 0.01, betas=(0.9, 0.95))])
    opt = torch.optim.AdamW(grps, eps=1e-10, weight_decay=0.0)
    return _run(m, opt, steps, sample_fn, data_seed, device)


def variant_linear(L, steps, sample_fn, data_seed, device, model_seed=0):
    """The fix currently committed in unlimited/train.py (c660582)."""
    m = base_model(L, device, model_seed)
    m.resid_path_scale = BD / L
    grps = prod_groups(m, [
        dict(params=[m.x0_lambdas], lr=0.1 * 0.25, betas=(0.96, 0.95)),
        dict(params=[m.skip_weights], lr=0.1 * 0.25 * 0.01, betas=(0.9, 0.95))])
    opt = torch.optim.AdamW(grps, eps=1e-10, weight_decay=0.0)
    return _run(m, opt, steps, sample_fn, data_seed, device)


def variant_combined(L, skip_beta, steps, sample_fn, data_seed, device, model_seed=0):
    """x0: log/exp reparam, exponent 1.0. Skip: log/exp reparam, exponent
    skip_beta. Each path's depth factor is baked into its OWN c0 -- a single
    shared resid_path_scale can't give the two paths different exponents.
    right_inverse would otherwise UNDO a depth-scaled c0 relative to the old
    (unscaled 0.1/1.0) parameter value, so the raw value is pre-set to the
    depth-scaled target before registering: registration is then a true no-op
    on the effective value, changing only how it moves from there on."""
    m = base_model(L, device, model_seed)
    m.resid_path_scale = 1.0

    x0_c0 = 0.1 * (BD / L) ** 1.0
    with torch.no_grad():
        m.x0_lambdas.fill_(x0_c0)
    P.register_parametrization(m, "x0_lambdas", ExpReparam(x0_c0))

    skip_c0 = 1.0 * (BD / L) ** skip_beta
    with torch.no_grad():
        m.skip_weights.fill_(skip_c0)
    P.register_parametrization(m, "skip_weights", ExpReparam(skip_c0))

    nd = dict(m.named_parameters())
    grps = prod_groups(m, [
        dict(params=[nd["parametrizations.x0_lambdas.original"]], lr=0.1 * 0.25, betas=(0.96, 0.95)),
        dict(params=[nd["parametrizations.skip_weights.original"]], lr=0.1 * 0.25 * 0.01, betas=(0.9, 0.95))])
    opt = torch.optim.AdamW(grps, eps=1e-10, weight_decay=0.0)
    return _run(m, opt, steps, sample_fn, data_seed, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    a = ap.parse_args(_ARGS)
    device = a.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    print(f"device: {device}\n")

    sample = bigram_source()

    print("=" * 74)
    print("(A) U-Net-skip exponent sweep, isolated, bigram data, 3 seeds, 40 steps")
    print("=" * 74)
    beta_rows = []
    for beta in (0.5, 0.65, 0.75, 0.85, 1.0):
        s0, s40 = [], []
        for ds in SEEDS:
            r = {L: variant_skip_beta(L, beta, STEPS, sample, ds, device) for L in DEPTHS}
            v0 = np.array([r[L][0] for L in DEPTHS])
            v40 = np.array([r[L][STEPS] for L in DEPTHS])
            s0.append(v0.max() / v0.min())
            s40.append(v40.max() / v40.min())
        beta_rows.append((beta, np.mean(s0), np.mean(s40)))
        print(f"  beta={beta:.2f}  mean st0={np.mean(s0):.3f}  mean st40={np.mean(s40):.3f}")

    print()
    print("=" * 74)
    print(f"(B) full model: broken vs linear (committed) vs combined (skip_beta={BEST_SKIP_BETA})")
    print("=" * 74)
    variants = {
        "broken": lambda L, ds: variant_broken(L, STEPS, sample, ds, device),
        "linear": lambda L, ds: variant_linear(L, STEPS, sample, ds, device),
        "combined": lambda L, ds: variant_combined(L, BEST_SKIP_BETA, STEPS, sample, ds, device),
    }
    full_rows = []  # (variant, seed, step, spread)
    for name, fn in variants.items():
        print(f"\n--- {name} ---")
        for ds in SEEDS:
            r = {L: fn(L, ds) for L in DEPTHS}
            line = f"  seed={ds:<3}"
            for st in CHECK:
                v = np.array([r[L][st] for L in DEPTHS])
                spread = v.max() / v.min()
                full_rows.append((name, ds, st, spread))
                line += f" st{st}:{spread:5.2f}"
            print(line)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    csv = OUTDIR / "reparam_search_depth.csv"
    with open(csv, "w") as fh:
        fh.write("section,beta_or_variant,seed,step,value\n")
        for beta, s0, s40 in beta_rows:
            fh.write(f"beta_sweep,{beta},,0,{s0:.6f}\n")
            fh.write(f"beta_sweep,{beta},,{STEPS},{s40:.6f}\n")
        for name, ds, st, spread in full_rows:
            fh.write(f"full_model,{name},{ds},{st},{spread:.6f}\n")
    print(f"\nSaved {csv}")

    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")
    plt.rcParams["axes.linewidth"] = 3.0
    plt.rcParams["grid.alpha"] = 0.25
    colors = sns.color_palette("cool", 3)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    ax = axes[0]
    betas = [b for b, _, _ in beta_rows]
    ax.plot(betas, [s0 for _, s0, _ in beta_rows], "o-", color=colors[0], lw=3.0, ms=9, label="step 0")
    ax.plot(betas, [s40 for _, _, s40 in beta_rows], "s-", color=colors[2], lw=3.0, ms=9, label="step 40")
    ax.axhline(1.0, color="0.3", ls="--", lw=2.0)
    ax.axvline(BEST_SKIP_BETA, color="0.5", ls=":", lw=2.0)
    ax.set_xlabel(r"U-Net skip exponent $\beta$  (scale $= (L_{base}/L)^\beta$)")
    ax.set_ylabel("residual RMS max/min across depth")
    ax.set_title("(A) U-Net-skip exponent sweep\n(isolated, mean over 3 seeds)", fontsize=14)
    ax.legend(frameon=True)

    ax = axes[1]
    for c, name in zip(colors, ("broken", "linear", "combined")):
        for ds in SEEDS:
            xs = [st for (n, s, st, v) in full_rows if n == name and s == ds]
            ys = [v for (n, s, st, v) in full_rows if n == name and s == ds]
            ax.plot(xs, ys, "-o", color=c, lw=2.0, ms=5, alpha=0.55)
        ax.plot([], [], "-o", color=c, lw=3.0, ms=8, label=name)
    ax.axhline(1.0, color="0.3", ls="--", lw=2.0)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("residual RMS max/min across depth")
    ax.set_title("(B) full model: 3 seeds per variant", fontsize=14)
    ax.legend(frameon=True, loc="upper left")

    fig.suptitle("Depth transfer without touching LR/WD: log/exp reparametrization "
                 "of the x0 and U-Net residual paths", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        p = OUTDIR / f"reparam_search_depth.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=300)
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
