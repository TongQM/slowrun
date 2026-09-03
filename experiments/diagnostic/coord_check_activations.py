"""Activation coordinate check for CompleteP (the muP-style diagnostic).

Motivation
----------
Our evidence that CompleteP transfers is currently indirect: we infer it from
where the *loss* is minimized as a function of an HP. That is weak — it needs a
full 40-epoch run per point, the optima are shallow, every point is one seed,
and the existing LR-transfer check (figures/08_coord_check) has every size
pinned at the left edge of its lr grid, so it cannot establish transfer at all.
It also scaled width and depth together on a w/L=64 ladder, so it could not
separate the two axes.

This measures the parameterization's *defining* invariant directly:

    under a correct parameterization the coordinate-wise magnitude of the
    activations at every layer is Theta(1) -- independent of width and depth --
    at initialization AND after each of the first few optimizer steps.

If activation RMS is flat across a ladder, that axis transfers. If it grows or
shrinks systematically, it does not, and the per-layer/per-step pattern says
where. Costs ~10 steps per cell instead of 40 epochs, and needs no HP grid, so
there is no boundary to get pinned against.

Two independent ladders (the point of the exercise -- vary ONE axis at a time):
    width ladder : d12 x w{384, 768, 1152, 1536}
    depth ladder : w768 x d{6, 12, 24, 48}

What is measured, per (cell, step, layer):
    resid   RMS of the residual stream leaving each Block
    attn    RMS of that Block's attention branch output (pre-residual-add)
    mlp     RMS of that Block's MLP branch output (pre-residual-add)

The attn/mlp branch outputs are the interesting ones for depth: CompleteP's
depth scaling is exactly the `depth_scale = L_base/L` factor applied to those
two branches before the residual add (train.py Block.forward), so if the depth
path is wrong, it shows up there first.

Optimizer note: this uses torch.optim.AdamW rather than train.py's custom fused
distributed optimizer, with the same per-group learning rates and weight decay.
The update rule is the same; the custom one exists for ZeRO sharding, which is
irrelevant at one GPU and would only add distributed machinery to a diagnostic.
What is under test is the parameterization (init + multipliers + depth_scale),
not the optimizer implementation.

Usage (on PSC, one GPU, a few minutes):
    python experiments/diagnostic/coord_check_activations.py \
        --out experiments/diagnostic/coord_check_activations.json
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from collections import defaultdict

# train.py parses argv at import time; give it a benign command line before
# importing, and keep our own flags out of its way.
_MY_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]

import types  # noqa: E402

# train.py imports wandb/tiktoken at module scope for the training path. The model
# classes need neither, and stubbing them keeps this diagnostic runnable on a
# laptop with only torch installed — which is the point: a parameterization bug
# is scale-free, so it reproduces on a 2-layer CPU model.
for _m in ("wandb", "tiktoken"):
    if _m not in sys.modules:
        try:
            __import__(_m)
        except ImportError:
            sys.modules[_m] = types.ModuleType(_m)

import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "unlimited"))
import train as T  # noqa: E402


def build(depth: int, width: int, device: str, *, head_dim: int, base_w: int,
          base_d: int, vocab: int, seqlen: int):
    """A CompleteP model at (depth, width). head_dim is held fixed across the
    ladder, exactly as every production grid does (n_head = width/head_dim)."""
    cfg = T.GPTConfig(
        sequence_len=seqlen,
        vocab_size=vocab,
        n_layer=depth,
        n_embd=width,
        n_head=width // head_dim,
        n_kv_head=width // head_dim,
        completep=True,
        no_ve_projs=True,
        mup_base_width=base_w,
        mup_base_depth=base_d,
        mup_base_head_dim=head_dim,
        dropout=0.0,
    )
    if device == "cpu":
        model = T.GPT(cfg).to(device)
    else:
        with torch.device("meta"):
            model = T.GPT(cfg)
        model.to_empty(device=device)
    model.init_weights(convert_embed=(device != "cpu"))
    return model, cfg


def attach_hooks(model, store: dict):
    """Record RMS of each Block's residual output and its attn/mlp branch outputs."""
    handles = []

    def mk(kind, idx):
        def hook(_mod, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            store[(kind, idx)] = o.detach().float().pow(2).mean().sqrt().item()
        return hook

    for i, block in enumerate(model.transformer.h):
        handles.append(block.register_forward_hook(mk("resid", i)))
        handles.append(block.attn.register_forward_hook(mk("attn", i)))
        handles.append(block.mlp.register_forward_hook(mk("mlp", i)))
    return handles


def param_groups(model, cfg):
    """Mirror train.py's LR split: matrix / embedding / unembedding / scalars."""
    matrix, embed, unembed, scalar = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "wte" in name:
            embed.append(p)
        elif "lm_head" in name:
            unembed.append(p)
        elif p.ndim >= 2:
            matrix.append(p)
        else:
            scalar.append(p)
    mult = 0.25  # train.py default lr_multiplier
    groups = []
    for ps, lr in ((matrix, 0.04 * mult), (embed, 0.3 * mult),
                   (unembed, 0.004 * mult), (scalar, 0.1 * mult)):
        if ps:
            groups.append(dict(params=ps, lr=lr))
    return groups


def run_cell(depth, width, loader, steps, device, weight_decay, **bk):
    model, cfg = build(depth, width, device, **bk)
    opt = torch.optim.AdamW(param_groups(model, cfg), betas=(0.9, 0.95),
                            eps=1e-10, weight_decay=weight_decay)
    store: dict = {}
    handles = attach_hooks(model, store)
    rows = []
    it = iter(loader)
    for step in range(steps + 1):          # step 0 = at initialization
        x, y = next(it)
        store.clear()
        if device == "cpu":
            loss = model(x, y)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, y)
        for (kind, idx), rms in store.items():
            rows.append(dict(depth=depth, width=width, step=step,
                             kind=kind, layer=idx, rms=rms))
        if step < steps:
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
    for h in handles:
        h.remove()
    del model, opt
    torch.cuda.empty_cache()
    return rows


class SyntheticLoader:
    """Random tokens. A coordinate check measures activation scale, not learning,
    so real text is unnecessary — and this lets the whole diagnostic run on CPU."""

    def __init__(self, vocab, B, T, seed=0):
        self.g = torch.Generator().manual_seed(seed)
        self.vocab, self.B, self.T = vocab, B, T

    def __iter__(self):
        return self

    def __next__(self):
        ids = torch.randint(0, self.vocab, (self.B, self.T + 1), generator=self.g)
        return ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


class RealLoader:
    """Adapter: train.DataLoader yields (x, y, epoch); we want (x, y)."""

    def __init__(self, inner):
        self.inner = iter(inner)

    def __iter__(self):
        return self

    def __next__(self):
        x, y, _ = next(self.inner)
        return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--synthetic", action="store_true",
                    help="random tokens instead of FineWeb (implied on CPU)")
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--base-width", type=int, default=768)
    ap.add_argument("--base-depth", type=int, default=12)
    ap.add_argument("--vocab", type=int, default=50257)
    ap.add_argument("--seqlen", type=int, default=None)
    ap.add_argument("--width-ladder", default="12:384,12:768,12:1152,12:1536")
    ap.add_argument("--depth-ladder", default="6:768,12:768,24:768,48:768")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="0 isolates the parameterization; rerun with 0.15 to see "
                         "whether decay itself breaks the invariance")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=os.path.join(REPO, "experiments/diagnostic/coord_check_activations.json"))
    a = ap.parse_args(_MY_ARGS)

    device = a.device
    synthetic = a.synthetic or device == "cpu"
    seqlen = a.seqlen or (128 if synthetic else T.MAX_SEQ_LEN)
    data = a.data or os.path.join(T.DATA_DIR, "fineweb_train.pt")

    def parse(spec):
        return [tuple(int(v) for v in c.split(":")) for c in spec.split(",")]

    ladders = {"width": parse(a.width_ladder), "depth": parse(a.depth_ladder)}
    bk = dict(head_dim=a.head_dim, base_w=a.base_width, base_d=a.base_depth,
              vocab=a.vocab, seqlen=seqlen)

    out = []
    for name, cells in ladders.items():
        for depth, width in cells:
            if synthetic:
                loader = SyntheticLoader(a.vocab, a.batch, seqlen, seed=42)
            else:
                loader = RealLoader(T.DataLoader(data, a.batch, seqlen,
                                                 device=device, seed=42, quiet=True))
            rows = run_cell(depth, width, loader, a.steps, device, a.weight_decay, **bk)
            for r in rows:
                r["ladder"] = name
            out.extend(rows)
            print(f"[{name}] d{depth}_w{width}: {len(rows)} measurements", flush=True)

    with open(a.out, "w") as fh:
        json.dump(dict(weight_decay=a.weight_decay, steps=a.steps,
                       batch=a.batch, rows=out), fh)
    print(f"wrote {a.out}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
