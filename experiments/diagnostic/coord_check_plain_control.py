"""Positive control: does CompleteP transfer across depth once the architecture
matches the one it was derived for?

Our model adds three residual-stream paths that a plain pre-LN transformer -- the
architecture CompleteP's Table 1 assumes -- does not have:
    ve_projs        a learned projection of x0 into the attention values
    x0_lambdas*x0   the raw embedding re-injected at EVERY layer
    skip_weights    the U-Net skips
Only the first is disableable in the codebase (--no-ve-projs). This strips all
three (zeroing AND freezing the two scalars, so training cannot reintroduce them)
and re-measures the depth ladder.

Result (w64, depths 4/8/16/32, residual-stream RMS averaged over layers):

    full arch, depth fix ON     step 0: 1.17    step 5: 1.51
    PLAIN                       step 0: 1.00    step 5: 1.02

So the CompleteP core -- width-aware init, forward multipliers, and L_base/L on
the attn/mlp branches -- is implemented CORRECTLY: it delivers exact depth
invariance, and holds it through training. The residual imperfection in the full
architecture is caused entirely by the extra paths, and L_base/L is evidently not
the exactly-right factor for them (the full-arch curve is now slightly DECREASING
with depth: 2.124 -> 1.810, i.e. mildly over-corrected).

Runs on MPS/CPU in seconds.
"""
from __future__ import annotations
import sys, types
sys.argv = [sys.argv[0]]
for _m in ("wandb", "tiktoken"):
    try: __import__(_m)
    except ImportError: sys.modules[_m] = types.ModuleType(_m)
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "unlimited"))
import train as T  # noqa: E402

BASE_D, BASE_W, HEAD_DIM, VOCAB, SEQ = 4, 64, 16, 256, 64


def resid_rms(depth, width, mode, steps=5, device="mps"):
    cfg = T.GPTConfig(sequence_len=SEQ, vocab_size=VOCAB, n_layer=depth, n_embd=width,
                      n_head=width // HEAD_DIM, n_kv_head=width // HEAD_DIM,
                      completep=True, no_ve_projs=True, mup_base_width=BASE_W,
                      mup_base_depth=BASE_D, mup_base_head_dim=HEAD_DIM, dropout=0.0)
    torch.manual_seed(0)
    m = T.GPT(cfg).to(device)
    m.init_weights(convert_embed=False)
    if mode == "plain":
        with torch.no_grad():
            m.x0_lambdas.zero_(); m.skip_weights.zero_()
        m.x0_lambdas.requires_grad_(False); m.skip_weights.requires_grad_(False)
    m.to(device)
    nd = dict(m.named_parameters()); M = 0.25
    mats = [p for n, p in nd.items() if p.ndim >= 2 and "wte" not in n and "lm_head" not in n]
    grps = [dict(params=mats, lr=0.04 * M, betas=(0.9, 0.95)),
            dict(params=[nd["transformer.wte.weight"]], lr=0.15 * M, betas=(0.9, 0.95)),
            dict(params=[nd["lm_head.weight"]], lr=0.002 * M, betas=(0.9, 0.95)),
            dict(params=[m.resid_lambdas], lr=0.1 * M * 0.01, betas=(0.9, 0.95))]
    if mode != "plain":
        grps += [dict(params=[m.x0_lambdas], lr=0.1 * M, betas=(0.96, 0.95)),
                 dict(params=[m.skip_weights], lr=0.1 * M * 0.01, betas=(0.9, 0.95))]
    opt = torch.optim.AdamW(grps, eps=1e-10, weight_decay=0.0)
    got = {}
    hs = [b.register_forward_hook(
        lambda _m, _i, o, i=i: got.__setitem__(
            i, (o[0] if isinstance(o, tuple) else o).detach().float().pow(2).mean().sqrt().item()))
        for i, b in enumerate(m.transformer.h)]
    g = torch.Generator().manual_seed(7); res = {}
    for st in range(steps + 1):
        ids = torch.randint(0, VOCAB, (4, SEQ + 1), generator=g).to(device)
        got.clear()
        loss = m(ids[:, :-1].contiguous(), ids[:, 1:].contiguous())
        res[st] = float(np.mean([got[i] for i in sorted(got)]))
        if st < steps:
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    for h in hs: h.remove()
    return res


if __name__ == "__main__":
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    DEPTHS = (4, 8, 16, 32)
    print(f"device {dev} — residual RMS, depth ladder at w64 (ve_projs OFF throughout)")
    for mode, lbl in (("full", "full arch (x0 + U-Net, depth fix ON)"),
                      ("plain", "PLAIN (x0=0, skips=0, frozen)")):
        r = {L: resid_rms(L, 64, mode, device=dev) for L in DEPTHS}
        for st in (0, 5):
            v = np.array([r[L][st] for L in DEPTHS])
            print(f"  {lbl:<38} step {st}: " + " ".join(f"{x:6.3f}" for x in v)
                  + f"   max/min={v.max()/v.min():5.2f}"
                  + ("   <-- FLAT" if v.max() / v.min() < 1.15 else ""))
