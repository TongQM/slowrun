"""Sharded worker for the LR-transfer + performance sweep.

One grid answers BOTH questions:
  transferability -> does argmin_lr align across depths, per variant?
  performance     -> best-achievable loss, variant vs variant, at each depth
                     (best-vs-best, i.e. each arm at its OWN optimal lr)

Variants, all with IDENTICAL production learning rates:
  logexp : the CURRENT committed reparametrization. x0/skip use
           ExpReparam(c0 = base * (L_base/L)^expo), raw init 0.
           Effective coefficient moves MULTIPLICATIVELY under AdamW.
  linear : faithful reproduction of the SUPERSEDED constant-factor fix
           (c660582). Effective = (L_base/L)^expo * raw, raw init at the
           unscaled base (0.1 / 1.0), so the effective coefficient's drift is
           scaled by the depth factor -- which is what the old forward code
           `resid_path_scale * x0_lambdas` did. NOT the same as baking the
           depth factor into a plain parameter's value (that would leave the
           drift unscaled), hence LinearReparam rather than
           remove_parametrizations(leave_parametrized=True).
  broken : no depth correction on either path (pre-fix behaviour).
"""
import sys, types, argparse, os, csv
sys.argv_backup = sys.argv[:]
_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]
for _m in ("wandb", "tiktoken"):
    try:
        __import__(_m)
    except ImportError:
        sys.modules[_m] = types.ModuleType(_m)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
import numpy as np

torch.set_num_threads(1)   # one thread per process; parallelism comes from sharding

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/Users/miaoyidi/Desktop/pretraining/slowrun/unlimited")
import train as T

BD, BW, HD, S = 4, 64, 16, 64
M = 0.25
X0_BASE, SKIP_BASE = 0.1, 1.0
X0_EXPO, SKIP_EXPO = 1.0, 0.75


class LinearReparam(nn.Module):
    """effective = c * raw  (the superseded constant-forward-multiplier form)."""
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, raw):
        return self.c * raw

    def right_inverse(self, value):
        return value / self.c


def make_task(kind, V):
    if kind == "bigram":
        g = torch.Generator().manual_seed(1234)
        Pm = torch.softmax(torch.randn(V, V, generator=g) * 3.0, dim=-1)
        def s(n, gen):
            idx = torch.randint(0, V, (n, 1), generator=gen)
            out = [idx]
            for _ in range(S):
                out.append(torch.multinomial(Pm[out[-1].squeeze(-1)], 1, generator=gen))
            return torch.cat(out, dim=1)
        return s
    if kind == "copy":
        half = (S + 1 + 1) // 2
        def s(n, gen):
            first = torch.randint(0, V, (n, half), generator=gen)
            return torch.cat([first, first], dim=1)[:, :S + 1].contiguous()
        return s
    raise ValueError(kind)


def build(L, N, variant, V, model_seed):
    cfg = T.GPTConfig(sequence_len=S, vocab_size=V, n_layer=L, n_embd=N,
                      n_head=N // HD, n_kv_head=N // HD, completep=True,
                      no_ve_projs=True, mup_base_width=BW, mup_base_depth=BD,
                      mup_base_head_dim=HD, dropout=0.0)
    torch.manual_seed(model_seed)
    m = T.GPT(cfg)
    m.init_weights(convert_embed=False)
    if variant == "logexp":
        return m                      # current committed behaviour, as-is
    # Rebuild the two paths under a different parametrization.
    ds = BD / L
    for name, base, expo in (("x0_lambdas", X0_BASE, X0_EXPO),
                             ("skip_weights", SKIP_BASE, SKIP_EXPO)):
        P.remove_parametrizations(m, name, leave_parametrized=True)
        if variant == "linear":
            c = ds ** expo
            # register_parametrization calls right_inverse(current) to pick raw, i.e. it
            # PRESERVES the current effective value. To land raw = base (the unscaled
            # 0.1/1.0 the old code used), the value at registration must be c*base --
            # filling with `base` here would instead give raw = base/c and leave the
            # effective value unscaled, silently reproducing "broken" instead of "linear".
            with torch.no_grad():
                getattr(m, name).fill_(c * base)
            P.register_parametrization(m, name, LinearReparam(c))
        elif variant == "broken":
            with torch.no_grad():
                getattr(m, name).fill_(base)          # no depth correction at all
            P.register_parametrization(m, name, LinearReparam(1.0))
        else:
            raise ValueError(variant)
    return m


def groups(m, lm):
    nd = dict(m.named_parameters())
    mats = [p for n, p in nd.items()
            if p.ndim >= 2 and "wte" not in n and "lm_head" not in n and "parametrizations" not in n]
    return [dict(params=mats, lr=0.04 * M * lm, betas=(0.9, 0.95)),
            dict(params=[nd["transformer.wte.weight"]], lr=0.15 * M * lm, betas=(0.9, 0.95)),
            dict(params=[nd["lm_head.weight"]], lr=0.002 * M * lm, betas=(0.9, 0.95)),
            dict(params=[m.resid_lambdas], lr=0.1 * M * lm * 0.01, betas=(0.9, 0.95)),
            dict(params=[nd["parametrizations.x0_lambdas.original"]], lr=0.1 * M * lm, betas=(0.96, 0.95)),
            dict(params=[nd["parametrizations.skip_weights.original"]], lr=0.1 * M * lm * 0.01, betas=(0.9, 0.95))]


def run_one(variant, L, N, lm, seed, steps, task, V, tail_only):
    m = build(L, N, variant, V, model_seed=seed)
    opt = torch.optim.AdamW(groups(m, lm), eps=1e-10, weight_decay=0.0)
    sample = make_task(task, V)
    g = torch.Generator().manual_seed(seed + 7)
    losses = []
    for st in range(steps):
        ids = sample(16, g)
        x, y = ids[:, :-1].contiguous(), ids[:, 1:].contiguous()
        if tail_only:
            logits = m(x)
            h = x.shape[1] // 2
            loss = F.cross_entropy(logits[:, h:].reshape(-1, logits.size(-1)), y[:, h:].reshape(-1))
        else:
            loss = m(x, y)
        if not torch.isfinite(loss):
            return float("nan")
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
    return float(np.mean(losses[-max(20, steps // 10):]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--task", default="bigram")
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--tail-only", action="store_true")
    ap.add_argument("--depths", default="4,8,16,32")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--variants", default="logexp,linear")
    ap.add_argument("--lrs", default="0.5,0.71,1.0,1.41,2.0,2.83,4.0")
    ap.add_argument("--seeds", default="0,1,2,3")
    a = ap.parse_args(_ARGS)

    depths = [int(x) for x in a.depths.split(",")]
    variants = a.variants.split(",")
    lrs = [float(x) for x in a.lrs.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]

    configs = [(v, L, lm, sd) for v in variants for L in depths for lm in lrs for sd in seeds]
    mine = [c for i, c in enumerate(configs) if i % a.nshards == a.shard]

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "depth", "width", "lr_mult", "seed", "loss"])
        for k, (v, L, lm, sd) in enumerate(mine):
            val = run_one(v, L, a.width, lm, sd, a.steps, a.task, a.vocab, a.tail_only)
            w.writerow([v, L, a.width, lm, sd, f"{val:.6f}"])
            fh.flush()
            print(f"[shard {a.shard}] {k+1}/{len(mine)} {v} d{L} lr{lm} s{sd} -> {val:.4f}", flush=True)


if __name__ == "__main__":
    main()
