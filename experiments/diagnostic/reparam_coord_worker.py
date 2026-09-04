"""Comprehensive coordinate check: residual-stream RMS across WIDE depth and
width ladders, over a long horizon, multi-seed, for broken / linear / logexp.

This is the standard muP transfer diagnostic and -- unlike a loss-vs-lr sweep --
it has real power at toy scale, because the effect it measures (activation
scale drifting with depth) is 5.45x for the broken parameterization versus a
~0.02 seed noise floor. The loss-vs-lr route was attempted at length and is
inconclusive here: every task small enough to run locally saturates, so all
depths and all variants reach the same loss and the "optimal lr" is set by the
stability edge rather than by the parameterization.
"""
import sys, types, argparse, os, csv
_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]
for _m in ("wandb", "tiktoken"):
    try:
        __import__(_m)
    except ImportError:
        sys.modules[_m] = types.ModuleType(_m)

import torch
import numpy as np

torch.set_num_threads(1)
sys.path.insert(0, "/private/tmp/claude-501/-Users-miaoyidi-Desktop-pretraining/124e307c-a54e-495e-9e13-8fa59adb3ccf/scratchpad")
import sweep_worker as W   # reuse build() / groups() / make_task()

S, V = W.S, 64


def measure(variant, L, N, seed, steps, lm):
    m = W.build(L, N, variant, V, model_seed=seed)
    opt = torch.optim.AdamW(W.groups(m, lm), eps=1e-10, weight_decay=0.0)
    sample = W.make_task("bigram", V)
    got = {}
    hs = [b.register_forward_hook(
        lambda _m, _i, o, i=i: got.__setitem__(
            i, (o[0] if isinstance(o, tuple) else o).detach().float().pow(2).mean().sqrt().item()))
        for i, b in enumerate(m.transformer.h)]
    g = torch.Generator().manual_seed(seed + 7)
    out = {}
    for st in range(steps + 1):
        ids = sample(16, g)
        got.clear()
        loss = m(ids[:, :-1].contiguous(), ids[:, 1:].contiguous())
        out[st] = float(np.mean([got[i] for i in sorted(got)]))
        if st < steps:
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
    for h in hs:
        h.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--lm", type=float, default=1.0)  # 1.0 == production lr_multiplier 0.25
    a = ap.parse_args(_ARGS)
    seeds = [int(x) for x in a.seeds.split(",")]
    checkpoints = [0, 5, 10, 25, 50, 100, 150, 200]
    checkpoints = [c for c in checkpoints if c <= a.steps]

    configs = []
    for v in ("broken", "linear", "logexp"):
        for sd in seeds:
            for L in (2, 4, 8, 16, 32, 64):        # depth ladder, width fixed 64
                configs.append((v, "depth", L, 64, sd))
            for N in (32, 64, 128, 256, 512):      # width ladder, depth fixed 4
                configs.append((v, "width", 4, N, sd))
    mine = [c for i, c in enumerate(configs) if i % a.nshards == a.shard]

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "ladder", "depth", "width", "seed", "step", "rms"])
        for k, (v, lad, L, N, sd) in enumerate(mine):
            r = measure(v, L, N, sd, a.steps, a.lm)
            for st in checkpoints:
                w.writerow([v, lad, L, N, sd, st, f"{r[st]:.6f}"])
            fh.flush()
            print(f"[shard {a.shard}] {k+1}/{len(mine)} {v} {lad} d{L}w{N} s{sd} done", flush=True)


if __name__ == "__main__":
    main()
