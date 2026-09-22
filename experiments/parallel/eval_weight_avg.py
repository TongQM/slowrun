"""Weight-averaging baselines from saved checkpoints (reviewer's Model Soups point).

Two kinds of average, evaluated on the same held-out split as everything else:
  soup   : average the weights of E independently trained members at one epoch
           (expected to fail: different initialisations)
  ckpt   : average the last K epoch checkpoints of ONE member (checkpoint averaging)
Compare with: the single member at the same epoch, and the logit ensemble (from the
replay logs, not recomputed here).

Usage:
  python experiments/parallel/eval_weight_avg.py --checkpoint-dir DIR --models 0 1 2 3 \
      --epochs 5 10 ... --soup-sizes 2 4 --ckpt-windows 2 4 --out results.json
Checkpoint filenames: model_{i}_epoch_{e}.pt.
"""
from __future__ import annotations
import argparse, json, os, sys, importlib
import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--models", type=int, nargs="+", required=True)
    p.add_argument("--epochs", type=int, nargs="+", required=True, help="epochs with checkpoints for every model")
    p.add_argument("--soup-sizes", type=int, nargs="*", default=[2, 4])
    p.add_argument("--ckpt-windows", type=int, nargs="*", default=[2, 4], help="average the last K listed epochs of model 0")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cfg = json.load(open(os.path.join(a.checkpoint_dir, "config.json")))
    m = cfg["model"]
    sys.argv = [sys.argv[0], f"--n_layer={m['n_layer']}", f"--n_head={m['n_head']}", f"--n_embd={m['n_embd']}",
                f"--dropout={m['dropout']}", f"--mup-base-width={m.get('mup_base_width', 768)}",
                f"--mup-base-depth={m.get('mup_base_depth', 12)}", f"--mup-base-head-dim={m.get('mup_base_head_dim', 64)}",
                "--device-batch-size=2", "--compile-mode=eager"] + (["--completep"] if m.get("completep") else []) \
               + (["--no-ve-projs"] if m.get("no_ve_projs") else [])
    T = importlib.import_module("unlimited.train")
    import tiktoken
    dev = torch.device("cuda")
    enc = tiktoken.get_encoding("gpt2"); eot = enc._special_tokens["<|endoftext|>"]
    token_bytes = torch.tensor([0 if i == eot else len(enc.decode_single_token_bytes(i)) for i in range(enc.n_vocab)],
                               dtype=torch.int32, device=dev)
    config = T.GPTConfig(vocab_size=enc.n_vocab, n_layer=m["n_layer"], n_head=m["n_head"], n_embd=m["n_embd"],
                         dropout=m["dropout"], completep=m.get("completep", False), mup_base_width=m.get("mup_base_width", 768),
                         mup_base_depth=m.get("mup_base_depth", 12), mup_base_head_dim=m.get("mup_base_head_dim", 64),
                         no_ve_projs=m.get("no_ve_projs", False), optimizer=cfg["optimizer"]["name"])
    B = 8
    steps = T.EVAL_TOKENS // (B * T.MAX_SEQ_LEN)
    val_path = os.path.join(T.DATA_DIR, "fineweb_val.pt")
    ac = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    def ck(i, e): return os.path.join(a.checkpoint_dir, f"model_{i}_epoch_{e}.pt")
    def avg_state(paths):
        sd = None
        for p_ in paths:
            s = torch.load(p_, map_location="cpu", weights_only=True)
            if sd is None: sd = {k: v.float().clone() for k, v in s.items()}
            else:
                for k in sd: sd[k] += s[k].float()
        return {k: v / len(paths) for k, v in sd.items()}
    @torch.no_grad()
    def evaluate(sd):
        with torch.device("meta"): model = T.GPT(config)
        model.to_empty(device=dev); model.init_weights(convert_embed=False)
        ref = torch.load(ck(a.models[0], a.epochs[-1]), map_location="cpu", weights_only=True)
        model.load_state_dict({k: v.to(ref[k].dtype) for k, v in sd.items()}); model.eval()
        loader = T.DataLoader(val_path, B, T.MAX_SEQ_LEN, device=dev, seed=0, quiet=True)
        with ac: bpb, loss = T.evaluate_bpb(model, loader, steps, token_bytes)
        del model; torch.cuda.empty_cache(); return float(loss)

    res = []
    for e in a.epochs:
        single = evaluate(avg_state([ck(a.models[0], e)]))
        res.append(dict(kind="single", model=a.models[0], epoch=e, loss=single)); print(res[-1], flush=True)
        for E in a.soup_sizes:
            if E <= len(a.models):
                l = evaluate(avg_state([ck(i, e) for i in a.models[:E]]))
                res.append(dict(kind="soup", E=E, epoch=e, loss=l)); print(res[-1], flush=True)
        for K in a.ckpt_windows:
            eps = [x for x in a.epochs if x <= e][-K:]
            if len(eps) == K:
                l = evaluate(avg_state([ck(a.models[0], x) for x in eps]))
                res.append(dict(kind="ckpt_avg", K=K, epochs=eps, epoch=e, loss=l)); print(res[-1], flush=True)
    json.dump(dict(checkpoint_dir=a.checkpoint_dir, results=res), open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
