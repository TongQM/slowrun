"""
Fused replay: one forward pass per (model, eval-point) instead of one per (model, ens_size, eval-point).

Compared to replay.py (which computes a single ensemble size per process invocation), this script:

  - Takes a list of ensemble sizes and produces results for ALL of them in a single pass.
  - For each evaluation point (per-epoch checkpoint), loads all N models, then iterates
    over models forwarding each ONCE while incrementally accumulating the running logit sum.
    Whenever the running count S hits a target ensemble size, we compute that size's
    ensemble val loss + bpb on-the-fly.
  - Saves a factor of (sum(sizes) / max_size) in compute. For sizes={2,5,10,15,20} with
    N=20 individuals, that's 52/20 ≈ 2.6× fewer forward passes vs the per-size replay.

Output: one wandb run per ensemble size (so analysis tools that key on `ens/val_loss` /
`ens/tokens_seen` keep working). Runs are filled in sequentially after eval completes.

Mode: only `logit` averaging is implemented (project default). For `prob` averaging
the math doesn't telescope as cleanly across sizes; would require per-model softmax
storage and a separate code path.

Usage:
    python experiments/parallel/replay_fused.py \
        --checkpoint-dir checkpoints/<RUN_ID> \
        --num-models 20 \
        --ens-sizes 2 5 10 15 20 \
        --num-epochs 25 \
        --wandb-group <group> \
        --wandb-run-name-prefix <group>_<strategy>
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import wandb
import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--num-models", type=int, required=True,
                   help="Total individuals trained (ckpts model_0..N-1 must exist)")
    p.add_argument("--ens-sizes", type=int, nargs="+", required=True,
                   help="Ensemble sizes to evaluate (each ≤ num-models)")
    p.add_argument("--num-epochs", type=int, required=True)
    p.add_argument("--ensemble-mode", default="logit", choices=["logit"],
                   help="Only 'logit' supported in fused replay")
    p.add_argument("--wandb-group", required=True)
    p.add_argument("--wandb-run-name-prefix", required=True,
                   help="Run names will be <prefix>_ens<N>_replay")
    p.add_argument("--wandb-project", default="slowrun")
    p.add_argument("--input-val-bin", default=None)
    p.add_argument("--device-batch-size", type=int, default=2)
    p.add_argument("--start-epoch", type=int, default=1)
    p.add_argument("--end-epoch", type=int, default=None)
    args = p.parse_args()

    if args.end_epoch is None:
        args.end_epoch = args.num_epochs

    sizes = sorted(set(int(s) for s in args.ens_sizes))
    max_size = max(sizes)
    assert max_size <= args.num_models, "ens-size cannot exceed num-models"

    # Load training config
    config_path = os.path.join(args.checkpoint_dir, "config.json")
    with open(config_path) as f:
        train_config = json.load(f)
    m_cfg = train_config["model"]
    sys.argv = [sys.argv[0],
                f"--n_layer={m_cfg['n_layer']}",
                f"--n_head={m_cfg['n_head']}",
                f"--n_embd={m_cfg['n_embd']}",
                f"--dropout={m_cfg['dropout']}",
                f"--mup-base-width={m_cfg.get('mup_base_width', 768)}",
                f"--mup-base-depth={m_cfg.get('mup_base_depth', 12)}",
                f"--mup-base-head-dim={m_cfg.get('mup_base_head_dim', 64)}",
                f"--device-batch-size={args.device_batch_size}",
                "--compile-mode=eager"]
    if m_cfg.get("completep"):
        sys.argv.append("--completep")

    import importlib
    train_mod = importlib.import_module("unlimited.train")
    GPT = train_mod.GPT
    GPTConfig = train_mod.GPTConfig
    DataLoader = train_mod.DataLoader
    DATA_DIR = train_mod.DATA_DIR
    EVAL_TOKENS = train_mod.EVAL_TOKENS
    MAX_SEQ_LEN = train_mod.MAX_SEQ_LEN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else __import__("contextlib").nullcontext())

    encoder = tiktoken.get_encoding("gpt2")
    vocab_size = encoder.n_vocab
    eot_id = encoder._special_tokens['<|endoftext|>']
    token_bytes_list = []
    for i in range(vocab_size):
        token_bytes_list.append(0 if i == eot_id else len(encoder.decode_single_token_bytes(i)))
    token_bytes = torch.tensor(token_bytes_list, dtype=torch.int32, device=device)

    config = GPTConfig(
        vocab_size=vocab_size,
        n_layer=m_cfg["n_layer"], n_head=m_cfg["n_head"], n_embd=m_cfg["n_embd"],
        dropout=m_cfg["dropout"],
        completep=m_cfg.get("completep", False),
        mup_base_width=m_cfg.get("mup_base_width", 768),
        mup_base_depth=m_cfg.get("mup_base_depth", 12),
        mup_base_head_dim=m_cfg.get("mup_base_head_dim", 64),
        no_ve_projs=m_cfg.get("no_ve_projs", False),
        optimizer=train_config["optimizer"]["name"],
    )

    val_path = args.input_val_bin or os.path.join(DATA_DIR, "fineweb_val.pt")
    ens_eval_B = 1
    ens_eval_steps = EVAL_TOKENS // (ens_eval_B * MAX_SEQ_LEN * 1)
    tokens_per_epoch = train_config["training"].get("tokens_per_epoch")
    print(f"[fused replay] num_models={args.num_models} sizes={sizes} "
          f"epochs={args.start_epoch}..{args.end_epoch} eval_steps={ens_eval_steps}")

    # results[size] = list of dicts {epoch, tokens_seen, val_loss, val_bpb}
    results = {s: [] for s in sizes}

    # ---- Iterate over per-epoch checkpoint events ----
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        ckpt_paths = [os.path.join(args.checkpoint_dir, f"model_{i}_epoch_{epoch}.pt")
                      for i in range(args.num_models)]
        missing = [p for p in ckpt_paths if not os.path.exists(p)]
        if missing:
            print(f"[epoch {epoch}] SKIP — missing checkpoints: {missing[:3]}{'...' if len(missing)>3 else ''}")
            continue

        t0 = time.time()
        tokens_seen = epoch * tokens_per_epoch

        # Load all N models on GPU
        models = []
        for ckpt_path in ckpt_paths:
            with torch.device("meta"):
                m = GPT(config)
            m.to_empty(device=device)
            m.init_weights(convert_embed=False)
            sd = torch.load(ckpt_path, map_location=device, weights_only=True)
            m.load_state_dict(sd)
            m.eval()
            models.append(m)
            del sd
        load_dt = time.time() - t0
        print(f"[epoch {epoch}] loaded {args.num_models} models in {load_dt:.1f}s")

        # ---- Streaming evaluation: cumulative logit sum across models, per-batch ----
        # Per-size accumulators (all on device)
        total_loss = {s: torch.tensor(0.0, dtype=torch.float64, device=device) for s in sizes}
        total_nats = {s: torch.tensor(0.0, dtype=torch.float64, device=device) for s in sizes}
        total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
        total_tokens = torch.tensor(0, dtype=torch.int64, device=device)

        # One val_loader per epoch (deterministic seed=0 for reproducibility)
        val_loader = DataLoader(val_path, ens_eval_B, MAX_SEQ_LEN, device=device, seed=0, quiet=True)
        batch_iter = iter(val_loader)

        eval_t0 = time.time()
        # Explicit no_grad — model.eval() only disables dropout/batchnorm; without
        # no_grad, every forward pass keeps activations alive for a hypothetical
        # backward, blowing memory at N=20.
        with torch.no_grad():
            for step in range(ens_eval_steps):
                x, y, _ = next(batch_iter)
                flat_y = y.view(-1)
                mask = flat_y != -1

                num_bytes = token_bytes[flat_y.clamp(min=0)]
                valid_bytes = (num_bytes > 0)

                # Per-batch byte / token totals (computed once)
                total_tokens += mask.sum()
                total_bytes += num_bytes[mask].sum()

                # Cumulative logit sum across N models, snapshot at each target size
                logit_sum = None
                for i, m in enumerate(models):
                    with autocast_ctx:
                        logits = m.forward_logits(x).float()
                    flat_logits = logits.view(-1, logits.size(-1))
                    if logit_sum is None:
                        logit_sum = torch.zeros_like(flat_logits)
                    logit_sum += flat_logits
                    del logits, flat_logits

                    S = i + 1
                    if S in sizes:
                        avg_logits = logit_sum / S
                        loss_per_pos = F.cross_entropy(avg_logits, flat_y.clamp(min=0), reduction='none')
                        total_loss[S] += loss_per_pos[mask].sum().double()
                        total_nats[S] += (loss_per_pos[mask].double() * valid_bytes[mask].double()).sum()
                        del avg_logits, loss_per_pos
                del logit_sum
        eval_dt = time.time() - eval_t0

        # Finalize per-size loss/bpb
        log2 = float(torch.log(torch.tensor(2.0)))
        for s in sizes:
            val_loss = (total_loss[s] / total_tokens.double()).item()
            val_bpb = (total_nats[s] / (total_bytes.double() * log2)).item()
            results[s].append({
                "epoch": epoch,
                "tokens_seen": int(tokens_seen),
                "val_loss": val_loss,
                "val_bpb": val_bpb,
            })
            print(f"[epoch {epoch} ens={s}] val_loss={val_loss:.6f} val_bpb={val_bpb:.6f}")
        print(f"[epoch {epoch}] eval done in {eval_dt:.1f}s ({load_dt+eval_dt:.1f}s total)")

        # Free model memory before next epoch
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Dump to wandb: one run per ensemble size (compatible with plot.py) ----
    print(f"\n[fused replay] eval done; writing {len(sizes)} wandb runs")
    for s in sizes:
        run_name = f"{args.wandb_run_name_prefix}_ens{s}_replay"
        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=run_name,
            config={"replay": True, "fused": True,
                    "ensemble_mode": args.ensemble_mode,
                    "num_models": s,
                    "num_epochs": args.num_epochs,
                    "source_run_id": os.path.basename(args.checkpoint_dir.rstrip("/")),
                    "training_config": train_config},
            reinit=True,
        )
        run.define_metric("ens/tokens_seen")
        run.define_metric("ens/*", step_metric="ens/tokens_seen")
        for entry in results[s]:
            run.log({
                "ens/val_loss": entry["val_loss"],
                "ens/val_bpb": entry["val_bpb"],
                "ens/num_models": s,
                "ens/epoch": entry["epoch"],
                "ens/tokens_seen": entry["tokens_seen"],
            }, commit=True)
        run.finish()
        print(f"  wrote {run_name} ({len(results[s])} epochs)")

    print("Fused replay complete.")


if __name__ == "__main__":
    main()
