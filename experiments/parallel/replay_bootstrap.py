"""Bootstrap (membership-variance, scheme B) variant of replay_fused.py.

Differences vs replay_fused.py:
  - Each --bootstrap-iter is a deterministic random PERMUTATION of {0..N-1}.
    Members are distinct (no duplicates in any single ensemble). Cumulative
    E-ensembles are the first E entries of the permutation, so each iter
    gives one E-subset per ensemble size, and across B iters we get B
    different E-subsets per size (except E=N where every iter yields the
    full set).
  - Cumulative L_ens snapshots are taken along the PERMUTATION ORDER, not the
    canonical 0..N-1 order.
  - NO wandb logging at all. Results are saved to disk only at
    experiments/figures/02_ensemble_scaling/bootstrap/{strategy}_iter{idx:03d}.npz.
  - This lets us draw B independent (T,E) measurements per strategy without
    overwriting any of the original wandb runs.

Usage example:
    python experiments/parallel/replay_bootstrap.py \\
        --checkpoint-dir checkpoints/parallel_init_ens_q2_20260502_234110_d12_w768_df0.2 \\
        --strategy init_ens \\
        --num-models 20 \\
        --ens-sizes 2 5 10 15 20 \\
        --num-epochs 40 \\
        --bootstrap-iter 0
"""
import argparse
import json
import os
import sys
import time
import numpy as np

import torch
import torch.nn.functional as F
import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--strategy", required=True, choices=["init_ens", "init_shuffle_ens"])
    p.add_argument("--num-models", type=int, required=True)
    p.add_argument("--ens-sizes", type=int, nargs="+", required=True)
    p.add_argument("--num-epochs", type=int, required=True)
    p.add_argument("--bootstrap-iter", type=int, required=True,
                   help="Iteration index, also used as the RNG seed for the resample.")
    p.add_argument("--input-val-bin", default=None)
    p.add_argument("--device-batch-size", type=int, default=2)
    p.add_argument("--start-epoch", type=int, default=1)
    p.add_argument("--end-epoch", type=int, default=None)
    p.add_argument("--out-dir", default="experiments/figures/02_ensemble_scaling/bootstrap")
    args = p.parse_args()

    if args.end_epoch is None:
        args.end_epoch = args.num_epochs

    sizes = sorted(set(int(s) for s in args.ens_sizes))
    max_size = max(sizes)
    assert max_size <= args.num_models, "ens-size cannot exceed num-models"

    # ---- Bootstrap sample (permutation, no duplicates) ----
    # For each iter, draw a random permutation of {0..N-1}. Cumulative E-ensembles
    # are the first E entries of the permutation: E=2 → first 2, E=5 → first 5, etc.
    # Each iter gives one E-subset of distinct members per ensemble size; at E=N
    # all iters yield the full set (zero variance there, by construction).
    rng = np.random.default_rng(args.bootstrap_iter)
    boot_indices = rng.permutation(args.num_models).astype(int)
    unique_indices = sorted(int(i) for i in boot_indices)
    print(f"[bootstrap iter {args.bootstrap_iter}] strategy={args.strategy}")
    print(f"  permutation: {boot_indices.tolist()}")
    print(f"  unique models loaded per epoch: {len(unique_indices)} of {args.num_models}")

    # ---- Load training config ----
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
    print(f"  num_models={args.num_models} sizes={sizes} "
          f"epochs={args.start_epoch}..{args.end_epoch} eval_steps={ens_eval_steps}")

    # results[size] = list of (epoch, tokens_seen, val_loss, val_bpb)
    results = {s: [] for s in sizes}

    # ---- Iterate over per-epoch checkpoints ----
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        ckpt_paths = {i: os.path.join(args.checkpoint_dir, f"model_{i}_epoch_{epoch}.pt")
                      for i in unique_indices}
        missing = [p for p in ckpt_paths.values() if not os.path.exists(p)]
        if missing:
            print(f"[epoch {epoch}] SKIP — missing checkpoints: {missing[:3]}")
            continue

        t0 = time.time()
        tokens_seen = epoch * tokens_per_epoch

        # Load only the UNIQUE models needed by the bootstrap
        models = {}
        for idx, ckpt_path in ckpt_paths.items():
            with torch.device("meta"):
                m = GPT(config)
            m.to_empty(device=device)
            m.init_weights(convert_embed=False)
            sd = torch.load(ckpt_path, map_location=device, weights_only=True)
            m.load_state_dict(sd)
            m.eval()
            models[idx] = m
            del sd
        load_dt = time.time() - t0
        print(f"[epoch {epoch}] loaded {len(models)} unique models in {load_dt:.1f}s")

        # Per-size accumulators
        total_loss = {s: torch.tensor(0.0, dtype=torch.float64, device=device) for s in sizes}
        total_nats = {s: torch.tensor(0.0, dtype=torch.float64, device=device) for s in sizes}
        total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
        total_tokens = torch.tensor(0, dtype=torch.int64, device=device)

        val_loader = DataLoader(val_path, ens_eval_B, MAX_SEQ_LEN, device=device, seed=0, quiet=True)
        batch_iter = iter(val_loader)

        eval_t0 = time.time()
        with torch.no_grad():
            for step in range(ens_eval_steps):
                x, y, _ = next(batch_iter)
                flat_y = y.view(-1)
                mask = flat_y != -1

                num_bytes = token_bytes[flat_y.clamp(min=0)]
                valid_bytes = (num_bytes > 0)

                total_tokens += mask.sum()
                total_bytes += num_bytes[mask].sum()

                # Forward each UNIQUE model once per batch; cache flat logits
                cache = {}
                for idx, m in models.items():
                    with autocast_ctx:
                        logits = m.forward_logits(x).float()
                    cache[idx] = logits.view(-1, logits.size(-1))
                    del logits

                # Walk the bootstrap sequence (with repeats), accumulate cumulative sum
                logit_sum = torch.zeros_like(next(iter(cache.values())))
                for i_in_seq, boot_idx in enumerate(boot_indices):
                    logit_sum += cache[int(boot_idx)]
                    S = i_in_seq + 1
                    if S in sizes:
                        avg_logits = logit_sum / S
                        loss_per_pos = F.cross_entropy(avg_logits, flat_y.clamp(min=0), reduction='none')
                        total_loss[S] += loss_per_pos[mask].sum().double()
                        total_nats[S] += (loss_per_pos[mask].double() * valid_bytes[mask].double()).sum()
                        del avg_logits, loss_per_pos
                del logit_sum, cache
        eval_dt = time.time() - eval_t0

        log2 = float(torch.log(torch.tensor(2.0)))
        for s in sizes:
            val_loss = (total_loss[s] / total_tokens.double()).item()
            val_bpb = (total_nats[s] / (total_bytes.double() * log2)).item()
            results[s].append((epoch, int(tokens_seen), val_loss, val_bpb))
            print(f"[epoch {epoch} ens={s}] val_loss={val_loss:.6f} val_bpb={val_bpb:.6f}")
        print(f"[epoch {epoch}] eval done in {eval_dt:.1f}s")

        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Save to disk ----
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir,
                            f"{args.strategy}_iter{args.bootstrap_iter:03d}.npz")
    epochs_arr = np.array([row[0] for row in results[sizes[0]]], dtype=np.int32)
    tokens_arr = np.array([row[1] for row in results[sizes[0]]], dtype=np.int64)
    L_grid = np.array([[row[2] for row in results[s]] for s in sizes], dtype=np.float64)
    bpb_grid = np.array([[row[3] for row in results[s]] for s in sizes], dtype=np.float64)
    np.savez(out_path,
             L=L_grid, bpb=bpb_grid,
             epochs=epochs_arr, tokens=tokens_arr,
             sizes=np.array(sizes, dtype=np.int32),
             boot_indices=boot_indices.astype(np.int32),
             unique_indices=np.array(unique_indices, dtype=np.int32),
             strategy=args.strategy,
             iter_idx=args.bootstrap_iter)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
