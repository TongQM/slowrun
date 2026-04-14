"""
Post-hoc ensemble val eval from per-epoch checkpoints.

After parallel training jobs complete, this script loads each
(epoch_k, all-N-models) checkpoint set and computes ensemble val loss.
Logs `ens/val_loss`, `ens/val_bpb`, `ens/num_models`, `ens/epoch` to wandb,
matching the schema produced by sync-mode training.

Usage:
    python data_eff/replay_ensemble_eval.py \
        --checkpoint-dir checkpoints/<RUN_ID> \
        --num-models 5 \
        --num-epochs 30 \
        --ensemble-mode logit \
        --wandb-run-name <run_id>_replay \
        --wandb-group <group_name>
"""
import argparse
import json
import os
import sys
import time

import torch
import wandb
import tiktoken

# Ensure we can import from unlimited/train.py
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True,
                   help="Directory containing model_{i}_epoch_{k}.pt and config.json")
    p.add_argument("--num-models", type=int, required=True)
    p.add_argument("--num-epochs", type=int, required=True)
    p.add_argument("--ensemble-mode", default="logit", choices=["logit", "prob"])
    p.add_argument("--wandb-run-name", required=True)
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-project", default="slowrun")
    p.add_argument("--input-val-bin", default=None,
                   help="Path to val.pt (defaults to fineweb_data/fineweb_val.pt)")
    p.add_argument("--device-batch-size", type=int, default=2,
                   help="Used for evaluate_bpb path; ensemble eval uses B=1")
    args = p.parse_args()

    # Load training config to recover model architecture
    config_path = os.path.join(args.checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json missing in {args.checkpoint_dir}")
    with open(config_path) as f:
        train_config = json.load(f)

    # Inject the loaded config into argv so train.py's argparse picks up matching values
    # (we actually only need n_layer/n_head/n_embd/dropout/completep/mup_base_width)
    m_cfg = train_config["model"]
    val_cfg = train_config.get("val", {})
    sys.argv = [sys.argv[0],
                f"--n_layer={m_cfg['n_layer']}",
                f"--n_head={m_cfg['n_head']}",
                f"--n_embd={m_cfg['n_embd']}",
                f"--dropout={m_cfg['dropout']}",
                f"--mup-base-width={m_cfg['mup_base_width']}",
                f"--device-batch-size={args.device_batch_size}",
                "--compile-mode=eager"]
    if m_cfg.get("completep"):
        sys.argv.append("--completep")

    # Now import (this evaluates the module-level argparse and constants)
    import importlib
    train_mod = importlib.import_module("unlimited.train")

    GPT = train_mod.GPT
    GPTConfig = train_mod.GPTConfig
    DataLoader = train_mod.DataLoader
    evaluate_ensemble_in_memory = train_mod.evaluate_ensemble_in_memory
    DATA_DIR = train_mod.DATA_DIR
    EVAL_TOKENS = train_mod.EVAL_TOKENS
    MAX_SEQ_LEN = train_mod.MAX_SEQ_LEN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else __import__("contextlib").nullcontext())

    # Build token_bytes for BPB calc
    encoder = tiktoken.get_encoding("gpt2")
    vocab_size = encoder.n_vocab
    eot_id = encoder._special_tokens['<|endoftext|>']
    token_bytes_list = []
    for i in range(vocab_size):
        if i == eot_id:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(encoder.decode_single_token_bytes(i)))
    token_bytes = torch.tensor(token_bytes_list, dtype=torch.int32, device=device)

    config = GPTConfig(
        vocab_size=vocab_size,
        n_layer=m_cfg["n_layer"],
        n_head=m_cfg["n_head"],
        n_embd=m_cfg["n_embd"],
        dropout=m_cfg["dropout"],
        completep=m_cfg.get("completep", False),
        mup_base_width=m_cfg.get("mup_base_width", 256),
        optimizer=train_config["optimizer"]["name"],
    )

    val_path = args.input_val_bin or os.path.join(DATA_DIR, "fineweb_val.pt")
    ens_eval_B = 1
    ens_eval_steps = EVAL_TOKENS // (ens_eval_B * MAX_SEQ_LEN * 1)  # single GPU, world_size=1

    # Init wandb
    wandb_kwargs = {"project": args.wandb_project, "name": args.wandb_run_name,
                    "config": {"replay": True,
                               "ensemble_mode": args.ensemble_mode,
                               "num_models": args.num_models,
                               "num_epochs": args.num_epochs,
                               "source_run_id": os.path.basename(args.checkpoint_dir.rstrip("/")),
                               "training_config": train_config}}
    if args.wandb_group:
        wandb_kwargs["group"] = args.wandb_group
    run = wandb.init(**wandb_kwargs)
    print(f"Wandb run: {run.url}")

    # Process each epoch
    for epoch in range(1, args.num_epochs + 1):
        ckpt_paths = [os.path.join(args.checkpoint_dir, f"model_{i}_epoch_{epoch}.pt")
                      for i in range(args.num_models)]
        missing = [p for p in ckpt_paths if not os.path.exists(p)]
        if missing:
            print(f"[epoch {epoch}] SKIP — missing checkpoints: {missing}")
            continue

        t0 = time.time()
        print(f"[epoch {epoch}] Loading {args.num_models} models...")
        models = []
        for ckpt_path in ckpt_paths:
            with torch.device("meta"):
                m = GPT(config)
            m.to_empty(device=device)
            m.init_weights(convert_embed=False)
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
            m.load_state_dict(state_dict)
            m.eval()
            models.append(m)
            del state_dict

        val_loader = DataLoader(val_path, ens_eval_B, MAX_SEQ_LEN,
                                device=device, seed=0, quiet=True)
        ebpb, eloss = evaluate_ensemble_in_memory(
            models, val_loader, ens_eval_steps, token_bytes,
            device, autocast_ctx, mode=args.ensemble_mode)

        dt = time.time() - t0
        print(f"[epoch {epoch}] ens_val_loss={eloss:.6f} ens_val_bpb={ebpb:.6f}  ({dt:.1f}s)")

        run.log({
            "ens/val_loss": eloss,
            "ens/val_bpb": ebpb,
            "ens/num_models": args.num_models,
            "ens/epoch": epoch,
        }, commit=True)

        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    run.finish()
    print("Replay complete.")


if __name__ == "__main__":
    main()
