"""
Post-hoc ensemble val eval from per-epoch checkpoints.

After parallel training jobs complete, this script loads each
(epoch_k, all-N-models) checkpoint set and computes ensemble val loss.
Logs `ens/val_loss`, `ens/val_bpb`, `ens/num_models`, `ens/epoch` to wandb,
matching the schema produced by sync-mode training.

Usage:
    python experiments/parallel/replay.py \
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
ROOT = os.path.dirname(os.path.dirname(HERE))  # experiments/parallel/.. /.. = repo root
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
    p.add_argument("--start-epoch", type=int, default=1,
                   help="Resume from this epoch (1-indexed). Use for crash recovery.")
    p.add_argument("--end-epoch", type=int, default=None,
                   help="Stop after this epoch (defaults to --num-epochs)")
    p.add_argument("--wandb-resume-id", type=str, default=None,
                   help="If set, resume an existing wandb run with this ID instead of creating a new one.")
    p.add_argument("--skip-individual-val", action="store_true", default=False,
                   help="Skip per-model val eval in replay (just compute ensemble val). "
                        "Per-model val is already logged during training, so this saves ~Nx time.")
    p.add_argument("--progress-file", type=str, default=None,
                   help="JSON file recording {last_completed_epoch, wandb_run_id}. "
                        "If it exists on startup, resume from last_completed_epoch+1 and reuse "
                        "the stored wandb run id. Written atomically after each ensemble epoch.")
    args = p.parse_args()
    if args.end_epoch is None:
        args.end_epoch = args.num_epochs

    # Resume from progress file if present
    resumed_from = None
    if args.progress_file and os.path.exists(args.progress_file):
        with open(args.progress_file) as f:
            prog = json.load(f)
        last = int(prog.get("last_completed_epoch", 0))
        resumed_from = last
        # never redo completed epochs, but respect a higher CLI start-epoch
        args.start_epoch = max(args.start_epoch, last + 1)
        if args.wandb_resume_id is None and prog.get("wandb_run_id"):
            args.wandb_resume_id = prog["wandb_run_id"]
        if last >= args.end_epoch:
            print(f"[resume] Progress file says epoch {last} already done "
                  f"(>= end_epoch={args.end_epoch}); nothing to do.")
            return

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
    indiv_eval_B = args.device_batch_size
    indiv_eval_steps = EVAL_TOKENS // (indiv_eval_B * MAX_SEQ_LEN * 1)
    evaluate_bpb = train_mod.evaluate_bpb

    # Steps-per-epoch for x-axis (so per-model val maps to step count)
    steps_per_epoch = train_config["training"].get("steps_per_epoch", 1)

    # Init wandb (optionally resume an existing run)
    wandb_kwargs = {"project": args.wandb_project, "name": args.wandb_run_name,
                    "config": {"replay": True,
                               "ensemble_mode": args.ensemble_mode,
                               "num_models": args.num_models,
                               "num_epochs": args.num_epochs,
                               "source_run_id": os.path.basename(args.checkpoint_dir.rstrip("/")),
                               "training_config": train_config}}
    if args.wandb_group:
        wandb_kwargs["group"] = args.wandb_group
    if args.wandb_resume_id:
        wandb_kwargs["id"] = args.wandb_resume_id
        wandb_kwargs["resume"] = "allow"
    run = wandb.init(**wandb_kwargs)
    print(f"Wandb run: {run.url}")
    if resumed_from is not None:
        print(f"[resume] Restarting after epoch {resumed_from}; "
              f"replaying epochs {args.start_epoch}..{args.end_epoch}")

    # Define metric step axes (clean per-model curves in wandb)
    for i in range(args.num_models):
        run.define_metric(f"model_{i+1}/step")
        run.define_metric(f"model_{i+1}/*", step_metric=f"model_{i+1}/step")
    run.define_metric("ens/epoch")
    run.define_metric("ens/*", step_metric="ens/epoch")

    # Process each epoch (with optional resume range)
    print(f"Replay epochs {args.start_epoch}..{args.end_epoch}")
    for epoch in range(args.start_epoch, args.end_epoch + 1):
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

        # Per-model individual val eval (skip if requested — use training-time per-model logs instead)
        if not args.skip_individual_val:
            for i, m in enumerate(models):
                indiv_loader = DataLoader(val_path, indiv_eval_B, MAX_SEQ_LEN,
                                          device=device, seed=0, quiet=True)
                with autocast_ctx:
                    vbpb, vloss = evaluate_bpb(m, indiv_loader, indiv_eval_steps, token_bytes)
                print(f"  [model {i+1}] epoch {epoch}: val_loss={vloss:.4f} val_bpb={vbpb:.4f}")
                run.log({
                    f"model_{i+1}/val_loss": vloss,
                    f"model_{i+1}/val_bpb": vbpb,
                    f"model_{i+1}/epoch": epoch,
                    f"model_{i+1}/step": epoch * steps_per_epoch,
                }, commit=True)

        # Ensemble val eval
        val_loader = DataLoader(val_path, ens_eval_B, MAX_SEQ_LEN,
                                device=device, seed=0, quiet=True)
        ebpb, eloss = evaluate_ensemble_in_memory(
            models, val_loader, ens_eval_steps, token_bytes,
            device, autocast_ctx, mode=args.ensemble_mode)

        dt = time.time() - t0
        print(f"[epoch {epoch}] ens_val_loss={eloss:.6f} ens_val_bpb={ebpb:.6f}  ({dt:.1f}s total)")

        run.log({
            "ens/val_loss": eloss,
            "ens/val_bpb": ebpb,
            "ens/num_models": args.num_models,
            "ens/epoch": epoch,
        }, commit=True)

        # Persist progress atomically (rename is atomic on POSIX)
        if args.progress_file:
            tmp = args.progress_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"last_completed_epoch": epoch, "wandb_run_id": run.id}, f)
            os.replace(tmp, args.progress_file)

        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    run.finish()
    print("Replay complete.")


if __name__ == "__main__":
    main()
