"""
CompleteP Orchestrator — launches unlimited/train.py runs across model sizes and ensemble strategies.

For each model size, launches up to 3 ensemble strategy runs:
  1. No ensemble:           --num-models 1
  2. Init ensemble:         --num-models N --ensemble-type init
  3. Init+shuffle ensemble: --num-models N --ensemble-type init_shuffle

Model sizes are specified as comma-separated "layer:head:embd" triples.
All runs use CompleteP parametrization (muP width scaling + 1/L depth scaling).

Usage:
    # Single node, 8 GPUs (default)
    python experiments/sync/sweep.py \
        --model-sizes 12:12:768 \
        --num-models 5 --num-epochs 12

    # Multiple model sizes (muP width/depth sweep)
    python experiments/sync/sweep.py \
        --model-sizes 12:12:768,20:10:1280,26:14:1792 \
        --num-models 5 --num-epochs 12 \
        --wandb-group completep_sweep

    # Multi-node (4 nodes x 8 GPUs = 32 GPUs)
    python experiments/sync/sweep.py \
        --model-sizes 30:16:2048 \
        --launch-prefix "torchrun --nnodes=4 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=host:29500" \
        --num-models 5

    # SLURM cluster
    python experiments/sync/sweep.py \
        --model-sizes 30:16:2048 \
        --launch-prefix "srun torchrun --standalone --nproc_per_node=8" \
        --num-models 5
"""

import argparse
import subprocess
import time


def parse_model_sizes(spec):
    """Parse 'layer:head:embd,layer:head:embd,...' into list of (n_layer, n_head, n_embd)."""
    sizes = []
    for entry in spec.split(","):
        parts = entry.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid model size '{entry}'. Expected format: layer:head:embd")
        n_layer, n_head, n_embd = int(parts[0]), int(parts[1]), int(parts[2])
        sizes.append((n_layer, n_head, n_embd))
    return sizes


def build_cmd(args, n_layer, n_head, n_embd, num_models, ensemble_type, run_suffix):
    """Build a launch command for one unlimited/train.py run."""
    # Split launch prefix into tokens (e.g. "srun torchrun --nproc_per_node=8")
    cmd = args.launch_prefix.split() + [
        "unlimited/train.py",
        "--completep",
        f"--n_layer={n_layer}",
        f"--n_head={n_head}",
        f"--n_embd={n_embd}",
        f"--num-models={num_models}",
        f"--ensemble-type={ensemble_type}",
        f"--num-epochs={args.num_epochs}",
        f"--optimizer={args.optimizer}",
        f"--mup-base-width={args.mup_base_width}",
        f"--weight-decay={args.weight_decay}",
        f"--dropout={args.dropout}",
        f"--ensemble-mode={args.ensemble_mode}",
        f"--run={run_suffix}",
    ]
    if args.num_epochs_model_0 is not None:
        cmd.append(f"--num-epochs-model-0={args.num_epochs_model_0}")
    if args.wandb_group:
        cmd.append(f"--wandb_group={args.wandb_group}")
    if args.input_bin:
        cmd.append(f"--input_bin={args.input_bin}")
    if args.input_val_bin:
        cmd.append(f"--input_val_bin={args.input_val_bin}")
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="CompleteP orchestrator: run ensemble experiments with muP + 1/L depth scaling")

    # Model config — supports multiple sizes
    parser.add_argument("--model-sizes", type=str, required=True,
                        help="Comma-separated model configs as 'layer:head:embd'. "
                             "E.g. '12:12:768' or '12:12:768,20:10:1280,26:14:1792'")

    # Experiment config
    parser.add_argument("--num-models", type=int, default=5,
                        help="Number of ensemble members for init/init_shuffle runs")
    parser.add_argument("--num-epochs", type=int, default=12)
    parser.add_argument("--num-epochs-model-0", type=int, default=None,
                        help="Epochs for first model (defaults to --num-epochs)")
    parser.add_argument("--optimizer", type=str, default="hybrid",
                        choices=["hybrid", "muon", "adamw"])
    parser.add_argument("--launch-prefix", type=str,
                        default="torchrun --standalone --nproc_per_node=8",
                        help="Launch command prefix before 'unlimited/train.py'. "
                             "Examples: 'torchrun --standalone --nproc_per_node=8' (single node, default), "
                             "'torchrun --nnodes=4 --nproc_per_node=8 --rdzv_backend=c10d "
                             "--rdzv_endpoint=host:29500' (multi-node), "
                             "'srun torchrun --standalone --nproc_per_node=8' (SLURM)")
    parser.add_argument("--run-prefix", type=str, default="completep",
                        help="Prefix for wandb run names")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="Wandb group to organize all runs together")

    # Pass-through flags for unlimited/train.py
    parser.add_argument("--mup-base-width", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1.3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ensemble-mode", type=str, default="logit",
                        choices=["logit", "prob"])
    parser.add_argument("--input_bin", type=str, default=None)
    parser.add_argument("--input_val_bin", type=str, default=None)

    # Control which ensemble strategies to run
    parser.add_argument("--skip-no-ensemble", action="store_true",
                        help="Skip the single-model (no ensemble) run")
    parser.add_argument("--skip-init", action="store_true",
                        help="Skip the init-only ensemble run")
    parser.add_argument("--skip-init-shuffle", action="store_true",
                        help="Skip the init+shuffle ensemble run")

    args = parser.parse_args()

    model_sizes = parse_model_sizes(args.model_sizes)

    # Build ensemble strategies
    ensemble_strategies = []
    if not args.skip_no_ensemble:
        ensemble_strategies.append((1, "init_shuffle", "no_ensemble"))
    if not args.skip_init:
        ensemble_strategies.append((args.num_models, "init", "init_ens"))
    if not args.skip_init_shuffle:
        ensemble_strategies.append((args.num_models, "init_shuffle", "init_shuffle_ens"))

    if not ensemble_strategies:
        print("All ensemble strategies skipped. Nothing to do.")
        return

    total_runs = len(model_sizes) * len(ensemble_strategies)

    print(f"CompleteP Experiment")
    print(f"  Launch: {args.launch_prefix}")
    print(f"  Model sizes: {len(model_sizes)}")
    for n_layer, n_head, n_embd in model_sizes:
        print(f"    d{n_layer} (n_layer={n_layer}, n_head={n_head}, n_embd={n_embd})"
              f"  depth_scale={1.0/n_layer:.4f}, output_mult={args.mup_base_width/n_embd:.4f}")
    print(f"  Optimizer: {args.optimizer}, CompleteP: True")
    print(f"  Ensemble strategies: {len(ensemble_strategies)}, members: {args.num_models}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Total runs: {total_runs}")
    print()

    results = []
    run_idx = 0
    for n_layer, n_head, n_embd in model_sizes:
        size_tag = f"d{n_layer}_w{n_embd}"
        for num_models, ens_type, ens_suffix in ensemble_strategies:
            run_idx += 1
            run_suffix = f"{args.run_prefix}_{size_tag}_{ens_suffix}"

            cmd = build_cmd(args, n_layer, n_head, n_embd,
                            num_models, ens_type, run_suffix)

            print(f"{'='*60}")
            print(f"[{run_idx}/{total_runs}] {size_tag} / {ens_suffix}"
                  f" (num_models={num_models}, ensemble_type={ens_type})")
            print(f"Command: {' '.join(cmd)}")
            print(f"{'='*60}")

            start_time = time.time()
            result = subprocess.run(cmd)
            elapsed = time.time() - start_time

            status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
            results.append((size_tag, ens_suffix, status, elapsed))
            print(f"\n  {size_tag}/{ens_suffix}: {status} ({elapsed/60:.1f} min)")

            if result.returncode != 0:
                print(f"  WARNING: run failed. Continuing with remaining runs...")

    # Summary
    print(f"\n{'='*60}")
    print("CompleteP Experiment Summary")
    print(f"{'='*60}")
    for size_tag, ens_suffix, status, elapsed in results:
        label = f"{size_tag}/{ens_suffix}"
        print(f"  {label:40s} {status:20s} ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
