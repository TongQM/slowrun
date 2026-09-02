"""Audit wandb cloud data vs ground-truth train logs.

For each (group, run_name), report:
  - Whether wandb run exists
  - Epoch range covered by val_loss in wandb
  - Epoch range covered by val_loss in local train log
  - Mismatch flag if log has more epochs than wandb
"""
import os, re, sys
import wandb
from collections import defaultdict

ENT = "xjtumyd-carnegie-mellon-university/slowrun"
LOGS = "experiments/logs"
EPOCH_VAL_RE = re.compile(r"\[model \d+\]\s+epoch\s+(\d+)\s+val_loss=([0-9.]+)")


def parse_log_epochs(path):
    """Return set of epoch numbers with val_loss data."""
    if not os.path.isfile(path):
        return set()
    eps = set()
    with open(path) as f:
        for line in f:
            m = EPOCH_VAL_RE.search(line)
            if m:
                eps.add(int(m.group(1)))
    return eps


def wandb_run_epochs(api, group, display_name):
    """Return (run_exists: bool, epoch_set: set).
    Tries all model_{k}/ namespaces for k in 1..30 because train.py uses
    different namespaces depending on launcher (single-model with NUM_MODELS=20
    logs under model_{model_idx+1}/, but Q1 v2 single-model with NUM_MODELS=1
    logs under model_1/). Just scan all of them and union."""
    runs = list(api.runs(ENT, filters={"group": group, "display_name": display_name}))
    if not runs:
        return False, set()
    eps = set()
    for r in runs:
        for h in r.scan_history():
            for k in range(1, 30):
                vl = h.get(f"model_{k}/val_loss")
                ep = h.get(f"model_{k}/epoch")
                if vl is not None and ep is not None:
                    eps.add(int(ep))
                    break
    return True, eps


def fmt(eps):
    if not eps: return "-"
    s = sorted(eps)
    return f"{len(s)} eps [{s[0]}..{s[-1]}]"


def audit_q2():
    """Q2 original train (40532724) + Q2 ext (40583426)."""
    api = wandb.Api()
    group = "q2_ensemble_size_q2_20260502_234110_d12_w768_df0.2"
    print(f"\n=== Q2 audit: {group} ===")
    print(f"{'task':>4}  {'name':<55}  {'wandb':<22}  {'logs':<22}  status")
    issues = 0
    for strat_idx, strat in enumerate(["init_ens", "init_shuffle_ens"]):
        for m in range(20):
            task = strat_idx * 20 + m
            name = f"{group}_{strat}_model{m}"
            log_orig = f"{LOGS}/q2_train_d12_w768_df0.2_40532724_{task}.out"
            log_ext = f"{LOGS}/q2ext_train_40583426_{task}.out"
            log_eps = parse_log_epochs(log_orig) | parse_log_epochs(log_ext)
            exists, w_eps = wandb_run_epochs(api, group, name)
            ok = exists and (log_eps == w_eps or w_eps.issuperset(log_eps))
            status = "OK" if ok else ("MISSING_RUN" if not exists else f"MISSING_{len(log_eps - w_eps)}_EPS")
            if not ok:
                issues += 1
            print(f"  {task:>3}  {name[-50:]:<55}  {fmt(w_eps):<22}  {fmt(log_eps):<22}  {status}")
    print(f"\nQ2 issues: {issues}/40 runs")
    return issues


def audit_q1v2():
    """Q1 v2 trains (40542399, 40542401, 40542403, 40542405)."""
    api = wandb.Api()
    print(f"\n=== Q1 v2 audit ===")
    issues = 0
    df_to_jobid = {"0.2": 40542399, "0.3": 40542401, "0.4": 40542403, "0.5": 40542405}
    print(f"{'df':>5}  {'task':>4}  {'name':<60}  {'wandb':<22}  {'logs':<22}  status")
    for df, jobid in df_to_jobid.items():
        group = f"q1_data_size_q1v2_20260503_115259_d12_w768_df{df}"
        for strat_idx, strat in enumerate(["init_ens", "init_shuffle_ens"]):
            task = strat_idx
            name = f"{group}_{strat}_model0"
            log = f"{LOGS}/q1v2_train_d12_w768_df{df}_{jobid}_{task}.out"
            log_eps = parse_log_epochs(log)
            exists, w_eps = wandb_run_epochs(api, group, name)
            ok = exists and w_eps.issuperset(log_eps)
            status = "OK" if ok else ("MISSING_RUN" if not exists else f"MISSING_{len(log_eps - w_eps)}_EPS")
            if not ok: issues += 1
            print(f"  {df:>5}  {task:>3}  {name[-55:]:<60}  {fmt(w_eps):<22}  {fmt(log_eps):<22}  {status}")
    print(f"\nQ1 v2 issues: {issues}/8 runs")
    return issues


def audit_grid_df02():
    """Original df=0.2 grid (grid_20260430_152533): 9 cells × 5 ind × 2 strats."""
    api = wandb.Api()
    print(f"\n=== Original df=0.2 grid audit (logs likely gone, just check wandb-only) ===")
    # We don't have ground truth from logs (cleaned up); just check existence + epoch count.
    issues = 0
    cells = [(d, w) for d in (6, 12, 24) for w in (384, 768, 1536)]
    print(f"{'cell':<11}  strategy            model  wandb")
    for d, w in cells:
        group = f"grid_20260430_152533_d{d}_w{w}_df0.2"
        for strat in ["init_ens", "init_shuffle_ens"]:
            for m in range(5):
                name = f"{group}_{strat}_model{m}"
                exists, w_eps = wandb_run_epochs(api, group, name)
                if not exists or len(w_eps) < 25:
                    issues += 1
                    print(f"  d{d}/w{w:<5}  {strat:<19}  {m}      {fmt(w_eps)}  ISSUE")
    print(f"\ndf=0.2 grid issues: {issues}/90 runs")
    return issues


def audit_grid_df04_w384():
    """df=0.4 grid cells 1-3 (w384 column) — done via launch_grid_v2.sh."""
    api = wandb.Api()
    print(f"\n=== df=0.4 grid cells 1-3 audit ===")
    issues = 0
    for d in (6, 12, 24):
        group = f"grid_20260502_013702_d{d}_w384_df0.4"
        for strat in ["init_ens", "init_shuffle_ens"]:
            for m in range(5):
                name = f"{group}_{strat}_model{m}"
                exists, w_eps = wandb_run_epochs(api, group, name)
                if not exists or len(w_eps) < 25:
                    issues += 1
                    print(f"  d{d}/w384  {strat:<19}  {m}      {fmt(w_eps)}  ISSUE")
    print(f"\ndf=0.4 cells 1-3 issues: {issues}/30 runs")
    return issues


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    total = 0
    if target in ("all", "q2"): total += audit_q2()
    if target in ("all", "q1v2"): total += audit_q1v2()
    if target in ("all", "grid02"): total += audit_grid_df02()
    if target in ("all", "grid04"): total += audit_grid_df04_w384()
    print(f"\n=========================")
    print(f"TOTAL ISSUES: {total}")
