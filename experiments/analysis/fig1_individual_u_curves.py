"""Pick representative individual (single-model) val loss trajectories across
configurations to visualize the L(t) U-curve under limited-data training.

Curves selected:
  Q1 v2 (wd=0, single ind per df, d12/w768) — clean U-curves, varied by df:
    - df=0.2  → dramatic overfit (val: 4.7 → 7.7)
    - df=0.3  → clear U
    - df=0.4  → mild U
    - df=0.5  → just past minimum

  df=0.2 grid (wd=0.1, trapezoidal LR, 5 individuals × 25 ep, smallest-batch):
    - d12/w1536 init_shuffle model 0 — large width, milder overfit
    - d24/w1536 init_shuffle model 0 — largest cell, clearer overfit

X-axis: cumulative tokens_seen (per-model, including multi-epoch repeats).
"""
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ENT = "xjtumyd-carnegie-mellon-university/slowrun"


def fetch_individual_val_tokens(api, group, run_name):
    """Return (tokens_seen, val_loss) for one individual training run.
    If multiple wandb runs share the display name (resume case), merge by tokens_seen."""
    runs = list(api.runs(ENT, filters={"group": group, "display_name": run_name}))
    runs.sort(key=lambda r: r.created_at)
    by_tok = {}
    for run in runs:
        for row in run.scan_history():
            for k in range(1, 30):
                vl = row.get(f"model_{k}/val_loss")
                ts = row.get(f"model_{k}/tokens_seen")
                if vl is not None and ts is not None:
                    by_tok[int(ts)] = float(vl)
                    break
    if not by_tok:
        return np.array([]), np.array([])
    toks = np.array(sorted(by_tok))
    losses = np.array([by_tok[t] for t in toks])
    return toks, losses


# ----- Curve selection -----
CURVES = [
    # (group, run_name, label, palette_index)
    ("q1_data_size_q1v2_20260503_115259_d12_w768_df0.2",
     "q1_data_size_q1v2_20260503_115259_d12_w768_df0.2_init_ens_model0",
     "d12/w768, df=0.2, wd=0"),
    ("q1_data_size_q1v2_20260503_115259_d12_w768_df0.3",
     "q1_data_size_q1v2_20260503_115259_d12_w768_df0.3_init_ens_model0",
     "d12/w768, df=0.3, wd=0"),
    ("q1_data_size_q1v2_20260503_115259_d12_w768_df0.4",
     "q1_data_size_q1v2_20260503_115259_d12_w768_df0.4_init_ens_model0",
     "d12/w768, df=0.4, wd=0"),
    ("q1_data_size_q1v2_20260503_115259_d12_w768_df0.5",
     "q1_data_size_q1v2_20260503_115259_d12_w768_df0.5_init_ens_model0",
     "d12/w768, df=0.5, wd=0"),
    ("grid_20260430_152533_d12_w1536_df0.2",
     "grid_20260430_152533_d12_w1536_df0.2_init_shuffle_ens_model0",
     "d12/w1536, df=0.2, wd=0.1"),
    ("grid_20260430_152533_d24_w1536_df0.2",
     "grid_20260430_152533_d24_w1536_df0.2_init_shuffle_ens_model0",
     "d24/w1536, df=0.2, wd=0.1"),
]


def main():
    # Project plot style (per CLAUDE.md)
    sns.set(font_scale=1.5)
    sns.set_style('whitegrid')
    sns.set_palette('cool', len(CURVES))

    plt.rcParams['axes.labelsize']  = 24
    plt.rcParams['axes.linewidth']  = 4.0
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['grid.alpha']      = 0.25
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    api = wandb.Api()
    fig, ax = plt.subplots(figsize=(13, 7))

    for i, (group, run_name, label) in enumerate(CURVES):
        toks, losses = fetch_individual_val_tokens(api, group, run_name)
        if len(toks) == 0:
            print(f"MISSING: {run_name}")
            continue
        ax.plot(toks, losses, lw=3.0, alpha=0.95, label=label)
        i_min = int(np.argmin(losses))
        ax.plot([toks[i_min]], [losses[i_min]], marker="v", markersize=11,
                markeredgecolor="black", markeredgewidth=0.7, zorder=5)
        print(f"  {label}: {len(toks)} pts | min {losses.min():.3f} @ "
              f"{toks[i_min]/1e6:.0f}M tokens | end {losses[-1]:.3f}")

    ax.set_xlabel("tokens seen (per model, multi-epoch repeats included)")
    ax.set_ylabel("val loss")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax.legend(loc="upper left", framealpha=0.95)

    out = "experiments/figures/01_overfit_demos/individual_u_curves.pdf"
    plt.savefig(out, bbox_inches='tight', dpi=300)
    out_png = out.replace(".pdf", ".png")
    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    print(f"\nSaved {out}")
    print(f"Saved {out_png} (preview)")


if __name__ == "__main__":
    main()
