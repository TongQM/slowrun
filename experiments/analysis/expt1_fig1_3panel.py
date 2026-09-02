"""Figure 1 (replot, extended) — overfit U-curve at the base cell d=12, w=768.

Data:
  data_export/expt4_datasize/wd0_fixed_tokens/  (10 df × 2 strategies, wd=0,
  constant LR, ~1B-token budget per run; epochs scale inversely with df).

Per the lab's empirical finding, init and init+shuffle individuals are
indistinguishable at this cell, so both strategies are averaged per df.

Layout:
  L | val loss vs cumulative tokens (B)
  M | val loss vs continuous epoch (= tokens / P)
  R | overfit-onset position vs unique tokens P
        filled markers (left axis):  nadir epoch
        open markers   (right axis): nadir cumulative tokens (M)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data_export" / "expt4_datasize" / "wd0_fixed_tokens"
CORPUS_TOKENS = 100_000_000               # 100M-token FineWeb subset
BATCH_SIZE = 131072                        # project default --total-batch-size


def load_runs():
    """runs[df] = list of dicts with val + train arrays per strategy."""
    runs: dict[float, list] = {}
    for f in sorted(DATA.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        df = round(float(str(d["df"])), 2)
        entry = dict(
            strat=str(d["strat"]),
            tokens=d["tokens"].astype(np.float64),
            val_loss=d["val_loss"].astype(np.float64),
        )
        if "train_tokens" in d.files and "train_loss" in d.files:
            entry["train_tokens"] = d["train_tokens"].astype(np.float64)
            entry["train_loss"] = d["train_loss"].astype(np.float64)
        runs.setdefault(df, []).append(entry)
    return runs


def find_nadir(tokens, val_loss):
    """Argmin of raw val_loss. Returns (idx, tokens_at_nadir, vl_at_nadir).
    Curves are clean enough that no smoothing is needed; smoothing with
    edge-padding caused fake minima at the last point.
    """
    idx = int(np.argmin(val_loss))
    return idx, float(tokens[idx]), float(val_loss[idx])


def setup_style():
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams["axes.labelsize"] = 24
    plt.rcParams["axes.linewidth"] = 4.0
    plt.rcParams["legend.fontsize"] = 18
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "experiments/analysis/expt1_fig1_3panel"))
    args = ap.parse_args()

    setup_style()
    runs = load_runs()
    dfs = sorted(runs.keys())
    palette = sns.color_palette("cool", len(dfs))

    fig = plt.figure(figsize=(30, 11.5))
    gs = fig.add_gridspec(1, 3, wspace=0.26, top=0.94, bottom=0.24)
    ax_t = fig.add_subplot(gs[0, 0])     # val loss vs tokens
    ax_e = fig.add_subplot(gs[0, 1])     # val loss vs epochs
    ax_n = fig.add_subplot(gs[0, 2])     # nadir vs P (twin y)
    ax_n2 = ax_n.twinx()

    STRAT_MARKER = {"init_ens": "o", "init_shuffle_ens": "s"}
    STRAT_PRETTY = {"init_ens": "init", "init_shuffle_ens": "init + shuffle"}

    # collect per-(df, strat) nadirs for the right panel
    nadir_per_strat = {s: {"P": [], "ep": [], "tok": [], "color": []}
                       for s in STRAT_MARKER}

    for color, df in zip(palette, dfs):
        P = CORPUS_TOKENS * df
        strat_data = runs[df]
        ref_tokens = strat_data[0]["tokens"]

        # ---- average val loss across strategies on the shared grid ----
        vls = []
        for entry in strat_data:
            if (len(entry["tokens"]) == len(ref_tokens)
                    and np.allclose(entry["tokens"], ref_tokens)):
                vls.append(entry["val_loss"])
            else:
                vls.append(np.interp(ref_tokens, entry["tokens"], entry["val_loss"]))
            # per-strategy val nadir for right panel
            idx_s, tok_s, _ = find_nadir(entry["tokens"], entry["val_loss"])
            nadir_per_strat[entry["strat"]]["P"].append(P)
            nadir_per_strat[entry["strat"]]["ep"].append(tok_s / P)
            nadir_per_strat[entry["strat"]]["tok"].append(tok_s)
            nadir_per_strat[entry["strat"]]["color"].append(color)
        vl_mean = np.stack(vls, axis=0).mean(axis=0)

        # ---- average train loss across strategies on the shared train grid ----
        train_mean = train_tokens_ref = None
        if all("train_tokens" in e for e in strat_data):
            ref_train_tokens = strat_data[0]["train_tokens"]
            tls = []
            for entry in strat_data:
                if (len(entry["train_tokens"]) == len(ref_train_tokens)
                        and np.allclose(entry["train_tokens"], ref_train_tokens)):
                    tls.append(entry["train_loss"])
                else:
                    tls.append(np.interp(ref_train_tokens,
                                         entry["train_tokens"], entry["train_loss"]))
            train_mean = np.stack(tls, axis=0).mean(axis=0)
            train_tokens_ref = ref_train_tokens

        label = fr"$P = {int(P/1e6)}$M"

        # ---- panel (A): val + train vs steps s ----
        ax_t.plot(ref_tokens / BATCH_SIZE, vl_mean, color=color, lw=3.0, label=label)
        if train_mean is not None:
            ax_t.plot(train_tokens_ref / BATCH_SIZE, train_mean, color=color,
                      lw=1.2, ls="--", alpha=0.40, zorder=1)
        idx_avg, _, _ = find_nadir(ref_tokens, vl_mean)
        ax_t.scatter([ref_tokens[idx_avg] / BATCH_SIZE], [vl_mean[idx_avg]],
                     marker="v", s=110, color=color,
                     edgecolor="black", linewidth=1.0, zorder=5)

        # ---- panel (B): val + train vs epoch ----
        ax_e.plot(ref_tokens / P, vl_mean, color=color, lw=3.0, label=label)
        if train_mean is not None:
            ax_e.plot(train_tokens_ref / P, train_mean, color=color,
                      lw=1.2, ls="--", alpha=0.40, zorder=1)
        ax_e.scatter([ref_tokens[idx_avg] / P], [vl_mean[idx_avg]],
                     marker="v", s=110, color=color,
                     edgecolor="black", linewidth=1.0, zorder=5)

    # right panel: per-(df, strat) markers, two-marker convention
    for strat, d in nadir_per_strat.items():
        Ps_arr = np.array(d["P"])
        ep_arr = np.array(d["ep"])
        tok_arr = np.array(d["tok"])
        # filled (epoch, left axis) — colored by df
        ax_n.scatter(Ps_arr, ep_arr, marker=STRAT_MARKER[strat], s=180,
                     c=d["color"], edgecolor="black", linewidth=0.8,
                     zorder=4, label=f"{STRAT_PRETTY[strat]}  (epoch)")
        # open (steps, right axis) — gray edge
        ax_n2.scatter(Ps_arr, tok_arr / BATCH_SIZE,
                      marker=STRAT_MARKER[strat], s=140,
                      facecolors="none", edgecolors="0.35", linewidth=1.4,
                      zorder=3)
    nad_ep = np.concatenate([nadir_per_strat[s]["ep"] for s in STRAT_MARKER])
    nad_tok = np.concatenate([nadir_per_strat[s]["tok"] for s in STRAT_MARKER])
    nad_steps = nad_tok / BATCH_SIZE

    # capture df-only handles BEFORE adding the marker / line-style explainers
    df_handles, df_labels = ax_t.get_legend_handles_labels()

    # cosmetics
    ax_t.set_xlabel(r"steps  $s$", fontsize=28)
    ax_t.set_ylabel(r"$\mathcal{L}$", fontsize=28)
    ax_t.set_xlim(0, None)
    # train collapses to ~0.1 for small df; keep zero-baseline so divergence reads
    ax_t.set_ylim(0, 8)
    ax_t.set_title(r"(A)  val + train loss vs steps", fontsize=22, loc="left")

    # inline legend: line-style + marker explainers (panel A only)
    style_handles = [
        plt.Line2D([], [], color="0.4", lw=3.0, ls="-",  label="val loss"),
        plt.Line2D([], [], color="0.4", lw=1.2, ls="--", alpha=0.55,
                   label="train loss"),
        plt.Line2D([], [], color="0.5", marker="v", linestyle="",
                   markeredgecolor="black", markersize=10, label="val nadir"),
    ]
    ax_t.legend(handles=style_handles, loc="upper right",
                frameon=True, framealpha=0.92, fontsize=18)

    ax_e.set_xlabel("epoch", fontsize=28)
    ax_e.set_ylabel(r"$\mathcal{L}$", fontsize=28)
    # most U-curves' meaningful range sits in 0–30 epochs; cap at 50 so the
    # nadirs aren't squashed against the y-axis
    ax_e.set_xlim(0, 50)
    ax_e.set_ylim(0, 8)
    ax_e.set_title("(B)  val + train loss vs epoch", fontsize=22, loc="left")
    # mirror panel-A's val / train / nadir legend so each panel is self-explanatory
    style_handles_b = [
        plt.Line2D([], [], color="0.4", lw=3.0, ls="-",  label="val loss"),
        plt.Line2D([], [], color="0.4", lw=1.2, ls="--", alpha=0.55,
                   label="train loss"),
        plt.Line2D([], [], color="0.5", marker="v", linestyle="",
                   markeredgecolor="black", markersize=10, label="val nadir"),
    ]
    ax_e.legend(handles=style_handles_b, loc="upper right",
                frameon=True, framealpha=0.92, fontsize=18)

    # one shared color-by-df legend, horizontal, below all three panels
    fig.legend(df_handles, df_labels, loc="lower center", frameon=True,
               framealpha=0.92, fontsize=18, bbox_to_anchor=(0.5, 0.03),
               ncol=5, title=None)

    ax_n.set_xscale("log")
    ax_n.set_xlabel(r"unique tokens $P$", fontsize=28)
    ax_n.set_ylabel(r"nadir epoch  $\mathcal{E}^\ast$  (filled)",
                    color="black", fontsize=24)
    ax_n2.set_ylabel(r"nadir steps  $s^\ast$  (open)",
                     color="0.35", fontsize=24)
    ax_n.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1e6)}M"))
    ax_n.set_title("(C)  overfit-onset position vs $P$",
                   fontsize=22, loc="left")
    ax_n.set_ylim(max(0, nad_ep.min() - 1.5), nad_ep.max() + 1.5)
    ax_n2.set_ylim(0, nad_steps.max() * 1.10)
    ax_n.grid(True, alpha=0.25)
    ax_n.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=14)

    out_pdf = Path(args.out).with_suffix(".pdf")
    out_png = Path(args.out).with_suffix(".png")
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    print(f"saved {out_pdf}")
    print(f"saved {out_png}")

    # text summary per (df, strategy)
    print()
    print(f"{'df':>5}  {'P (M)':>7}  {'strategy':>16}  "
          f"{'nadir epoch':>12}  {'nadir steps':>14}")
    for strat in STRAT_MARKER:
        d = nadir_per_strat[strat]
        for P, ep_, tok_ in zip(d["P"], d["ep"], d["tok"]):
            df_ = P / CORPUS_TOKENS
            print(f"{df_:>5.2f}  {P/1e6:>7.0f}  {strat:>16}  "
                  f"{ep_:>12.2f}  {tok_/BATCH_SIZE:>14.0f}")


if __name__ == "__main__":
    main()
