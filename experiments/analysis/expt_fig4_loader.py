"""Shared loader for Expt 4 figures.

Reads the d=12, w=768 base-cell runs at varying df (data fraction) from
data_export/expt4_datasize/wd0_fixed_tokens/. Each (df, strat) is one
individual training run (~1B total tokens).
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np


REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data_export" / "expt4_datasize" / "wd0_fixed_tokens"

DFS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
STRATEGIES = ("init_ens", "init_shuffle_ens")
STRAT_PRETTY = {"init_ens": "init", "init_shuffle_ens": "init + shuffle"}

# Unique-token universe for df = 1.0 (full FineWeb subset used in this project).
TOTAL_UNIQUE_TOKENS = 100_000_000   # 100M


def df_to_P(df: float) -> int:
    """Unique tokens P for a given data fraction."""
    return int(round(TOTAL_UNIQUE_TOKENS * df))


@dataclass
class RunData:
    df: float
    strat: str
    tokens: np.ndarray   # (n_snap,)
    val_loss: np.ndarray # (n_snap,)
    epoch: np.ndarray    # (n_snap,)

    @property
    def P(self) -> int:
        return df_to_P(self.df)

    def min_val_loss(self) -> float:
        return float(self.val_loss.min())

    def min_val_idx(self) -> int:
        return int(self.val_loss.argmin())


def load_all(data_dir: Path = DATA_DIR) -> dict[tuple[float, str], RunData]:
    out = {}
    for df in DFS:
        for strat in STRATEGIES:
            p = data_dir / f"df{df}_{strat}.npz"
            if not p.exists():
                continue
            d = np.load(p, allow_pickle=True)
            tok, vl, ep = d["tokens"], d["val_loss"], d["epoch"]
            order = np.argsort(tok)
            out[(float(df), strat)] = RunData(
                df=float(df), strat=strat,
                tokens=tok[order].copy(),
                val_loss=vl[order].copy(),
                epoch=ep[order].copy(),
            )
    return out


if __name__ == "__main__":
    runs = load_all()
    print(f"Loaded {len(runs)} runs.\n")
    print(f"  {'df':>4}  {'P':>5}  {'strat':<18}  {'epochs':>6}  {'tokens':>9}  {'min val':>9}")
    for df in DFS:
        for strat in STRATEGIES:
            r = runs.get((df, strat))
            if r is None: continue
            print(f"  {df:>4}  {r.P/1e6:>4.0f}M  {strat:<18}  "
                  f"{r.epoch.max():>6}  {r.tokens.max()/1e9:>5.2f}B  "
                  f"{r.min_val_loss():>9.4f}")
