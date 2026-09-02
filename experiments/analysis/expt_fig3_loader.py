"""Shared loader for Expt 3 figures.

Reads the 4×4 (depth × width) grid from data_export/expt3_grid/ and returns
a tidy nested dict. All Expt 3 figures (3.1 grid, 3.2 heatmap, 3.3 compute-
matched, 3.4 per-cell b, 3.5 scaling-law fit) reuse this.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field

import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO / "data_export" / "expt3_grid"

DEPTHS = (6, 12, 18, 24)
WIDTHS = (384, 768, 1152, 1536)
STRATEGIES = ("init_ens", "init_shuffle_ens")
N_INDIV = 5
E_SIZES = (2, 3, 4, 5)
TOKENS_PER_EPOCH_DF02 = 19_988_480  # exact from npz["tokens"][0] at any cell


@dataclass
class CellData:
    d: int
    w: int
    strat: str
    individuals: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    # ensembles[E] -> (tokens, val_loss)
    ensembles: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def n_indiv(self) -> int:
        return len(self.individuals)

    def n_epochs(self) -> int:
        if self.individuals:
            return len(self.individuals[0][0])
        if self.ensembles:
            return len(next(iter(self.ensembles.values()))[0])
        return 0

    def individual_array(self) -> tuple[np.ndarray, np.ndarray]:
        """Stack individuals into (n_inds, n_epochs) array. Token grid taken from individual 0."""
        if not self.individuals:
            return np.array([]), np.array([])
        toks = self.individuals[0][0]
        L = np.stack([vl for _, vl in self.individuals], axis=0)
        return toks, L


def load_grid(data_dir: Path = DEFAULT_DATA_DIR) -> dict[tuple[int, int, str], CellData]:
    """Return {(d, w, strat): CellData}. Missing files silently skipped."""
    ind_dir = data_dir / "individuals"
    ens_dir = data_dir / "ensembles"
    grid: dict[tuple[int, int, str], CellData] = {}
    for d in DEPTHS:
        for w in WIDTHS:
            for strat in STRATEGIES:
                cell = CellData(d=d, w=w, strat=strat)
                for m in range(N_INDIV):
                    p = ind_dir / f"d{d}_w{w}_{strat}_model{m}.npz"
                    if p.exists():
                        npz = np.load(p, allow_pickle=True)
                        cell.individuals.append((npz["tokens"].copy(), npz["val_loss"].copy()))
                for E in E_SIZES:
                    p = ens_dir / f"d{d}_w{w}_{strat}_E{E}.npz"
                    if p.exists():
                        npz = np.load(p, allow_pickle=True)
                        cell.ensembles[E] = (npz["tokens"].copy(), npz["val_loss"].copy())
                grid[(d, w, strat)] = cell
    return grid


def grid_summary(grid):
    print("Cell coverage (n_indiv | E={2,3,4,5}):")
    for d in DEPTHS:
        for w in WIDTHS:
            for strat in STRATEGIES:
                c = grid[(d, w, strat)]
                marks = "".join("Y" if E in c.ensembles else "." for E in E_SIZES)
                print(f"  d={d:>2} w={w:>4} {strat:<18}  ind={c.n_indiv()} ens={marks} epochs={c.n_epochs()}")


if __name__ == "__main__":
    grid_summary(load_grid())
