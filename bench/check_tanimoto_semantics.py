#!/usr/bin/env python3
"""Check the claims in docs/01 §2.2 about Tanimoto and magnitude.

The claim splits into two statements, checked separately:
 A) on L2-normalized vectors T = c/(2-c), a strictly monotone transform of cosine (neighbor
    ranking unchanged);
 B) in which space does the original CosTaL compute Tanimoto — normalized, or with magnitude?

(B) can be settled by reading the code: `find_communities` writes the **normalized** X to the
.ijv for p_knng to select neighbors, but `build_tani_graph(clu, X)` receives the **un-normalized**
X, and `cell_norm = norm(X, axis=1)` is the raw magnitude. Here we confirm it with numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "reference" / "CosTaL-0.0.2.4"))

from CosTaL.clustering import pp, tani_from_cos  # noqa: E402
from profile_costal import knn_blas_blocked, load_levine  # noqa: E402


def main() -> int:
    x = pp(load_levine(20000), platform="mass")
    idx, cos = knn_blas_blocked(x, 10)

    norms = np.linalg.norm(x, axis=1)
    rows = np.repeat(np.arange(len(x)), idx.shape[1])
    cols = idx.ravel()
    c = cos.ravel().astype(np.float64)

    # What the original CosTaL actually computes (magnitude kept)
    t_costal = np.array([tani_from_cos(norms[i], norms[j], v)
                         for i, j, v in zip(rows, cols, c)])
    # What L2-normalizing before Tanimoto would give (the degenerate form)
    t_degen = c / (2.0 - c)

    print(f"pairs                  : {len(c)}")
    print(f"cos                    : [{c.min():.4f}, {c.max():.4f}]")
    print(f"row norm ‖x‖           : [{norms.min():.3f}, {norms.max():.3f}]  "
          f"(ratio max/min = {norms.max()/norms.min():.1f})")
    print(f"T_CosTaL (with magnitude) : [{t_costal.min():.4f}, {t_costal.max():.4f}]")
    print(f"T = c/(2-c) (normalized)  : [{t_degen.min():.4f}, {t_degen.max():.4f}]")
    print(f"max abs difference        : {np.abs(t_costal - t_degen).max():.4f}")
    print(f"Pearson r                 : {np.corrcoef(t_costal, t_degen)[0,1]:.4f}")
    print(f"same Spearman order?      : "
          f"{np.array_equal(np.argsort(t_costal), np.argsort(t_degen))}")

    # A) monotonicity: c/(2-c) is strictly increasing on c∈[-1,1)
    cc = np.linspace(-0.99, 0.999, 1000)
    print(f"c/(2-c) strictly increasing? : {bool(np.all(np.diff(cc/(2-cc)) > 0))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
