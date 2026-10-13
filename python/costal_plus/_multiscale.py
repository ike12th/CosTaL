"""``find_communities_multiscale()``: an opt-in answer to the resolution limit (docs/05 §2.3).

No single Leiden resolution recovers both the large and the rare populations: a low one swallows the
rare ones, a high one shatters the large ones (docs/05 §1.1). This runs Leiden at a range of
resolutions on one exact kNN graph and keeps, from coarse to fine, every finer cluster that is
cohesive on its own: a candidate C is carved out of the cluster H currently hosting it when less
than ``phi_max`` of its edge weight goes to the rest of H,

    φ(C | H) = w(C, H∖C) / vol(C) ≤ phi_max.

``phi_max = 0.04`` was fixed on Levine_32 and then checked unchanged on six other datasets (docs/05
§2.3): the rare populations come out better than at any single resolution on six of seven, including
the paper's own rare-population benchmarks, where it also beats the supplement's two-level option.
Not part of the published method; ``compat="paper"`` refuses it.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from . import _costal_plus as _ext
from ._api import knn, reweight

__all__ = ["RESOLUTIONS", "find_communities_multiscale"]

# 0.1 … 3.2, ratio √2 (docs/05 §2.3)
RESOLUTIONS = tuple(round(0.1 * 2 ** (i / 2), 3) for i in range(11))
_NESTED, _PROPER = 0.8, 0.9


def _ensemble(graph, resolutions, seeds, leiden: str, n_iterations: int) -> np.ndarray:
    out = np.empty((len(resolutions), len(seeds), graph.vcount()), dtype=np.int64)
    for r, res in enumerate(resolutions):
        for s, seed in enumerate(seeds):
            if leiden == "fast":
                random.seed(seed)
                m = graph.community_leiden(objective_function="modularity", weights="weight",
                                           resolution=res, n_iterations=n_iterations).membership
            else:
                import leidenalg as la

                m = la.find_partition(graph, la.RBConfigurationVertexPartition, weights="weight",
                                      n_iterations=n_iterations, resolution_parameter=res,
                                      seed=seed).membership
            out[r, s] = m
    return out


def _carve(ms: np.ndarray, rows: np.ndarray, cols: np.ndarray, w: np.ndarray, deg: np.ndarray,
           phi_max: float, min_size: int, resolutions) -> tuple[np.ndarray, list[dict]]:
    R, S, _ = ms.shape
    counts = [len(np.unique(ms[0, s])) for s in range(S)]
    labels = ms[0, int(np.argsort(counts)[len(counts) // 2])].copy()  # coarsest level, median seed
    next_label = labels.max() + 1
    carved: list[dict] = []

    def check(cells: np.ndarray) -> tuple[np.ndarray, float] | None:
        hosts = np.bincount(labels[cells])
        host = int(hosts.argmax())
        host_mask = labels == host
        in_host = cells & host_mask
        if hosts[host] < _NESTED * cells.sum() or in_host.sum() > _PROPER * host_mask.sum():
            return None
        rest = host_mask & ~in_host
        cut = w[(in_host[rows] & rest[cols]) | (in_host[cols] & rest[rows])].sum()
        return in_host, float(cut / deg[in_host].sum())

    for r in range(1, R):
        scored = []
        for s in range(S):
            m = ms[r, s]
            for c in np.flatnonzero(np.bincount(m) >= min_size):
                hit = check(m == c)
                if hit is not None and hit[1] <= phi_max:
                    scored.append((hit[1], s, int(c)))
        for _, s, c in sorted(scored):
            hit = check(ms[r, s] == c)  # the hosts may have changed since scoring
            if hit is None or hit[1] > phi_max:
                continue
            labels[hit[0]] = next_label
            next_label += 1
            carved.append({"resolution": resolutions[r], "seed_index": s, "cells": int(hit[0].sum()),
                           "phi": hit[1]})
    return np.unique(labels, return_inverse=True)[1].astype(np.int64), carved


def find_communities_multiscale(
    X: Any,
    k: int = 10,
    *,
    phi_max: float = 0.04,
    select: str = "cosine",
    weight: str = "tanimoto",
    normalize: str | None = None,
    resolutions: tuple | list | None = None,
    seeds: tuple | list | int = 4,
    leiden: str = "fast",
    n_iterations: int = 2,
    min_size: int | None = None,
    compat: str | None = None,
    n_threads: int | None = None,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Multi-resolution clustering that keeps cohesive rare populations (opt-in, docs/05 §2.3).

    The exact kNN graph is built once (as in :func:`find_communities`), Leiden runs at every
    resolution × seed, and finer clusters are carved out of coarser ones when at most ``phi_max`` of
    their edge weight leaks into the rest of their parent.

    Args:
        phi_max: the leak threshold; 0.04 by default (fixed on one dataset, checked on six others).
            Lower carves less.
        resolutions: Leiden resolutions, ascending; default 0.1 … 3.2 with ratio √2.
        seeds: Leiden seeds per resolution, or how many (``0 … seeds−1``).
        leiden: ``"fast"`` (igraph's native Leiden, default) or ``"paper"`` (leidenalg).
        n_iterations: Leiden iterations per run (2 reproduces the paper's 10, docs/03 §9.1).
        min_size: smallest candidate cluster; default ``2 * k``.
        return_info: also return a dict with the carved clusters and the ensemble's settings.

    Returns:
        ``(n,)`` int64 labels, and the info dict if ``return_info``. Costs one kNN plus
        ``len(resolutions) × seeds`` Leiden runs (about 80 s for 161k cells with ``"fast"``).
    """
    if compat == "paper":
        raise ValueError("compat='paper' pins the published single-resolution method; "
                         "find_communities_multiscale is not part of it (docs/05 §2.3)")
    if leiden not in ("fast", "paper"):
        raise ValueError(f"leiden must be 'fast' or 'paper', got {leiden!r}")
    if not 0 < phi_max < 1:
        raise ValueError(f"phi_max must be in (0, 1), got {phi_max}")
    resolutions = tuple(RESOLUTIONS if resolutions is None else resolutions)
    if list(resolutions) != sorted(resolutions) or len(resolutions) < 2:
        raise ValueError("resolutions must be ascending, at least two")
    seeds = tuple(range(seeds)) if isinstance(seeds, int) else tuple(seeds)
    min_size = 2 * k if min_size is None else int(min_size)
    try:
        import igraph as ig
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("find_communities_multiscale needs igraph: pip install costal-plus[leiden]") from exc

    g = knn(X, k, select=select, normalize=normalize, compat=compat, n_threads=n_threads)
    rows, cols, w = _ext._fuzzy_union(g.idx, reweight(g, weight))
    rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
    graph = ig.Graph(n=g.n, edges=np.column_stack([rows, cols]).tolist(), edge_attrs={"weight": w.tolist()})
    deg = np.bincount(rows, weights=w, minlength=g.n) + np.bincount(cols, weights=w, minlength=g.n)

    ms = _ensemble(graph, resolutions, seeds, leiden, n_iterations)
    labels, carved = _carve(ms, rows, cols, np.asarray(w), deg, phi_max, min_size, resolutions)
    if not return_info:
        return labels
    return labels, {"phi_max": phi_max, "resolutions": resolutions, "seeds": seeds, "leiden": leiden,
                    "n_iterations": n_iterations, "min_size": min_size, "carved": carved,
                    "clusters": int(labels.max() + 1)}
