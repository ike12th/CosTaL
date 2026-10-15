"""``costal.leiden``: CosTaL's own Leiden on any weighted undirected graph given as an edge list.

The same implementation as ``find_communities(leiden="fast")`` (crates/costal-core/src/leiden.rs,
docs/07 §5), for graphs built elsewhere. Part of the stable API: no breaking changes within 2.x.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

from . import _costal as _ext
from ._api import _seed64

_OBJECTIVES = ("modularity", "cpm")


def _node_ids(a: Any, name: str, n: int) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {a.shape}")
    if a.size and not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"{name} must hold integer node ids, got dtype {a.dtype}")
    if a.size and (a.min() < 0 or a.max() >= n):
        bad = a[(a < 0) | (a >= n)][0]
        raise ValueError(f"{name} holds node id {bad}, outside 0 … {n - 1} (n_nodes={n})")
    return np.ascontiguousarray(a, dtype=np.uint32)


def leiden(
    n_nodes: int,
    rows: Any,
    cols: Any,
    weights: Any,
    *,
    objective: str = "modularity",
    resolution: float = 1.0,
    n_iterations: int = 2,
    beta: float = 0.01,
    seed: int | None = None,
    n_threads: int | None = None,
    initial_membership: Any = None,
    return_quality: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Leiden communities of an undirected weighted graph given as an edge list.

    The Leiden algorithm (Traag, Waltman & van Eck 2019), implemented in Rust as CosTaL's
    ``leiden="fast"`` (docs/07 §5): fast local moving, refinement with the paper's
    well-connectedness tests and randomness β, aggregation; every cluster is connected.

    Args:
        n_nodes: number of nodes; node ids are ``0 … n_nodes − 1``. Nodes without edges stay alone.
        rows, cols: integer arrays, one entry per edge (COO). Either every undirected edge once,
            in either orientation, or every edge in both directions with equal weights (a
            symmetric sparse matrix, e.g. ``scipy.sparse.coo_matrix(A)``'s ``row``/``col``).
            ``ValueError`` for anything else: a pair listed twice in the same direction (even
            with equal weights; combine duplicates first), the two directions of a pair with
            different weights, a mix of pairs given once and pairs given both ways, a self-loop,
            or a node id outside ``0 … n_nodes − 1``.
        weights: edge weights, finite and ≥ 0 (``ValueError`` otherwise). Edges of weight 0 are
            ignored.
        objective: ``"modularity"`` (leidenalg's RBConfiguration: node weights are the strengths,
            a pair costs γ k_i k_j / 2m) or ``"cpm"`` (the constant Potts model: a pair costs γ).
        resolution: γ ≥ 0. Higher gives more, smaller clusters.
        n_iterations: Leiden iterations; a negative value iterates until an iteration moves no
            node (capped at 1,000). 0 is an error.
        beta: randomness of the refinement, in edge-weight units (0.01, as igraph and the
            authors' reference implementation).
        seed: random seed (any integer, taken modulo 2⁶⁴); ``None`` draws one.
        n_threads: threads; default all cores. The result does not depend on it.
        initial_membership: optional labels, one integer per node (any values); the run starts
            from this partition instead of from every node alone.
        return_quality: also return the partition's quality.

    Returns:
        ``(n_nodes,)`` int64 cluster labels ``0 … k − 1``, numbered by first appearance in node
        order; with ``return_quality=True``, ``(labels, quality)``. The quality is the objective
        being optimized, at the given ``resolution``:

        - ``"modularity"``: Q_γ = (1/2m) Σ_ij (A_ij − γ k_i k_j / 2m) δ(c_i, c_j), igraph's and
          networkx's ``modularity(..., resolution=γ)``. It is the standard modularity Q only at
          γ = 1;
        - ``"cpm"``: Σ_{i≠j} (A_ij − γ) δ(c_i, c_j) over ordered pairs, i.e. Σ_c [2 w_c − γ n_c
          (n_c − 1)] with w_c the weight of the edges inside cluster c and n_c its size: exactly
          leidenalg's ``CPMVertexPartition.quality()``.

    The same inputs and seed give the same partition on every platform and at any ``n_threads``.
    """
    n = operator.index(n_nodes)
    if not 0 <= n < 2**32:
        raise ValueError(f"n_nodes must be in 0 … 2³² − 1, got {n}")
    if objective not in _OBJECTIVES:
        raise ValueError(f"objective must be one of {_OBJECTIVES}, got {objective!r}")
    rows = _node_ids(rows, "rows", n)
    cols = _node_ids(cols, "cols", n)
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError(f"weights must be one-dimensional, got shape {w.shape}")
    if not len(rows) == len(cols) == len(w):
        raise ValueError(f"rows, cols and weights differ in length: {len(rows)}, {len(cols)}, {len(w)}")
    initial = None
    if initial_membership is not None:
        m = np.asarray(initial_membership)
        if m.shape != (n,):
            raise ValueError(f"initial_membership must have one label per node ({n}), got shape {m.shape}")
        if m.size and not np.issubdtype(m.dtype, np.integer):
            raise ValueError(f"initial_membership must hold integer labels, got dtype {m.dtype}")
        initial = np.unique(m, return_inverse=True)[1].astype(np.uint32).reshape(-1)
    labels, quality, _, _, _ = _ext._leiden_coo(
        n, rows, cols, np.ascontiguousarray(w), objective=objective, resolution=float(resolution),
        beta=float(beta), n_iterations=operator.index(n_iterations), seed=_seed64(seed),
        initial=initial, n_threads=n_threads,
    )
    return (labels, float(quality)) if return_quality else labels
