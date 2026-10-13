"""Public API: ``knn()`` / ``reweight()`` / ``truncate()`` / ``build_graph()`` (docs/04 §6).

No optional arguments = the published CosTaL: select neighbors by cosine (exact kNN, k excludes
the row itself), weight edges by Tanimoto with the un-normalized row norms, then symmetrize by
fuzzy union (docs/04 §0.0).

kNN is the only expensive step; edge weights and truncation work on the same kNN and cost
almost nothing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import _costal_plus as _ext

__all__ = [
    "CostalWarning",
    "KnnGraph",
    "build_graph",
    "find_communities",
    "knn",
    "reweight",
    "truncate",
]


class CostalWarning(UserWarning):
    """A note from the input diagnostics: results are still correct, but a parameter may not mean
    what you think it means."""


@dataclass(frozen=True)
class KnnGraph:
    """An exact kNN.

    Attributes:
        idx: ``(n, k)`` int32 original row indices; each row sorted by (cosine desc, index asc).
        cos: ``(n, k)`` float64 cosine after f64 rescoring.
        norms: ``(n,)`` float64 raw (un-normalized) row norms, used by Tanimoto edge weights.
        meta: parameters, statistics and input diagnostics (docs/04 §6.6).
    """

    idx: np.ndarray
    cos: np.ndarray
    norms: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.idx.shape[0]

    @property
    def k(self) -> int:
        return self.idx.shape[1]


_WEIGHTS = ("tanimoto", "cosine", "jaccard")


def _check_choices(
    select: str, normalize: str | None, backend: str, compat: str | None
) -> str:
    """Validate the knobs and return the effective normalization."""
    if compat not in (None, "paper"):
        raise ValueError(f"compat must be None or 'paper', got {compat!r}")
    if select not in ("cosine", "tanimoto"):
        raise ValueError(f"select must be 'cosine' or 'tanimoto', got {select!r}")
    if normalize is None:
        # Cosine is scale-invariant, so L2 is just its natural description; Tanimoto selection
        # only means something on the raw magnitudes (docs/04 §0.1)
        normalize = "l2" if select == "cosine" else "none"
    if normalize not in ("l2", "none"):
        raise ValueError(f"normalize must be 'l2' or 'none', got {normalize!r}")
    if backend in ("approx", "hnsw"):
        raise NotImplementedError("approximate backends are stage E and never enter the default path (docs/03 §6)")
    if backend == "sparse":
        raise NotImplementedError("the sparse backend is stage S (docs/06 §5)")
    if backend != "exact":
        raise ValueError(f"backend must be 'exact', got {backend!r}")
    return normalize


def _as_matrix(X: Any) -> np.ndarray:
    if hasattr(X, "tocsr") or hasattr(X, "toarray"):
        raise TypeError(
            "sparse input needs the sparse backend (stage S, docs/06 §5), not implemented yet; "
            "use .toarray() or reduce dimensionality first"
        )
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D matrix, got {X.ndim} dimensions")
    if not (np.issubdtype(X.dtype, np.floating) or np.issubdtype(X.dtype, np.integer)):
        raise TypeError(f"X must be a real-valued matrix, got dtype={X.dtype}")
    return np.ascontiguousarray(X, dtype=np.float32)


def _diagnose(X: np.ndarray) -> dict[str, Any]:
    """Input diagnostics: record only, never modify the data (docs/04 §6.6)."""
    norms = np.sqrt(np.einsum("ij,ij->i", X, X, dtype=np.float64))
    mean_norm = norms.mean()
    col_mean = X.mean(axis=0, dtype=np.float64)
    col_std = X.std(axis=0, dtype=np.float64)
    live = col_std > 0
    return {
        # ≈ 0 ⇒ rows are already unit-normalized, so Tanimoto ≡ c/(2−c)
        "row_norm_cv": float(norms.std() / mean_norm) if mean_norm > 0 else 0.0,
        # Every non-constant column has mean ≪ std ⇒ looks like PCA / z-score / Pearson residuals
        "columns_centered": bool(
            live.any() and np.all(np.abs(col_mean[live]) <= 0.05 * col_std[live])
        ),
        "negative_fraction": float(np.count_nonzero(X < 0) / X.size),
        "nonzero_fraction": float(np.count_nonzero(X) / X.size),
    }


def knn(
    X: Any,
    k: int = 10,
    *,
    select: str = "cosine",
    normalize: str | None = None,
    backend: str = "exact",
    compat: str | None = None,
    n_threads: int | None = None,
) -> KnnGraph:
    """Exact kNN (docs/04 §6.1).

    Selects each row's k neighbors (excluding the row itself) by cosine — the published CosTaL —
    or by Tanimoto on the raw rows (``select="tanimoto"``, docs/04 §2). All N² similarities are
    computed, and the final ranking comes from fixed-order f64 recomputation: results are
    bit-identical across platforms and thread counts (docs/06 §1).

    Args:
        X: ``(n, d)`` dense real-valued matrix, converted to float32 internally.
        k: neighbors per row, ``1 ≤ k ≤ n-1``.
        select: ``"cosine"`` (default) or ``"tanimoto"``.
        normalize: ``"l2"`` or ``"none"``; defaults to ``"l2"`` for cosine and ``"none"`` for
            Tanimoto. Tanimoto on L2-normalized rows is a monotone function of cosine, so
            ``select="tanimoto", normalize="l2"`` runs cosine and warns (docs/04 §0.1).
        compat: ``"paper"`` explicitly pins the semantics of the published CosTaL (docs/04 §6.5).
            The defaults currently are the paper's semantics; this only records the intent in
            the metadata.
        n_threads: number of threads; ``None`` uses all cores.
    """
    normalize = _check_choices(select, normalize, backend, compat)
    effective = select
    if select == "tanimoto" and normalize == "l2":
        warnings.warn(
            "select='tanimoto' with normalize='l2' is equivalent to cosine selection (on unit rows "
            "T = c/(2−c) is monotone in c); running cosine (docs/04 §0.1)",
            CostalWarning,
            stacklevel=2,
        )
        effective = "cosine"
    X = _as_matrix(X)
    (idx, cos, norms), stats = _ext._knn_exact(X, int(k), select=effective, n_threads=n_threads)
    meta = {
        "k": int(k),
        "select": select,
        "select_effective": effective,
        "normalize": normalize,
        "backend": backend,
        "compat": compat,
        "version": _ext.__version__,
        "stats": dict(stats),
        "diagnostics": _diagnose(X),
    }
    if effective == "tanimoto":
        # The selection itself uses magnitude here, so warn now rather than at reweight()
        _warn_about_magnitude(meta)
    return KnnGraph(idx=idx, cos=cos, norms=norms, meta=meta)


def _warn_about_magnitude(meta: dict[str, Any]) -> None:
    diag = meta.get("diagnostics")
    if not diag:
        return
    if diag["row_norm_cv"] < 1e-4:
        warnings.warn(
            "row norms are nearly constant (rows already unit-normalized?): Tanimoto edge weights "
            "reduce to the monotone cosine transform c/(2−c) and add no information (docs/04 §0.1)",
            CostalWarning,
            stacklevel=3,
        )
    if diag["columns_centered"]:
        warnings.warn(
            "columns are centered (PCA / z-score / Pearson residuals?): the row norm means "
            "\"distance from the data center\" rather than abundance, so Tanimoto edge weights change "
            "meaning and can be negative (docs/04 §6.6)",
            CostalWarning,
            stacklevel=3,
        )


def reweight(knn: KnnGraph, weight: str = "tanimoto") -> np.ndarray:
    """Edge weights on the same kNN, as ``(n, k)`` float64 (docs/04 §6.1).

    - ``"tanimoto"``: ``T = c·‖a‖‖b‖ / (‖a‖² + ‖b‖² − c·‖a‖‖b‖)`` with the raw row norms — the
      published CosTaL;
    - ``"cosine"``: the rescored cosine;
    - ``"jaccard"``: ``s / (2k − s)`` for s shared neighbors; compares neighbors of neighbors,
      O(N·k²).

    Reweighting only changes the numbers on edges, never which edges exist (docs/04 §6.3).
    """
    if weight not in _WEIGHTS:
        raise ValueError(f"weight must be one of {_WEIGHTS}, got {weight!r}")
    if weight == "tanimoto":
        _warn_about_magnitude(knn.meta)
    w = _ext._reweight(knn.idx, knn.cos, knn.norms, weight)
    n_negative = int(np.count_nonzero(w < 0))
    if n_negative:
        warnings.warn(
            f"{n_negative} edges have negative weights (Tanimoto's lower bound on signed input is "
            "−1/3). Paper mode keeps them as is; how to handle them is to be decided by experiment "
            "(docs/04 §6.6)",
            CostalWarning,
            stacklevel=2,
        )
    return w


def truncate(knn: KnnGraph, k: int) -> KnnGraph:
    """Keep the first k neighbors of each row. Rows are totally ordered by (cosine desc, index asc),
    so the result **is** the exact kNN for the smaller k, not an approximation (docs/04 §6.1)."""
    if not 1 <= k <= knn.k:
        raise ValueError(f"k must be within 1..{knn.k}, got {k}")
    meta = {**knn.meta, "k": int(k), "truncated_from": knn.k}
    return KnnGraph(
        idx=np.ascontiguousarray(knn.idx[:, :k]),
        cos=np.ascontiguousarray(knn.cos[:, :k]),
        norms=knn.norms,
        meta=meta,
    )


def build_graph(
    X: Any,
    k: int = 10,
    *,
    select: str = "cosine",
    weight: str = "tanimoto",
    normalize: str | None = None,
    backend: str = "exact",
    symmetrize: str = "fuzzy_union",
    compat: str | None = None,
    n_threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One call: kNN → edge weights → symmetrization, returning the upper-triangular edge list
    ``(rows, cols, weights)``.

    Symmetrization is the original's fuzzy union ``S + Sᵀ − S∘Sᵀ`` (**not** ``(S + Sᵀ)/2``), with
    self-loops and exactly-zero edges removed; edges are in ascending (row, col) order, matching
    the original scipy pipeline bit for bit (docs/04 §0.0).

    Hand it to Leiden the same way as the original::

        rows, cols, w = costal_plus.build_graph(X)
        G = igraph.Graph(n=len(X), edges=list(zip(rows.tolist(), cols.tolist())),
                         edge_attrs={"weight": w.tolist()})
    """
    if symmetrize != "fuzzy_union":
        raise ValueError(f"symmetrize currently supports only 'fuzzy_union', got {symmetrize!r}")
    g = knn(
        X,
        k,
        select=select,
        normalize=normalize,
        backend=backend,
        compat=compat,
        n_threads=n_threads,
    )
    w = reweight(g, weight)
    return _ext._fuzzy_union(g.idx, w)


_LEIDEN = {
    # The published CosTaL: leidenalg, RBConfiguration, 10 iterations (docs/04 §0.0 item 5)
    "paper": 10,
    # igraph's native Leiden on the same objective: 38x faster on Levine, same accuracy against the
    # gating labels, but a different implementation whose partitions differ more (docs/03 §9.1)
    "fast": 2,
}


def find_communities(
    X: Any,
    k: int = 10,
    *,
    select: str = "cosine",
    weight: str = "tanimoto",
    normalize: str | None = None,
    resolution: float = 0.8,
    leiden: str = "paper",
    n_iterations: int | None = None,
    seed: int | None = None,
    compat: str | None = None,
    n_threads: int | None = None,
) -> np.ndarray:
    """End to end: exact kNN → edge weights → fuzzy union → Leiden. Returns ``(n,)`` int64 labels.

    With no optional arguments this is the published CosTaL (docs/04 §0.0): cosine selection,
    Tanimoto weights, fuzzy-union symmetrization, then leidenalg's RBConfiguration partition at
    resolution 0.8 with 10 iterations.

    Args:
        leiden: ``"paper"`` uses leidenalg exactly as the original; ``"fast"`` uses igraph's native
            Leiden on the same modularity objective (docs/03 §9.1).
        n_iterations: Leiden iterations; ``None`` means 10 for ``"paper"`` and 2 for ``"fast"``.
            Negative runs until the partition is stable.
        seed: Leiden seed. For ``"fast"`` it seeds Python's ``random`` module, which python-igraph
            draws from.
        compat: ``"paper"`` refuses any setting that departs from the published method.

    Needs ``igraph`` (and ``leidenalg`` for ``"paper"``): ``pip install costal-plus[leiden]``.
    """
    if leiden not in _LEIDEN:
        raise ValueError(f"leiden must be one of {tuple(_LEIDEN)}, got {leiden!r}")
    iterations = _LEIDEN[leiden] if n_iterations is None else int(n_iterations)
    if compat == "paper" and (
        leiden != "paper" or resolution != 0.8 or iterations != 10
        or select != "cosine" or weight != "tanimoto"
    ):
        raise ValueError(
            "compat='paper' pins cosine selection, Tanimoto weights and leidenalg with "
            "resolution 0.8 and 10 iterations (docs/04 §0.0)"
        )
    try:
        import igraph as ig
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("find_communities needs igraph: pip install costal-plus[leiden]") from exc

    g = knn(X, k, select=select, normalize=normalize, compat=compat, n_threads=n_threads)
    w = reweight(g, weight)
    rows, cols, ew = _ext._fuzzy_union(g.idx, w)
    graph = ig.Graph(n=g.n, edges=np.column_stack([rows, cols]).tolist(),
                     edge_attrs={"weight": ew.tolist()})
    if leiden == "paper":
        try:
            import leidenalg as la
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "leiden='paper' needs leidenalg: pip install costal-plus[leiden]"
            ) from exc
        part = la.find_partition(graph, la.RBConfigurationVertexPartition, weights="weight",
                                 n_iterations=iterations, resolution_parameter=resolution,
                                 seed=seed)
        membership = part.membership
    else:
        if seed is not None:
            import random

            random.seed(seed)
        membership = graph.community_leiden(objective_function="modularity", weights="weight",
                                            resolution=resolution,
                                            n_iterations=iterations).membership
    return np.asarray(membership, dtype=np.int64)
