"""The CosTaL 0.0.2 interface, kept in CosTaL 1.0 so that 0.0.2 scripts run unchanged.

``pp``, ``clustering`` and the 0.0.2 form of ``find_communities`` keep their signatures and return a
Python list of labels; the evaluation helpers of ``CosTaL.descriptions`` (``return_scores`` and
friends) are ported as they were. What runs underneath is the 1.0 implementation: the exact kNN of
the published method on every platform (docs/user_guide.md §11 lists the differences from 0.0.2).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._api import _find_communities, build_graph
from ._preprocess import COFACTORS, preprocess

__all__ = [
    "calculate_sparsity",
    "clustering",
    "find_communities",
    "get_f1_matrix",
    "get_rare_f1",
    "normalize_arcsinh",
    "pp",
    "return_Ff1_score",
    "return_Hf1_score",
    "return_scores",
]

_METHODS = {"tani": "tanimoto", "cosine": "cosine", "jaccard": "jaccard"}


def normalize_arcsinh(X: Any, base: float = 5) -> np.ndarray:
    """``arcsinh(X / base)``: base 5 for mass cytometry, 150 for flow."""
    return np.arcsinh(np.asarray(X, dtype=np.float64) / base)


def pp(X: Any, platform: str | None = None) -> Any:
    """The original preprocessing: ``"mass"``, ``"flow"``, ``"spectral flow"``, ``"scrnaseq"``, or
    ``None`` to take the numeric columns as they are."""
    if platform in (*COFACTORS, "scrnaseq"):
        return preprocess(X, platform)
    # anything else, as the original: numeric columns, values untouched
    return X.select_dtypes([np.number]).to_numpy() if hasattr(X, "select_dtypes") else X


def find_communities(X: Any, pp_method: str | None = None, method: str = "tani", nbr_num: int = 10,
                     n_threads: int | None = None, verbose: bool = False, keep_temp_files: bool = False,
                     temp_name: str = "", partition_type: Any = None, resolution: float = 0.8,
                     seed: int | None = None, local: bool = False, **kwargs: Any) -> list:
    """CosTaL's ``find_communities``: preprocess, exact cosine kNN, edge weights, Leiden (10
    iterations). ``keep_temp_files`` / ``temp_name`` / ``verbose`` are accepted and ignored (there
    are no temporary files). Returns the cluster label of each row, as a list."""
    if local:
        raise NotImplementedError(
            "local=True is not carried over: the original passed similarities where UMAP expects "
            "distances, and the corrected version gave no consistent gain (docs/05 §2.2). For rare "
            "populations use CosTaL.find_communities_multiscale.")
    if method not in _METHODS:
        raise RuntimeError("Method not supported")
    X = pp(X, platform=pp_method)
    if hasattr(X, "toarray"):
        X = X.toarray()
    if partition_type is None or getattr(partition_type, "__name__", "") == "RBConfigurationVertexPartition":
        labels = _find_communities(X, nbr_num, weight=_METHODS[method], resolution=resolution, seed=seed,
                                   n_threads=n_threads)
        return labels.tolist()
    # Another leidenalg partition type: the same graph, the original's call
    import igraph as ig
    import leidenalg as la

    rows, cols, w = build_graph(X, nbr_num, weight=_METHODS[method], n_threads=n_threads)
    graph = ig.Graph(n=len(X), edges=np.column_stack([rows, cols]), edge_attrs={"weight": w})
    options = {"resolution_parameter": resolution} if hasattr(partition_type, "resolution_parameter") else {}
    part = la.find_partition(graph, partition_type, weights="weight", n_iterations=10, seed=seed, **options)
    return list(part.membership)


def clustering(X: Any, nbr_num: int = 10, seed: int | None = None, local: bool = False, method: str = "tani",
               pp_method: str | None = None, **kwargs: Any) -> list:
    """CosTaL's ``clustering``. The original ran ``find_communities`` on Linux and a brute-force path
    on other systems; CosTaL 1.0 runs the published (Linux) path everywhere."""
    return find_communities(X, pp_method=pp_method, method=method, nbr_num=nbr_num, seed=seed, local=local,
                            **kwargs)


# ---------------------------------------------------------------- CosTaL.descriptions

def _labels(y: Any) -> list:
    if isinstance(y, list) and any(isinstance(v, list) for v in y):
        return ["_".join(map(str, v)) for v in y]
    return list(y)


def get_f1_matrix(ytrue: Any, ypred: Any) -> np.ndarray:
    """F1 between every true class (rows) and every predicted cluster (columns); rows with a NaN
    true label are dropped."""
    ytrue = np.asarray(ytrue, dtype=float)
    keep = ~np.isnan(ytrue)
    t_cls, t = np.unique(ytrue[keep], return_inverse=True)
    p_cls, p = np.unique(np.asarray(ypred)[keep], return_inverse=True)
    conf = np.zeros((len(t_cls), len(p_cls)))
    np.add.at(conf, (t, p), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.nan_to_num(conf / conf.sum(axis=0, keepdims=True))
        recall = np.nan_to_num(conf / conf.sum(axis=1, keepdims=True))
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    return f1


def return_Hf1_score(ytrue: Any, ypred: Any) -> float:  # noqa: N802 - the original's name
    """Mean F1 of the true classes under the Hungarian (one-to-one) matching."""
    from scipy.optimize import linear_sum_assignment

    f1 = get_f1_matrix(ytrue, ypred)
    rows, cols = linear_sum_assignment(1 - f1)
    matched = dict(zip(rows, cols))
    return float(np.mean([f1[i, matched[i]] if i in matched else 0.0 for i in range(f1.shape[0])]))


def return_Ff1_score(ytrue: Any, ypred: Any) -> float:  # noqa: N802 - the original's name
    """FlowCAP F1: each true class's best F1, weighted by its size."""
    f1 = get_f1_matrix(ytrue, ypred)
    ytrue = np.asarray(ytrue, dtype=float)
    sizes = np.unique(ytrue[~np.isnan(ytrue)], return_counts=True)[1]
    return float(np.sum(sizes / sizes.sum() * f1.max(axis=1)))


def return_scores(ytrue: Any, ypred: Any) -> tuple:
    """AMI, ARI, FlowCAP F1, Hungarian F1, Fowlkes–Mallows, homogeneity, completeness, V-measure,
    and the number of singleton, small (≤ 10 cells), decent and all clusters — as the original."""
    import pandas as pd
    from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score,
                                 homogeneity_completeness_v_measure)
    from sklearn.preprocessing import LabelEncoder

    ytrue, ypred = pd.Series(_labels(ytrue)), pd.Series(_labels(ypred))
    keep = ytrue.notna()
    original = LabelEncoder().fit_transform(ytrue[keep])
    result = LabelEncoder().fit_transform(ypred[keep])
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(original, result)
    sizes = np.unique(ypred, return_counts=True)[1]
    return (
        adjusted_mutual_info_score(original, result),
        adjusted_rand_score(original, result),
        return_Ff1_score(original, result),
        return_Hf1_score(original, result),
        fowlkes_mallows_score(original, result),
        homogeneity, completeness, v_measure,
        int(np.sum(sizes == 1)),
        int(np.sum(sizes <= 10) - np.sum(sizes == 1)),
        int(len(sizes) - np.sum(sizes <= 10)),
        int(len(sizes)),
    )


def get_rare_f1(ytrue: Any, ypred: Any) -> float:
    """The best F1 of the second true class (label-encoded order), the original's rare-population
    score for the one-population datasets."""
    from sklearn.preprocessing import LabelEncoder

    original = LabelEncoder().fit_transform([str(v) for v in ytrue])
    result = LabelEncoder().fit_transform([str(v) for v in ypred])
    return float(get_f1_matrix(original, result)[1, :].max())


def calculate_sparsity(data: Any) -> float:
    """Share of exactly-zero entries."""
    data = np.asarray(data)
    return 1.0 - np.count_nonzero(data) / float(data.size)
