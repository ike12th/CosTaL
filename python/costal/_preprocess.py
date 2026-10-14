"""``preprocess()``: the original CosTaL ``pp()``, plus one opt-in scRNA-seq preset (docs/04 §6.7).

Cytometry is an arcsinh with the platform's cofactor; scRNA-seq is scanpy's ``filter_cells`` →
``filter_genes`` → 2000 HVGs by ``seurat_v3`` → ``log1p``. Unlike the original, which returns only
the matrix, a dropped cell is reported (and its index available), so labels can't silently shift.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ._api import CostalWarning, _row_chunks

__all__ = ["COFACTORS", "PRESETS", "preprocess"]

# arcsinh cofactors of the original pp()
COFACTORS = {"mass": 5.0, "flow": 150.0, "spectral flow": 6000.0}
PRESETS = ("paper", "normalize_total")


def _numeric(X: Any) -> Any:
    if hasattr(X, "select_dtypes"):  # pandas DataFrame: numeric columns only, as the original
        return X.select_dtypes([np.number]).to_numpy()
    return X


def preprocess(
    X: Any,
    platform: str,
    *,
    preset: str = "paper",
    n_top_genes: int = 2000,
    return_cells: bool = False,
    compat: str | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Turn raw measurements into the matrix CosTaL clusters.

    Args:
        X: cells × features — a numpy array, a pandas DataFrame (numeric columns are used) or, for
            scRNA-seq, a scipy sparse matrix of raw counts.
        platform: ``"mass"`` (arcsinh(x/5)), ``"flow"`` (arcsinh(x/150)), ``"spectral flow"``
            (arcsinh(x/6000)) or ``"scrnaseq"``.
        preset: scRNA-seq only. ``"paper"`` is the original (no library-size normalization, so
            row norms mostly record sequencing depth); ``"normalize_total"`` scales every cell to
            the median library size before ``log1p`` (docs/04 §6.7).
        n_top_genes: scRNA-seq only; the paper uses 2000.
        return_cells: also return the indices of the cells kept (scRNA-seq drops cells with fewer
            than 200 detected genes; cytometry keeps all).
        compat: ``"paper"`` refuses any setting that departs from the original ``pp()``.

    Returns:
        A dense float32 matrix (and the kept cell indices if ``return_cells``). scRNA-seq needs
        ``scanpy`` and ``scikit-misc``, installed with CosTaL (except on Windows on Arm).
    """
    if compat not in (None, "paper"):
        raise ValueError(f"compat must be None or 'paper', got {compat!r}")
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {PRESETS}, got {preset!r}")
    if compat == "paper" and (preset != "paper" or n_top_genes != 2000):
        raise ValueError("compat='paper' pins preset='paper' and n_top_genes=2000 (docs/04 §6.7)")
    X = _numeric(X)

    if platform in COFACTORS:
        if preset != "paper":
            raise ValueError("preset applies to platform='scrnaseq' only")
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2-D matrix, got {X.ndim} dimensions")
        # arcsinh in f64 then rounded to f32, row chunk by row chunk: the output is the only full-size
        # array (values identical to the one-shot version)
        out = np.empty(X.shape, dtype=np.float32)
        for a, b in _row_chunks(*X.shape):
            out[a:b] = np.arcsinh(X[a:b].astype(np.float64) / COFACTORS[platform])
        return (out, np.arange(out.shape[0])) if return_cells else out
    if platform != "scrnaseq":
        raise ValueError(f"platform must be one of {(*COFACTORS, 'scrnaseq')}, got {platform!r}")

    try:
        import anndata
        import scanpy as sc
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("platform='scrnaseq' needs scanpy and scikit-misc (installed with CosTaL except "
                          "on Windows on Arm, where they have no wheels yet)") from exc

    n = X.shape[0]
    ad = anndata.AnnData(X=X.copy() if hasattr(X, "copy") else np.array(X))
    ad.obs_names = [str(i) for i in range(n)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scanpy's own notes on non-integer counts, implicit copies
        sc.pp.filter_cells(ad, min_genes=200)
        sc.pp.filter_genes(ad, min_cells=3)
        sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes, flavor="seurat_v3")
        if preset == "normalize_total":
            sc.pp.normalize_total(ad)
        sc.pp.log1p(ad)
    out = ad.X[:, ad.var["highly_variable"].to_numpy()]
    out = np.ascontiguousarray(out.toarray() if hasattr(out, "toarray") else out, dtype=np.float32)
    cells = ad.obs_names.astype(int).to_numpy()
    if len(cells) < n:
        warnings.warn(
            f"filter_cells(min_genes=200) dropped {n - len(cells)} of {n} cells; subset your labels "
            "with the indices from return_cells=True",
            CostalWarning,
            stacklevel=2,
        )
    return (out, cells) if return_cells else out
