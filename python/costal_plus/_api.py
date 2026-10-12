"""公开 API：``knn()`` / ``reweight()`` / ``truncate()`` / ``build_graph()``（docs/04 §6）。

不传任何可选参数 = 已发表的 CosTaL：按 cosine 选邻居（精确 kNN，k 不含自身），
用未归一化的行范数算 Tanimoto 边权，再做模糊并集对称化（docs/04 §0.0）。

kNN 是唯一贵的一步；边权与截断都在同一份 kNN 上做，几乎不花时间。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import _costal_plus as _ext

__all__ = ["CostalWarning", "KnnGraph", "build_graph", "knn", "reweight", "truncate"]


class CostalWarning(UserWarning):
    """输入诊断给出的提醒：结果仍然正确，但某个参数的含义可能和你想的不一样。"""


@dataclass(frozen=True)
class KnnGraph:
    """一份精确 kNN。

    Attributes:
        idx: ``(n, k)`` int32，原始行索引；每行按（cosine 降序, 索引升序）排好。
        cos: ``(n, k)`` float64，f64 精排后的 cosine。
        norms: ``(n,)`` float64，原始（未归一化）行范数，Tanimoto 边权用它。
        meta: 参数、统计与输入诊断（docs/04 §6.6）。
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


def _check_choices(select: str, normalize: str, backend: str, compat: str | None) -> None:
    if compat not in (None, "paper"):
        raise ValueError(f"compat 只能是 None 或 'paper'，收到 {compat!r}")
    if select == "tanimoto":
        raise NotImplementedError("select='tanimoto' 在阶段 A+ 加入（docs/04 §7）")
    if select != "cosine":
        raise ValueError(f"select 只能是 'cosine' 或 'tanimoto'，收到 {select!r}")
    if normalize == "none":
        raise NotImplementedError(
            "normalize='none' 只对 select='tanimoto' 有意义，阶段 A+ 加入（docs/04 §0）"
        )
    if normalize != "l2":
        raise ValueError(f"normalize 只能是 'l2' 或 'none'，收到 {normalize!r}")
    if backend in ("approx", "hnsw"):
        raise NotImplementedError("近似后端是阶段 E，且永不进默认路径（docs/03 §6）")
    if backend == "sparse":
        raise NotImplementedError("稀疏后端是阶段 S（docs/06 §5）")
    if backend != "exact":
        raise ValueError(f"backend 只能是 'exact'，收到 {backend!r}")


def _as_matrix(X: Any) -> np.ndarray:
    if hasattr(X, "tocsr") or hasattr(X, "toarray"):
        raise TypeError(
            "稀疏输入要走稀疏后端（阶段 S，docs/06 §5），尚未实现；可以先 .toarray() 或降维"
        )
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X 必须是二维矩阵，收到 {X.ndim} 维")
    if not (np.issubdtype(X.dtype, np.floating) or np.issubdtype(X.dtype, np.integer)):
        raise TypeError(f"X 必须是实数矩阵，收到 dtype={X.dtype}")
    return np.ascontiguousarray(X, dtype=np.float32)


def _diagnose(X: np.ndarray) -> dict[str, Any]:
    """输入诊断：只记录、不改数据（docs/04 §6.6）。"""
    norms = np.sqrt(np.einsum("ij,ij->i", X, X, dtype=np.float64))
    mean_norm = norms.mean()
    col_mean = X.mean(axis=0, dtype=np.float64)
    col_std = X.std(axis=0, dtype=np.float64)
    live = col_std > 0
    return {
        # ≈ 0 ⇒ 输入已逐行单位化，Tanimoto ≡ c/(2−c)
        "row_norm_cv": float(norms.std() / mean_norm) if mean_norm > 0 else 0.0,
        # 所有非常数列的均值都远小于标准差 ⇒ 像 PCA / z-score / Pearson residuals
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
    normalize: str = "l2",
    backend: str = "exact",
    compat: str | None = None,
    n_threads: int | None = None,
) -> KnnGraph:
    """精确 kNN（docs/04 §6.1）。

    按 cosine 选每行的 k 个邻居（不含自身）。全部 N² 个相似度都算过，
    最终名次由 f64 固定顺序重算决定：换平台、换线程数，结果逐位相同（docs/06 §1）。

    Args:
        X: ``(n, d)`` 稠密实数矩阵，内部转成 float32。
        k: 每行的邻居数，``1 ≤ k ≤ n-1``。
        compat: ``"paper"`` 显式锁定已发表 CosTaL 的语义（docs/04 §6.5）。
            目前默认值就是论文语义，这个参数只是把意图写进元数据。
        n_threads: 线程数；``None`` 用全部核。
    """
    _check_choices(select, normalize, backend, compat)
    X = _as_matrix(X)
    (idx, cos, norms), stats = _ext._knn_exact(X, int(k), n_threads=n_threads)
    meta = {
        "k": int(k),
        "select": select,
        "normalize": normalize,
        "backend": backend,
        "compat": compat,
        "version": _ext.__version__,
        "stats": dict(stats),
        "diagnostics": _diagnose(X),
    }
    return KnnGraph(idx=idx, cos=cos, norms=norms, meta=meta)


def _warn_about_magnitude(meta: dict[str, Any]) -> None:
    diag = meta.get("diagnostics")
    if not diag:
        return
    if diag["row_norm_cv"] < 1e-4:
        warnings.warn(
            "输入的行范数几乎恒定（已逐行单位化？）：Tanimoto 边权等价于 cosine 的单调变换 "
            "c/(2−c)，不提供额外信息（docs/04 §0.1）",
            CostalWarning,
            stacklevel=3,
        )
    if diag["columns_centered"]:
        warnings.warn(
            "各列已居中（PCA / z-score / Pearson residuals？）：行范数表示“离数据中心多远”"
            "而不是丰度，Tanimoto 边权的含义随之改变，且可能为负（docs/04 §6.6）",
            CostalWarning,
            stacklevel=3,
        )


def reweight(knn: KnnGraph, weight: str = "tanimoto") -> np.ndarray:
    """在同一份 kNN 上算边权，返回 ``(n, k)`` float64（docs/04 §6.1）。

    - ``"tanimoto"``：``T = c·‖a‖‖b‖ / (‖a‖² + ‖b‖² − c·‖a‖‖b‖)``，用原始行范数——已发表的 CosTaL；
    - ``"cosine"``：精排后的 cosine；
    - ``"jaccard"``：共享邻居数 s 的 ``s / (2k − s)``，要比较邻居的邻居，O(N·k²)。

    重算边权只改边上的数，改不了图里有哪些边（docs/04 §6.3）。
    """
    if weight not in _WEIGHTS:
        raise ValueError(f"weight 只能是 {_WEIGHTS}，收到 {weight!r}")
    if weight == "tanimoto":
        _warn_about_magnitude(knn.meta)
    w = _ext._reweight(knn.idx, knn.cos, knn.norms, weight)
    n_negative = int(np.count_nonzero(w < 0))
    if n_negative:
        warnings.warn(
            f"{n_negative} 条边的权重为负（带符号输入上 Tanimoto 的下界是 −1/3）。"
            "论文模式原样保留；怎么处理待实验决定（docs/04 §6.6）",
            CostalWarning,
            stacklevel=2,
        )
    return w


def truncate(knn: KnnGraph, k: int) -> KnnGraph:
    """取每行前 k 个邻居。每行已按（cosine 降序, 索引升序）全序排好，
    所以结果**就是** k 更小时的精确 kNN，不是近似（docs/04 §6.1）。"""
    if not 1 <= k <= knn.k:
        raise ValueError(f"k 必须在 1..{knn.k} 之间，收到 {k}")
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
    normalize: str = "l2",
    backend: str = "exact",
    symmetrize: str = "fuzzy_union",
    compat: str | None = None,
    n_threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一步到位：kNN → 边权 → 对称化，返回上三角边表 ``(rows, cols, weights)``。

    对称化是原版的模糊并集 ``S + Sᵀ − S∘Sᵀ``（**不是** ``(S + Sᵀ)/2``），去自环、
    去掉恰为 0 的边；边按（行, 列）升序，与原版 scipy 管线逐位一致（docs/04 §0.0）。

    接 Leiden 的方式与原版相同::

        rows, cols, w = costal_plus.build_graph(X)
        G = igraph.Graph(n=len(X), edges=list(zip(rows.tolist(), cols.tolist())),
                         edge_attrs={"weight": w.tolist()})
    """
    if symmetrize != "fuzzy_union":
        raise ValueError(f"symmetrize 目前只有 'fuzzy_union'，收到 {symmetrize!r}")
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
