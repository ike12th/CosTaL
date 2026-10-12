"""CosTaL+ — CosTaL 的高性能重写。

方法来源：Li Y, Nguyen J, Anastasiu DC, Arriaga EA. *CosTaL: an accurate and
scalable graph-based clustering algorithm for high-dimensional single-cell data
analysis.* Briefings in Bioinformatics, 2023. doi:10.1093/bib/bbad157

阶段 A：精确 kNN + 边权 + 对称化已就位，Leiden 仍用 leidenalg（docs/03 §5）::

    rows, cols, w = costal_plus.build_graph(X, k=10)
"""

from ._api import CostalWarning, KnnGraph, build_graph, knn, reweight, truncate
from ._costal_plus import (
    __version__,
    core_version,
    default_matmul,
    hello,
    l2_bytes_per_thread,
    row_l2_norms,
    tile_budget_bytes,
)

__all__ = [
    "CostalWarning",
    "KnnGraph",
    "__version__",
    "build_graph",
    "core_version",
    "default_matmul",
    "hello",
    "knn",
    "l2_bytes_per_thread",
    "reweight",
    "row_l2_norms",
    "tile_budget_bytes",
    "truncate",
]
