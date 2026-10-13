"""CosTaL+ — a high-performance rewrite of CosTaL.

Method: Li Y, Nguyen J, Anastasiu DC, Arriaga EA. *CosTaL: an accurate and
scalable graph-based clustering algorithm for high-dimensional single-cell data
analysis.* Briefings in Bioinformatics, 2023. doi:10.1093/bib/bbad157

End to end, with the published CosTaL's defaults (docs/04 §0.0)::

    labels = costal_plus.find_communities(X, k=10)

or step by step::

    rows, cols, w = costal_plus.build_graph(X, k=10)
"""

from ._api import CostalWarning, KnnGraph, build_graph, find_communities, knn, reweight, truncate
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
    "find_communities",
    "hello",
    "knn",
    "l2_bytes_per_thread",
    "reweight",
    "row_l2_norms",
    "tile_budget_bytes",
    "truncate",
]
