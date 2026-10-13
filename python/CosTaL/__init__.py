"""CosTaL — fast, exact graph-based clustering for single-cell data.

Method: Li Y, Nguyen J, Anastasiu DC, Arriaga EA. *CosTaL: an accurate and scalable graph-based
clustering algorithm for high-dimensional single-cell data analysis.* Briefings in Bioinformatics
24(3), bbad157 (2023). doi:10.1093/bib/bbad157

End to end, as published (docs/04 §0.0)::

    import CosTaL

    X = CosTaL.preprocess(raw, "mass")      # or "flow", "spectral flow", "scrnaseq"
    labels = CosTaL.find_communities(X)

or step by step::

    rows, cols, w = CosTaL.build_graph(X, k=10)

CosTaL 0.0.2 code keeps working: ``CosTaL.find_communities(X, pp_method="mass", nbr_num=10)``,
``CosTaL.clustering``, ``CosTaL.pp``, ``CosTaL.return_scores`` and ``from CosTaL.clustering import …``
behave as they did (docs/user_guide.md §11).
"""

from ._api import CostalWarning, KnnGraph, build_graph, find_communities, knn, reweight, truncate
from ._costal import (
    __version__,
    core_version,
    default_matmul,
    hello,
    l2_bytes_per_thread,
    row_l2_norms,
    tile_budget_bytes,
)
from ._multiscale import find_communities_multiscale
from ._preprocess import preprocess

# The 0.0.2 interface. As in 0.0.2, importing the `clustering` submodule and then its `clustering`
# function leaves `CosTaL.clustering` bound to the function, while `from CosTaL.clustering import …`
# still finds the submodule.
from .clustering import clustering, normalize_arcsinh, pp  # noqa: E402  (after the 1.0 API on purpose)
from .descriptions import (  # noqa: E402
    calculate_sparsity,
    get_f1_matrix,
    get_rare_f1,
    return_Ff1_score,
    return_Hf1_score,
    return_scores,
)

__all__ = [
    "CostalWarning",
    "KnnGraph",
    "__version__",
    "build_graph",
    "calculate_sparsity",
    "clustering",
    "core_version",
    "default_matmul",
    "find_communities",
    "find_communities_multiscale",
    "get_f1_matrix",
    "get_rare_f1",
    "hello",
    "knn",
    "l2_bytes_per_thread",
    "normalize_arcsinh",
    "pp",
    "preprocess",
    "return_Ff1_score",
    "return_Hf1_score",
    "return_scores",
    "reweight",
    "row_l2_norms",
    "tile_budget_bytes",
    "truncate",
]
