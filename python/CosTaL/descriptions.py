"""CosTaL 0.0.2's ``CosTaL.descriptions`` module: the clustering-evaluation helpers."""

from ._legacy import (
    calculate_sparsity,
    get_f1_matrix,
    get_rare_f1,
    return_Ff1_score,
    return_Hf1_score,
    return_scores,
)

__all__ = [
    "calculate_sparsity",
    "get_f1_matrix",
    "get_rare_f1",
    "return_Ff1_score",
    "return_Hf1_score",
    "return_scores",
]
