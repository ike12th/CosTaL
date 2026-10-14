"""The pipeline in steps: one kNN, many edge weights, k and resolutions.

    python examples/step_by_step.py
"""

import sys

import numpy as np

import costal
from cytometry import synthetic


def main(argv):
    raw, truth = synthetic(n=4000)
    X = costal.preprocess(raw, "mass")

    g = costal.knn(X, k=30)                    # the only expensive step
    print("kernel stats:", {k: g.meta["stats"][k] for k in ("matmul", "fallback_rows")})
    print("input diagnostics:", g.meta["diagnostics"])
    g10 = costal.truncate(g, 10)                # the exact 10-NN, for free
    weights = {w: costal.reweight(g10, w) for w in ("tanimoto", "cosine", "jaccard")}
    print({w: round(float(v.mean()), 3) for w, v in weights.items()})

    import igraph
    import leidenalg as la
    from sklearn.metrics import adjusted_rand_score

    rows, cols, w = costal.build_graph(X, k=10)
    G = igraph.Graph(n=len(X), edges=np.column_stack([rows, cols]), edge_attrs={"weight": w})
    for resolution in (0.2, 0.4, 0.8, 1.6):
        part = la.find_partition(G, la.RBConfigurationVertexPartition, weights="weight", n_iterations=10,
                                 resolution_parameter=resolution, seed=0)
        labels = np.asarray(part.membership)
        print(f"resolution {resolution}: {labels.max() + 1} clusters, ARI {adjusted_rand_score(truth, labels):.3f}")


if __name__ == "__main__":
    main(sys.argv)
