"""Cluster scRNA-seq counts.

    python examples/scrnaseq.py                 # synthetic counts
    python examples/scrnaseq.py counts.h5ad     # raw UMI counts in .X (pip install anndata)
"""

import sys

import numpy as np

import costal


def synthetic(n=600, genes=3000, types=4, seed=0):
    """Poisson counts: each cell type over-expresses its own block of genes."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.4, 1.0, genes)
    labels = rng.integers(0, types, n)
    rate = np.tile(base, (n, 1))
    for t in range(types):
        rate[labels == t, t * 150:(t + 1) * 150] *= 8
    return rng.poisson(rate * rng.uniform(0.5, 2.0, (n, 1))).astype(np.float32), labels


def main(argv):
    if len(argv) > 1:
        import anndata

        counts, truth = anndata.read_h5ad(argv[1]).X, None
    else:
        counts, truth = synthetic()

    # cells with < 200 detected genes are dropped: keep their indices to align labels
    X, cells = costal.preprocess(counts, "scrnaseq", return_cells=True)
    labels = costal.find_communities(X, seed=0)
    print(f"{len(cells)} of {counts.shape[0]} cells kept, {X.shape[1]} genes -> {labels.max() + 1} clusters")
    if truth is not None:
        from sklearn.metrics import adjusted_rand_score

        print(f"ARI against the true types: {adjusted_rand_score(truth[cells], labels):.3f}")
    return labels


if __name__ == "__main__":
    main(sys.argv)
