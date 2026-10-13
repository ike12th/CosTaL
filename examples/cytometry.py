"""Cluster cytometry data with the published CosTaL.

    python examples/cytometry.py                      # synthetic data
    python examples/cytometry.py cells.fcs mass       # an FCS file (pip install fcsparser); platform mass/flow/"spectral flow"
"""

import sys

import numpy as np

import costal_plus


def synthetic(n=3000, markers=12, populations=6, seed=0):
    """Raw intensities: a few populations, each a different marker profile, plus noise."""
    rng = np.random.default_rng(seed)
    profiles = rng.choice([0.0, 20.0, 200.0], size=(populations, markers))
    labels = rng.integers(0, populations, n)
    return profiles[labels] * rng.lognormal(0, 0.3, (n, markers)) + rng.normal(0, 3, (n, markers)), labels


def main(argv):
    if len(argv) > 1:
        import fcsparser

        _, df = fcsparser.parse(argv[1], reformat_meta=True)
        drop = {"Time", "Event_length", "Cell_length", "label"}
        raw, truth = df[[c for c in df.columns if c not in drop]], None
        platform = argv[2] if len(argv) > 2 else "mass"
    else:
        raw, truth = synthetic()
        platform = "mass"

    X = costal_plus.preprocess(raw, platform)
    labels = costal_plus.find_communities(X, seed=0)
    sizes = np.bincount(labels)
    print(f"{len(X)} cells, {X.shape[1]} markers -> {len(sizes)} clusters; sizes {sorted(sizes, reverse=True)}")
    if truth is not None:
        from sklearn.metrics import adjusted_rand_score

        print(f"ARI against the true populations: {adjusted_rand_score(truth, labels):.3f}")
    return labels


if __name__ == "__main__":
    main(sys.argv)
