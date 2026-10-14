"""Rare populations: one resolution vs find_communities_multiscale, and how to read its report.

    python examples/rare_populations.py                    # synthetic data
    python examples/rare_populations.py cells.fcs flow     # an FCS file

Synthetic populations this distinct are easy for any resolution, so here both runs find the rare one;
the resolution limit shows on real data, where single resolutions lose rare populations that the
multiscale mode keeps (Nilsson_rare HSC F1 0.17 at the paper's 0.8 vs 0.53; docs/05 §2.3).
"""

import sys

import numpy as np

import costal


def synthetic(n=20000, markers=10, rare=60, seed=1):
    """Five large populations and a small one that sits next to the first."""
    rng = np.random.default_rng(seed)
    profiles = rng.choice([0.0, 30.0, 300.0], size=(5, markers))
    labels = rng.integers(0, 5, n)
    x = profiles[labels] * rng.lognormal(0, 0.25, (n, markers))
    rare_profile = profiles[0].copy()
    rare_profile[np.argmin(rare_profile)] = 300.0  # one extra marker switched on
    x[:rare] = rare_profile * rng.lognormal(0, 0.25, (rare, markers))
    labels[:rare] = 5
    return x + rng.normal(0, 3, x.shape), labels


def rare_f1(truth, labels, population=5):
    cells = truth == population
    best = 0.0
    for c in np.unique(labels[cells]):
        tp = np.sum(labels[cells] == c)
        precision, recall = tp / np.sum(labels == c), tp / cells.sum()
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def main(argv, n=20000):
    if len(argv) > 1:
        _, df = costal.read_fcs(argv[1])
        raw, truth = df[[c for c in df.columns if c not in {"Time", "Event_length", "label"}]], None
        platform = argv[2] if len(argv) > 2 else "mass"
    else:
        raw, truth = synthetic(n=n)
        platform = "mass"
    X = costal.preprocess(raw, platform)
    single = costal.find_communities(X, leiden="fast", seed=0)
    multi, info = costal.find_communities_multiscale(X, return_info=True)
    print(f"one resolution (0.8): {single.max() + 1} clusters")
    print(f"multiscale:           {info['clusters']} clusters, {len(info['carved'])} split off from coarser ones")
    for c in info["carved"][:10]:
        print(f"  {c['cells']:6d} cells at resolution {c['resolution']}: {100 * c['phi']:.1f}% of their edge weight "
              f"leaks into the parent cluster")
    if truth is not None:
        print(f"rare-population F1: one resolution {rare_f1(truth, single):.2f}, multiscale {rare_f1(truth, multi):.2f}")
    return single, multi


if __name__ == "__main__":
    main(sys.argv)
