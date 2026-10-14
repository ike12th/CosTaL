"""leiden="fast": CosTaL's own Leiden (crates/costal-core/src/leiden.rs, docs/07 §5).

The Rust unit tests check the algorithm itself (Leiden's guarantees, optimality on small graphs,
determinism across thread counts). These check it through Python: against igraph on the same
objective, through the public API, and bit for bit against fixed partitions, so that the wheel
tests on macOS, Linux and Windows catch any platform difference.
"""

import hashlib
import random

import numpy as np
import pytest

import costal as cp
from costal import _costal as ext


def planted(n_blocks=12, size=40, p_in=0.15, n_cross=500, seed=0):
    """Upper-triangular edge list of a planted partition with weights in (0.5, 1)."""
    rng = np.random.default_rng(seed)
    pairs = {}
    for b in range(n_blocks):
        nodes = np.arange(b * size, (b + 1) * size)
        i, j = np.triu_indices(size, 1)
        keep = rng.random(len(i)) < p_in
        for a, c in zip(nodes[i[keep]], nodes[j[keep]]):
            pairs[(int(a), int(c))] = 0.5 + 0.5 * rng.random()
    n = n_blocks * size
    while n_cross:
        a, c = rng.integers(0, n, 2)
        if a // size != c // size:
            pairs[(int(min(a, c)), int(max(a, c)))] = 0.5 + 0.5 * rng.random()
            n_cross -= 1
    keys = sorted(pairs)
    rows = np.array([k[0] for k in keys], dtype=np.int32)
    cols = np.array([k[1] for k in keys], dtype=np.int32)
    w = np.array([pairs[k] for k in keys])
    return n, rows, cols, w


def digest(labels) -> str:
    return hashlib.sha256(np.asarray(labels, dtype="<i8").tobytes()).hexdigest()[:16]


def test_recovers_planted_blocks_and_is_thread_independent():
    n, rows, cols, w = planted(p_in=0.3)
    m, k, iterations, _ = ext._leiden(n, rows, cols, w, resolution=1.0, seed=0)
    assert k == 12 and iterations == 2
    assert all(len(np.unique(m[b * 40:(b + 1) * 40])) == 1 for b in range(12))
    for threads in (1, 2, 4):
        assert np.array_equal(ext._leiden(n, rows, cols, w, resolution=1.0, seed=5, n_threads=threads)[0],
                              ext._leiden(n, rows, cols, w, resolution=1.0, seed=5)[0])


def test_modularity_agrees_with_igraph():
    ig = pytest.importorskip("igraph")
    n, rows, cols, w = planted(p_in=0.08, seed=1)
    graph = ig.Graph(n=n, edges=list(zip(rows.tolist(), cols.tolist())))
    graph.es["weight"] = w.tolist()
    rng = np.random.default_rng(2)
    for res in (0.5, 0.8, 2.0):
        m = rng.integers(0, 30, n).astype(np.int32)
        assert ext._modularity(n, rows, cols, w, m, res) == pytest.approx(
            graph.modularity(m.tolist(), weights="weight", resolution=res), abs=1e-12)


def test_optimizes_as_well_as_igraph():
    ig = pytest.importorskip("igraph")
    n, rows, cols, w = planted(n_blocks=30, size=30, p_in=0.06, n_cross=1500, seed=3)
    graph = ig.Graph(n=n, edges=list(zip(rows.tolist(), cols.tolist())))
    graph.es["weight"] = w.tolist()
    native, igraph = [], []
    for s in range(10):
        m = ext._leiden(n, rows, cols, w, resolution=1.0, seed=s)[0]
        native.append(ext._modularity(n, rows, cols, w, m, 1.0))
        random.seed(s)
        m = graph.community_leiden(objective_function="modularity", weights="weight", resolution=1.0,
                                   n_iterations=2).membership
        igraph.append(ext._modularity(n, rows, cols, w, np.asarray(m, dtype=np.int32), 1.0))
    assert np.mean(native) >= np.mean(igraph) - 0.5 * np.std(igraph) - 1e-4


def test_many_matches_single_runs():
    n, rows, cols, w = planted(seed=4)
    many = ext._leiden_many(n, rows, cols, w, resolutions=[0.5, 1.0, 2.0], seeds=[0, 7])
    assert many.shape == (6, n)
    for i, (res, seed) in enumerate([(r, s) for r in (0.5, 1.0, 2.0) for s in (0, 7)]):
        assert np.array_equal(many[i], ext._leiden(n, rows, cols, w, resolution=res, seed=seed)[0])


def test_until_stable_and_bad_arguments():
    n, rows, cols, w = planted(seed=6)
    m, _, iterations, converged = ext._leiden(n, rows, cols, w, resolution=1.0, n_iterations=-1, seed=0)
    assert converged and iterations >= 1
    with pytest.raises(ValueError, match="n_iterations"):
        ext._leiden(n, rows, cols, w, resolution=1.0, n_iterations=0)
    with pytest.raises(ValueError, match="resolution"):
        ext._leiden(n, rows, cols, w, resolution=-1.0)
    with pytest.raises(ValueError, match="out of range"):
        ext._leiden(n - 1, rows, cols, w, resolution=1.0)


def test_public_api():
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 4, size=(5, 12))
    x = np.abs(np.vstack([c + rng.normal(size=(300, 12)) for c in centers]))
    truth = np.repeat(np.arange(5), 300)
    from sklearn.metrics import adjusted_rand_score

    labels = cp.find_communities(x, leiden="fast", seed=0)
    assert labels.dtype == np.int64 and adjusted_rand_score(truth, labels) > 0.95
    assert np.array_equal(labels, cp.find_communities(x, leiden="fast", seed=0, n_threads=1))
    assert np.array_equal(labels, cp.find_communities(x, leiden="fast", seed=2**64))  # seed mod 2^64
    ms = cp.find_communities_multiscale(x, leiden="fast")
    assert adjusted_rand_score(truth, ms) > 0.95
    with pytest.raises(ValueError, match="compat"):
        cp.find_communities(x, leiden="fast", compat="paper")


# Fixed partitions: the same bits on every platform (the wheel tests run this on macOS, Linux and
# Windows). If the algorithm changes on purpose, update the digests.
GOLDEN_GRAPH = "36b0210aa49dc294"
GOLDEN_PIPELINE = "ab4b3e17a8b1138d"


def test_same_partition_on_every_platform():
    n, rows, cols, w = planted(n_blocks=20, size=50, p_in=0.1, n_cross=2000, seed=11)
    m = ext._leiden(n, rows, cols, w, resolution=1.3, seed=12345)[0]
    assert digest(m) == GOLDEN_GRAPH
    rng = np.random.default_rng(7)
    x = np.abs(rng.normal(size=(3000, 16)) + np.repeat(rng.normal(0, 3, size=(6, 16)), 500, axis=0))
    assert digest(cp.find_communities(x, leiden="fast", seed=3)) == GOLDEN_PIPELINE
