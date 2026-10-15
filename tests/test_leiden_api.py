"""costal.leiden: Leiden on an external edge list (the public API around the Rust implementation)."""


import numpy as np
import pytest

import costal
from test_leiden_native import digest, planted


def both_directions(rows, cols, w):
    return np.concatenate([rows, cols]), np.concatenate([cols, rows]), np.concatenate([w, w])


def test_each_edge_once_or_symmetric_give_the_same_result():
    n, rows, cols, w = planted(seed=1)
    base = costal.leiden(n, rows, cols, w, seed=3, return_quality=True)
    flip = np.arange(len(rows)) % 2 == 1  # each edge once, in mixed orientation
    r2, c2 = np.where(flip, cols, rows), np.where(flip, rows, cols)
    for args in [(r2, c2, w), both_directions(rows, cols, w)]:
        labels, q = costal.leiden(n, *args, seed=3, return_quality=True)
        assert np.array_equal(labels, base[0]) and q == base[1]


def test_a_symmetric_scipy_matrix():
    sp = pytest.importorskip("scipy.sparse")
    n, rows, cols, w = planted(seed=2)
    a = sp.coo_matrix((w, (rows, cols)), shape=(n, n))
    sym = (a + a.T).tocoo()
    assert np.array_equal(costal.leiden(n, sym.row, sym.col, sym.data, seed=0),
                          costal.leiden(n, rows, cols, w, seed=0))


@pytest.mark.parametrize("rows, cols, w, match", [
    ([0, 0], [1, 1], [1.0, 1.0], "same direction"),        # a duplicate, even with equal weights
    ([0, 1], [1, 0], [1.0, 2.0], "different weights"),
    ([0, 1, 1], [1, 0, 2], [1.0, 1.0, 1.0], "others once"),
    ([1], [1], [1.0], "self-loop"),
    ([0], [6], [1.0], "outside"),
    ([-1], [1], [1.0], "outside"),
    ([0], [1], [np.nan], "weight"),
    ([0], [1], [np.inf], "weight"),
    ([0], [1], [-0.5], "weight"),
    ([0, 1], [1], [1.0], "length"),
    ([0.0], [1.0], [1.0], "integer"),
])
def test_bad_edges_are_value_errors(rows, cols, w, match):
    with pytest.raises(ValueError, match=match):
        costal.leiden(6, np.asarray(rows), np.asarray(cols), np.asarray(w, dtype=float))


def test_bad_arguments_are_value_errors():
    rows, cols, w = np.array([0, 1]), np.array([1, 2]), np.ones(2)
    for kwargs, match in [(dict(objective="rb"), "objective"), (dict(n_iterations=0), "n_iterations"),
                          (dict(resolution=-1.0), "resolution"), (dict(beta=0.0), "beta"),
                          (dict(initial_membership=[0, 1]), "one label per node"),
                          (dict(initial_membership=[0.0, 1.0, 1.0]), "integer")]:
        with pytest.raises(ValueError, match=match):
            costal.leiden(3, rows, cols, w, **kwargs)
    with pytest.raises(ValueError, match="n_nodes"):
        costal.leiden(-1, rows, cols, w)


def test_labels_quality_and_small_cases():
    n, rows, cols, w = planted(seed=4)
    labels = costal.leiden(n, rows, cols, w, seed=0)
    assert labels.dtype == np.int64 and labels.shape == (n,)
    _, first = np.unique(labels, return_index=True)
    assert np.array_equal(labels[np.sort(first)], np.arange(labels.max() + 1))  # 0 … k−1, first appearance
    # Zero-weight edges are ignored; isolated nodes stay alone
    r2, c2, w2 = np.append(rows, [0, 5]), np.append(cols, [n + 1, n + 2]), np.append(w, [0.0, 0.0])
    out = costal.leiden(n + 3, r2, c2, w2, seed=0)
    assert np.array_equal(out[:n], labels) and len(set(out[n:])) == 3 and not set(out[n:]) & set(out[:n])
    assert costal.leiden(3, [], [], np.array([]), seed=0).tolist() == [0, 1, 2]
    assert costal.leiden(0, [], [], np.array([])).shape == (0,)


def test_modularity_quality_is_igraphs_at_any_resolution():
    ig = pytest.importorskip("igraph")
    n, rows, cols, w = planted(seed=5)
    graph = ig.Graph(n=n, edges=list(zip(rows.tolist(), cols.tolist())))
    graph.es["weight"] = w.tolist()
    for res in (1.0, 0.5, 2.0):
        labels, q = costal.leiden(n, rows, cols, w, resolution=res, seed=1, return_quality=True)
        assert q == pytest.approx(graph.modularity(labels.tolist(), weights="weight", resolution=res), abs=1e-12)


def test_cpm_quality_is_leidenalgs_and_optimizes_as_well():
    ig = pytest.importorskip("igraph")
    la = pytest.importorskip("leidenalg")
    n, rows, cols, w = planted(n_blocks=20, size=30, p_in=0.15, n_cross=800, seed=6)
    graph = ig.Graph(n=n, edges=list(zip(rows.tolist(), cols.tolist())))
    graph.es["weight"] = w.tolist()
    for res in (0.02, 0.1):
        ours, theirs = [], []
        for s in range(5):
            labels, q = costal.leiden(n, rows, cols, w, objective="cpm", resolution=res, seed=s,
                                      return_quality=True)
            part = la.CPMVertexPartition(graph, initial_membership=labels.tolist(), weights="weight",
                                         resolution_parameter=res)
            assert q == pytest.approx(part.quality(), rel=1e-12, abs=1e-9)
            ours.append(q)
            theirs.append(la.find_partition(graph, la.CPMVertexPartition, weights="weight",
                                            resolution_parameter=res, n_iterations=2, seed=s).quality())
        assert np.mean(ours) >= np.mean(theirs) - 0.01 * abs(np.mean(theirs))


def test_initial_membership_and_until_stable():
    n, rows, cols, w = planted(seed=7)
    stable, q = costal.leiden(n, rows, cols, w, n_iterations=-1, seed=0, return_quality=True)
    # Arbitrary label values are accepted; from a stable partition one more iteration changes nothing
    relabeled = stable * 7 + 100
    again, q2 = costal.leiden(n, rows, cols, w, n_iterations=1, seed=9, initial_membership=relabeled,
                              return_quality=True)
    assert np.array_equal(again, stable) and q2 == q
    # Starting from one cluster still gets to a good partition
    start = costal.leiden(n, rows, cols, w, seed=0, initial_membership=np.zeros(n, dtype=int))
    assert len(np.unique(start)) > 1


def test_deterministic_for_a_seed_at_any_thread_count():
    n, rows, cols, w = planted(n_blocks=150, size=40, p_in=0.2, n_cross=3000, seed=8)
    for objective, res in (("modularity", 1.0), ("cpm", 0.05)):
        a = costal.leiden(n, rows, cols, w, objective=objective, resolution=res, seed=11, n_threads=1)
        b = costal.leiden(n, rows, cols, w, objective=objective, resolution=res, seed=11)
        c = costal.leiden(n, rows, cols, w, objective=objective, resolution=res, seed=11 + 2**64)
        assert np.array_equal(a, b) and np.array_equal(a, c)


# Fixed partitions: the same bits on every platform (checked by the wheel tests on macOS, Linux and
# Windows). If the algorithm changes on purpose, update the digests.
GOLDEN_MODULARITY = "36b0210aa49dc294"  # the same graph and seed as test_leiden_native.GOLDEN_GRAPH
GOLDEN_CPM = "12fe6df4ac48f2e6"


def test_same_partition_on_every_platform():
    n, rows, cols, w = planted(n_blocks=20, size=50, p_in=0.1, n_cross=2000, seed=11)
    r, c, ww = both_directions(rows, cols, w)
    assert digest(costal.leiden(n, r, c, ww, resolution=1.3, seed=12345)) == GOLDEN_MODULARITY
    assert digest(costal.leiden(n, r, c, ww, objective="cpm", resolution=0.04, seed=7)) == GOLDEN_CPM
