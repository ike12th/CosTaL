"""find_communities_multiscale(): the conductance carving rule (docs/05 §2.3) and the API around it.
Its accuracy is measured on real data in bench/multiscale.py; bench and API give identical partitions
(checked on GSE84133 and Levine_32)."""

import numpy as np
import pytest

import costal_plus as cp
from costal_plus._multiscale import _adjacency, _carve


def graph(edges):
    """CSR adjacency (indptr, neighbors, weights, degrees) of an undirected weighted edge list."""
    rows = np.array([min(a, b) for a, b, _ in edges])
    cols = np.array([max(a, b) for a, b, _ in edges])
    w = np.array([x for _, _, x in edges], dtype=np.float64)
    return _adjacency(int(cols.max() + 1), rows, cols, w)


def clique(nodes, weight=1.0):
    return [(a, b, weight) for i, a in enumerate(nodes) for b in nodes[i + 1:]]


def test_carves_a_cohesive_subcluster_but_not_a_fragment():
    big = list(range(0, 60))
    rare = list(range(60, 80))
    edges = clique(big) + clique(rare) + [(0, 60, 1.0)]  # the rare clique hangs on by one edge
    adj = graph(edges)
    n = 80
    coarse = np.zeros(n, dtype=np.int64)  # one cluster: the rare clique swallowed
    fine = np.zeros(n, dtype=np.int64)
    fine[rare] = 1  # the rare clique on its own
    fine[30:60] = 2  # and the big clique cut in half: a fragment, half its edges cross the cut
    ms = np.stack([np.stack([coarse, coarse]), np.stack([fine, fine])])
    labels, carved = _carve(ms, *adj, phi_max=0.04, min_size=10, resolutions=(0.1, 0.2))
    assert len(carved) == 1 and carved[0]["cells"] == 20
    assert carved[0]["phi"] < 0.01
    assert len(np.unique(labels[rare])) == 1 and labels[rare[0]] != labels[big[0]]
    assert len(np.unique(labels[big])) == 1  # the fragment was not carved
    # a threshold above the fragment's leak carves it too
    _, carved = _carve(ms, *adj, phi_max=0.9, min_size=10, resolutions=(0.1, 0.2))
    assert len(carved) == 2


def test_min_size_and_proper_part():
    # 100 + 8: the big side is 93% of the host, so only the small side can be a proper part
    adj = graph(clique(list(range(100))) + clique(list(range(100, 108))) + [(0, 100, 1.0)])
    coarse = np.zeros(108, dtype=np.int64)
    fine = np.r_[np.zeros(100, dtype=np.int64), np.ones(8, dtype=np.int64)]
    ms = np.stack([coarse[None], fine[None]])
    assert _carve(ms, *adj, 0.04, min_size=10, resolutions=(0.1, 0.2))[1] == []  # too small
    assert len(_carve(ms, *adj, 0.04, min_size=5, resolutions=(0.1, 0.2))[1]) == 1
    same = np.stack([coarse[None], coarse[None]])  # a candidate equal to its host is not a proper part
    assert _carve(same, *adj, 0.9, min_size=5, resolutions=(0.1, 0.2))[1] == []


def blobs(seed=0, n=400, d=8):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 4, size=(3, d))
    x = np.vstack([c + rng.normal(0, 0.4, (n, d)) for c in centers])
    return np.clip(x, 0, None).astype(np.float32)


def test_api_is_deterministic_and_reports_what_it_carved():
    pytest.importorskip("igraph")
    x = blobs()
    a, info = cp.find_communities_multiscale(x, return_info=True)
    b = cp.find_communities_multiscale(x)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (len(x),) and a.dtype == np.int64
    assert info["clusters"] == a.max() + 1 and info["phi_max"] == 0.04
    assert len(info["resolutions"]) == 11 and info["seeds"] == (0, 1, 2, 3)
    assert all(c["phi"] <= 0.04 and c["cells"] >= 20 for c in info["carved"])


def test_api_rejects_bad_arguments():
    x = blobs(n=50)
    with pytest.raises(ValueError, match="compat='paper'"):
        cp.find_communities_multiscale(x, compat="paper")
    with pytest.raises(ValueError, match="phi_max"):
        cp.find_communities_multiscale(x, phi_max=0)
    with pytest.raises(ValueError, match="ascending"):
        cp.find_communities_multiscale(x, resolutions=[1.0, 0.5])
    with pytest.raises(ValueError, match="leiden"):
        cp.find_communities_multiscale(x, leiden="louvain")
