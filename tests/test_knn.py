"""阶段 A 的正确性网（docs/04 §4）。

- 与 Rust 里不分块、不剪枝、全程 f64 的暴力解**逐位相同**；
- 与 numpy float64 的独立实现在数值上一致（防止两边共用同一个 bug）；
- 边权与对称化与原版 CosTaL 的代码逐位一致；
- 分块尺寸、matmul 后端、布局、线程数都不改变输出（docs/06 §1）。
"""

import itertools
import warnings

import numpy as np
import pytest

import costal_plus as cp
from costal_plus import _costal_plus as ext


def clustered(n, d, seed, n_centers=8, noise=0.5):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 4, size=(n_centers, d))
    x = centers[rng.integers(0, n_centers, size=n)] + rng.normal(0, noise, size=(n, d))
    return np.ascontiguousarray(x, dtype=np.float32)


def assert_same_knn(a, b):
    ia, ca, na = a
    ib, cb, nb = b
    np.testing.assert_array_equal(ia, ib)
    np.testing.assert_array_equal(ca, cb)
    np.testing.assert_array_equal(na, nb)


# ---------------------------------------------------------------- kNN 本身


@pytest.mark.parametrize("seed", [0, 1])
def test_exact_matches_bruteforce_bitwise(seed):
    x = clustered(3000, 32, seed)
    got, stats = ext._knn_exact(x, 10)
    assert_same_knn(got, ext._knn_bruteforce(x, 10))
    assert stats["fallback_rows"] == 0


def test_matches_independent_numpy_oracle():
    x = clustered(1500, 20, 2)
    idx, cos, _ = ext._knn_exact(x, 12)[0]
    x64 = x.astype(np.float64)
    xn = x64 / np.linalg.norm(x64, axis=1, keepdims=True)
    sim = xn @ xn.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1, kind="stable")[:, :13]
    want_cos = np.take_along_axis(sim, order, axis=1)
    # 前 k 个 cosine 的值逐行一致（与并列怎么排无关）
    np.testing.assert_allclose(cos, want_cos[:, :12], rtol=0, atol=1e-12)
    # 第 k 名与第 k+1 名拉得开的行，邻居集合也必须一致
    clear = (want_cos[:, 11] - want_cos[:, 12]) > 1e-9
    assert clear.mean() > 0.99
    for i in np.flatnonzero(clear):
        assert set(idx[i]) == set(order[i, :12])


def test_output_independent_of_blocks_backend_layout_and_threads():
    x = clustered(1200, 17, 3)
    want = ext._knn_bruteforce(x, 7)
    backends = ["gemm", cp.default_matmul()]
    for backend, (rb, cb), sort, threads in itertools.product(
        backends, [(1, 7), (16, 128), (200, 5000)], [False, True], [1, 3]
    ):
        got, _ = ext._knn_exact(
            x, 7, row_block=rb, col_block=cb, matmul=backend,
            sort_by_norm=sort, n_threads=threads,
        )
        assert_same_knn(got, want)


def test_ties_break_by_original_index():
    rng = np.random.default_rng(4)
    protos = rng.normal(size=(10, 5)).astype(np.float32)
    x = np.ascontiguousarray(np.tile(protos, (30, 1)))  # 行 i 是原型 i % 10
    idx, cos, _ = ext._knn_exact(x, 10)[0]
    assert_same_knn((idx, cos, _), ext._knn_bruteforce(x, 10))
    np.testing.assert_array_equal(idx[0], np.arange(10, 101, 10))


def test_tie_heavy_integer_data_stays_exact():
    rng = np.random.default_rng(5)
    x = np.floor(rng.uniform(0, 3, size=(500, 6))).astype(np.float32)
    x[[0, 99, 499]] = 0.0
    got, stats = ext._knn_exact(x, 12)
    assert_same_knn(got, ext._knn_bruteforce(x, 12))
    # 大量精确并列，兜底路径必然被触发
    assert stats["fallback_rows"] > 0


def test_public_knn_and_truncate():
    x = clustered(800, 10, 6)
    g = cp.knn(x, 15)
    assert g.idx.dtype == np.int32 and g.cos.dtype == np.float64
    assert (g.idx != np.arange(800)[:, None]).all(), "k 不含自身"
    t = cp.truncate(g, 6)
    g6 = cp.knn(x, 6)
    np.testing.assert_array_equal(t.idx, g6.idx)
    np.testing.assert_array_equal(t.cos, g6.cos)
    np.testing.assert_array_equal(
        cp.reweight(t, "jaccard"), cp.reweight(g6, "jaccard")
    )


def test_input_validation():
    x = clustered(50, 4, 7)
    with pytest.raises(ValueError):
        cp.knn(x, 50)
    with pytest.raises(ValueError):
        cp.knn(x, 0)
    bad = x.copy()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        cp.knn(bad, 5)
    with pytest.raises(NotImplementedError):
        cp.knn(x, 5, select="tanimoto")
    with pytest.raises(ValueError):
        cp.knn(x, 5, compat="nope")
    sp = pytest.importorskip("scipy.sparse")
    with pytest.raises(TypeError):
        cp.knn(sp.csr_matrix(x), 5)
    # float64、整数、非连续输入都会被转成 C 连续 float32
    np.testing.assert_array_equal(cp.knn(x.astype(np.float64), 5).idx, cp.knn(x, 5).idx)
    xf = np.asfortranarray(x)
    np.testing.assert_array_equal(cp.knn(xf, 5).idx, cp.knn(x, 5).idx)


# ---------------------------------------------------------------- 边权：与原版逐位一致


def original_tani_from_cos(a, b, cos_val):
    """原版 clustering.py 的 tani_from_cos（去掉 numba 装饰器，运算顺序原样）。"""
    d = a * b * cos_val
    if d == 0:
        return 0
    else:
        return d / (a * a + b * b - d)


def original_calc_jaccard(i, idx):
    """原版 clustering.py 的 calc_jaccard。"""
    coefficients = np.fromiter(
        (len(set(idx[i]).intersection(set(idx[j]))) for j in idx[i]), dtype=float
    )
    coefficients /= 2 * idx.shape[1] - coefficients
    return coefficients


def test_tanimoto_matches_original_bitwise():
    g = cp.knn(clustered(600, 12, 8), 10)
    w = cp.reweight(g, "tanimoto")
    want = np.array(
        [
            [original_tani_from_cos(g.norms[i], g.norms[j], c) for j, c in zip(g.idx[i], g.cos[i])]
            for i in range(g.n)
        ]
    )
    np.testing.assert_array_equal(w, want)
    np.testing.assert_array_equal(cp.reweight(g, "cosine"), g.cos)


def test_jaccard_matches_original_bitwise():
    g = cp.knn(clustered(400, 8, 9), 10)
    want = np.array([original_calc_jaccard(i, g.idx) for i in range(g.n)])
    np.testing.assert_array_equal(cp.reweight(g, "jaccard"), want)


# ---------------------------------------------------------------- 对称化：与原版逐位一致


def original_build_graph_edges(rows, columns, weights, shape):
    """原版 build_graph 去掉 igraph 那一步：返回 (sources, targets, weights)。"""
    sp = pytest.importorskip("scipy.sparse")
    csr = sp.csr_matrix((np.array(weights), (np.array(rows), np.array(columns))), shape=shape)
    csr = csr + csr.transpose() - csr.multiply(csr.transpose())
    csr.setdiag(0)
    csr = sp.triu(csr)
    csr.eliminate_zeros()
    sources, targets = csr.nonzero()
    return sources, targets, csr.data


def test_fuzzy_union_matches_original_bitwise():
    x = clustered(1000, 16, 10)
    g = cp.knn(x, 10)
    w = cp.reweight(g, "tanimoto")
    rows = np.repeat(np.arange(g.n), g.k)
    want = original_build_graph_edges(rows, g.idx.ravel(), w.ravel(), (g.n, g.n))
    got = cp.build_graph(x, 10)
    for a, b in zip(got, want):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------- 输入诊断（docs/04 §6.6）


def test_warns_when_rows_are_unit_normalised():
    x = clustered(300, 8, 11)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    g = cp.knn(x, 5)
    assert g.meta["diagnostics"]["row_norm_cv"] < 1e-4
    with pytest.warns(cp.CostalWarning, match="单位化"):
        cp.reweight(g, "tanimoto")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cp.reweight(g, "cosine")  # cosine 边权不涉及幅度，不该警告


def test_warns_when_columns_are_centered():
    x = clustered(300, 8, 12)
    x -= x.mean(axis=0)
    g = cp.knn(x, 5)
    assert g.meta["diagnostics"]["columns_centered"]
    with pytest.warns(cp.CostalWarning, match="居中"):
        cp.reweight(g, "tanimoto")


def test_no_warnings_on_cytof_like_input():
    g = cp.knn(clustered(300, 8, 13), 5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cp.reweight(g, "tanimoto")
