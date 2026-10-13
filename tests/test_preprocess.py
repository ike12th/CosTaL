"""preprocess(): the original pp() (bench/datasets.py checks it against the paper's Table 2 on
real data; these tests pin the behavior on synthetic data)."""

import warnings

import numpy as np
import pytest

import CosTaL as cp


@pytest.mark.parametrize("platform,cofactor", [("mass", 5), ("flow", 150), ("spectral flow", 6000)])
def test_cytometry_is_arcsinh_with_the_platform_cofactor(platform, cofactor):
    raw = np.random.default_rng(0).normal(50, 200, size=(100, 8))
    out, cells = cp.preprocess(raw, platform, return_cells=True)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.arcsinh(raw / cofactor).astype(np.float32))
    np.testing.assert_array_equal(cells, np.arange(100))


def test_dataframe_uses_numeric_columns():
    pd = pytest.importorskip("pandas")
    raw = np.random.default_rng(1).uniform(0, 100, size=(20, 3))
    df = pd.DataFrame(raw, columns=["a", "b", "c"]).assign(label=["x"] * 20)
    np.testing.assert_array_equal(cp.preprocess(df, "mass"), cp.preprocess(raw, "mass"))


def test_rejects_bad_arguments():
    x = np.ones((4, 3))
    with pytest.raises(ValueError, match="platform"):
        cp.preprocess(x, "cytof")
    with pytest.raises(ValueError, match="scrnaseq.* only"):
        cp.preprocess(x, "mass", preset="normalize_total")
    with pytest.raises(ValueError, match="compat='paper'"):
        cp.preprocess(x, "scrnaseq", preset="normalize_total", compat="paper")


def need_scanpy():
    """scanpy plus a working scikit-misc loess (seurat_v3); a binary mismatch with numpy fails
    only at import time of the extension, so probe it."""
    pytest.importorskip("scanpy")
    try:
        from skmisc.loess import loess  # noqa: F401
    except Exception as exc:  # ImportError, or ValueError on a numpy ABI mismatch
        pytest.skip(f"scikit-misc loess unavailable: {exc}")


def counts(seed=0, n=300, genes=3000, sparse_cells=(5, 77)):
    rng = np.random.default_rng(seed)
    mean = rng.gamma(0.3, 1.0, size=genes) * rng.uniform(0.5, 2.0, size=(n, 1))
    x = rng.poisson(mean).astype(np.float32)
    for i in sparse_cells:  # fewer than 200 detected genes: filter_cells drops them
        keep = np.flatnonzero(x[i])[:50]
        row = np.zeros(genes, dtype=np.float32)
        row[keep] = x[i, keep]
        x[i] = row
    return x


@pytest.mark.parametrize("preset", ["paper", "normalize_total"])
def test_scrnaseq_matches_the_scanpy_pipeline(preset):
    need_scanpy()
    import anndata
    import scanpy as sc

    x = counts()
    with pytest.warns(cp.CostalWarning, match="dropped 2 of 300"):
        out, cells = cp.preprocess(x, "scrnaseq", preset=preset, return_cells=True)
    np.testing.assert_array_equal(cells, np.setdiff1d(np.arange(300), [5, 77]))

    ad = anndata.AnnData(X=x.copy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.filter_cells(ad, min_genes=200)
        sc.pp.filter_genes(ad, min_cells=3)
        sc.pp.highly_variable_genes(ad, n_top_genes=2000, flavor="seurat_v3")
        if preset == "normalize_total":
            sc.pp.normalize_total(ad)
        sc.pp.log1p(ad)
    expected = ad.X[:, ad.var["highly_variable"].to_numpy()]
    assert out.shape == (298, 2000) and out.dtype == np.float32
    np.testing.assert_array_equal(out, np.asarray(expected, dtype=np.float32))
    assert out.min() >= 0  # log1p of counts: non-negative, zeros preserved (docs/04 §6.7)


def test_scrnaseq_accepts_sparse_counts():
    need_scanpy()
    sp = pytest.importorskip("scipy.sparse")
    x = counts(seed=2, sparse_cells=())
    np.testing.assert_array_equal(cp.preprocess(sp.csr_matrix(x), "scrnaseq"), cp.preprocess(x, "scrnaseq"))
