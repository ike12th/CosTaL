"""costal_plus.legacy: the CosTaL 0.0.2 API on CosTaL+, and the CosTaL compatibility package."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import costal_plus as cp
from costal_plus import legacy

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DESCRIPTIONS = ROOT / "bench" / "reference" / "CosTaL-0.0.2.4" / "CosTaL" / "descriptions.py"


def cells(seed=0, n=300, d=6):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(20, 400, size=(3, d))
    return np.abs(centers[rng.integers(0, 3, n)] + rng.normal(0, 30, (n, d)))


def test_find_communities_is_the_published_method():
    pytest.importorskip("leidenalg")
    x = cells()
    got = legacy.find_communities(x, pp_method="mass", seed=0)
    assert isinstance(got, list) and len(got) == len(x)
    assert got == cp.find_communities(cp.preprocess(x, "mass"), seed=0).tolist()
    assert legacy.clustering(x, pp_method="mass", seed=0) == got
    cosine = legacy.find_communities(x, pp_method="mass", method="cosine", seed=0)
    assert cosine == cp.find_communities(cp.preprocess(x, "mass"), weight="cosine", seed=0).tolist()


def test_other_partition_types_and_unsupported_options():
    la = pytest.importorskip("leidenalg")
    x = cells(seed=1)
    labels = legacy.find_communities(x, pp_method="mass", partition_type=la.ModularityVertexPartition, seed=0)
    assert len(labels) == len(x) and max(labels) >= 1
    with pytest.raises(NotImplementedError, match="local=True"):
        legacy.find_communities(x, local=True)
    with pytest.raises(RuntimeError, match="Method not supported"):
        legacy.find_communities(x, method="euclid")


def test_pp_matches_the_original_conventions():
    x = cells()
    np.testing.assert_array_equal(legacy.pp(x, "mass"), cp.preprocess(x, "mass"))
    assert legacy.pp(x) is x  # no platform: values untouched
    np.testing.assert_allclose(legacy.normalize_arcsinh(x, 150), np.arcsinh(x / 150))


def original_descriptions():
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    if not ORIGINAL_DESCRIPTIONS.exists():  # pragma: no cover
        pytest.skip("the original CosTaL source is not in bench/reference")
    spec = importlib.util.spec_from_file_location("original_descriptions", ORIGINAL_DESCRIPTIONS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scores_match_the_original(seed):
    orig = original_descriptions()
    rng = np.random.default_rng(seed)
    ytrue = rng.integers(0, 6, 400).astype(float)
    ytrue[rng.random(400) < 0.1] = np.nan  # unlabeled cells
    ypred = np.where(rng.random(400) < 0.7, np.nan_to_num(ytrue, nan=0).astype(int), rng.integers(0, 9, 400))
    ours, theirs = legacy.return_scores(ytrue, ypred), orig.return_scores(ytrue, ypred)
    assert len(ours) == len(theirs) == 12
    np.testing.assert_allclose(np.array(ours, dtype=float), np.array(theirs, dtype=float), rtol=1e-12)
    assert legacy.get_rare_f1(ytrue[:50] > 3, ypred[:50]) == pytest.approx(orig.get_rare_f1(ytrue[:50] > 3, ypred[:50]))
    x = rng.normal(size=(20, 5)) * (rng.random((20, 5)) < 0.5)
    assert legacy.calculate_sparsity(x) == orig.calculate_sparsity(x)


def test_compatibility_package(monkeypatch):
    pytest.importorskip("leidenalg")
    monkeypatch.syspath_prepend(str(ROOT / "packaging" / "costal-compat"))
    for name in [m for m in sys.modules if m == "CosTaL" or m.startswith("CosTaL.")]:
        monkeypatch.delitem(sys.modules, name)
    with pytest.warns(FutureWarning, match="CosTaL\\+"):
        import CosTaL
    from CosTaL.clustering import find_communities
    from CosTaL.descriptions import return_scores

    assert CosTaL.find_communities is find_communities is legacy.find_communities
    assert CosTaL.return_scores is return_scores is legacy.return_scores
    assert callable(CosTaL.clustering)  # as in the original, the function shadows the module name
    x = cells(seed=3)
    assert CosTaL.clustering(x, pp_method="mass", seed=0) == legacy.clustering(x, pp_method="mass", seed=0)
