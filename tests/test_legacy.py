"""CosTaL 1.0 code keeps working on 2.0: the same names, arguments, import paths and list results."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import CosTaL

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DESCRIPTIONS = ROOT / "bench" / "reference" / "CosTaL-0.0.2.4" / "CosTaL" / "descriptions.py"


def cells(seed=0, n=300, d=6):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(20, 400, size=(3, d))
    return np.abs(centers[rng.integers(0, 3, n)] + rng.normal(0, 30, (n, d)))


def test_find_communities_v1_call_is_the_published_method():
    pytest.importorskip("leidenalg")
    x = cells()
    got = CosTaL.find_communities(x, pp_method="mass", seed=0)
    assert isinstance(got, list) and len(got) == len(x)
    assert got == CosTaL.find_communities(CosTaL.preprocess(x, "mass"), seed=0).tolist()
    assert CosTaL.find_communities(x, "mass", seed=0) == got  # 1.0's positional pp_method
    assert CosTaL.clustering(x, pp_method="mass", seed=0) == got
    cosine = CosTaL.find_communities(x, pp_method="mass", method="cosine", nbr_num=10, seed=0)
    assert cosine == CosTaL.find_communities(CosTaL.preprocess(x, "mass"), weight="cosine", seed=0).tolist()
    accepted = CosTaL.find_communities(x, pp_method="mass", seed=0, verbose=True, keep_temp_files=True, temp_name="t")
    assert accepted == got  # 1.0 housekeeping arguments are accepted and have nothing to do


def test_new_call_returns_an_array():
    pytest.importorskip("leidenalg")
    labels = CosTaL.find_communities(CosTaL.preprocess(cells(), "mass"), seed=0)
    assert isinstance(labels, np.ndarray) and labels.dtype == np.int64


def test_mixing_interfaces_and_typos_are_errors():
    x = cells(n=50)
    with pytest.raises(ValueError, match="can't be combined"):
        CosTaL.find_communities(x, nbr_num=10, leiden="fast")
    with pytest.raises(TypeError, match="unexpected keyword"):
        CosTaL.find_communities(x, resolutoin=0.5)


def test_other_partition_types_and_unsupported_options():
    la = pytest.importorskip("leidenalg")
    x = cells(seed=1)
    labels = CosTaL.find_communities(x, pp_method="mass", partition_type=la.ModularityVertexPartition, seed=0)
    assert len(labels) == len(x) and max(labels) >= 1
    with pytest.raises(NotImplementedError, match="local=True"):
        CosTaL.find_communities(x, local=True)
    with pytest.raises(RuntimeError, match="Method not supported"):
        CosTaL.find_communities(x, method="euclid")


def test_import_paths_of_v1():
    from CosTaL.clustering import clustering, find_communities, normalize_arcsinh, pp
    from CosTaL.descriptions import return_scores

    assert find_communities is CosTaL.find_communities and clustering is CosTaL.clustering
    assert callable(CosTaL.clustering)  # as in 1.0, the function shadows the submodule's name
    assert return_scores is CosTaL.return_scores
    x = cells()
    np.testing.assert_array_equal(pp(x, "mass"), CosTaL.preprocess(x, "mass"))
    assert pp(x) is x  # no platform: values untouched
    np.testing.assert_allclose(normalize_arcsinh(x, 150), np.arcsinh(x / 150))


def original_descriptions():
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    if not ORIGINAL_DESCRIPTIONS.exists():  # pragma: no cover
        pytest.skip("the 1.0 source is not in bench/reference")
    spec = importlib.util.spec_from_file_location("costal_v1_descriptions", ORIGINAL_DESCRIPTIONS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scores_match_v1(seed):
    orig = original_descriptions()
    rng = np.random.default_rng(seed)
    ytrue = rng.integers(0, 6, 400).astype(float)
    ytrue[rng.random(400) < 0.1] = np.nan  # unlabeled cells
    ypred = np.where(rng.random(400) < 0.7, np.nan_to_num(ytrue, nan=0).astype(int), rng.integers(0, 9, 400))
    ours, theirs = CosTaL.return_scores(ytrue, ypred), orig.return_scores(ytrue, ypred)
    assert len(ours) == len(theirs) == 12
    np.testing.assert_allclose(np.array(ours, dtype=float), np.array(theirs, dtype=float), rtol=1e-12)
    assert CosTaL.get_rare_f1(ytrue[:50] > 3, ypred[:50]) == pytest.approx(orig.get_rare_f1(ytrue[:50] > 3, ypred[:50]))
    x = rng.normal(size=(20, 5)) * (rng.random((20, 5)) < 0.5)
    assert CosTaL.calculate_sparsity(x) == orig.calculate_sparsity(x)
