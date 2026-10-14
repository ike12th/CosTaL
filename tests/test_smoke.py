"""Skeleton self-check: the extension loads, numpy round-trips, and the versions match."""

import numpy as np
import pytest

import costal


def test_hello_mentions_version():
    assert costal.core_version() in costal.hello()


def test_wheel_and_core_versions_match():
    assert costal.__version__ == costal.core_version()


def test_row_l2_norms_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((257, 33), dtype=np.float32)
    got = costal.row_l2_norms(x)
    want = np.linalg.norm(x, axis=1)
    assert got.dtype == np.float32
    assert got.shape == (257,)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_row_l2_norms_rejects_non_contiguous():
    x = np.zeros((8, 8), dtype=np.float32)[:, ::2]
    with pytest.raises(ValueError):
        costal.row_l2_norms(x)


def test_pyproject_version_matches_the_extension():
    import re
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.exists():  # pragma: no cover - tests run outside the source tree
        pytest.skip("no pyproject.toml next to the tests")
    version = re.search(r'^version = "([^"]+)"', pyproject.read_text(), re.M).group(1)
    assert version == costal.__version__
