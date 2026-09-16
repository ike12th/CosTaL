"""骨架自检：扩展模块能加载、numpy 能来回穿、版本号对得上。"""

import numpy as np
import pytest

import costal_plus


def test_hello_mentions_version():
    assert costal_plus.core_version() in costal_plus.hello()


def test_wheel_and_core_versions_match():
    assert costal_plus.__version__ == costal_plus.core_version()


def test_row_l2_norms_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((257, 33), dtype=np.float32)
    got = costal_plus.row_l2_norms(x)
    want = np.linalg.norm(x, axis=1)
    assert got.dtype == np.float32
    assert got.shape == (257,)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_row_l2_norms_rejects_non_contiguous():
    x = np.zeros((8, 8), dtype=np.float32)[:, ::2]
    with pytest.raises(ValueError):
        costal_plus.row_l2_norms(x)
