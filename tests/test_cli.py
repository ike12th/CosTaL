"""The command line (docs/01 §3): same labels as the Python API, on every input format it reads."""

import json

import numpy as np
import pytest

import costal as cp
from costal.cli import main


def cells(seed=0, n=300, d=6):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(20, 400, size=(3, d))
    return np.abs(centers[rng.integers(0, 3, n)] + rng.normal(0, 30, (n, d)))


def read_labels(path):
    rows = path.read_text().splitlines()
    assert rows[0] == "cell,cluster"
    return np.array([int(r.split(",")[1]) for r in rows[1:]])


def test_info():
    assert main(["info"]) == 0


def test_npy_with_platform_matches_the_api(tmp_path, capsys):
    pytest.importorskip("igraph")
    x = cells()
    np.save(tmp_path / "x.npy", x)
    out = tmp_path / "labels.csv"
    assert main(["cluster", str(tmp_path / "x.npy"), "--platform", "mass", "--leiden", "fast", "--seed", "0",
                 "-o", str(out), "--json"]) == 0
    expected = cp.find_communities(cp.preprocess(x, "mass"), leiden="fast", seed=0)
    np.testing.assert_array_equal(read_labels(out), expected)
    summary = json.loads(capsys.readouterr().out)
    assert summary["cells"] == 300 and summary["clusters"] == expected.max() + 1


def test_csv_drops_text_columns_and_selects(tmp_path):
    pytest.importorskip("igraph")
    pd = pytest.importorskip("pandas")
    x = cells(seed=1)
    df = pd.DataFrame(x, columns=[f"m{i}" for i in range(6)]).assign(sample=["s"] * 300)
    df.to_csv(tmp_path / "x.csv", index=False)
    assert main(["cluster", str(tmp_path / "x.csv"), "--leiden", "fast", "--seed", "0",
                 "--exclude", "m5"]) == 0
    expected = cp.find_communities(x[:, :5], leiden="fast", seed=0)
    np.testing.assert_array_equal(read_labels(tmp_path / "x_clusters.csv"), expected)  # default output name


def test_unknown_column_and_paper_compat_errors(tmp_path):
    np.save(tmp_path / "x.npy", cells(n=60))
    with pytest.raises(SystemExit, match="not in the input"):
        main(["cluster", str(tmp_path / "x.npy"), "--columns", "0,99"])
    assert main(["cluster", str(tmp_path / "x.npy"), "--multiscale", "--compat", "paper"]) == 2


def test_multiscale(tmp_path):
    pytest.importorskip("igraph")
    x = cells(seed=2)
    np.save(tmp_path / "x.npy", x)
    out = tmp_path / "labels.npy"
    assert main(["cluster", str(tmp_path / "x.npy"), "--multiscale", "-o", str(out)]) == 0
    np.testing.assert_array_equal(np.load(out), cp.find_communities_multiscale(x))
