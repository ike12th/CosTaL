"""Every script in examples/ runs (on its synthetic data), so the documented usage keeps working."""

import runpy
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("name, needs", [
    ("cytometry", ["igraph", "leidenalg", "sklearn"]),
    ("scrnaseq", ["igraph", "leidenalg", "sklearn", "scanpy", "skmisc.loess"]),
    ("step_by_step", ["igraph", "leidenalg", "sklearn"]),
    ("rare_populations", ["igraph"]),
    ("migrate_from_costal", ["igraph", "leidenalg", "sklearn", "pandas", "scipy"]),
])
def test_example_runs(name, needs, monkeypatch, capsys):
    for module in needs:
        try:
            __import__(module)
        except Exception as exc:  # ImportError, or a binary mismatch (scikit-misc on numpy 2)
            pytest.skip(f"{module} unavailable: {exc}")
    monkeypatch.syspath_prepend(str(EXAMPLES))
    monkeypatch.setattr(sys, "argv", [name])
    module = runpy.run_path(str(EXAMPLES / f"{name}.py"))
    if name == "rare_populations":
        module["main"]([name], n=4000)
    else:
        module["main"]([name])
    assert capsys.readouterr().out.strip()
