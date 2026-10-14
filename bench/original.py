"""The original CosTaL 0.0.2.4 source (bench/reference), loaded under its own module name.

In 2.0, ``import CosTaL`` (1.0's import name) is an alias of the ``costal`` package, so
``import CosTaL.clustering`` reaches 2.0, never the 1.0 source. The 1.0 ``clustering.py`` has no
package-internal imports, so it loads on its own as ``costal_002_clustering``, next to CosTaL 2.0.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent / "reference" / "CosTaL-0.0.2.4"


def clustering():
    """The 0.0.2.4 ``CosTaL/clustering.py`` module."""
    name = "costal_002_clustering"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REFERENCE / "CosTaL" / "clustering.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
