"""``import CosTaL``: CosTaL 1.0's import name, kept for 1.0 code. The package is ``costal``.

Everything here is the ``costal`` package itself, not a copy: ``CosTaL.find_communities is
costal.find_communities``, and ``from CosTaL.clustering import …`` / ``from CosTaL.descriptions
import …`` reach ``costal.clustering`` / ``costal.descriptions``. (A second package directory can't
be used: on macOS and Windows ``CosTaL/`` and ``costal/`` would be the same directory.)
"""

import sys as _sys

from costal import *  # noqa: F401,F403
from costal import __all__, __version__  # noqa: F401

# 1.0's submodules: the import system finds these before looking for a package directory
_sys.modules[__name__ + ".clustering"] = _sys.modules["costal.clustering"]
_sys.modules[__name__ + ".descriptions"] = _sys.modules["costal.descriptions"]
