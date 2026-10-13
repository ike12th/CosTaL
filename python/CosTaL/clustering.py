"""CosTaL 1.0's ``CosTaL.clustering`` module: ``from CosTaL.clustering import find_communities``
keeps working. ``find_communities`` is the 2.0 function, which accepts the 1.0 arguments."""

from ._api import find_communities
from ._legacy import clustering, normalize_arcsinh, pp

__all__ = ["clustering", "find_communities", "normalize_arcsinh", "pp"]
