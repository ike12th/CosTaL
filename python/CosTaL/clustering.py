"""CosTaL 0.0.2's ``CosTaL.clustering`` module: ``from CosTaL.clustering import find_communities``
keeps working. ``find_communities`` is the 1.0 function, which accepts the 0.0.2 arguments."""

from ._api import find_communities
from ._legacy import clustering, normalize_arcsinh, pp

__all__ = ["clustering", "find_communities", "normalize_arcsinh", "pp"]
