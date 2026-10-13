"""CosTaL 0.0.2's API, running on CosTaL+ (https://github.com/ike12th/costal-plus)."""

import warnings

from CosTaL.clustering import *  # noqa: F401,F403  (as the original: the function shadows the module)
from CosTaL.descriptions import *  # noqa: F401,F403

warnings.warn(
    "CosTaL is now CosTaL+: this package runs the CosTaL 0.0.2 API on costal-plus. New code can "
    "`import costal_plus` directly (see its user guide, section 11).",
    FutureWarning,
    stacklevel=2,
)
