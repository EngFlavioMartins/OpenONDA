"""Native finite-volume, vortex-particle, and vortex-lattice solver internals.

Use ``openonda.fvm``, ``openonda.vpm``, and ``openonda.coupler`` to configure
and run simulations.
"""

from . import solvers
from .version import __version__, __version_info__

__all__ = [
    "__version__",
    "__version_info__",
    "solvers",
]


def get_config():
    """Return basic OpenONDA configuration."""
    return {
        "version": __version__,
        "solvers": ("FVM", "VPM"),
        "coupling": "FVM-VPM",
    }
