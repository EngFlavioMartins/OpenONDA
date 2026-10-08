"""Infrastructure shared by the solvers: restart, output layout, logging and slab images.

Use ``openonda.fvm``, ``openonda.vpm``, and ``openonda.coupler`` to configure
and run simulations.
"""

from .version import __version__, __version_info__

__all__ = [
    "__version__",
    "__version_info__",
]
