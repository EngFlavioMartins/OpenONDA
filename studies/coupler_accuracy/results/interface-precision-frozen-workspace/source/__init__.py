# OpenONDA/__init__.py
"""
OpenONDA: Native FVM and vortex-method CFD
===========================================

OpenONDA provides native finite-volume, vortex-particle, and vortex-lattice
solvers with a conservative FVM-VPM coupling layer.

Available modules:
- solvers.FVM: Pure-Python finite-volume solver
- solvers.VPM: Taichi-accelerated vortex-particle and vortex-lattice solvers
- coupler: Conservative native FVM-VPM coupling

Usage:
    from source.solvers.fvm import Solver
    from source.solvers.vpm import Solver as VPMSolver
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
