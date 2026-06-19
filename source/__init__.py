# OpenONDA/__init__.py
"""
OpenONDA: Python Interface for OpenFOAM
========================================

OpenONDA provides a Cython-based Python wrapper for stepwise control of
OpenFOAM's PIMPLE solver, enabling in-line coupling with external solvers
and scientific computing libraries (NumPy, SciPy).

Available modules:
- solvers.OFW: Cython-based OpenFOAM wrapper (requires a local OpenFOAM build)
- solvers.FVM: Pure-Python finite volume solver (NumPy/SciPy, no OpenFOAM needed)

Usage:
    from source.solvers.OFW import Solver   # OpenFOAM wrapper
    from source.solvers.FVM import Solver   # Pure-Python FVM
"""

from . import solvers
from .version import __version__, __version_info__

# Don't eagerly import utilities to avoid blocking FVM import when VPM fails
# Users can import from utilities explicitly if needed:
#   from source.utilities import OfflineFlowDiagnosticsVPM
#   from source.utilities import LambOseenFVM
# or
#   from source import utilities

__all__ = [
    "__version__",
    "__version_info__",
    "solvers",
]

try:
    import importlib.util as _ilu

    _has_ofw = _ilu.find_spec("source.solvers.OFW") is not None
    del _ilu
except Exception:
    _has_ofw = False


def get_config():
    """Return basic OpenONDA configuration."""
    return {
        "version": __version__,
        "has_ofw": _has_ofw,
        "installation_type": "full" if _has_ofw else "fvm-only",
    }
