# OpenONDA/solvers/__init__.py

try:
    from . import FVM  # noqa: F401

    _has_fvm = True
except ImportError as e:
    _has_fvm = False
    import warnings

    warnings.warn(
        f"FVM module failed to import: {e}. FVM functionality will be unavailable.",
        ImportWarning,
        stacklevel=2,
    )

# OFW is intentionally NOT imported at package-load time — the Cython
# extension links against OpenFOAM shared libraries.  Import explicitly:
#   from source.solvers.OFW import Solver
import importlib.util as _importlib_util

_has_ofw = _importlib_util.find_spec("source.solvers.OFW") is not None
del _importlib_util

__all__ = []
if _has_fvm:
    __all__.append("FVM")
if _has_ofw:
    __all__.append("OFW")
