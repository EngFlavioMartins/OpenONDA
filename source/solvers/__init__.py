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

__all__ = []
if _has_fvm:
    __all__.append("FVM")
