"""Numerical scheme registry for the OpenONDA FVM solver.

Currently hosts the TVD flux limiters used by the high-resolution convection
schemes (:mod:`..assemble.convection`).  This package is the intended home for
the wider discretisation selection (div/grad/laplacian/ddt) as it grows.
"""

from .boundaries import validate_boundary_conditions
from .limiters import LIMITERS, apply_limiter, is_limited_scheme

__all__ = [
    "LIMITERS",
    "apply_limiter",
    "is_limited_scheme",
    "validate_boundary_conditions",
]
