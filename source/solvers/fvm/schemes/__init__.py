"""Numerical scheme registry for the OpenONDA FVM solver.

Currently hosts the TVD flux limiters used by the high-resolution convection
schemes (:mod:`..assemble.convection`).  This package is the intended home for
the wider discretisation selection (div/grad/laplacian/ddt) as it grows.
"""

from .limiters import LIMITERS, apply_limiter, is_limited_scheme
from .validation import (
    CONVECTION_SCHEMES,
    GRADIENT_SCHEMES,
    TIME_SCHEMES,
    TURBULENCE_MODELS,
    validate_acceptance_limits,
    validate_boundary_conditions,
    validate_solver_params,
    validate_turbulence,
)

__all__ = [
    "LIMITERS",
    "apply_limiter",
    "is_limited_scheme",
    "CONVECTION_SCHEMES",
    "GRADIENT_SCHEMES",
    "TIME_SCHEMES",
    "TURBULENCE_MODELS",
    "validate_boundary_conditions",
    "validate_acceptance_limits",
    "validate_solver_params",
    "validate_turbulence",
]
