"""Numerical stabilization mechanisms for the VPM solver.

Every mechanism that trades a declared amount of physics for numerical
robustness lives here, ordered from cheapest to most invasive:

``operators``
    Per-particle residual stretching viscosity and Pedrizzetti relaxation.
``filament_refinement``
    Invariant-preserving bisection of over-stretched Lagrangian elements.
``divergence_relaxation``
    Moment-constrained Winckelmans/Helmholtz projection of the strengths.
``regularization``
    Health-triggered conservative redistribution of a distorted cloud.
``manager``
    :class:`StabilizationManager`, the single per-step entry point the solver
    calls, and the owner of every stabilization diagnostic.

Configuration for all of them is centralized in
:class:`~source.solvers.VPM.config.types.StabilizationConfig`.
"""

from .divergence_relaxation import DivergenceRelaxationError, DivergenceRelaxationResult
from .filament_refinement import FilamentRefinementError, FilamentRefinementResult
from .manager import StabilizationError, StabilizationHealth, StabilizationManager
from .operators import StabilizationOperatorsMixin
from .regularization import RegularizationOutcome, regularize

__all__ = [
    "DivergenceRelaxationError",
    "DivergenceRelaxationResult",
    "FilamentRefinementError",
    "FilamentRefinementResult",
    "RegularizationOutcome",
    "StabilizationError",
    "StabilizationHealth",
    "StabilizationManager",
    "StabilizationOperatorsMixin",
    "regularize",
]
