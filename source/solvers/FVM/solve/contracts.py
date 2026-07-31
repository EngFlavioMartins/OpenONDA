"""Structured convergence and run-health records for the FVM solver."""

from __future__ import annotations

from dataclasses import dataclass, field

from .linear_interface import LinearSolveResult


@dataclass(frozen=True)
class OuterCorrectorDiagnostics:
    """Convergence state after one PIMPLE outer corrector pass.

    The outer-corrector loop repeats the momentum-predict and pressure-
    correct sequence.  Each pass produces the reported residuals and the
    maximum cell divergence after the correction.

    Attributes
    ----------
    index : int
        Zero-based outer-corrector index.
    momentum_residual : float
        Dimensionless residual of the momentum equation after the predict.
    pressure_residual : float
        Dimensionless residual of the pressure Poisson equation.
    continuity_max : float
        Maximum cell divergence of velocity (|div U|) in 1/s.
    """

    index: int
    momentum_residual: float
    pressure_residual: float
    continuity_max: float


@dataclass(frozen=True)
class StepDiagnostics:
    """Immutable health record for one solver time step.

    Aggregates convergence metrics, field extrema, CFL number, and
    turbulence diagnostics for a single accepted (or rejected) step.
    The solver's diagnostic pipeline writes these records to the JSONL
    output and the acceptance policy evaluates them for abort conditions.

    Attributes
    ----------
    algorithm : str
        Solver algorithm name (``"PIMPLE"``, ``"SIMPLE"``, or ``"PISO"``).
    step : int
        Zero-based time-step index.
    time : float
        Physical time at the end of the step.
    dt : float
        Time-step size used.
    residuals : dict[str, float]
        Equation residuals keyed by field name.
    outer_correctors : tuple[OuterCorrectorDiagnostics, ...]
        Per-corrector convergence data (PIMPLE/PISO only).
    linear_solves : tuple[LinearSolveResult, ...]
        Backend-neutral records for every linear solve in the step.
    continuity_max : float
        Maximum cell divergence in 1/s.
    continuity_sum : float
        Sum of absolute cell divergences in 1/s.
    boundary_mass_balance : float
        Net boundary mass flux (should be near zero for incompressible flow).
    cfl_max : float
        Maximum Courant number.
    velocity_min / velocity_max : tuple[float, float, float]
        Component-wise velocity extrema.
    pressure_min / pressure_max : float
        Scalar pressure extrema.
    nonfinite_count : int
        Number of non-finite entries detected (zero for a healthy step).
    kinetic_energy : float
        Volume-integrated kinetic energy.
    enstrophy : float
        Volume-integrated enstrophy.
    turbulence_min / turbulence_max : float or None
        Eddy-viscosity extrema (``None`` for DNS/ILES).
    warnings : tuple[str, ...]
        Human-readable warnings that did not trigger an abort.
    """

    algorithm: str
    step: int
    time: float
    dt: float
    residuals: dict[str, float]
    outer_correctors: tuple[OuterCorrectorDiagnostics, ...]
    linear_solves: tuple[LinearSolveResult, ...]
    continuity_max: float
    continuity_sum: float
    boundary_mass_balance: float
    cfl_max: float
    velocity_min: tuple[float, float, float]
    velocity_max: tuple[float, float, float]
    pressure_min: float
    pressure_max: float
    nonfinite_count: int
    kinetic_energy: float
    enstrophy: float
    turbulence_min: float | None = None
    turbulence_max: float | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
