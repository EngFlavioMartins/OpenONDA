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
    velocity_residual : float
        Dimensionless residual of the momentum equation after the predict.
    kinematic_pressure_residual : float
        Dimensionless residual of the pressure Poisson equation.
    max_continuity_error : float
        Maximum cell divergence of velocity (|div velocity|) in 1/s.
    """

    index: int
    velocity_residual: float
    kinematic_pressure_residual: float
    max_continuity_error: float


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
    time_step_size : float
        Time-step size used.
    residuals : dict[str, float]
        Equation residuals keyed by field name.
    outer_correctors : tuple[OuterCorrectorDiagnostics, ...]
        Per-corrector convergence data (PIMPLE/PISO only).
    linear_solves : tuple[LinearSolveResult, ...]
        Backend-neutral records for every linear solve in the step.
    max_continuity_error : float
        Maximum cell divergence in 1/s.
    sum_absolute_continuity_error : float
        Sum of absolute cell divergences in 1/s.
    net_boundary_volumetric_flux : float
        Net boundary volumetric flux [m³/s] (near zero for incompressible flow).
    max_courant_number : float
        Maximum Courant number.
    min_velocity / max_velocity : tuple[float, float, float]
        Component-wise velocity extrema.
    min_kinematic_pressure / max_kinematic_pressure : float
        Kinematic-pressure extrema.
    n_nonfinite_values : int
        Number of non-finite entries detected (zero for a healthy step).
    total_kinetic_energy : float
        Volume-integrated kinetic energy.
    total_enstrophy : float
        Volume-integrated enstrophy.
    min_eddy_viscosity / max_eddy_viscosity : float or None
        Eddy-viscosity extrema (``None`` for DNS/ILES).
    state_projection : dict[str, float]
        Maximum amplitudes removed by an optional accepted-state projection.
        Empty when no projection is installed.
    warnings : tuple[str, ...]
        Human-readable warnings that did not trigger an abort.
    """

    algorithm: str
    step: int
    time: float
    time_step_size: float
    residuals: dict[str, float]
    outer_correctors: tuple[OuterCorrectorDiagnostics, ...]
    linear_solves: tuple[LinearSolveResult, ...]
    max_continuity_error: float
    sum_absolute_continuity_error: float
    net_boundary_volumetric_flux: float
    max_courant_number: float
    min_velocity: tuple[float, float, float]
    max_velocity: tuple[float, float, float]
    min_kinematic_pressure: float
    max_kinematic_pressure: float
    n_nonfinite_values: int
    total_kinetic_energy: float
    total_enstrophy: float
    min_eddy_viscosity: float | None = None
    max_eddy_viscosity: float | None = None
    state_projection: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)
