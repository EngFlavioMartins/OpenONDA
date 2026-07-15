"""Structured convergence and run-health records for the FVM solver."""

from __future__ import annotations

from dataclasses import dataclass, field

from .linear_interface import LinearSolveResult


@dataclass(frozen=True)
class OuterCorrectorDiagnostics:
    """Convergence state after one PIMPLE outer corrector."""

    index: int
    momentum_residual: float
    pressure_residual: float
    continuity_max: float


@dataclass(frozen=True)
class StepDiagnostics:
    """Machine-readable health record for an accepted or rejected step."""

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
