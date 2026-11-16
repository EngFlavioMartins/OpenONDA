"""Convergence and field records of one FVM time step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..solve.linear_interface import LinearSolveResult
from . import fields as field_diagnostics

if TYPE_CHECKING:
    from ..core.solver import FVMSolver


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
    state_update : float
        Dimensionless maximum normalized change of the velocity and pressure
        state across this outer corrector.  This is the nonlinear stopping
        metric; the two residual fields above remain linear-solve telemetry.
    """

    index: int
    velocity_residual: float
    kinematic_pressure_residual: float
    max_continuity_error: float
    state_update: float = 0.0


@dataclass(frozen=True)
class StepDiagnostics:
    """Immutable field and convergence record for one solver time step.

    Aggregates convergence metrics, field extrema, CFL number, and
    turbulence diagnostics for a single accepted (or rejected) step.
    The solver's diagnostic sequence writes these records to the JSONL
    output and the acceptance limits evaluate them for abort conditions.

    Attributes
    ----------
    step : int
        Number of the step just completed, counted from one.
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
    max_velocity_magnitude : float
        Maximum cell-centred speed over owned cells [m/s].
    min_kinematic_pressure / max_kinematic_pressure : float
        Kinematic-pressure extrema.
    n_nonfinite_values : int
        Number of non-finite entries detected (zero for a finite step).
    total_kinetic_energy : float
        Volume-integrated kinetic energy.
    total_enstrophy : float
        Volume-integrated enstrophy.
    min_eddy_viscosity / max_eddy_viscosity : float or None
        Eddy-viscosity extrema (``None`` for DNS/ILES).
    warnings : tuple[str, ...]
        Human-readable warnings that did not trigger an abort.
    """

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
    max_velocity_magnitude: float
    min_eddy_viscosity: float | None = None
    max_eddy_viscosity: float | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def build_step_diagnostics(
    solver: FVMSolver, step_time_step_size: float, residuals: dict[str, object]
) -> StepDiagnostics:
    """Build the backend-neutral field and convergence record of the solved candidate step.

    Reads the solved fields of ``solver`` and sets its ``max_courant_number``.
    """
    n_cells = solver.mesh_data["n_cells"]
    n_owned = solver.parallel.n_owned if solver.parallel.is_partitioned else n_cells
    interior_velocity = np.asarray(solver.velocity[:n_owned])
    interior_kinematic_pressure = np.asarray(solver.kinematic_pressure[:n_owned])
    solver.max_courant_number = solver._measure_maximum_courant_number(step_time_step_size)
    local_nonfinite = int(
        np.count_nonzero(~np.isfinite(interior_velocity))
        + np.count_nonzero(~np.isfinite(interior_kinematic_pressure))
        + np.count_nonzero(~np.isfinite(solver.volumetric_face_flux))
    )
    n_nonfinite_values = int(solver.parallel.global_sum(local_nonfinite))
    min_eddy_viscosity = None
    max_eddy_viscosity = None
    if solver.eddy_viscosity is not None:
        n_nonfinite_values += int(
            solver.parallel.global_sum(
                int(np.count_nonzero(~np.isfinite(solver.eddy_viscosity[:n_owned])))
            )
        )
        local_eddy = np.asarray(solver.eddy_viscosity[:n_owned])
        min_eddy_viscosity = float(
            solver.parallel.global_min(float(np.nanmin(local_eddy)) if n_owned else float("inf"))
        )
        max_eddy_viscosity = float(
            solver.parallel.global_max(float(np.nanmax(local_eddy)) if n_owned else float("-inf"))
        )
    n_interior = solver.mesh_data["n_interior_faces"]
    linear_results = tuple(getattr(solver.algorithm, "last_linear_results", ()))
    local_min_velocity = np.nanmin(interior_velocity, axis=0) if n_owned else np.full(3, np.inf)
    local_max_velocity = np.nanmax(interior_velocity, axis=0) if n_owned else np.full(3, -np.inf)
    min_velocity = np.asarray(
        [solver.parallel.global_min(float(value)) for value in local_min_velocity]
    )
    max_velocity = np.asarray(
        [solver.parallel.global_max(float(value)) for value in local_max_velocity]
    )
    if n_owned:
        finite_velocity = np.where(np.isfinite(interior_velocity), interior_velocity, 0.0)
        local_max_velocity_magnitude = float(np.max(np.linalg.norm(finite_velocity, axis=1)))
    else:
        local_max_velocity_magnitude = 0.0
    max_velocity_magnitude = float(solver.parallel.global_max(local_max_velocity_magnitude))
    min_kinematic_pressure = float(
        solver.parallel.global_min(
            float(np.nanmin(interior_kinematic_pressure)) if n_owned else float("inf")
        )
    )
    max_kinematic_pressure = float(
        solver.parallel.global_max(
            float(np.nanmax(interior_kinematic_pressure)) if n_owned else float("-inf")
        )
    )
    local_kinetic_energy = (
        0.5
        * solver._configuration.numerics.transport.density
        * float(
            np.sum(
                solver.geo_data["cell_volume"][:n_owned]
                * np.sum(interior_velocity * interior_velocity, axis=1)
            )
        )
    )
    local_enstrophy_integral = field_diagnostics.enstrophy_from_gradient(
        solver._velocity_gradient(),
        solver.geo_data["cell_volume"],
        n_owned,
    )
    return StepDiagnostics(
        step=solver.step + 1,
        time=solver.time + step_time_step_size,
        time_step_size=float(step_time_step_size),
        residuals={key: float(value) for key, value in residuals.items()},
        outer_correctors=tuple(getattr(solver.algorithm, "last_outer_diagnostics", ())),
        linear_solves=linear_results,
        max_continuity_error=solver.max_continuity_error,
        sum_absolute_continuity_error=solver.sum_absolute_continuity_error,
        net_boundary_volumetric_flux=float(
            solver.parallel.global_sum(float(np.sum(solver.volumetric_face_flux[n_interior:])))
        ),
        max_courant_number=solver.max_courant_number,
        min_velocity=(
            float(min_velocity[0]),
            float(min_velocity[1]),
            float(min_velocity[2]),
        ),
        max_velocity=(
            float(max_velocity[0]),
            float(max_velocity[1]),
            float(max_velocity[2]),
        ),
        max_velocity_magnitude=max_velocity_magnitude,
        min_kinematic_pressure=min_kinematic_pressure,
        max_kinematic_pressure=max_kinematic_pressure,
        n_nonfinite_values=n_nonfinite_values,
        total_kinetic_energy=float(solver.parallel.global_sum(local_kinetic_energy)),
        total_enstrophy=float(solver.parallel.global_sum(local_enstrophy_integral)),
        min_eddy_viscosity=min_eddy_viscosity,
        max_eddy_viscosity=max_eddy_viscosity,
    )
