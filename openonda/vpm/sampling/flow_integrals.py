"""Global flow integrals and particle statistics of the VPM flow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from openonda.core.file_output import append_csv_rows
from openonda.core.sampling import Sampler
from openonda.core.schedules import EverySteps, Schedule

from ..io.logging import Logging

if TYPE_CHECKING:
    from ..core.solver import VPMSolver
    from .manager import SamplingContext

__all__ = ["FlowIntegralsSampler"]

_RESOLUTION_COLUMNS = (
    "mean_core_radius",
    "mean_overlap_ratio",
    "max_overlap_ratio",
    "vorticity_divergence_error",
    "vortex_strength_misalignment_degrees",
)


def flow_integral_row(solver: VPMSolver) -> dict[str, Any]:
    """Return the flow integrals and particle statistics of the current state.

    Circulation is in m³/s, impulses per density in m⁴/s and m⁵/s, energy
    per density in m⁵/s², helicity in m⁴/s², enstrophy in m³/s² and
    viscosities in m²/s.
    """
    integrals = solver._flow_integrals
    linear_impulse = integrals.get("linear_impulse", np.zeros(3))
    angular_impulse = integrals.get("angular_impulse", np.zeros(3))
    net_vortex_strength = integrals.get("net_vortex_strength", np.zeros(3))
    strength = solver.particle_vortex_strength
    core_radius = solver.particle_core_radius
    eddy_viscosity = solver.particles.eddy_viscosity_cpu()
    effective_viscosity = solver.particles.effective_viscosity_cpu()

    def mean(values: np.ndarray) -> float:
        return float(values.mean()) if len(values) else 0.0

    row: dict[str, Any] = {
        "time": solver.time,
        "total_kinetic_energy": solver.total_kinetic_energy,
        "energy_measurement": integrals.get("energy_measurement", "unknown"),
        "total_enstrophy": solver.total_enstrophy,
        "test_filtered_enstrophy": integrals.get("test_filtered_enstrophy", 0.0),
        "kinetic_energy_rate": solver.kinetic_energy_rate,
        "kinetic_energy_rate_source": integrals.get("kinetic_energy_rate_source", "unknown"),
        "viscous_kinetic_energy_rate": solver.viscous_kinetic_energy_rate,
        "total_helicity": solver.total_helicity,
        "vortex_strength_magnitude_sum": solver.vortex_strength_magnitude_sum,
    }
    for index, axis in enumerate("xyz"):
        row[f"net_vortex_strength_{axis}"] = float(net_vortex_strength[index])
    for index, axis in enumerate("xyz"):
        row[f"linear_impulse_{axis}"] = float(linear_impulse[index])
    for index, axis in enumerate("xyz"):
        row[f"angular_impulse_{axis}"] = float(angular_impulse[index])
    row.update(
        {
            "n_particles": solver.particles.n_particles_total,
            "max_vortex_strength_magnitude": float(
                np.linalg.norm(strength, axis=1).max(initial=0.0)
            ),
            "min_particle_core_radius": float(core_radius.min()) if len(core_radius) else 0.0,
            "mean_particle_core_radius": mean(core_radius),
            "max_particle_core_radius": float(core_radius.max(initial=0.0)),
            "mean_eddy_viscosity": mean(eddy_viscosity),
            "max_eddy_viscosity": float(eddy_viscosity.max(initial=0.0)),
            "mean_effective_viscosity": mean(effective_viscosity),
            "max_effective_viscosity": float(effective_viscosity.max(initial=0.0)),
        }
    )
    resolution = solver._particle_resolution_metrics
    row.update({name: resolution.get(name, float("nan")) for name in _RESOLUTION_COLUMNS})
    row.update(solver.stabilization.diagnostics)
    vlm = solver.vlm_solver
    if vlm is not None:
        bound_impulse = vlm.compute_bound_linear_impulse()
        bound_strength = vlm.compute_total_bound_vortex_strength()
        for index, axis in enumerate("xyz"):
            row[f"bound_linear_impulse_{axis}"] = float(bound_impulse[index])
            row[f"coupled_linear_impulse_{axis}"] = float(
                linear_impulse[index] + bound_impulse[index]
            )
            row[f"bound_vortex_strength_{axis}"] = float(bound_strength[index])
            row[f"coupled_vortex_strength_{axis}"] = float(
                net_vortex_strength[index] + bound_strength[index]
            )
    metrics = solver._accepted_state_metrics
    nan = float("nan")
    row.update(
        {
            "strain_increment_infinity": nan
            if metrics is None
            else metrics.strain_increment_infinity,
            "strain_increment_spectral": nan
            if metrics is None
            else metrics.strain_increment_spectral,
            "maximum_particle_strength": nan
            if metrics is None
            else metrics.maximum_particle_strength,
            "maximum_particle_vorticity": nan if metrics is None else metrics.maximum_vorticity,
        }
    )
    return row


class FlowIntegralsSampler(Sampler):
    """Append the flow integrals of each sampled state to ``flow_integrals.csv``.

    The table holds kinetic energy, enstrophy and helicity, their rates,
    circulation, linear and angular impulse, particle statistics,
    resolution metrics and stabilization diagnostics; an attached vortex
    lattice adds its bound circulation and impulse.

    Parameters
    ----------
    file_name : str, default='flow_integrals'
    schedule : EverySteps, EveryTime or FinalOnly, default=EverySteps(1)
    initial : bool, default=True
        Output arguments of :class:`~openonda.core.sampling.Sampler`.
    """

    requires_flow_integrals = True

    def __init__(
        self,
        *,
        file_name: str = "flow_integrals",
        schedule: Schedule = EverySteps(1),
        initial: bool = True,
    ) -> None:
        super().__init__(file_name=file_name, schedule=schedule, initial=initial)

    def write(self, context: SamplingContext) -> None:
        """Append the integrals of the current state and log them."""
        solver = context.solver
        Logging.flow_diagnostics(solver)
        if solver.turbulence_model is not None:
            solver.turbulence_model.update_turbulence_statistics(solver.particles)
            Logging.les_diagnostics(solver)
        path = context.output_directory / f"{self.name}.csv"
        if not context.continuing_output:
            path.unlink(missing_ok=True)
        row = flow_integral_row(solver)
        append_csv_rows(path, tuple(row), [tuple(row.values())])
