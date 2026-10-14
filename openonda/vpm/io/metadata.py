"""Run metadata of the VPM solver, ``solution/vpm_metadata.json``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from openonda.core.metadata import metadata_value, write_run_metadata
from openonda.core.status import RunStatus

if TYPE_CHECKING:
    from ..core.solver import VPMSolver

__all__ = ["write_vpm_metadata"]


def _configuration(solver: VPMSolver) -> dict[str, Any]:
    """Return the case configuration with the loaded vortex-lattice geometry."""
    configuration = metadata_value(solver.case)
    vlm = solver.vlm_solver
    if vlm is not None:
        from ..boundary_elements.vlm.geometry.surface_io import surface_to_dict
        from ..boundary_elements.vlm.solver.restart import (
            restart_configuration_hash,
            restart_physics_hash,
        )

        record = configuration["numerics"]["vlm"]
        record["configuration_hash"] = restart_configuration_hash(vlm)
        record["physics_hash"] = restart_physics_hash(vlm)
        # The geometry the solver loaded, not its source file, which may change later.
        for surface, (geometry, _) in zip(record["surfaces"], vlm.surfaces.values(), strict=True):
            surface["geometry"] = surface_to_dict(geometry)
    return configuration


def write_vpm_metadata(solver: VPMSolver, path: Path, *, status: RunStatus) -> Path:
    """Write the configuration, run status and accepted state of a VPM run.

    Sections after the shared keys: ``state`` (initial and current step and
    time, target step, particle counts), ``runtime`` (requested and used
    compute device) and, after a restart with changed settings, ``restart``.
    """
    particle_count = solver._restart_particle_count
    if particle_count is None:
        particle_count = int(solver.particles.n_particles_total)
    sections: dict[str, object] = {
        "state": {
            "initial_step": int(solver._run_initial_step),
            "initial_time": float(solver._run_initial_time),
            "step": int(solver.step),
            "time": float(solver.time),
            "target_step": int(solver.case.n_steps),
            "n_initial_particles": int(solver._n_initial_particles),
            "n_particles": int(particle_count),
        },
        "runtime": {
            "configured_compute_device": solver.numerics.compute_device,
            "requested_compute_device": solver.case.run.runtime_compute_device,
            "compute_device": solver.compute_device,
        },
    }
    if solver._restart_details is not None:
        sections["restart"] = solver._restart_details
    return write_run_metadata(
        path,
        solver="vpm",
        case_name=solver.case.name,
        status=status,
        configuration=_configuration(solver),
        sections=sections,
    )
