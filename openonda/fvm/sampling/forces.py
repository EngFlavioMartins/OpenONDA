"""Force and wall-diagnostic samplers for the FVM solver.

``ForceSampler`` integrates pressure and viscous loads on boundary patches and
appends one row per patch to a force-history CSV.  ``YPlusSampler`` computes
wall y+ statistics without a force-sampling dependency.

The numerical force and y+ routines live in
:mod:`openonda.fvm.fields.diagnostics`; these samplers only call them and
serialise the results, so offline post-processing and live runs share the exact
same mathematics.

Examples
--------
>>> from openonda.fvm.config import RunSchedule
>>> forces = ForceSampler(
...     patch_names=["cube"],
...     reference_velocity=1.0, reference_area=1.0, reference_length=1.0,
...     moment_centre=[0, 0, 0],
...     file_name="forces_history",
...     schedule=RunSchedule(every_n_steps=10),
... )
>>> y_plus = YPlusSampler(patch_names=["cube"])
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..fields import diagnostics
from .base import Sampler, _register_sampler, append_csv_rows

if TYPE_CHECKING:
    from ..config.scheduling import RunSchedule
    from ..config.types import FVMSetup
    from ..core.solver import FVMSolver

FORCES_HEADER = [
    "time",
    "step",
    "accepted_time_step_size",
    "patch",
    "pressure_force_x",
    "pressure_force_y",
    "pressure_force_z",
    "viscous_force_x",
    "viscous_force_y",
    "viscous_force_z",
    "total_force_x",
    "total_force_y",
    "total_force_z",
    "moment_x",
    "moment_y",
    "moment_z",
    "drag_coefficient",
    "lift_coefficient",
    "side_force_coefficient",
    "pitching_moment_coefficient",
]


def _context_setup(context: FVMSolver) -> FVMSetup:
    """Return the validated setup used by live and offline sampling contexts."""
    return getattr(context, "_resolved_setup", context.setup)


def _context_transport(context: FVMSolver) -> tuple[float, float | np.ndarray]:
    """Return ``(density, effective_kinematic_viscosity)`` for the sampled state."""
    setup = _context_setup(context)
    density = setup.transport.density
    effective_kinematic_viscosity = getattr(
        context,
        "_kinematic_viscosity",
        setup.transport.kinematic_viscosity,
    )
    eddy_viscosity = getattr(context, "eddy_viscosity", None)
    if eddy_viscosity is not None:
        effective_kinematic_viscosity = (
            effective_kinematic_viscosity + eddy_viscosity[: context.mesh_data["n_cells"]]
        )
    return density, effective_kinematic_viscosity


class ForceSampler(Sampler):
    """Compute per-patch surface forces/coefficients and append them to CSV.

    Wraps :func:`openonda.fvm.fields.diagnostics.compute_surface_forces`,
    which it calls directly — no force mathematics lives in the sampler.
    ``sample()`` is MPI-collective in partitioned runs (it merges patch-force
    fragments across ranks); ``write_csv()`` is the root-only file step.

    Examples
    --------
    >>> sampler = ForceSampler(
    ...     patch_names=["cube"],
    ...     reference_velocity=1.0,
    ...     reference_area=1.0,
    ...     reference_length=1.0,
    ...     moment_centre=[0, 0, 0],
    ...     file_name="forces_history",
    ... )
    >>> sampler.name
    'forces_history'
    """

    def __init__(
        self,
        patch_names: Sequence[str] | None = None,
        reference_velocity: float = 1.0,
        reference_area: float = 1.0,
        reference_length: float = 1.0,
        moment_centre: Sequence[float] = (0.0, 0.0, 0.0),
        file_name: str = "forces_history",
        schedule: RunSchedule | None = None,
    ) -> None:
        """Create a surface-force sampler.

        Parameters
        ----------
        patch_names : sequence[str] or None, default=None
            Boundary names to integrate; ``None`` selects all wall patches.
        reference_velocity : float, default=1.0
            Reference speed in m/s for force coefficients.
        reference_area : float, default=1.0
            Reference area in m².
        reference_length : float, default=1.0
            Reference length in m for the pitching-moment coefficient.
        moment_centre : sequence[float], default=(0, 0, 0)
            Moment origin, shape ``(3,)``, in m. On a moving mesh it is the
            start position of a point that moves with the mesh.
        file_name : str, default="forces_history"
            Output CSV stem.
        schedule : RunSchedule or None, default=None
            Accepted-step/time cadence.
        """
        super().__init__(file_name=file_name, schedule=schedule)
        self.patch_names = patch_names
        self.reference_velocity = reference_velocity
        self.reference_area = reference_area
        self.reference_length = reference_length
        self.moment_centre = list(moment_centre)

    def config_dict(self) -> dict:
        """Return JSON-safe constructor settings for persistence."""
        spec = super().config_dict()
        spec.update(
            {
                "patch_names": self.patch_names,
                "reference_velocity": self.reference_velocity,
                "reference_area": self.reference_area,
                "reference_length": self.reference_length,
                "moment_centre": list(self.moment_centre),
            }
        )
        return spec

    def sample(self, context: FVMSolver) -> dict[str, dict[str, object]]:
        """Compute per-patch forces/coefficients for the current state.

        Uses the LES-aware effective viscosity (molecular plus ``eddy_viscosity``) that the
        momentum equation relies on, and merges patch fragments across ranks
        when partitioned — call on every rank, not just root.

        Returns
        -------
        dict[str, dict[str, object]]
            Patch force vectors in N, moments in N·m, face counts, and
            coefficient scalars.
        """
        density, effective_kinematic_viscosity = _context_transport(context)
        dynamic_viscosity = effective_kinematic_viscosity * density
        moment_centre = np.asarray(self.moment_centre, dtype=np.float64)
        mesh_motion = getattr(context, "mesh_motion", None)
        if mesh_motion is not None:
            moment_centre = mesh_motion.current_position(moment_centre)

        forces = diagnostics.compute_surface_forces(
            context.velocity,
            context.kinematic_pressure,
            dynamic_viscosity,
            density,
            context.mesh_data,
            context.geo_data,
            context.boundaries,
            patch_names=self.patch_names,
            reference_velocity=self.reference_velocity,
            reference_area=self.reference_area,
            reference_length=self.reference_length,
            moment_centre=moment_centre,
            gradient=context._velocity_gradient(),
        )
        if context.parallel.is_partitioned:
            forces = diagnostics.merge_partition_forces(context.parallel.comm.allgather(forces))
        return forces

    def write_csv(self, context: FVMSolver, samples_dir: str, forces: dict) -> None:
        """Append one row per patch for an already-sampled force result.

        Parameters
        ----------
        context : FVMSolver
            Accepted solver state supplying time, step, and dt.
        samples_dir : str
            Solver-owned output directory.
        forces : dict
            Result returned by :meth:`sample`.
        """
        rows = []
        for pname, fdata in forces.items():
            pressure_force = fdata.get("pressure_force", [0, 0, 0])
            viscous_force = fdata.get("viscous_force", [0, 0, 0])
            total_force = fdata.get("total_force", [0, 0, 0])
            moment = fdata.get("moment", [0, 0, 0])
            C = fdata.get("coeffs", {})
            rows.append(
                [
                    context.time,
                    context.step,
                    context._accepted_time_step_size,
                    pname,
                    pressure_force[0],
                    pressure_force[1],
                    pressure_force[2],
                    viscous_force[0],
                    viscous_force[1],
                    viscous_force[2],
                    total_force[0],
                    total_force[1],
                    total_force[2],
                    moment[0],
                    moment[1],
                    moment[2],
                    C.get("drag_coefficient"),
                    C.get("lift_coefficient"),
                    C.get("side_force_coefficient"),
                    C.get("pitching_moment_coefficient"),
                ]
            )
        append_csv_rows(f"{samples_dir}/{self.file_name}.csv", FORCES_HEADER, rows)


class YPlusSampler(Sampler):
    """Compute wall y+ statistics for selected patches.

    A diagnostic sampler with no dependency on ``ForceSampler`` or on any
    force cadence.  Statistics are logged; when ``file_name`` is set it also
    appends one row per patch to ``<name>.csv``.

    Examples
    --------
    >>> sampler = YPlusSampler(patch_names=["cube"])
    >>> sampler.patch_names
    ['cube']
    """

    def __init__(
        self,
        patch_names: Sequence[str] | None = None,
        file_name: str | None = None,
        schedule: RunSchedule | None = None,
    ) -> None:
        """Create a wall-unit diagnostic sampler.

        Parameters
        ----------
        patch_names : sequence[str] or None, default=None
            Patches to inspect; ``None`` selects all wall patches.
        file_name : str or None, default=None
            Optional CSV stem. ``None`` only logs the statistics.
        schedule : RunSchedule or None, default=None
            Accepted-step/time cadence.
        """
        super().__init__(file_name=file_name, schedule=schedule)
        self.patch_names = patch_names

    def config_dict(self) -> dict:
        """Return JSON-safe constructor settings for persistence."""
        spec = super().config_dict()
        spec["patch_names"] = self.patch_names
        return spec

    def sample(self, context: FVMSolver) -> dict[str, dict[str, float]]:
        """Compute y+ statistics for the current state (collective).

        Returns
        -------
        dict[str, dict[str, float]]
            Per-patch minimum, maximum, average, and face count.
        """
        stats = diagnostics.compute_y_plus(
            context.velocity,
            getattr(
                context,
                "_kinematic_viscosity",
                _context_setup(context).transport.kinematic_viscosity,
            ),
            context.mesh_data,
            context.geo_data,
            context.boundaries,
            patch_names=self.patch_names,
        )
        if context.parallel.is_partitioned:
            stats = diagnostics.merge_partition_yplus(context.parallel.comm.allgather(stats))
        return stats

    def write_csv(
        self,
        context: FVMSolver,
        samples_dir: str,
        stats: dict[str, dict[str, float]],
    ) -> None:
        """Append one y+ row per patch to the history CSV."""
        if self.file_name is None:
            return
        header = ["time", "step", "patch", "min", "max", "avg"]
        rows = [
            [context.time, context.step, name, v["min"], v["max"], v["avg"]]
            for name, v in sorted(stats.items())
        ]
        append_csv_rows(f"{samples_dir}/{self.file_name}.csv", header, rows)


_register_sampler(ForceSampler)
_register_sampler(YPlusSampler)
