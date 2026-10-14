"""Surface-force and wall-resolution samplers of the FVM solver.

:class:`ForceSampler` integrates the pressure and viscous forces on boundary
patches and appends one row per patch to ``samples/<name>.csv``;
:class:`YPlusSampler` computes the wall y+ statistics. The force and y+
computations are in :mod:`openonda.fvm.diagnostics.fields`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from openonda.core.file_output import append_csv_rows
from openonda.core.sampling import Sampler
from openonda.core.schedules import EverySteps, Schedule

from ..diagnostics import fields as diagnostics

if TYPE_CHECKING:
    from ..core.solver import FVMSolver

FORCE_SAMPLE_COLUMNS = (
    "time",
    "time_step_size",
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
)
Y_PLUS_SAMPLE_COLUMNS = ("time", "patch", "min", "max", "avg")


def _effective_viscosity(context: FVMSolver) -> tuple[float, float | np.ndarray]:
    """Return the density in kg/m³ and the effective kinematic viscosity in m²/s."""
    density = context.case.numerics.transport.density
    viscosity = context._kinematic_viscosity
    if context.eddy_viscosity is not None:
        viscosity = viscosity + context.eddy_viscosity[: context.mesh_data["n_cells"]]
    return density, viscosity


class ForceSampler(Sampler):
    """Compute per-patch surface forces/coefficients and append them to CSV.

    Wraps :func:`openonda.fvm.diagnostics.fields.compute_surface_forces`,
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
        *,
        file_name: str | None = "forces_history",
        schedule: Schedule = EverySteps(1),
        initial: bool = True,
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
            Moment origin, shape ``(3,)``, in m.
        file_name : str, default="forces_history"
            Output CSV stem.
        schedule : EverySteps, EveryTime or FinalOnly, default=EverySteps(1)
            Accepted states that are sampled.
        initial : bool, default=True
            Sample the initial state.
        """
        super().__init__(file_name=file_name, schedule=schedule, initial=initial)
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
        density, effective_kinematic_viscosity = _effective_viscosity(context)
        dynamic_viscosity = effective_kinematic_viscosity * density

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
            moment_centre=self.moment_centre,
            gradient=context._velocity_gradient(),
        )
        if context.parallel.is_partitioned:
            forces = diagnostics.merge_partition_forces(context.parallel.comm.allgather(forces))
        return forces

    def write_csv(self, context: FVMSolver, samples_dir: str | Path, forces: dict) -> None:
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
        append_csv_rows(Path(samples_dir) / f"{self.name}.csv", FORCE_SAMPLE_COLUMNS, rows)


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
        *,
        file_name: str | None = None,
        schedule: Schedule = EverySteps(1),
        initial: bool = True,
    ) -> None:
        """Create a wall-unit diagnostic sampler.

        Parameters
        ----------
        patch_names : sequence[str] or None, default=None
            Patches to inspect; ``None`` selects all wall patches.
        file_name : str or None, default=None
            Optional CSV stem. ``None`` only logs the statistics.
        schedule : EverySteps, EveryTime or FinalOnly, default=EverySteps(1)
            Accepted states that are sampled.
        initial : bool, default=True
            Sample the initial state.
        """
        super().__init__(file_name=file_name, schedule=schedule, initial=initial)
        self.write_file = file_name is not None
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
            context._kinematic_viscosity,
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
        samples_dir: str | Path,
        stats: dict[str, dict[str, float]],
    ) -> None:
        """Append one y+ row per patch to ``<samples_dir>/<name>.csv``."""
        if not self.write_file:
            return
        rows = [
            [context.time, name, values["min"], values["max"], values["avg"]]
            for name, values in sorted(stats.items())
        ]
        append_csv_rows(Path(samples_dir) / f"{self.name}.csv", Y_PLUS_SAMPLE_COLUMNS, rows)
