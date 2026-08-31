"""Force and wall-diagnostic samplers for the FVM solver.

``ForceSampler`` integrates pressure and viscous loads on boundary patches and
appends one row per patch to a force-history CSV.  ``IBMForceSampler`` logs
per-body loads from the immersed-boundary forcing.  ``YPlusSampler`` computes
wall y+ statistics and exposes them through ``context.last_y_plus`` without a
force-sampling dependency.

The numerical force and y+ routines live in
:mod:`source.solvers.fvm.fields.diagnostics`; these samplers only call them and
serialise the results, so offline post-processing and live runs share the exact
same mathematics.

Examples
--------
>>> from source.solvers.fvm.config import RunSchedule
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
from typing import TYPE_CHECKING, Any

import numpy as np

from ..fields import diagnostics
from .base import Sampler, _register_sampler, append_csv_rows

if TYPE_CHECKING:
    from ..config.scheduling import RunSchedule
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


def _context_transport(context):
    """Return ``(density, effective_kinematic_viscosity)`` for the sampled state."""
    density = context.setup.transport.density
    effective_kinematic_viscosity = context.setup.transport.kinematic_viscosity
    eddy_viscosity = getattr(context, "eddy_viscosity", None)
    if eddy_viscosity is not None:
        effective_kinematic_viscosity = (
            effective_kinematic_viscosity + eddy_viscosity[: context.mesh_data["n_cells"]]
        )
    return density, effective_kinematic_viscosity


class ForceSampler(Sampler):
    """Compute per-patch surface forces/coefficients and append them to CSV.

    Wraps :func:`source.solvers.fvm.fields.diagnostics.compute_surface_forces`,
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

    sampler_kind = "ForceSampler"

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
        """Initialize the force sampler.

        Args:
            patch_names: Patch names to integrate loads over. ``None``
                integrates all wall patches.
            reference_velocity: Reference velocity for coefficient calculation.
            reference_area: Reference area for coefficient calculation.
            reference_length: Reference length for moment coefficients.
            moment_centre: Moment reference point [x, y, z].
            file_name: Base name for the output CSV.
            schedule: Optional :class:`~source.solvers.fvm.config.RunSchedule`.
        """
        super().__init__(file_name=file_name, schedule=schedule)
        self.patch_names = patch_names
        self.reference_velocity = reference_velocity
        self.reference_area = reference_area
        self.reference_length = reference_length
        self.moment_centre = list(moment_centre)

    def config_dict(self) -> dict:
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

    def sample(self, context: FVMSolver) -> dict[str, dict[str, Any]]:
        """Compute per-patch forces/coefficients for the current state.

        Uses the LES-aware effective viscosity (molecular plus ``eddy_viscosity``) that the
        momentum equation relies on, and merges patch fragments across ranks
        when partitioned — call on every rank, not just root.
        """
        density, effective_kinematic_viscosity = _context_transport(context)
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

    def write_csv(self, context: FVMSolver, samples_dir: str, forces: dict) -> None:
        """Append one row per patch for an already-sampled ``forces`` dict."""
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

    A diagnostic sampler: it has no dependency on ``ForceSampler`` or on any
    force cadence, so ``last_y_plus`` cannot go stale from an inappropriate
    force interval.  When ``file_name`` is set it also appends one row per
    patch to ``<name>.csv``; otherwise it only updates ``context.last_y_plus``.

    Examples
    --------
    >>> sampler = YPlusSampler(patch_names=["cube"])
    >>> sampler.patch_names
    ['cube']
    """

    sampler_kind = "YPlusSampler"

    def __init__(
        self,
        patch_names=None,
        file_name: str | None = None,
        schedule: RunSchedule | None = None,
    ) -> None:
        """Initialize the y+ sampler.

        Args:
            patch_names: Patch names to compute y+ over; ``None`` selects all
                wall patches.
            file_name: Optional base name for a y+ history CSV; ``None``
                disables file output.
            schedule: Optional :class:`~source.solvers.fvm.config.RunSchedule`.
        """
        super().__init__(file_name=file_name, schedule=schedule)
        self.patch_names = patch_names

    def config_dict(self) -> dict:
        spec = super().config_dict()
        spec["patch_names"] = self.patch_names
        return spec

    def sample(self, context) -> dict[str, dict[str, float]]:
        """Compute y+ statistics for the current state (collective)."""
        stats = diagnostics.compute_y_plus(
            context.velocity,
            context.setup.transport.kinematic_viscosity,
            context.mesh_data,
            context.geo_data,
            context.boundaries,
            patch_names=self.patch_names,
        )
        if context.parallel.is_partitioned:
            stats = diagnostics.merge_partition_yplus(context.parallel.comm.allgather(stats))
        return stats

    def write_csv(self, context, samples_dir: str, stats: dict) -> None:
        """Append one row per patch to ``<samples_dir>/<name>.csv``."""
        if self.file_name is None:
            return
        header = ["time", "step", "patch", "min", "max", "avg"]
        rows = [
            [context.time, context.step, name, v["min"], v["max"], v["avg"]]
            for name, v in sorted(stats.items())
        ]
        append_csv_rows(f"{samples_dir}/{self.file_name}.csv", header, rows)


class IBMForceSampler(Sampler):
    """Log per-body immersed-boundary forces and coefficients.

    Converts the solver's former hand-written IBM CSV path into a sampler:
    forces come from :meth:`IBMForcing.body_forces`, the no-slip ``slip_error``
    from :meth:`IBMForcing.slip_error`, and coefficients are normalised by the
    dynamic pressure built from this sampler's reference values.

    Examples
    --------
    >>> sampler = IBMForceSampler(reference_velocity=1.0, reference_area=1.0)
    >>> sampler.file_name
    'ibm_forces_history'
    """

    sampler_kind = "IBMForceSampler"

    def __init__(
        self,
        reference_velocity: float = 1.0,
        reference_area: float = 1.0,
        file_name: str = "ibm_forces_history",
        schedule: RunSchedule | None = None,
    ) -> None:
        """Initialize the IBM force sampler.

        Args:
            reference_velocity: Reference velocity for coefficient calculation.
            reference_area: Reference area for coefficient calculation.
            file_name: Base name for the output CSV.
            schedule: Optional :class:`~source.solvers.fvm.config.RunSchedule`.
        """
        super().__init__(file_name=file_name, schedule=schedule)
        self.reference_velocity = reference_velocity
        self.reference_area = reference_area

    def config_dict(self) -> dict:
        spec = super().config_dict()
        spec.update(
            {"reference_velocity": self.reference_velocity, "reference_area": self.reference_area}
        )
        return spec

    def sample(self, context: FVMSolver) -> dict[str, np.ndarray]:
        """Compute per-body forces and slip for the current state."""
        if context.parallel.is_partitioned:
            raise NotImplementedError(
                "IBM force sampling is not qualified for partitioned execution"
            )
        ibm = getattr(context, "ibm", None)
        if ibm is None:
            raise RuntimeError(
                "IBMForceSampler requires a solver with immersed bodies; "
                "call FVMSolver.set_immersed_bodies(...) first"
            )
        return {
            "forces": ibm.body_forces(density=context.setup.transport.density),
            "slip_error": ibm.slip_error(context.velocity),
        }

    def summary(
        self,
        context: FVMSolver,
        data: dict,
    ) -> dict[str, tuple[float, float]]:
        """Return per-body ``(Cd, Cl)`` pairs for logging."""
        density = context.setup.transport.density
        q = 0.5 * density * self.reference_velocity**2 * self.reference_area
        return {name: (float(F[0] / q), float(F[1] / q)) for name, F in data["forces"].items()}

    def write_csv(self, context: FVMSolver, samples_dir: str, data: dict) -> None:
        """Append one row per body to ``<samples_dir>/<name>.csv``."""
        density = context.setup.transport.density
        q = 0.5 * density * self.reference_velocity**2 * self.reference_area
        rows = []
        for name, F in data["forces"].items():
            rows.append(
                [
                    context.time,
                    context.step,
                    context._accepted_time_step_size,
                    name,
                    F[0],
                    F[1],
                    F[2],
                    F[0] / q,
                    F[1] / q,
                    data["slip_error"],
                ]
            )
        header = [
            "time",
            "step",
            "accepted_time_step_size",
            "body_id",
            "force_x",
            "force_y",
            "force_z",
            "drag_coefficient",
            "lift_coefficient",
            "slip_error",
        ]
        append_csv_rows(f"{samples_dir}/{self.file_name}.csv", header, rows)


_register_sampler(ForceSampler)
_register_sampler(YPlusSampler)
_register_sampler(IBMForceSampler)
