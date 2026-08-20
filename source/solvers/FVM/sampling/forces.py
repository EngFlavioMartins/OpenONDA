"""Force and wall-diagnostic samplers for the FVM solver.

``ForceSampler`` integrates pressure and viscous loads on boundary patches and
appends one row per patch to a force-history CSV.  ``IBMForceSampler`` logs
per-body loads from the immersed-boundary forcing.  ``YPlusSampler`` computes
wall y+ statistics and exposes them through ``context.last_yplus`` without a
force-sampling dependency.

The numerical force and y+ routines live in
:mod:`source.solvers.FVM.fields.diagnostics`; these samplers only call them and
serialise the results, so offline post-processing and live runs share the exact
same mathematics.

Examples
--------
>>> from source.solvers.FVM.sampling.base import SamplingSchedule
>>> forces = ForceSampler(
...     patch_names=["cube"],
...     ref_velocity=1.0, ref_area=1.0, ref_length=1.0,
...     moment_centre=[0, 0, 0],
...     file_name="forces_history",
...     schedule=SamplingSchedule(every_n_steps=10),
... )
>>> yplus = YPlusSampler(patch_names=["cube"])
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..fields import diagnostics
from .base import Sampler, _register_sampler, append_csv_rows

FORCES_HEADER = [
    "time",
    "step",
    "dt",
    "patch",
    "Fpx",
    "Fpy",
    "Fpz",
    "Fvx",
    "Fvy",
    "Fvz",
    "Ftx",
    "Fty",
    "Ftz",
    "Cd",
    "Cl",
    "Cz",
    "Cm",
]


def _context_transport(context):
    """Return ``(rho, nu_eff)`` for the sampled state."""
    rho = context.setup.transport.density
    nu_eff = context.setup.transport.nu
    nut = getattr(context, "nut", None)
    if nut is not None:
        nu_eff = nu_eff + nut[: context.mesh_data["n_elements"]]
    return rho, nu_eff


class ForceSampler(Sampler):
    """Compute per-patch surface forces/coefficients and append them to CSV.

    Wraps :func:`source.solvers.FVM.fields.diagnostics.compute_surface_forces`,
    which it calls directly — no force mathematics lives in the sampler.
    ``sample()`` is MPI-collective in partitioned runs (it merges patch-force
    fragments across ranks); ``write_csv()`` is the root-only file step.

    Examples
    --------
    >>> sampler = ForceSampler(
    ...     patch_names=["cube"],
    ...     ref_velocity=1.0,
    ...     ref_area=1.0,
    ...     ref_length=1.0,
    ...     moment_centre=[0, 0, 0],
    ...     file_name="forces_history",
    ... )
    >>> sampler.name
    'forces_history'
    """

    sampler_kind = "ForceSampler"

    def __init__(
        self,
        patch_names=None,
        ref_velocity: float = 1.0,
        ref_area: float = 1.0,
        ref_length: float = 1.0,
        moment_centre=(0.0, 0.0, 0.0),
        file_name: str = "forces_history",
        schedule=None,
    ):
        """Initialize the force sampler.

        Args:
            patch_names: Patch names to integrate loads over. ``None``
                integrates all wall patches.
            ref_velocity: Reference velocity for coefficient calculation.
            ref_area: Reference area for coefficient calculation.
            ref_length: Reference length for moment coefficients.
            moment_centre: Moment reference point [x, y, z].
            file_name: Base name for the output CSV.
            schedule: Optional :class:`~.base.SamplingSchedule`.
        """
        super().__init__(file_name=file_name, schedule=schedule)
        self.patch_names = patch_names
        self.ref_velocity = ref_velocity
        self.ref_area = ref_area
        self.ref_length = ref_length
        self.moment_centre = list(moment_centre)

    def config_dict(self) -> dict:
        spec = super().config_dict()
        spec.update(
            {
                "patch_names": self.patch_names,
                "ref_velocity": self.ref_velocity,
                "ref_area": self.ref_area,
                "ref_length": self.ref_length,
                "moment_centre": list(self.moment_centre),
            }
        )
        return spec

    def sample(self, context) -> dict[str, dict[str, Any]]:
        """Compute per-patch forces/coefficients for the current state.

        Uses the LES-aware effective viscosity (``nu + nut``) that the
        momentum equation relies on, and merges patch fragments across ranks
        when partitioned — call on every rank, not just root.
        """
        rho, nu_eff = _context_transport(context)
        mu = nu_eff * rho

        forces = diagnostics.compute_surface_forces(
            context.U,
            context.p,
            mu,
            rho,
            context.mesh_data,
            context.geo_data,
            context.boundaries,
            patch_names=self.patch_names,
            ref_U=self.ref_velocity,
            ref_area=self.ref_area,
            ref_length=self.ref_length,
            moment_centre=self.moment_centre,
            gradient=context._velocity_gradient(),
        )
        if context.parallel.is_partitioned:
            forces = diagnostics.merge_partition_forces(context.parallel.comm.allgather(forces))
        return forces

    def write_csv(self, context, samples_dir: str, forces: dict) -> None:
        """Append one row per patch for an already-sampled ``forces`` dict."""
        rows = []
        for pname, fdata in forces.items():
            Fp = fdata.get("Fp", [0, 0, 0])
            Fv = fdata.get("Fv", [0, 0, 0])
            Ft = fdata.get("Ftot", [0, 0, 0])
            C = fdata.get("coeffs", {})
            rows.append(
                [
                    context.time,
                    context.step,
                    context._current_time_step_size,
                    pname,
                    Fp[0],
                    Fp[1],
                    Fp[2],
                    Fv[0],
                    Fv[1],
                    Fv[2],
                    Ft[0],
                    Ft[1],
                    Ft[2],
                    C.get("Cd"),
                    C.get("Cl"),
                    C.get("Cz"),
                    C.get("Cm"),
                ]
            )
        append_csv_rows(f"{samples_dir}/{self.file_name}.csv", FORCES_HEADER, rows)


class YPlusSampler(Sampler):
    """Compute wall y+ statistics for selected patches.

    A diagnostic sampler: it has no dependency on ``ForceSampler`` or on any
    force cadence, so ``last_yplus`` cannot go stale from an inappropriate
    force interval.  When ``file_name`` is set it also appends one row per
    patch to ``<name>.csv``; otherwise it only updates ``context.last_yplus``.

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
        schedule=None,
    ):
        """Initialize the y+ sampler.

        Args:
            patch_names: Patch names to compute y+ over; ``None`` selects all
                wall patches.
            file_name: Optional base name for a y+ history CSV; ``None``
                disables file output.
            schedule: Optional :class:`~.base.SamplingSchedule`.
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
            context.U,
            context.setup.transport.nu,
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
    forces come from :meth:`IBMForcing.body_forces`, the no-slip slip error
    from :meth:`IBMForcing.slip_error`, and coefficients are normalised by the
    dynamic pressure built from this sampler's reference values.

    Examples
    --------
    >>> sampler = IBMForceSampler(ref_velocity=1.0, ref_area=1.0)
    >>> sampler.file_name
    'ibm_forces_history'
    """

    sampler_kind = "IBMForceSampler"

    def __init__(
        self,
        ref_velocity: float = 1.0,
        ref_area: float = 1.0,
        file_name: str = "ibm_forces_history",
        schedule=None,
    ):
        """Initialize the IBM force sampler.

        Args:
            ref_velocity: Reference velocity for coefficient calculation.
            ref_area: Reference area for coefficient calculation.
            file_name: Base name for the output CSV.
            schedule: Optional :class:`~.base.SamplingSchedule`.
        """
        super().__init__(file_name=file_name, schedule=schedule)
        self.ref_velocity = ref_velocity
        self.ref_area = ref_area

    def config_dict(self) -> dict:
        spec = super().config_dict()
        spec.update({"ref_velocity": self.ref_velocity, "ref_area": self.ref_area})
        return spec

    def sample(self, context) -> dict[str, np.ndarray]:
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
            "forces": ibm.body_forces(rho=context.setup.transport.density),
            "slip": ibm.slip_error(context.U),
        }

    def summary(self, context, data: dict) -> dict[str, tuple[float, float]]:
        """Return per-body ``(Cd, Cl)`` pairs for logging."""
        rho = context.setup.transport.density
        q = 0.5 * rho * self.ref_velocity**2 * self.ref_area
        return {name: (float(F[0] / q), float(F[1] / q)) for name, F in data["forces"].items()}

    def write_csv(self, context, samples_dir: str, data: dict) -> None:
        """Append one row per body to ``<samples_dir>/<name>.csv``."""
        rho = context.setup.transport.density
        q = 0.5 * rho * self.ref_velocity**2 * self.ref_area
        rows = []
        for name, F in data["forces"].items():
            rows.append(
                [
                    context.time,
                    context.step,
                    context._current_time_step_size,
                    name,
                    F[0],
                    F[1],
                    F[2],
                    F[0] / q,
                    F[1] / q,
                    data["slip"],
                ]
            )
        header = ["time", "step", "dt", "body", "Fx", "Fy", "Fz", "Cd", "Cl", "slip"]
        append_csv_rows(f"{samples_dir}/{self.file_name}.csv", header, rows)


_register_sampler(ForceSampler)
_register_sampler(YPlusSampler)
_register_sampler(IBMForceSampler)
