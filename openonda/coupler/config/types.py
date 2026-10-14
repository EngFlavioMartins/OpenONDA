"""Coupling settings and coupled backups of an FVM--VPM simulation."""

from dataclasses import asdict, dataclass, field, fields

import numpy as np

from openonda.core.schedules import EverySteps, EveryTime, FinalOnly, Schedule
from openonda.core.simulation.forcing import VelocityRamp

__all__ = ["Backup", "CouplerConfig"]


@dataclass(frozen=True, slots=True)
class Backup:
    """Coupled restart backups ``solution/coupled/coupled_{step:06d}/``.

    One coupled backup holds the FVM fields, the particles and the interface
    trace of one exchange, and the FVM and particle frames of that time.

    Parameters
    ----------
    schedule : EverySteps, EveryTime, FinalOnly or None, default=None
        Exchanges with a backup. ``EverySteps(n)`` counts exchanges; an
        ``EveryTime`` interval and start time must be whole numbers of
        exchange time steps. ``None`` writes no periodic backups.
    write_at_end : bool, default=True
        Write a backup of the last exchange when the run ends.
    """

    schedule: Schedule | None = None
    write_at_end: bool = True

    def __post_init__(self) -> None:
        if self.schedule is not None and not isinstance(
            self.schedule, EverySteps | EveryTime | FinalOnly
        ):
            raise TypeError("Backup.schedule must be EverySteps, EveryTime, FinalOnly or None")
        if not isinstance(self.write_at_end, bool):
            raise TypeError("Backup.write_at_end must be a bool")


@dataclass(frozen=True, slots=True)
class CouplerConfig:
    """Overlap transfer, interface iteration and backups of the FVM--VPM exchange.

    The FVM and VPM cases own viscosity, density, geometry, mesh and particle
    resolution and the time steps; the coupled solver checks that they agree.
    Particle strengths are integrated vorticity, in m³/s. Renewal requires
    GBD diffusion with an absolute vorticity threshold.
    """

    freestream_velocity: tuple[float, float, float] = (1.0, 0.0, 0.0)
    """Cartesian background velocity in m/s; must match the VPM."""
    freestream: VelocityRamp | None = None
    """Optional background velocity history; part of the restart configuration."""
    transfer_region_bounds: tuple[float, float, float, float, float, float] | None = None
    """FVM-derived bounds (xmin, xmax, ymin, ymax, zmin, zmax), in m."""
    eta_blend_width: float = 0.0
    """Width of the smooth blending ramp inside the transfer faces, in m."""
    vpm_only_width: float = 0.0
    """VPM-owned band inside transfer faces, in m; smaller than the ramp."""
    transfer_vorticity_cutoff: float = 0.05
    """Interior soft-pruning threshold in 1/s, at least the VPM GBD floor."""
    transfer_amplification_cap: float = 1.8
    """Correction gain and coefficient bound relative to the target peak; at least one.

    Existing larger coefficients are retained when the fields already agree.
    """
    transfer_diagnostic_interval_steps: int = 1
    """Accepted transfers between expensive diagnostics; positive."""
    transfer_discretization_error_limit: float = 0.08
    """Maximum relative closure-correction fraction, in (0, 1]."""
    coupling_patch: str = "numericalBoundary"
    """Outer FVM patch receiving normal velocity and tangential gradient."""
    interface_iterations: int = 1
    """Maximum fixed-predictor sweeps; FVM output must fall on exchanges when above one."""
    interface_normal_tolerance: float = 1.0e-6
    """Area-weighted RMS normal-velocity residual tolerance, in m/s."""
    interface_gradient_tolerance: float = 1.0e-6
    """Area-weighted RMS tangential-gradient residual tolerance, in 1/s."""
    backup: Backup = field(default_factory=Backup)
    """Coupled restart backups."""

    def __post_init__(self) -> None:
        velocity = np.asarray(self.freestream_velocity, dtype=np.float64)
        if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
            raise ValueError("freestream_velocity must be a finite three-component vector")
        object.__setattr__(self, "freestream_velocity", tuple(float(v) for v in velocity))
        if self.freestream is not None and not isinstance(self.freestream, VelocityRamp):
            raise TypeError("freestream must be a VelocityRamp or None")
        if self.transfer_region_bounds is not None:
            bounds = np.asarray(self.transfer_region_bounds, dtype=np.float64)
            if bounds.shape != (6,) or not np.all(np.isfinite(bounds)):
                raise ValueError("transfer_region_bounds must contain six finite bounds")
            if np.any(bounds[1::2] <= bounds[::2]):
                raise ValueError(
                    "Each transfer_region_bounds upper bound must exceed its lower bound"
                )
            object.__setattr__(self, "transfer_region_bounds", tuple(float(v) for v in bounds))
        for name in ("interface_iterations", "transfer_diagnostic_interval_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("interface_normal_tolerance", "interface_gradient_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("Interface tolerances must be finite and positive")
        for name in ("eta_blend_width", "vpm_only_width", "transfer_vorticity_cutoff"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.eta_blend_width == 0.0 and self.vpm_only_width != 0.0:
            raise ValueError("vpm_only_width requires a positive eta_blend_width")
        if self.eta_blend_width > 0.0 and self.vpm_only_width >= self.eta_blend_width:
            raise ValueError("vpm_only_width must be smaller than eta_blend_width")
        if (
            not np.isfinite(self.transfer_amplification_cap)
            or self.transfer_amplification_cap < 1.0
        ):
            raise ValueError("transfer_amplification_cap must be at least one")
        if not np.isfinite(self.transfer_discretization_error_limit) or not (
            0.0 < self.transfer_discretization_error_limit <= 1.0
        ):
            raise ValueError("transfer_discretization_error_limit must lie in (0, 1]")
        if not isinstance(self.backup, Backup):
            raise TypeError("backup must be a coupler Backup")

    @property
    def freestream_velocity_vector(self) -> np.ndarray:
        """Return the Cartesian freestream vector in m/s."""
        return np.asarray(self.freestream_velocity, dtype=np.float64)

    def validate_transfer_region_box(self, fvm_box: tuple | np.ndarray) -> None:
        """Require the transfer region to lie inside the FVM domain."""
        outer = np.asarray(fvm_box, dtype=np.float64)
        inner = (
            outer
            if self.transfer_region_bounds is None
            else np.asarray(self.transfer_region_bounds, dtype=np.float64)
        )
        if np.any(inner[::2] < outer[::2]) or np.any(inner[1::2] > outer[1::2]):
            raise ValueError("transfer_region_bounds must be contained within the FVM domain")

    def to_dict(self) -> dict[str, object]:
        """Return the settings that change the solution, for restart validation.

        The backup schedule only selects saved states and is left out.
        """
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        del values["backup"]
        values["freestream_velocity"] = list(self.freestream_velocity)
        values["freestream"] = None if self.freestream is None else asdict(self.freestream)
        if self.transfer_region_bounds is not None:
            values["transfer_region_bounds"] = dict(
                zip(
                    ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
                    self.transfer_region_bounds,
                    strict=True,
                )
            )
        return {"coupler": values}
