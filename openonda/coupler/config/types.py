"""Coupling settings and coupled backups of an FVM--VPM simulation."""

from dataclasses import asdict, dataclass, field, fields

import numpy as np

from openonda.core.schedules import EverySteps, EveryTime, FinalOnly, Schedule
from openonda.core.simulation.forcing import VelocityRamp

__all__ = ["Backup", "CouplerConfig"]

# Interface depths in particle spacings, recommended by the cylinder error study
# (docs/reports/2026-10-23-coupling-error-sources.md): the inset of the transfer surface from
# the outer FVM boundary, and the depths below the transfer surface where the blending weight
# starts to rise (end of the zero-weight layer) and reaches one (start of the replacement region).
TRANSFER_INSET_SPACINGS = 8.75
BLEND_START_SPACINGS = 2.0
BLEND_END_SPACINGS = 6.0


@dataclass(frozen=True, slots=True)
class Backup:
    """Coupled restart backups ``solution/coupled/coupled_{step:06d}/``.

    One coupled backup holds the FVM fields, the particles and the boundary
    data of one coupling step, and the FVM and particle frames of that time.

    Parameters
    ----------
    schedule : EverySteps, EveryTime, FinalOnly or None, default=None
        Coupling steps with a backup. ``EverySteps(n)`` counts accepted shared
        steps; ``EveryTime`` selects fixed physical-time intervals. ``None``
        writes no periodic backups.
    write_at_end : bool, default=True
        Write a backup of the last coupling step when the run ends.
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
    """Vorticity transfer, coupling iterations and backups of the FVM--VPM coupling.

    The FVM and VPM cases set viscosity, density, geometry, mesh and particle
    resolution and the time steps; the coupled solver checks that they agree.
    Particle strengths are integrated vorticity, in m³/s. The vorticity transfer
    requires GBD diffusion with an absolute vorticity threshold.
    """

    freestream_velocity: tuple[float, float, float] = (1.0, 0.0, 0.0)
    """Cartesian free-stream velocity in m/s; must match the VPM."""
    freestream: VelocityRamp | None = None
    """Optional free-stream velocity history; part of the restart configuration."""
    transfer_region_bounds: tuple[float, float, float, float, float, float] | None = None
    """Transfer-region bounds (xmin, xmax, ymin, ymax, zmin, zmax), in m.

    ``None`` insets the outer FVM boundary by ``transfer_inset_spacings`` on every face (not on
    slip planes).
    """
    transfer_inset_spacings: float = TRANSFER_INSET_SPACINGS
    """Transfer inset ``L_inset``, from the outer FVM boundary to the transfer surface, in particle
    spacings.

    Used when ``transfer_region_bounds`` is ``None``; the default is the inset recommended by the
    cylinder error study.
    """
    blend_start_depth: float | None = None
    """Depth ``δ_0`` below the transfer surface where the blending weight starts to rise, in m.

    The zero-weight layer ``0 < d ≤ δ_0`` keeps its particle vorticity. ``None`` uses
    ``BLEND_START_SPACINGS`` particle spacings, and zero when ``blend_end_depth`` is zero.
    """
    blend_end_depth: float | None = None
    """Depth ``δ_1`` below the transfer surface where the blending weight reaches one, in m.

    Deeper lies the replacement region, where FVM vorticity replaces the particle vorticity.
    ``None`` uses ``BLEND_END_SPACINGS`` particle spacings; zero disables blending, so the FVM
    vorticity replaces the particle vorticity everywhere inside the transfer surface.
    """
    transfer_removal_threshold: float = 0.05
    """Removal threshold ``τ_transfer`` of the replacement region, in 1/s; at least the GBD
    threshold ``τ_VPM``."""
    max_strength_ratio: float = 1.8
    """Strength-ratio limit ``κ_max`` of the strength correction; at least one.

    It sets the correction gain ``γ = min(κ_max − 1, 1)`` and bounds each corrected lattice strength
    by ``κ_max`` times the peak target strength. Larger existing strengths are kept when the FVM
    and particle vorticity already agree.
    """
    transfer_diagnostic_interval_steps: int = 1
    """Accepted transfers between full diagnostics; positive."""
    max_moment_correction_fraction: float = 0.08
    """Largest moment correction, as a fraction of the transferred strength L1 norm, in (0, 1]."""
    outer_boundary_patch: str = "numericalBoundary"
    """Mesh patch of the outer FVM boundary, which receives the normal velocity and the normal
    gradient of the tangential velocity."""
    max_coupling_iterations: int = 6
    """Largest number of coupling iterations at a fixed VPM predictor; above one, FVM output must
    fall on coupling times.

    With the tolerances below the cylinder study converged in 1.8 coupling iterations on average;
    a single iteration changed the drag by 3%.
    """
    normal_velocity_tolerance: float = 1.0e-5
    """Tolerance of the area-weighted RMS normal-velocity residual, in m/s."""
    normal_gradient_tolerance: float = 1.0e-5
    """Tolerance of the area-weighted RMS normal-gradient residual, in 1/s."""
    boundary_mismatch_warning: float | None = 0.2
    """Tangential FVM–particle velocity mismatch on the outer FVM boundary, in units of the
    reference speed, above which the coupler logs a warning with the face location; ``None``
    disables it. Inflow faces take the particle velocity, so mismatch arises on outflow faces."""
    boundary_mismatch_limit: float | None = 1.0
    """Mismatch, in units of the reference speed, above which the run stops with an error
    naming the face. A mismatch of the order of the flow speed is not physical; it signals
    an unstable boundary closure or corrupted boundary data. ``None`` disables the check."""
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
        for name in ("max_coupling_iterations", "transfer_diagnostic_interval_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("normal_velocity_tolerance", "normal_gradient_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("blend_start_depth", "blend_end_depth", "transfer_removal_threshold"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.blend_end_depth == 0.0 and self.blend_start_depth not in (None, 0.0):
            raise ValueError("blend_start_depth must be zero when blend_end_depth is zero")
        if (
            self.blend_start_depth is not None
            and self.blend_end_depth is not None
            and self.blend_end_depth > 0.0
            and self.blend_start_depth >= self.blend_end_depth
        ):
            raise ValueError("blend_start_depth must be smaller than blend_end_depth")
        if not np.isfinite(self.transfer_inset_spacings) or self.transfer_inset_spacings < 0.0:
            raise ValueError("transfer_inset_spacings must be finite and non-negative")
        if not np.isfinite(self.max_strength_ratio) or self.max_strength_ratio < 1.0:
            raise ValueError("max_strength_ratio must be at least one")
        if not np.isfinite(self.max_moment_correction_fraction) or not (
            0.0 < self.max_moment_correction_fraction <= 1.0
        ):
            raise ValueError("max_moment_correction_fraction must lie in (0, 1]")
        for name in ("boundary_mismatch_warning", "boundary_mismatch_limit"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive, or None")
        if (
            self.boundary_mismatch_warning is not None
            and self.boundary_mismatch_limit is not None
            and self.boundary_mismatch_warning > self.boundary_mismatch_limit
        ):
            raise ValueError("boundary_mismatch_warning must not exceed boundary_mismatch_limit")
        if not isinstance(self.backup, Backup):
            raise TypeError("backup must be a coupler Backup")

    @property
    def freestream_velocity_vector(self) -> np.ndarray:
        """Return the Cartesian free-stream vector in m/s."""
        return np.asarray(self.freestream_velocity, dtype=np.float64)

    def transfer_region(
        self, fvm_domain_bounds: tuple | np.ndarray, particle_spacing: float, slip_planes: bool
    ) -> np.ndarray:
        """Transfer-region bounds in m, inset from the outer FVM boundary when none are given.

        Slip planes bound a planar flow in z, so the default inset leaves them in place.
        """
        if self.transfer_region_bounds is not None:
            return np.asarray(self.transfer_region_bounds, dtype=np.float64)
        inset = self.transfer_inset_spacings * float(particle_spacing)
        offsets = np.array([inset, -inset] * 3)
        if slip_planes:
            offsets[4:6] = 0.0
        return np.asarray(fvm_domain_bounds, dtype=np.float64) + offsets

    def blend_depths(self, particle_spacing: float) -> tuple[float, float]:
        """Blending depths ``(δ_0, δ_1)`` in m, from the particle spacing when unset."""
        spacing = float(particle_spacing)
        end = BLEND_END_SPACINGS * spacing if self.blend_end_depth is None else self.blend_end_depth
        if self.blend_start_depth is not None:
            start = self.blend_start_depth
        else:
            start = 0.0 if end == 0.0 else BLEND_START_SPACINGS * spacing
        if end > 0.0 and start >= end:
            raise ValueError(
                f"blend_start_depth {start:g} m must be smaller than blend_end_depth {end:g} m"
            )
        return float(start), float(end)

    def validate_transfer_region(
        self, fvm_domain_bounds: tuple | np.ndarray, particle_spacing: float, slip_planes: bool
    ) -> None:
        """Require the transfer region to lie inside the FVM domain."""
        outer = np.asarray(fvm_domain_bounds, dtype=np.float64)
        inner = self.transfer_region(outer, particle_spacing, slip_planes)
        if np.any(inner[::2] < outer[::2]) or np.any(inner[1::2] > outer[1::2]):
            raise ValueError("transfer_region_bounds must be contained within the FVM domain")
        if np.any(inner[1::2] <= inner[::2]):
            raise ValueError("the transfer inset leaves no transfer region inside the FVM domain")

    def to_dict(self) -> dict[str, object]:
        """Return the settings that change the solution, for restart validation.

        The backup schedule, the interval of the full transfer diagnostics and the
        boundary-mismatch guard only select outputs or stop a run, and are left out.
        """
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        del values["backup"], values["transfer_diagnostic_interval_steps"]
        del values["boundary_mismatch_warning"], values["boundary_mismatch_limit"]
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
