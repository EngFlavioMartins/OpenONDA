"""Coupling settings and coupled backups of an FVM--VPM simulation."""

from dataclasses import asdict, dataclass, field, fields

import numpy as np

from openonda.core.schedules import EverySteps, EveryTime, FinalOnly, Schedule
from openonda.core.simulation.forcing import VelocityRamp

__all__ = ["Backup", "CouplerConfig"]

# Interface depths in particle spacings, from the interface sensitivity study of
# tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow: the inset of the transfer surface from
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
    """Cartesian free-stream velocity in m/s; must equal ``VPMCase.numerics.freestream_velocity``,
    whose default is (0, 0, 0)."""
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

    Used when ``transfer_region_bounds`` is ``None``; the default comes from the interface
    sensitivity study of the cylinder case.
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
    """Removal threshold ``τ_transfer`` of the replacement region, in 1/s. The strength threshold
    ``τ_transfer h³`` must be at least the absolute GBD threshold of the VPM, in m³/s."""
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
    circulation_budget_limit: float | None = 0.02
    """Largest difference between the total strength emitted in the replacement region and the
    FVM total strength there, as a fraction of the FVM strength L1 norm; above it the transfer
    stops before it changes the particles. ``None`` disables the check."""
    wall_circulation_ratio_bounds: tuple[float, float] | None = (0.9, 1.1)
    """Bounds of the near-wall circulation ratio, emitted over FVM total strength within three
    particle spacings of a wall, checked where the net near-wall FVM circulation exceeds 5 % of its
    L1 norm (a lifting body); outside them the transfer stops. ``None`` disables the check."""
    particle_net_strength_warning: float | None = 1.0e-3
    """Net vortex strength of the whole particle set, as a fraction of its L1 norm, above which each
    transfer logs a warning. A healthy three-dimensional run stays near 1e-4. ``None`` disables it."""
    particle_net_strength_limit: float | None = None
    """The same fraction above which the run stops; ``None`` (default) only warns."""
    outer_boundary_patch: str = "numericalBoundary"
    """Mesh patch of the outer FVM boundary, which receives the normal velocity and the normal
    gradient of the tangential velocity."""
    max_coupling_iterations: int = 6
    """Largest number of coupling iterations at a fixed VPM predictor; above one, FVM output must
    fall on coupling times.

    The interface sensitivity study of the cylinder case compares a single iteration with
    iterations converged to the tolerances below.
    """
    stop_on_unconverged_coupling: bool = False
    """Stop the run when the coupling iterations reach ``max_coupling_iterations`` without meeting
    the tolerances; otherwise the interval is accepted with the last boundary data the FVM used,
    and a warning is logged."""
    normal_velocity_tolerance: float = 1.0e-5
    """Tolerance of the area-weighted RMS normal-velocity residual, relative to the free-stream
    speed U."""
    normal_gradient_tolerance: float = 1.0e-5
    """Tolerance of the area-weighted RMS normal-gradient residual, relative to U/h with h the
    particle spacing."""
    boundary_data_extrapolation: str = "linear"
    """First estimate of the end-of-interval boundary data: ``"linear"`` extrapolates the
    corrections of the last two intervals, ``"constant"`` reuses the last one."""
    boundary_mismatch_warning: float | None = 0.2
    """Tangential FVM–particle velocity mismatch on the outer FVM boundary, in units of the
    reference speed, above which the coupler logs a warning with the face location; ``None``
    disables it. The mixed condition prescribes the normal velocity on every face, so the mismatch
    is the tangential velocity that the condition leaves to the FVM."""
    boundary_mismatch_limit: float | None = 1.0
    """Mismatch, in units of the reference speed, above which the run stops with an error
    naming the face. A mismatch of the order of the flow speed is not physical; it signals
    an unstable boundary condition or corrupted boundary data. ``None`` disables the check."""
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
        if self.boundary_data_extrapolation not in ("linear", "constant"):
            raise ValueError("boundary_data_extrapolation must be 'linear' or 'constant'")
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
        for name in (
            "boundary_mismatch_warning",
            "boundary_mismatch_limit",
            "circulation_budget_limit",
            "particle_net_strength_warning",
            "particle_net_strength_limit",
        ):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive, or None")
        if (
            self.boundary_mismatch_warning is not None
            and self.boundary_mismatch_limit is not None
            and self.boundary_mismatch_warning > self.boundary_mismatch_limit
        ):
            raise ValueError("boundary_mismatch_warning must not exceed boundary_mismatch_limit")
        if self.wall_circulation_ratio_bounds is not None:
            bounds = tuple(float(v) for v in self.wall_circulation_ratio_bounds)
            if (
                len(bounds) != 2
                or not np.all(np.isfinite(bounds))
                or not 0.0 < bounds[0] < 1.0 < bounds[1]
            ):
                raise ValueError("wall_circulation_ratio_bounds must hold 0 < lower < 1 < upper")
            object.__setattr__(self, "wall_circulation_ratio_bounds", bounds)
        if not isinstance(self.backup, Backup):
            raise TypeError("backup must be a coupler Backup")

    @property
    def freestream_velocity_vector(self) -> np.ndarray:
        """Return the Cartesian free-stream vector in m/s."""
        return np.asarray(self.freestream_velocity, dtype=np.float64)

    @property
    def maximum_freestream_speed(self) -> float:
        """Return the largest free-stream speed of the run in m/s.

        A velocity ramp blends its two end velocities monotonically, so the largest
        speed is at one of them.
        """
        speeds = [float(np.linalg.norm(self.freestream_velocity_vector))]
        if self.freestream is not None:
            speeds += [
                float(np.linalg.norm(self.freestream.initial)),
                float(np.linalg.norm(self.freestream.final)),
            ]
        return max(speeds)

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
        warnings and stops only select outputs or stop a run, and are left out.
        """
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        for name in (
            "backup",
            "transfer_diagnostic_interval_steps",
            "boundary_mismatch_warning",
            "boundary_mismatch_limit",
            "circulation_budget_limit",
            "wall_circulation_ratio_bounds",
            "particle_net_strength_warning",
            "particle_net_strength_limit",
            "stop_on_unconverged_coupling",
        ):
            del values[name]
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
