"""Configuration of the FVM--VPM coupling operations."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class CouplerSetup:
    """Numerical choices owned by the coupling algorithm.

    Fluid properties, time integration, mesh geometry, and wall definitions
    are read from the native FVM and VPM solvers during initialization.

    Parameters are grouped by the coupler subsystem that owns them: flow
    state, VPM discretization, the overlap (FVM/VPM authority) zone, the
    FVM -> VPM vorticity transfer, the VPM boundary-condition trace on the
    FVM, the pressure reference, and run-level operational settings.
    """

    # ---- FLOW STATE ----
    freestream_velocity: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    """Freestream velocity (u, v, w) in m/s; must be a finite three-component vector."""

    # ---- VPM DISCRETIZATION ----
    vpm_particle_spacing: float = 0.05
    """VPM particle spacing (m); sets the lattice spacing and particle core size."""
    vpm_core_radius_ratio: float = 1.0
    """Particle core radius sigma as a ratio of the particle spacing; must match
    the VPM regeneration radius ratio and be at least one."""

    # ---- OVERLAP ZONE (FVM/VPM authority profile) ----
    authority_ramp_width: float = 0.30
    """Width (m) of the C1 ramp over which FVM authority rises from zero to one
    inside the overlap zone; must exceed ``vpm_only_width``."""
    vpm_only_width: float = 0.15
    """Width (m) of the band just inside the overlap-zone faces where FVM
    authority is exactly zero; must be non-negative."""

    # ---- VORTICITY TRANSFER (FVM -> VPM) ----
    transfer_region_bounds: tuple[float, float, float, float, float, float] | None = None
    """(xmin, xmax, ymin, ymax, zmin, zmax) region over which FVM vorticity is
    transferred to VPM particles; must lie inside the FVM domain; None uses the
    full FVM domain."""
    transfer_diagnostic_interval_steps: int = 1
    """Coupling steps between transfer diagnostics; at least one."""

    # ---- VPM BOUNDARY-CONDITION TRACE ON THE FVM ----
    coupling_patch: str = "numericalBoundary"
    """Name of the FVM patch on which the VPM boundary condition is imposed."""
    boundary_condition_mode: Literal[
        "dirichlet",
        "characteristic",
        "directional_outflow",
        "pressure_gradient",
        "vorticity_mixed",
    ] = "dirichlet"
    """VPM boundary-condition mode: dirichlet, characteristic, directional_outflow,
    pressure_gradient, or vorticity_mixed."""
    bc_resync_after_transfer: bool = True
    """Re-synchronize the VPM boundary trace after each vorticity transfer."""

    # ---- PRESSURE REFERENCE ----
    pressure_anchor_to_freestream: bool = True
    """Anchor the mean upstream total-pressure reference to the freestream value."""

    # ---- RUN-LEVEL OPERATIONAL ----
    checkpoint_interval_steps: int = 1
    """Coupling steps between automatic checkpoints; non-negative (0 disables checkpoints)."""
    checkpoint_interval_time: float | None = None
    """Flow-time interval between automatic checkpoints; mutually exclusive with steps."""

    def __post_init__(self) -> None:
        freestream_velocity = np.asarray(self.freestream_velocity, dtype=np.float64)
        if freestream_velocity.shape != (3,) or not np.all(np.isfinite(freestream_velocity)):
            raise ValueError("freestream_velocity must be a finite three-component vector")

        if self.boundary_condition_mode not in {
            "dirichlet",
            "characteristic",
            "directional_outflow",
            "pressure_gradient",
            "vorticity_mixed",
        }:
            raise ValueError(
                "boundary_condition_mode must be 'dirichlet', 'characteristic', "
                "'directional_outflow', 'pressure_gradient', or 'vorticity_mixed'"
            )

        if self.transfer_region_bounds is not None:
            transfer_region_bounds = np.asarray(self.transfer_region_bounds, dtype=np.float64)
            if transfer_region_bounds.shape != (6,) or not np.all(
                np.isfinite(transfer_region_bounds)
            ):
                raise ValueError("transfer_region_bounds must contain six finite bounds")
            if np.any(transfer_region_bounds[1::2] <= transfer_region_bounds[::2]):
                raise ValueError(
                    "Each transfer_region_bounds upper bound must exceed its lower bound"
                )

        positive = {
            "vpm_particle_spacing": self.vpm_particle_spacing,
            "authority_ramp_width": self.authority_ramp_width,
            "vpm_core_radius_ratio": self.vpm_core_radius_ratio,
        }
        invalid = [
            name for name, value in positive.items() if not np.isfinite(value) or value <= 0.0
        ]
        if invalid:
            raise ValueError(f"Coupling values must be positive: {', '.join(invalid)}")
        if self.checkpoint_interval_steps < 0:
            raise ValueError("checkpoint_interval_steps must be non-negative")
        if self.checkpoint_interval_time is not None:
            if (
                not np.isfinite(self.checkpoint_interval_time)
                or self.checkpoint_interval_time <= 0.0
            ):
                raise ValueError("checkpoint_interval_time must be finite and positive")
            if self.checkpoint_interval_steps > 0:
                raise ValueError(
                    "Provide only one of checkpoint_interval_steps or checkpoint_interval_time"
                )
        if self.vpm_only_width < 0.0:
            raise ValueError("vpm_only_width must be non-negative")
        if self.transfer_diagnostic_interval_steps < 1:
            raise ValueError("transfer_diagnostic_interval must be at least one")
        if self.authority_ramp_width <= self.vpm_only_width:
            raise ValueError("authority_ramp_width must exceed vpm_only_width")
        if self.vpm_core_radius_ratio < 1.0:
            raise ValueError("vpm_core_radius_ratio must be at least one")

    @property
    def freestream_velocity_vector(self) -> np.ndarray:
        return np.asarray(self.freestream_velocity, dtype=np.float64)

    def validate_transfer_region_box(self, fvm_box) -> None:
        """Require the vorticity-transfer region to lie inside the FVM domain."""
        if self.transfer_region_bounds is None:
            return
        outer = np.asarray(fvm_box, dtype=np.float64)
        inner = np.asarray(self.transfer_region_bounds, dtype=np.float64)
        if np.any(inner[::2] < outer[::2]) or np.any(inner[1::2] > outer[1::2]):
            raise ValueError("transfer_region_bounds must be contained within the FVM domain")

    def to_dict(self) -> dict:
        transfer_region_bounds = None
        if self.transfer_region_bounds is not None:
            transfer_region_bounds = dict(
                zip(
                    ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
                    self.transfer_region_bounds,
                    strict=True,
                )
            )
        return {
            "coupler": {
                "freestream_velocity": self.freestream_velocity,
                "checkpoint_interval_steps": self.checkpoint_interval_steps,
                "checkpoint_interval_time": self.checkpoint_interval_time,
                "coupling_patch": self.coupling_patch,
                "boundary_condition_mode": self.boundary_condition_mode,
                "transfer_region_bounds": transfer_region_bounds,
                "vpm_particle_spacing": self.vpm_particle_spacing,
                "authority_ramp_width": self.authority_ramp_width,
                "vpm_only_width": self.vpm_only_width,
                "vpm_core_radius_ratio": self.vpm_core_radius_ratio,
                "transfer_diagnostic_interval_steps": self.transfer_diagnostic_interval_steps,
                "bc_resync_after_transfer": self.bc_resync_after_transfer,
                "pressure_anchor_to_freestream": self.pressure_anchor_to_freestream,
            }
        }
