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
    overlap_zone_ramp_width: float = 0.30
    """Width (m) of the C1 ramp over which FVM authority rises from zero to one
    inside the overlap zone; also scales the blending relaxation. Must exceed
    ``overlap_zone_dead_zone_width``."""
    overlap_zone_dead_zone_width: float = 0.15
    """Width (m) of the band just inside the overlap-zone faces where FVM
    authority is exactly zero; must be non-negative."""

    # ---- VORTICITY TRANSFER (FVM -> VPM) ----
    transfer_region_box: tuple[float, float, float, float, float, float] | None = None
    """(xmin, xmax, ymin, ymax, zmin, zmax) region over which FVM vorticity is
    transferred to VPM particles; must lie inside the FVM domain; None uses the
    full FVM domain."""
    transfer_prune_vorticity_min: float = 0.005
    """Minimum |vorticity| (1/s) kept after pruning; must be non-negative."""
    transfer_boundary_prune_multiplier: float = 1.0
    """Smooth multiplier on the prune threshold where FVM authority vanishes at
    the transfer-region boundary; must be at least one."""
    transfer_max_particles: int | None = None
    """Post-transfer VPM particle population cap; None is unlimited."""
    transfer_amplification_cap: float = 2.0
    """Maximum gain of the bounded local circulation correction; at least one."""
    transfer_diagnostic_interval_steps: int = 1
    """Coupling steps between transfer diagnostics; at least one."""

    # ---- VPM BOUNDARY-CONDITION TRACE ON THE FVM ----
    bc_patch_name: str = "numericalBoundary"
    """Name of the FVM patch on which the VPM boundary condition is imposed."""
    vpm_bc_mode: Literal[
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

    def __post_init__(self) -> None:
        freestream_velocity = np.asarray(self.freestream_velocity, dtype=np.float64)
        if freestream_velocity.shape != (3,) or not np.all(np.isfinite(freestream_velocity)):
            raise ValueError("freestream_velocity must be a finite three-component vector")

        if self.vpm_bc_mode not in {
            "dirichlet",
            "characteristic",
            "directional_outflow",
            "pressure_gradient",
            "vorticity_mixed",
        }:
            raise ValueError(
                "vpm_bc_mode must be 'dirichlet', 'characteristic', "
                "'directional_outflow', 'pressure_gradient', or 'vorticity_mixed'"
            )

        if self.transfer_region_box is not None:
            transfer_region_box = np.asarray(self.transfer_region_box, dtype=np.float64)
            if transfer_region_box.shape != (6,) or not np.all(np.isfinite(transfer_region_box)):
                raise ValueError("transfer_region_box must contain six finite bounds")
            if np.any(transfer_region_box[1::2] <= transfer_region_box[::2]):
                raise ValueError("Each transfer_region_box upper bound must exceed its lower bound")

        positive = {
            "vpm_particle_spacing": self.vpm_particle_spacing,
            "overlap_zone_ramp_width": self.overlap_zone_ramp_width,
            "vpm_core_radius_ratio": self.vpm_core_radius_ratio,
            "transfer_amplification_cap": self.transfer_amplification_cap,
            "transfer_boundary_prune_multiplier": self.transfer_boundary_prune_multiplier,
        }
        invalid = [
            name for name, value in positive.items() if not np.isfinite(value) or value <= 0.0
        ]
        if invalid:
            raise ValueError(f"Coupling values must be positive: {', '.join(invalid)}")
        if self.checkpoint_interval_steps < 0:
            raise ValueError("checkpoint_interval_steps must be non-negative")
        if self.overlap_zone_dead_zone_width < 0.0 or self.transfer_prune_vorticity_min < 0.0:
            raise ValueError(
                "overlap_zone_dead_zone_width and transfer_prune_vorticity_min must be non-negative"
            )
        if self.transfer_amplification_cap < 1.0:
            raise ValueError("transfer_amplification_cap must be at least one")
        if self.transfer_boundary_prune_multiplier < 1.0:
            raise ValueError("transfer_boundary_prune_multiplier must be at least one")
        if self.transfer_diagnostic_interval_steps < 1:
            raise ValueError("transfer_diagnostic_interval must be at least one")
        if self.overlap_zone_ramp_width <= self.overlap_zone_dead_zone_width:
            raise ValueError("overlap_zone_ramp_width must exceed overlap_zone_dead_zone_width")
        if self.vpm_core_radius_ratio < 1.0:
            raise ValueError("vpm_core_radius_ratio must be at least one")
        if self.transfer_max_particles is not None and self.transfer_max_particles < 1:
            raise ValueError("transfer_max_particles must be positive")

    @property
    def freestream_velocity_vector(self) -> np.ndarray:
        return np.asarray(self.freestream_velocity, dtype=np.float64)

    def validate_transfer_region_box(self, fvm_box) -> None:
        """Require the vorticity-transfer region to lie inside the FVM domain."""
        if self.transfer_region_box is None:
            return
        outer = np.asarray(fvm_box, dtype=np.float64)
        inner = np.asarray(self.transfer_region_box, dtype=np.float64)
        if np.any(inner[::2] < outer[::2]) or np.any(inner[1::2] > outer[1::2]):
            raise ValueError("transfer_region_box must be contained within the FVM domain")

    def to_dict(self) -> dict:
        transfer_region_box = None
        if self.transfer_region_box is not None:
            transfer_region_box = dict(
                zip(
                    ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
                    self.transfer_region_box,
                    strict=True,
                )
            )
        return {
            "coupler": {
                "freestream_velocity": self.freestream_velocity,
                "checkpoint_interval_steps": self.checkpoint_interval_steps,
                "bc_patch_name": self.bc_patch_name,
                "vpm_bc_mode": self.vpm_bc_mode,
                "transfer_region_box": transfer_region_box,
                "vpm_particle_spacing": self.vpm_particle_spacing,
                "overlap_zone_ramp_width": self.overlap_zone_ramp_width,
                "overlap_zone_dead_zone_width": self.overlap_zone_dead_zone_width,
                "transfer_prune_vorticity_min": self.transfer_prune_vorticity_min,
                "transfer_boundary_prune_multiplier": self.transfer_boundary_prune_multiplier,
                "transfer_max_particles": self.transfer_max_particles,
                "vpm_core_radius_ratio": self.vpm_core_radius_ratio,
                "transfer_amplification_cap": self.transfer_amplification_cap,
                "transfer_diagnostic_interval_steps": self.transfer_diagnostic_interval_steps,
                "bc_resync_after_transfer": self.bc_resync_after_transfer,
                "pressure_anchor_to_freestream": self.pressure_anchor_to_freestream,
            }
        }
