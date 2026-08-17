"""Configuration for the supported FVM–VPM coupling path."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class CouplerSetup:
    """Configuration shared by the native FVM and VPM solvers."""

    u_inf: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    nu: float | None = None
    rho: float | None = None
    dt: float | None = None
    t_end: float | None = None
    backup_period: int = 1
    log_period: int = 1

    fvm_box: tuple[float, float, float, float, float, float] | None = None
    handoff_box: tuple[float, float, float, float, float, float] | None = None
    patch_name: str = "numericalBoundary"
    vpm_bc_mode: Literal[
        "dirichlet", "characteristic", "directional_outflow", "pressure_gradient"
    ] = "dirichlet"
    wall_patch_name: str | None = "cube"
    grid_spacing: float | None = None
    initial_U: list[float] | None = None
    case_dir: str = "."
    surface: dict = field(default_factory=dict)

    h: float = 0.05
    buffer_thickness: float = 0.30
    dead_zone_h: float = 3.0
    prune_vorticity_min: float = 0.005
    boundary_prune_multiplier: float = 1.0
    handoff_max_particles: int | None = None
    overlap_radius_ratio: float = 1.0
    # Cap on the inverse-mollification gain. What the lattice cannot carry is
    # reported, not amplified into grid-scale noise.
    transfer_amplification_cap: float = 2.0
    # Re-evaluate the VPM BC from the corrected particles after the hand-off.
    # Otherwise the next interval starts from a stale prediction.
    resync_vpm_bc_after_handoff: bool = True
    # Datum the FVM pressure. Every coupling face is Neumann in pressure.
    anchor_pressure: bool = True

    def __post_init__(self) -> None:
        u_inf = np.asarray(self.u_inf, dtype=np.float64)
        if u_inf.shape != (3,) or not np.all(np.isfinite(u_inf)):
            raise ValueError("u_inf must be a finite three-component vector")

        if self.initial_U is not None:
            initial_u = np.asarray(self.initial_U, dtype=np.float64)
            if initial_u.shape != (3,) or not np.all(np.isfinite(initial_u)):
                raise ValueError("initial_U must be a finite three-component vector")

        if self.vpm_bc_mode not in {
            "dirichlet",
            "characteristic",
            "directional_outflow",
            "pressure_gradient",
        }:
            raise ValueError(
                "vpm_bc_mode must be 'dirichlet', 'characteristic', "
                "'directional_outflow', or 'pressure_gradient'"
            )

        if self.fvm_box is not None:
            box = np.asarray(self.fvm_box, dtype=np.float64)
            if box.shape != (6,) or not np.all(np.isfinite(box)):
                raise ValueError("fvm_box must contain six finite bounds")
            if np.any(box[1::2] <= box[::2]):
                raise ValueError("Each fvm_box upper bound must exceed its lower bound")
        if self.handoff_box is not None:
            handoff_box = np.asarray(self.handoff_box, dtype=np.float64)
            if handoff_box.shape != (6,) or not np.all(np.isfinite(handoff_box)):
                raise ValueError("handoff_box must contain six finite bounds")
            if np.any(handoff_box[1::2] <= handoff_box[::2]):
                raise ValueError("Each handoff_box upper bound must exceed its lower bound")
            if self.fvm_box is not None:
                self.validate_handoff_box()

        positive = {
            "nu": self.nu,
            "rho": self.rho,
            "dt": self.dt,
            "t_end": self.t_end,
            "grid_spacing": self.grid_spacing,
            "h": self.h,
            "buffer_thickness": self.buffer_thickness,
            "overlap_radius_ratio": self.overlap_radius_ratio,
            "transfer_amplification_cap": self.transfer_amplification_cap,
            "boundary_prune_multiplier": self.boundary_prune_multiplier,
        }
        invalid = [
            name
            for name, value in positive.items()
            if value is not None and (not np.isfinite(value) or value <= 0.0)
        ]
        if invalid:
            raise ValueError(f"Coupling values must be positive: {', '.join(invalid)}")
        if self.backup_period < 0 or self.log_period < 1:
            raise ValueError("backup_period must be non-negative and log_period positive")
        if self.dead_zone_h < 0.0 or self.prune_vorticity_min < 0.0:
            raise ValueError("dead_zone_h and prune_vorticity_min must be non-negative")
        if self.transfer_amplification_cap < 1.0:
            raise ValueError("transfer_amplification_cap must be at least one")
        if self.boundary_prune_multiplier < 1.0:
            raise ValueError("boundary_prune_multiplier must be at least one")
        if self.buffer_thickness <= self.dead_zone_h * self.h:
            raise ValueError("buffer_thickness must exceed dead_zone_h * h")
        if self.overlap_radius_ratio < 1.0:
            raise ValueError("overlap_radius_ratio must be at least one")
        if self.handoff_max_particles is not None and self.handoff_max_particles < 1:
            raise ValueError("handoff_max_particles must be positive")

    @property
    def U_inf(self) -> np.ndarray:
        return np.asarray(self.u_inf, dtype=np.float64)

    def validate_handoff_box(self) -> None:
        """Require the vorticity-transfer box to lie inside the FVM domain."""
        if self.handoff_box is None or self.fvm_box is None:
            return
        outer = np.asarray(self.fvm_box, dtype=np.float64)
        inner = np.asarray(self.handoff_box, dtype=np.float64)
        if np.any(inner[::2] < outer[::2]) or np.any(inner[1::2] > outer[1::2]):
            raise ValueError("handoff_box must be contained within fvm_box")

    def to_dict(self) -> dict:
        domain = None
        if self.fvm_box is not None:
            domain = dict(
                zip(
                    ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
                    self.fvm_box,
                    strict=True,
                )
            )
        handoff_domain = None
        if self.handoff_box is not None:
            handoff_domain = dict(
                zip(
                    ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
                    self.handoff_box,
                    strict=True,
                )
            )
        return {
            "physics": {
                "u_inf": self.u_inf,
                "nu": self.nu,
                "rho": self.rho,
                "dt": self.dt,
                "t_end": self.t_end,
                "backup_period": self.backup_period,
                "log_period": self.log_period,
            },
            "fvm_solver": {
                "patch_name": self.patch_name,
                "vpm_bc_mode": self.vpm_bc_mode,
                "wall_patch_name": self.wall_patch_name,
                "grid_spacing": self.grid_spacing,
                "initial_U": self.initial_U,
                "fvm_domain": domain,
                "surface": self.surface,
            },
            "vpm_solver": {
                "particle_spacing": self.h,
                "buffer_thickness": self.buffer_thickness,
                "dead_zone_h": self.dead_zone_h,
                "h": self.h,
            },
            "coupler": {
                "handoff_domain": handoff_domain,
                "prune_vorticity_min": self.prune_vorticity_min,
                "boundary_prune_multiplier": self.boundary_prune_multiplier,
                "handoff_max_particles": self.handoff_max_particles,
                "overlap_radius_ratio": self.overlap_radius_ratio,
                "transfer_amplification_cap": self.transfer_amplification_cap,
                "resync_vpm_bc_after_handoff": self.resync_vpm_bc_after_handoff,
                "anchor_pressure": self.anchor_pressure,
            },
        }
