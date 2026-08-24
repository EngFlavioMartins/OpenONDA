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

    # ---- VORTICITY TRANSFER (FVM -> VPM) ----
    transfer_region_bounds: tuple[float, float, float, float, float, float] | None = None
    """FVM-authoritative replacement region ``(xmin, xmax, ymin, ymax, zmin,
    zmax)``. It must lie inside the FVM domain; ``None`` uses the full domain."""
    eta_blend_width: float = 0.0
    """Width (m) of the C1 state-blending ramp inside the replacement-region
    faces. Zero disables eta blending and performs hard delete/reinjection."""
    transfer_diagnostic_interval_steps: int = 1
    """Replacement steps between transfer diagnostics; at least one."""

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
    # ---- RUN-LEVEL OPERATIONAL ----
    checkpoint_interval_steps: int = 1
    """Coupling steps between automatic checkpoints; non-negative (0 disables checkpoints)."""

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

        if self.checkpoint_interval_steps < 0:
            raise ValueError("checkpoint_interval_steps must be non-negative")
        if not np.isfinite(self.eta_blend_width) or self.eta_blend_width < 0.0:
            raise ValueError("eta_blend_width must be finite and non-negative")
        if self.transfer_diagnostic_interval_steps < 1:
            raise ValueError("transfer_diagnostic_interval must be at least one")

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
                "coupling_patch": self.coupling_patch,
                "boundary_condition_mode": self.boundary_condition_mode,
                "transfer_region_bounds": transfer_region_bounds,
                "eta_blend_width": self.eta_blend_width,
                "transfer_diagnostic_interval_steps": self.transfer_diagnostic_interval_steps,
            }
        }
