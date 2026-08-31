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
    transfer_method: Literal[
        "buffered_m4_renewal",
        "common_lattice",
        "projected_renewal",
    ] = "common_lattice"
    """FVM-to-VPM state transfer. ``buffered_m4_renewal`` is the whole-belt
    M4' method proven by the historical 20-second GBD cube run.
    ``projected_renewal`` and ``common_lattice`` remain experimental paths."""
    transfer_region_bounds: tuple[float, float, float, float, float, float] | None = None
    """FVM-authoritative replacement region ``(xmin, xmax, ymin, ymax, zmin,
    zmax)``. It must lie inside the FVM domain; ``None`` uses the full domain."""
    eta_blend_width: float = 0.0
    """Width (m) of the C1 FVM-authority ramp measured inward from the transfer
    faces. The stable renewal leaves the face and its release buffer under VPM
    authority; zero selects a hard interior authority profile."""
    vpm_only_width: float = 0.0
    """Width (m) just inside the transfer faces where stable renewal keeps
    FVM authority exactly zero. It must be smaller than ``eta_blend_width``."""
    transfer_vorticity_cutoff: float = 0.05
    """Stable-renewal soft-prune threshold in vorticity units (1/s)."""
    transfer_boundary_prune_multiplier: float = 10.0
    """Multiplier on the stable-renewal prune threshold as FVM authority
    approaches zero at the transfer boundary."""
    transfer_amplification_cap: float = 1.8
    """Maximum gain used by the stable represented-state correction."""
    transfer_diagnostic_interval_steps: int = 1
    """Replacement steps between transfer diagnostics; at least one."""
    transfer_discretization_error_limit: float = 0.08
    """Maximum Gaussian-particle vorticity-divergence error admitted before transfer."""
    renewal_vorticity_error_limit: float = 5.0e-3
    """Maximum independent relative vorticity mismatch for projected renewal."""
    renewal_velocity_error_limit: float = 1.0e-3
    """Maximum relative normal-velocity mismatch at the ownership boundary."""
    renewal_gaussian_tail_cutoff: float = 1.0e-8
    """Relative Gaussian kernel weight omitted from the production sparse operator."""
    renewal_solver_tolerance: float = 1.0e-9
    """Relative LSMR tolerance for the sparse absolute-strength solve."""

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
    fvm_consistency_width: float = 0.0
    """Width (m) of the resolved-scale VPM-to-FVM consistency band measured
    inward from the outer FVM boundary. Zero disables the band. A positive
    value must fit entirely outside ``transfer_region_bounds``."""
    # ---- RUN-LEVEL OPERATIONAL ----
    backup_interval_steps: int = 1
    """Coupling steps between automatic backups; non-negative (0 disables backups)."""

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

        if self.transfer_method not in {
            "buffered_m4_renewal",
            "common_lattice",
            "projected_renewal",
        }:
            raise ValueError(
                "transfer_method must be 'buffered_m4_renewal', 'common_lattice', "
                "or 'projected_renewal'"
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

        if self.backup_interval_steps < 0:
            raise ValueError("backup_interval_steps must be non-negative")
        if not np.isfinite(self.fvm_consistency_width) or self.fvm_consistency_width < 0.0:
            raise ValueError("fvm_consistency_width must be finite and non-negative")
        if not np.isfinite(self.eta_blend_width) or self.eta_blend_width < 0.0:
            raise ValueError("eta_blend_width must be finite and non-negative")
        if not np.isfinite(self.vpm_only_width) or self.vpm_only_width < 0.0:
            raise ValueError("vpm_only_width must be finite and non-negative")
        if self.eta_blend_width == 0.0 and self.vpm_only_width != 0.0:
            raise ValueError("vpm_only_width requires a positive eta_blend_width")
        if self.eta_blend_width > 0.0 and self.vpm_only_width >= self.eta_blend_width:
            raise ValueError("vpm_only_width must be smaller than eta_blend_width")
        if not np.isfinite(self.transfer_vorticity_cutoff) or self.transfer_vorticity_cutoff < 0.0:
            raise ValueError("transfer_vorticity_cutoff must be finite and non-negative")
        if (
            not np.isfinite(self.transfer_boundary_prune_multiplier)
            or self.transfer_boundary_prune_multiplier < 1.0
        ):
            raise ValueError("transfer_boundary_prune_multiplier must be at least one")
        if (
            not np.isfinite(self.transfer_amplification_cap)
            or self.transfer_amplification_cap < 1.0
        ):
            raise ValueError("transfer_amplification_cap must be at least one")
        if self.transfer_diagnostic_interval_steps < 1:
            raise ValueError("transfer_diagnostic_interval must be at least one")
        if (
            not np.isfinite(self.transfer_discretization_error_limit)
            or not 0.0 < self.transfer_discretization_error_limit <= 1.0
        ):
            raise ValueError("transfer_discretization_error_limit must lie in (0, 1]")
        if (
            not np.isfinite(self.renewal_vorticity_error_limit)
            or self.renewal_vorticity_error_limit <= 0.0
        ):
            raise ValueError("renewal_vorticity_error_limit must be finite and positive")
        if (
            not np.isfinite(self.renewal_velocity_error_limit)
            or self.renewal_velocity_error_limit <= 0.0
        ):
            raise ValueError("renewal_velocity_error_limit must be finite and positive")
        if (
            not np.isfinite(self.renewal_gaussian_tail_cutoff)
            or not 0.0 < self.renewal_gaussian_tail_cutoff < 1.0
        ):
            raise ValueError("renewal_gaussian_tail_cutoff must lie between zero and one")
        if not np.isfinite(self.renewal_solver_tolerance) or self.renewal_solver_tolerance <= 0.0:
            raise ValueError("renewal_solver_tolerance must be finite and positive")
        if self.transfer_method == "projected_renewal" and self.eta_blend_width != 0.0:
            raise ValueError("projected_renewal requires eta_blend_width=0")

    @property
    def freestream_velocity_vector(self) -> np.ndarray:
        return np.asarray(self.freestream_velocity, dtype=np.float64)

    def validate_transfer_region_box(self, fvm_box) -> None:
        """Require the vorticity-transfer region to lie inside the FVM domain."""
        outer = np.asarray(fvm_box, dtype=np.float64)
        if self.transfer_method == "projected_renewal" and self.transfer_region_bounds is None:
            raise ValueError(
                "projected_renewal requires explicit transfer_region_bounds with room "
                "inside the FVM domain for its runtime GBD guard"
            )
        inner = (
            outer
            if self.transfer_region_bounds is None
            else np.asarray(self.transfer_region_bounds, dtype=np.float64)
        )
        if np.any(inner[::2] < outer[::2]) or np.any(inner[1::2] > outer[1::2]):
            raise ValueError("transfer_region_bounds must be contained within the FVM domain")
        if self.fvm_consistency_width > 0.0:
            if self.transfer_region_bounds is None:
                raise ValueError("fvm_consistency_width requires explicit transfer_region_bounds")
            margins = np.column_stack((inner[::2] - outer[::2], outer[1::2] - inner[1::2]))
            if np.any(margins + 1.0e-14 < self.fvm_consistency_width):
                raise ValueError(
                    "fvm_consistency_width must fit between every transfer-region face "
                    "and the outer FVM boundary"
                )

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
                "backup_interval_steps": self.backup_interval_steps,
                "coupling_patch": self.coupling_patch,
                "boundary_condition_mode": self.boundary_condition_mode,
                "fvm_consistency_width": self.fvm_consistency_width,
                "transfer_method": self.transfer_method,
                "transfer_region_bounds": transfer_region_bounds,
                "eta_blend_width": self.eta_blend_width,
                "vpm_only_width": self.vpm_only_width,
                "transfer_vorticity_cutoff": self.transfer_vorticity_cutoff,
                "transfer_boundary_prune_multiplier": (self.transfer_boundary_prune_multiplier),
                "transfer_amplification_cap": self.transfer_amplification_cap,
                "transfer_diagnostic_interval_steps": self.transfer_diagnostic_interval_steps,
                "transfer_discretization_error_limit": (self.transfer_discretization_error_limit),
                "renewal_vorticity_error_limit": self.renewal_vorticity_error_limit,
                "renewal_velocity_error_limit": self.renewal_velocity_error_limit,
                "renewal_gaussian_tail_cutoff": self.renewal_gaussian_tail_cutoff,
                "renewal_solver_tolerance": self.renewal_solver_tolerance,
            }
        }
