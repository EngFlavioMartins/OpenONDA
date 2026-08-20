"""Viscous-diffusion configuration for the VPM solver."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

_THRESHOLD_MODES = {
    "budget",
    "relative_max",
    "absolute",
    "relative_local",
}


@dataclass(frozen=True)
class ViscousConfig:
    """Configure molecular viscous diffusion in the VPM solver.

    Supported schemes are Core Spreading (``CS``), Random Walk (``RWM``),
    Diffused Vortex Hydrodynamics (``DVH``), Grid-Based Diffusion (``GBD``),
    and ``NONE``.
    """

    scheme: Literal["CS", "RWM", "NONE", "DVH", "GBD"] = "CS"

    rwm_noise_amplitude: float = 1.0
    """Multiplier applied to Random-Walk Brownian displacement."""

    core_radius_ratio: float = 2.5
    """Regenerated-particle core radius divided by regeneration-grid spacing."""

    dvh_grid_spacing: float | None = None
    """DVH regeneration-grid spacing [m]."""

    dvh_domain_padding: float = 3.0
    """DVH domain padding in grid cells."""

    dvh_threshold: float = 0.01
    """DVH pruning threshold interpreted by ``dvh_threshold_mode``."""

    dvh_threshold_mode: str = "budget"
    """DVH pruning mode."""

    gbd_grid_spacing: float | None = None
    """GBD regeneration-grid spacing [m]."""

    gbd_domain_padding: float = 3.0
    """GBD domain padding in grid cells."""

    gbd_threshold: float = 0.01
    """GBD pruning threshold interpreted by ``gbd_threshold_mode``."""

    gbd_threshold_mode: str = "budget"
    """GBD pruning mode."""

    regeneration_threshold_window: int = 3
    """Half-width of the local regeneration-threshold window [grid cells]."""

    regeneration_cap_absolute_fraction: float = 0.99
    """Minimum fraction of surviving absolute vortex strength protected by a cap."""

    gbd_max_nodes: int | None = None
    """Optional cap on surviving GBD grid nodes."""

    dvh_support_radius_ratio: int = 4
    """DVH compact-support radius ``R_d / h``; allowed values are 3, 4, and 5."""

    dvh_max_nodes: int | None = None
    """Optional cap on surviving DVH grid nodes."""

    kinematic_viscosity: float | None = None
    """Molecular kinematic viscosity ``nu`` [m²/s]."""

    particle_spacing: float | None = None
    """Representative inter-particle spacing ``h`` [m]."""

    def __post_init__(self) -> None:
        scheme = self.scheme.upper()
        if scheme not in {"CS", "RWM", "NONE", "DVH", "GBD"}:
            raise ValueError(f"Invalid viscous scheme: {self.scheme!r}")
        object.__setattr__(self, "scheme", scheme)

        if self.rwm_noise_amplitude < 0.0 or not math.isfinite(self.rwm_noise_amplitude):
            raise ValueError("rwm_noise_amplitude must be finite and non-negative")
        if self.core_radius_ratio <= 0.0 or not math.isfinite(self.core_radius_ratio):
            raise ValueError("core_radius_ratio must be finite and positive")

        for name in ("particle_spacing", "dvh_grid_spacing", "gbd_grid_spacing"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive when set")

        if self.kinematic_viscosity is not None and (
            not math.isfinite(self.kinematic_viscosity) or self.kinematic_viscosity < 0.0
        ):
            raise ValueError("kinematic_viscosity must be finite and non-negative when set")

        if self.dvh_support_radius_ratio not in {3, 4, 5}:
            raise ValueError(
                f"dvh_support_radius_ratio must be 3, 4, or 5 (got {self.dvh_support_radius_ratio})"
            )

        if self.dvh_threshold_mode not in _THRESHOLD_MODES:
            raise ValueError(f"dvh_threshold_mode must be one of {sorted(_THRESHOLD_MODES)}")
        if self.gbd_threshold_mode not in _THRESHOLD_MODES:
            raise ValueError(f"gbd_threshold_mode must be one of {sorted(_THRESHOLD_MODES)}")
        if self.regeneration_threshold_window < 0:
            raise ValueError("regeneration_threshold_window must be non-negative")
        if not 0.0 < self.regeneration_cap_absolute_fraction <= 1.0:
            raise ValueError("regeneration_cap_absolute_fraction must be in (0, 1]")
        if self.gbd_max_nodes is not None and self.gbd_max_nodes < 1:
            raise ValueError("gbd_max_nodes must be positive when set")
        if self.dvh_max_nodes is not None and self.dvh_max_nodes < 1:
            raise ValueError("dvh_max_nodes must be positive when set")

    def rwm_accuracy_time_step_size(self) -> float:
        """Return the configured RWM accuracy bound ``h² / (4 nu)`` [s]."""
        if self.particle_spacing is None:
            raise ValueError("particle_spacing must be set for the RWM accuracy check")
        if self.kinematic_viscosity is None or self.kinematic_viscosity <= 0.0:
            raise ValueError("kinematic_viscosity must be positive for the RWM accuracy check")
        return self.particle_spacing**2 / (4.0 * self.kinematic_viscosity)

    def dvh_required_time_step_size(self) -> float:
        """Return the DVH diffusion increment ``beta R_d² / (4 nu)`` [s]."""
        from .constants import _DVH_BETA

        if self.dvh_grid_spacing is None:
            raise ValueError("dvh_grid_spacing must be set to a positive value")
        if self.kinematic_viscosity is None or self.kinematic_viscosity <= 0.0:
            raise ValueError("kinematic_viscosity must be positive for DVH")

        support_radius = self.dvh_support_radius_ratio * self.dvh_grid_spacing
        return _DVH_BETA * support_radius * support_radius / (4.0 * self.kinematic_viscosity)

    def gbd_max_time_step_size(self) -> float:
        """Return the explicit GBD stability bound ``h² / (6 nu)`` [s]."""
        if self.gbd_grid_spacing is None:
            raise ValueError("gbd_grid_spacing must be set to a positive value")
        if self.kinematic_viscosity is None or self.kinematic_viscosity <= 0.0:
            raise ValueError("kinematic_viscosity must be positive for GBD")
        return self.gbd_grid_spacing**2 / (6.0 * self.kinematic_viscosity)

    @staticmethod
    def cs(
        kinematic_viscosity: float | None = None,
        particle_spacing: float | None = None,
    ) -> ViscousConfig:
        """Return Core-Spreading viscous configuration."""
        return ViscousConfig(
            scheme="CS",
            kinematic_viscosity=kinematic_viscosity,
            particle_spacing=particle_spacing,
        )

    @staticmethod
    def rwm(
        kinematic_viscosity: float | None = None,
        particle_spacing: float | None = None,
    ) -> ViscousConfig:
        """Return Random-Walk viscous configuration."""
        return ViscousConfig(
            scheme="RWM",
            kinematic_viscosity=kinematic_viscosity,
            particle_spacing=particle_spacing,
        )

    @staticmethod
    def inviscid() -> ViscousConfig:
        """Return configuration with molecular diffusion disabled."""
        return ViscousConfig(scheme="NONE")

    @staticmethod
    def dvh(
        particle_spacing: float | None = None,
        padding: float = 20.0,
        threshold: float = 1e-5,
        threshold_mode: str = "budget",
        threshold_window: int = 3,
        dvh_support_radius_ratio: int = 4,
        kinematic_viscosity: float | None = None,
        max_nodes: int | None = None,
        cap_absolute_fraction: float = 0.99,
        core_radius_ratio: float = 2.5,
    ) -> ViscousConfig:
        """Return Diffused Vortex Hydrodynamics configuration."""
        return ViscousConfig(
            scheme="DVH",
            particle_spacing=particle_spacing,
            dvh_grid_spacing=particle_spacing,
            dvh_domain_padding=padding,
            dvh_threshold=threshold,
            dvh_threshold_mode=threshold_mode,
            regeneration_threshold_window=threshold_window,
            dvh_support_radius_ratio=dvh_support_radius_ratio,
            kinematic_viscosity=kinematic_viscosity,
            dvh_max_nodes=max_nodes,
            regeneration_cap_absolute_fraction=cap_absolute_fraction,
            core_radius_ratio=core_radius_ratio,
        )

    @staticmethod
    def gbd(
        particle_spacing: float | None = None,
        padding: float = 20.0,
        threshold: float = 1e-5,
        threshold_mode: str = "budget",
        threshold_window: int = 3,
        kinematic_viscosity: float | None = None,
        max_nodes: int | None = None,
        cap_absolute_fraction: float = 0.99,
        core_radius_ratio: float = 2.5,
    ) -> ViscousConfig:
        """Return Grid-Based Diffusion configuration."""
        return ViscousConfig(
            scheme="GBD",
            particle_spacing=particle_spacing,
            gbd_grid_spacing=particle_spacing,
            gbd_domain_padding=padding,
            gbd_threshold=threshold,
            gbd_threshold_mode=threshold_mode,
            regeneration_threshold_window=threshold_window,
            kinematic_viscosity=kinematic_viscosity,
            gbd_max_nodes=max_nodes,
            regeneration_cap_absolute_fraction=cap_absolute_fraction,
            core_radius_ratio=core_radius_ratio,
        )
