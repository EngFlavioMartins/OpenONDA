"""Declarative, immutable configuration for the vortex-lattice solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ForceConfig:
    """VLM aerodynamic-force evaluation policy."""

    method: Literal["KUTTA_JOUKOWSKI"] = "KUTTA_JOUKOWSKI"
    kj_smoothing: bool = False

    @staticmethod
    def kutta_joukowski(*, smoothing: bool = False) -> ForceConfig:
        return ForceConfig(kj_smoothing=smoothing)


@dataclass(frozen=True)
class VLMMeshSetup:
    """Global panel-distribution policy for all declared surfaces."""

    spacing: Literal["uniform", "geometric"] = "uniform"
    ratio: float = 1.0
    region: Literal["start", "end", "both"] = "both"

    def __post_init__(self) -> None:
        if self.ratio <= 0:
            raise ValueError("VLM mesh ratio must be positive")
        if self.spacing == "uniform" and self.ratio != 1.0:
            raise ValueError("A non-unit VLM mesh ratio requires geometric spacing")

    @staticmethod
    def geometric(
        ratio: float = 3.0,
        *,
        region: Literal["start", "end", "both"] = "both",
    ) -> VLMMeshSetup:
        return VLMMeshSetup(spacing="geometric", ratio=ratio, region=region)


@dataclass(frozen=True)
class VLMSurfaceSetup:
    """One lifting surface and all of its placement/coupling metadata."""

    surface: Any
    name: str | None = None
    kinematics: Any | None = None
    translation: tuple[float, float, float] | None = None
    rotation_degrees: tuple[float, float, float] | None = None
    rotation_centre: tuple[float, float, float] | None = None
    group_id: int = 0
    sample_forces: bool | None = None

    def __post_init__(self) -> None:
        for field_name in ("translation", "rotation_degrees", "rotation_centre"):
            value = getattr(self, field_name)
            if value is not None:
                if len(value) != 3:
                    raise ValueError(f"{field_name} must contain three coordinates")
                object.__setattr__(self, field_name, tuple(float(item) for item in value))
        if self.group_id < 0:
            raise ValueError("VLM surface group_id must be non-negative")


@dataclass(frozen=True)
class VLMSetup:
    """Complete VLM solver definition.

    ``max_n_panels`` is optional. When omitted, capacity is derived exactly from
    the declared surfaces, so normal cases need no allocation tuning.
    """

    surfaces: tuple[VLMSurfaceSetup, ...]
    mesh: VLMMeshSetup = field(default_factory=VLMMeshSetup)
    max_n_panels: int | None = None
    dtype: Literal["f32", "f64"] = "f32"
    linear_solver: Literal["SCIPY", "BICGSTAB_GPU", "CG_GPU"] | None = None
    circulation_relaxation: float = 1.0
    kinematic_viscosity: float = 1.0
    density: float = 1.0
    sigma_factor: float = 2.5
    freestream_velocity: tuple[float, float, float] | None = None
    logging_interval_steps: int = 1
    force: ForceConfig = field(default_factory=ForceConfig.kutta_joukowski)
    sample_surface_forces: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "surfaces", tuple(self.surfaces))
        if not self.surfaces:
            raise ValueError("VLMSetup requires at least one surface")
        if self.max_n_panels is not None and self.max_n_panels < 1:
            raise ValueError("VLM max_n_panels must be positive when provided")
        if not 0 < self.circulation_relaxation <= 1:
            raise ValueError("circulation_relaxation must be in (0, 1]")
        if self.kinematic_viscosity < 0:
            raise ValueError("VLM kinematic_viscosity must be non-negative")
        if self.density <= 0:
            raise ValueError("VLM density must be positive")
        if self.sigma_factor <= 0:
            raise ValueError("VLM sigma_factor must be positive")
        if self.logging_interval_steps < 1:
            raise ValueError("VLM logging_interval_steps must be positive")
        if self.freestream_velocity is not None:
            if len(self.freestream_velocity) != 3:
                raise ValueError("freestream_velocity must contain three coordinates")
            object.__setattr__(
                self,
                "freestream_velocity",
                tuple(float(item) for item in self.freestream_velocity),
            )


__all__ = ["ForceConfig", "VLMMeshSetup", "VLMSurfaceSetup", "VLMSetup"]
