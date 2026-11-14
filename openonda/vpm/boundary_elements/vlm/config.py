"""Declarative, immutable configuration for the vortex-lattice solver."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Literal

from openonda.core.kinematics import RigidMotion, RigidPose, finite_placement, placement_pose

if TYPE_CHECKING:
    from .geometry.aircraft import Aircraft


@dataclass(frozen=True, slots=True)
class ForceConfig:
    """Select aerodynamic load evaluation for an attached vortex-lattice model.

    Loads are the Kutta–Joukowski forces on the bound legs, optionally with the
    unsteady pressure term.

    Parameters
    ----------
    unsteady : bool, default=True
        Include the backward time difference of the surface potential jump
        from unsteady Bernoulli, integrated over the physical panels. Requires
        a positive physical time step in postprocessing.
    """

    unsteady: bool = True

    @staticmethod
    def kutta_joukowski(*, unsteady: bool = True) -> ForceConfig:
        """Configure bound-leg loads and optional pressure-time forces/moments."""
        return ForceConfig(unsteady=unsteady)


@dataclass(frozen=True, slots=True)
class VLMMeshConfig:
    """Configure the spanwise panel spacing within each wing segment of all declared surfaces.

    Parameters
    ----------
    spacing : {'uniform', 'geometric'}, default='uniform'
        Distribution family applied by the surface mesher.
    ratio : float, default=1.0
        Positive largest-to-smallest spacing ratio. Uniform spacing requires 1.
    region : {'start', 'end', 'both'}, default='both'
        Edge or edges toward which geometric cells are clustered.
    """

    spacing: Literal["uniform", "geometric"] = "uniform"
    ratio: float = 1.0
    region: Literal["start", "end", "both"] = "both"

    def __post_init__(self) -> None:
        """Validate declared settings and normalize immutable configuration values."""
        if self.spacing not in {"uniform", "geometric"}:
            raise ValueError("VLM mesh spacing must be uniform or geometric")
        if self.region not in {"start", "end", "both"}:
            raise ValueError("VLM mesh region must be start, end, or both")
        if not math.isfinite(self.ratio) or self.ratio <= 0:
            raise ValueError("VLM mesh ratio must be finite and positive")
        if self.spacing == "uniform" and self.ratio != 1.0:
            raise ValueError("A non-unit VLM mesh ratio requires geometric spacing")

    @staticmethod
    def geometric(
        ratio: float = 3.0,
        *,
        region: Literal["start", "end", "both"] = "both",
    ) -> VLMMeshConfig:
        """Configure geometric panel spacing toward the selected region."""
        return VLMMeshConfig(spacing="geometric", ratio=ratio, region=region)


@dataclass(frozen=True, slots=True)
class VLMSurface:
    """Declare one lifting surface and its placement in a VLM/VPM case.

    Parameters
    ----------
    surface : str or Aircraft
        Surface geometry: a surface JSON path, or an :class:`Aircraft` built
        directly or imported from an OpenVSP description.
    name : str or None, optional
        Stable user-facing identifier; the geometry name is used when omitted.
    kinematics : RigidMotion or None, optional
        Prescribed rigid motion of the placed surface, one of the shared
        motions of :mod:`openonda.core.kinematics`. ``None`` is static.
    translation : tuple[float, float, float] or None, optional
        Initial Cartesian translation in m.
    rotation_degrees : tuple[float, float, float] or None, optional
        Initial x/y/z rotations in degrees.
    rotation_centre : tuple[float, float, float] or None, optional
        Cartesian pivot in the input geometry coordinates, before translation, in m.
    group_id : int, default=0
        Non-negative label copied to shed particles for grouping/diagnostics.
    """

    surface: str | Aircraft
    name: str | None = None
    kinematics: RigidMotion | None = None
    translation: tuple[float, float, float] | None = None
    rotation_degrees: tuple[float, float, float] | None = None
    rotation_centre: tuple[float, float, float] | None = None
    group_id: int = 0

    def __post_init__(self) -> None:
        """Validate declared settings and normalize immutable configuration values."""
        translation, rotation_degrees, rotation_centre = finite_placement(
            "VLMSurface", self.translation, self.rotation_degrees, self.rotation_centre
        )
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "rotation_degrees", rotation_degrees)
        object.__setattr__(self, "rotation_centre", rotation_centre)
        if self.kinematics is not None and not isinstance(self.kinematics, RigidMotion):
            raise TypeError("VLMSurface.kinematics must be a RigidMotion instance or None")
        if (
            isinstance(self.group_id, bool)
            or not isinstance(self.group_id, int)
            or self.group_id < 0
        ):
            raise ValueError("VLM surface group_id must be a non-negative integer")

    def placement(self) -> RigidPose:
        """Return the initial placement pose of the surface geometry."""
        return placement_pose(self.translation, self.rotation_degrees, self.rotation_centre)


@dataclass(frozen=True, slots=True)
class VLMConfig:
    """Complete VLM solver definition.

    Parameters
    ----------
    surfaces : tuple[VLMSurface, ...]
        One or more lifting-surface declarations.
    mesh : VLMMeshConfig
        Shared panel-distribution settings.
    kinematic_viscosity : float, default=1
        Non-negative viscosity in m²/s assigned to shed wake particles.
    density : float, default=1
        Positive fluid density in kg/m³ used to dimensionalize forces.
    wake_core_overlap : float, default=2.5
        Core radius divided by the larger local span spacing and convected
        row length, for both trailing and transverse wake elements.
        Positive values above one provide overlapping blobs.
    freestream_velocity : tuple[float, float, float] or None, optional
        Uniform background velocity in m/s; ``None`` inherits the VPM value.
    force : ForceConfig
        Aerodynamic-force model.
    """

    surfaces: tuple[VLMSurface, ...]
    mesh: VLMMeshConfig = field(default_factory=VLMMeshConfig)
    kinematic_viscosity: float = 1.0
    density: float = 1.0
    freestream_velocity: tuple[float, float, float] | None = None
    force: ForceConfig = field(default_factory=ForceConfig.kutta_joukowski)
    wake_core_overlap: float = 2.5

    def __post_init__(self) -> None:
        """Validate declared settings and normalize immutable configuration values."""
        object.__setattr__(self, "surfaces", tuple(self.surfaces))
        if not self.surfaces:
            raise ValueError("VLMConfig requires at least one surface")
        if not math.isfinite(self.kinematic_viscosity) or self.kinematic_viscosity < 0:
            raise ValueError("VLM kinematic_viscosity must be non-negative")
        if not math.isfinite(self.density) or self.density <= 0:
            raise ValueError("VLM density must be positive")
        if not math.isfinite(self.wake_core_overlap) or self.wake_core_overlap <= 0:
            raise ValueError("VLM wake_core_overlap must be finite and positive")
        if self.freestream_velocity is not None:
            if len(self.freestream_velocity) != 3:
                raise ValueError("freestream_velocity must contain three coordinates")
            if not all(math.isfinite(float(value)) for value in self.freestream_velocity):
                raise ValueError("freestream_velocity coordinates must be finite")
            object.__setattr__(
                self,
                "freestream_velocity",
                tuple(float(item) for item in self.freestream_velocity),
            )


__all__ = ["ForceConfig", "VLMConfig", "VLMMeshConfig", "VLMSurface"]
