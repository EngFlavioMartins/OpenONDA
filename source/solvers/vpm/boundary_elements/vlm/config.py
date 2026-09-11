"""Declarative, immutable configuration for the vortex-lattice solver."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal


@dataclass(frozen=True)
class ForceConfig:
    """Select aerodynamic load evaluation for an attached vortex-lattice model.

    Parameters
    ----------
    method : {'KUTTA_JOUKOWSKI'}, default='KUTTA_JOUKOWSKI'
        Bound-leg Kutta–Joukowski loads, optionally with unsteady pressure.
    kj_smoothing : bool, default=False
        Average current and previous circulation for Kutta–Joukowski loads.
        This temporal filter changes reported forces, not solved circulation.
    unsteady : bool, default=False
        Include the backward time difference of the surface potential jump
        from unsteady Bernoulli, integrated over the physical panels. Requires
        a positive physical time step in postprocessing.
    """

    method: Literal["KUTTA_JOUKOWSKI"] = "KUTTA_JOUKOWSKI"
    kj_smoothing: bool = False
    unsteady: bool = False

    @staticmethod
    def kutta_joukowski(*, smoothing: bool = False, unsteady: bool = False) -> ForceConfig:
        """Configure bound-leg loads and optional pressure-time forces/moments."""
        return ForceConfig(kj_smoothing=smoothing, unsteady=unsteady)


@dataclass(frozen=True)
class VLMMeshSetup:
    """Configure chordwise/spanwise panel spacing for all declared VLM surfaces.

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
        """Configure geometric panel spacing toward the selected region."""
        return VLMMeshSetup(spacing="geometric", ratio=ratio, region=region)


@dataclass(frozen=True)
class VLMSurfaceSetup:
    """Declare one lifting surface and its placement in a VLM/VPM case.

    Parameters
    ----------
    surface : object
        Surface geometry accepted by the VLM mesh builder; commonly a surface
        mapping loaded from an OpenVSP/JSON description.
    name : str or None, optional
        Stable user-facing identifier; the geometry name is used when omitted.
    kinematics : object or None, optional
        Motion object implementing the VLM kinematics interface. ``None`` is static.
    translation : tuple[float, float, float] or None, optional
        Initial Cartesian translation in m.
    rotation_degrees : tuple[float, float, float] or None, optional
        Initial x/y/z rotations in degrees.
    rotation_centre : tuple[float, float, float] or None, optional
        Cartesian pivot in the input geometry coordinates, before translation, in m.
    group_id : int, default=0
        Non-negative label copied to shed particles for grouping/diagnostics.
    sample_forces : bool or None, optional
        Per-surface override for force-history output.
    """

    surface: object
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

    Parameters
    ----------
    surfaces : tuple[VLMSurfaceSetup, ...]
        One or more lifting-surface declarations.
    mesh : VLMMeshSetup
        Shared panel-distribution policy.
    max_n_panels : int or None, optional
        Positive allocation ceiling; ``None`` derives exact capacity.
    dtype : {'f32', 'f64'}, default='f32'
        VLM field precision.
    linear_solver : {'SCIPY', 'BICGSTAB_GPU'} or None, optional
        Circulation solver; ``None`` selects a backend-appropriate default.
    circulation_relaxation : float, default=1
        New/old circulation blend fraction in ``(0, 1]``.
    kinematic_viscosity : float, default=1
        Non-negative viscosity in m²/s assigned to shed wake particles.
    density : float, default=1
        Positive fluid density in kg/m³ used to dimensionalize forces.
    sigma_factor : float, default=2.5
        Legacy transverse-element radius factor on the convected row length.
        Used when ``wake_core_overlap`` is omitted; trailing cores retain
        their legacy local-spacing radius.
    wake_core_overlap : float or None, optional
        Common core radius divided by the larger local span spacing and
        convected row length, for both trailing and transverse elements.
        A supplied value replaces the legacy radius rule; values above one
        provide overlapping blobs in a thin, high-Reynolds-number wake.
    freestream_velocity : tuple[float, float, float] or None, optional
        Uniform background velocity in m/s; ``None`` inherits the VPM value.
    logging_interval_steps : int, default=1
        Positive accepted-step force-table cadence for standalone VLM use.
        Coupled VPM cases own scientific output cadence and require this to
        remain at its owner-step value of one.
    force : ForceConfig
        Aerodynamic-force model.
    sample_surface_forces : bool, default=True
        Write per-surface as well as aggregate force histories. Coupled VPM
        cases require this owner-sample output and reject an explicit opt-out.
    """

    surfaces: tuple[VLMSurfaceSetup, ...]
    mesh: VLMMeshSetup = field(default_factory=VLMMeshSetup)
    max_n_panels: int | None = None
    dtype: Literal["f32", "f64"] = "f32"
    linear_solver: Literal["SCIPY", "BICGSTAB_GPU"] | None = None
    circulation_relaxation: float = 1.0
    kinematic_viscosity: float = 1.0
    density: float = 1.0
    sigma_factor: float = 2.5
    freestream_velocity: tuple[float, float, float] | None = None
    logging_interval_steps: int = 1
    force: ForceConfig = field(default_factory=ForceConfig.kutta_joukowski)
    sample_surface_forces: bool = True
    wake_core_overlap: float | None = None

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
        if not math.isfinite(self.sigma_factor) or self.sigma_factor <= 0:
            raise ValueError("VLM sigma_factor must be positive")
        if self.wake_core_overlap is not None and (
            not math.isfinite(self.wake_core_overlap) or self.wake_core_overlap <= 0
        ):
            raise ValueError("VLM wake_core_overlap must be finite and positive")
        if (
            isinstance(self.logging_interval_steps, bool)
            or not isinstance(self.logging_interval_steps, int)
            or self.logging_interval_steps < 1
        ):
            raise ValueError("VLM logging_interval_steps must be a positive integer")
        if self.freestream_velocity is not None:
            if len(self.freestream_velocity) != 3:
                raise ValueError("freestream_velocity must contain three coordinates")
            object.__setattr__(
                self,
                "freestream_velocity",
                tuple(float(item) for item in self.freestream_velocity),
            )


__all__ = ["ForceConfig", "VLMMeshSetup", "VLMSurfaceSetup", "VLMSetup"]
