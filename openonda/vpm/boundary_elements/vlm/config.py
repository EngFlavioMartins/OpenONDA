"""Declarative, immutable configuration for the vortex-lattice solver."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Literal

from openonda.core.kinematics import RigidMotion

if TYPE_CHECKING:
    from .geometry.aircraft import Aircraft


@dataclass(frozen=True, slots=True)
class ForceConfig:
    """Select aerodynamic load evaluation for an attached vortex-lattice model.

    Parameters
    ----------
    method : {'kutta_joukowski'}, default='kutta_joukowski'
        Bound-leg Kutta–Joukowski loads, optionally with unsteady pressure.
    unsteady : bool, default=False
        Include the backward time difference of the surface potential jump
        from unsteady Bernoulli, integrated over the physical panels. Requires
        a positive physical time step in postprocessing.
    """

    method: Literal["kutta_joukowski"] = "kutta_joukowski"
    unsteady: bool = False

    @staticmethod
    def kutta_joukowski(*, unsteady: bool = False) -> ForceConfig:
        """Configure bound-leg loads and optional pressure-time forces/moments."""
        return ForceConfig(unsteady=unsteady)


@dataclass(frozen=True, slots=True)
class VLMMeshConfig:
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
    sample_forces : bool or None, optional
        Per-surface override for force-history output.
    """

    surface: str | Aircraft
    name: str | None = None
    kinematics: RigidMotion | None = None
    translation: tuple[float, float, float] | None = None
    rotation_degrees: tuple[float, float, float] | None = None
    rotation_centre: tuple[float, float, float] | None = None
    group_id: int = 0
    sample_forces: bool | None = None

    def __post_init__(self) -> None:
        """Validate declared settings and normalize immutable configuration values."""
        for field_name in ("translation", "rotation_degrees", "rotation_centre"):
            value = getattr(self, field_name)
            if value is not None:
                if len(value) != 3:
                    raise ValueError(f"{field_name} must contain three coordinates")
                if not all(math.isfinite(float(item)) for item in value):
                    raise ValueError(f"{field_name} coordinates must be finite")
                object.__setattr__(self, field_name, tuple(float(item) for item in value))
        if self.kinematics is not None and not isinstance(self.kinematics, RigidMotion):
            raise TypeError("VLMSurface.kinematics must be a RigidMotion instance or None")
        if (
            isinstance(self.group_id, bool)
            or not isinstance(self.group_id, int)
            or self.group_id < 0
        ):
            raise ValueError("VLM surface group_id must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class VLMConfig:
    """Complete VLM solver definition.

    ``max_n_panels`` is optional. When omitted, capacity is derived exactly from
    the declared surfaces, so normal cases need no allocation tuning.

    Parameters
    ----------
    surfaces : tuple[VLMSurface, ...]
        One or more lifting-surface declarations.
    mesh : VLMMeshConfig
        Shared panel-distribution settings.
    max_n_panels : int or None, optional
        Positive allocation ceiling; ``None`` derives exact capacity.
    dtype : {'f32', 'f64'}, default='f32'
        VLM field precision.
    linear_solver : {'scipy', 'bicgstab_gpu'} or None, optional
        Circulation solver; ``None`` selects a backend-appropriate default.
    circulation_relaxation : float, default=1
        New/old circulation blend fraction in ``(0, 1]``.
    kinematic_viscosity : float, default=1
        Non-negative viscosity in m²/s assigned to shed wake particles.
    density : float, default=1
        Positive fluid density in kg/m³ used to dimensionalize forces.
    wake_core_overlap : float, default=2.5
        Core radius divided by the larger local span spacing and convected
        row length, for both trailing and transverse wake elements.
        Positive values above one provide overlapping blobs.
    boundary_response : {'lagged', 'responsive'}, default='lagged'
        Coupled VPM boundary-response settings. ``'lagged'`` preserves the
        historical accepted-step solve. ``'responsive'`` solves a pure
        temporary VLM system at every particle RK stage from that stage's
        incident wake and prescribed surface geometry; only the accepted
        solve emits the wake row and publishes history.
        The responsive formulation remains experimental pending coupled
        temporal-convergence qualification. The virtual row retains its old
        closing vector and contributes transport/stretching, but uses the
        first-order row-convection rule also used by accepted emission.
        A small stage-system residual does not establish physical accuracy.
    freestream_velocity : tuple[float, float, float] or None, optional
        Uniform background velocity in m/s; ``None`` inherits the VPM value.
    logging_interval_steps : int, default=1
        Positive accepted-step force-table cadence for standalone VLM use.
        Coupled VPM cases own scientific output cadence and require this to
        remain at its solver-step value of one.
    force : ForceConfig
        Aerodynamic-force model.
    sample_surface_forces : bool, default=True
        Write per-surface as well as aggregate force histories. Coupled VPM
        cases require this solver-sample output and reject an explicit opt-out.
    """

    surfaces: tuple[VLMSurface, ...]
    mesh: VLMMeshConfig = field(default_factory=VLMMeshConfig)
    max_n_panels: int | None = None
    dtype: Literal["f32", "f64"] = "f32"
    linear_solver: Literal["scipy", "bicgstab_gpu"] | None = None
    circulation_relaxation: float = 1.0
    kinematic_viscosity: float = 1.0
    density: float = 1.0
    freestream_velocity: tuple[float, float, float] | None = None
    logging_interval_steps: int = 1
    force: ForceConfig = field(default_factory=ForceConfig.kutta_joukowski)
    sample_surface_forces: bool = True
    wake_core_overlap: float = 2.5
    boundary_response: Literal["lagged", "responsive"] = "lagged"

    def __post_init__(self) -> None:
        """Validate declared settings and normalize immutable configuration values."""
        object.__setattr__(self, "surfaces", tuple(self.surfaces))
        if not self.surfaces:
            raise ValueError("VLMConfig requires at least one surface")
        if self.max_n_panels is not None and (
            isinstance(self.max_n_panels, bool)
            or not isinstance(self.max_n_panels, int)
            or self.max_n_panels < 1
        ):
            raise ValueError("VLM max_n_panels must be a positive integer when provided")
        if self.dtype not in {"f32", "f64"}:
            raise ValueError("VLM dtype must be f32 or f64")
        if self.linear_solver not in {None, "scipy", "bicgstab_gpu"}:
            raise ValueError("VLM linear_solver must be SCIPY or BICGSTAB_GPU")
        if not 0 < self.circulation_relaxation <= 1:
            raise ValueError("circulation_relaxation must be in (0, 1]")
        if not math.isfinite(self.kinematic_viscosity) or self.kinematic_viscosity < 0:
            raise ValueError("VLM kinematic_viscosity must be non-negative")
        if not math.isfinite(self.density) or self.density <= 0:
            raise ValueError("VLM density must be positive")
        if not math.isfinite(self.wake_core_overlap) or self.wake_core_overlap <= 0:
            raise ValueError("VLM wake_core_overlap must be finite and positive")
        if self.boundary_response not in {"lagged", "responsive"}:
            raise ValueError("boundary_response must be 'lagged' or 'responsive'")
        if (
            isinstance(self.logging_interval_steps, bool)
            or not isinstance(self.logging_interval_steps, int)
            or self.logging_interval_steps < 1
        ):
            raise ValueError("VLM logging_interval_steps must be a positive integer")
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
