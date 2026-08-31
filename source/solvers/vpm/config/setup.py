"""Top-level setup for the Vortex Particle Method solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from source.write_precision import (
    DEFAULT_WRITE_PRECISION,
    WritePrecision,
    validate_write_precision,
)

from ..boundary_elements.vlm.config import VLMSetup
from .advection import AdvectionConfig
from .artifacts import Backup, Samplers
from .constants import (
    DEFAULT_CUTOFF_RADIUS_FACTOR,
    DEFAULT_TIME_STEP,
    MAX_N_PARTICLES,
    TREECODE_SUPPORTED_KERNELS,
)
from .diagnostics import DiagnosticsConfig
from .health import HealthLimits
from .stabilization import StabilizationConfig
from .stretching import StretchingConfig
from .turbulence import TurbulenceConfig
from .velocity import VelocityConfig
from .viscous import ViscousConfig


@dataclass(frozen=True)
class PanelBodySetup:
    """Declarative setup for one closed STL body in the panel solver."""

    stl: str
    uid: str
    group_id: int = 0
    kinematics: object | None = None
    translation: tuple[float, float, float] | None = None
    rotation_degrees: tuple[float, float, float] | None = None
    rotation_centre: tuple[float, float, float] | None = None
    reference_area: float | None = None

    def __post_init__(self) -> None:
        if not str(self.stl).strip():
            raise ValueError("PanelBodySetup.stl must be a non-empty path")
        if not str(self.uid).strip():
            raise ValueError("PanelBodySetup.uid must be non-empty")
        for field_name in ("translation", "rotation_degrees", "rotation_centre"):
            value = getattr(self, field_name)
            if value is not None:
                if len(value) != 3:
                    raise ValueError(f"{field_name} must contain three coordinates")
                object.__setattr__(self, field_name, tuple(float(item) for item in value))
        if self.group_id < 0:
            raise ValueError("Panel body group_id must be non-negative")
        if self.reference_area is not None and self.reference_area <= 0.0:
            raise ValueError("Panel body reference_area must be positive when provided")


@dataclass(frozen=True)
class VPMSetup:
    """Complete immutable setup for a VPM simulation."""

    # Accepted-step duration. Runtime time and step are owned by VPMSolver.
    time_step_size: float = DEFAULT_TIME_STEP

    # Evolution
    time_integration: Literal["FRACTIONAL", "COUPLED"] = "FRACTIONAL"
    axisymmetric_no_swirl_axis: Literal["x", "y", "z"] | None = None

    advection: AdvectionConfig = field(default_factory=AdvectionConfig)
    stretching: StretchingConfig = field(default_factory=StretchingConfig.transposed)
    viscous: ViscousConfig = field(default_factory=ViscousConfig.cs)
    turbulence: TurbulenceConfig = field(default_factory=TurbulenceConfig.dns)
    stabilization: StabilizationConfig = field(default_factory=StabilizationConfig.disabled)
    vlm: VLMSetup | None = None

    particle_kernel: Literal[
        "GAUSSIAN",
        "HIGH_ORDER_GAUSSIAN",
        "SUPER_GAUSSIAN",
        "WINCKELMANS",
    ] = "GAUSSIAN"

    max_n_particles: int = MAX_N_PARTICLES
    max_evaluation_points: int = 200_000

    # Compute
    compute_device: Literal[
        "AUTO",
        "CPU",
        "VULKAN",
        "CUDA",
        "METAL",
    ] = "AUTO"

    precision: Literal["f32", "f64"] = "f32"
    write_precision: WritePrecision = DEFAULT_WRITE_PRECISION
    random_seed: int = 42
    device_memory_fraction: float = 0.5
    debug_mode: bool = False
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    health_limits: HealthLimits = field(default_factory=HealthLimits)

    # Output
    backup: Backup = field(default_factory=Backup)
    samplers: Samplers = field(default_factory=Samplers)

    # Flow and numerical controls
    cutoff_radius_factor: float = DEFAULT_CUTOFF_RADIUS_FACTOR
    freestream_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    verbose: bool = True
    velocity: VelocityConfig | None = None

    # Optional coupled solvers and sampling
    panel_solver: object | None = None
    bodies: tuple[PanelBodySetup, ...] = ()

    domain_bounds: tuple[float, ...] | None = None
    """Optional VPM domain ``(xmin, xmax, ymin, ymax, zmin, zmax)``."""

    def __post_init__(self) -> None:
        validate_write_precision(self.write_precision)
        if not isinstance(self.backup, Backup):
            raise TypeError("backup must be a Backup instance")
        if not isinstance(self.samplers, Samplers):
            raise TypeError("samplers must be a Samplers instance")
        if not isinstance(self.diagnostics, DiagnosticsConfig):
            raise TypeError("diagnostics must be a DiagnosticsConfig instance")
        if not isinstance(self.health_limits, HealthLimits):
            raise TypeError("health_limits must be a HealthLimits instance")
        if len(self.freestream_velocity) != 3:
            raise ValueError("freestream_velocity must contain three components")
        object.__setattr__(
            self,
            "freestream_velocity",
            tuple(float(value) for value in self.freestream_velocity),
        )

        object.__setattr__(self, "bodies", tuple(self.bodies))
        body_uids = [body.uid for body in self.bodies]
        if len(body_uids) != len(set(body_uids)):
            duplicates = sorted({uid for uid in body_uids if body_uids.count(uid) > 1})
            raise ValueError("Duplicate panel body uid(s): " + ", ".join(duplicates))

        if self.domain_bounds is not None:
            if len(self.domain_bounds) != 6:
                raise ValueError("domain_bounds must contain (xmin, xmax, ymin, ymax, zmin, zmax)")
            object.__setattr__(
                self,
                "domain_bounds",
                tuple(float(value) for value in self.domain_bounds),
            )

        if self.velocity is None:
            if self.particle_kernel.upper() in {
                "GAUSSIAN",
                "WINCKELMANS",
            }:
                velocity = VelocityConfig.treecode(theta=0.3)
            else:
                velocity = VelocityConfig.direct()
            object.__setattr__(self, "velocity", velocity)

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate cross-subsystem setup constraints."""
        if self.time_step_size <= 0.0:
            raise ValueError("time_step_size must be positive")

        integration = self.time_integration.upper()
        if integration not in {"FRACTIONAL", "COUPLED"}:
            raise ValueError("time_integration must be 'FRACTIONAL' or 'COUPLED'")

        if integration == "COUPLED":
            advection_scheme = self.advection.scheme.upper()
            stretching_scheme = self.stretching.scheme.upper()
            if not self.stretching.enabled:
                raise ValueError("COUPLED time integration requires stretching")
            if advection_scheme != stretching_scheme or advection_scheme not in {"RK2", "RK3"}:
                raise ValueError(
                    "COUPLED time integration requires matching "
                    "RK2 or RK3 advection and stretching schemes"
                )
            if self.viscous.scheme.upper() not in {
                "NONE",
                "CS",
                "RWM",
                "DVH",
                "GBD",
            }:
                raise ValueError(
                    "COUPLED time integration supports NONE, CS, RWM, DVH, or GBD diffusion"
                )
        elif self.stretching.conserve_moments or self.stretching.conserve_energy:
            raise ValueError("stretching invariant projection requires COUPLED time integration")

        if self.axisymmetric_no_swirl_axis is not None:
            axis = self.axisymmetric_no_swirl_axis.lower()
            object.__setattr__(
                self,
                "axisymmetric_no_swirl_axis",
                axis,
            )
            if integration != "COUPLED":
                raise ValueError("axisymmetric_no_swirl_axis requires COUPLED time integration")
            if self.stabilization.remove_particles_by_bounds is not None:
                raise ValueError(
                    "axisymmetric_no_swirl_axis is incompatible with particle retention"
                )
            if (
                self.stabilization.filament_refinement.enabled
                or self.stabilization.divergence_relaxation.enabled
            ):
                raise ValueError(
                    "axisymmetric_no_swirl_axis is incompatible "
                    "with refinement or divergence relaxation"
                )

        if self.max_n_particles < 1:
            raise ValueError("max_n_particles must be at least one")
        if self.max_evaluation_points < 1:
            raise ValueError("max_evaluation_points must be at least one")

        valid_devices = {"AUTO", "CPU", "VULKAN", "CUDA", "METAL"}
        if self.compute_device.upper() not in valid_devices:
            raise ValueError(f"compute_device must be one of {sorted(valid_devices)}")

        particle_kernel = self.particle_kernel.upper()
        valid_kernels = {
            "GAUSSIAN",
            "HIGH_ORDER_GAUSSIAN",
            "SUPER_GAUSSIAN",
            "WINCKELMANS",
        }
        if particle_kernel not in valid_kernels:
            raise ValueError(f"particle_kernel must be one of {sorted(valid_kernels)}")
        object.__setattr__(
            self,
            "particle_kernel",
            particle_kernel,
        )

        treecode = self.velocity is not None and self.velocity.method == "TREECODE"
        if treecode and particle_kernel not in TREECODE_SUPPORTED_KERNELS:
            raise ValueError(
                f"particle_kernel={particle_kernel!r} cannot be used "
                "with TREECODE velocity evaluation; supported kernels are "
                f"{list(TREECODE_SUPPORTED_KERNELS)}"
            )
        if treecode and self.precision == "f64":
            raise ValueError(
                "precision='f64' cannot be used with TREECODE because the current treecode is f32"
            )

        if self.stabilization.filament_refinement.enabled and particle_kernel != "GAUSSIAN":
            raise ValueError("filament refinement currently requires GAUSSIAN particles")
        if self.stabilization.divergence_relaxation.enabled and particle_kernel != "GAUSSIAN":
            raise ValueError("divergence relaxation currently requires GAUSSIAN particles")
        if (
            self.stabilization.divergence_relaxation.enabled
            and not self.stabilization.filament_refinement.enabled
        ):
            raise ValueError("divergence relaxation requires filament refinement")
        if self.stabilization.regularization_interval_steps > 0 and particle_kernel != "GAUSSIAN":
            raise ValueError("conservative regularization currently requires GAUSSIAN particles")
        if (
            self.stabilization.regularization_interval_steps > 0
            and self.stabilization.divergence_relaxation.enabled
        ):
            raise ValueError(
                "conservative regularization cannot be combined with divergence relaxation"
            )
        if (
            self.stabilization.regularization_max_particles is not None
            and self.stabilization.regularization_max_particles > self.max_n_particles
        ):
            raise ValueError("regularization_max_particles cannot exceed VPMSetup.max_n_particles")
        if (
            self.stabilization.regularization_capacity_max_particles is not None
            and self.stabilization.regularization_capacity_max_particles > self.max_n_particles
        ):
            raise ValueError(
                "regularization_capacity_max_particles cannot exceed VPMSetup.max_n_particles"
            )
        if (
            self.stabilization.filament_refinement.max_n_particles is not None
            and self.stabilization.filament_refinement.max_n_particles > self.max_n_particles
        ):
            raise ValueError(
                "filament-refinement max_n_particles cannot exceed VPMSetup.max_n_particles"
            )

    def __str__(self) -> str:
        """Return a concise engineering summary of the setup."""
        lines = [
            "VPM Setup:",
            f"  Flow model: {self.turbulence.flow_model}",
            f"  Time-step size: {self.time_step_size:.3e} s",
            f"  Advection: {self.advection.scheme}",
            (f"  Stretching: {self.stretching.scheme} / {self.stretching.mode}"),
            f"  Diffusion: {self.viscous.scheme}",
            f"  Compute device: {self.compute_device}",
            f"  Particle kernel: {self.particle_kernel}",
            f"  Backup interval: {self.backup.interval_steps} steps",
            f"  Freestream velocity: {self.freestream_velocity} m/s",
        ]
        return "\n".join(lines)
