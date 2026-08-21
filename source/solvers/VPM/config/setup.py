"""Top-level setup for the Vortex Particle Method solver."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
from pathlib import Path
import re
import sys
from typing import Any, Literal

from ..boundary_elements.vlm.config import VLMSetup
from .advection import AdvectionConfig
from .constants import (
    DEFAULT_BACKUP_FILENAME,
    DEFAULT_CUTOFF_RADIUS_FACTOR,
    DEFAULT_TIME_STEP,
    MAX_PARTICLES,
    TREECODE_SUPPORTED_KERNELS,
)
from .divergence_relaxation import DivergenceRelaxationConfig
from .filament_refinement import FilamentRefinementConfig
from .stabilization import StabilizationConfig
from .stretching import StretchingConfig
from .turbulence import TurbulenceConfig
from .velocity import VelocityConfig
from .viscous import ViscousConfig

sys.tracebacklimit = 0


@dataclass(frozen=True)
class VPMSetup:
    """Complete immutable setup for a VPM simulation."""

    # Time
    time_step_size: float = DEFAULT_TIME_STEP
    time: float = 0.0
    step: int = 0

    # Evolution
    time_integration: Literal["FRACTIONAL", "COUPLED"] = "FRACTIONAL"
    coupled_max_strain_increment: float = 0.08
    coupled_max_advection_fraction: float = 0.25
    coupled_max_substeps: int = 128
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

    max_particles: int = MAX_PARTICLES
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
    random_seed: int = 42
    device_memory_fraction: float = 0.5
    debug_mode: bool = False

    # Monitoring and output
    logging_interval_steps: int = 0
    timing_interval_steps: int = 0
    checkpoint_interval_steps: int = 0
    checkpoint_name: str = DEFAULT_BACKUP_FILENAME
    checkpoint_directory: str = "solution"
    sample_subdirectory: str | None = None
    clean: bool = False

    export_flow_integrals: bool = True
    export_discretization_health: bool = True
    log_mode: Literal["file", "tee", "console"] = "tee"

    # Flow and numerical controls
    cutoff_radius_factor: float = DEFAULT_CUTOFF_RADIUS_FACTOR
    freestream_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    verbose: bool = True
    velocity: VelocityConfig | None = None

    # Optional coupled solvers and sampling
    panel_solver: Any | None = None
    samplers: tuple[Any, ...] | None = None
    final_samplers: tuple[Any, ...] | None = None
    body_stl: str | None = None

    domain_bounds: tuple[float, ...] | None = None
    """Optional VPM domain ``(xmin, xmax, ymin, ymax, zmin, zmax)``."""

    def __post_init__(self) -> None:
        if len(self.freestream_velocity) != 3:
            raise ValueError("freestream_velocity must contain three components")
        object.__setattr__(
            self,
            "freestream_velocity",
            tuple(float(value) for value in self.freestream_velocity),
        )

        if self.samplers is not None:
            object.__setattr__(self, "samplers", tuple(self.samplers))
        if self.final_samplers is not None:
            object.__setattr__(
                self,
                "final_samplers",
                tuple(self.final_samplers),
            )

        if self.domain_bounds is not None:
            if len(self.domain_bounds) != 6:
                raise ValueError("domain_bounds must contain (xmin, xmax, ymin, ymax, zmin, zmax)")
            object.__setattr__(
                self,
                "domain_bounds",
                tuple(float(value) for value in self.domain_bounds),
            )

        checkpoint_name = self.checkpoint_name.strip()
        if checkpoint_name and (
            re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_-]*",
                checkpoint_name,
            )
            is None
            or checkpoint_name.startswith(("vpm_", "vlm_"))
        ):
            raise ValueError(
                "checkpoint_name must be a filename-safe infix without "
                "a path, extension, or solver prefix"
            )
        object.__setattr__(self, "checkpoint_name", checkpoint_name)

        if self.sample_subdirectory is not None:
            sample_path = Path(self.sample_subdirectory)
            if (
                not self.sample_subdirectory
                or sample_path.is_absolute()
                or any(part in {".", ".."} for part in sample_path.parts)
            ):
                raise ValueError(
                    "sample_subdirectory must be a non-empty relative path below samples/"
                )
            object.__setattr__(
                self,
                "sample_subdirectory",
                str(sample_path),
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
        if self.time < 0.0:
            raise ValueError("time must be non-negative")
        if self.step < 0:
            raise ValueError("step must be non-negative")

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
                "DVH",
                "GBD",
            }:
                raise ValueError(
                    "COUPLED time integration supports NONE, CS, DVH, or GBD diffusion"
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

        if self.coupled_max_strain_increment <= 0.0:
            raise ValueError("coupled_max_strain_increment must be positive")
        if self.coupled_max_advection_fraction <= 0.0:
            raise ValueError("coupled_max_advection_fraction must be positive")
        if self.coupled_max_substeps < 1:
            raise ValueError("coupled_max_substeps must be at least one")

        for name in (
            "logging_interval_steps",
            "timing_interval_steps",
            "checkpoint_interval_steps",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

        if self.max_particles < 1:
            raise ValueError("max_particles must be at least one")
        if self.max_evaluation_points < 1:
            raise ValueError("max_evaluation_points must be at least one")

        if self.log_mode not in {"file", "tee", "console"}:
            raise ValueError("log_mode must be 'file', 'tee', or 'console'")

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
        if self.stabilization.regularization_interval_steps > 0 and (
            self.stabilization.filament_refinement.enabled
            or self.stabilization.divergence_relaxation.enabled
        ):
            raise ValueError(
                "conservative regularization replaces refinement and divergence relaxation"
            )
        if (
            self.stabilization.regularization_max_particles is not None
            and self.stabilization.regularization_max_particles > self.max_particles
        ):
            raise ValueError("regularization_max_particles cannot exceed VPMSetup.max_particles")
        if (
            self.stabilization.filament_refinement.max_particles is not None
            and self.stabilization.filament_refinement.max_particles > self.max_particles
        ):
            raise ValueError(
                "filament-refinement max_particles cannot exceed VPMSetup.max_particles"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable setup using canonical field names."""
        if self.vlm is not None:
            raise ValueError(
                "VPMSetup.to_dict() cannot serialize an attached VLMSetup: VLM "
                "surfaces and kinematics are live Python objects, so a serialized "
                "copy would silently discard the coupling configuration. Construct "
                "the coupled solver programmatically instead of round-tripping it "
                "through a file."
            )

        def as_serializable(value: Any) -> Any:
            if hasattr(value, "__dataclass_fields__"):
                return {
                    item.name: as_serializable(getattr(value, item.name)) for item in fields(value)
                }
            if isinstance(value, tuple):
                return [as_serializable(item) for item in value]
            return value

        return {
            "time_step_size": self.time_step_size,
            "time": self.time,
            "step": self.step,
            "time_integration": self.time_integration,
            "coupled_max_strain_increment": (self.coupled_max_strain_increment),
            "coupled_max_advection_fraction": (self.coupled_max_advection_fraction),
            "coupled_max_substeps": self.coupled_max_substeps,
            "axisymmetric_no_swirl_axis": (self.axisymmetric_no_swirl_axis),
            "advection": as_serializable(self.advection),
            "stretching": as_serializable(self.stretching),
            "viscous": as_serializable(self.viscous),
            "turbulence": as_serializable(self.turbulence),
            "stabilization": as_serializable(self.stabilization),
            "vlm": None,
            "particle_kernel": self.particle_kernel,
            "max_particles": self.max_particles,
            "max_evaluation_points": self.max_evaluation_points,
            "compute_device": self.compute_device,
            "logging_interval_steps": self.logging_interval_steps,
            "timing_interval_steps": self.timing_interval_steps,
            "checkpoint_interval_steps": (self.checkpoint_interval_steps),
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_directory": self.checkpoint_directory,
            "sample_subdirectory": self.sample_subdirectory,
            "export_flow_integrals": self.export_flow_integrals,
            "export_discretization_health": (self.export_discretization_health),
            "log_mode": self.log_mode,
            "clean": self.clean,
            "cutoff_radius_factor": self.cutoff_radius_factor,
            "precision": self.precision,
            "random_seed": self.random_seed,
            "device_memory_fraction": self.device_memory_fraction,
            "debug_mode": self.debug_mode,
            "freestream_velocity": list(self.freestream_velocity),
            "verbose": self.verbose,
            "velocity": (as_serializable(self.velocity) if self.velocity is not None else None),
            "body_stl": self.body_stl,
            "domain_bounds": (list(self.domain_bounds) if self.domain_bounds is not None else None),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VPMSetup:
        """Reconstruct a setup from the canonical serialized schema."""
        values = dict(data)
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError("Unknown VPMSetup field(s): " + ", ".join(unknown))

        if isinstance(values.get("viscous"), dict):
            values["viscous"] = ViscousConfig(**values["viscous"])
        if isinstance(values.get("turbulence"), dict):
            turbulence = dict(values["turbulence"])
            turbulence.pop("flow_model", None)
            values["turbulence"] = TurbulenceConfig(**turbulence)
        for name, config_type in {
            "advection": AdvectionConfig,
            "stretching": StretchingConfig,
            "velocity": VelocityConfig,
        }.items():
            if isinstance(values.get(name), dict):
                values[name] = config_type(**values[name])
        if isinstance(values.get("stabilization"), dict):
            values["stabilization"] = cls._stabilization_from_dict(values)
        return cls(**values)

    @staticmethod
    def _stabilization_from_dict(values: dict[str, Any]) -> StabilizationConfig:
        """Rebuild stabilization from the canonical nested schema."""
        section = values.pop("stabilization", None)
        if section is None:
            return StabilizationConfig.disabled()
        if isinstance(section, StabilizationConfig):
            return section
        if not isinstance(section, dict):
            raise TypeError("stabilization must be a mapping")
        section = dict(section)
        for name, config_type in {
            "filament_refinement": FilamentRefinementConfig,
            "divergence_relaxation": DivergenceRelaxationConfig,
        }.items():
            entry = section.get(name)
            if isinstance(entry, dict):
                section[name] = config_type(**entry)
            elif entry is not None and not isinstance(entry, config_type):
                raise TypeError(f"{name} must be a mapping")
        known = {item.name for item in fields(StabilizationConfig)}
        unknown = sorted(set(section) - known)
        if unknown:
            raise ValueError("Unknown StabilizationConfig field(s): " + ", ".join(unknown))
        return StabilizationConfig(**section)

    @staticmethod
    def viscous_flow_simulation(
        time_step_size: float = 0.01,
        freestream_velocity: tuple[float, float, float] = (
            0.0,
            0.0,
            0.0,
        ),
        viscous: ViscousConfig | None = None,
        **kwargs: Any,
    ) -> VPMSetup:
        """Return a standard viscous-flow setup."""
        if viscous is None:
            viscous = ViscousConfig.cs()

        stretching = kwargs.pop(
            "stretching",
            StretchingConfig.disabled(),
        )
        return VPMSetup(
            time_step_size=time_step_size,
            stretching=stretching,
            viscous=viscous,
            freestream_velocity=freestream_velocity,
            **kwargs,
        )

    @staticmethod
    def dns_simulation(
        time_step_size: float = 0.01,
        **kwargs: Any,
    ) -> VPMSetup:
        """Return a DNS setup."""
        stretching = kwargs.pop(
            "stretching",
            StretchingConfig.transposed(),
        )
        viscous = kwargs.pop(
            "viscous",
            ViscousConfig.cs(),
        )
        turbulence = kwargs.pop(
            "turbulence",
            TurbulenceConfig.dns(),
        )
        return VPMSetup(
            time_step_size=time_step_size,
            stretching=stretching,
            viscous=viscous,
            turbulence=turbulence,
            **kwargs,
        )

    @staticmethod
    def les_simulation(
        time_step_size: float = 0.01,
        c_s: float = 0.17,
        c_e: float = 1.048,
        **kwargs: Any,
    ) -> VPMSetup:
        """Return an equilibrium Smagorinsky LES setup."""
        stretching = kwargs.pop(
            "stretching",
            StretchingConfig.transposed(),
        )
        viscous = kwargs.pop(
            "viscous",
            ViscousConfig.cs(),
        )
        turbulence = kwargs.pop(
            "turbulence",
            TurbulenceConfig.les_smagorinsky(
                c_s=c_s,
                c_e=c_e,
            ),
        )
        return VPMSetup(
            time_step_size=time_step_size,
            stretching=stretching,
            viscous=viscous,
            turbulence=turbulence,
            **kwargs,
        )

    def save_to_file(self, filename: str) -> None:
        """Write the setup to JSON using canonical field names."""
        with open(filename, "w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2)

    @classmethod
    def load_from_file(cls, filename: str) -> VPMSetup:
        """Load a setup from canonical or legacy JSON."""
        with open(filename, encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

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
            (f"  Logging interval: {self.logging_interval_steps} steps"),
            (f"  Checkpoint interval: {self.checkpoint_interval_steps} steps"),
            f"  Freestream velocity: {self.freestream_velocity} m/s",
        ]
        return "\n".join(lines)
