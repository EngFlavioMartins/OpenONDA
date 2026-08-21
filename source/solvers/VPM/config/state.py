"""Serializable VPM state models and state-management helpers."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

F = TypeVar("F", bound=Callable[..., Any])


def cached_particle_property(func: F) -> F:
    """Cache an expensive particle property for the current solver step."""
    cache_name = f"_{func.__name__}_cache"
    cache_step_name = f"_{func.__name__}_cache_step"

    @wraps(func)
    def wrapper(self, use_cache: bool = True):
        cache_valid = (
            use_cache
            and getattr(self, "_cache_step", None) != -1
            and getattr(self, cache_step_name, None) == self.step
            and hasattr(self, cache_name)
        )
        if not cache_valid:
            setattr(self, cache_name, func(self))
            setattr(self, cache_step_name, self.step)
        return getattr(self, cache_name)

    return wrapper  # type: ignore[return-value]


class SolverState(BaseModel):
    """Serializable scalar state for a VPM solver (canonical names only)."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    time_step_size: float = Field(gt=0.0)
    time: float = Field(
        default=0.0,
        ge=0.0,
    )
    step: int = Field(
        default=0,
        ge=0,
    )
    advection_scheme: str = "RK3"
    stretching_scheme: str = "RK3"
    compute_device: str = Field(
        default="AUTO",
    )
    flow_model: str = "DNS"
    viscous_scheme: str = "CS"
    stretching_enabled: bool = True
    stretching_mode: str = "TRANSPOSED"
    particle_kernel: str = Field(
        default="GAUSSIAN",
    )
    checkpoint_name: str = Field(
        default="",
    )
    checkpoint_directory: str = Field(
        default="solution",
    )
    logging_interval_steps: int = Field(
        default=0,
        ge=0,
    )
    timing_interval_steps: int = Field(
        default=0,
        ge=0,
    )
    checkpoint_interval_steps: int = Field(
        default=0,
        ge=0,
    )
    wall_time: float | None = Field(
        default=0.0,
        ge=0.0,
    )
    cache_step: int | None = Field(
        default=0,
    )

    @field_validator("compute_device")
    @classmethod
    def validate_compute_device(cls, value: str) -> str:
        valid = {"AUTO", "CPU", "VULKAN", "CUDA", "METAL"}
        value = value.upper()
        if value not in valid:
            raise ValueError(f"compute_device must be one of {sorted(valid)}")
        return value

    @field_validator("flow_model")
    @classmethod
    def validate_flow_model(cls, value: str) -> str:
        valid = {"DNS", "LES", "INVISCID"}
        value = value.upper()
        if value not in valid:
            raise ValueError(f"flow_model must be one of {sorted(valid)}")
        return value

    @classmethod
    def from_solver(cls, solver: Any) -> SolverState:
        """Create a scalar-state snapshot from a canonical ``VPMSolver``."""
        return cls(
            time_step_size=solver.time_step_size,
            time=solver.time,
            step=solver.step,
            advection_scheme=solver.advection_scheme,
            stretching_scheme=solver.stretching_scheme,
            compute_device=solver.compute_device,
            flow_model=solver.flow_model,
            viscous_scheme=solver.viscous_scheme,
            stretching_enabled=solver.stretching_enabled,
            stretching_mode=solver.stretching_mode,
            particle_kernel=solver.particle_kernel,
            checkpoint_name=solver.checkpoint_name,
            checkpoint_directory=solver.checkpoint_directory,
            logging_interval_steps=solver.logging_interval_steps,
            timing_interval_steps=solver.timing_interval_steps,
            checkpoint_interval_steps=solver.checkpoint_interval_steps,
            wall_time=solver.wall_time,
            cache_step=getattr(solver, "_cache_step", 0),
        )


class ParticlesState(BaseModel):
    """Serializable VPM particle state with canonical live-field names."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    position: list[list[float]] = Field()
    velocity: list[list[float]] = Field()
    vortex_strength: list[list[float]] = Field()
    core_radius: list[float] = Field()
    volume: list[float] = Field()
    kinematic_viscosity: list[float] = Field()
    eddy_viscosity: list[float] = Field()
    effective_viscosity: list[float] | None = Field(
        default=None,
    )
    group_id: list[int] = Field()
    velocity_gradient: list[list[list[float]]] | None = Field(
        default=None,
    )
    strain_rate: list[list[list[float]]] | None = None
    vorticity: list[list[float]] | None = Field(
        default=None,
    )
    zone_id: list[int] | None = Field(
        default=None,
    )

    @field_validator("position", "velocity", "vortex_strength")
    @classmethod
    def validate_vector_fields(
        cls,
        values: list[list[float]],
    ) -> list[list[float]]:
        for index, vector in enumerate(values):
            if len(vector) != 3:
                raise ValueError(
                    f"Vector at index {index} must have 3 components, got {len(vector)}"
                )
        return values

    @field_validator("vorticity")
    @classmethod
    def validate_optional_vector_field(
        cls,
        values: list[list[float]] | None,
    ) -> list[list[float]] | None:
        if values is None:
            return None
        for index, vector in enumerate(values):
            if len(vector) != 3:
                raise ValueError(
                    f"Vector at index {index} must have 3 components, got {len(vector)}"
                )
        return values

    @field_validator(
        "core_radius",
        "volume",
        "kinematic_viscosity",
        "eddy_viscosity",
    )
    @classmethod
    def validate_non_negative_scalars(
        cls,
        values: list[float],
    ) -> list[float]:
        for index, value in enumerate(values):
            if value < 0.0:
                raise ValueError(f"Value at index {index} must be non-negative, got {value}")
        return values

    @classmethod
    def from_particles(
        cls,
        particles: Any,
    ) -> ParticlesState:
        """Create a serializable snapshot from the live particle container."""
        state = cls(
            position=particles.position_cpu().tolist(),
            velocity=particles.velocity_cpu().tolist(),
            vortex_strength=(particles.vortex_strength_cpu().tolist()),
            core_radius=particles.core_radius_cpu().tolist(),
            volume=particles.volume_cpu().tolist(),
            kinematic_viscosity=(particles.kinematic_viscosity_cpu().tolist()),
            eddy_viscosity=(particles.eddy_viscosity_cpu().tolist()),
            effective_viscosity=(particles.effective_viscosity_cpu().tolist()),
            group_id=particles.group_id_cpu().tolist(),
            velocity_gradient=(particles.velocity_gradient_cpu().tolist()),
            strain_rate=particles.strain_rate_cpu().tolist(),
            vorticity=particles.vorticity_cpu().tolist(),
            zone_id=particles.zone_id_cpu().tolist(),
        )
        state.validate_consistency()
        return state

    def validate_consistency(self) -> None:
        """Require all particle fields to contain the same particle count."""
        n_particles = len(self.position)
        for name in (
            "velocity",
            "vortex_strength",
            "core_radius",
            "volume",
            "kinematic_viscosity",
            "eddy_viscosity",
            "effective_viscosity",
            "group_id",
            "velocity_gradient",
            "strain_rate",
            "vorticity",
            "zone_id",
        ):
            value = getattr(self, name)
            if value is not None and len(value) != n_particles:
                raise ValueError(f"{name} has {len(value)} entries; expected {n_particles}")


def set_flow_model(
    solver: Any,
    flow_model: str,
) -> None:
    """Set the solver flow model and concise physical description."""
    descriptions = {
        "DNS": "DNS ::: (ω·∇)u + ν∇²ω",
        "LES": "LES ::: (ω·∇)u + (ν+νt)∇²ω",
        "INVISCID": "INV ::: (ω·∇)u",
    }
    solver.flow_model = flow_model
    if flow_model in descriptions:
        solver.flow_model_description = descriptions[flow_model]
