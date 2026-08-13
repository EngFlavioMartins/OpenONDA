"""Restart/state models and the flow-model validator for the VPM solver.
Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


def CachedParticleProperty(func):
    """
    Decorator for caching expensive particle property calculations.

    Caches results based on the current time step to avoid redundant computations.
    Uses per-property timestamp to ensure independence.
    """
    cache_name = f"_{func.__name__}_cache"
    step_name = f"_{func.__name__}_step"

    def wrapper(self, use_cache: bool = True):
        # Check if cache is valid and should be used
        cache_valid = (
            use_cache
            and hasattr(self, step_name)
            and getattr(self, step_name) == self.time_step
            and hasattr(self, cache_name)
        )

        # Also check global invalidation signal (-1)
        if hasattr(self, "_cached_step") and self._cached_step == -1:
            cache_valid = False

        if not cache_valid:
            result = func(self)
            setattr(self, cache_name, result)
            setattr(self, step_name, self.time_step)

        return getattr(self, cache_name)

    return wrapper


# =========================================================
# PYDANTIC MODELS FOR SERIALIZATION AND STATE MANAGEMENT
# =========================================================
class SolverState(BaseModel):
    """
    Pydantic model for serializing and deserializing solver state.

    This model handles the complete simulation state including parameters,
    timing information, and configuration settings. It supports robust
    backup/restore operations with validation.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    # Core simulation parameters (required for initialization)
    time_step_size: float = Field(gt=0.0, description="Time step size in seconds")
    flow_time: float = Field(ge=0.0, default=0.0, description="Current simulation time")
    time_step: int = Field(ge=0, default=0, description="Current time step number")

    # Method and model configuration
    # Method and model configuration
    # time_integration_scheme: removed in favor of specific schemes
    advection_scheme: str = Field(default="RK3", description="Advection time integration scheme")
    stretching_scheme: str = Field(default="RK3", description="Stretching time integration scheme")

    processing_unit: str = Field(default="AUTO", description="Computation backend")
    flow_model: str = Field(default="DNS", description="Flow physics model")
    viscous_scheme: str = Field(default="CS", description="Viscous modeling scheme")

    # Additional config fields for full reconstruction
    stretching_enabled: bool = Field(default=True, description="Whether stretching is enabled")
    stretching_mode: str = Field(default="TRANSPOSED", description="Stretching mode")
    particles_kernel: str = Field(default="GAUSSIAN", description="Particle kernel function")

    # Simulation control parameters
    backup_file_name: str = Field(default="", description="Optional backup file name infix")
    backup_directory: str = Field(default="solution", description="Output directory for backups")
    logging_frequency: int = Field(default=0, description="Log frequency in steps")
    timing_frequency: int = Field(
        default=0, description="Runtime-profile report frequency in steps"
    )
    backup_frequency: int = Field(default=0, description="Backup frequency in steps")

    # Runtime state (optional, set during execution)
    simulation_time: float | None = Field(default=0.0, ge=0.0, description="Total wall-clock time")
    cached_step: int | None = Field(default=0, description="Last cached computation step")
    E_previous: float | None = Field(
        default=0.0, description="Previous energy for decay calculation"
    )
    E_previous2: float | None = Field(default=0.0, description="Energy from 2 steps ago")

    @field_validator("processing_unit")
    @classmethod
    def validate_processing_unit(cls, v: str) -> str:
        """Validate processing unit."""
        valid_units = {"AUTO", "CPU", "VULKAN", "CUDA", "METAL"}
        v_upper = v.upper()
        if v_upper not in valid_units:
            raise ValueError(f"Invalid processing unit: {v}. Must be one of {valid_units}")
        return v_upper

    @field_validator("flow_model")
    @classmethod
    def validate_flow_model(cls, v: str) -> str:
        """Validate flow model."""
        valid_models = {"DNS", "LES"}
        v_upper = v.upper()
        if v_upper not in valid_models:
            raise ValueError(f"Invalid flow model: {v}. Must be one of {valid_models}")
        return v_upper

    @classmethod
    def from_solver(cls, solver) -> "SolverState":
        """
        Convert solver object to SolverState for serialization.

        Args:
              solver: The solver instance to convert

        Returns:
              SolverState: Serializable state representation

        Raises:
              ValueError: If solver has invalid or missing required attributes

        Examples:
              >>> state = SolverState.from_solver(solver)
              >>> state.time_step_size
              0.01
        """
        try:
            # Extract core attributes, handling missing values gracefully
            solver_dict = {}
            for key, value in solver.__dict__.items():
                # Skip non-serializable attributes and private attributes
                if key in ["particles", "physics", "turbulence", "config", "io"] or key.startswith(
                    "_"
                ):
                    continue

                # Include only serializable types
                if isinstance(value, int | float | str | list | bool | type(None)):
                    solver_dict[key] = value

            return cls(**solver_dict)
        except Exception as e:
            raise ValueError(f"Failed to convert solver to state: {e}") from e

    def to_solver(self):
        """
        Convert SolverState back to solver object.

        Returns:
              Solver: Fully initialized solver instance

        Raises:
              ValueError: If state contains invalid parameters
        """
        try:
            # Import locally to avoid circular dependencies
            from source.solvers.VPM.config.types import (
                AdvectionConfig,
                StretchingConfig,
                TurbulenceConfig,
                ViscousConfig,
                VPMSetup,
            )
            from source.solvers.VPM.core.solver import Solver

            # Reconstruct Configuration Objects
            advection = AdvectionConfig(scheme=self.advection_scheme)

            stretching = StretchingConfig(
                mode=self.stretching_mode,
                scheme=self.stretching_scheme,
                enabled=self.stretching_enabled,
            )

            viscous = ViscousConfig(
                scheme=self.viscous_scheme,
            )

            # Turbulence config reconstruction
            # Same issue for cs, etc.
            turbulence = TurbulenceConfig.dns()
            if self.flow_model == "LES":
                turbulence = TurbulenceConfig.les_smagorinsky()  # Default/Placeholder

            # Reconstruct full VPMSetup
            config = VPMSetup(
                time_step_size=self.time_step_size,
                flow_time=self.flow_time,
                time_step=self.time_step,
                advection=advection,
                stretching=stretching,
                viscous=viscous,
                turbulence=turbulence,
                particles_kernel=self.particles_kernel,
                logging_frequency=self.logging_frequency,
                timing_frequency=self.timing_frequency,
                backup_frequency=self.backup_frequency,
                backup_file_name=self.backup_file_name,
                backup_directory=self.backup_directory,
                processing_unit=self.processing_unit,
                # precision?
            )

            # Create new solver instance with config
            new_solver = Solver(setup=config)

            # Restore additional attributes that aren't constructor parameters
            for key, value in self.__dict__.items():
                if not hasattr(new_solver, key) and not key.startswith("_"):
                    setattr(new_solver, key, value)

            return new_solver

        except Exception as e:
            raise ValueError(f"Failed to create solver from state: {e}") from e


class ParticlesState(BaseModel):
    """
    Pydantic model for serializing complete particle state.

    This model handles all particle data including positions, velocities, strengths,
    and computed fields. It provides robust validation and efficient conversion
    to/from the Particles class.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    # Core particle data (always present)
    positions: list[list[float]] = Field(description="Particle positions [N, 3]")
    velocities: list[list[float]] = Field(description="Particle velocities [N, 3]")
    strengths: list[list[float]] = Field(description="Particle strengths [N, 3]")
    radii: list[float] = Field(description="Particle radii [N]")
    volumes: list[float] = Field(description="Particle volumes [N]")
    viscosities: list[float] = Field(description="Particle molecular viscosities [N]")
    viscosities_t: list[float] = Field(description="Particle turbulent viscosities [N]")
    group_ids: list[int] = Field(description="Particle group identifiers [N]")

    # Optional computed fields (may not always be present)
    grad_u: list[list[list[float]]] | None = Field(
        default=None, description="Velocity gradient tensors [N, 3, 3]"
    )
    vorticities: list[list[float]] | None = Field(
        default=None, description="Particle vorticities [N, 3]"
    )

    @field_validator("positions", "velocities", "strengths")
    @classmethod
    def validate_vector_fields(cls, v: list[list[float]]) -> list[list[float]]:
        """Validate vector fields have consistent 3D structure."""
        if not v:
            return v
        for i, vec in enumerate(v):
            if len(vec) != 3:
                raise ValueError(f"Vector at index {i} must have 3 components, got {len(vec)}")
        return v

    @field_validator("radii", "volumes", "viscosities", "viscosities_t")
    @classmethod
    def validate_positive_scalars(cls, v: list[float]) -> list[float]:
        """Validate scalar fields are positive."""
        for i, val in enumerate(v):
            if val < 0:
                raise ValueError(f"Value at index {i} must be non-negative, got {val}")
        return v

    def validate_consistency(self) -> None:
        """Validate that all fields have consistent sizes."""
        n_particles = len(self.positions)

        # Check all required fields have same length
        fields_to_check = [
            ("velocities", self.velocities),
            ("strengths", self.strengths),
            ("radii", self.radii),
            ("volumes", self.volumes),
            ("viscosities", self.viscosities),
            ("viscosities_t", self.viscosities_t),
            ("group_ids", self.group_ids),
        ]

        for field_name, field_data in fields_to_check:
            if len(field_data) != n_particles:
                raise ValueError(
                    f"Field '{field_name}' has {len(field_data)} elements, "
                    f"expected {n_particles} to match positions"
                )

        # Check optional fields if present
        if self.grad_u is not None and len(self.grad_u) != n_particles:
            raise ValueError(f"grad_u field size mismatch: {len(self.grad_u)} != {n_particles}")
        if self.vorticities is not None and len(self.vorticities) != n_particles:
            raise ValueError(
                f"vorticities field size mismatch: {len(self.vorticities)} != {n_particles}"
            )

    @classmethod
    def from_particles(cls, particles) -> "ParticlesState":
        """
        Convert Particles object to ParticlesState for serialization.

        Args:
              particles: The particles instance to convert

        Returns:
              ParticlesState: Serializable particle state

        Raises:
              ValueError: If particles object is invalid or conversion fails

        Examples:
              >>> state = ParticlesState.from_particles(particles)
              >>> len(state.positions)
              10000
        """
        try:
            # Convert Taichi fields to numpy arrays, then to lists
            data = {
                "positions": particles.positions.to_numpy().tolist(),
                "velocities": particles.velocities.to_numpy().tolist(),
                "strengths": particles.strengths.to_numpy().tolist(),
                "radii": particles.radii.to_numpy().tolist(),
                "volumes": particles.volumes.to_numpy().tolist(),
                "viscosities": particles.viscosities.to_numpy().tolist(),
                "viscosities_t": particles.viscosities_t.to_numpy().tolist(),
                "group_ids": particles.group_ids.to_numpy().tolist(),
                "vorticities": particles.vorticities.to_numpy().tolist(),
            }

            # Add optional fields if they exist and are properly initialized
            if hasattr(particles, "grad_u") and particles.grad_u is not None:
                try:
                    grad_u_array = particles.grad_u.to_numpy()
                    if grad_u_array.size > 0:
                        data["grad_u"] = grad_u_array.tolist()
                except Exception:
                    pass  # Skip if conversion fails

            state = cls(**data)
            state.validate_consistency()
            return state

        except Exception as e:
            raise ValueError(f"Failed to convert particles to state: {e}") from e

    def to_particles(self):
        """
        Convert ParticlesState back to a fully initialized Particles object.

        Returns:
              Particles: Fully initialized particles instance

        Raises:
              ValueError: If state is invalid or conversion fails
        """
        try:
            # Import locally to avoid circular dependencies
            from source.solvers.VPM.particles.container import Particles

            # Validate consistency before conversion
            self.validate_consistency()

            # Determine the number of particles
            n_particles = len(self.positions)
            if n_particles == 0:
                raise ValueError("Cannot create particles from empty state")

            # Create a new Particles object with sufficient capacity
            particles = Particles(max_particles=max(n_particles, 100))

            # Convert all lists to numpy arrays with proper dtypes
            positions = np.array(self.positions, dtype=np.float32)
            velocities = np.array(self.velocities, dtype=np.float32)
            strengths = np.array(self.strengths, dtype=np.float32)
            radii = np.array(self.radii, dtype=np.float32)
            volumes = np.array(self.volumes, dtype=np.float32)
            viscosities = np.array(self.viscosities, dtype=np.float32)
            viscosities_t = np.array(self.viscosities_t, dtype=np.float32)
            group_ids = np.array(self.group_ids, dtype=np.int32)

            # Handle optional fields safely
            grad_u = None
            if self.grad_u is not None:
                grad_u = np.array(self.grad_u, dtype=np.float32)

            # Use add_vortex_particles for robust initialization
            particles.add_vortex_particles(
                positions=positions,
                velocities=velocities,
                strengths=strengths,
                radii=radii,
                volumes=volumes,
                viscosities=viscosities,
                viscosities_t=viscosities_t,
                group_id=group_ids,
                grad_u=grad_u,
            )

            # Ensure particle count is set correctly
            particles.number_of_particles = n_particles

            return particles

        except Exception as e:
            raise ValueError(f"Failed to create particles from state: {e}") from e


# =========================================================
# UTILITY FUNCTIONS FOR FLOW MODEL SETTING
# =========================================================
def SetFlowModel(psys, flow_model: str):
    """
    Set flow model and configure associated parameters.
    Note: Validation is already done in VPMSetup, so this just sets the model.
    """
    if flow_model == "DNS":
        psys.flow_model_description = "DNS ::: (ω.∇)u + (v)(∇²)ω"

    elif flow_model == "LES":
        # LES model description will be set based on smagorinsky type in ParticlesLES
        psys.flow_model_description = "LES ::: (ω.∇)u + (v+vt)(∇²)ω"

    elif flow_model == "INVISCID":
        psys.flow_model_description = "INV ::: (ω.∇)u (stretching only)"

    psys.flow_model = flow_model
