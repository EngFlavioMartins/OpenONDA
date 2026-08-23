"""Stabilization configuration for the VPM solver."""

from dataclasses import dataclass, field

import numpy as np

from .divergence_relaxation import DivergenceRelaxationConfig
from .filament_refinement import FilamentRefinementConfig


@dataclass(frozen=True)
class StabilizationConfig:
    """Optional VPM stabilization and particle-retention policy."""

    stretching_viscosity_coefficient: float = 0.0

    pedrizzetti_relaxation_factor: float = 0.0
    pedrizzetti_relaxation_interval_steps: int = 1
    pedrizzetti_relaxation_start_step: int = 0
    pedrizzetti_relaxation_preserve_vortex_strength: bool = True

    filament_refinement: FilamentRefinementConfig = field(
        default_factory=FilamentRefinementConfig.disabled
    )
    divergence_relaxation: DivergenceRelaxationConfig = field(
        default_factory=DivergenceRelaxationConfig.disabled
    )

    remove_particles_by_bounds: tuple[float, ...] | None = None

    regularization_interval_steps: int = 0
    regularization_start_step: int = 0
    regularization_grid_spacing: float | None = None
    regularization_tail_budget: float = 3.0e-3
    regularization_max_particles: int | None = None
    regularization_total_kinetic_energy_dissipation_limit: float = 0.15
    regularization_total_enstrophy_dissipation_limit: float = 0.15
    regularization_divergence_trigger: float = 0.04
    regularization_misalignment_trigger: float = 20.0
    regularization_capacity_divergence_trigger: float | None = None
    regularization_capacity_misalignment_trigger: float | None = None
    regularization_capacity_fraction: float = 1.0
    regularization_capacity_grid_spacing: float | None = None
    regularization_core_radius: float | None = None
    regularization_capacity_core_radius: float | None = None
    regularization_projection_trigger: float = 0.08
    regularization_projection_max_correction: float = 0.20

    max_vortex_strength_error: float = 1.0e-5
    max_vortex_strength_growth: float = 1.0e-3
    max_vorticity_growth: float = 5.0e-2

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.stretching_viscosity_coefficient)
            or self.stretching_viscosity_coefficient < 0.0
        ):
            raise ValueError("stretching_viscosity_coefficient must be finite and non-negative")
        if (
            not np.isfinite(self.pedrizzetti_relaxation_factor)
            or not 0.0 <= self.pedrizzetti_relaxation_factor <= 1.0
        ):
            raise ValueError("pedrizzetti_relaxation_factor must lie in [0, 1]")
        if self.pedrizzetti_relaxation_interval_steps < 1:
            raise ValueError("pedrizzetti_relaxation_interval_steps must be at least one")
        if self.pedrizzetti_relaxation_start_step < 0:
            raise ValueError("pedrizzetti_relaxation_start_step must be non-negative")

        if self.remove_particles_by_bounds is not None:
            bounds = tuple(float(value) for value in self.remove_particles_by_bounds)
            if len(bounds) != 6:
                raise ValueError("remove_particles_by_bounds must contain six values")
            object.__setattr__(
                self,
                "remove_particles_by_bounds",
                bounds,
            )

        if self.regularization_interval_steps < 0:
            raise ValueError("regularization_interval_steps must be non-negative")
        if self.regularization_start_step < 0:
            raise ValueError("regularization_start_step must be non-negative")
        if self.regularization_interval_steps > 0 and (
            self.regularization_grid_spacing is None
            or not np.isfinite(self.regularization_grid_spacing)
            or self.regularization_grid_spacing <= 0.0
        ):
            raise ValueError("enabled regularization requires finite positive grid spacing")
        if self.regularization_max_particles is not None and self.regularization_max_particles <= 0:
            raise ValueError("regularization_max_particles must be positive or None")
        if not 0.0 < self.regularization_tail_budget < 1.0:
            raise ValueError("regularization_tail_budget must lie in (0, 1)")
        if not 0.0 < self.regularization_total_kinetic_energy_dissipation_limit < 1.0:
            raise ValueError(
                "regularization_total_kinetic_energy_dissipation_limit must lie in (0, 1)"
            )
        if not 0.0 < self.regularization_total_enstrophy_dissipation_limit < 1.0:
            raise ValueError("regularization_total_enstrophy_dissipation_limit must lie in (0, 1)")
        if (
            not np.isfinite(self.regularization_divergence_trigger)
            or self.regularization_divergence_trigger < 0.0
        ):
            raise ValueError("regularization_divergence_trigger must be non-negative")
        if not 0.0 <= self.regularization_misalignment_trigger <= 180.0:
            raise ValueError("regularization_misalignment_trigger must lie in [0, 180]")
        if (
            self.regularization_capacity_divergence_trigger is not None
            and self.regularization_capacity_divergence_trigger < 0.0
        ):
            raise ValueError("regularization_capacity_divergence_trigger must be non-negative")
        if self.regularization_capacity_misalignment_trigger is not None and not (
            0.0 <= self.regularization_capacity_misalignment_trigger <= 180.0
        ):
            raise ValueError("regularization_capacity_misalignment_trigger must lie in [0, 180]")
        if not 0.0 < self.regularization_capacity_fraction <= 1.0:
            raise ValueError("regularization_capacity_fraction must lie in (0, 1]")
        if self.regularization_capacity_grid_spacing is not None and (
            not np.isfinite(self.regularization_capacity_grid_spacing)
            or self.regularization_capacity_grid_spacing <= 0.0
        ):
            raise ValueError("regularization_capacity_grid_spacing must be finite and positive")
        if self.regularization_core_radius is not None and (
            not np.isfinite(self.regularization_core_radius)
            or self.regularization_core_radius <= 0.0
        ):
            raise ValueError("regularization_core_radius must be finite and positive")
        if self.regularization_capacity_core_radius is not None and (
            not np.isfinite(self.regularization_capacity_core_radius)
            or self.regularization_capacity_core_radius <= 0.0
        ):
            raise ValueError("regularization_capacity_core_radius must be finite and positive")
        if (
            not np.isfinite(self.regularization_projection_trigger)
            or self.regularization_projection_trigger < 0.0
        ):
            raise ValueError("regularization_projection_trigger must be non-negative")
        if not 0.0 < self.regularization_projection_max_correction < 1.0:
            raise ValueError("regularization_projection_max_correction must lie in (0, 1)")

        for name in (
            "max_vortex_strength_error",
            "max_vortex_strength_growth",
            "max_vorticity_growth",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def pedrizzetti_relaxation_enabled(self) -> bool:
        return self.pedrizzetti_relaxation_factor > 0.0

    @staticmethod
    def disabled() -> "StabilizationConfig":
        """Return stabilization with every optional mechanism disabled."""
        return StabilizationConfig()

    @staticmethod
    def bounded_domain(
        bounds: list[float] | tuple[float, ...],
    ) -> "StabilizationConfig":
        """Remove particles outside one declared domain."""
        return StabilizationConfig(remove_particles_by_bounds=tuple(bounds))

    @staticmethod
    def stretching_viscosity(
        coefficient: float = 0.5,
    ) -> "StabilizationConfig":
        """Enable stretching-aware residual viscosity."""
        return StabilizationConfig(stretching_viscosity_coefficient=coefficient)

    @staticmethod
    def pedrizzetti_relaxation(
        *,
        factor: float = 0.3,
        interval_steps: int = 1,
        start_step: int = 0,
        preserve_vortex_strength: bool = True,
    ) -> "StabilizationConfig":
        """Enable Pedrizzetti relaxation at a fixed step interval."""
        return StabilizationConfig(
            pedrizzetti_relaxation_factor=factor,
            pedrizzetti_relaxation_interval_steps=interval_steps,
            pedrizzetti_relaxation_start_step=start_step,
            pedrizzetti_relaxation_preserve_vortex_strength=(preserve_vortex_strength),
        )

    @staticmethod
    def conservative_filter(
        *,
        coefficient: float = 0.5,
        interval_steps: int,
        start_step: int,
        grid_spacing: float,
        max_n_particles: int,
        tail_budget: float = 3.0e-3,
        total_kinetic_energy_dissipation_limit: float = 0.15,
        total_enstrophy_dissipation_limit: float = 0.15,
        divergence_trigger: float = 0.04,
        misalignment_trigger: float = 20.0,
        capacity_divergence_trigger: float | None = None,
        capacity_misalignment_trigger: float | None = None,
        capacity_fraction: float = 1.0,
        capacity_grid_spacing: float | None = None,
        core_radius: float | None = None,
        capacity_core_radius: float | None = None,
        projection_trigger: float = 0.08,
        projection_max_correction: float = 0.20,
    ) -> "StabilizationConfig":
        """Enable residual viscosity plus conservative redistribution."""
        return StabilizationConfig(
            stretching_viscosity_coefficient=coefficient,
            regularization_interval_steps=interval_steps,
            regularization_start_step=start_step,
            regularization_grid_spacing=grid_spacing,
            regularization_max_particles=max_n_particles,
            regularization_tail_budget=tail_budget,
            regularization_total_kinetic_energy_dissipation_limit=(
                total_kinetic_energy_dissipation_limit
            ),
            regularization_total_enstrophy_dissipation_limit=(total_enstrophy_dissipation_limit),
            regularization_divergence_trigger=divergence_trigger,
            regularization_misalignment_trigger=misalignment_trigger,
            regularization_capacity_divergence_trigger=(capacity_divergence_trigger),
            regularization_capacity_misalignment_trigger=(capacity_misalignment_trigger),
            regularization_capacity_fraction=capacity_fraction,
            regularization_capacity_grid_spacing=capacity_grid_spacing,
            regularization_core_radius=core_radius,
            regularization_capacity_core_radius=capacity_core_radius,
            regularization_projection_trigger=projection_trigger,
            regularization_projection_max_correction=(projection_max_correction),
        )
