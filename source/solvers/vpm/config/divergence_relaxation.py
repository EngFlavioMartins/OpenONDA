"""Divergence-relaxation configuration for the VPM solver."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DivergenceRelaxationConfig:
    """Configure the iterated divergence-relaxation projection."""

    interval_steps: int = 0
    start_step: int = 0
    grid_spacing: float | None = None
    regularization: float = 0.1
    solver_relative_tolerance: float = 1e-5
    max_iterations: int = 30
    max_projection_sweeps: int = 3
    max_grid_nodes: int = 8_000_000
    max_correction_norm: float = 2e-2
    max_residual_ratio: float = 0.9
    total_kinetic_energy_tolerance: float = 1e-6
    total_enstrophy_tolerance: float = 1e-4
    total_helicity_tolerance: float = 1e-4
    variation_tolerance: float = 1e-3
    vortex_strength_reference_scale: float | None = None
    linear_impulse_reference_scale: float | None = None
    angular_impulse_reference_scale: float | None = None
    vortex_strength_reference_tolerance: float = 1e-3
    linear_impulse_reference_tolerance: float = 1e-2
    angular_impulse_reference_tolerance: float = 1e-2
    spectral_convergence_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.interval_steps < 0:
            raise ValueError("divergence-relaxation interval_steps must be non-negative")
        if self.start_step < 0:
            raise ValueError("divergence-relaxation start_step must be non-negative")
        if self.interval_steps > 0 and (self.grid_spacing is None or self.grid_spacing <= 0.0):
            raise ValueError("enabled divergence relaxation requires positive grid_spacing")
        if self.regularization <= 0.0:
            raise ValueError("regularization must be positive")
        if self.solver_relative_tolerance <= 0.0:
            raise ValueError("solver_relative_tolerance must be positive")
        if self.max_iterations < 1 or self.max_projection_sweeps < 1 or self.max_grid_nodes < 1:
            raise ValueError("iteration, projection-sweep, and grid-node limits must be positive")
        if self.max_correction_norm <= 0.0:
            raise ValueError("max_correction_norm must be positive")
        if not 0.0 < self.max_residual_ratio < 1.0:
            raise ValueError("max_residual_ratio must lie in (0, 1)")
        if not 0.0 < self.spectral_convergence_fraction <= 1.0:
            raise ValueError("spectral_convergence_fraction must lie in (0, 1]")

        reference_scales = (
            self.vortex_strength_reference_scale,
            self.linear_impulse_reference_scale,
            self.angular_impulse_reference_scale,
        )
        for name, value in (
            ("vortex_strength_reference_scale", reference_scales[0]),
            ("linear_impulse_reference_scale", reference_scales[1]),
            ("angular_impulse_reference_scale", reference_scales[2]),
        ):
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive when provided")

        if any(value is not None for value in reference_scales) and not all(
            value is not None for value in reference_scales
        ):
            raise ValueError("all three reference scales must be provided together")

        for name in (
            "total_kinetic_energy_tolerance",
            "total_enstrophy_tolerance",
            "total_helicity_tolerance",
            "variation_tolerance",
            "vortex_strength_reference_tolerance",
            "linear_impulse_reference_tolerance",
            "angular_impulse_reference_tolerance",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def enabled(self) -> bool:
        """Whether divergence relaxation is active."""
        return self.interval_steps > 0

    @staticmethod
    def disabled() -> "DivergenceRelaxationConfig":
        """Return disabled divergence relaxation."""
        return DivergenceRelaxationConfig()

    @staticmethod
    def constrained(
        *,
        interval_steps: int,
        grid_spacing: float,
        start_step: int = 0,
        regularization: float = 0.1,
        solver_relative_tolerance: float = 1e-5,
        max_iterations: int = 30,
        max_projection_sweeps: int = 3,
        max_grid_nodes: int = 8_000_000,
        max_correction_norm: float = 2e-2,
        max_residual_ratio: float = 0.9,
        total_kinetic_energy_tolerance: float = 1e-6,
        total_enstrophy_tolerance: float = 1e-4,
        total_helicity_tolerance: float = 1e-4,
        variation_tolerance: float = 1e-3,
        vortex_strength_reference_scale: float | None = None,
        linear_impulse_reference_scale: float | None = None,
        angular_impulse_reference_scale: float | None = None,
        vortex_strength_reference_tolerance: float = 1e-3,
        linear_impulse_reference_tolerance: float = 1e-2,
        angular_impulse_reference_tolerance: float = 1e-2,
        spectral_convergence_fraction: float = 0.1,
    ) -> "DivergenceRelaxationConfig":
        """Return an enabled constrained divergence-relaxation setup."""
        return DivergenceRelaxationConfig(
            interval_steps=interval_steps,
            start_step=start_step,
            grid_spacing=grid_spacing,
            regularization=regularization,
            solver_relative_tolerance=solver_relative_tolerance,
            max_iterations=max_iterations,
            max_projection_sweeps=max_projection_sweeps,
            max_grid_nodes=max_grid_nodes,
            max_correction_norm=max_correction_norm,
            max_residual_ratio=max_residual_ratio,
            total_kinetic_energy_tolerance=total_kinetic_energy_tolerance,
            total_enstrophy_tolerance=total_enstrophy_tolerance,
            total_helicity_tolerance=total_helicity_tolerance,
            variation_tolerance=variation_tolerance,
            vortex_strength_reference_scale=vortex_strength_reference_scale,
            linear_impulse_reference_scale=linear_impulse_reference_scale,
            angular_impulse_reference_scale=angular_impulse_reference_scale,
            vortex_strength_reference_tolerance=(vortex_strength_reference_tolerance),
            linear_impulse_reference_tolerance=(linear_impulse_reference_tolerance),
            angular_impulse_reference_tolerance=(angular_impulse_reference_tolerance),
            spectral_convergence_fraction=spectral_convergence_fraction,
        )
