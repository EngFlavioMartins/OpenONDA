"""Divergence-relaxation configuration for the VPM solver.
Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DivergenceRelaxationConfig:
    """Atomic iterated Winckelmans projection with hard physics gates."""

    frequency: int = 0
    start_step: int = 0
    grid_spacing: float | None = None
    regularization: float = 0.1
    solver_rtol: float = 1e-5
    max_iterations: int = 30
    max_projection_sweeps: int = 3
    max_grid_nodes: int = 8_000_000
    max_correction_norm: float = 2e-2
    max_residual_ratio: float = 0.9
    energy_tolerance: float = 1e-6
    enstrophy_tolerance: float = 1e-4
    helicity_tolerance: float = 1e-4
    variation_tolerance: float = 1e-3
    circulation_reference_scale: float | None = None
    linear_impulse_reference_scale: float | None = None
    angular_impulse_reference_scale: float | None = None
    circulation_reference_tolerance: float = 1e-3
    linear_impulse_reference_tolerance: float = 1e-2
    angular_impulse_reference_tolerance: float = 1e-2
    spectral_convergence_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.frequency < 0:
            raise ValueError("divergence-relaxation frequency must be non-negative")
        if self.start_step < 0:
            raise ValueError("divergence-relaxation start_step must be non-negative")
        if self.frequency > 0 and (self.grid_spacing is None or self.grid_spacing <= 0.0):
            raise ValueError("enabled divergence relaxation requires a positive grid_spacing")
        if self.regularization <= 0.0 or self.solver_rtol <= 0.0:
            raise ValueError("regularization and solver_rtol must be positive")
        if self.max_iterations < 1 or self.max_projection_sweeps < 1 or self.max_grid_nodes < 1:
            raise ValueError("iteration, projection-sweep, and grid-node limits must be positive")
        if self.max_correction_norm <= 0.0:
            raise ValueError("max_correction_norm must be positive")
        if not 0.0 < self.max_residual_ratio < 1.0:
            raise ValueError("max_residual_ratio must lie in (0, 1)")
        if not 0.0 < self.spectral_convergence_fraction <= 1.0:
            raise ValueError("spectral_convergence_fraction must lie in (0, 1]")
        for name in (
            "circulation_reference_scale",
            "linear_impulse_reference_scale",
            "angular_impulse_reference_scale",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive when provided")
        reference_scales = (
            self.circulation_reference_scale,
            self.linear_impulse_reference_scale,
            self.angular_impulse_reference_scale,
        )
        if any(value is not None for value in reference_scales) and not all(
            value is not None for value in reference_scales
        ):
            raise ValueError("all three reference scales must be provided together")
        for name in (
            "energy_tolerance",
            "enstrophy_tolerance",
            "helicity_tolerance",
            "variation_tolerance",
            "circulation_reference_tolerance",
            "linear_impulse_reference_tolerance",
            "angular_impulse_reference_tolerance",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def enabled(self) -> bool:
        return self.frequency > 0

    @staticmethod
    def disabled() -> "DivergenceRelaxationConfig":
        return DivergenceRelaxationConfig()

    @staticmethod
    def constrained(
        *,
        frequency: int,
        grid_spacing: float,
        start_step: int = 0,
        regularization: float = 0.1,
        solver_rtol: float = 1e-5,
        max_iterations: int = 30,
        max_projection_sweeps: int = 3,
        max_grid_nodes: int = 8_000_000,
        max_correction_norm: float = 2e-2,
        max_residual_ratio: float = 0.9,
        energy_tolerance: float = 1e-6,
        enstrophy_tolerance: float = 1e-4,
        helicity_tolerance: float = 1e-4,
        variation_tolerance: float = 1e-3,
        circulation_reference_scale: float | None = None,
        linear_impulse_reference_scale: float | None = None,
        angular_impulse_reference_scale: float | None = None,
        circulation_reference_tolerance: float = 1e-3,
        linear_impulse_reference_tolerance: float = 1e-2,
        angular_impulse_reference_tolerance: float = 1e-2,
        spectral_convergence_fraction: float = 0.1,
    ) -> "DivergenceRelaxationConfig":
        return DivergenceRelaxationConfig(
            frequency=frequency,
            start_step=start_step,
            grid_spacing=grid_spacing,
            regularization=regularization,
            solver_rtol=solver_rtol,
            max_iterations=max_iterations,
            max_projection_sweeps=max_projection_sweeps,
            max_grid_nodes=max_grid_nodes,
            max_correction_norm=max_correction_norm,
            max_residual_ratio=max_residual_ratio,
            energy_tolerance=energy_tolerance,
            enstrophy_tolerance=enstrophy_tolerance,
            helicity_tolerance=helicity_tolerance,
            variation_tolerance=variation_tolerance,
            circulation_reference_scale=circulation_reference_scale,
            linear_impulse_reference_scale=linear_impulse_reference_scale,
            angular_impulse_reference_scale=angular_impulse_reference_scale,
            circulation_reference_tolerance=circulation_reference_tolerance,
            linear_impulse_reference_tolerance=linear_impulse_reference_tolerance,
            angular_impulse_reference_tolerance=angular_impulse_reference_tolerance,
            spectral_convergence_fraction=spectral_convergence_fraction,
        )
