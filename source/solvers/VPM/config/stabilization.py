"""Stabilization configuration for the VPM solver.
Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass, field

import numpy as np

from .divergence_relaxation import DivergenceRelaxationConfig
from .filament_refinement import FilamentRefinementConfig


@dataclass(frozen=True)
class StabilizationConfig:
    """Optional numerical stabilization and particle-retention policy.

    ``stretching_viscosity_coefficient`` enables a stretching-aware residual
    viscosity,

    ``nu_stab = C_stab Delta^2 max(Gamma.S.Gamma / |Gamma|^2, 0)``.

    It acts only on locally amplifying vortex-line elements and enters the
    configured viscous diffusion operator through ``nu_eff``.  It therefore
    removes energy through the same auditable mechanism as molecular and SGS
    viscosity; it does not clip or directly modify particle circulation.

    Conservative regularization is a separate, health-triggered LES filter.
    It rebuilds an under-resolved Gaussian cloud while restoring circulation,
    linear impulse, and angular impulse, while forbidding energy or enstrophy
    injection.  Its accepted kinetic-energy loss is reported explicitly as a
    filter transfer.

    Pedrizzetti relaxation is a third, purely local mechanism.  It rotates each
    ``Gamma_p`` toward the vorticity direction reconstructed at that particle,

    ``Gamma <- (1 - f) Gamma + f |Gamma| omega / |omega|``,

    which drives the particle field back toward the vortex-line structure the
    continuum equations imply.  It is a model choice, not a projection: the
    rotation changes vector circulation and both impulses, so it is reported as
    a measured transfer rather than gated as an invariant.

    Filament refinement and divergence relaxation are configured through the
    nested ``filament_refinement`` and ``divergence_relaxation`` entries, so one
    object declares the complete stabilization policy of a run.

    The ``max_*`` limits at the end are the only criteria applied uniformly to
    every mechanism.  They are global, physical, and cheap to evaluate, and the
    stabilization master enforces them on the uploaded field after each event.
    Everything finer belongs to the mechanism that can act on it.
    """

    stretching_viscosity_coefficient: float = 0.0
    """Dimensionless coefficient ``C_stab``; zero disables residual viscosity."""

    pedrizzetti_relaxation_factor: float = 0.0
    """Relaxation factor ``f`` in [0, 1]; zero disables Pedrizzetti relaxation."""

    pedrizzetti_relaxation_frequency: int = 1
    """Steps between relaxation events once the factor is positive."""

    pedrizzetti_relaxation_start_step: int = 0
    """First time step on which Pedrizzetti relaxation may act."""

    pedrizzetti_relaxation_conserve_strength: bool = True
    """Rescale each relaxed vector back to its own ``|Gamma|``."""

    filament_refinement: FilamentRefinementConfig = field(
        default_factory=FilamentRefinementConfig.disabled
    )
    """Adaptive bisection of over-stretched Lagrangian elements."""

    divergence_relaxation: DivergenceRelaxationConfig = field(
        default_factory=DivergenceRelaxationConfig.disabled
    )
    """Moment-constrained Helmholtz projection of the particle strengths."""

    remove_particles_by_bounds: tuple[float, ...] | None = None
    """[xmin, xmax, ymin, ymax, zmin, zmax] — remove particles outside box.  None = disabled."""

    regularization_frequency: int = 0
    """Steps between conservative particle-field regularizations; zero disables it."""

    regularization_start_step: int = 0
    """First time step on which conservative regularization may act."""

    regularization_grid_spacing: float | None = None
    """Spacing of the temporary Gaussian redistribution grid."""

    regularization_tail_budget: float = 3.0e-3
    """Maximum fraction of absolute circulation discarded before moment restoration."""

    regularization_max_particles: int | None = None
    """Maximum number of particles retained by a regularization event."""

    regularization_energy_dissipation_limit: float = 0.15
    """Maximum admissible fractional energy removal in one event."""

    regularization_enstrophy_dissipation_limit: float = 0.15
    """Maximum admissible fractional enstrophy removal in one event."""

    regularization_divergence_trigger: float = 0.04
    """Apply a scheduled regularization above this divergence-health error."""

    regularization_misalignment_trigger: float = 20.0
    """Apply a scheduled regularization above this mean misalignment angle [deg]."""

    regularization_capacity_divergence_trigger: float | None = None
    """Optional higher divergence trigger after the particle budget is reached."""

    regularization_capacity_misalignment_trigger: float | None = None
    """Optional higher misalignment trigger after the particle budget is reached."""

    regularization_capacity_fraction: float = 1.0
    """Fraction of the particle budget at which capacity settings take effect."""

    regularization_capacity_grid_spacing: float | None = None
    """Optional coarser redistribution spacing used in capacity mode."""

    regularization_core_radius: float | None = None
    """Optional regenerated Gaussian core radius, independent of grid spacing."""

    regularization_capacity_core_radius: float | None = None
    """Optional regenerated core radius used only in capacity mode."""

    regularization_projection_trigger: float = 0.08
    """Apply a moment-constrained solenoidal projection above this post-filter error."""

    regularization_projection_max_correction: float = 0.20
    """Largest relative circulation correction admitted by the solenoidal projection."""

    max_circulation_error: float = 1.0e-5
    """Admissible ``|d sum Gamma| / sum |Gamma|`` for a circulation-preserving event."""

    max_strength_growth: float = 1.0e-3
    """Admissible relative growth of ``sum |Gamma|``; removal is unrestricted here."""

    max_vorticity_growth: float = 5.0e-2
    """Admissible relative growth of ``max |omega|``, the instability signature."""

    def __post_init__(self) -> None:
        """Normalize and validate stabilization inputs."""
        if (
            not np.isfinite(self.stretching_viscosity_coefficient)
            or self.stretching_viscosity_coefficient < 0.0
        ):
            raise ValueError("stretching_viscosity_coefficient must be finite and non-negative")
        if (
            not np.isfinite(self.pedrizzetti_relaxation_factor)
            or not 0.0 <= self.pedrizzetti_relaxation_factor <= 1.0
        ):
            raise ValueError("pedrizzetti relaxation factor must be finite and lie in [0, 1]")
        if self.pedrizzetti_relaxation_frequency < 1:
            raise ValueError("pedrizzetti relaxation frequency must be at least one")
        if self.pedrizzetti_relaxation_start_step < 0:
            raise ValueError("pedrizzetti relaxation start step must be non-negative")
        if self.remove_particles_by_bounds is not None:
            object.__setattr__(
                self,
                "remove_particles_by_bounds",
                tuple(float(value) for value in self.remove_particles_by_bounds),
            )
        if (
            self.remove_particles_by_bounds is not None
            and len(self.remove_particles_by_bounds) != 6
        ):
            raise ValueError("remove_particles_by_bounds must have 6 elements")
        if self.regularization_frequency < 0 or self.regularization_start_step < 0:
            raise ValueError("regularization frequency and start step must be non-negative")
        if self.regularization_frequency > 0 and (
            self.regularization_grid_spacing is None or self.regularization_grid_spacing <= 0.0
        ):
            raise ValueError("enabled regularization requires a positive grid spacing")
        if self.regularization_max_particles is not None and self.regularization_max_particles <= 0:
            raise ValueError("regularization_max_particles must be positive or None")
        if not 0.0 < self.regularization_tail_budget < 1.0:
            raise ValueError("regularization tail budget must lie in (0, 1)")
        if not 0.0 < self.regularization_energy_dissipation_limit < 1.0:
            raise ValueError("regularization energy-dissipation limit must lie in (0, 1)")
        if not 0.0 < self.regularization_enstrophy_dissipation_limit < 1.0:
            raise ValueError("regularization enstrophy-dissipation limit must lie in (0, 1)")
        if self.regularization_divergence_trigger < 0.0:
            raise ValueError("regularization divergence trigger must be non-negative")
        if not 0.0 <= self.regularization_misalignment_trigger <= 180.0:
            raise ValueError("regularization misalignment trigger must lie in [0, 180]")
        if (
            self.regularization_capacity_divergence_trigger is not None
            and self.regularization_capacity_divergence_trigger < 0.0
        ):
            raise ValueError("regularization capacity divergence trigger must be non-negative")
        if self.regularization_capacity_misalignment_trigger is not None and not (
            0.0 <= self.regularization_capacity_misalignment_trigger <= 180.0
        ):
            raise ValueError("regularization capacity misalignment trigger must lie in [0, 180]")
        if not 0.0 < self.regularization_capacity_fraction <= 1.0:
            raise ValueError("regularization capacity fraction must lie in (0, 1]")
        if (
            self.regularization_capacity_grid_spacing is not None
            and not self.regularization_capacity_grid_spacing > 0.0
        ):
            raise ValueError("regularization capacity grid spacing must be positive")
        if (
            self.regularization_core_radius is not None
            and not self.regularization_core_radius > 0.0
        ):
            raise ValueError("regularization core radius must be positive")
        if (
            self.regularization_capacity_core_radius is not None
            and not self.regularization_capacity_core_radius > 0.0
        ):
            raise ValueError("regularization capacity core radius must be positive")
        if self.regularization_projection_trigger < 0.0:
            raise ValueError("regularization projection trigger must be non-negative")
        if not 0.0 < self.regularization_projection_max_correction < 1.0:
            raise ValueError("regularization projection correction limit must lie in (0, 1)")
        for name in ("max_circulation_error", "max_strength_growth", "max_vorticity_growth"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def pedrizzetti_relaxation_enabled(self) -> bool:
        return self.pedrizzetti_relaxation_factor > 0.0

    # -- Factory methods -------------------------------------------------------
    @staticmethod
    def disabled() -> "StabilizationConfig":
        """Do not remove particles automatically."""
        return StabilizationConfig()

    @staticmethod
    def bounded_domain(bounds: list[float] | tuple[float, ...]) -> "StabilizationConfig":
        """Remove particles outside one declared six-coordinate domain."""
        return StabilizationConfig(remove_particles_by_bounds=tuple(bounds))

    @staticmethod
    def stretching_viscosity(coefficient: float = 0.5) -> "StabilizationConfig":
        """Add selective core spreading where strain amplifies ``Gamma``.

        The default gives the residual diffusion rate half the local positive
        stretching rate.  Smooth rotation and compressive strain are untouched.
        """
        return StabilizationConfig(stretching_viscosity_coefficient=coefficient)

    @staticmethod
    def pedrizzetti_relaxation(
        *,
        factor: float = 0.3,
        frequency: int = 1,
        start_step: int = 0,
        conserve_strength: bool = True,
    ) -> "StabilizationConfig":
        """Rotate every ``Gamma_p`` toward its local vorticity direction.

        One event moves each strength the fraction ``factor`` of the way to
        ``omega(x_p)``, so a sustained misalignment decays over roughly
        ``1 / factor`` events.  The relaxation is the cheapest of the three
        stabilization mechanisms here — a local rotation, no grid and no solve —
        and the one that acts directly on the misalignment that drives the
        classical three-dimensional vortex-method instability.

        ``conserve_strength`` rescales the relaxed vector back to its original
        ``|Gamma|``, making the update an exact rotation that cannot drain
        particle strength.  Without it the plain convex combination shortens
        ``Gamma_p`` by ``sqrt(1 - 2 f (1 - f)(1 - cos(theta)))``, so the
        relaxation also dissipates enstrophy wherever it acts.
        """

        return StabilizationConfig(
            pedrizzetti_relaxation_factor=factor,
            pedrizzetti_relaxation_frequency=frequency,
            pedrizzetti_relaxation_start_step=start_step,
            pedrizzetti_relaxation_conserve_strength=conserve_strength,
        )

    @staticmethod
    def conservative_filter(
        *,
        coefficient: float = 0.5,
        frequency: int,
        start_step: int,
        grid_spacing: float,
        max_particles: int,
        tail_budget: float = 3.0e-3,
        energy_dissipation_limit: float = 0.15,
        enstrophy_dissipation_limit: float = 0.15,
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
        """Residual viscosity plus moment/enstrophy-constrained redistribution."""

        return StabilizationConfig(
            stretching_viscosity_coefficient=coefficient,
            regularization_frequency=frequency,
            regularization_start_step=start_step,
            regularization_grid_spacing=grid_spacing,
            regularization_max_particles=max_particles,
            regularization_tail_budget=tail_budget,
            regularization_energy_dissipation_limit=energy_dissipation_limit,
            regularization_enstrophy_dissipation_limit=enstrophy_dissipation_limit,
            regularization_divergence_trigger=divergence_trigger,
            regularization_misalignment_trigger=misalignment_trigger,
            regularization_capacity_divergence_trigger=capacity_divergence_trigger,
            regularization_capacity_misalignment_trigger=capacity_misalignment_trigger,
            regularization_capacity_fraction=capacity_fraction,
            regularization_capacity_grid_spacing=capacity_grid_spacing,
            regularization_core_radius=core_radius,
            regularization_capacity_core_radius=capacity_core_radius,
            regularization_projection_trigger=projection_trigger,
            regularization_projection_max_correction=projection_max_correction,
        )


# =========================================================
# VELOCITY CONFIGURATION
# =========================================================
