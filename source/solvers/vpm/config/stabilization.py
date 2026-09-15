"""Stabilization configuration for the VPM solver."""

from dataclasses import dataclass, field

import numpy as np

from .divergence_relaxation import DivergenceRelaxationConfig
from .filament_refinement import FilamentRefinementConfig


@dataclass(frozen=True)
class StabilizationConfig:
    """Configure accepted-step VPM correction, refinement, and retention policies.

    Parameters
    ----------
    selective_eddy_viscosity_coefficient : float, default=0
        Selective eddy-viscosity coefficient C=2*C_w**2 (Winckelmans 1995,
        Eq. 26, positive-production version, with h=V**(1/3)). Persisted old
        configuration keys are translated by the backup reader.
    selective_eddy_viscosity_start_step : int, default=0
        First accepted step on which selective eddy viscosity may act.
    selective_eddy_viscosity_feedback_gain : float, default=0
        Non-negative feedback gain driven by measured vorticity growth.
    selective_eddy_viscosity_feedback_interval_steps : int, default=5
        Positive feedback-update cadence.
    selective_eddy_viscosity_feedback_growth_limit : float, default=0.25
        Target fractional growth in ``(0, 1]``.
    selective_eddy_viscosity_max_coefficient : float or None, optional
        Optional coefficient ceiling, no smaller than the initial coefficient.
    pedrizzetti_relaxation_factor : float, default=0
        Blend fraction in ``[0, 1]``; zero disables relaxation.
    pedrizzetti_relaxation_interval_steps : int, default=1
        Positive accepted-step cadence.
    pedrizzetti_relaxation_start_step : int, default=0
        First eligible accepted step.
    pedrizzetti_relaxation_end_step : int or None, optional
        Last eligible window boundary; it cannot precede the start step.
    pedrizzetti_relaxation_preserve_vortex_strength : bool, default=True
        Renormalize each relaxed particle to retain its strength magnitude.
        This does not preserve the net vector strength of the particle field.
    pedrizzetti_relaxation_preserve_moments : bool, default=False
        Restore net vector strength and first moments when the correction is
        well conditioned; this can change individual particle magnitudes.
    filament_refinement : FilamentRefinementConfig
        Conservative particle-splitting policy.
    divergence_relaxation : DivergenceRelaxationConfig
        Guarded divergence-projection policy.
    remove_particles_by_bounds : tuple[float, ...] or None, optional
        Cartesian keep-box ``(xmin, xmax, ymin, ymax, zmin, zmax)`` in m.
    regularization_interval_steps, regularization_start_step : int
        Non-negative cadence and first eligible step for conservative remeshing.
    regularization_grid_spacing : float or None
        Positive standard remeshing lattice spacing in m when enabled.
    regularization_tail_budget : float, default=3e-3
        Fractional circulation-magnitude budget in ``(0, 1)`` available to pruning.
    regularization_solenoidal_remesh : bool, default=False
        Project the remeshed lattice toward a divergence-free field.
    regularization_transfer_only : bool, default=False
        Apply Gaussian redistribution and moment restoration only. Disable
        adaptive core broadening, enstrophy adjustment and projection. The
        energy/enstrophy limits then bound absolute transfer errors, allowing
        either sign rather than enforcing dissipation.
    regularization_max_particles, regularization_capacity_max_particles : int or None
        Positive standard and capacity-triggered post-remesh population ceilings.
    regularization_max_events : int or None
        Optional positive lifetime cap on regularization events.
    regularization_total_kinetic_energy_dissipation_limit,
    regularization_total_enstrophy_dissipation_limit : float
        Allowed fractional losses in ``(0, 1)`` for an accepted proposal.
    regularization_divergence_trigger : float or None
        Non-negative normalized divergence trigger.
    regularization_misalignment_trigger : float or None
        Trigger angle in degrees within ``[0, 180]``.
    regularization_core_radius_trigger : float or None
        Positive maximum-core trigger in m.
    regularization_capacity_divergence_trigger : float or None
        Optional non-negative divergence gate for capacity-triggered remeshing.
    regularization_capacity_misalignment_trigger : float or None
        Optional capacity-path angle gate in degrees.
    regularization_capacity_energy_rate_trigger : float or None
        Optional non-negative kinetic-energy rate gate.
    regularization_capacity_fraction : float, default=1
        Active/device-capacity fraction in ``(0, 1]`` that triggers the capacity path.
    regularization_capacity_grid_spacing : float or None
        Optional positive capacity-path lattice spacing in m.
    regularization_core_radius, regularization_capacity_core_radius : float or None
        Optional positive output core radii in m for standard/capacity remeshing.
    regularization_projection_trigger : float, default=0.08
        Non-negative normalized divergence at which projection is attempted.
    regularization_projection_max_correction : float, default=0.20
        Maximum relative projection correction, strictly between 0 and 1.
    max_vortex_strength_error, max_vortex_strength_growth,
    max_vorticity_growth : float
        Non-negative conservation and relative-growth gates applied to proposals.

    Notes
    -----
    All enabled operators run after a physical time step has been accepted.
    They mutate particle fields transactionally: rejected proposals restore the
    pre-event state and do not advance time.
    """

    selective_eddy_viscosity_coefficient: float = 0.0
    selective_eddy_viscosity_start_step: int = 0
    selective_eddy_viscosity_feedback_gain: float = 0.0
    selective_eddy_viscosity_feedback_interval_steps: int = 5
    selective_eddy_viscosity_feedback_growth_limit: float = 0.25
    selective_eddy_viscosity_max_coefficient: float | None = None

    pedrizzetti_relaxation_factor: float = 0.0
    pedrizzetti_relaxation_interval_steps: int = 1
    pedrizzetti_relaxation_start_step: int = 0
    pedrizzetti_relaxation_end_step: int | None = None
    pedrizzetti_relaxation_preserve_vortex_strength: bool = True
    pedrizzetti_relaxation_preserve_moments: bool = False

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
    regularization_solenoidal_remesh: bool = False
    regularization_transfer_only: bool = False
    regularization_preserve_groups: bool = False
    """Remap each group/zone contribution and restore its moments separately.

    Requires transfer-only redistribution; overlapping groups cost additional
    particles. Labels identify vorticity contributions, not post-merger cores.
    """
    regularization_max_particles: int | None = None
    regularization_capacity_max_particles: int | None = None
    regularization_max_events: int | None = None
    regularization_total_kinetic_energy_dissipation_limit: float = 0.15
    regularization_total_enstrophy_dissipation_limit: float = 0.15
    regularization_divergence_trigger: float | None = 0.04
    regularization_misalignment_trigger: float | None = 20.0
    regularization_core_radius_trigger: float | None = None
    regularization_capacity_divergence_trigger: float | None = None
    regularization_capacity_misalignment_trigger: float | None = None
    regularization_capacity_energy_rate_trigger: float | None = None
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
            not np.isfinite(self.selective_eddy_viscosity_coefficient)
            or self.selective_eddy_viscosity_coefficient < 0.0
        ):
            raise ValueError("selective_eddy_viscosity_coefficient must be finite and non-negative")
        if self.selective_eddy_viscosity_start_step < 0:
            raise ValueError("selective_eddy_viscosity_start_step must be non-negative")
        if (
            not np.isfinite(self.selective_eddy_viscosity_feedback_gain)
            or self.selective_eddy_viscosity_feedback_gain < 0.0
        ):
            raise ValueError(
                "selective_eddy_viscosity_feedback_gain must be finite and non-negative"
            )
        if self.selective_eddy_viscosity_feedback_interval_steps < 1:
            raise ValueError(
                "selective_eddy_viscosity_feedback_interval_steps must be at least one"
            )
        if (
            not np.isfinite(self.selective_eddy_viscosity_feedback_growth_limit)
            or not 0.0 < self.selective_eddy_viscosity_feedback_growth_limit <= 1.0
        ):
            raise ValueError("selective_eddy_viscosity_feedback_growth_limit must lie in (0, 1]")
        if self.selective_eddy_viscosity_max_coefficient is not None and (
            not np.isfinite(self.selective_eddy_viscosity_max_coefficient)
            or self.selective_eddy_viscosity_max_coefficient
            < self.selective_eddy_viscosity_coefficient
        ):
            raise ValueError(
                "selective_eddy_viscosity_max_coefficient must be finite and no smaller "
                "than selective_eddy_viscosity_coefficient"
            )
        if (
            not np.isfinite(self.pedrizzetti_relaxation_factor)
            or not 0.0 <= self.pedrizzetti_relaxation_factor <= 1.0
        ):
            raise ValueError("pedrizzetti_relaxation_factor must lie in [0, 1]")
        if self.pedrizzetti_relaxation_interval_steps < 1:
            raise ValueError("pedrizzetti_relaxation_interval_steps must be at least one")
        if self.pedrizzetti_relaxation_start_step < 0:
            raise ValueError("pedrizzetti_relaxation_start_step must be non-negative")
        if self.pedrizzetti_relaxation_end_step is not None and (
            self.pedrizzetti_relaxation_end_step < self.pedrizzetti_relaxation_start_step
        ):
            raise ValueError(
                "pedrizzetti_relaxation_end_step must be no smaller than the start step"
            )

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
        if self.regularization_transfer_only and self.regularization_solenoidal_remesh:
            raise ValueError("transfer-only redistribution cannot enable solenoidal projection")
        if self.regularization_preserve_groups and not self.regularization_transfer_only:
            raise ValueError("group-preserving redistribution requires transfer-only mode")
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
        if (
            self.regularization_capacity_max_particles is not None
            and self.regularization_capacity_max_particles <= 0
        ):
            raise ValueError("regularization_capacity_max_particles must be positive or None")
        if self.regularization_max_events is not None and self.regularization_max_events <= 0:
            raise ValueError("regularization_max_events must be positive or None")
        if not 0.0 < self.regularization_tail_budget < 1.0:
            raise ValueError("regularization_tail_budget must lie in (0, 1)")
        if not 0.0 < self.regularization_total_kinetic_energy_dissipation_limit < 1.0:
            raise ValueError(
                "regularization_total_kinetic_energy_dissipation_limit must lie in (0, 1)"
            )
        if not 0.0 < self.regularization_total_enstrophy_dissipation_limit < 1.0:
            raise ValueError("regularization_total_enstrophy_dissipation_limit must lie in (0, 1)")
        if self.regularization_divergence_trigger is not None and (
            not np.isfinite(self.regularization_divergence_trigger)
            or self.regularization_divergence_trigger < 0.0
        ):
            raise ValueError("regularization_divergence_trigger must be non-negative or None")
        if self.regularization_misalignment_trigger is not None and not (
            0.0 <= self.regularization_misalignment_trigger <= 180.0
        ):
            raise ValueError("regularization_misalignment_trigger must lie in [0, 180] or be None")
        if self.regularization_core_radius_trigger is not None and (
            not np.isfinite(self.regularization_core_radius_trigger)
            or self.regularization_core_radius_trigger <= 0.0
        ):
            raise ValueError("regularization_core_radius_trigger must be finite and positive")
        if (
            self.regularization_capacity_divergence_trigger is not None
            and self.regularization_capacity_divergence_trigger < 0.0
        ):
            raise ValueError("regularization_capacity_divergence_trigger must be non-negative")
        if self.regularization_capacity_misalignment_trigger is not None and not (
            0.0 <= self.regularization_capacity_misalignment_trigger <= 180.0
        ):
            raise ValueError("regularization_capacity_misalignment_trigger must lie in [0, 180]")
        if self.regularization_capacity_energy_rate_trigger is not None and (
            not np.isfinite(self.regularization_capacity_energy_rate_trigger)
            or self.regularization_capacity_energy_rate_trigger < 0.0
        ):
            raise ValueError("regularization_capacity_energy_rate_trigger must be non-negative")
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
        """Whether a non-zero Pedrizzetti strength-alignment blend is configured."""
        return self.pedrizzetti_relaxation_factor > 0.0

    @staticmethod
    def disabled() -> "StabilizationConfig":
        """Return no field-modifying stabilization; retain the solution check."""
        return StabilizationConfig()

    @staticmethod
    def bounded_domain(
        bounds: list[float] | tuple[float, ...],
    ) -> "StabilizationConfig":
        """Remove particles outside one declared domain."""
        return StabilizationConfig(remove_particles_by_bounds=tuple(bounds))

    @staticmethod
    def selective_eddy_viscosity(
        coefficient: float = 0.5,
        start_step: int = 0,
    ) -> "StabilizationConfig":
        """Enable selective eddy viscosity."""
        return StabilizationConfig(
            selective_eddy_viscosity_coefficient=coefficient,
            selective_eddy_viscosity_start_step=start_step,
        )

    @staticmethod
    def pedrizzetti_relaxation(
        *,
        factor: float = 0.3,
        interval_steps: int = 1,
        start_step: int = 0,
        end_step: int | None = None,
        preserve_vortex_strength: bool = True,
        preserve_moments: bool = False,
    ) -> "StabilizationConfig":
        """Enable Pedrizzetti relaxation at a fixed step interval."""
        return StabilizationConfig(
            pedrizzetti_relaxation_factor=factor,
            pedrizzetti_relaxation_interval_steps=interval_steps,
            pedrizzetti_relaxation_start_step=start_step,
            pedrizzetti_relaxation_end_step=end_step,
            pedrizzetti_relaxation_preserve_vortex_strength=(preserve_vortex_strength),
            pedrizzetti_relaxation_preserve_moments=preserve_moments,
        )

    @staticmethod
    def conservative_filter(
        *,
        coefficient: float = 0.5,
        interval_steps: int,
        start_step: int,
        grid_spacing: float,
        max_n_particles: int,
        capacity_max_n_particles: int | None = None,
        max_events: int | None = None,
        tail_budget: float = 3.0e-3,
        solenoidal_remesh: bool = False,
        total_kinetic_energy_dissipation_limit: float = 0.15,
        total_enstrophy_dissipation_limit: float = 0.15,
        divergence_trigger: float | None = 0.04,
        misalignment_trigger: float | None = 20.0,
        core_radius_trigger: float | None = None,
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
            selective_eddy_viscosity_coefficient=coefficient,
            regularization_interval_steps=interval_steps,
            regularization_start_step=start_step,
            regularization_grid_spacing=grid_spacing,
            regularization_max_particles=max_n_particles,
            regularization_capacity_max_particles=capacity_max_n_particles,
            regularization_max_events=max_events,
            regularization_tail_budget=tail_budget,
            regularization_solenoidal_remesh=solenoidal_remesh,
            regularization_total_kinetic_energy_dissipation_limit=(
                total_kinetic_energy_dissipation_limit
            ),
            regularization_total_enstrophy_dissipation_limit=(total_enstrophy_dissipation_limit),
            regularization_divergence_trigger=divergence_trigger,
            regularization_misalignment_trigger=misalignment_trigger,
            regularization_core_radius_trigger=core_radius_trigger,
            regularization_capacity_divergence_trigger=(capacity_divergence_trigger),
            regularization_capacity_misalignment_trigger=(capacity_misalignment_trigger),
            regularization_capacity_fraction=capacity_fraction,
            regularization_capacity_grid_spacing=capacity_grid_spacing,
            regularization_core_radius=core_radius,
            regularization_capacity_core_radius=capacity_core_radius,
            regularization_projection_trigger=projection_trigger,
            regularization_projection_max_correction=(projection_max_correction),
        )
