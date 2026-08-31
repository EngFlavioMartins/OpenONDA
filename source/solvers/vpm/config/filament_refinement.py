"""Filament-refinement configuration for the VPM solver."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FilamentRefinementConfig:
    """Adaptive refinement of stretched Lagrangian vortex-line elements."""

    interval_steps: int = 0
    """Steps between refinement events; zero disables refinement."""

    max_vortex_strength_factor: float = 2.0
    """Refine once ``|alpha_p|`` exceeds this multiple of its lineage reference."""

    offset_fraction: float = 0.25
    """Child offset as a fraction of the estimated material-line length."""

    max_n_particles: int | None = None
    """Maximum particle population permitted after refinement."""

    max_absolute_vortex_strength: float | None = None
    """Optional absolute strength threshold for refinement."""

    late_interval_steps: int | None = None
    """Optional shorter cadence after ``late_start_step``."""

    late_start_step: int | None = None
    """First step using ``late_interval_steps``."""

    late_absolute_only: bool = False
    """After ``late_start_step``, refine only at the absolute strength threshold."""

    end_step: int | None = None
    """First step at which refinement is no longer applied."""

    def __post_init__(self) -> None:
        if self.interval_steps < 0:
            raise ValueError("filament-refinement interval_steps must be non-negative")
        if self.max_vortex_strength_factor <= 1.0:
            raise ValueError("max_vortex_strength_factor must be greater than one")
        if not 0.0 <= self.offset_fraction <= 0.5:
            raise ValueError("offset_fraction must be in [0, 0.5]")
        if self.max_n_particles is not None and self.max_n_particles <= 0:
            raise ValueError("max_n_particles must be positive or None")
        if self.max_absolute_vortex_strength is not None and (
            not np.isfinite(self.max_absolute_vortex_strength)
            or self.max_absolute_vortex_strength <= 0.0
        ):
            raise ValueError("max_absolute_vortex_strength must be finite and positive")
        if (self.late_interval_steps is None) != (self.late_start_step is None):
            raise ValueError("late refinement interval and start step must be set together")
        if self.late_interval_steps is not None and self.late_interval_steps <= 0:
            raise ValueError("late_interval_steps must be positive or None")
        if self.late_start_step is not None and self.late_start_step < 0:
            raise ValueError("late_start_step must be non-negative or None")
        if self.late_absolute_only and (
            self.late_start_step is None or self.max_absolute_vortex_strength is None
        ):
            raise ValueError(
                "late_absolute_only requires late_start_step and "
                "max_absolute_vortex_strength"
            )
        if self.end_step is not None and self.end_step < 0:
            raise ValueError("end_step must be non-negative or None")

    @property
    def enabled(self) -> bool:
        """Whether filament refinement is active."""
        return self.interval_steps > 0

    @staticmethod
    def disabled() -> "FilamentRefinementConfig":
        """Return disabled filament refinement."""
        return FilamentRefinementConfig()

    @staticmethod
    def adaptive(
        *,
        interval_steps: int,
        max_vortex_strength_factor: float = 2.0,
        offset_fraction: float = 0.25,
        max_n_particles: int | None = None,
        max_absolute_vortex_strength: float | None = None,
        late_interval_steps: int | None = None,
        late_start_step: int | None = None,
        late_absolute_only: bool = False,
        end_step: int | None = None,
    ) -> "FilamentRefinementConfig":
        """Refine over-stretched particles at the requested step interval."""
        return FilamentRefinementConfig(
            interval_steps=interval_steps,
            max_vortex_strength_factor=max_vortex_strength_factor,
            offset_fraction=offset_fraction,
            max_n_particles=max_n_particles,
            max_absolute_vortex_strength=max_absolute_vortex_strength,
            late_interval_steps=late_interval_steps,
            late_start_step=late_start_step,
            late_absolute_only=late_absolute_only,
            end_step=end_step,
        )
