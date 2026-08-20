"""Filament-refinement configuration for the VPM solver."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FilamentRefinementConfig:
    """Adaptive refinement of stretched Lagrangian vortex-line elements."""

    interval_steps: int = 0
    """Steps between refinement events; zero disables refinement."""

    max_vortex_strength_factor: float = 2.0
    """Refine once ``|alpha_p|`` exceeds this multiple of its lineage reference."""

    offset_fraction: float = 0.25
    """Child offset as a fraction of the estimated material-line length."""

    max_particles: int | None = None
    """Maximum particle population permitted after refinement."""

    def __post_init__(self) -> None:
        if self.interval_steps < 0:
            raise ValueError("filament-refinement interval_steps must be non-negative")
        if self.max_vortex_strength_factor <= 1.0:
            raise ValueError("max_vortex_strength_factor must be greater than one")
        if not 0.0 <= self.offset_fraction <= 0.5:
            raise ValueError("offset_fraction must be in [0, 0.5]")
        if self.max_particles is not None and self.max_particles <= 0:
            raise ValueError("max_particles must be positive or None")

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
        max_particles: int | None = None,
    ) -> "FilamentRefinementConfig":
        """Refine over-stretched particles at the requested step interval."""
        return FilamentRefinementConfig(
            interval_steps=interval_steps,
            max_vortex_strength_factor=max_vortex_strength_factor,
            offset_fraction=offset_fraction,
            max_particles=max_particles,
        )
