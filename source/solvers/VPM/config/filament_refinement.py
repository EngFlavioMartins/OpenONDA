"""Filament-refinement configuration for the VPM solver.
Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FilamentRefinementConfig:
    """Adaptive resolution for stretched Lagrangian vortex-line elements.

    At each refinement event, a particle whose strength exceeds
    ``max_strength_factor`` times its own lineage reference is bisected along
    its circulation direction.  Each child starts a new lineage reference.
    The transformation preserves circulation, total strength variation,
    volume, linear impulse, and the Gaussian kernel-corrected angular impulse
    without resetting the viscous core radius.
    """

    frequency: int = 0
    """Steps between refinement events; zero disables refinement."""

    max_strength_factor: float = 2.0
    """Bisect a particle once ``|Gamma|`` exceeds this multiple of its reference."""

    offset_fraction: float = 0.25
    """Child offset as a fraction of the estimated material line length."""

    max_particles: int | None = None
    """Particle budget the refinement may not exceed."""

    def __post_init__(self) -> None:
        if self.frequency < 0:
            raise ValueError("filament-refinement frequency must be non-negative")
        if self.max_strength_factor <= 1.0:
            raise ValueError("max_strength_factor must be greater than one")
        if not 0.0 <= self.offset_fraction <= 0.5:
            raise ValueError("offset_fraction must be in [0, 0.5]")
        if self.max_particles is not None and self.max_particles <= 0:
            raise ValueError("max_particles must be positive or None")

    @property
    def enabled(self) -> bool:
        return self.frequency > 0

    @staticmethod
    def disabled() -> "FilamentRefinementConfig":
        return FilamentRefinementConfig()

    @staticmethod
    def adaptive(
        *,
        frequency: int,
        max_strength_factor: float = 2.0,
        offset_fraction: float = 0.25,
        max_particles: int | None = None,
    ) -> "FilamentRefinementConfig":
        """Bisect over-stretched elements every ``frequency`` steps."""
        return FilamentRefinementConfig(
            frequency=frequency,
            max_strength_factor=max_strength_factor,
            offset_fraction=offset_fraction,
            max_particles=max_particles,
        )
