"""Filament-refinement configuration for the VPM solver."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FilamentRefinementConfig:
    """Configure conservative splitting of stretched vortex-line particles.

    Parameters
    ----------
    interval_steps : int, default=0
        Accepted-step cadence; zero disables refinement.
    max_vortex_strength_factor : float, default=2.0
        Split when ``|Gamma|`` reaches this factor times its stored lineage
        reference; must exceed one.
    offset_fraction : float, default=0.25
        Symmetric child offset divided by estimated material-line length, in
        ``[0, 0.5]``.
    max_absolute_vortex_strength : float or None, optional
        Additional positive ``|Gamma|`` trigger in m³/s.

    Notes
    -----
    Winckelmans (1989), thesis p. 89: bisect along the strength direction at
    x +/- h(t)/4, halve strength and volume, and reset the child references.
    This implements the fixed-core branch (threshold 2); children retain the
    parent's Gaussian radius, group and zone. Array indices are not persistent
    particle identifiers. Splitting preserves moments but changes the resolved
    field; it does not replace viscous core redistribution.
    """

    interval_steps: int = 0
    """Steps between refinement events; zero disables refinement."""

    max_vortex_strength_factor: float = 2.0
    """Refine once ``|Gamma_p|`` exceeds this multiple of its lineage reference."""

    offset_fraction: float = 0.25
    """Child offset as a fraction of the estimated material-line length."""

    max_absolute_vortex_strength: float | None = None
    """Optional absolute strength threshold for refinement."""

    def __post_init__(self) -> None:
        if self.interval_steps < 0:
            raise ValueError("filament-refinement interval_steps must be non-negative")
        if np.isnan(self.max_vortex_strength_factor) or self.max_vortex_strength_factor <= 1.0:
            raise ValueError("max_vortex_strength_factor must be greater than one")
        if not 0.0 <= self.offset_fraction <= 0.5:
            raise ValueError("offset_fraction must be in [0, 0.5]")
        if self.max_absolute_vortex_strength is not None and (
            not np.isfinite(self.max_absolute_vortex_strength)
            or self.max_absolute_vortex_strength <= 0.0
        ):
            raise ValueError("max_absolute_vortex_strength must be finite and positive")

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
        max_absolute_vortex_strength: float | None = None,
    ) -> "FilamentRefinementConfig":
        """Refine over-stretched particles at the requested step interval."""
        return FilamentRefinementConfig(
            interval_steps=interval_steps,
            max_vortex_strength_factor=max_vortex_strength_factor,
            offset_fraction=offset_fraction,
            max_absolute_vortex_strength=max_absolute_vortex_strength,
        )
