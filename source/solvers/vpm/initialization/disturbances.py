"""Explicit disturbance specifications for canonical vortex initializers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class WidnallDisturbance:
    """Single-mode or broadband radial disturbance of a vortex-ring centreline.

    ``amplitude`` is nondimensional and relative to the unperturbed ring radius.
    """

    amplitude: float
    mode: int | None = None
    phase: float = 0.0
    number_of_modes: int = 24
    seed: int = 42

    def __post_init__(self) -> None:
        if not np.isfinite(self.amplitude) or self.amplitude < 0.0:
            raise ValueError("amplitude must be finite and non-negative")
        if self.mode is not None and self.mode < 1:
            raise ValueError("mode must be positive")
        if not np.isfinite(self.phase):
            raise ValueError("phase must be finite")
        if self.number_of_modes < 1:
            raise ValueError("number_of_modes must be positive")

    @classmethod
    def single_mode(cls, *, amplitude: float, mode: int, phase: float = 0.0) -> WidnallDisturbance:
        return cls(amplitude=amplitude, mode=mode, phase=phase)

    @classmethod
    def broadband(
        cls, *, amplitude: float, number_of_modes: int = 24, seed: int = 42
    ) -> WidnallDisturbance:
        return cls(amplitude=amplitude, number_of_modes=number_of_modes, seed=seed)

    def centreline(self, azimuth: np.ndarray, ring_radius: float) -> tuple[np.ndarray, np.ndarray]:
        """Return disturbed radius and derivative with respect to azimuth."""
        if self.mode is not None:
            argument = self.mode * azimuth + self.phase
            radius = ring_radius * (1.0 + self.amplitude * np.sin(argument))
            slope = ring_radius * self.amplitude * self.mode * np.cos(argument)
            return radius, slope

        rng = np.random.default_rng(self.seed)
        phases = 2.0 * np.pi * rng.random(self.number_of_modes)
        shape = np.zeros_like(azimuth, dtype=float)
        slope = np.zeros_like(azimuth, dtype=float)
        for mode in range(1, self.number_of_modes + 1):
            argument = mode * azimuth + phases[mode - 1]
            shape += np.cos(argument)
            slope -= mode * np.sin(argument)
        normalization = np.sqrt(self.number_of_modes)
        return (
            ring_radius * (1.0 + self.amplitude * shape / normalization),
            ring_radius * self.amplitude * slope / normalization,
        )


@dataclass(frozen=True, slots=True)
class FilamentDisturbance:
    """Sinusoidal transverse displacement of a straight vortex filament."""

    amplitude: float
    wavelength: float
    phase: float = 0.0
    polarization_angle: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.amplitude) or self.amplitude < 0.0:
            raise ValueError("amplitude must be finite and non-negative")
        if not np.isfinite(self.wavelength) or self.wavelength <= 0.0:
            raise ValueError("wavelength must be finite and positive")
        if not np.isfinite(self.phase) or not np.isfinite(self.polarization_angle):
            raise ValueError("disturbance angles must be finite")
