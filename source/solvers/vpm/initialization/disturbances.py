"""Explicit disturbance specifications for canonical vortex initializers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class WidnallDisturbance:
    """Configure a single-mode or broadband vortex-ring centreline disturbance.

    Parameters
    ----------
    amplitude : float
        Non-negative dimensionless displacement relative to the unperturbed
        ring radius.
    mode : int or None, default=None
        Positive azimuthal Fourier mode for a single sinusoid. ``None`` builds
        an equal-weight random-phase spectrum over modes 1 through
        ``number_of_modes``.
    phase : float, default=0.0
        Finite single-mode phase in radians; ignored for broadband mode.
    number_of_modes : int, default=24
        Positive broadband mode count; ignored when ``mode`` is set.
    seed : int, default=42
        NumPy seed for deterministic broadband phases.
    direction : {'radial', 'axial'}, default='radial'
        Direction in the local ring frame in which displacement is applied.

    Notes
    -----
    This object defines geometry only. :class:`ToroidalDistribution` and
    :class:`VortexRing` use the same waveform so particle geometry and
    attributed tangent/vorticity remain consistent.
    """

    amplitude: float
    mode: int | None = None
    phase: float = 0.0
    number_of_modes: int = 24
    seed: int = 42
    direction: str = "radial"

    def __post_init__(self) -> None:
        if self.direction not in ("radial", "axial"):
            raise ValueError("direction must be radial or axial")
        if not np.isfinite(self.amplitude) or self.amplitude < 0.0:
            raise ValueError("amplitude must be finite and non-negative")
        if self.mode is not None and self.mode < 1:
            raise ValueError("mode must be positive")
        if not np.isfinite(self.phase):
            raise ValueError("phase must be finite")
        if self.number_of_modes < 1:
            raise ValueError("number_of_modes must be positive")

    @classmethod
    def single_mode(
        cls, *, amplitude: float, mode: int, phase: float = 0.0, direction: str = "radial"
    ) -> WidnallDisturbance:
        """Create one sinusoidal azimuthal ring mode.

        ``amplitude`` is dimensionless, ``mode`` is a positive integer,
        ``phase`` is in radians, and ``direction`` is ``'radial'`` or
        ``'axial'``.
        """
        return cls(amplitude=amplitude, mode=mode, phase=phase, direction=direction)

    @classmethod
    def broadband(
        cls,
        *,
        amplitude: float,
        number_of_modes: int = 24,
        seed: int = 42,
        direction: str = "radial",
    ) -> WidnallDisturbance:
        """Create a reproducible random-phase equal-weight mode spectrum.

        ``amplitude`` is the dimensionless RMS-scale displacement relative to
        ring radius; ``number_of_modes`` is positive and ``seed`` fixes phases.
        """
        return cls(
            amplitude=amplitude, number_of_modes=number_of_modes, seed=seed, direction=direction
        )

    def centreline(self, azimuth: np.ndarray, ring_radius: float) -> tuple[np.ndarray, np.ndarray]:
        """Return radial centreline position and azimuthal derivative.

        Parameters
        ----------
        azimuth : ndarray, shape (N,)
            Angular coordinates in radians.
        ring_radius : float
            Positive unperturbed centreline radius in m.

        Returns
        -------
        radius, derivative : tuple[ndarray, ndarray]
            Arrays of shape ``(N,)`` in m. The derivative is with respect to
            dimensionless azimuth, so it also has units m.
        """
        if self.direction == "axial":
            return np.full_like(azimuth, ring_radius, dtype=float), np.zeros_like(
                azimuth, dtype=float
            )
        displacement, slope = self.displacement(azimuth, ring_radius)
        return ring_radius + displacement, slope

    def axial_centreline(
        self, azimuth: np.ndarray, ring_radius: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return axial centreline displacement and azimuthal derivative.

        Both returned arrays have the shape of ``azimuth`` and units m;
        radial disturbances return zeros.
        """
        if self.direction == "radial":
            return np.zeros_like(azimuth, dtype=float), np.zeros_like(azimuth, dtype=float)
        return self.displacement(azimuth, ring_radius)

    def displacement(
        self, azimuth: np.ndarray, ring_radius: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the configured displacement waveform.

        Parameters
        ----------
        azimuth : ndarray, shape (N,)
            Angular coordinates in radians.
        ring_radius : float
            Reference radius in m that dimensionalizes ``amplitude``.

        Returns
        -------
        displacement, derivative : tuple[ndarray, ndarray]
            Displacement and derivative with respect to azimuth, both shape
            ``(N,)`` in m.
        """
        if self.mode is not None:
            argument = self.mode * azimuth + self.phase
            radius = ring_radius * self.amplitude * np.sin(argument)
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
            ring_radius * self.amplitude * shape / normalization,
            ring_radius * self.amplitude * slope / normalization,
        )


@dataclass(frozen=True, slots=True)
class FilamentDisturbance:
    """Configure sinusoidal transverse displacement of a vortex filament.

    Parameters
    ----------
    amplitude : float
        Non-negative transverse displacement amplitude in m.
    wavelength : float
        Positive axial wavelength in m.
    phase : float, default=0.0
        Finite phase offset in radians.
    polarization_angle : float, default=0.0
        Finite angle in radians selecting the displacement direction in the
        plane normal to the filament.
    """

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
