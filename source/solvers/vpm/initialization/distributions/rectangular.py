"""Rectangular particle-distribution construction objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data import ParticleDistribution
from ._common import Bounds3D, centred_coordinates, validate_bounds, validate_spacing


@dataclass(frozen=True, slots=True)
class RectangularDistribution:
    """Configure a Cartesian midpoint particle lattice in a finite box.

    Parameters
    ----------
    bounds : sequence of three (float, float) pairs
        Increasing Cartesian ``((xmin, xmax), (ymin, ymax), (zmin, zmax))``
        bounds in m. Particle locations are centred within nominal cells.
    spacing : float
        Positive nominal lattice spacing ``h`` in m.
    core_radius_ratio : float
        Positive dimensionless ratio ``sigma/h`` used for every particle core.

    Notes
    -----
    This is an immutable construction object; :meth:`build` performs the
    allocation. Returned quadrature volumes are ``h**3`` in m³ even when the
    requested extents are not exact multiples of ``h``.

    Examples
    --------
    >>> grid = RectangularDistribution(
    ...     bounds=((-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5)),
    ...     spacing=0.1,
    ...     core_radius_ratio=2.5,
    ... ).build()
    """

    bounds: Bounds3D
    spacing: float
    core_radius_ratio: float

    def build(self) -> ParticleDistribution:
        """Build immutable lattice geometry and midpoint quadrature.

        Returns
        -------
        ParticleDistribution
            Geometry with positions ``(N, 3)`` in m, core radii ``(N,)`` in
            m, and particle volumes ``(N,)`` equal to ``h**3`` in m³.

        Raises
        ------
        ValueError
            If spacing/core ratio is non-positive or bounds are non-finite or
            non-increasing.
        """
        spacing, ratio = validate_spacing(self.spacing, self.core_radius_ratio)
        limits = validate_bounds(self.bounds)
        coordinates = [centred_coordinates(*limits[axis], spacing) for axis in range(3)]
        position = np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1).reshape(-1, 3)
        return ParticleDistribution(
            position=position,
            core_radius=np.full(len(position), ratio * spacing),
            particle_volume=np.full(len(position), spacing**3),
            spacing=spacing,
        )


@dataclass(frozen=True, slots=True)
class NoisyRectangularDistribution:
    """Configure a reproducibly jittered Cartesian particle lattice.

    Parameters
    ----------
    bounds : sequence of three (float, float) pairs
        Increasing Cartesian domain bounds in m.
    spacing : float
        Positive nominal lattice spacing ``h`` in m.
    core_radius_ratio : float
        Positive dimensionless core ratio ``sigma/h``.
    noise_fraction : float, default=0.3
        Jitter amplitude relative to one nominal cell, in ``[0, 1]``. Each
        coordinate receives a uniform displacement in
        ``[-noise_fraction*h/2, noise_fraction*h/2]``.
    seed : int or None, default=None
        NumPy random seed. Use an integer for deterministic geometry.

    Notes
    -----
    Jittered coordinates are reflected at the bounds rather than clipped,
    avoiding artificial point piles at a face. Core radii and quadrature
    volumes remain those of the unperturbed grid.
    """

    bounds: Bounds3D
    spacing: float
    core_radius_ratio: float
    noise_fraction: float = 0.3
    seed: int | None = None

    def build(self) -> ParticleDistribution:
        """Build bounded, jittered geometry with nominal midpoint weights.

        Returns
        -------
        ParticleDistribution
            Immutable positions ``(N, 3)`` in m, core radii ``(N,)`` in m,
            and volumes ``(N,)`` in m³.

        Raises
        ------
        ValueError
            If ``noise_fraction`` is outside ``[0, 1]`` or the base lattice
            configuration is invalid.
        """
        if not np.isfinite(self.noise_fraction) or not 0.0 <= self.noise_fraction <= 1.0:
            raise ValueError("noise_fraction must be finite and between zero and one")
        base = RectangularDistribution(
            bounds=self.bounds, spacing=self.spacing, core_radius_ratio=self.core_radius_ratio
        ).build()
        limits = validate_bounds(self.bounds)
        position = np.array(base.position, copy=True)
        position += np.random.default_rng(self.seed).uniform(
            -0.5 * self.noise_fraction * base.spacing,
            0.5 * self.noise_fraction * base.spacing,
            size=position.shape,
        )
        lower, upper = limits[:, 0], limits[:, 1]
        position = np.where(position < lower, 2.0 * lower - position, position)
        position = np.where(position > upper, 2.0 * upper - position, position)
        return ParticleDistribution(
            position=position,
            core_radius=base.core_radius,
            particle_volume=base.particle_volume,
            spacing=base.spacing,
        )
