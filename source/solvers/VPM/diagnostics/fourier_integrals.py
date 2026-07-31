from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft


@dataclass(frozen=True)
class FourierIntegrals:
    energy: float
    enstrophy: float
    enstrophy_test: float
    helicity: float
    previous_order_energy: float
    previous_order_enstrophy: float
    previous_order_helicity: float
    radius_expansion_order: int


@dataclass(frozen=True)
class CartesianGrid:
    """Temporary grid used only to audit particle-field integrals."""

    origin: np.ndarray
    spacing: float
    shape: tuple[int, int, int]


def _m4_prime(distance: np.ndarray) -> np.ndarray:
    distance = np.abs(np.asarray(distance))
    weight = np.zeros_like(distance, dtype=np.result_type(distance, np.float64))
    inner = distance <= 1.0
    outer = (distance > 1.0) & (distance <= 2.0)
    weight[inner] = 1.0 - 2.5 * distance[inner] ** 2 + 1.5 * distance[inner] ** 3
    weight[outer] = 0.5 * (2.0 - distance[outer]) ** 2 * (1.0 - distance[outer])
    return weight


def _grid_for_particles(
    positions: np.ndarray,
    spacing: float,
    padding: int = 3,
) -> CartesianGrid:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError("positions must have shape (N, 3) with N > 0")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    lower = np.floor(positions.min(axis=0) / spacing).astype(np.int64) - padding
    upper = np.ceil(positions.max(axis=0) / spacing).astype(np.int64) + padding
    shape = tuple(int(value) for value in upper - lower + 1)
    return CartesianGrid(lower.astype(np.float64) * spacing, float(spacing), shape)


def _scatter_circulation_m4(
    positions: np.ndarray,
    circulation: np.ndarray,
    grid: CartesianGrid,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    circulation = np.asarray(circulation, dtype=np.float64)
    if positions.shape != circulation.shape or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions and circulation must both have shape (N, 3)")

    coordinates = (positions - grid.origin) / grid.spacing
    base = np.floor(coordinates).astype(np.int64)
    fractions = coordinates - base
    offsets = np.arange(-1, 3, dtype=np.int64)
    weights = tuple(
        np.stack([_m4_prime(fractions[:, axis] - offset) for offset in offsets], axis=1)
        for axis in range(3)
    )

    result = np.zeros((*grid.shape, 3), dtype=np.float64)
    flat = result.reshape(-1, 3)
    _, ny, nz = grid.shape
    for oi, di in enumerate(offsets):
        ix = base[:, 0] + di
        for oj, dj in enumerate(offsets):
            iy = base[:, 1] + dj
            wxy = weights[0][:, oi] * weights[1][:, oj]
            for ok, dk in enumerate(offsets):
                iz = base[:, 2] + dk
                valid = (
                    (ix >= 0)
                    & (ix < grid.shape[0])
                    & (iy >= 0)
                    & (iy < grid.shape[1])
                    & (iz >= 0)
                    & (iz < grid.shape[2])
                )
                linear = (ix[valid] * ny + iy[valid]) * nz + iz[valid]
                weight = wxy[valid] * weights[2][valid, ok]
                np.add.at(flat, linear, weight[:, None] * circulation[valid])
    return result


def _wave_numbers(
    shape: tuple[int, int, int],
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies = [
        2.0 * np.pi * fft.fftfreq(shape[0], d=spacing),
        2.0 * np.pi * fft.fftfreq(shape[1], d=spacing),
        2.0 * np.pi * fft.rfftfreq(shape[2], d=spacing),
    ]
    for axis, size in enumerate(shape):
        if size % 2 == 0:
            frequencies[axis][size // 2] = 0.0
    kx = frequencies[0][:, None, None]
    ky = frequencies[1][None, :, None]
    kz = frequencies[2][None, None, :]
    return np.broadcast_arrays(kx, ky, kz)


def gaussian_fourier_integrals(
    position: np.ndarray,
    circulation: np.ndarray,
    radius: np.ndarray,
    volume: np.ndarray,
    *,
    spacing: float | None = None,
    grid: CartesianGrid | None = None,
    radius_expansion_order: int = 3,
) -> FourierIntegrals:
    """Audit Gaussian-blob quadratic integrals on a padded Fourier grid.

    Each particle keeps its own core radius.  For a fixed reference variance
    ``s0`` the exact blob multiplier is expanded as

    ``exp(-sigma_p^2 k^2/4) = exp(-s0 k^2/4)
       sum_n [-(sigma_p^2-s0) k^2/4]^n / n!``.

    The midpoint of the radius-variance range minimizes the largest expansion
    argument and, unlike a circulation-weighted effective radius, is unchanged
    when a relaxation candidate changes particle strengths.  The resulting
    energy is therefore a genuine quadratic form in circulation.  Integrals
    from the penultimate order are returned so transfer convergence can be
    hard-gated without another set of FFTs.
    """

    if radius_expansion_order < 1:
        raise ValueError("radius_expansion_order must be at least one")
    position = np.asarray(position, dtype=np.float64)
    circulation = np.asarray(circulation, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    if position.shape != circulation.shape or position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position and circulation must both have shape (N, 3)")
    if radius.shape != (len(position),) or volume.shape != (len(position),):
        raise ValueError("radius and volume must have shape (N,)")
    if np.any(radius <= 0.0) or np.any(volume <= 0.0):
        raise ValueError("all particle radii and volumes must be positive")
    if grid is not None:
        if spacing is not None and not np.isclose(spacing, grid.spacing):
            raise ValueError("spacing must match the supplied Cartesian grid")
        spacing = grid.spacing
    elif spacing is None:
        spacing = float(np.median(np.cbrt(volume)))
    if spacing <= 0.0:
        raise ValueError("particle spacing and Gaussian radius must be positive")

    if grid is None:
        grid = _grid_for_particles(position, spacing)
    padded_shape = tuple(2 * size for size in grid.shape)
    kx, ky, kz = _wave_numbers(padded_shape, spacing)
    norm_sq = kx * kx + ky * ky + kz * kz
    radius_sq = radius * radius
    reference_variance = 0.5 * (float(radius_sq.min()) + float(radius_sq.max()))
    variance_offset = radius_sq - reference_variance
    reference_gaussian = np.exp(-0.25 * reference_variance * norm_sq)
    transformed = [np.zeros(norm_sq.shape, dtype=np.complex128) for _ in range(3)]
    transformed_previous: list[np.ndarray] | None = None
    factorial = 1
    for order in range(radius_expansion_order + 1):
        if order > 0:
            factorial *= order
        compact = _scatter_circulation_m4(
            position,
            circulation * variance_offset[:, None] ** order,
            grid,
        )
        padding = tuple((size // 2, size - size // 2) for size in compact.shape[:3])
        field = np.pad(compact, (*padding, (0, 0)))
        multiplier = reference_gaussian * (-0.25 * norm_sq) ** order / factorial
        for axis in range(3):
            transformed[axis] += fft.rfftn(field[..., axis], workers=-1) * multiplier
        if order == radius_expansion_order - 1:
            transformed_previous = [component.copy() for component in transformed]
    assert transformed_previous is not None

    multiplicity = np.ones(transformed[0].shape, dtype=np.float64)
    multiplicity[:, :, 1:] = 2.0
    if padded_shape[2] % 2 == 0:
        multiplicity[:, :, -1] = 1.0
    domain_volume = float(np.prod(padded_shape) * spacing**3)
    nonzero = norm_sq > 0.0

    def quadratic_integrals(
        spectrum: list[np.ndarray],
    ) -> tuple[float, float, float, float]:
        cross = (
            ky * spectrum[2] - kz * spectrum[1],
            kz * spectrum[0] - kx * spectrum[2],
            kx * spectrum[1] - ky * spectrum[0],
        )
        energy = sum(
            np.sum(
                multiplicity[nonzero] * np.abs(component[nonzero]) ** 2 / norm_sq[nonzero] ** 2,
                dtype=np.float64,
            )
            for component in cross
        ) / (2.0 * domain_volume)
        enstrophy = (
            sum(
                np.sum(
                    multiplicity * np.abs(component) ** 2,
                    dtype=np.float64,
                )
                for component in spectrum
            )
            / domain_volume
        )
        filter_width = 2.0 * spacing
        test_filter = np.exp(-0.25 * filter_width**2 * norm_sq)
        enstrophy_test = (
            sum(
                np.sum(
                    multiplicity * np.abs(component) ** 2 * test_filter,
                    dtype=np.float64,
                )
                for component in spectrum
            )
            / domain_volume
        )
        velocity = []
        for component in cross:
            value = np.zeros_like(component)
            value[nonzero] = 1j * component[nonzero] / norm_sq[nonzero]
            velocity.append(value)
        helicity = (
            sum(
                np.sum(
                    multiplicity * np.real(velocity[axis] * np.conjugate(spectrum[axis])),
                    dtype=np.float64,
                )
                for axis in range(3)
            )
            / domain_volume
        )
        return (
            float(energy),
            float(enstrophy),
            float(enstrophy_test),
            float(helicity),
        )

    energy, enstrophy, enstrophy_test, helicity = quadratic_integrals(transformed)
    previous_energy, previous_enstrophy, _, previous_helicity = quadratic_integrals(
        transformed_previous
    )
    return FourierIntegrals(
        energy=energy,
        enstrophy=enstrophy,
        enstrophy_test=enstrophy_test,
        helicity=helicity,
        previous_order_energy=previous_energy,
        previous_order_enstrophy=previous_enstrophy,
        previous_order_helicity=previous_helicity,
        radius_expansion_order=radius_expansion_order,
    )
