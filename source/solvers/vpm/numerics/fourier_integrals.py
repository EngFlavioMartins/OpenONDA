from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft
from scipy.special import erf


@dataclass(frozen=True)
class FourierIntegrals:
    total_kinetic_energy: float
    total_enstrophy: float
    test_filtered_enstrophy: float
    total_helicity: float
    previous_order_total_kinetic_energy: float
    previous_order_total_enstrophy: float
    previous_order_total_helicity: float
    radius_expansion_order: int
    viscous_kinetic_energy_rate: float | None
    energy_measurement: str = "periodic_fourier_energy"


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
    position: np.ndarray,
    spacing: float,
    padding: int = 3,
) -> CartesianGrid:
    position = np.asarray(position, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3 or len(position) == 0:
        raise ValueError("position must have shape (N, 3) with N > 0")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    lower = np.floor(position.min(axis=0) / spacing).astype(np.int64) - padding
    upper = np.ceil(position.max(axis=0) / spacing).astype(np.int64) + padding
    shape = tuple(int(value) for value in upper - lower + 1)
    return CartesianGrid(lower.astype(np.float64) * spacing, float(spacing), shape)


def _scatter_vortex_strength_m4(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    grid: CartesianGrid,
) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    vortex_strength = np.asarray(vortex_strength, dtype=np.float64)
    if position.shape != vortex_strength.shape or position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position and vortex_strength must both have shape (N, 3)")

    coordinates = (position - grid.origin) / grid.spacing
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
                np.add.at(flat, linear, weight[:, None] * vortex_strength[valid])
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
    # These are physical wave numbers for Gaussian filtering and quadratic
    # integrals, not a real-grid first-derivative stencil. Zeroing Nyquist here
    # makes high-frequency aliases look like k=0 and leaves them unsmoothed.
    kx = frequencies[0][:, None, None]
    ky = frequencies[1][None, :, None]
    kz = frequencies[2][None, None, :]
    return np.broadcast_arrays(kx, ky, kz)


def _free_space_energy_and_power(compact, spacing, core_radius, viscosity):
    """Linear correlations with the unbounded Gaussian transverse Green tensor.

    Zero padding prevents wraparound of particle pairs. Unlike division by
    discrete k², this includes the far-field energy of a cloud with nonzero
    net strength and is independent of the FFT box size.
    """
    shape = tuple(fft.next_fast_len(2 * n - 1) for n in compact.shape[:3])
    offsets = [fft.fftfreq(n) * n * spacing for n in shape]
    r = np.meshgrid(*offsets, indexing="ij", sparse=True)
    radius_sq = sum(component**2 for component in r)
    sigma = np.sqrt(2.0) * core_radius
    rho_sq = radius_sq / sigma**2
    rho = np.sqrt(rho_sq)
    near = rho < 0.05
    safe_rho = np.where(near, 1.0, rho)
    g = erf(safe_rho) / (4.0 * np.pi * safe_rho)
    zeta = np.exp(-rho_sq) / np.pi**1.5
    aux = ((2.0 * safe_rho**2 - 1.0) * g + 0.5 * zeta) / (4.0 * safe_rho**2)
    isotropic = (g - aux) / sigma
    radial = (-g + 3.0 * aux) / (safe_rho**2 * sigma**3)
    c = 1.0 / np.pi**1.5
    isotropic[near] = c / sigma * (1 / 3 - 2 / 15 * rho_sq[near] + 3 / 70 * rho_sq[near] ** 2)
    radial[near] = c / sigma**3 * (1 / 15 - 1 / 35 * rho_sq[near] + 1 / 126 * rho_sq[near] ** 2)
    q_over_rho3 = (erf(safe_rho) - 2.0 / np.sqrt(np.pi) * safe_rho * np.exp(-(safe_rho**2))) / (
        4.0 * np.pi * safe_rho**3
    )
    diss_iso = (zeta - q_over_rho3) / sigma**3
    diss_radial = (-zeta + 3.0 * q_over_rho3) / (sigma**5 * safe_rho**2)
    diss_iso[near] = c / sigma**3 * (2 / 3 - 4 / 5 * rho_sq[near] + 3 / 7 * rho_sq[near] ** 2)
    diss_radial[near] = c / sigma**5 * (2 / 5 - 2 / 7 * rho_sq[near] + 1 / 9 * rho_sq[near] ** 2)
    active = [axis for axis in range(3) if np.any(compact[..., axis])]
    spectra = {axis: fft.rfftn(compact[..., axis], s=shape, workers=-1) for axis in active}
    energy = power = 0.0
    for i in active:
        for j in active:
            if j < i:
                continue
            correlation = fft.irfftn(spectra[i] * spectra[j].conj(), s=shape, workers=-1)
            tensor = radial * r[i] * r[j]
            diss_tensor = diss_radial * r[i] * r[j]
            if i == j:
                tensor = tensor + isotropic
                diss_tensor = diss_tensor + diss_iso
            multiplicity = 1 if i == j else 2
            energy += 0.5 * multiplicity * float(np.sum(correlation * tensor))
            power -= viscosity * multiplicity * float(np.sum(correlation * diss_tensor))
    return energy, power


def gaussian_fourier_integrals(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
    particle_volume: np.ndarray,
    effective_viscosity: np.ndarray | None = None,
    *,
    spacing: float | None = None,
    grid: CartesianGrid | None = None,
    radius_expansion_order: int = 3,
    free_space: bool = False,
) -> FourierIntegrals:
    """Audit Gaussian-blob quadratic integrals on a padded Fourier grid.

    Each particle keeps its own core radius. For a fixed reference variance
    ``s0`` the exact blob multiplier is expanded as

    ``exp(-sigma_p^2 k^2/4) = exp(-s0 k^2/4)
       sum_n [-(sigma_p^2-s0) k^2/4]^n / n!``.

    The midpoint of the core-radius variance range minimizes the largest expansion
    argument and, unlike a vortex-strength-weighted effective core radius, is unchanged
    when a relaxation candidate changes particle vortex_strength.  The resulting
    energy is therefore a genuine quadratic form in vortex strength. Integrals
    from the penultimate order are returned so transfer convergence can be
    hard-gated without another set of FFTs.

    ``free_space=True`` replaces periodic energy and viscous power with linear
    correlations against the unbounded transverse Gaussian tensors. This mode
    requires a common core radius and uniform viscosity; the default spectral
    quadratic form remains available for variable-core transfer audits.
    """

    if radius_expansion_order < 1:
        raise ValueError("radius_expansion_order must be at least one")
    position = np.asarray(position, dtype=np.float64)
    vortex_strength = np.asarray(vortex_strength, dtype=np.float64)
    core_radius = np.asarray(core_radius, dtype=np.float64)
    particle_volume = np.asarray(particle_volume, dtype=np.float64)
    if effective_viscosity is not None:
        effective_viscosity = np.asarray(effective_viscosity, dtype=np.float64)
    if position.shape != vortex_strength.shape or position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position and vortex_strength must both have shape (N, 3)")
    if core_radius.shape != (len(position),) or particle_volume.shape != (len(position),):
        raise ValueError("core_radius and particle_volume must have shape (N,)")
    if effective_viscosity is not None and effective_viscosity.shape != (len(position),):
        raise ValueError("effective_viscosity must have shape (N,)")
    if np.any(core_radius <= 0.0) or np.any(particle_volume <= 0.0):
        raise ValueError("all particle core_radius and particle_volume values must be positive")
    if effective_viscosity is not None and (
        not np.isfinite(effective_viscosity).all() or np.any(effective_viscosity < 0.0)
    ):
        raise ValueError("all effective_viscosity values must be finite and non-negative")
    if grid is not None:
        if spacing is not None and not np.isclose(spacing, grid.spacing):
            raise ValueError("spacing must match the supplied Cartesian grid")
        spacing = grid.spacing
    elif spacing is None:
        spacing = float(np.median(np.cbrt(particle_volume)))
    if spacing <= 0.0:
        raise ValueError("particle spacing and Gaussian core radius must be positive")

    if grid is None:
        grid = _grid_for_particles(position, spacing)
    padded_shape = tuple(2 * size for size in grid.shape)
    kx, ky, kz = _wave_numbers(padded_shape, spacing)
    norm_sq = kx * kx + ky * ky + kz * kz
    radius_sq = core_radius * core_radius
    reference_variance = 0.5 * (float(radius_sq.min()) + float(radius_sq.max()))
    variance_offset = radius_sq - reference_variance
    common_radius = bool(np.all(variance_offset == 0.0))
    uniform_viscosity = effective_viscosity is not None and bool(
        np.all(effective_viscosity == effective_viscosity[0])
    )
    if free_space and not (common_radius and uniform_viscosity):
        raise ValueError("free-space Fourier integrals require common cores and uniform viscosity")
    reference_gaussian = np.exp(-0.25 * reference_variance * norm_sq)
    transformed = [np.zeros(norm_sq.shape, dtype=np.complex128) for _ in range(3)]
    viscosity_transformed = (
        [np.zeros(norm_sq.shape, dtype=np.complex128) for _ in range(3)]
        if effective_viscosity is not None and not uniform_viscosity
        else None
    )
    transformed_previous: list[np.ndarray] | None = None
    factorial = 1
    for order in range(1 if common_radius else radius_expansion_order + 1):
        if order > 0:
            factorial *= order
        compact = _scatter_vortex_strength_m4(
            position,
            vortex_strength * variance_offset[:, None] ** order,
            grid,
        )
        padding = tuple((size // 2, size - size // 2) for size in compact.shape[:3])
        field = np.pad(compact, (*padding, (0, 0)))
        viscosity_field = None
        if viscosity_transformed is not None:
            viscosity_compact = _scatter_vortex_strength_m4(
                position,
                vortex_strength * effective_viscosity[:, None] * variance_offset[:, None] ** order,
                grid,
            )
            viscosity_field = np.pad(viscosity_compact, (*padding, (0, 0)))
        multiplier = reference_gaussian * (-0.25 * norm_sq) ** order / factorial
        for axis in range(3):
            transformed[axis] += fft.rfftn(field[..., axis], workers=-1) * multiplier
            if viscosity_transformed is not None and viscosity_field is not None:
                viscosity_transformed[axis] += (
                    fft.rfftn(viscosity_field[..., axis], workers=-1) * multiplier
                )
        if common_radius or order == radius_expansion_order - 1:
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
        total_kinetic_energy = sum(
            np.sum(
                multiplicity[nonzero] * np.abs(component[nonzero]) ** 2 / norm_sq[nonzero] ** 2,
                dtype=np.float64,
            )
            for component in cross
        ) / (2.0 * domain_volume)
        total_enstrophy = (
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
        total_helicity = (
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
            float(total_kinetic_energy),
            float(total_enstrophy),
            float(enstrophy_test),
            float(total_helicity),
        )

    total_kinetic_energy, total_enstrophy, test_filtered_enstrophy, total_helicity = (
        quadratic_integrals(transformed)
    )
    viscous_kinetic_energy_rate = None
    if uniform_viscosity:
        viscous_kinetic_energy_rate = -float(effective_viscosity[0]) * total_enstrophy
    if viscosity_transformed is not None:
        viscous_kinetic_energy_rate = -float(
            sum(
                np.sum(
                    multiplicity
                    * np.real(transformed[axis] * np.conjugate(viscosity_transformed[axis])),
                    dtype=np.float64,
                )
                for axis in range(3)
            )
            / domain_volume
        )
    (
        previous_order_total_kinetic_energy,
        previous_order_total_enstrophy,
        _,
        previous_order_total_helicity,
    ) = quadratic_integrals(transformed_previous)
    if free_space:
        total_kinetic_energy, viscous_kinetic_energy_rate = _free_space_energy_and_power(
            compact, spacing, float(core_radius[0]), float(effective_viscosity[0])
        )
        previous_order_total_kinetic_energy = total_kinetic_energy
    return FourierIntegrals(
        total_kinetic_energy=total_kinetic_energy,
        total_enstrophy=total_enstrophy,
        test_filtered_enstrophy=test_filtered_enstrophy,
        total_helicity=total_helicity,
        previous_order_total_kinetic_energy=previous_order_total_kinetic_energy,
        previous_order_total_enstrophy=previous_order_total_enstrophy,
        previous_order_total_helicity=previous_order_total_helicity,
        radius_expansion_order=radius_expansion_order,
        viscous_kinetic_energy_rate=viscous_kinetic_energy_rate,
        energy_measurement="unbounded_energy" if free_space else "periodic_fourier_energy",
    )
