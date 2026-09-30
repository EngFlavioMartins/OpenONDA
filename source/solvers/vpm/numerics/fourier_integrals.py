from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft
from scipy.special import erf


@dataclass(frozen=True)
class FourierIntegrals:
    """Quadratic flow-integral diagnostics reconstructed from a Fourier grid.

    Attributes
    ----------
    total_kinetic_energy, previous_order_total_kinetic_energy : float
        ``0.5 * integral(|u|² dV)`` and the value from the penultimate core
        expansion order.  These are mass-normalized kinetic integrals with
        units m⁵/s²; multiply by density for physical kinetic energy.
    total_enstrophy, test_filtered_enstrophy,
    previous_order_total_enstrophy : float
        Vorticity-square integrals in m³/s², with the test-filtered value using
        a filter width of ``2*h``.
    total_helicity, previous_order_total_helicity : float
        ``integral(u dot omega dV)`` in m⁴/s².
    radius_expansion_order : int
        Number of terms used for variable-core Gaussian reconstruction.
    viscous_kinetic_energy_rate : float or None
        Viscous rate associated with the energy integral, in m⁵/s³, or None
        when viscosity was not supplied.
    energy_measurement : str
        ``"periodic_fourier_energy"`` or ``"unbounded_energy"``; the label is
        part of the diagnostic identity and must be considered when comparing
        histories.
    """

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


def _spectral_blocks(shape):
    """Bound quadratic-reduction scratch to 65,536 spectral cells."""
    for i in range(0, shape[0], 16):
        for j in range(0, shape[1], 32):
            for k in range(0, shape[2], 128):
                yield (slice(i, i + 16), slice(j, j + 32), slice(k, k + 128))


def _quadratic_integrals(spectrum, wave_numbers, padded_shape, spacing, viscosity_spectrum=None):
    """Reduce the unchanged Fourier quadratic forms using bounded scratch.

    Full-volume curls, velocities and conjugate products can each exceed
    hundreds of MB for long wakes. Only the spectra need to stay resident.
    The block reduction changes summation order, not the grid or precision.
    """
    multiplicity = np.full(spectrum[0].shape[2], 2.0)
    multiplicity[0] = 1.0
    if padded_shape[2] % 2 == 0:
        multiplicity[-1] = 1.0
    totals = np.zeros(5, dtype=np.float64)
    for block in _spectral_blocks(spectrum[0].shape):
        waves = tuple(component[block] for component in wave_numbers)
        norm_sq = sum(component * component for component in waves)
        inverse = np.divide(1.0, norm_sq, out=np.zeros_like(norm_sq), where=norm_sq > 0.0)
        test_filter = np.exp(-(spacing**2) * norm_sq)
        weight = multiplicity[block[2]][None, None, :]
        values = tuple(component[block] for component in spectrum)
        for axis in range(3):
            squared = np.abs(values[axis]) ** 2
            totals[1] += np.sum(weight * squared, dtype=np.float64)
            totals[2] += np.sum(weight * squared * test_filter, dtype=np.float64)
            first, second = (axis + 1) % 3, (axis + 2) % 3
            velocity = (
                1j * (waves[first] * values[second] - waves[second] * values[first]) * inverse
            )
            totals[0] += np.sum(weight * np.abs(velocity) ** 2, dtype=np.float64)
            totals[3] += np.sum(weight * np.real(velocity * values[axis].conj()), dtype=np.float64)
            if viscosity_spectrum is not None:
                totals[4] -= np.sum(
                    weight * np.real(values[axis] * viscosity_spectrum[axis][block].conj()),
                    dtype=np.float64,
                )
    totals /= float(np.prod(padded_shape) * spacing**3)
    totals[0] *= 0.5
    return tuple(float(value) for value in totals)


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

    Parameters
    ----------
    position, vortex_strength : ndarray, shape (N, 3)
        Particle centres in metres and vector strengths ``Gamma`` in m³/s.
    core_radius, particle_volume : ndarray, shape (N,)
        Particle core radii in metres and quadrature volumes in m³.
    effective_viscosity : ndarray, shape (N,), optional
        Per-particle viscosity in m²/s.  Its absence omits the viscous-rate
        diagnostic.
    spacing : float, optional
        Fourier/M4 grid spacing ``h`` in metres.  If omitted, the median
        cube-root particle volume is used.
    grid : CartesianGrid, optional
        Precomputed audit grid.  When supplied, its spacing takes precedence
        and any explicit ``spacing`` must match.
    radius_expansion_order : int, default=3
        Number of variable-core Gaussian expansion terms used for the primary
        and penultimate-order estimates.
    free_space : bool, default=False
        Use unbounded correlation tensors instead of the periodic spectral
        form.  Requires common core radii and uniform effective viscosity.

    Returns
    -------
    FourierIntegrals
        Immutable energy, enstrophy, helicity, convergence, and viscous-rate
        diagnostics.

    Raises
    ------
    ValueError
        If array shapes, positivity, grid spacing, expansion order, or
        free-space prerequisites are invalid.
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
    previous_integrals = None
    factorial = 1
    for order in range(1 if common_radius else radius_expansion_order + 1):
        if order > 0:
            factorial *= order
        compact = _scatter_vortex_strength_m4(
            position,
            vortex_strength * variance_offset[:, None] ** order,
            grid,
        )
        viscosity_compact = None
        if viscosity_transformed is not None:
            viscosity_compact = _scatter_vortex_strength_m4(
                position,
                vortex_strength * effective_viscosity[:, None] * variance_offset[:, None] ** order,
                grid,
            )
        multiplier = reference_gaussian * (-0.25 * norm_sq) ** order / factorial
        for axis in range(3):
            # Let the FFT pad one component at a time. Centering all three
            # padded components explicitly consumes eight times the compact
            # grid storage. The omitted translation is one common spectral
            # phase, which cancels from every quadratic integral below.
            component = fft.rfftn(compact[..., axis], s=padded_shape, workers=-1)
            component *= multiplier
            transformed[axis] += component
            del component
            if viscosity_transformed is not None and viscosity_compact is not None:
                component = fft.rfftn(viscosity_compact[..., axis], s=padded_shape, workers=-1)
                component *= multiplier
                viscosity_transformed[axis] += component
                del component
        if not common_radius and order == radius_expansion_order - 1:
            # Retain five scalars instead of copying three full complex grids.
            previous_integrals = _quadratic_integrals(
                transformed, (kx, ky, kz), padded_shape, spacing
            )

    del multiplier, reference_gaussian, viscosity_compact, norm_sq
    integrals = _quadratic_integrals(
        transformed, (kx, ky, kz), padded_shape, spacing, viscosity_transformed
    )
    total_kinetic_energy, total_enstrophy, test_filtered_enstrophy, total_helicity, power = (
        integrals
    )
    viscous_kinetic_energy_rate = None
    if uniform_viscosity:
        viscous_kinetic_energy_rate = -float(effective_viscosity[0]) * total_enstrophy
    elif viscosity_transformed is not None:
        viscous_kinetic_energy_rate = power
    if common_radius:
        previous_integrals = integrals
    assert previous_integrals is not None
    (
        previous_order_total_kinetic_energy,
        previous_order_total_enstrophy,
        _,
        previous_order_total_helicity,
        _,
    ) = previous_integrals
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
