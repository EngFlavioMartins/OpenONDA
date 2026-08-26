"""Conservative circulation transfer and blending on a regular VPM lattice.

This module intentionally contains no particle mutation, viscous diffusion, or
time-advancement logic. It maps integrated vortex strength onto complete M4'
target stencils and forms an absolute FVM/VPM state blend on one common
lattice.
"""

from __future__ import annotations

from dataclasses import dataclass

from numba import njit
import numpy as np


def m4_prime(distance: np.ndarray | float) -> np.ndarray:
    """Return the interpolating M4' kernel for dimensionless distances.

    Its tensor product has compact support over four nodes per axis and
    reproduces zeroth and first moments on a complete, untruncated stencil.
    """
    q = np.abs(np.asarray(distance, dtype=np.float64))
    result = np.zeros_like(q)
    inner = q < 1.0
    outer = (q >= 1.0) & (q < 2.0)
    result[inner] = 1.0 - 2.5 * q[inner] ** 2 + 1.5 * q[inner] ** 3
    result[outer] = 0.5 * (1.0 - q[outer]) * (2.0 - q[outer]) ** 2
    return result


@njit(cache=True, fastmath=False)
def _m4_prime_scalar(distance: float) -> float:
    q = abs(distance)
    if q < 1.0:
        return 1.0 - 2.5 * q * q + 1.5 * q * q * q
    if q < 2.0:
        return 0.5 * (1.0 - q) * (2.0 - q) * (2.0 - q)
    return 0.0


@njit(cache=True, fastmath=False)
def _scatter_complete_m4_prime(
    relative: np.ndarray,
    circulation: np.ndarray,
    lower_index: np.ndarray,
    shape: np.ndarray,
) -> np.ndarray:
    """Scatter on a lattice already known to contain every 4^3 stencil."""
    result = np.zeros((shape[0], shape[1], shape[2], 3), dtype=np.float64)
    for donor in range(len(relative)):
        base_x = int(np.floor(relative[donor, 0])) - 1
        base_y = int(np.floor(relative[donor, 1])) - 1
        base_z = int(np.floor(relative[donor, 2])) - 1
        for ox in range(4):
            ix = base_x + ox
            wx = _m4_prime_scalar(relative[donor, 0] - ix)
            for oy in range(4):
                iy = base_y + oy
                wy = _m4_prime_scalar(relative[donor, 1] - iy)
                for oz in range(4):
                    iz = base_z + oz
                    wz = _m4_prime_scalar(relative[donor, 2] - iz)
                    weight = wx * wy * wz
                    result[ix - lower_index[0], iy - lower_index[1], iz - lower_index[2], 0] += (
                        weight * circulation[donor, 0]
                    )
                    result[ix - lower_index[0], iy - lower_index[1], iz - lower_index[2], 1] += (
                        weight * circulation[donor, 1]
                    )
                    result[ix - lower_index[0], iy - lower_index[1], iz - lower_index[2], 2] += (
                        weight * circulation[donor, 2]
                    )
    return result


@dataclass(frozen=True)
class LatticeTransfer:
    """A complete regular target lattice and its integrated vortex strength."""

    position: np.ndarray
    vortex_strength: np.ndarray
    origin: np.ndarray
    shape: tuple[int, int, int]
    spacing: float
    donor_position: np.ndarray
    donor_vortex_strength: np.ndarray

    @property
    def target_cell_volume(self) -> float:
        return self.spacing**3

    @property
    def donor_gamma_net(self) -> np.ndarray:
        return self.donor_vortex_strength.sum(axis=0, dtype=np.float64)

    @property
    def target_gamma_net(self) -> np.ndarray:
        return self.vortex_strength.sum(axis=0, dtype=np.float64)

    @property
    def donor_first_moment(self) -> np.ndarray:
        return first_vorticity_moment(self.donor_position, self.donor_vortex_strength)

    @property
    def target_first_moment(self) -> np.ndarray:
        return first_vorticity_moment(self.position, self.vortex_strength)


@dataclass(frozen=True)
class RenewalLattice:
    """Fixed lattice covering a renewable particle belt and complete M4' support."""

    position: np.ndarray
    origin: np.ndarray
    shape: tuple[int, int, int]
    spacing: float
    lattice_anchor: np.ndarray
    lower_index: np.ndarray
    renewal_bounds: np.ndarray


@dataclass(frozen=True)
class LatticeStateBlend:
    """Absolute FVM/VPM state on one regular lattice.

    ``vortex_strength`` has units m^3/s. ``vpm_source_mask`` identifies every
    VPM particle represented by the absolute lattice state, including regular
    nodes in its complete outer support guard. ``vpm_replace_mask`` is the
    narrower physical renewal-belt classification used by diagnostics.
    """

    position: np.ndarray
    vortex_strength: np.ndarray
    partitioned_vortex_strength: np.ndarray
    fvm_vortex_strength: np.ndarray
    vpm_vortex_strength: np.ndarray
    eta: np.ndarray
    origin: np.ndarray
    shape: tuple[int, int, int]
    spacing: float
    fvm_donor_vortex_strength_net: np.ndarray
    fvm_donor_first_moment: np.ndarray
    vpm_source_mask: np.ndarray
    vpm_replace_mask: np.ndarray
    hard_replacement: bool
    cross_divergence_l2_before: float
    cross_divergence_l2_after: float
    cross_divergence_relative: float

    @property
    def first_moment(self) -> np.ndarray:
        return first_vorticity_moment(self.position, self.vortex_strength)

    @property
    def gamma_net(self) -> np.ndarray:
        return self.vortex_strength.sum(axis=0, dtype=np.float64)


def state_blend_weight(
    points: np.ndarray,
    box: np.ndarray | list[float] | tuple[float, ...],
    blend_width: float,
) -> np.ndarray:
    """Return the FVM state weight on or inside an ownership box.

    A positive width uses the tensor product of six one-dimensional cosine
    face windows. Unlike a ramp based on the minimum face distance, this is a
    globally C1 Cartesian window: its value and gradient agree on all face,
    edge, and corner bisectors. Zero gives hard ownership.
    """
    position = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    bounds = np.asarray(box, dtype=np.float64).reshape(6)
    width = float(blend_width)
    if not np.all(np.isfinite(bounds)) or np.any(bounds[1::2] <= bounds[::2]):
        raise ValueError("replacement box must contain six finite increasing bounds")
    if not np.isfinite(width) or width < 0.0:
        raise ValueError("eta_blend_width must be finite and non-negative")

    lower_distance = position - bounds[::2]
    upper_distance = bounds[1::2] - position
    inside = np.all((lower_distance >= 0.0) & (upper_distance >= 0.0), axis=1)
    if width == 0.0:
        eta = np.zeros(len(position), dtype=np.float64)
        eta[inside] = 1.0
        return eta

    def face_window(distance: np.ndarray) -> np.ndarray:
        window = np.ones(len(distance), dtype=np.float64)
        ramp = (distance >= 0.0) & (distance < width)
        window[ramp] = 0.5 * (1.0 - np.cos(np.pi * distance[ramp] / width))
        window[distance < 0.0] = 0.0
        return window

    eta = np.ones(len(position), dtype=np.float64)
    for axis in range(3):
        eta *= face_window(lower_distance[:, axis])
        eta *= face_window(upper_distance[:, axis])
    return eta


def release_blend_weight(
    points: np.ndarray,
    box: np.ndarray | list[float] | tuple[float, ...],
    blend_width: float,
) -> np.ndarray:
    """Return an FVM weight that is one in ``box`` and decays outside it.

    This is the state-partition weight for a release overlap.  It is distinct
    from :func:`state_blend_weight`, whose inward ramp classifies particles
    that the FVM-authoritative region may replace.  Separating the two avoids
    deleting released particles merely because M4' support crosses the
    ownership face.
    """
    position = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    bounds = np.asarray(box, dtype=np.float64).reshape(6)
    width = float(blend_width)
    if not np.all(np.isfinite(bounds)) or np.any(bounds[1::2] <= bounds[::2]):
        raise ValueError("replacement box must contain six finite increasing bounds")
    if not np.isfinite(width) or width < 0.0:
        raise ValueError("eta_blend_width must be finite and non-negative")

    lower_excess = np.maximum(bounds[::2] - position, 0.0)
    upper_excess = np.maximum(position - bounds[1::2], 0.0)
    excess = lower_excess + upper_excess
    inside = np.all(excess == 0.0, axis=1)
    if width == 0.0:
        eta = np.zeros(len(position), dtype=np.float64)
        eta[inside] = 1.0
        return eta

    eta = np.ones(len(position), dtype=np.float64)
    for axis in range(3):
        distance = excess[:, axis]
        window = np.zeros(len(position), dtype=np.float64)
        window[distance == 0.0] = 1.0
        ramp = (distance > 0.0) & (distance < width)
        window[ramp] = 0.5 * (1.0 + np.cos(np.pi * distance[ramp] / width))
        eta *= window
    return eta


@njit(cache=True, fastmath=False)
def _evaluate_gaussian_vorticity_direct(
    evaluation_position: np.ndarray,
    particle_position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
) -> np.ndarray:
    """Evaluate the exact untruncated Gaussian vortex field in float64."""
    result = np.zeros((len(evaluation_position), 3), dtype=np.float64)
    normalization = np.pi ** (-1.5)
    for target in range(len(evaluation_position)):
        for particle in range(len(particle_position)):
            dx = evaluation_position[target, 0] - particle_position[particle, 0]
            dy = evaluation_position[target, 1] - particle_position[particle, 1]
            dz = evaluation_position[target, 2] - particle_position[particle, 2]
            sigma = core_radius[particle]
            zeta_over_volume = (
                normalization
                * np.exp(-(dx * dx + dy * dy + dz * dz) / (sigma * sigma))
                / (sigma * sigma * sigma)
            )
            for component in range(3):
                result[target, component] += zeta_over_volume * vortex_strength[particle, component]
    return result


def evaluate_gaussian_vorticity(
    evaluation_position: np.ndarray,
    particle_position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
) -> np.ndarray:
    r"""Evaluate untruncated Gaussian-particle vorticity at arbitrary points.

    This deterministic reference uses OpenONDA's Gaussian convention,
    ``pi**(-3/2) exp(-(r/sigma)**2) / sigma**3``. It is intended for focused
    certification and diagnostics, not the cube's per-step production path.
    """
    targets = np.ascontiguousarray(np.asarray(evaluation_position, dtype=np.float64).reshape(-1, 3))
    position = np.ascontiguousarray(np.asarray(particle_position, dtype=np.float64).reshape(-1, 3))
    strength = np.ascontiguousarray(np.asarray(vortex_strength, dtype=np.float64).reshape(-1, 3))
    radius = np.ascontiguousarray(np.asarray(core_radius, dtype=np.float64).reshape(-1))
    if len(position) != len(strength) or len(position) != len(radius):
        raise ValueError("Gaussian particle position, strength, and core-radius counts must match")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(strength)):
        raise ValueError("Gaussian particle position and strength must be finite")
    if not np.all(np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("Gaussian particle core radii must be finite and positive")
    if not np.all(np.isfinite(targets)):
        raise ValueError("Gaussian evaluation positions must be finite")
    if not len(position):
        return np.zeros((len(targets), 3), dtype=np.float64)
    return _evaluate_gaussian_vorticity_direct(targets, position, strength, radius)


def _spectral_wave_numbers(
    shape: tuple[int, int, int], spacing: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies = [2.0 * np.pi * np.fft.fftfreq(size, d=spacing) for size in shape]
    frequencies[2] = 2.0 * np.pi * np.fft.rfftfreq(shape[2], d=spacing)
    for axis, size in enumerate(shape):
        if size % 2 == 0:
            frequencies[axis][size // 2] = 0.0
    kx, ky, kz = np.broadcast_arrays(
        frequencies[0][:, None, None],
        frequencies[1][None, :, None],
        frequencies[2][None, None, :],
    )
    return kx, ky, kz


def spectral_lattice_divergence(field: np.ndarray, *, spacing: float) -> np.ndarray:
    """Return the periodic spectral divergence used by blend correction.

    This is deliberately the single discrete operator used to form the blend
    residual, solve its Poisson equation, and report the remaining defect.
    """
    vector = np.asarray(field, dtype=np.float64)
    if vector.ndim != 4 or vector.shape[-1] != 3:
        raise ValueError("field must have shape (nx, ny, nz, 3)")
    h = float(spacing)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("spacing must be finite and positive")
    wave_numbers = _spectral_wave_numbers(vector.shape[:3], h)
    divergence_hat = sum(
        1j * wave_numbers[component] * np.fft.rfftn(vector[..., component])
        for component in range(3)
    )
    return np.fft.irfftn(divergence_hat, s=vector.shape[:3], axes=(0, 1, 2)).real


def correct_state_blend_cross_divergence(
    fvm_vortex_strength: np.ndarray,
    vpm_vortex_strength: np.ndarray,
    eta: np.ndarray,
    *,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Remove only the divergence introduced by a variable state weight.

    The correction solves a periodic lattice Poisson equation for the
    mean-free difference between the blended divergence and the same weighted
    source divergences. Its gradient has zero net circulation. If the two
    source states match, the correction is exactly zero.
    """
    fvm = np.asarray(fvm_vortex_strength, dtype=np.float64)
    vpm = np.asarray(vpm_vortex_strength, dtype=np.float64)
    weight = np.asarray(eta, dtype=np.float64)
    if fvm.shape != vpm.shape or fvm.ndim != 4 or fvm.shape[-1] != 3:
        raise ValueError("FVM and VPM lattice fields must both have shape (nx, ny, nz, 3)")
    if weight.shape != fvm.shape[:3]:
        raise ValueError("eta must match the lattice shape")
    h = float(spacing)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("spacing must be finite and positive")

    if np.array_equal(fvm, vpm):
        return fvm.copy(), fvm.copy(), 0.0, 0.0

    partitioned = weight[..., None] * fvm + (1.0 - weight[..., None]) * vpm
    fvm_divergence = spectral_lattice_divergence(fvm, spacing=h)
    vpm_divergence = spectral_lattice_divergence(vpm, spacing=h)
    cross_divergence = spectral_lattice_divergence(partitioned, spacing=h) - (
        weight * fvm_divergence + (1.0 - weight) * vpm_divergence
    )
    cross_mean = float(np.mean(cross_divergence))
    removable = cross_divergence - cross_mean
    before = float(np.sqrt(np.mean(cross_divergence**2)))
    if before == 0.0:
        return partitioned, partitioned.copy(), 0.0, 0.0

    shape = fvm.shape[:3]
    wave_numbers = _spectral_wave_numbers(shape, h)
    wave_number_sq = np.zeros_like(wave_numbers[0])
    for component in wave_numbers:
        wave_number_sq = wave_number_sq + component * component
    source_hat = np.fft.rfftn(removable)
    potential_hat = np.zeros_like(source_hat)
    nonzero = wave_number_sq > 0.0
    potential_hat[nonzero] = -source_hat[nonzero] / wave_number_sq[nonzero]
    correction = np.stack(
        [
            np.fft.irfftn(
                1j * wave_numbers[component] * potential_hat,
                s=shape,
                axes=(0, 1, 2),
            ).real
            for component in range(3)
        ],
        axis=-1,
    )
    corrected = partitioned - correction
    corrected_divergence = spectral_lattice_divergence(corrected, spacing=h)
    residual = corrected_divergence - (weight * fvm_divergence + (1.0 - weight) * vpm_divergence)
    after = float(np.sqrt(np.mean(residual**2)))
    net_error = partitioned.sum(axis=(0, 1, 2)) - corrected.sum(axis=(0, 1, 2))
    corrected += net_error / float(np.prod(shape))
    return partitioned, corrected, before, after


def _validate_donors(
    position: np.ndarray,
    cell_volume: np.ndarray,
    vorticity: np.ndarray,
    solid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    donor_position = np.asarray(position, dtype=np.float64).reshape(-1, 3)
    volume = np.asarray(cell_volume, dtype=np.float64).reshape(-1)
    donor_vorticity = np.asarray(vorticity, dtype=np.float64).reshape(-1, 3)
    if len(donor_position) != len(volume) or len(donor_position) != len(donor_vorticity):
        raise ValueError("donor position, volume, and vorticity counts must match")
    if not np.all(np.isfinite(donor_position)) or not np.all(np.isfinite(donor_vorticity)):
        raise ValueError("donor position and vorticity must be finite")
    if not np.all(np.isfinite(volume)) or np.any(volume <= 0.0):
        raise ValueError("donor volumes must be finite and positive")
    keep = np.ones(len(donor_position), dtype=bool)
    if solid_mask is not None:
        solid = np.asarray(solid_mask, dtype=bool).reshape(-1)
        if len(solid) != len(keep):
            raise ValueError("solid_mask must match the donor count")
        keep &= ~solid
    return donor_position[keep], (volume[keep, None] * donor_vorticity[keep])


def _validate_vortex_strength(
    position: np.ndarray,
    vortex_strength: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_position = np.asarray(position, dtype=np.float64).reshape(-1, 3)
    source_strength = np.asarray(vortex_strength, dtype=np.float64).reshape(-1, 3)
    if len(source_position) != len(source_strength):
        raise ValueError("position and vortex_strength counts must match")
    if not np.all(np.isfinite(source_position)) or not np.all(np.isfinite(source_strength)):
        raise ValueError("position and vortex_strength must be finite")
    return source_position, source_strength


def _map_integrated_vortex_strength_to_lattice(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    *,
    lattice_anchor: np.ndarray,
    spacing: float,
) -> LatticeTransfer:
    h = float(spacing)
    anchor = np.asarray(lattice_anchor, dtype=np.float64).reshape(3)
    if not np.isfinite(h) or h <= 0.0 or not np.all(np.isfinite(anchor)):
        raise ValueError("lattice anchor and spacing must be finite, with positive spacing")
    donor_position, donor_strength = _validate_vortex_strength(position, vortex_strength)
    if len(donor_position) == 0:
        return LatticeTransfer(
            position=np.empty((0, 3), dtype=np.float64),
            vortex_strength=np.empty((0, 3), dtype=np.float64),
            origin=anchor.copy(),
            shape=(0, 0, 0),
            spacing=h,
            donor_position=donor_position,
            donor_vortex_strength=donor_strength,
        )

    relative = (donor_position - anchor) / h
    base = np.floor(relative).astype(np.int64) - 1
    lower = base.min(axis=0)
    upper = (base + 3).max(axis=0)
    shape_array = upper - lower + 1
    field = _scatter_complete_m4_prime(
        np.ascontiguousarray(relative),
        np.ascontiguousarray(donor_strength),
        np.ascontiguousarray(lower),
        np.ascontiguousarray(shape_array),
    )
    axes = tuple(anchor[axis] + h * np.arange(lower[axis], upper[axis] + 1) for axis in range(3))
    mesh = np.meshgrid(*axes, indexing="ij")
    target_position = np.column_stack([axis.ravel() for axis in mesh])
    return LatticeTransfer(
        position=target_position,
        vortex_strength=field.reshape(-1, 3),
        origin=anchor + h * lower,
        shape=(int(shape_array[0]), int(shape_array[1]), int(shape_array[2])),
        spacing=h,
        donor_position=donor_position,
        donor_vortex_strength=donor_strength,
    )


def build_renewal_lattice(
    renewal_bounds: np.ndarray | list[float] | tuple[float, ...],
    *,
    lattice_anchor: np.ndarray,
    spacing: float,
) -> RenewalLattice:
    """Build a fixed lattice containing every M4' stencil from a renewal belt.

    ``renewal_bounds`` classifies the VPM particles whose state is renewed.
    The returned lattice extends by the complete four-node M4' stencil, so no
    source inside that belt is clipped or renormalised at its boundary.
    """
    bounds = np.asarray(renewal_bounds, dtype=np.float64).reshape(6)
    anchor = np.asarray(lattice_anchor, dtype=np.float64).reshape(3)
    h = float(spacing)
    if not np.all(np.isfinite(bounds)) or np.any(bounds[1::2] <= bounds[::2]):
        raise ValueError("renewal_bounds must contain six finite increasing bounds")
    if not np.all(np.isfinite(anchor)) or not np.isfinite(h) or h <= 0.0:
        raise ValueError("lattice anchor and spacing must be finite, with positive spacing")

    lower_relative = (bounds[::2] - anchor) / h
    upper_relative = (bounds[1::2] - anchor) / h
    # M4' uses floor(q)-1 ... floor(q)+2.  These limits cover every phase in
    # the closed renewal belt, including sources exactly on a belt face.
    lower_index = np.floor(lower_relative).astype(np.int64) - 1
    upper_index = np.floor(upper_relative).astype(np.int64) + 2
    shape_array = upper_index - lower_index + 1
    axes = tuple(
        anchor[axis] + h * np.arange(lower_index[axis], upper_index[axis] + 1) for axis in range(3)
    )
    mesh = np.meshgrid(*axes, indexing="ij")
    position = np.column_stack([component.ravel() for component in mesh])
    return RenewalLattice(
        position=position,
        origin=anchor + h * lower_index,
        shape=(int(shape_array[0]), int(shape_array[1]), int(shape_array[2])),
        spacing=h,
        lattice_anchor=anchor,
        lower_index=lower_index,
        renewal_bounds=bounds,
    )


def scatter_vortex_strength_to_renewal_lattice(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    lattice: RenewalLattice,
    *,
    position_dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Scatter integrated vortex strength to one fixed complete M4' lattice."""
    source_position, source_strength = _validate_vortex_strength(position, vortex_strength)
    if len(source_position) == 0:
        return np.zeros((*lattice.shape, 3), dtype=np.float64)

    scale = np.maximum(1.0, np.abs(source_position))
    dtype = np.dtype(position_dtype)
    if not np.issubdtype(dtype, np.floating):
        raise ValueError("position_dtype must be floating point")
    coordinate_tolerance = 16.0 * np.finfo(dtype).eps * scale
    relative = (source_position - lattice.lattice_anchor) / lattice.spacing
    nearest = np.rint(relative).astype(np.int64)
    reconstructed = lattice.lattice_anchor + lattice.spacing * nearest
    regular = np.all(np.abs(source_position - reconstructed) <= coordinate_tolerance, axis=1)
    bounds = lattice.renewal_bounds
    inside = np.all(
        (source_position >= bounds[::2] - coordinate_tolerance)
        & (source_position <= bounds[1::2] + coordinate_tolerance),
        axis=1,
    )
    # The lattice has a complete outer M4' guard.  A regular node deposited
    # into that guard is an exact identity source and may be reconciled there;
    # genuinely off-lattice sources still require the physical renewal belt.
    if np.any(~inside & ~regular):
        raise ValueError("off-lattice renewal sources must lie inside renewal_bounds")

    base = np.floor(relative).astype(np.int64) - 1
    upper = base + 3
    lattice_upper = lattice.lower_index + np.asarray(lattice.shape, dtype=np.int64) - 1
    if np.any(nearest[regular] < lattice.lower_index) or np.any(nearest[regular] > lattice_upper):
        raise RuntimeError("regular renewal source lies outside the fixed lattice")
    if np.any(base[~regular] < lattice.lower_index) or np.any(upper[~regular] > lattice_upper):
        raise RuntimeError("renewal lattice does not contain a complete M4' source stencil")

    # GBD/DVH and a just-renewed cloud already lie on this lattice.  M4' is
    # exactly the identity at integer phase, so embed those nodes directly and
    # scatter only genuinely off-lattice particles.
    field = np.zeros((*lattice.shape, 3), dtype=np.float64)
    if np.any(regular):
        local = nearest[regular] - lattice.lower_index
        np.add.at(field, (local[:, 0], local[:, 1], local[:, 2]), source_strength[regular])
    if np.any(~regular):
        field += _scatter_complete_m4_prime(
            np.ascontiguousarray(relative[~regular]),
            np.ascontiguousarray(source_strength[~regular]),
            np.ascontiguousarray(lattice.lower_index),
            np.ascontiguousarray(np.asarray(lattice.shape, dtype=np.int64)),
        )
    return field


def map_vortex_strength_to_lattice(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    *,
    lattice_anchor: np.ndarray,
    spacing: float,
) -> LatticeTransfer:
    """Scatter particle vortex strength onto a complete M4' lattice."""
    return _map_integrated_vortex_strength_to_lattice(
        position,
        vortex_strength,
        lattice_anchor=lattice_anchor,
        spacing=spacing,
    )


def map_cell_circulation_to_lattice(
    position: np.ndarray,
    cell_volume: np.ndarray,
    vorticity: np.ndarray,
    *,
    lattice_anchor: np.ndarray,
    spacing: float,
    solid_mask: np.ndarray | None = None,
) -> LatticeTransfer:
    """Conservatively scatter FVM-cell circulation to a complete M4' lattice.

    The returned target bounds are inferred from all donor stencils, never
    clipped to an ownership box.  Consequently each donor has all 64 target
    nodes required by M4' and no zeroth- or first-moment boundary repair is
    required.
    """
    donor_position, donor_strength = _validate_donors(position, cell_volume, vorticity, solid_mask)
    return _map_integrated_vortex_strength_to_lattice(
        donor_position,
        donor_strength,
        lattice_anchor=lattice_anchor,
        spacing=spacing,
    )


def _common_lattice_fields(
    first: LatticeTransfer,
    second: LatticeTransfer,
    *,
    lattice_anchor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int], np.ndarray]:
    """Embed two aligned lattice fields in their smallest common box."""
    nonempty = [field for field in (first, second) if len(field.position)]
    anchor = np.asarray(lattice_anchor, dtype=np.float64).reshape(3)
    if not nonempty:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, empty.copy(), empty.copy(), (0, 0, 0), anchor.copy()

    h = float(nonempty[0].spacing)
    lower = np.min(
        [np.rint((field.origin - anchor) / h).astype(np.int64) for field in nonempty], axis=0
    )
    upper = np.max(
        [
            np.rint((field.origin - anchor) / h).astype(np.int64)
            + np.asarray(field.shape, dtype=np.int64)
            - 1
            for field in nonempty
        ],
        axis=0,
    )
    shape_array = upper - lower + 1
    fields = []
    for source in (first, second):
        target = np.zeros((*shape_array, 3), dtype=np.float64)
        if len(source.position):
            source_lower = np.rint((source.origin - anchor) / h).astype(np.int64)
            offset = source_lower - lower
            slices = tuple(
                slice(int(offset[axis]), int(offset[axis] + source.shape[axis]))
                for axis in range(3)
            )
            target[slices] = source.vortex_strength.reshape(*source.shape, 3)
        fields.append(target.reshape(-1, 3))

    axes = tuple(anchor[axis] + h * np.arange(lower[axis], upper[axis] + 1) for axis in range(3))
    mesh = np.meshgrid(*axes, indexing="ij")
    position = np.column_stack([axis.ravel() for axis in mesh])
    shape = (int(shape_array[0]), int(shape_array[1]), int(shape_array[2]))
    return position, fields[0], fields[1], shape, anchor + h * lower


def blend_fvm_vpm_circulation_on_lattice(
    *,
    fvm_position: np.ndarray,
    fvm_cell_volume: np.ndarray,
    fvm_vorticity: np.ndarray,
    vpm_position: np.ndarray,
    vpm_vortex_strength: np.ndarray,
    transfer_box: np.ndarray | list[float] | tuple[float, ...],
    blend_width: float,
    lattice_anchor: np.ndarray,
    spacing: float,
    fvm_solid_mask: np.ndarray | None = None,
) -> LatticeStateBlend:
    r"""Form ``Gamma = eta Gamma_F + (1-eta) Gamma_V`` on one lattice.

    Both states are first scattered with complete M4' support. Existing VPM
    particles with positive ``eta`` are the source state replaced by the
    returned lattice. With zero blend width, the operation is the conservative
    hard lattice transfer and retains the complete FVM release stencil.
    """
    fvm_points = np.asarray(fvm_position, dtype=np.float64).reshape(-1, 3)
    volumes = np.asarray(fvm_cell_volume, dtype=np.float64).reshape(-1)
    fvm_omega = np.asarray(fvm_vorticity, dtype=np.float64).reshape(-1, 3)
    if len(fvm_points) != len(volumes) or len(fvm_points) != len(fvm_omega):
        raise ValueError("FVM position, volume, and vorticity counts must match")
    if not np.all(np.isfinite(fvm_points)) or not np.all(np.isfinite(fvm_omega)):
        raise RuntimeError("FVM transfer donors must be finite")
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise RuntimeError("FVM cell volumes must be finite and positive")
    bounds = np.asarray(transfer_box, dtype=np.float64).reshape(6)
    in_fvm_authority = state_blend_weight(fvm_points, bounds, 0.0) > 0.0
    if fvm_solid_mask is not None:
        solid = np.asarray(fvm_solid_mask, dtype=bool).reshape(-1)
        if len(solid) != len(fvm_points):
            raise ValueError("fvm_solid_mask must match the FVM cell count")
        in_fvm_authority &= ~solid
    if not np.any(in_fvm_authority):
        raise ValueError("FVM transfer region contains no fluid donor cells")

    donor_strength = volumes[in_fvm_authority, None] * fvm_omega[in_fvm_authority]
    fvm_lattice = map_cell_circulation_to_lattice(
        fvm_points[in_fvm_authority],
        volumes[in_fvm_authority],
        fvm_omega[in_fvm_authority],
        lattice_anchor=lattice_anchor,
        spacing=spacing,
    )
    vpm_points, vpm_strength = _validate_vortex_strength(vpm_position, vpm_vortex_strength)
    source_eta = state_blend_weight(vpm_points, bounds, blend_width)
    tolerance = 32.0 * np.finfo(np.float64).eps
    replace = source_eta > tolerance
    vpm_lattice = map_vortex_strength_to_lattice(
        vpm_points[replace],
        vpm_strength[replace],
        lattice_anchor=lattice_anchor,
        spacing=spacing,
    )
    position, fvm_field, vpm_field, shape, origin = _common_lattice_fields(
        fvm_lattice,
        vpm_lattice,
        lattice_anchor=lattice_anchor,
    )
    eta = state_blend_weight(position, bounds, blend_width)
    hard = float(blend_width) == 0.0
    if hard:
        partitioned_strength = fvm_field.copy()
        strength = partitioned_strength.copy()
        divergence_before = 0.0
        divergence_after = 0.0
    else:
        partitioned, corrected, divergence_before, divergence_after = (
            correct_state_blend_cross_divergence(
                fvm_field.reshape(*shape, 3),
                vpm_field.reshape(*shape, 3),
                eta.reshape(shape),
                spacing=spacing,
            )
        )
        partitioned_strength = partitioned.reshape(-1, 3)
        strength = corrected.reshape(-1, 3)
    return LatticeStateBlend(
        position=position,
        vortex_strength=strength,
        partitioned_vortex_strength=partitioned_strength,
        fvm_vortex_strength=fvm_field,
        vpm_vortex_strength=vpm_field,
        eta=eta,
        origin=origin,
        shape=shape,
        spacing=float(spacing),
        fvm_donor_vortex_strength_net=donor_strength.sum(axis=0, dtype=np.float64),
        fvm_donor_first_moment=first_vorticity_moment(fvm_points[in_fvm_authority], donor_strength),
        vpm_source_mask=replace,
        vpm_replace_mask=replace,
        hard_replacement=hard,
        cross_divergence_l2_before=divergence_before,
        cross_divergence_l2_after=divergence_after,
        cross_divergence_relative=(
            divergence_after
            / max(
                float(np.sqrt(np.mean(np.sum(strength.reshape(*shape, 3) ** 2, axis=-1))))
                / float(spacing),
                np.finfo(np.float64).tiny,
            )
        ),
    )


def blend_fvm_vpm_circulation_in_renewal_belt(
    *,
    fvm_position: np.ndarray,
    fvm_cell_volume: np.ndarray,
    fvm_vorticity: np.ndarray,
    vpm_position: np.ndarray,
    vpm_vortex_strength: np.ndarray,
    transfer_box: np.ndarray | list[float] | tuple[float, ...],
    blend_width: float,
    lattice: RenewalLattice,
    fvm_solid_mask: np.ndarray | None = None,
    vpm_position_dtype: np.dtype | type = np.float64,
    compute_divergence_diagnostic: bool = True,
) -> LatticeStateBlend:
    r"""Form an absolute FVM/VPM state over one buffered renewal lattice.

    Every VPM particle in the travel-plus-support belt is renewed.  Particles
    beyond it remain Lagrangian.  The FVM state is authoritative inside the
    transfer box and blends into the VPM state across the exterior M4' release
    support, while mutation ownership remains a separate geometric decision.
    """
    fvm_points = np.asarray(fvm_position, dtype=np.float64).reshape(-1, 3)
    volumes = np.asarray(fvm_cell_volume, dtype=np.float64).reshape(-1)
    fvm_omega = np.asarray(fvm_vorticity, dtype=np.float64).reshape(-1, 3)
    if len(fvm_points) != len(volumes) or len(fvm_points) != len(fvm_omega):
        raise ValueError("FVM position, volume, and vorticity counts must match")
    if not np.all(np.isfinite(fvm_points)) or not np.all(np.isfinite(fvm_omega)):
        raise RuntimeError("FVM transfer donors must be finite")
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise RuntimeError("FVM cell volumes must be finite and positive")
    bounds = np.asarray(transfer_box, dtype=np.float64).reshape(6)
    in_fvm_authority = state_blend_weight(fvm_points, bounds, 0.0) > 0.0
    if fvm_solid_mask is not None:
        solid = np.asarray(fvm_solid_mask, dtype=bool).reshape(-1)
        if len(solid) != len(fvm_points):
            raise ValueError("fvm_solid_mask must match the FVM cell count")
        in_fvm_authority &= ~solid
    if not np.any(in_fvm_authority):
        raise ValueError("FVM transfer region contains no fluid donor cells")

    donor_position = fvm_points[in_fvm_authority]
    donor_strength = volumes[in_fvm_authority, None] * fvm_omega[in_fvm_authority]
    fvm_field = scatter_vortex_strength_to_renewal_lattice(
        donor_position,
        donor_strength,
        lattice,
    )
    vpm_points, vpm_strength = _validate_vortex_strength(vpm_position, vpm_vortex_strength)
    in_renewal_belt = state_blend_weight(vpm_points, lattice.renewal_bounds, 0.0) > 0.0
    position_dtype = np.dtype(vpm_position_dtype)
    relative = (vpm_points - lattice.lattice_anchor) / lattice.spacing
    nearest = np.rint(relative)
    reconstructed = lattice.lattice_anchor + lattice.spacing * nearest
    coordinate_tolerance = 16.0 * np.finfo(position_dtype).eps * np.maximum(1.0, np.abs(vpm_points))
    regular = np.all(np.abs(vpm_points - reconstructed) <= coordinate_tolerance, axis=1)
    lattice_bounds = np.column_stack(
        (lattice.position.min(axis=0), lattice.position.max(axis=0))
    ).reshape(-1)
    in_complete_lattice = state_blend_weight(vpm_points, lattice_bounds, 0.0) > 0.0
    # Renew the whole physical belt.  Regular nodes in its complete outer M4'
    # guard are also embedded so a repeated handoff reconciles, rather than
    # re-adds, support deposited at the persistent boundary.
    replace = in_renewal_belt
    source = replace | (in_complete_lattice & regular)
    vpm_field = scatter_vortex_strength_to_renewal_lattice(
        vpm_points[source],
        vpm_strength[source],
        lattice,
        position_dtype=vpm_position_dtype,
    )
    relative_donor = (donor_position - lattice.lattice_anchor) / lattice.spacing
    donor_lower = np.min(np.floor(relative_donor).astype(np.int64) - 1, axis=0)
    donor_upper = np.max(np.floor(relative_donor).astype(np.int64) + 2, axis=0)
    target_index = np.rint((lattice.position - lattice.lattice_anchor) / lattice.spacing).astype(
        np.int64
    )
    fvm_support = np.all((target_index >= donor_lower) & (target_index <= donor_upper), axis=1)
    if float(blend_width) == 0.0:
        # Hard ownership still needs the complete donor stencil: support nodes
        # are reconciled in the release belt, not classified for hard deletion.
        eta_flat = fvm_support.astype(np.float64)
    else:
        eta_flat = release_blend_weight(lattice.position, bounds, blend_width)
        eta_flat *= fvm_support
    eta = eta_flat.reshape(lattice.shape)
    partitioned = eta[..., None] * fvm_field + (1.0 - eta[..., None]) * vpm_field

    divergence = 0.0
    if compute_divergence_diagnostic and not np.array_equal(fvm_field, vpm_field):
        fvm_divergence = spectral_lattice_divergence(fvm_field, spacing=lattice.spacing)
        vpm_divergence = spectral_lattice_divergence(vpm_field, spacing=lattice.spacing)
        cross = spectral_lattice_divergence(partitioned, spacing=lattice.spacing) - (
            eta * fvm_divergence + (1.0 - eta) * vpm_divergence
        )
        divergence = float(np.sqrt(np.mean(cross**2)))
    field_scale = float(np.sqrt(np.mean(np.sum(partitioned**2, axis=-1)))) / lattice.spacing
    divergence_relative = divergence / max(field_scale, np.finfo(np.float64).tiny)

    flat_fvm = fvm_field.reshape(-1, 3)
    flat_vpm = vpm_field.reshape(-1, 3)
    flat_partitioned = partitioned.reshape(-1, 3)
    return LatticeStateBlend(
        position=lattice.position,
        vortex_strength=flat_partitioned,
        partitioned_vortex_strength=flat_partitioned.copy(),
        fvm_vortex_strength=flat_fvm,
        vpm_vortex_strength=flat_vpm,
        eta=eta.reshape(-1),
        origin=lattice.origin,
        shape=lattice.shape,
        spacing=lattice.spacing,
        fvm_donor_vortex_strength_net=donor_strength.sum(axis=0, dtype=np.float64),
        fvm_donor_first_moment=first_vorticity_moment(donor_position, donor_strength),
        vpm_source_mask=source,
        vpm_replace_mask=replace,
        hard_replacement=float(blend_width) == 0.0,
        cross_divergence_l2_before=divergence,
        cross_divergence_l2_after=divergence,
        cross_divergence_relative=divergence_relative,
    )


def first_vorticity_moment(position: np.ndarray, vortex_strength: np.ndarray) -> np.ndarray:
    """Return componentwise first moments ``sum(x_j * Gamma_i)``."""
    points = np.asarray(position, dtype=np.float64).reshape(-1, 3)
    strength = np.asarray(vortex_strength, dtype=np.float64).reshape(-1, 3)
    if len(points) != len(strength):
        raise ValueError("position and vortex_strength counts must match")
    return points.T @ strength


__all__ = [
    "LatticeStateBlend",
    "LatticeTransfer",
    "RenewalLattice",
    "blend_fvm_vpm_circulation_in_renewal_belt",
    "blend_fvm_vpm_circulation_on_lattice",
    "build_renewal_lattice",
    "correct_state_blend_cross_divergence",
    "evaluate_gaussian_vorticity",
    "first_vorticity_moment",
    "m4_prime",
    "map_cell_circulation_to_lattice",
    "map_vortex_strength_to_lattice",
    "release_blend_weight",
    "scatter_vortex_strength_to_renewal_lattice",
    "spectral_lattice_divergence",
    "state_blend_weight",
]
