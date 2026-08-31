"""Conservative refinement of stretched vortex-line elements.

The particle vortex-strength vector is a material line element in a three-
dimensional vortex method.  A particle whose strength has grown too large is
therefore refined along its own vortex strength direction:

* two children receive half the parent vortex strength and particle_volume;
* their position are displaced symmetrically along the vortex-strength vector;
* their Gaussian core radius and material properties are unchanged.

For a displacement parallel to ``vortex_strength`` this construction preserves total
vector vortex strength, total strength variation, linear impulse, and the
kernel-corrected angular impulse algebraically.  It also leaves molecular
diffusion in the core radius untouched.  Unlike a strength limiter, refinement
does not remove stretched vorticity; it adds the Lagrangian degrees of freedom
needed to represent it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class FilamentRefinementError(RuntimeError):
    """A requested conservative refinement cannot fit its declared budget."""


@dataclass(frozen=True)
class FilamentRefinementResult:
    """Particle arrays and transfer diagnostics produced by one refinement."""

    position: np.ndarray
    vortex_strength: np.ndarray
    core_radius: np.ndarray
    particle_volume: np.ndarray
    reference_vortex_strength: np.ndarray
    reference_length: np.ndarray
    source_index: np.ndarray
    refined_parent_index: np.ndarray
    refined_particles: int
    max_stretch_ratio: float
    vortex_strength_error: float
    vortex_strength_variation_error: float
    linear_impulse_error: float
    angular_impulse_error: float
    isolated_kinetic_energy_change: float


@dataclass(frozen=True)
class GaussianIntegralTransfer:
    """Exact changes in the Gaussian-blob quadratic flow integrals."""

    total_kinetic_energy_change: float
    total_enstrophy_change: float
    total_helicity_change: float


def particle_moments(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
    *,
    angular_core_coefficient: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Return ``(sum vortex_strength, sum|vortex_strength|, I, A)`` for a supported blob kernel."""

    position = np.asarray(position, dtype=np.float64)
    vortex_strength = np.asarray(vortex_strength, dtype=np.float64)
    core_radius = np.asarray(core_radius, dtype=np.float64)
    if position.shape != vortex_strength.shape or position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position and vortex strength must both have shape (N, 3)")
    if core_radius.shape != (len(position),):
        raise ValueError("core_radius must have shape (N,)")

    total = vortex_strength.sum(axis=0, dtype=np.float64)
    variation = float(np.linalg.norm(vortex_strength, axis=1).sum(dtype=np.float64))
    impulse = 0.5 * np.cross(position, vortex_strength).sum(axis=0, dtype=np.float64)
    angular = np.cross(position, np.cross(position, vortex_strength)).sum(
        axis=0, dtype=np.float64
    ) / 3.0 - angular_core_coefficient * (core_radius[:, None] ** 2 * vortex_strength).sum(
        axis=0, dtype=np.float64
    )
    return total, variation, impulse, angular


def gaussian_particle_moments(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Return ``(sum vortex_strength, sum|vortex_strength|, I, A)`` for Gaussian vortex blobs."""
    return particle_moments(
        position,
        vortex_strength,
        core_radius,
        angular_core_coefficient=1.0 / 3.0,
    )


def _gaussian_energy_kernel(normalized_distance: np.ndarray) -> np.ndarray:
    """Gaussian blob energy kernel ``erf(rho)/(4*pi*rho)``."""

    from scipy.special import erf

    normalized_distance = np.asarray(normalized_distance, dtype=np.float64)
    origin = 1.0 / (2.0 * np.pi**1.5)
    result = np.full_like(normalized_distance, origin)
    nonzero = np.abs(normalized_distance) >= 1e-12
    result[nonzero] = erf(normalized_distance[nonzero]) / (
        4.0 * np.pi * normalized_distance[nonzero]
    )
    return result


def _isolated_split_energy_change(
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
    displacement: np.ndarray,
) -> float:
    """Exact self-plus-sibling energy change, excluding other particles."""

    magnitude_sq = np.einsum("ij,ij->i", vortex_strength, vortex_strength)
    sigma_pair = np.sqrt(2.0) * core_radius
    normalized_distance = 2.0 * displacement / sigma_pair
    origin = 1.0 / (2.0 * np.pi**1.5)
    sibling = _gaussian_energy_kernel(normalized_distance)
    parent = 0.5 * magnitude_sq * origin / sigma_pair
    children = 0.25 * magnitude_sq * (origin + sibling) / sigma_pair
    return float((children - parent).sum(dtype=np.float64))


def _gaussian_pair_integrals(
    position_a: np.ndarray,
    vortex_strength_a: np.ndarray,
    core_radius_a: np.ndarray,
    position_b: np.ndarray,
    vortex_strength_b: np.ndarray,
    core_radius_b: np.ndarray,
    *,
    target_pairs_per_chunk: int = 500_000,
) -> tuple[float, float, float]:
    """Return ordered ``A x B`` Gaussian energy, enstrophy, and helicity sums.

    The energy sum omits the outer factor ``1/2`` used for a full ordered
    particle-particle sum.  Chunking bounds temporary storage when ``B`` is a
    production-sized cloud.
    """

    from scipy.special import erf

    if len(position_a) == 0 or len(position_b) == 0:
        return 0.0, 0.0, 0.0

    b_count = len(position_b)
    chunk_size = max(1, target_pairs_per_chunk // b_count)
    total_kinetic_energy = 0.0
    total_enstrophy = 0.0
    total_helicity = 0.0
    gaussian_origin = 1.0 / (2.0 * np.pi**1.5)
    vorticity_origin = 1.0 / np.pi**1.5

    for start in range(0, len(position_a), chunk_size):
        stop = min(start + chunk_size, len(position_a))
        displacement = position_a[start:stop, None, :] - position_b[None, :, :]
        distance_sq = np.einsum("abk,abk->ab", displacement, displacement)
        distance = np.sqrt(distance_sq)
        convolved_core_radius = np.sqrt(
            core_radius_a[start:stop, None] ** 2 + core_radius_b[None, :] ** 2
        )
        normalized_distance = distance / convolved_core_radius
        dot_product = np.einsum(
            "ak,bk->ab",
            vortex_strength_a[start:stop],
            vortex_strength_b,
        )

        energy_kernel = np.full_like(normalized_distance, gaussian_origin)
        nonzero_distance = normalized_distance >= 1e-12
        energy_kernel[nonzero_distance] = erf(normalized_distance[nonzero_distance]) / (
            4.0 * np.pi * normalized_distance[nonzero_distance]
        )
        total_kinetic_energy += float(
            np.sum(
                dot_product * energy_kernel / convolved_core_radius,
                dtype=np.float64,
            )
        )

        vorticity_kernel = (
            vorticity_origin
            * np.exp(-(normalized_distance * normalized_distance))
            / convolved_core_radius**3
        )
        total_enstrophy += float(np.sum(dot_product * vorticity_kernel, dtype=np.float64))

        has_nonzero_separation = distance >= 1e-12
        q_value = np.empty_like(normalized_distance)
        q_value[nonzero_distance] = (
            erf(normalized_distance[nonzero_distance])
            - 2.0
            / np.sqrt(np.pi)
            * normalized_distance[nonzero_distance]
            * np.exp(-(normalized_distance[nonzero_distance] ** 2))
        ) / (4.0 * np.pi)
        q_value[~nonzero_distance] = normalized_distance[~nonzero_distance] ** 3 / (
            3.0 * np.pi**1.5
        )
        triple_product = np.einsum(
            "abk,abk->ab",
            displacement,
            np.cross(
                vortex_strength_a[start:stop, None, :],
                vortex_strength_b[None, :, :],
            ),
        )
        helicity_kernel = np.zeros_like(distance)
        helicity_kernel[has_nonzero_separation] = q_value[has_nonzero_separation] / (
            distance_sq[has_nonzero_separation] * distance[has_nonzero_separation]
        )
        total_helicity += float(np.sum(triple_product * helicity_kernel, dtype=np.float64))

    return total_kinetic_energy, total_enstrophy, total_helicity


def gaussian_refinement_integral_transfer(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
    result: FilamentRefinementResult,
) -> GaussianIntegralTransfer:
    """Return the exact flow-integral jump caused by one refinement transform.

    Unchanged-particle pairs cancel identically.  The calculation therefore
    evaluates only parent-parent/parent-retained interactions before the
    transform and child-child/child-retained interactions after it.
    """

    selected = result.refined_parent_index
    if len(selected) == 0:
        return GaussianIntegralTransfer(0.0, 0.0, 0.0)
    retained_mask = np.ones(len(position), dtype=bool)
    retained_mask[selected] = False
    retained = np.flatnonzero(retained_mask)
    child_start = len(retained)

    old_self = _gaussian_pair_integrals(
        position[selected],
        vortex_strength[selected],
        core_radius[selected],
        position[selected],
        vortex_strength[selected],
        core_radius[selected],
    )
    old_cross = _gaussian_pair_integrals(
        position[selected],
        vortex_strength[selected],
        core_radius[selected],
        position[retained],
        vortex_strength[retained],
        core_radius[retained],
    )
    new_self = _gaussian_pair_integrals(
        result.position[child_start:],
        result.vortex_strength[child_start:],
        result.core_radius[child_start:],
        result.position[child_start:],
        result.vortex_strength[child_start:],
        result.core_radius[child_start:],
    )
    new_cross = _gaussian_pair_integrals(
        result.position[child_start:],
        result.vortex_strength[child_start:],
        result.core_radius[child_start:],
        result.position[:child_start],
        result.vortex_strength[:child_start],
        result.core_radius[:child_start],
    )

    return GaussianIntegralTransfer(
        total_kinetic_energy_change=0.5 * (new_self[0] - old_self[0]) + new_cross[0] - old_cross[0],
        total_enstrophy_change=(new_self[1] - old_self[1]) + 2.0 * (new_cross[1] - old_cross[1]),
        total_helicity_change=(new_self[2] - old_self[2]) + 2.0 * (new_cross[2] - old_cross[2]),
    )


def _assert_transform_is_exact(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
    before: tuple[np.ndarray, float, np.ndarray, np.ndarray],
    after: tuple[np.ndarray, float, np.ndarray, np.ndarray],
) -> None:
    """Verify the bisection preserved its four invariants to roundoff.

    Bisection is exactly moment-preserving by construction, so any measurable
    drift here is an implementation error rather than a modelling choice; it is
    checked where the transform is built instead of being reported upward.
    """

    impulse_scale = max(
        0.5
        * float(np.linalg.norm(np.cross(position, vortex_strength), axis=1).sum(dtype=np.float64)),
        np.finfo(float).tiny,
    )
    angular_terms = (
        np.cross(position, np.cross(position, vortex_strength)) / 3.0
        - core_radius[:, None] ** 2 * vortex_strength / 3.0
    )
    angular_scale = max(
        float(np.linalg.norm(angular_terms, axis=1).sum(dtype=np.float64)),
        np.finfo(float).tiny,
    )
    allowance = 512.0 * np.finfo(float).eps
    checks = (
        ("vector vortex strength", float(np.linalg.norm(after[0] - before[0])), before[1]),
        ("total strength variation", abs(after[1] - before[1]), before[1]),
        ("linear impulse", float(np.linalg.norm(after[2] - before[2])), impulse_scale),
        ("angular impulse", float(np.linalg.norm(after[3] - before[3])), angular_scale),
    )
    for name, error, scale in checks:
        limit = allowance * max(float(scale), np.finfo(float).tiny)
        if error > limit:
            raise FilamentRefinementError(
                f"filament refinement changed {name} by {error:.3e}, beyond its "
                f"roundoff allowance {limit:.3e}"
            )


def split_stretched_filaments(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
    particle_volume: np.ndarray,
    *,
    reference_vortex_strength: np.ndarray,
    reference_length: np.ndarray,
    max_stretch_factor: float,
    offset_fraction: float = 0.25,
    max_n_particles: int | None = None,
    max_absolute_vortex_strength: float | None = None,
) -> FilamentRefinementResult:
    """Bisect over-stretched particles along their vortex-line direction.

    ``reference_vortex_strength`` and ``reference_length`` describe each particle at
    its most recent reference time (initialization, or the time its parent was
    split).  Since a vortex-particle strength vector is a material line
    element, ``|vortex_strength| / |vortex_strength_ref|`` estimates its line-stretch ratio.  The
    children are placed at the quarter points of that estimated material line
    by the default ``offset_fraction=0.25``.  Each child then receives its own
    current strength and half-line length as the next reference state.

    A single call performs one bisection per selected parent.  Repeated calls
    can therefore resolve arbitrarily large stretching without placing a
    single pair over an arbitrarily long line segment.
    """

    position = np.asarray(position, dtype=np.float64)
    vortex_strength = np.asarray(vortex_strength, dtype=np.float64)
    core_radius = np.asarray(core_radius, dtype=np.float64)
    particle_volume = np.asarray(particle_volume, dtype=np.float64)
    reference_vortex_strength = np.asarray(reference_vortex_strength, dtype=np.float64)
    reference_length = np.asarray(reference_length, dtype=np.float64)
    if position.shape != vortex_strength.shape or position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position and vortex strength must both have shape (N, 3)")
    if core_radius.shape != (len(position),) or particle_volume.shape != (len(position),):
        raise ValueError("core_radius and particle_volume must both have shape (N,)")
    if reference_vortex_strength.shape != (len(position),) or reference_length.shape != (
        len(position),
    ):
        raise ValueError("reference_vortex_strength and reference_length must both have shape (N,)")
    if max_stretch_factor <= 1.0:
        raise ValueError("max_stretch_factor must be greater than one")
    if max_absolute_vortex_strength is not None and (
        not np.isfinite(max_absolute_vortex_strength) or max_absolute_vortex_strength <= 0.0
    ):
        raise ValueError("max_absolute_vortex_strength must be finite and positive")
    if not 0.0 <= offset_fraction <= 0.5:
        raise ValueError("offset_fraction must be in [0, 0.5]")
    if (
        np.any(core_radius <= 0.0)
        or np.any(particle_volume <= 0.0)
        or np.any(reference_vortex_strength <= 0.0)
        or np.any(reference_length <= 0.0)
    ):
        raise ValueError(
            "core_radius, particle_volume, reference_vortex_strength, and "
            "reference_length must be positive"
        )
    if not (
        np.isfinite(position).all()
        and np.isfinite(vortex_strength).all()
        and np.isfinite(core_radius).all()
        and np.isfinite(particle_volume).all()
        and np.isfinite(reference_vortex_strength).all()
        and np.isfinite(reference_length).all()
    ):
        raise ValueError("particle arrays must be finite")

    magnitude = np.linalg.norm(vortex_strength, axis=1)
    stretch_ratio = magnitude / reference_vortex_strength
    risk = stretch_ratio / max_stretch_factor
    if max_absolute_vortex_strength is not None:
        risk = np.maximum(risk, magnitude / max_absolute_vortex_strength)
    eligible = np.flatnonzero(risk > 1.0)
    selected = eligible
    if max_n_particles is not None:
        available = max(0, max_n_particles - len(position))
        if len(selected) > available:
            priority = np.lexsort((selected, -risk[selected]))
            selected = selected[priority[:available]]
    refined_count = len(selected)

    if refined_count == 0:
        source = np.arange(len(position), dtype=np.int64)
        return FilamentRefinementResult(
            position=position.copy(),
            vortex_strength=vortex_strength.copy(),
            core_radius=core_radius.copy(),
            particle_volume=particle_volume.copy(),
            reference_vortex_strength=reference_vortex_strength.copy(),
            reference_length=reference_length.copy(),
            source_index=source,
            refined_parent_index=selected,
            refined_particles=0,
            max_stretch_ratio=(float(stretch_ratio.max()) if len(stretch_ratio) else 0.0),
            vortex_strength_error=0.0,
            vortex_strength_variation_error=0.0,
            linear_impulse_error=0.0,
            angular_impulse_error=0.0,
            isolated_kinetic_energy_change=0.0,
        )

    retained_mask = np.ones(len(position), dtype=bool)
    retained_mask[selected] = False
    retained = np.flatnonzero(retained_mask)
    direction = vortex_strength[selected] / magnitude[selected, None]
    current_line_length = reference_length[selected] * stretch_ratio[selected]
    displacement_magnitude = offset_fraction * current_line_length
    displacement = displacement_magnitude[:, None] * direction

    new_position = np.concatenate(
        (
            position[retained],
            position[selected] + displacement,
            position[selected] - displacement,
        ),
        axis=0,
    )
    new_vortex_strength = np.concatenate(
        (
            vortex_strength[retained],
            0.5 * vortex_strength[selected],
            0.5 * vortex_strength[selected],
        ),
        axis=0,
    )
    new_core_radius = np.concatenate(
        (core_radius[retained], core_radius[selected], core_radius[selected])
    )
    new_particle_volume = np.concatenate(
        (
            particle_volume[retained],
            0.5 * particle_volume[selected],
            0.5 * particle_volume[selected],
        )
    )
    child_reference_vortex_strength = 0.5 * magnitude[selected]
    child_reference_length = 0.5 * current_line_length
    new_reference_vortex_strength = np.concatenate(
        (
            reference_vortex_strength[retained],
            child_reference_vortex_strength,
            child_reference_vortex_strength,
        )
    )
    new_reference_length = np.concatenate(
        (
            reference_length[retained],
            child_reference_length,
            child_reference_length,
        )
    )
    source = np.concatenate((retained, selected, selected)).astype(np.int64, copy=False)

    before = gaussian_particle_moments(position, vortex_strength, core_radius)
    after = gaussian_particle_moments(new_position, new_vortex_strength, new_core_radius)
    _assert_transform_is_exact(position, vortex_strength, core_radius, before, after)
    return FilamentRefinementResult(
        position=new_position,
        vortex_strength=new_vortex_strength,
        core_radius=new_core_radius,
        particle_volume=new_particle_volume,
        reference_vortex_strength=new_reference_vortex_strength,
        reference_length=new_reference_length,
        source_index=source,
        refined_parent_index=selected,
        refined_particles=refined_count,
        max_stretch_ratio=float(stretch_ratio.max()),
        vortex_strength_error=float(np.linalg.norm(after[0] - before[0])),
        vortex_strength_variation_error=abs(after[1] - before[1]),
        linear_impulse_error=float(np.linalg.norm(after[2] - before[2])),
        angular_impulse_error=float(np.linalg.norm(after[3] - before[3])),
        isolated_kinetic_energy_change=_isolated_split_energy_change(
            vortex_strength[selected],
            core_radius[selected],
            displacement_magnitude,
        ),
    )


__all__ = [
    "FilamentRefinementError",
    "FilamentRefinementResult",
    "GaussianIntegralTransfer",
    "gaussian_particle_moments",
    "gaussian_refinement_integral_transfer",
    "split_stretched_filaments",
]
