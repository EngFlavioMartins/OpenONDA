"""Gaussian field projection on the particle basis produced by the VPM.

This module implements the numerical core of the projected renewal interface.
In particular, a GBD regeneration is treated as the VPM's legitimate geometry
update: coupling authority is reconstructed from the *current* coordinates and
the FVM transfer changes strengths, not the basis.  New support positions are
eligible only when the current basis fails the requested physical-field gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
from scipy.spatial import KDTree
from scipy.special import gammainc


@dataclass(frozen=True)
class AdvectedRenewalProjectionResult:
    """Dense float64 projection result and its numerical-health diagnostics."""

    vortex_strength: np.ndarray
    fitted_vorticity: np.ndarray
    fitted_velocity: np.ndarray | None
    fitted_normal_velocity: np.ndarray | None
    singular_values: np.ndarray
    rank: int
    condition_number: float
    vorticity_relative_error: float
    velocity_relative_error: float | None
    prior_relative_change: float | None
    maximum_strength: float
    rms_strength: float
    maximum_to_rms_strength: float


@dataclass(frozen=True)
class GBDRenewalProjectionResult:
    """Absolute state produced on one post-GBD particle basis."""

    renewable_mask: np.ndarray
    preserved_mask: np.ndarray
    updated_vortex_strength: np.ndarray
    birth_position: np.ndarray
    birth_vortex_strength: np.ndarray
    birth_core_radius: np.ndarray
    projection: AdvectedRenewalProjectionResult | SparseRenewalProjectionResult
    used_selective_births: bool


@dataclass(frozen=True)
class SparseRenewalProjectionResult:
    """Production sparse solve on a local Gaussian particle basis."""

    vortex_strength: np.ndarray
    fitted_vorticity: np.ndarray
    condition_number: float
    rank: int
    vorticity_relative_error: float
    velocity_relative_error: float | None
    prior_relative_change: float | None
    maximum_strength: float
    rms_strength: float
    maximum_to_rms_strength: float
    iteration_count: int
    operator_nonzeros: int
    converged: bool


def _core_radius_array(core_radius: np.ndarray | float, count: int) -> np.ndarray:
    radius = np.asarray(core_radius, dtype=np.float64)
    if radius.ndim == 0:
        if not np.isfinite(float(radius)) or float(radius) <= 0.0:
            raise ValueError("core radii must be finite and positive")
        radius = np.full(count, float(radius), dtype=np.float64)
    else:
        radius = radius.reshape(-1)
    if len(radius) != count:
        raise ValueError("core_radius must be scalar or have one value per particle")
    if not np.all(np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("core radii must be finite and positive")
    return radius


def _relative_norm(residual: np.ndarray, reference: np.ndarray) -> float:
    reference_norm = float(np.linalg.norm(reference))
    residual_norm = float(np.linalg.norm(residual))
    return residual_norm / reference_norm if reference_norm > 0.0 else residual_norm


def gaussian_vorticity_basis(
    evaluation_position: np.ndarray,
    particle_position: np.ndarray,
    core_radius: np.ndarray | float,
) -> np.ndarray:
    """Return OpenONDA's scalar Gaussian vorticity collocation matrix."""
    targets = np.asarray(evaluation_position, dtype=np.float64).reshape(-1, 3)
    particles = np.asarray(particle_position, dtype=np.float64).reshape(-1, 3)
    radius = _core_radius_array(core_radius, len(particles))
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(particles)):
        raise ValueError("evaluation and particle positions must be finite")
    displacement = targets[:, None, :] - particles[None, :, :]
    distance_squared = np.einsum("mni,mni->mn", displacement, displacement)
    return np.exp(-distance_squared / radius[None, :] ** 2) / (np.pi**1.5 * radius[None, :] ** 3)


def sparse_gaussian_vorticity_basis(
    evaluation_position: np.ndarray,
    particle_position: np.ndarray,
    core_radius: np.ndarray | float,
    *,
    relative_tail_cutoff: float = 1.0e-8,
) -> sparse.csr_matrix:
    """Return a local sparse Gaussian basis with a quantified kernel cutoff.

    Entries are omitted only when their scalar Gaussian weight is below
    ``relative_tail_cutoff`` times that particle's peak weight.  For the
    default this retains support through ``sqrt(log(1e8)) sigma = 4.29 sigma``.
    """
    targets = np.asarray(evaluation_position, dtype=np.float64).reshape(-1, 3)
    particles = np.asarray(particle_position, dtype=np.float64).reshape(-1, 3)
    radius = _core_radius_array(core_radius, len(particles))
    cutoff = float(relative_tail_cutoff)
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(particles)):
        raise ValueError("evaluation and particle positions must be finite")
    if not np.isfinite(cutoff) or cutoff <= 0.0 or cutoff >= 1.0:
        raise ValueError("relative_tail_cutoff must lie strictly between zero and one")
    shape = (len(targets), len(particles))
    if not len(targets) or not len(particles):
        return sparse.csr_matrix(shape, dtype=np.float64)

    support_multiple = float(np.sqrt(-np.log(cutoff)))
    maximum_distance = support_multiple * float(radius.max())
    distances = KDTree(targets).sparse_distance_matrix(
        KDTree(particles),
        maximum_distance,
        output_type="coo_matrix",
    )
    row = np.asarray(distances.row, dtype=np.int64)
    column = np.asarray(distances.col, dtype=np.int64)
    distance = np.asarray(distances.data, dtype=np.float64)
    local_radius = radius[column]
    retain = distance <= support_multiple * local_radius
    row = row[retain]
    column = column[retain]
    distance = distance[retain]
    local_radius = local_radius[retain]
    value = np.exp(-((distance / local_radius) ** 2)) / (np.pi**1.5 * local_radius**3)
    return sparse.coo_matrix((value, (row, column)), shape=shape).tocsr()


def evaluate_sparse_gaussian_vorticity(
    evaluation_position: np.ndarray,
    particle_position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray | float,
    *,
    relative_tail_cutoff: float = 1.0e-8,
) -> np.ndarray:
    """Evaluate a Gaussian vorticity field using the production sparse cutoff."""
    particles = np.asarray(particle_position, dtype=np.float64).reshape(-1, 3)
    strength = np.asarray(vortex_strength, dtype=np.float64).reshape(-1, 3)
    if strength.shape != particles.shape or not np.all(np.isfinite(strength)):
        raise ValueError("vortex_strength must be finite with shape (N, 3)")
    basis = sparse_gaussian_vorticity_basis(
        evaluation_position,
        particles,
        core_radius,
        relative_tail_cutoff=relative_tail_cutoff,
    )
    return np.asarray(basis @ strength, dtype=np.float64).reshape(-1, 3)


def solve_sparse_renewal_projection(
    *,
    collocation_position: np.ndarray,
    target_vorticity: np.ndarray,
    particle_position: np.ndarray,
    core_radius: np.ndarray | float,
    prior_vortex_strength: np.ndarray | None = None,
    prior_weight: float = 0.0,
    relative_tail_cutoff: float = 1.0e-8,
    relative_tolerance: float = 1.0e-10,
    maximum_iterations: int | None = None,
) -> SparseRenewalProjectionResult:
    """Solve absolute strengths with sparse LSMR on the local Gaussian basis."""
    particles = np.asarray(particle_position, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_vorticity, dtype=np.float64).reshape(-1, 3)
    collocation = np.asarray(collocation_position, dtype=np.float64).reshape(-1, 3)
    if len(target) != len(collocation) or not np.all(np.isfinite(target)):
        raise ValueError("target_vorticity must be finite and match collocation_position")
    tolerance = float(relative_tolerance)
    penalty = float(prior_weight)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive")
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError("prior_weight must be finite and non-negative")

    basis = sparse_gaussian_vorticity_basis(
        collocation,
        particles,
        core_radius,
        relative_tail_cutoff=relative_tail_cutoff,
    )
    prior: np.ndarray | None = None
    if prior_vortex_strength is not None:
        prior = np.asarray(prior_vortex_strength, dtype=np.float64).reshape(-1, 3)
        if prior.shape != particles.shape or not np.all(np.isfinite(prior)):
            raise ValueError("prior_vortex_strength must be finite with shape (N, 3)")
    if penalty > 0.0:
        if prior is None:
            raise ValueError("prior_vortex_strength is required when prior_weight is nonzero")
        operator = sparse.vstack(
            (basis, penalty * sparse.eye(len(particles), format="csr")),
            format="csr",
        )
        right_hand_side = np.vstack((target, penalty * prior))
    else:
        operator = basis
        right_hand_side = target

    if not len(particles):
        fitted = np.zeros_like(target)
        return SparseRenewalProjectionResult(
            vortex_strength=np.empty((0, 3), dtype=np.float64),
            fitted_vorticity=fitted,
            condition_number=0.0,
            rank=0,
            vorticity_relative_error=_relative_norm(fitted - target, target),
            velocity_relative_error=None,
            prior_relative_change=None,
            maximum_strength=0.0,
            rms_strength=0.0,
            maximum_to_rms_strength=0.0,
            iteration_count=0,
            operator_nonzeros=0,
            converged=not np.any(target),
        )

    iteration_limit = (
        max(100, min(2000, 2 * len(particles)))
        if maximum_iterations is None
        else int(maximum_iterations)
    )
    if iteration_limit < 1:
        raise ValueError("maximum_iterations must be positive")
    strength = np.empty((len(particles), 3), dtype=np.float64)
    condition_number = 0.0
    iteration_count = 0
    converged = True
    for component in range(3):
        result = lsmr(
            operator,
            right_hand_side[:, component],
            atol=tolerance,
            btol=tolerance,
            maxiter=iteration_limit,
            x0=None if prior is None else prior[:, component],
        )
        strength[:, component] = result[0]
        stop_reason = int(result[1])
        iteration_count = max(iteration_count, int(result[2]))
        condition_number = max(condition_number, float(result[6]))
        converged &= stop_reason in {0, 1, 2, 4, 5}

    fitted = np.asarray(basis @ strength, dtype=np.float64).reshape(-1, 3)
    magnitude = np.linalg.norm(strength, axis=1)
    maximum_strength = float(magnitude.max(initial=0.0))
    rms_strength = float(np.sqrt(np.mean(magnitude**2))) if len(magnitude) else 0.0
    return SparseRenewalProjectionResult(
        vortex_strength=strength,
        fitted_vorticity=fitted,
        condition_number=condition_number,
        rank=-1,
        vorticity_relative_error=_relative_norm(fitted - target, target),
        velocity_relative_error=None,
        prior_relative_change=(None if prior is None else _relative_norm(strength - prior, prior)),
        maximum_strength=maximum_strength,
        rms_strength=rms_strength,
        maximum_to_rms_strength=(maximum_strength / rms_strength if rms_strength > 0.0 else 0.0),
        iteration_count=iteration_count,
        operator_nonzeros=int(basis.nnz),
        converged=converged,
    )


def gaussian_vorticity_divergence_operator(
    evaluation_position: np.ndarray,
    particle_position: np.ndarray,
    core_radius: np.ndarray | float,
) -> np.ndarray:
    """Return ``D`` such that ``D @ Gamma.ravel()`` is ``div(omega)``."""
    targets = np.asarray(evaluation_position, dtype=np.float64).reshape(-1, 3)
    particles = np.asarray(particle_position, dtype=np.float64).reshape(-1, 3)
    radius = _core_radius_array(core_radius, len(particles))
    basis = gaussian_vorticity_basis(targets, particles, radius)
    displacement = targets[:, None, :] - particles[None, :, :]
    gradient = -2.0 * displacement * basis[..., None] / radius[None, :, None] ** 2
    return gradient.reshape(len(targets), 3 * len(particles))


def gaussian_velocity_operator(
    evaluation_position: np.ndarray,
    particle_position: np.ndarray,
    core_radius: np.ndarray | float,
    *,
    normal: np.ndarray | None = None,
) -> np.ndarray:
    """Return the Gaussian-regularized Biot--Savart linear operator."""
    targets = np.asarray(evaluation_position, dtype=np.float64).reshape(-1, 3)
    particles = np.asarray(particle_position, dtype=np.float64).reshape(-1, 3)
    radius = _core_radius_array(core_radius, len(particles))
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(particles)):
        raise ValueError("evaluation and particle positions must be finite")

    displacement = targets[:, None, :] - particles[None, :, :]
    distance_squared = np.einsum("mni,mni->mn", displacement, displacement)
    density = np.sqrt(distance_squared) / radius[None, :]
    enclosed = gammainc(1.5, density**2) / (4.0 * np.pi)
    scale = np.divide(
        enclosed,
        distance_squared * np.sqrt(distance_squared),
        out=np.zeros_like(enclosed),
        where=distance_squared > 0.0,
    )

    cross_matrix = np.zeros((*displacement.shape[:2], 3, 3), dtype=np.float64)
    cross_matrix[..., 0, 1] = -displacement[..., 2]
    cross_matrix[..., 0, 2] = displacement[..., 1]
    cross_matrix[..., 1, 0] = displacement[..., 2]
    cross_matrix[..., 1, 2] = -displacement[..., 0]
    cross_matrix[..., 2, 0] = -displacement[..., 1]
    cross_matrix[..., 2, 1] = displacement[..., 0]
    velocity_blocks = -scale[..., None, None] * cross_matrix
    if normal is None:
        return velocity_blocks.transpose(0, 2, 1, 3).reshape(3 * len(targets), 3 * len(particles))

    normals = np.asarray(normal, dtype=np.float64).reshape(-1, 3)
    if len(normals) != len(targets) or not np.all(np.isfinite(normals)):
        raise ValueError("normal must contain one finite vector per velocity point")
    magnitude = np.linalg.norm(normals, axis=1)
    if np.any(magnitude <= 0.0):
        raise ValueError("normal vectors must be nonzero")
    unit_normal = normals / magnitude[:, None]
    return np.einsum("mi,mnij->mnj", unit_normal, velocity_blocks).reshape(
        len(targets), 3 * len(particles)
    )


def geometric_renewal_mask(
    particle_position: np.ndarray,
    renewal_bounds: np.ndarray | tuple[float, ...],
    *,
    particle_spacing: float,
) -> np.ndarray:
    """Classify the current post-remesh cloud with scale-aware bound tolerance.

    No particle identity or regenerated winner ID participates in this rule.
    The spacing-scaled term absorbs harmless lattice construction roundoff; the
    ULP term also remains valid at large coordinate offsets.
    """
    position = np.asarray(particle_position, dtype=np.float64).reshape(-1, 3)
    bounds = np.asarray(renewal_bounds, dtype=np.float64).reshape(-1)
    spacing = float(particle_spacing)
    if bounds.shape != (6,) or not np.all(np.isfinite(bounds)):
        raise ValueError("renewal_bounds must contain six finite values")
    if np.any(bounds[1::2] <= bounds[::2]):
        raise ValueError("renewal bound upper values must exceed lower values")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("particle_spacing must be finite and positive")
    if not np.all(np.isfinite(position)):
        raise ValueError("particle positions must be finite")

    lower = bounds[::2]
    upper = bounds[1::2]
    bound_scale = np.maximum(1.0, np.maximum(np.abs(lower), np.abs(upper)))
    bound_tolerance = np.maximum(
        64.0 * np.finfo(np.float64).eps * bound_scale,
        1.0e-12 * spacing,
    )
    coordinate_tolerance = np.maximum(
        8.0 * np.abs(np.spacing(position)),
        bound_tolerance[None, :],
    )
    return np.all(
        (position >= lower[None, :] - coordinate_tolerance)
        & (position <= upper[None, :] + coordinate_tolerance),
        axis=1,
    )


def gbd_guard_width(*, particle_spacing: float, diffusion_substeps: int) -> float:
    """Conservative normal reach of one M4' scatter plus GBD Laplacian stages."""
    spacing = float(particle_spacing)
    stages = int(diffusion_substeps)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("particle_spacing must be finite and positive")
    if stages < 1 or stages != diffusion_substeps:
        raise ValueError("diffusion_substeps must be a positive integer")
    return (2.0 + stages) * spacing


def solve_advected_renewal_projection(
    *,
    collocation_position: np.ndarray,
    target_vorticity: np.ndarray,
    particle_position: np.ndarray,
    core_radius: np.ndarray | float,
    prior_vortex_strength: np.ndarray | None = None,
    prior_weight: float = 0.0,
    velocity_position: np.ndarray | None = None,
    target_velocity: np.ndarray | None = None,
    velocity_normal: np.ndarray | None = None,
    velocity_weight: float = 0.0,
    singular_value_cutoff: float | None = None,
) -> AdvectedRenewalProjectionResult:
    """Solve absolute strengths on retained coordinates with dense float64 SVD."""
    particles = np.asarray(particle_position, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_vorticity, dtype=np.float64).reshape(-1, 3)
    basis = gaussian_vorticity_basis(collocation_position, particles, core_radius)
    if len(target) != len(basis) or not np.all(np.isfinite(target)):
        raise ValueError("target_vorticity must be finite and match collocation_position")
    prior_penalty = float(prior_weight)
    velocity_penalty = float(velocity_weight)
    if not np.isfinite(prior_penalty) or prior_penalty < 0.0:
        raise ValueError("prior_weight must be finite and non-negative")
    if not np.isfinite(velocity_penalty) or velocity_penalty < 0.0:
        raise ValueError("velocity_weight must be finite and non-negative")
    cutoff = None if singular_value_cutoff is None else float(singular_value_cutoff)
    if cutoff is not None and (not np.isfinite(cutoff) or cutoff <= 0.0):
        raise ValueError("singular_value_cutoff must be finite and positive")

    vorticity_operator = np.kron(basis, np.eye(3, dtype=np.float64))
    operator_terms = [vorticity_operator]
    target_terms = [target.reshape(-1)]
    prior: np.ndarray | None = None
    if prior_vortex_strength is not None:
        prior = np.asarray(prior_vortex_strength, dtype=np.float64).reshape(-1, 3)
        if prior.shape != particles.shape or not np.all(np.isfinite(prior)):
            raise ValueError("prior_vortex_strength must be finite with shape (N, 3)")
    if prior_penalty > 0.0:
        if prior is None:
            raise ValueError("prior_vortex_strength is required when prior_weight is nonzero")
        operator_terms.append(prior_penalty * np.eye(3 * len(particles)))
        target_terms.append(prior_penalty * prior.reshape(-1))

    velocity_operator: np.ndarray | None = None
    velocity_target: np.ndarray | None = None
    if velocity_penalty > 0.0:
        if velocity_position is None or target_velocity is None:
            raise ValueError("velocity targets are required when velocity_weight is nonzero")
        velocity_operator = gaussian_velocity_operator(
            velocity_position,
            particles,
            core_radius,
            normal=velocity_normal,
        )
        target_velocity_array = np.asarray(target_velocity, dtype=np.float64).reshape(-1, 3)
        if velocity_normal is None:
            velocity_target = target_velocity_array.reshape(-1)
        else:
            normals = np.asarray(velocity_normal, dtype=np.float64).reshape(-1, 3)
            if len(normals) != len(target_velocity_array):
                raise ValueError("target_velocity and velocity_normal counts must match")
            normals = normals / np.linalg.norm(normals, axis=1)[:, None]
            velocity_target = np.einsum("ij,ij->i", normals, target_velocity_array)
        if len(velocity_target) != len(velocity_operator) or not np.all(
            np.isfinite(velocity_target)
        ):
            raise ValueError("target_velocity must be finite and match velocity_position")
        operator_terms.append(velocity_penalty * velocity_operator)
        target_terms.append(velocity_penalty * velocity_target)

    augmented_operator = np.vstack(operator_terms)
    augmented_target = np.concatenate(target_terms)
    if augmented_operator.shape[1] == 0:
        flat_strength = np.empty(0, dtype=np.float64)
        rank = 0
        singular_values = np.empty(0, dtype=np.float64)
    else:
        flat_strength, _residual_sum, rank, singular_values = lstsq(
            augmented_operator,
            augmented_target,
            cond=cutoff,
            lapack_driver="gelsd",
            check_finite=False,
        )
    strength = np.asarray(flat_strength, dtype=np.float64).reshape(-1, 3)
    fitted_vorticity = basis @ strength
    vorticity_error = _relative_norm(fitted_vorticity - target, target)

    fitted_velocity: np.ndarray | None = None
    fitted_normal_velocity: np.ndarray | None = None
    velocity_error: float | None = None
    if velocity_operator is not None and velocity_target is not None:
        flat_velocity = velocity_operator @ flat_strength
        velocity_error = _relative_norm(flat_velocity - velocity_target, velocity_target)
        if velocity_normal is None:
            fitted_velocity = flat_velocity.reshape(-1, 3)
        else:
            fitted_normal_velocity = flat_velocity

    prior_change = None if prior is None else _relative_norm(strength - prior, prior)
    magnitude = np.linalg.norm(strength, axis=1)
    maximum_strength = float(magnitude.max(initial=0.0))
    rms_strength = float(np.sqrt(np.mean(magnitude**2))) if len(magnitude) else 0.0
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if len(singular_values) and singular_values[-1] > 0.0
        else float("inf")
    )
    return AdvectedRenewalProjectionResult(
        vortex_strength=strength,
        fitted_vorticity=fitted_vorticity,
        fitted_velocity=fitted_velocity,
        fitted_normal_velocity=fitted_normal_velocity,
        singular_values=np.asarray(singular_values, dtype=np.float64),
        rank=int(rank),
        condition_number=condition_number,
        vorticity_relative_error=vorticity_error,
        velocity_relative_error=velocity_error,
        prior_relative_change=prior_change,
        maximum_strength=maximum_strength,
        rms_strength=rms_strength,
        maximum_to_rms_strength=(maximum_strength / rms_strength if rms_strength > 0.0 else 0.0),
    )


def select_residual_support_positions(
    *,
    candidate_position: np.ndarray,
    existing_position: np.ndarray,
    collocation_position: np.ndarray,
    vorticity_residual: np.ndarray,
    renewal_bounds: np.ndarray | tuple[float, ...],
    particle_spacing: float,
    residual_fraction: float = 1.0e-3,
    maximum_births: int | None = None,
) -> np.ndarray:
    """Choose separated lattice support only where the current basis is deficient."""
    candidates = np.asarray(candidate_position, dtype=np.float64).reshape(-1, 3)
    existing = np.asarray(existing_position, dtype=np.float64).reshape(-1, 3)
    collocation = np.asarray(collocation_position, dtype=np.float64).reshape(-1, 3)
    residual = np.asarray(vorticity_residual, dtype=np.float64).reshape(-1, 3)
    spacing = float(particle_spacing)
    fraction = float(residual_fraction)
    if len(collocation) != len(residual):
        raise ValueError("vorticity_residual must match collocation_position")
    if not np.isfinite(fraction) or fraction < 0.0:
        raise ValueError("residual_fraction must be finite and non-negative")
    if maximum_births is not None and maximum_births < 1:
        raise ValueError("maximum_births must be positive when supplied")
    if not len(candidates) or not len(collocation):
        return np.empty((0, 3), dtype=np.float64)
    inside = geometric_renewal_mask(
        candidates,
        renewal_bounds,
        particle_spacing=spacing,
    )
    candidates = candidates[inside]
    if not len(candidates):
        return np.empty((0, 3), dtype=np.float64)

    distance_tolerance = 1.0e-12 * spacing
    if len(existing):
        nearest_existing = KDTree(existing).query(candidates, k=1)[0]
        candidates = candidates[nearest_existing >= spacing - distance_tolerance]
    if not len(candidates):
        return np.empty((0, 3), dtype=np.float64)

    tree = KDTree(collocation)
    _distance, nearest = tree.query(candidates, k=1)
    score = np.linalg.norm(residual[nearest], axis=1)
    peak = float(np.linalg.norm(residual, axis=1).max(initial=0.0))
    if peak == 0.0:
        return np.empty((0, 3), dtype=np.float64)
    eligible = score >= fraction * peak
    candidates = candidates[eligible]
    score = score[eligible]
    if not len(candidates):
        return np.empty((0, 3), dtype=np.float64)

    order = np.argsort(-score, kind="stable")
    accepted: list[np.ndarray] = []
    accepted_bins: dict[tuple[int, int, int], list[int]] = {}
    neighbour_offsets = tuple((i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1))
    for index in order:
        point = candidates[index]
        cell_values = np.floor(point / spacing).astype(np.int64)
        cell = (int(cell_values[0]), int(cell_values[1]), int(cell_values[2]))
        nearby = [
            accepted[accepted_index]
            for offset in neighbour_offsets
            for accepted_index in accepted_bins.get(
                (cell[0] + offset[0], cell[1] + offset[1], cell[2] + offset[2]),
                (),
            )
        ]
        if nearby and np.min(np.linalg.norm(np.asarray(nearby) - point, axis=1)) < (
            spacing - distance_tolerance
        ):
            continue
        accepted.append(point.copy())
        accepted_bins.setdefault(cell, []).append(len(accepted) - 1)
        if maximum_births is not None and len(accepted) >= maximum_births:
            break
    return np.asarray(accepted, dtype=np.float64).reshape(-1, 3)


def project_gbd_renewal_basis(
    *,
    collocation_position: np.ndarray,
    target_vorticity: np.ndarray,
    particle_position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray | float,
    renewal_bounds: np.ndarray | tuple[float, ...],
    particle_spacing: float,
    velocity_position: np.ndarray | None = None,
    target_velocity: np.ndarray | None = None,
    velocity_normal: np.ndarray | None = None,
    velocity_weight: float = 0.0,
    prior_weight: float = 0.0,
    maximum_vorticity_error: float = 5.0e-3,
    maximum_velocity_error: float = 1.0e-3,
    support_candidate_position: np.ndarray | None = None,
    support_core_radius: float | None = None,
    maximum_births: int | None = None,
) -> GBDRenewalProjectionResult:
    """Project the FVM residual on the current post-GBD basis.

    Preserved strengths are copied exactly.  A second basis is never created
    after a successful current-basis solve.  If that basis is insufficient,
    only residual-backed, spacing-safe support candidates are admitted and the
    absolute solve is repeated once.
    """
    position = np.asarray(particle_position, dtype=np.float64).reshape(-1, 3)
    strength = np.asarray(vortex_strength, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_vorticity, dtype=np.float64).reshape(-1, 3)
    collocation = np.asarray(collocation_position, dtype=np.float64).reshape(-1, 3)
    if strength.shape != position.shape or not np.all(np.isfinite(strength)):
        raise ValueError("vortex_strength must be finite with shape (N, 3)")
    if len(target) != len(collocation):
        raise ValueError("target_vorticity must match collocation_position")
    radius = _core_radius_array(core_radius, len(position))
    omega_limit = float(maximum_vorticity_error)
    velocity_limit = float(maximum_velocity_error)
    if not np.isfinite(omega_limit) or omega_limit < 0.0:
        raise ValueError("maximum_vorticity_error must be finite and non-negative")
    if not np.isfinite(velocity_limit) or velocity_limit < 0.0:
        raise ValueError("maximum_velocity_error must be finite and non-negative")

    renewable = geometric_renewal_mask(
        position,
        renewal_bounds,
        particle_spacing=particle_spacing,
    )
    preserved = ~renewable
    preserved_vorticity = (
        gaussian_vorticity_basis(
            collocation,
            position[preserved],
            radius[preserved],
        )
        @ strength[preserved]
    )
    renewable_target = target - preserved_vorticity

    renewable_velocity_target: np.ndarray | None = None
    if float(velocity_weight) > 0.0:
        if velocity_position is None or target_velocity is None:
            raise ValueError("velocity targets are required when velocity_weight is nonzero")
        velocity_points = np.asarray(velocity_position, dtype=np.float64).reshape(-1, 3)
        velocity_values = np.asarray(target_velocity, dtype=np.float64).reshape(-1, 3)
        preserved_velocity = (
            gaussian_velocity_operator(
                velocity_points,
                position[preserved],
                radius[preserved],
            )
            @ strength[preserved].reshape(-1)
        ).reshape(-1, 3)
        renewable_velocity_target = velocity_values - preserved_velocity

    def solve(
        solve_position: np.ndarray,
        solve_radius: np.ndarray,
        solve_prior: np.ndarray,
    ) -> AdvectedRenewalProjectionResult:
        return solve_advected_renewal_projection(
            collocation_position=collocation,
            target_vorticity=renewable_target,
            particle_position=solve_position,
            core_radius=solve_radius,
            prior_vortex_strength=solve_prior,
            prior_weight=prior_weight,
            velocity_position=velocity_position,
            target_velocity=renewable_velocity_target,
            velocity_normal=velocity_normal,
            velocity_weight=velocity_weight,
        )

    solve_position = position[renewable]
    solve_radius = radius[renewable]
    solve_prior = strength[renewable]
    projection = solve(solve_position, solve_radius, solve_prior)

    def passes(result: AdvectedRenewalProjectionResult) -> bool:
        velocity_ok = (
            result.velocity_relative_error is None
            or result.velocity_relative_error <= velocity_limit
        )
        return result.vorticity_relative_error <= omega_limit and velocity_ok

    birth_position = np.empty((0, 3), dtype=np.float64)
    used_births = False
    if not passes(projection) and support_candidate_position is not None:
        birth_position = select_residual_support_positions(
            candidate_position=support_candidate_position,
            existing_position=position,
            collocation_position=collocation,
            vorticity_residual=renewable_target - projection.fitted_vorticity,
            renewal_bounds=renewal_bounds,
            particle_spacing=particle_spacing,
            maximum_births=maximum_births,
        )
        if len(birth_position):
            if support_core_radius is None and not len(radius):
                raise ValueError(
                    "support_core_radius is required when the current VPM cloud is empty"
                )
            birth_sigma = (
                float(support_core_radius)
                if support_core_radius is not None
                else float(np.median(radius))
            )
            if not np.isfinite(birth_sigma) or birth_sigma <= 0.0:
                raise ValueError("support_core_radius must be finite and positive")
            solve_position = np.vstack((solve_position, birth_position))
            solve_radius = np.concatenate((solve_radius, np.full(len(birth_position), birth_sigma)))
            solve_prior = np.vstack((solve_prior, np.zeros((len(birth_position), 3))))
            projection = solve(solve_position, solve_radius, solve_prior)
            used_births = True

    if not passes(projection):
        velocity_text = (
            "not fitted"
            if projection.velocity_relative_error is None
            else f"{projection.velocity_relative_error:.6%}"
        )
        raise RuntimeError(
            "post-GBD renewal basis cannot meet the field gate: "
            f"vorticity={projection.vorticity_relative_error:.6e}, "
            f"velocity={velocity_text}, births={len(birth_position)}"
        )

    updated = strength.copy()
    renewable_count = int(np.count_nonzero(renewable))
    updated[renewable] = projection.vortex_strength[:renewable_count]
    birth_strength = projection.vortex_strength[renewable_count:]
    birth_radius = solve_radius[renewable_count:]
    return GBDRenewalProjectionResult(
        renewable_mask=renewable,
        preserved_mask=preserved,
        updated_vortex_strength=updated,
        birth_position=birth_position,
        birth_vortex_strength=birth_strength,
        birth_core_radius=birth_radius,
        projection=projection,
        used_selective_births=used_births,
    )


__all__ = [
    "AdvectedRenewalProjectionResult",
    "GBDRenewalProjectionResult",
    "SparseRenewalProjectionResult",
    "evaluate_sparse_gaussian_vorticity",
    "gaussian_velocity_operator",
    "gaussian_vorticity_basis",
    "gaussian_vorticity_divergence_operator",
    "gbd_guard_width",
    "geometric_renewal_mask",
    "project_gbd_renewal_basis",
    "select_residual_support_positions",
    "solve_advected_renewal_projection",
    "solve_sparse_renewal_projection",
    "sparse_gaussian_vorticity_basis",
]
