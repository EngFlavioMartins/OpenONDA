"""UNWIRED bounded finite-image Gaussian particle-mesh reference.

This is a proof-of-feasibility, NOT a runtime induction backend. Original and
axially reflected source families are compact. Their smooth kernels contain
the EXACT SUPPLIED FINITE image shifts before FFT embedding; images are never
folded modulo the FFT period. Zero-padding embeds the discrete linear
convolution for every needed source-target grid displacement.

Off-grid particles use tensor-product Lagrange assignment. Target velocity
and its full Jacobian are first/second derivatives of the SAME interpolated
vector potential; no finite-difference or incompressibility repair is used.
These piecewise polynomial interpolants are not globally C2 across stencil
changes. Their gridding/interpolation/FFT errors require explicit convergence
qualification and are NOT enclosed by the local-core truncation bound.

Every image uses the original SOURCE core. The Gaussian identity
K_sigma = K_tau + (K_sigma-K_tau) supplies the separate local correction.
No physical primary with target/source averaged cores is silently introduced.
The primary is present only if explicitly requested as image (0, False), in
which case this generic-target reference still uses source-only cores.
"""

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np
from scipy.fft import irfftn, next_fast_len, rfftn
from scipy.special import erf

from tests.vpm._gaussian_broadening_reference import broadening_tail_bound, correction_fields
from tests.vpm._slip_periodic_gaussian_oracle import gaussian_pairs


@dataclass(frozen=True)
class FiniteImageMeshResult:
    velocity: np.ndarray
    gradient: np.ndarray
    smooth_velocity: np.ndarray
    smooth_gradient: np.ndarray
    correction_velocity: np.ndarray
    correction_gradient: np.ndarray
    correction_velocity_tail: np.ndarray
    correction_gradient_tail: np.ndarray
    diagnostics: dict


def _positive_cap(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer work cap")
    return int(value)


def _inputs(position, strength, core, targets, images, tau, max_images):
    position, strength, core, targets = (
        np.asarray(value, dtype=np.float64) for value in (position, strength, core, targets)
    )
    if (position.ndim != 2 or position.shape[1:] != (3,) or strength.shape != position.shape
            or core.shape != (len(position),) or targets.ndim != 2 or targets.shape[1:] != (3,)):
        raise ValueError("expected source/strength/target vectors and one source core per particle")
    if not all(np.isfinite(value).all() for value in (position, strength, core, targets)):
        raise ValueError("finite particle and target inputs required")
    if not math.isfinite(tau) or tau <= 0 or np.any(core <= 0) or np.any(core > tau):
        raise ValueError("require 0 < every source core <= finite common broadening tau")
    supplied = []
    for shift, odd in images:
        if len(supplied) >= max_images:
            raise ValueError("finite image count exceeds explicit qualification cap")
        if type(odd) not in (bool, np.bool_) or not math.isfinite(float(shift)):
            raise ValueError("each finite image must be (finite shift, boolean axial reflection)")
        supplied.append((float(shift), bool(odd)))
    return position, strength, core, targets, tuple(supplied)


def _image_sources(position, strength, shift, odd):
    points, vectors = position.copy(), strength.copy()
    if odd:
        points[:, 2] *= -1
        vectors[:, :2] *= -1
    points[:, 2] += shift
    return points, vectors


def direct_finite_images(position, strength, core, targets, images, *, max_pairs=200_000, max_images=513):
    """Independent small-cloud direct u/J oracle, retaining finite self J."""
    tau = max(float(np.max(core, initial=0)), 1e-300)
    max_pairs, max_images = _positive_cap(max_pairs, "max_pairs"), _positive_cap(max_images, "max_images")
    x, gamma, sigma, query, images = _inputs(position, strength, core, targets, images, tau, max_images)
    if len(x) * len(query) * len(images) > max_pairs:
        raise ValueError("direct qualification pair budget exceeded")
    velocity, gradient = np.zeros((len(query), 3)), np.zeros((len(query), 3, 3))
    conditioning = np.zeros((len(query), 2))
    for shift, odd in images:
        sources, strengths = _image_sources(x, gamma, shift, odd)
        u, j = gaussian_pairs(query[:, None] - sources[None], strengths[None], sigma[None])
        velocity += u.sum(axis=1)
        gradient += j.sum(axis=1)
        conditioning[:, 0] += np.linalg.norm(u, axis=-1).sum(axis=1)
        conditioning[:, 1] += np.linalg.norm(j, axis=(-2, -1)).sum(axis=1)
    if not all(np.isfinite(field).all() for field in (velocity, gradient, conditioning)):
        raise FloatingPointError("nonfinite direct finite-image reference result")
    return velocity, gradient, conditioning


def _stencil(points, origin, spacing, order):
    coordinate = (points - origin) / spacing
    first = np.floor(coordinate).astype(np.int64) - order // 2 + 1
    argument = coordinate - first
    weights = np.empty((len(points), 3, 3, order), np.float64)
    for index in range(order):
        others = np.delete(np.arange(order, dtype=np.float64), index)
        polynomial = np.polynomial.Polynomial.fromroots(others)
        polynomial /= np.prod(index - others)
        for derivative in range(3):
            weights[:, :, derivative, index] = polynomial.deriv(derivative)(argument) / spacing**derivative
    if not np.isfinite(weights).all():
        raise FloatingPointError("nonfinite stencil arithmetic: rescale the qualification problem")
    return first, weights


def _validate_stencil(first, order, shape):
    # Large finite world offsets can erase the padding during subtraction.
    # Negative NumPy indices would silently wrap, corrupting linear convolution.
    if (first.ndim != 2 or first.shape[1:] != (3,) or np.any(first < 0)
            or np.any(first > np.asarray(shape)-order)):
        raise ValueError("source/target stencil is outside the admitted linear-convolution grid")


def _assign(shape, first, weights, strength):
    _validate_stencil(first, weights.shape[-1], shape)
    grid = np.zeros((*shape, 3), np.float64)
    order = weights.shape[-1]
    for a in range(order):
        for b in range(order):
            for c in range(order):
                indices = first + (a, b, c)
                weight = weights[:, 0, 0, a] * weights[:, 1, 0, b] * weights[:, 2, 0, c]
                np.add.at(grid, tuple(indices.T), weight[:, None] * strength)
    return grid


def _gaussian_potential(radius, tau):
    result = np.full_like(radius, 1 / (2 * math.pi**1.5 * tau))
    nonzero = radius > 0
    result[nonzero] = erf(radius[nonzero] / tau) / (4 * math.pi * radius[nonzero])
    return result


def _finite_kernel(shape, fft_shape, spacing, shifts, tau):
    # For density indices j and requested output indices i, i-j belongs to
    # [-(n-1),n-1]. Distinct such lags do not alias with padding >=2*n-1.
    coordinates, valid = [], []
    for size, padded in zip(shape, fft_shape, strict=True):
        index = np.arange(padded)
        coordinates.append(np.where(index < size, index, index - padded) * spacing)
        valid.append((index < size) | (index >= padded - size + 1))
    xx, yy, zz = np.meshgrid(*coordinates, indexing="ij", sparse=True)
    mask = valid[0][:, None, None] & valid[1][None, :, None] & valid[2][None, None, :]
    kernel = np.zeros(fft_shape, np.float64)
    for shift in shifts:
        radius = np.sqrt(xx * xx + yy * yy + (zz - shift)**2)
        values = _gaussian_potential(radius, tau)
        # A constant potential has identically zero u/J. Removing the same
        # centre value from every lag improves distant-block conditioning;
        # this is not a removal of a physical zero Fourier mode.
        gauge = float(_gaussian_potential(np.array(abs(shift)), tau))
        kernel += np.where(mask, values - gauge, 0)
    return kernel


def _gather_derivatives(potential, first, weights):
    _validate_stencil(first, weights.shape[-1], potential.shape[:3])
    count, order = len(first), weights.shape[-1]
    gradient = np.zeros((count, 3, 3))  # derivative axis, vector component
    hessian = np.zeros((count, 3, 3, 3))
    for a in range(order):
        for b in range(order):
            for c in range(order):
                indices = first + (a, b, c)
                value = potential[tuple(indices.T)]
                w = (weights[:, 0, :, a], weights[:, 1, :, b], weights[:, 2, :, c])
                for axis in range(3):
                    weight = np.ones(count)
                    for d in range(3):
                        weight *= w[d][:, int(d == axis)]
                    gradient[:, axis] += weight[:, None] * value
                    for second in range(axis, 3):
                        weight = np.ones(count)
                        for d in range(3):
                            weight *= w[d][:, int(d == axis) + int(d == second)]
                        contribution = weight[:, None] * value
                        hessian[:, axis, second] += contribution
                        if second != axis:
                            hessian[:, second, axis] += contribution
    velocity = np.stack((gradient[:, 1, 2] - gradient[:, 2, 1],
                         gradient[:, 2, 0] - gradient[:, 0, 2],
                         gradient[:, 0, 1] - gradient[:, 1, 0]), axis=1)
    jacobian = np.stack((hessian[:, :, 1, 2] - hessian[:, :, 2, 1],
                         hessian[:, :, 2, 0] - hessian[:, :, 0, 2],
                         hessian[:, :, 0, 1] - hessian[:, :, 1, 0]), axis=1)
    return velocity, jacobian


def finite_image_mesh(
    position, strength, core, targets, images, *, tau, spacing, order=6,
    correction_cutoff=None, max_grid_nodes=2_000_000, max_pairs=200_000, max_images=513,
):
    """Finite smooth mesh + explicit actual-source-core correction.

    No mesh error threshold is used to accept a result. ``diagnostics`` always
    reports it as unqualified, independently of observed mesh refinement.
    The correction cutoff is optional; without it all correction pairs are
    evaluated and its analytic omitted-pair bound is exactly zero.
    """
    max_grid_nodes = _positive_cap(max_grid_nodes, "max_grid_nodes")
    max_pairs, max_images = _positive_cap(max_pairs, "max_pairs"), _positive_cap(max_images, "max_images")
    x, gamma, sigma, query, images = _inputs(position, strength, core, targets, images, float(tau), max_images)
    if (not math.isfinite(spacing) or spacing <= 0 or not isinstance(order, int)
            or not 4 <= order <= 10 or order % 2):
        raise ValueError("require positive finite spacing and even interpolation order4..10")
    if correction_cutoff is not None and (not math.isfinite(correction_cutoff) or correction_cutoff <= 0):
        raise ValueError("correction cutoff must be positive finite or None")
    pair_count = len(x) * len(query) * len(images)
    if pair_count > max_pairs:
        raise ValueError("small-cloud qualification pair budget exceeded")
    u, j = np.zeros((len(query), 3)), np.zeros((len(query), 3, 3))
    du, dj, vu, vj = u.copy(), j.copy(), np.zeros(len(query)), np.zeros(len(query))
    diagnostics = {"fields_runtime_qualified": False, "finite_image_count": len(images),
                   "finite_images": images, "infinite_periodic_operator": False,
                   "source_only_cores": True, "spacing": spacing, "order": order, "tau": tau,
                   "mesh_error_certified": False, "roundoff_certified": False,
                   "pair_count": pair_count, "correction_pairs": 0,
                   "potential_derivatives_consistent_within_stencil": True,
                   "interpolant_globally_C2": False}
    if not len(x) or not len(query) or not images:
        return FiniteImageMeshResult(u, j, u.copy(), j.copy(), du, dj, vu, vj, diagnostics)
    # Source reflection about zero is merely a representation. Recenter the
    # common z coordinate so translating a slab does not separate compact
    # original/reflected density grids by twice an arbitrary world offset.
    world_x, world_query, world_images = x, query, images
    centre_z = 0.5 * min(x[:, 2].min(), query[:, 2].min()) + 0.5 * max(x[:, 2].max(), query[:, 2].max())
    x, query = x.copy(), query.copy()
    x[:, 2] -= centre_z
    query[:, 2] -= centre_z
    images = tuple((shift - 2 * centre_z if odd else shift, odd) for shift, odd in images)
    if (not np.isfinite(x).all() or not np.isfinite(query).all()
            or not all(math.isfinite(shift) for shift, _ in images)):
        raise FloatingPointError("nonfinite recentered geometry")
    families = {}
    for odd in (False, True):
        shifts = [shift for shift, reflected in images if reflected == odd]
        if shifts:
            family_x, family_g = _image_sources(x, gamma, 0, odd)
            families[odd] = family_x, family_g, shifts
    all_points = np.concatenate([query, *(entry[0] for entry in families.values())])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        origin = np.floor(all_points.min(axis=0) / spacing) * spacing - order * spacing
        dimensions = np.ceil((all_points.max(axis=0) - origin) / spacing) + order + 1
    if not np.isfinite(dimensions).all() or np.any(dimensions > max_grid_nodes) or np.any(dimensions < 1):
        raise ValueError("bounded FFT grid exceeded before shape allocation")
    shape = tuple(dimensions.astype(np.int64).tolist())
    fft_shape = tuple(next_fast_len(2 * n - 1) for n in shape)
    if math.prod(fft_shape) > max_grid_nodes:
        raise ValueError(f"bounded FFT grid exceeded: {fft_shape}, {math.prod(fft_shape)} nodes")
    potential = np.zeros((*shape, 3), np.float64)
    window = tuple(slice(0, n) for n in shape)
    for family_x, family_g, shifts in families.values():
        first, weights = _stencil(family_x, origin, spacing, order)
        density = _assign(shape, first, weights, family_g)
        kernel_hat = rfftn(_finite_kernel(shape, fft_shape, spacing, shifts, tau), workers=1)
        for component in range(3):
            transformed = rfftn(density[..., component], s=fft_shape, workers=1)
            potential[..., component] += irfftn(transformed * kernel_hat, s=fft_shape, workers=1)[window]
    first, weights = _stencil(query, origin, spacing, order)
    u, j = _gather_derivatives(potential, first, weights)
    for world_shift, odd in world_images:
        sources, strengths = _image_sources(world_x, gamma, world_shift, odd)
        for target, point in enumerate(world_query):
            for source, vector, radius in zip(sources, strengths, sigma, strict=True):
                displacement = point - source
                distance = float(np.linalg.norm(displacement))
                if correction_cutoff is None or distance < correction_cutoff:
                    dv, dg = correction_fields(displacement, vector, radius, tau)
                    du[target] += dv
                    dj[target] += dg
                    diagnostics["correction_pairs"] += 1
                else:
                    bound = broadening_tail_bound(distance, radius, tau, np.linalg.norm(vector))
                    vu[target] += bound.velocity
                    vj[target] += bound.gradient
    diagnostics.update(grid_shape=shape, fft_shape=fft_shape, fft_nodes=math.prod(fft_shape),
                       families=len(families), no_discrete_linear_convolution_alias=True,
                       potential_gauge_removed=True, exact_coincidence_override=False,
                       z_recentering=centre_z)
    if not all(np.isfinite(field).all() for field in (u, j, du, dj, vu, vj, u+du, j+dj)):
        raise FloatingPointError("nonfinite finite-image mesh result: rescale qualification inputs")
    return FiniteImageMeshResult(u + du, j + dj, u, j, du, dj, vu, vj, diagnostics)


def observed_error(candidate, exact, conditioning):
    """No relative claim at an exactly zero norm; retain absolute/pointwise data."""
    candidate, exact, conditioning = (np.asarray(field, dtype=float) for field in (candidate, exact, conditioning))
    if (candidate.shape != exact.shape or candidate.ndim not in (2, 3)
            or candidate.shape[1:] not in ((3,), (3, 3))
            or conditioning.shape != (len(candidate),)
            or not all(np.isfinite(field).all() for field in (candidate, exact, conditioning))
            or np.any(conditioning < 0)):
        raise ValueError("identical finite u/J arrays and nonnegative per-target conditioning required")
    difference = candidate - exact
    axes = tuple(range(1, difference.ndim))
    pointwise = np.linalg.norm(difference, axis=axes)
    denominator = float(np.linalg.norm(exact))
    return {"absolute_l2": float(np.linalg.norm(difference)),
            "relative_l2": None if denominator == 0 else float(np.linalg.norm(difference) / denominator),
            "pointwise_absolute": pointwise.tolist(), "conditioning": np.asarray(conditioning).tolist()}
