"""UNWIRED analytic-field finite-image particle-mesh alternative.

This separate reference leaves the failed potential-derivative interpolation
screen untouched. It convolves sampled analytic Gaussian gradient/Hessian
kernels with source strengths, then gathers u and full J with the SAME
undifferentiated Lagrange weights. It needs more inverse FFTs but does not
differentiate interpolation error. J is NOT identically the derivative of
the interpolated u; their discrepancy is a required independent screen.

Odd zero-shift kernels cancel coincident same-stencil velocity as W^T K W.
This is NOT an exact-self guarantee for arbitrary reflected image shifts:
physically coincident pairs can have different compact-grid mesh phases.
There is no coincidence override, implicit primary, infinite image sum,
periodic folding, production import, or runtime accuracy certificate.
"""

import math

import numpy as np
from scipy.fft import irfftn, next_fast_len, rfftn
from scipy.special import erf

from tests.vpm._finite_image_mesh_reference import (
    FiniteImageMeshResult,
    _assign,
    _image_sources,
    _inputs,
    _positive_cap,
    _stencil,
    _validate_stencil,
)
from tests.vpm._gaussian_broadening_reference import broadening_tail_bound, correction_fields


def analytic_gaussian_derivatives(displacement, tau):
    """Independent vectorized analytic grad/Hess of erf(r/tau)/(4*pi*r)."""
    displacement = np.asarray(displacement, dtype=np.float64)
    if displacement.shape[-1:] != (3,) or not np.isfinite(displacement).all() or not math.isfinite(tau) or tau <= 0:
        raise ValueError("finite displacements and positive broadening required")
    radius2 = np.sum(displacement**2, axis=-1)
    rho2 = radius2/tau**2
    near = rho2 < 1
    a = np.empty_like(radius2)
    b = np.empty_like(radius2)
    coefficients = np.array([(-1.0)**n/(math.factorial(n)*(2*n+3)) for n in range(24)])
    derivative = np.array([-2*n*coefficients[n] for n in range(1, 24)])
    a[near] = math.pi**-1.5/tau**3*np.polynomial.polynomial.polyval(rho2[near], coefficients)
    b[near] = math.pi**-1.5/tau**5*np.polynomial.polynomial.polyval(rho2[near], derivative)
    radius = np.sqrt(radius2[~near])
    rho = radius/tau
    density = np.exp(-rho*rho)/(math.pi**1.5*tau**3)
    q = (erf(rho)-2*rho*np.exp(-rho*rho)/math.sqrt(math.pi))/(4*math.pi)
    a[~near] = q/radius**3
    b[~near] = 3*q/radius**5-density/radius**2
    gradient = -a[..., None]*displacement
    hessian = b[..., None, None]*displacement[..., :, None]*displacement[..., None, :]
    hessian -= a[..., None, None]*np.eye(3)
    if not np.isfinite(gradient).all() or not np.isfinite(hessian).all():
        raise FloatingPointError("nonfinite analytic Gaussian derivatives")
    return gradient, hessian


def _field_kernels(shape, fft_shape, spacing, shifts, tau):
    coordinates, valid = [], []
    for size, padded in zip(shape, fft_shape, strict=True):
        index = np.arange(padded)
        coordinates.append(np.where(index < size, index, index-padded)*spacing)
        valid.append((index < size) | (index >= padded-size+1))
    displacement = np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1)
    mask = valid[0][:, None, None] & valid[1][None, :, None] & valid[2][None, None, :]
    gradient, hessian = np.zeros((*fft_shape, 3)), np.zeros((*fft_shape, 3, 3))
    for shift in shifts:
        current = displacement.copy()
        current[..., 2] -= shift
        g, h = analytic_gaussian_derivatives(current, tau)
        gradient += np.where(mask[..., None], g, 0)
        hessian += np.where(mask[..., None, None], h, 0)
    return gradient, hessian


def _gather_fields(fields, first, weights):
    _validate_stencil(first, weights.shape[-1], fields.shape[:3])
    count, order = len(first), weights.shape[-1]
    result = np.zeros((count, fields.shape[-1]))
    for a in range(order):
        for b in range(order):
            for c in range(order):
                indices = first+(a, b, c)
                weight = weights[:, 0, 0, a]*weights[:, 1, 0, b]*weights[:, 2, 0, c]
                result += weight[:, None]*fields[tuple(indices.T)]
    return result


def finite_image_field_mesh(
    position, strength, core, targets, images, *, tau, spacing, order=8,
    correction_cutoff=None, max_grid_nodes=2_000_000, max_pairs=200_000, max_images=513,
):
    """Analytic-field convolution plus exact source-core correction; CPU only."""
    max_grid_nodes = _positive_cap(max_grid_nodes, "max_grid_nodes")
    max_pairs, max_images = _positive_cap(max_pairs, "max_pairs"), _positive_cap(max_images, "max_images")
    x, gamma, sigma, query, images = _inputs(position, strength, core, targets, images, float(tau), max_images)
    if not math.isfinite(spacing) or spacing <= 0 or not isinstance(order, int) or order not in (4, 6, 8, 10):
        raise ValueError("positive finite spacing and even order4..10 required")
    if correction_cutoff is not None and (not math.isfinite(correction_cutoff) or correction_cutoff <= 0):
        raise ValueError("positive finite correction cutoff or None required")
    pairs = len(x)*len(query)*len(images)
    if pairs > max_pairs:
        raise ValueError("small-cloud qualification pair budget exceeded")
    u, j = np.zeros((len(query), 3)), np.zeros((len(query), 3, 3))
    du, dj, vu, vj = u.copy(), j.copy(), np.zeros(len(query)), np.zeros(len(query))
    diagnostics = {"fields_runtime_qualified": False, "mesh_error_certified": False, "roundoff_certified": False,
                   "analytic_field_convolution": True, "interpolated_u_J_derivative_consistent": False,
                   "source_only_cores": True, "infinite_periodic_operator": False,
                   "finite_images": images, "finite_image_count": len(images), "exact_coincidence_override": False,
                   "spacing": spacing, "tau": tau, "order": order, "pair_count": pairs, "correction_pairs": 0}
    if not len(x) or not len(query) or not images:
        return FiniteImageMeshResult(u, j, u.copy(), j.copy(), du, dj, vu, vj, diagnostics)
    world_x, world_query, world_images = x, query, images
    centre_z = 0.5*min(x[:, 2].min(), query[:, 2].min())+0.5*max(x[:, 2].max(), query[:, 2].max())
    x, query = x.copy(), query.copy()
    x[:, 2] -= centre_z
    query[:, 2] -= centre_z
    images = tuple((shift-2*centre_z if odd else shift, odd) for shift, odd in images)
    if not np.isfinite(x).all() or not np.isfinite(query).all() or not all(math.isfinite(s) for s, _ in images):
        raise FloatingPointError("nonfinite recentered geometry")
    families = []
    for odd in (False, True):
        shifts = [shift for shift, reflected in images if reflected == odd]
        if shifts:
            family_x, family_g = _image_sources(x, gamma, 0, odd)
            families.append((family_x, family_g, shifts))
    all_points = np.concatenate([query, *(family[0] for family in families)])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        origin = np.floor(all_points.min(axis=0)/spacing)*spacing-order*spacing
        dimensions = np.ceil((all_points.max(axis=0)-origin)/spacing)+order+1
    if not np.isfinite(dimensions).all() or np.any(dimensions > max_grid_nodes) or np.any(dimensions < 1):
        raise ValueError("bounded FFT grid exceeded before shape allocation")
    shape = tuple(dimensions.astype(np.int64).tolist())
    fft_shape = tuple(next_fast_len(2*n-1) for n in shape)
    if math.prod(fft_shape) > max_grid_nodes:
        raise ValueError("bounded FFT grid exceeded")
    fields = np.zeros((*shape, 12))
    window = tuple(slice(0, size) for size in shape)
    for family_x, family_g, shifts in families:
        first, weights = _stencil(family_x, origin, spacing, order)
        density = _assign(shape, first, weights, family_g)
        density_hat = [rfftn(density[..., c], s=fft_shape, workers=1) for c in range(3)]
        kernel_g, kernel_h = _field_kernels(shape, fft_shape, spacing, shifts, tau)
        gradient_hat = [rfftn(kernel_g[..., c], workers=1) for c in range(3)]
        hessian_hat = {(a, b): rfftn(kernel_h[..., a, b], workers=1)
                       for a in range(3) for b in range(a, 3)}
        for component, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
            fields[..., component] += irfftn(gradient_hat[a]*density_hat[b]-gradient_hat[b]*density_hat[a],
                                             s=fft_shape, workers=1)[window]
            for derivative in range(3):
                first_h = hessian_hat[tuple(sorted((a, derivative)))]
                second_h = hessian_hat[tuple(sorted((b, derivative)))]
                fields[..., 3+3*component+derivative] += irfftn(first_h*density_hat[b]-second_h*density_hat[a],
                                                               s=fft_shape, workers=1)[window]
    first, weights = _stencil(query, origin, spacing, order)
    values = _gather_fields(fields, first, weights)
    u, j = values[:, :3], values[:, 3:].reshape(-1, 3, 3)
    for shift, odd in world_images:
        sources, vectors = _image_sources(world_x, gamma, shift, odd)
        for target, point in enumerate(world_query):
            for source, vector, radius in zip(sources, vectors, sigma, strict=True):
                displacement = point-source
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
    if not all(np.isfinite(value).all() for value in (u, j, du, dj, vu, vj, u+du, j+dj)):
        raise FloatingPointError("nonfinite analytic-field mesh result")
    diagnostics.update(grid_shape=shape, fft_shape=fft_shape, fft_nodes=math.prod(fft_shape),
                       families=len(families), inverse_transforms=12*len(families), z_recentering=centre_z,
                       no_discrete_linear_convolution_alias=True)
    return FiniteImageMeshResult(u+du, j+dj, u.copy(), j.copy(), du, dj, vu, vj, diagnostics)
