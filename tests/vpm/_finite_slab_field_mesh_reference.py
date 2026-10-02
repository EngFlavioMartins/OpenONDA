"""UNWIRED finite-image analytic-field mesh with slab-commensurate grid.

Slab images are explicit integer descriptors (k, odd): original sources at
z+2*k*L or axially reflected sources at 2*zmin-z+2*k*L. Let N=ceil(L/h),
h_z=L/N and t_z=N*((z-zmin)/L-1/2). The image displacements are then EXACT
INTEGER indices 2*k*N or (2*k-1)*N in the auxiliary grid. Physical particles
and slab width are not rounded; h_z may differ from requested x/y spacing.

This is a separate qualification reference, not a repair of the preserved
arbitrary-phase reference. Alignment removes its artificial image/grid
phase mismatch but does not certify interpolation accuracy, derivative
consistency, roundoff, FFT cost, or a production backend. All those remain
independent tests; physical particle self-FMM is outside this image API.
"""

import math
from numbers import Integral

import numpy as np
from scipy.fft import irfftn, next_fast_len, rfftn

from tests.vpm._finite_image_field_mesh_reference import (
    _gather_fields,
    analytic_gaussian_derivatives,
)
from tests.vpm._finite_image_mesh_reference import (
    FiniteImageMeshResult,
    _assign,
    _image_sources,
    _inputs,
    _positive_cap,
    _stencil,
)
from tests.vpm._gaussian_broadening_reference import broadening_tail_bound, correction_fields


def slab_coordinates(position, zmin, zmax, spacing):
    """World-to-auxiliary coordinates without snapping any physical position."""
    points = np.asarray(position, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
            or not all(math.isfinite(value) for value in (zmin, zmax, spacing))
            or spacing <= 0 or zmax <= zmin):
        raise ValueError("finite positions, ordered finite slab bounds and positive spacing required")
    width = zmax-zmin
    if not math.isfinite(width) or not math.isfinite(width/spacing):
        raise ValueError("nonfinite slab-to-grid ratio")
    count = math.ceil(width/spacing)
    if not 1 <= count <= 2**30:
        raise ValueError("slab-to-grid cell count outside bounded qualification range")
    steps = np.array([spacing, spacing, width/count])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = points/steps
        result[:, 2] = count*((points[:, 2]-zmin)/width-.5)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite normalized slab coordinates")
    return result, steps, count


def image_descriptor_shift(k, odd, count):
    if isinstance(k, bool) or not isinstance(k, Integral) or type(odd) not in (bool, np.bool_):
        raise ValueError("slab image descriptor requires integer k and boolean parity")
    count = _positive_cap(count, "slab cells")
    shift = (2*int(k)-int(odd))*count
    if abs(shift) > 2**52:
        raise ValueError("integer image shift exceeds exact f64 qualification range")
    return shift


def slab_world_images(descriptors, zmin, zmax, count, max_images=513):
    max_images = _positive_cap(max_images, "max_images")
    result, integer_images = [], []
    width = zmax-zmin
    for k, odd in descriptors:
        if len(result) >= max_images:
            raise ValueError("finite image count exceeds explicit qualification cap")
        shift = image_descriptor_shift(k, odd, count)
        world_shift = 2*int(k)*width+(2*zmin if odd else 0.)
        if not math.isfinite(world_shift):
            raise ValueError("nonfinite physical image shift")
        result.append((world_shift, bool(odd)))
        integer_images.append((shift, bool(odd)))
    return result, integer_images


def _slab_field_kernels(shape, fft_shape, spacing, shifts, tau):
    coordinates, valid = [], []
    for size, padded in zip(shape, fft_shape, strict=True):
        index = np.arange(padded)
        coordinates.append(np.where(index < size, index, index-padded))
        valid.append((index < size) | (index >= padded-size+1))
    lag = np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1)
    mask = valid[0][:, None, None] & valid[1][None, :, None] & valid[2][None, None, :]
    gradient, hessian = np.zeros((*fft_shape, 3)), np.zeros((*fft_shape, 3, 3))
    for shift in shifts:
        current = lag.astype(np.float64)
        current[..., 2] -= shift  # Exact integer image index, not floating physical-shift rounding.
        current *= spacing
        g, h = analytic_gaussian_derivatives(current, tau)
        gradient += np.where(mask[..., None], g, 0)
        hessian += np.where(mask[..., None, None], h, 0)
    return gradient, hessian


def finite_slab_field_mesh(
    position, strength, core, targets, images, *, zmin, zmax, tau, spacing, order=10,
    correction_cutoff=None, max_grid_nodes=2_000_000, max_pairs=200_000, max_images=513,
):
    """Finite integer slab-image set; source cores only and no runtime admission."""
    max_grid_nodes = _positive_cap(max_grid_nodes, "max_grid_nodes")
    max_pairs, max_images = _positive_cap(max_pairs, "max_pairs"), _positive_cap(max_images, "max_images")
    lattice_x, steps, count = slab_coordinates(position, zmin, zmax, spacing)
    lattice_q, _, _ = slab_coordinates(targets, zmin, zmax, spacing)
    world_images, integer_images = slab_world_images(images, zmin, zmax, count, max_images)
    x, gamma, sigma, query, world_images = _inputs(position, strength, core, targets, world_images, float(tau), max_images)
    if not isinstance(order, int) or order not in (4, 6, 8, 10):
        raise ValueError("even interpolation order4..10 required")
    if correction_cutoff is not None and (not math.isfinite(correction_cutoff) or correction_cutoff <= 0):
        raise ValueError("positive finite correction cutoff or None required")
    pairs = len(x)*len(query)*len(integer_images)
    if pairs > max_pairs:
        raise ValueError("small-cloud qualification pair budget exceeded")
    u, j = np.zeros((len(query), 3)), np.zeros((len(query), 3, 3))
    du, dj, vu, vj = u.copy(), j.copy(), np.zeros(len(query)), np.zeros(len(query))
    diagnostics = {"fields_runtime_qualified": False, "mesh_error_certified": False, "roundoff_certified": False,
                   "analytic_field_convolution": True, "interpolated_u_J_derivative_consistent": False,
                   "source_only_cores": True, "infinite_periodic_operator": False,
                   "integer_images": integer_images, "finite_image_count": len(integer_images),
                   "exact_coincidence_override": False, "spacing": steps.tolist(), "tau": tau,
                   "order": order, "pair_count": pairs, "correction_pairs": 0, "slab_cells": count,
                   "particle_positions_snapped": False}
    if not len(x) or not len(query) or not integer_images:
        return FiniteImageMeshResult(u, j, u.copy(), j.copy(), du, dj, vu, vj, diagnostics)
    families = []
    for odd in (False, True):
        shifts = [shift for shift, reflected in integer_images if reflected == odd]
        if shifts:
            family_x, family_g = _image_sources(lattice_x, gamma, 0, odd)
            families.append((family_x, family_g, shifts))
    all_points = np.concatenate([lattice_q, *(family[0] for family in families)])
    origin = np.floor(all_points.min(axis=0))-order
    dimensions = np.ceil(all_points.max(axis=0)-origin)+order+1
    if not np.isfinite(dimensions).all() or np.any(dimensions > max_grid_nodes) or np.any(dimensions < 1):
        raise ValueError("bounded FFT grid exceeded before shape allocation")
    shape = tuple(dimensions.astype(np.int64).tolist())
    fft_shape = tuple(next_fast_len(2*n-1) for n in shape)
    if math.prod(fft_shape) > max_grid_nodes:
        raise ValueError("bounded FFT grid exceeded")
    fields = np.zeros((*shape, 12))
    window = tuple(slice(0, size) for size in shape)
    for family_x, family_g, shifts in families:
        first, weights = _stencil(family_x, origin, 1., order)
        density = _assign(shape, first, weights, family_g)
        density_hat = [rfftn(density[..., c], s=fft_shape, workers=1) for c in range(3)]
        kernel_g, kernel_h = _slab_field_kernels(shape, fft_shape, steps, shifts, tau)
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
    first, weights = _stencil(lattice_q, origin, 1., order)
    values = _gather_fields(fields, first, weights)
    u, j = values[:, :3], values[:, 3:].reshape(-1, 3, 3)
    for shift, odd in world_images:
        sources, vectors = _image_sources(x, gamma, shift, odd)
        for target, point in enumerate(query):
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
        raise FloatingPointError("nonfinite commensurate analytic-field mesh result")
    diagnostics.update(grid_shape=shape, fft_shape=fft_shape, fft_nodes=math.prod(fft_shape),
                       grid_origin_lattice=origin.tolist(), families=len(families),
                       inverse_transforms=12*len(families), no_discrete_linear_convolution_alias=True)
    return FiniteImageMeshResult(u+du, j+dj, u.copy(), j.copy(), du, dj, vu, vj, diagnostics)
