"""Independent direct Gaussian fields for current solver validation."""

import numpy as np

from tests.vpm._slip_periodic_gaussian_reference import gaussian_pairs


def primary_direct(position, strength, core, indices, *, chunk=8192, max_pairs=10_000_000):
    """Independent Gaussian pair evaluation; identical source index is included."""
    x, gamma, sigma = (np.asarray(value, dtype=np.float64) for value in (position, strength, core))
    indices = np.asarray(indices)
    if (
        x.ndim != 2
        or x.shape[1:] != (3,)
        or gamma.shape != x.shape
        or sigma.shape != (len(x),)
        or indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0)
        or np.any(indices >= len(x))
        or not all(np.isfinite(value).all() for value in (x, gamma, sigma))
        or np.any(sigma <= 0)
    ):
        raise ValueError(
            "finite source vectors, positive cores and valid particle indices required"
        )
    if (
        isinstance(chunk, bool)
        or not isinstance(chunk, int)
        or not 1 <= chunk <= 65536
        or isinstance(max_pairs, bool)
        or not isinstance(max_pairs, int)
        or max_pairs < 0
        or len(x) * len(indices) > max_pairs
    ):
        raise ValueError("bounded direct pair work required")
    u, j = np.zeros((len(indices), 3)), np.zeros((len(indices), 3, 3))
    absolute_u, absolute_j = np.zeros_like(u), np.zeros_like(j)
    for first in range(0, len(x), chunk):
        last = min(first + chunk, len(x))
        displacement = x[indices, None] - x[None, first:last]
        pair_core = 0.5 * (sigma[indices, None] + sigma[None, first:last])
        du, dj = gaussian_pairs(displacement, gamma[None, first:last], pair_core)
        u += du.sum(axis=1)
        j += dj.sum(axis=1)
        absolute_u += np.abs(du).sum(axis=1)
        absolute_j += np.abs(dj).sum(axis=1)
    if not all(np.isfinite(value).all() for value in (u, j, absolute_u, absolute_j)):
        raise FloatingPointError("nonfinite independent primary field")
    return u, j, absolute_u, absolute_j


def transposed_rate_f32(gradient, strength, *, fused=False):
    """Ordered three-term f32 J^T Gamma, with explicit separate/FMA rounding.

    The FMA variant evaluates each f32 product-plus-accumulator exactly enough
    in f64 before one f32 rounding. Neither is called bitwise-native, because
    the CUDA compiler may choose another reassociation. Both protect the same
    transposed contraction, not a direct or symmetric-gradient substitute.
    """
    gradient, strength = np.asarray(gradient, np.float32), np.asarray(strength, np.float32)
    if gradient.shape != (len(strength), 3, 3) or strength.shape != (len(strength), 3):
        raise ValueError("matching full-J and strength arrays required")
    rate = np.zeros_like(strength)
    for component in range(3):
        for axis in range(3):
            if fused:
                rate[:, component] = (
                    gradient[:, axis, component].astype(np.float64) * strength[:, axis]
                    + rate[:, component].astype(np.float64)
                ).astype(np.float32)
            else:
                rate[:, component] += gradient[:, axis, component] * strength[:, axis]
    return rate


def source_only_direct(position, strength, core, targets, *, chunk=4096, max_pairs=40_000_000):
    x, gamma, sigma, q = (
        np.asarray(value, np.float64) for value in (position, strength, core, targets)
    )
    if (
        x.ndim != 2
        or x.shape[1:] != (3,)
        or gamma.shape != x.shape
        or sigma.shape != (len(x),)
        or q.ndim != 2
        or q.shape[1:] != (3,)
        or np.any(sigma <= 0)
        or not all(np.isfinite(value).all() for value in (x, gamma, sigma, q))
    ):
        raise ValueError("finite source/target arrays and positive source cores required")
    if (
        not isinstance(chunk, int)
        or isinstance(chunk, bool)
        or not 1 <= chunk <= 16384
        or len(x) * len(q) > max_pairs
    ):
        raise ValueError("bounded direct query work required")
    u, j = np.zeros_like(q), np.zeros((len(q), 3, 3))
    for first in range(0, len(x), chunk):
        last = min(first + chunk, len(x))
        du, dj = gaussian_pairs(
            q[:, None] - x[None, first:last], gamma[None, first:last], sigma[None, first:last]
        )
        u += du.sum(axis=1)
        j += dj.sum(axis=1)
    return u, j


def finite_wall_normal(position, strength, core, targets, *, zmin, zmax, shells, upper):
    """Exact finite-family normal component via reflection pairing.

    Include physical primary plus both families k=-K..K. At the lower plane,
    even k pairs with odd -k, hence exact zero. At the upper plane, even k
    pairs with odd1-k; only even(-K) and odd(-K) lack their K+1 partners.
    Those two explicit Gaussian sums are the entire finite normal field.
    This avoids calling a finite family equal to the infinite zero-normal
    boundary condition. Evaluation still has ordinary f64 arithmetic error.
    """
    x, gamma, sigma, q = (
        np.asarray(value, np.float64) for value in (position, strength, core, targets)
    )
    if (
        not isinstance(shells, int)
        or isinstance(shells, bool)
        or shells < 1
        or type(upper) is not bool
        or not zmax > zmin
        or q.ndim != 2
        or q.shape[1:] != (3,)
        or not np.all(q[:, 2] == (zmax if upper else zmin))
    ):
        raise ValueError("exact geometric plane queries and complete positive shells required")
    result = np.zeros(len(q))
    if not upper:
        return result
    shift = -2 * shells * (zmax - zmin)
    for odd in (False, True):
        source, vector = x.copy(), gamma.copy()
        source[:, 2] = shift + (2 * zmin - source[:, 2] if odd else source[:, 2])
        if odd:
            vector[:, :2] *= -1
        u, _ = source_only_direct(source, vector, sigma, q)
        result += u[:, 2]
    return result


def direct_finite_images(
    position, strength, core, targets, images, *, max_pairs=200_000, max_images=513
):
    """Direct source-core Gaussian sums for a bounded finite image family."""
    x, gamma, sigma, query = (
        np.asarray(value, np.float64) for value in (position, strength, core, targets)
    )
    if (
        x.ndim != 2
        or x.shape[1:] != (3,)
        or gamma.shape != x.shape
        or sigma.shape != (len(x),)
        or query.ndim != 2
        or query.shape[1:] != (3,)
        or np.any(sigma <= 0)
        or not all(np.isfinite(value).all() for value in (x, gamma, sigma, query))
    ):
        raise ValueError("finite source/target vectors and positive source cores required")
    for cap in (max_pairs, max_images):
        if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
            raise ValueError("positive integer direct work caps required")
    descriptors = []
    for shift, odd in images:
        if len(descriptors) >= max_images:
            raise ValueError("finite image count exceeds direct work cap")
        if type(odd) not in (bool, np.bool_) or not np.isfinite(float(shift)):
            raise ValueError("finite image shift and boolean reflection required")
        descriptors.append((float(shift), bool(odd)))
    if len(x) * len(query) * len(descriptors) > max_pairs:
        raise ValueError("direct qualification pair budget exceeded")
    velocity, gradient = np.zeros((len(query), 3)), np.zeros((len(query), 3, 3))
    conditioning = np.zeros((len(query), 2))
    for shift, odd in descriptors:
        sources, strengths = x.copy(), gamma.copy()
        if odd:
            sources[:, 2] *= -1
            strengths[:, :2] *= -1
        sources[:, 2] += shift
        u, j = gaussian_pairs(query[:, None] - sources[None], strengths[None], sigma[None])
        velocity += u.sum(axis=1)
        gradient += j.sum(axis=1)
        conditioning[:, 0] += np.linalg.norm(u, axis=-1).sum(axis=1)
        conditioning[:, 1] += np.linalg.norm(j, axis=(-2, -1)).sum(axis=1)
    if not all(np.isfinite(field).all() for field in (velocity, gradient, conditioning)):
        raise FloatingPointError("nonfinite direct finite-image result")
    return velocity, gradient, conditioning
