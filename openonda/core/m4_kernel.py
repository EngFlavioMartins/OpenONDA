"""M4' interpolation kernel (Monaghan 1985) of the particle remeshing and the vorticity transfer.

The polynomial has one host form (:func:`m4_prime`, NumPy), one compiled scalar form
(:func:`m4_prime_scalar`, Numba) and one device form (:func:`m4_prime_ti`, Taichi).
"""

import numpy as np
import taichi as ti

from ._numba import cacheable_njit as njit


def m4_prime(distance: np.ndarray | float) -> np.ndarray:
    """Return the one-dimensional M4' weights at dimensionless distances.

    The support is [-2, 2]. The weights form a partition of unity and reproduce
    quadratic polynomials on a complete stencil, so remeshing is third-order
    accurate; the negative lobes on (1, 2) keep it free of aliasing. Floating
    inputs keep their precision.
    """
    q = np.abs(np.asarray(distance))
    if not np.issubdtype(q.dtype, np.floating):
        q = q.astype(np.float64)
    weight = np.zeros_like(q)
    inner = q <= 1.0
    outer = (q > 1.0) & (q <= 2.0)
    near, far = q[inner], q[outer]
    weight[inner] = 1.0 - 2.5 * near * near + 1.5 * near * near * near
    weight[outer] = 0.5 * (2.0 - far) * (2.0 - far) * (1.0 - far)
    return weight


@njit(cache=True, fastmath=False)
def m4_prime_scalar(distance: float) -> float:
    """Return the M4' weight at one dimensionless distance, for compiled loops."""
    q = abs(distance)
    if q <= 1.0:
        return 1.0 - 2.5 * q * q + 1.5 * q * q * q
    if q <= 2.0:
        return 0.5 * (2.0 - q) * (2.0 - q) * (1.0 - q)
    return 0.0


def m4_axis_weights(fractions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the M4' weights of the 4-node stencil along each axis.

    ``fractions`` has shape (N, 3): the position of each particle in grid cells
    minus its base node ``floor(position)``. Each returned array has shape
    (N, 4) and holds the weights of the nodes at offsets -1, 0, 1, 2 from the base.
    """
    offsets = np.arange(-1, 3, dtype=np.int64)
    return tuple(
        np.stack([m4_prime(fractions[:, axis] - offset) for offset in offsets], axis=1)
        for axis in range(3)
    )


@ti.func
def m4_prime_ti(q: ti.f32) -> ti.f32:
    """Taichi form of :func:`m4_prime`, in single precision."""
    r = ti.abs(q)
    w = 0.0
    if r <= 1.0:
        w = 1.0 - 2.5 * r * r + 1.5 * r * r * r
    elif r <= 2.0:
        w = 0.5 * (2.0 - r) * (2.0 - r) * (1.0 - r)
    return w
