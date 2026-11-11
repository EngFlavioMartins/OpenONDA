"""M4' interpolation kernel (Monaghan 1985) of the particle remeshing and the vorticity transfer."""

import numpy as np

from openonda.core._numba import cacheable_njit as njit


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
