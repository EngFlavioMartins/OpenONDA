"""M4-prime interpolation kernel shared by renewal and GBD diffusion."""

import numpy as np

from source._numba import cacheable_njit as njit


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
