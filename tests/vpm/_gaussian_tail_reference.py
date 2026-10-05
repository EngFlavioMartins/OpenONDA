"""Independent direct shell sums and small clouds for tail validation."""

import numpy as np

from tests.vpm._slip_periodic_gaussian_reference import gaussian_pairs


def explicit_tail(x, gamma, sigma, targets, zmin, zmax, first, last):
    """Direct Gaussian pairs, independent of all moment/remainder arithmetic."""
    u, j = np.zeros((len(targets), 3)), np.zeros((len(targets), 3, 3))
    period = 2 * (zmax - zmin)
    for odd in (False, True):
        source, strength = x.copy(), gamma.copy()
        if odd:
            source[:, 2] = 2 * zmin - source[:, 2]
            strength[:, :2] *= -1
        for lo in range(first, last + 1, 32):
            k = np.arange(lo, min(last + 1, lo + 32), dtype=float)
            shifts = np.concatenate((k, -k)) * period
            d = targets[:, None, None, :] - source[None, None, :, :]
            d = np.broadcast_to(d, (len(targets), len(shifts), len(x), 3)).copy()
            d[..., 2] -= shifts[None, :, None]
            du, dj = gaussian_pairs(d, strength[None, None, :, :], sigma[None, None, :])
            u += du.sum(axis=(1, 2))
            j += dj.sum(axis=(1, 2))
    return u, j


def cloud(kind):
    rng = np.random.default_rng(8675309)
    x = rng.uniform(-0.4, 0.4, (7, 3))
    g = rng.normal(size=x.shape)
    sigma = rng.uniform(0.03, 0.2, len(x))
    t = rng.uniform(-0.7, 0.7, (5, 3))
    zmin, zmax, k = -0.5, 0.5, 4
    if kind == "cancelled":
        x = np.array([[q, 0.0, 0.0] for q in np.linspace(-0.15, 0.15, 5)])
        g = np.array([1.0, -4.0, 6.0, -4.0, 1.0])[:, None] * np.array([0.3, -0.2, 1.0])[None, :]
        sigma = np.full(len(x), 0.09)
    elif kind == "axial":
        g[:, :2] = 0
        g[:, 2] = np.arange(1, len(x) + 1)
    elif kind == "translated":
        offset = np.array([1024.0, -512.0, 256.0])
        x, t = x + offset, t + offset
        zmin, zmax = zmin + offset[2], zmax + offset[2]
    elif kind == "near_distance_limit":
        x = np.array([[0.0, 0.0, 0.01], [0.02, -0.01, -0.02]])
        g, sigma = g[:2], np.full(2, 0.01)
        t = np.array([[0.0, 0.0, 2.8], [0.01, -0.02, 2.75]])
        k = 2
    return x, g, sigma, t, zmin, zmax, k
