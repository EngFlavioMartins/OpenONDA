"""Field-based ring measurements are independent of particle labels."""

import numpy as np

from source.solvers.vpm.diagnostics.axisymmetric_field import azimuthal_vorticity


def test_half_plane_circulation_matches_independent_numerical_integral():
    from scipy.integrate import quad

    from source.solvers.vpm.diagnostics.axisymmetric_field import azimuthal_circulation

    # Near-axis blobs expose cancellation that sum(alpha_theta/r) misses.
    p = np.array([[0, 0.04, 0], [0.1, 0.4, 0.2], [0, 0, 0]])
    a = np.array([[0, 0, 1], [0, -0.2, 0.4], [1, 2, 3]])
    sigma = np.array([0.2, 0.15, 0.2])
    integral = quad(
        lambda r: quad(
            lambda x: azimuthal_vorticity(p, a, sigma, [[x, r]])[0], -1.5, 1.5, epsabs=1e-8
        )[0],
        0,
        1.5,
        epsabs=1e-8,
    )[0]
    assert np.isclose(azimuthal_circulation(p, a, sigma), integral, rtol=1e-8)


def test_angular_integral_matches_independent_cartesian_gaussian_quadrature():
    rng = np.random.default_rng(912)
    position = rng.normal(size=(15, 3)) * 0.4
    strength = rng.normal(size=(15, 3))
    sigma = rng.uniform(0.15, 0.4, 15)
    targets = np.array([[0.1, 0.4], [-0.2, 0.6], [0, 0]])
    angle = np.arange(2048) * 2 * np.pi / 2048
    expected = []
    for x, r in targets:
        points = np.column_stack((np.full(len(angle), x), r * np.cos(angle), r * np.sin(angle)))
        delta = points[:, None, :] - position
        omega = (np.exp(-np.sum(delta**2, axis=2) / sigma**2) / (np.pi**1.5 * sigma**3)) @ strength
        expected.append(np.mean(-omega[:, 1] * np.sin(angle) + omega[:, 2] * np.cos(angle)))
    np.testing.assert_allclose(
        azimuthal_vorticity(position, strength, sigma, targets), expected, rtol=1e-12, atol=1e-13
    )
    np.testing.assert_allclose(
        azimuthal_vorticity(position, strength, sigma, np.tile(targets, (30, 1))),
        np.tile(expected, 30),
        rtol=1e-12,
        atol=1e-13,
    )


def test_thin_core_angular_integral_stays_finite():
    result = azimuthal_vorticity([[0, 1, 0]], [[0, 0, 1]], [0.001], [[0, 1]])
    assert np.isfinite(result).all()
    assert result[0] > 0
