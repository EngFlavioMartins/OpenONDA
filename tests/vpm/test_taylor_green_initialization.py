"""Independent correctness checks for the Taylor-Green initializer.

Reference vorticity derived by hand from omega = curl(u) for the canonical
field u = U sin(kx) cos(ky) cos(kz), v = -U cos(kx) sin(ky) cos(kz), w = 0:

    omega_x = -U k cos(kx) sin(ky) sin(kz)
    omega_y = -U k sin(kx) cos(ky) sin(kz)
    omega_z =  2 U k sin(kx) sin(ky) cos(kz)

(Brachet et al. 1983, eq. 2.1-2.2, canonical U=1, k=1 box [0, 2*pi)^3.)
"""

import numpy as np
import pytest

from source.solvers.VPM.initial_conditions.flow_models import TaylorGreenVortexVPM

pytestmark = pytest.mark.unit

BOX_SIZE = 2.0 * np.pi
U0 = 1.0


def _reference_velocity(x, y, z, k=1.0, U=U0):
    u = U * np.sin(k * x) * np.cos(k * y) * np.cos(k * z)
    v = -U * np.cos(k * x) * np.sin(k * y) * np.cos(k * z)
    w = np.zeros_like(x)
    return np.stack([u, v, w], axis=-1)


def _reference_vorticity(x, y, z, k=1.0, U=U0):
    wx = -U * k * np.cos(k * x) * np.sin(k * y) * np.sin(k * z)
    wy = -U * k * np.sin(k * x) * np.cos(k * y) * np.sin(k * z)
    wz = 2.0 * U * k * np.sin(k * x) * np.sin(k * y) * np.cos(k * z)
    return np.stack([wx, wy, wz], axis=-1)


def _random_positions(rng, n, box_size=BOX_SIZE):
    return rng.uniform(0.0, box_size, size=(n, 3))


def test_velocity_is_divergence_free_by_finite_difference():
    rng = np.random.default_rng(20260814)
    positions = _random_positions(rng, 200)
    volumes = np.ones(len(positions))
    h = 1e-6
    div = np.zeros(len(positions))
    for axis in range(3):
        plus, minus = positions.copy(), positions.copy()
        plus[:, axis] += h
        minus[:, axis] -= h
        v_plus, _, _ = TaylorGreenVortexVPM(0.0, BOX_SIZE, 0.0, plus, volumes)
        v_minus, _, _ = TaylorGreenVortexVPM(0.0, BOX_SIZE, 0.0, minus, volumes)
        div += (v_plus[:, axis] - v_minus[:, axis]) / (2 * h)
    assert np.max(np.abs(div)) < 1e-4


def test_vorticity_matches_curl_of_velocity_by_finite_difference():
    """Independent numerical curl of the function's own (verified-correct)
    velocity field must match the vorticity the function reports."""
    rng = np.random.default_rng(7)
    positions = _random_positions(rng, 200)
    volumes = np.ones(len(positions))
    h = 1e-6

    def velocity_at(p):
        v, _, _ = TaylorGreenVortexVPM(0.0, BOX_SIZE, 0.0, p, np.ones(len(p)))
        return v

    grads = []
    for axis in range(3):
        plus, minus = positions.copy(), positions.copy()
        plus[:, axis] += h
        minus[:, axis] -= h
        grads.append((velocity_at(plus) - velocity_at(minus)) / (2 * h))
    dudx, dudy, dudz = grads

    curl = np.empty_like(positions)
    curl[:, 0] = dudy[:, 2] - dudz[:, 1]
    curl[:, 1] = dudz[:, 0] - dudx[:, 2]
    curl[:, 2] = dudx[:, 1] - dudy[:, 0]

    _, _, strengths = TaylorGreenVortexVPM(0.0, BOX_SIZE, 0.0, positions, volumes)
    omega_returned = strengths / volumes[:, None]

    np.testing.assert_allclose(omega_returned, curl, atol=1e-4)


def test_vorticity_matches_independent_analytic_reference():
    rng = np.random.default_rng(11)
    positions = _random_positions(rng, 500)
    volumes = np.ones(len(positions))
    _, _, strengths = TaylorGreenVortexVPM(0.0, BOX_SIZE, 0.0, positions, volumes)
    omega_returned = strengths / volumes[:, None]

    k = 2.0 * np.pi / BOX_SIZE
    expected = _reference_vorticity(positions[:, 0], positions[:, 1], positions[:, 2], k=k)
    np.testing.assert_allclose(omega_returned, expected, atol=1e-10)


def test_periodicity_under_box_translation():
    rng = np.random.default_rng(3)
    positions = _random_positions(rng, 100)
    volumes = np.ones(len(positions))
    v0, _, s0 = TaylorGreenVortexVPM(0.0, BOX_SIZE, 0.0, positions, volumes)
    for axis in range(3):
        shifted = positions.copy()
        shifted[:, axis] += BOX_SIZE
        v1, _, s1 = TaylorGreenVortexVPM(0.0, BOX_SIZE, 0.0, shifted, volumes)
        np.testing.assert_allclose(v1, v0, atol=1e-10)
        np.testing.assert_allclose(s1, s0, atol=1e-10)


def test_representative_exact_point_values():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [np.pi / 2, 0.0, 0.0],
            [np.pi / 2, np.pi / 2, 0.0],
            [np.pi / 4, np.pi / 4, np.pi / 4],
        ]
    )
    volumes = np.ones(len(points))
    v, _, s = TaylorGreenVortexVPM(0.0, BOX_SIZE, 0.0, points, volumes)
    omega = s / volumes[:, None]

    expected_v = _reference_velocity(points[:, 0], points[:, 1], points[:, 2])
    expected_w = _reference_vorticity(points[:, 0], points[:, 1], points[:, 2])

    np.testing.assert_allclose(v, expected_v, atol=1e-10)
    np.testing.assert_allclose(omega, expected_w, atol=1e-10)


def test_volume_integrated_vorticity_vanishes_on_periodic_lattice():
    n = 8
    coords = np.linspace(0.0, BOX_SIZE, n, endpoint=False)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    positions = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    volumes = np.full(len(positions), (BOX_SIZE / n) ** 3)

    _, _, strengths = TaylorGreenVortexVPM(0.0, BOX_SIZE, 0.0, positions, volumes)
    total = strengths.sum(axis=0)
    scale = np.abs(strengths).sum(axis=0) + 1e-30
    np.testing.assert_allclose(total / scale, 0.0, atol=1e-10)


def test_strength_equals_vorticity_times_volume():
    rng = np.random.default_rng(99)
    positions = _random_positions(rng, 50)
    volumes = rng.uniform(0.5, 2.0, size=len(positions))
    _, visc, strengths = TaylorGreenVortexVPM(0.3, BOX_SIZE, 0.05, positions, volumes)
    k = 2.0 * np.pi / BOX_SIZE
    expected_omega = _reference_vorticity(positions[:, 0], positions[:, 1], positions[:, 2], k=k)
    np.testing.assert_allclose(strengths, expected_omega * volumes[:, None], atol=1e-10)
    assert np.all(visc == pytest.approx(0.3))
