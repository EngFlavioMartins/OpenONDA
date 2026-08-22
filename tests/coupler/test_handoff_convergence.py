"""Manufactured convergence tests for the compatible-curl handoff."""

from __future__ import annotations

import numpy as np

from source.coupler.vorticity_transfer import (
    build_transfer_lattice,
    solenoidal_velocity_correction,
    vortex_strength_from_velocity_trace,
)


def _scatter(lattice, result):
    field = np.zeros((len(lattice.positions), 3))
    index = {tuple(point): i for i, point in enumerate(lattice.positions)}
    for point, strength in zip(result.pos, result.circ, strict=True):
        field[index[tuple(point)]] = strength
    return field


def test_uniform_velocity_has_exactly_zero_curl():
    rng = np.random.default_rng(3)
    points = rng.uniform(-1.0, 1.0, (100, 3))
    strength = vortex_strength_from_velocity_trace(
        points,
        0.1,
        lambda query: np.tile([1.3, -0.4, 0.7], (len(query), 1)),
    )
    np.testing.assert_array_equal(strength, np.zeros_like(strength))


def test_solid_body_rotation_has_exact_constant_curl():
    rng = np.random.default_rng(4)
    points = rng.uniform(-1.0, 1.0, (100, 3))
    h = 0.05
    omega = np.array([0.3, -0.2, 1.1])
    strength = vortex_strength_from_velocity_trace(
        points,
        h,
        lambda query: 0.5 * np.cross(omega, np.asarray(query)),
    )
    np.testing.assert_allclose(strength, np.tile(omega * h**3, (len(points), 1)), atol=2.0e-18)


def test_quadratic_velocity_curl_is_exact_on_the_transfer_lattice():
    box = np.array([-1.0, 1.0] * 3)
    h = 0.1
    lattice = build_transfer_lattice(
        box,
        h,
    )

    def velocity(points):
        x, y, z = np.asarray(points).T
        return np.column_stack((y * z, x * z, x * y + x**2))

    result = solenoidal_velocity_correction(
        lattice,
        h,
        fvm_velocity_at=velocity,
        vpm_velocity_at=lambda points: np.zeros((len(points), 3)),
        authority_at=lambda points: np.ones(len(points)),
        core_radius_ratio=1.0,
    )
    numerical = _scatter(lattice, result) / h**3
    x, _y, _z = lattice.positions.T
    exact = np.column_stack((np.zeros_like(x), -2.0 * x, np.zeros_like(x)))
    np.testing.assert_allclose(numerical, exact, atol=3.0e-14)


def _analytic_vortex_error(h: float) -> float:
    box = np.array([-1.0, 1.0] * 3)
    lattice = build_transfer_lattice(
        box,
        h,
    )

    def velocity(points):
        x, y, _z = np.asarray(points).T
        envelope = np.exp(-(x**2 + y**2))
        return np.column_stack((-y * envelope, x * envelope, np.zeros(len(points))))

    result = solenoidal_velocity_correction(
        lattice,
        h,
        fvm_velocity_at=velocity,
        vpm_velocity_at=lambda points: np.zeros((len(points), 3)),
        authority_at=lambda points: np.ones(len(points)),
        core_radius_ratio=1.0,
    )
    numerical = _scatter(lattice, result)[:, 2] / h**3
    x, y, z = lattice.positions.T
    exact = 2.0 * (1.0 - x**2 - y**2) * np.exp(-(x**2 + y**2))
    core = (np.abs(x) <= 0.7) & (np.abs(y) <= 0.7) & (np.abs(z) <= 0.7)
    return float(np.linalg.norm(numerical[core] - exact[core]) / np.linalg.norm(exact[core]))


def test_analytic_vortex_is_second_order_convergent():
    errors = np.array([_analytic_vortex_error(h) for h in (0.2, 0.1, 0.05)])
    orders = np.log2(errors[:-1] / errors[1:])
    assert np.all(orders > 1.8), (errors, orders)
